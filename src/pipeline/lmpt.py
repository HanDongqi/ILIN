from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from src.language.foq import EFO1Query
from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.pipeline.reasoner import Reasoner
from src.transformer.TrmE import Transformer


class LMPTLayer(nn.Module):
    """
    data format [batch, dim]
    """
    def __init__(self, hidden_dim, nbp: NeuralBinaryPredicate, layers=2, agg_func='mean', pre_norm=True):
        super(LMPTLayer, self).__init__()
        self.nbp = nbp
        self.feature_dim = nbp.entity_embedding.size(1)

        self.hidden_dim = hidden_dim
        self.num_entities = nbp.num_entities
        self.agg_func = agg_func

        # Transformer
        self.transformer_dropout = 0.1
        self.transformer_activation = 'relu'
        self.num_encoder_layers = layers
        self.heads = 8

        if pre_norm:
            self.transformer = Transformer(self.num_encoder_layers, self.feature_dim, self.hidden_dim, self.transformer_dropout, self.heads)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                self.feature_dim, self.heads, self.hidden_dim, self.transformer_dropout, self.transformer_activation
            )
            encoder_norm = nn.LayerNorm(self.feature_dim)
            self.transformer = nn.TransformerEncoder(encoder_layer, self.num_encoder_layers, encoder_norm)  # post-norm

        self.existential_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        self.universal_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))
        self.free_embedding = nn.Parameter(
            torch.rand((1, self.feature_dim)))

    def message_passing(self, term_emb_dict, atomic_dict, pred_emb_dict, inv_pred_emb_dict):

        term_collect_embs_dict = defaultdict(list)
        for predicate, atomic in atomic_dict.items():
            head_name, tail_name = atomic.head.name, atomic.tail.name
            head_emb = term_emb_dict[head_name]
            tail_emb = term_emb_dict[tail_name]
            sign = -1 if atomic.negated else 1

            pred_emb = pred_emb_dict[atomic.relation]
            if head_emb.size(0) == 1:
                head_emb = head_emb.expand(pred_emb.size(0), -1)
            if tail_emb.size(0) == 1:
                tail_emb = tail_emb.expand(pred_emb.size(0), -1)

            assert head_emb.size(0) == pred_emb.size(0)
            assert tail_emb.size(0) == pred_emb.size(0)
            term_collect_embs_dict[tail_name].append(
                sign * self.nbp.estimate_tail_emb(head_emb, pred_emb)
            )

            term_collect_embs_dict[head_name].append(
                sign * self.nbp.estimate_head_emb(tail_emb, pred_emb)
            )

        return term_collect_embs_dict

    def forward(self, init_term_emb_dict, predicates, pred_emb_dict, inv_pred_emb_dict):
        # message passing
        term_collect_embs_dict = self.message_passing(
            init_term_emb_dict, predicates, pred_emb_dict, inv_pred_emb_dict
        )

        # node embedding updating
        out_term_emb_dict = {}
        for t, collect_emb_list in term_collect_embs_dict.items():
            temp = init_term_emb_dict[t]
            if temp.size(0) == 1:
                temp = temp.expand(collect_emb_list[0].size(0), -1)
            collect_emb_list.append(temp)
            x = torch.stack(collect_emb_list, 1)  # [batch, m, dim]
            agg_emb = self.transformer(x)
            if self.agg_func == 'sum':
                agg_emb = agg_emb.sum(dim=1)
            else:  # mean
                agg_emb = agg_emb.mean(dim=1)
            out_term_emb_dict[t] = agg_emb

        return out_term_emb_dict


class LMPTReasoner(Reasoner):
    def __init__(self,
                 nbp: NeuralBinaryPredicate,
                 lgnn_layer: LMPTLayer,
                 depth_shift=0):
        self.nbp = nbp
        self.lgnn_layer = lgnn_layer        # formula dependent
        self.depth_shift = depth_shift

        self.formula: EFO1Query = None
        self.term_local_emb_dict = {}

    def initialize_with_query(self, formula):
        self.formula = formula
        self.term_local_emb_dict = {term_name: None
                                    for term_name in self.formula.term_dict}

    def initialize_local_embedding(self):
        for term_name in self.formula.term_dict:
            if self.formula.has_term_grounded_entity_id_list(term_name):
                entity_id = self.formula.get_term_grounded_entity_id_list(term_name)
                emb = self.nbp.get_entity_emb(entity_id)
            elif self.formula.term_dict[term_name].is_existential:
                emb = self.lgnn_layer.existential_embedding
            elif self.formula.term_dict[term_name].is_free:
                emb = self.lgnn_layer.free_embedding
            elif self.formula.term_dict[term_name].is_universal:
                emb = self.lgnn_layer.universal_embedding
            else:
                raise KeyError(f"term name {term_name} cannot be initialized")
            self.set_local_embedding(term_name, emb)

    def estimate_variable_embeddings(self):
        self.initialize_local_embedding()
        term_emb_dict = self.term_local_emb_dict
        pred_emb_dict = {}
        inv_pred_emb_dict = {}
        for atomic_name in self.formula.atomic_dict:
            pred_name = self.formula.atomic_dict[atomic_name].relation
            if self.formula.has_pred_grounded_relation_id_list(pred_name):
                pred_emb_dict[pred_name] = self.get_rel_emb(pred_name)
                inv_pred_emb_dict[pred_name] = self.get_rel_emb(pred_name, inv=True)

        for _ in range(max(1, self.formula.quantifier_rank + self.depth_shift)):
            term_emb_dict = self.lgnn_layer(
                term_emb_dict,
                self.formula.atomic_dict,
                pred_emb_dict,
                inv_pred_emb_dict
            )

        for term_name in term_emb_dict:
            self.term_local_emb_dict[term_name] = term_emb_dict[term_name]