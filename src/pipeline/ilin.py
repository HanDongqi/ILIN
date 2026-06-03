from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from src.language.foq import EFO1Query
from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.pipeline.reasoner import Reasoner
from src.griffin.griffin import Griffin


class ILINLayer(nn.Module):
    def __init__(self, hidden_dim, nbp: NeuralBinaryPredicate, layers=2, agg_func='mean',pre_norm=True, refinement_steps=2, num_heads=4):
        super(ILINLayer, self).__init__()
        self.nbp = nbp
        self.feature_dim = nbp.entity_embedding.size(1)

        self.hidden_dim = hidden_dim
        self.num_entities = nbp.num_entities
        self.agg_func = agg_func

        self.dropout_rate = 0.1

        self.griffin_mlp_mult = 3
        self.griffin = Griffin(
            D=self.feature_dim,
            depth=layers,
            mlp_expansion_factor=self.griffin_mlp_mult,
            device=nbp.device
        )

        self.existential_embedding = nn.Parameter(torch.rand((1, self.feature_dim)))
        self.universal_embedding = nn.Parameter(torch.rand((1, self.feature_dim)))
        self.free_embedding = nn.Parameter(torch.rand((1, self.feature_dim)))

        self.refinement_steps = refinement_steps

        self.refinement_gru = nn.GRUCell(self.feature_dim, self.feature_dim)
        
        self.pos_injection_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU()
        )
        self.neg_injection_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU()
        )

        self.context_aggregator = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            # nn.Dropout(self.dropout_rate)
        )

        assert self.feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        self.context_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=num_heads,
            dropout=self.dropout_rate,
            batch_first=True 
        )

        self.message_norm = nn.LayerNorm(self.feature_dim)
        self.pre_aggregation_norm = nn.LayerNorm(self.feature_dim)

        self.neg_tail_residual = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            # nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.feature_dim)
        )
        self.neg_head_residual = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            # nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.feature_dim)
        )
        
        
    def forward(self, init_term_emb_dict, predicates, pred_emb_dict, inv_pred_emb_dict):
        initial_messages = defaultdict(list)
        for predicate, atomic in predicates.items():
            head_name, tail_name = atomic.head.name, atomic.tail.name
            head_emb = init_term_emb_dict[head_name]
            tail_emb = init_term_emb_dict[tail_name]

            pred_emb = pred_emb_dict[atomic.relation]
            if head_emb.size(0) == 1:
                head_emb = head_emb.expand(pred_emb.size(0), -1)
            if tail_emb.size(0) == 1:
                tail_emb = tail_emb.expand(pred_emb.size(0), -1)

            assert head_emb.size(0) == pred_emb.size(0)
            assert tail_emb.size(0) == pred_emb.size(0)
            
            p_tail_emb = self.nbp.estimate_tail_emb(head_emb, pred_emb)
            p_head_emb = self.nbp.estimate_head_emb(tail_emb, pred_emb)

            if atomic.negated:
                offset_tail = self.neg_tail_residual(p_tail_emb)
                n_tail_msg = -p_tail_emb + offset_tail
                initial_messages[atomic.tail.name].append({'type': 1, 'msg': n_tail_msg}) # type 1 for negative

                offset_head = self.neg_head_residual(p_head_emb)
                n_head_msg = -p_head_emb + offset_head
                initial_messages[atomic.head.name].append({'type': 1, 'msg': n_head_msg})
            else:
                initial_messages[atomic.tail.name].append({'type': 0, 'msg': p_tail_emb}) # type 0 for positive
                initial_messages[atomic.head.name].append({'type': 0, 'msg': p_head_emb})
        
        refined_messages = {}
        for var_name, msg_info_list in initial_messages.items():
            if not msg_info_list or len(msg_info_list) <= 1:
                if msg_info_list:
                    refined_messages[var_name] = torch.stack([m['msg'] for m in msg_info_list]).transpose(0, 1)
                continue
            
            msg_types = torch.tensor([m['type'] for m in msg_info_list], device=self.nbp.device)
            current_msgs = torch.stack([m['msg'] for m in msg_info_list]).transpose(0, 1)
            
            for _ in range(self.refinement_steps):
                batch_size, num_msgs, feat_dim = current_msgs.shape
                attn_output, _ = self.context_attention(
                    query=current_msgs, 
                    key=current_msgs, 
                    value=current_msgs
                )
                
                gru_input_flat = attn_output.reshape(-1, feat_dim)
                
                type_mask_flat = msg_types.repeat(batch_size)
                pos_mask_flat = (type_mask_flat == 0)
                neg_mask_flat = (type_mask_flat == 1)
                
                specialized_gru_input = gru_input_flat.clone()
                
                if pos_mask_flat.any():
                    specialized_gru_input[pos_mask_flat] = self.pos_injection_mlp(gru_input_flat[pos_mask_flat])

                if neg_mask_flat.any():
                    specialized_gru_input[neg_mask_flat] = self.neg_injection_mlp(gru_input_flat[neg_mask_flat])
                
                context_for_gru = self.context_aggregator(specialized_gru_input)

                updated_msgs = self.refinement_gru(
                    context_for_gru,
                    current_msgs.reshape(-1, feat_dim)
                )
                
                # updated_msgs = current_msgs.reshape(-1, feat_dim) + updated_msgs
                current_msgs = self.message_norm(updated_msgs.reshape(batch_size, num_msgs, feat_dim))
            
            refined_messages[var_name] = current_msgs

        out_term_emb_dict = {}
        all_vars = set(initial_messages.keys()) | set(init_term_emb_dict.keys())

        for t in all_vars:
            if t not in ['f', 'e1', 'e2', 'e3']:
                out_term_emb_dict[t] = init_term_emb_dict.get(t)
                continue
            
            refined_msg_tensor = refined_messages.get(t)
            self_emb = init_term_emb_dict.get(t)

            griffin_input_elements = []
            if refined_msg_tensor is not None:
                griffin_input_elements.append(refined_msg_tensor)
            
            if self_emb is not None:
                if self_emb.size(0) == 1 and refined_msg_tensor is not None:
                    self_emb = self_emb.expand(refined_msg_tensor.size(0), -1)
                griffin_input_elements.append(self_emb.unsqueeze(1))
            
            if not griffin_input_elements:
                out_term_emb_dict[t] = init_term_emb_dict.get(t)
                continue

            x = torch.cat(griffin_input_elements, dim=1)  # [batch, m, dim]
            x = self.pre_aggregation_norm(x)
            agg_emb = self.griffin(x)

            if self.agg_func == 'sum':
                agg_emb = agg_emb.sum(dim=1)
            elif self.agg_func == 'mean':
                agg_emb = agg_emb.mean(dim=1)
            else:  # max pooling
                agg_emb = agg_emb.transpose(1, 2)
                agg_emb, _ = torch.max(agg_emb, dim=-1)
            out_term_emb_dict[t] = agg_emb

        return out_term_emb_dict


class ILINReasoner(Reasoner):
    def __init__(self,
                 nbp: NeuralBinaryPredicate,
                 lgnn_layer: ILINLayer,
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
