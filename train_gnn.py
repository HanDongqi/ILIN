import argparse
import json
import logging
import os
import os.path as osp
import random
from collections import defaultdict

import numpy as np
import torch
import tqdm

from src.language.grammar import parse_lstr_to_lformula
from src.language.tnorm import Tnorm
from src.pipeline import (GradientEFOReasoner, Reasoner, LMPTReasoner, LMPTLayer,
                          CLMPTReasoner, CLMPTLayer,
                          ILINReasoner, ILINLayer,
                          LMPNNReasoner, LogicalMPNNLayer)
from src.structure import get_nbp_class
from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex
from src.structure.neural_binary_predicate import NeuralBinaryPredicate
from src.utils.data import QueryAnsweringSeqDataLoader
from src.utils.util import set_global_seed
from src.language.foq import EFO1Query


torch.autograd.set_detect_anomaly(True)


from convert_beta_dataset import beta_lstr2name
# from convert_q2b_dataset import q2b_lstr2name
# lstr2name = {parse_lstr_to_lformula(k).lstr: v for k, v in q2b_lstr2name.items()}

lstr2name = {parse_lstr_to_lformula(k).lstr: v for k, v in beta_lstr2name.items()}
name2lstr = {v: k for k, v in lstr2name.items()}

negation_query = [name for name in name2lstr if '!' in name]

class TaskWeightManager:
    def __init__(self, all_query_names: list, device, alpha=0.1):
        self.all_query_names = set(all_query_names)
        self.device = device
        self.avg_losses = {name: torch.tensor(1.0, device=device) for name in self.all_query_names}
        self.alpha = alpha 

    def update_loss(self, query_name: str, current_loss: torch.Tensor):
        if query_name in self.avg_losses:
            self.avg_losses[query_name] = self.alpha * current_loss + (1 - self.alpha) * self.avg_losses[query_name]

    def get_weights(self) -> dict:
        total_loss = sum(loss.item() for loss in self.avg_losses.values())
        if total_loss == 0:
            num_tasks = len(self.all_query_names)
            return {name: 1.0 / num_tasks for name in self.all_query_names}
        
        weights = {name: loss.item() / total_loss for name, loss in self.avg_losses.items()}
        return weights
    
parser = argparse.ArgumentParser()

# base environment
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--output_dir", type=str, default='log')

# input task folder, defines knowledge graph, index, and formulas
parser.add_argument("--task_folder", type=str, default='data/FB15k-237-betae')
parser.add_argument("--train_queries", action='append')
parser.add_argument("--eval_queries", action='append')
parser.add_argument("--batch_size", type=int, default=512, help="batch size for training")
parser.add_argument("--batch_size_eval_truth_value", type=int, default=32, help="batch size for evaluating the truth value")
parser.add_argument("--batch_size_eval_dataloader", type=int, default=5000, help="batch size for evaluation")

# model, defines the neural binary predicate
parser.add_argument("--model_name", type=str, default='complex')
parser.add_argument("--checkpoint_path", required=True, type=str, help="path to the KGE checkpoint")
parser.add_argument("--embedding_dim", type=int, default=1000)
parser.add_argument("--margin", type=float, default=10)
parser.add_argument("--scale", type=float, default=1)
parser.add_argument("--p", type=int, default=1)

# optimization for the entire process
parser.add_argument("--optimizer", type=str, default='AdamW')
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--noisy_sample_size", type=int, default=128)
parser.add_argument("--temp", type=float, default=0.05)

# reasoning machine
parser.add_argument("--reasoner", type=str, default='clmpt', choices=['gradient', 'beam', 'lmpnn', 'lmpnn_dnf', 'lmpt', 'clmpt', 'ilin'])
parser.add_argument("--tnorm", type=str, default='product', choices=['product', 'godel'])

# reasoner = gradient
parser.add_argument("--reasoning_rate", type=float, default=1e-1)
parser.add_argument("--reasoning_steps", type=int, default=1000)
parser.add_argument("--reasoning_optimizer", type=str, default='AdamW')

# reasoner = gnn
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--hidden_dim", type=int, default=4096)
parser.add_argument("--eps", type=float, default=0.1)
parser.add_argument("--depth_shift", type=int, default=0)
parser.add_argument("--agg_func", type=str, default='sum')
parser.add_argument("--checkpoint_path_gnn", type=str, default=None)

parser.add_argument("--seed", default=0, type=int, help="random seed")
parser.add_argument('--freeze_nlp', action='store_true', default=False, help='Whether to freeze the pre-trained neural link predictor.')
parser.add_argument('--pre_norm', action='store_true', default=False, help='Whether to use pre norm.')

parser.add_argument("--task_weight_alpha", type=float, default=0.1, help="Smoothing factor for dynamic task weighting.")

def train_gnn(
        desc: str,
        train_dataloader: QueryAnsweringSeqDataLoader,
        nbp: NeuralBinaryPredicate,
        reasoner: Reasoner,
        optimizer: torch.optim.Optimizer,
        task_weight_manager: TaskWeightManager,
        args):

    T = args.temp
    trajectory = defaultdict(list)

    fof_list = train_dataloader.get_fof_list()
    t = tqdm.tqdm(enumerate(fof_list), desc=desc, total=len(fof_list))

    nbp.eval()
    reasoner.lgnn_layer.train()

    # for each batch
    for ifof, fof in t:
        ####################
        loss = 0
        metric_step = {}

        query_name = lstr2name.get(fof.lstr, "unknown")

        reasoner.initialize_with_query(fof)

        # this procedure is somewhat of low efficiency
        reasoner.estimate_variable_embeddings()
        batch_fvar_emb = reasoner.get_ent_emb('f')
        pos_1answer_list = []
        neg_answers_list = []


        for i, pos_answer_dict in enumerate(fof.easy_answer_list):
            # this iteration is somehow redundant since there is only one free
            # variable in current case, i.e., fname='f'
            assert 'f' in pos_answer_dict
            pos_1answer_list.append(random.choice(pos_answer_dict['f']))
            neg_answers_list.append(torch.randint(0, nbp.num_entities,
                                                  (args.noisy_sample_size, 1)))

        batch_pos_emb = nbp.get_entity_emb(pos_1answer_list)
        neg_temp = torch.cat(neg_answers_list, dim=1)
        batch_neg_emb = nbp.get_entity_emb(neg_temp)

        contrastive_pos_score = torch.exp(torch.cosine_similarity(
            batch_pos_emb, batch_fvar_emb, dim=-1) / T)
        contrastive_neg_score = torch.exp(torch.cosine_similarity(
            batch_neg_emb, batch_fvar_emb, dim=-1) / T)
        contrastive_nll = - torch.log(
            contrastive_pos_score / (contrastive_pos_score + contrastive_neg_score.sum(0))
        ).mean()

        metric_step['contrastive_pos_score'] = contrastive_pos_score.mean().item()
        metric_step['contrastive_neg_score'] = contrastive_neg_score.mean().item()
        metric_step['contrastive_nll'] = contrastive_nll.item()
        
        task_weights = task_weight_manager.get_weights()
        current_weight = task_weights.get(query_name, 1.0) 

        loss += current_weight * contrastive_nll

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reasoner.lgnn_layer.parameters(), max_norm=1.0)
        optimizer.step()

        task_weight_manager.update_loss(query_name, contrastive_nll.detach())
        ####################
        metric_step = {
            'loss': loss.item(),
            'raw_loss': contrastive_nll.item(),
            'task_weight': current_weight
        }

        postfix = {'step': ifof+1}
        for k in metric_step:
            postfix[k] = np.mean(metric_step[k])
            trajectory[k].append(postfix[k])
        postfix['acc_loss'] = np.mean(trajectory['loss'])
        t.set_postfix(postfix)

        metric_step['acc_loss'] = postfix['acc_loss']
        metric_step['lstr'] = fof.lstr

        logging.info(f"[train GNN {desc}] {json.dumps(metric_step)}")

    t.close()

    with open(osp.join(args.output_dir, 'task_weights.log'), 'a') as f:
        f.write(json.dumps(task_weights) + '\n')

    metric = {}
    for k in trajectory:
        metric[k] = np.mean(trajectory[k])
    return metric


def compute_evaluation_scores(fof, batch_entity_rankings, metric):
    k = 'f'
    for i, ranking in enumerate(torch.split(batch_entity_rankings, 1)):
        ranking = ranking.squeeze()
        if fof.hard_answer_list[i]:
            # [1, num_entities]
            hard_answers = torch.tensor(fof.hard_answer_list[i][k],
                                        device=nbp.device)
            hard_answer_rank = ranking[hard_answers]

            # remove better easy answers from its rankings
            if fof.easy_answer_list[i][k]:
                easy_answers = torch.tensor(fof.easy_answer_list[i][k],
                                            device=nbp.device)
                easy_answer_rank = ranking[easy_answers].view(-1, 1)

                num_skipped_answers = torch.sum(
                    hard_answer_rank > easy_answer_rank, dim=0)
                pure_hard_ans_rank = hard_answer_rank - num_skipped_answers
            else:
                pure_hard_ans_rank = hard_answer_rank.squeeze()

        else:
            pure_hard_ans_rank = ranking[
                torch.tensor(fof.easy_answer_list[i][k], device=nbp.device)]

        # remove better hard answers from its ranking
        _reference_hard_ans_rank = pure_hard_ans_rank.reshape(-1, 1)
        num_skipped_answers = torch.sum(
            pure_hard_ans_rank > _reference_hard_ans_rank, dim=0
        )
        pure_hard_ans_rank -= num_skipped_answers.reshape(
            pure_hard_ans_rank.shape)

        rr = (1 / (1+pure_hard_ans_rank)).detach().cpu().float().numpy()
        hit1 = (pure_hard_ans_rank < 1).detach().cpu().float().numpy()
        hit3 = (pure_hard_ans_rank < 3).detach().cpu().float().numpy()
        hit10 = (pure_hard_ans_rank < 10).detach().cpu().float().numpy()
        metric['mrr'].append(rr.mean())
        metric['hit1'].append(hit1.mean())
        metric['hit3'].append(hit3.mean())
        metric['hit10'].append(hit10.mean())


def evaluate_by_search_emb_then_rank_truth_value(
        e,
        desc,
        dataloader,
        nbp: NeuralBinaryPredicate,
        reasoner: Reasoner):
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    foqs = dataloader.get_fof_list()

    # conduct reasoning
    with tqdm.tqdm(foqs, desc=desc) as t:
        for query in t:
            reasoner.initialize_with_query(query)
            reasoner.estimate_variable_embeddings()
            with torch.no_grad():
                truth_value_entity_batch = reasoner.evaluate_truth_values(
                    free_var_emb_dict={
                        'f': nbp.entity_embedding.unsqueeze(1)
                    },
                    batch_size_eval=args.batch_size_eval_truth_value)  # [num_entities batch_size]
            ranking_score = torch.transpose(truth_value_entity_batch, 0, 1)
            ranked_entity_ids = torch.argsort(
                ranking_score, dim=-1, descending=True)
            batch_entity_rankings = torch.argsort(
                ranked_entity_ids, dim=-1, descending=False)
            compute_evaluation_scores(
                query, batch_entity_rankings, metric[query.lstr])

            sum_metric = defaultdict(dict)
            for lstr in metric:
                for score_name in metric[lstr]:
                    sum_metric[lstr2name[lstr]][score_name] = float(
                        np.mean(metric[lstr][score_name]))

            postfix = {}
            # postfix['reasoning_steps'] = len(traj)
            postfix['lstr'] = query.lstr
            for name in ['1p', '2p', '3p', '2i', 'inp']:
                if name in sum_metric:
                    postfix[name + '_hit3'] = sum_metric[name]['hit3']
            t.set_postfix(postfix)
            torch.cuda.empty_cache()

    sum_metric['epoch'] = e
    logging.info(f"[{desc}][final] {json.dumps(sum_metric)}")


def evaluate_by_nearest_search(
        e,
        desc,
        dataloader,
        nbp: NeuralBinaryPredicate,
        reasoner: GradientEFOReasoner):
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    fofs = dataloader.get_fof_list()

    # conduct reasoning
    with tqdm.tqdm(fofs, desc=desc) as t:
        for fof in t:
            with torch.no_grad():
                reasoner.initialize_with_query(fof)
                reasoner.estimate_variable_embeddings()
                batch_fvar_emb = reasoner.get_ent_emb('f')
                batch_entity_rankings = nbp.get_all_entity_rankings(
                    batch_fvar_emb, score="cos")
            # [batch_size, num_entities]
            compute_evaluation_scores(
                fof, batch_entity_rankings, metric[fof.lstr])
            t.set_postfix({'lstr': fof.lstr})

            sum_metric = defaultdict(dict)
            for lstr in metric:
                for score_name in metric[lstr]:
                    sum_metric[lstr2name[lstr]][score_name] = float(
                        np.mean(metric[lstr][score_name]))

        postfix = {}
        for name in ['1p', '2p', '3p', '2i', 'inp']:
            if name in sum_metric:
                postfix[name + '_hit3'] = sum_metric[name]['hit3']
        torch.cuda.empty_cache()

    sum_metric['epoch'] = e
    logging.info(f"[{desc}][final] {json.dumps(sum_metric)}")


def evaluate_by_nearest_search_DNF(
        e,
        desc,
        dataloader,
        nbp: NeuralBinaryPredicate,
        reasoner: GradientEFOReasoner):
    # first level key: lstr
    # second level key: metric name
    metric = defaultdict(lambda: defaultdict(list))
    fofs = dataloader.get_fof_list()

    # conduct reasoning
    with tqdm.tqdm(fofs, desc=desc) as t:
        for fof in t:
            with torch.no_grad():
                if fof.lstr == '(r1(s1,f))|(r2(s2,f))':
                    # query1
                    lformula1 = parse_lstr_to_lformula('r1(s1,f)')
                    query1 = EFO1Query(lformula1)
                    query1.term_grounded_entity_id_dict['s1'] = fof.term_grounded_entity_id_dict['s1']
                    query1.term_grounded_entity_id_dict['f'] = fof.term_grounded_entity_id_dict['f']
                    query1.pred_grounded_relation_id_dict['r1'] = fof.pred_grounded_relation_id_dict['r1']
                    query1.easy_answer_list = fof.easy_answer_list
                    query1.hard_answer_list = fof.hard_answer_list
                    query1.noisy_answer_list = fof.noisy_answer_list
                    reasoner.initialize_with_query(query1)
                    reasoner.estimate_variable_embeddings()
                    batch_fvar_emb1 = reasoner.get_ent_emb('f')
                    # query2
                    lformula2 = parse_lstr_to_lformula('r1(s1,f)')
                    query2 = EFO1Query(lformula2)
                    query2.term_grounded_entity_id_dict['s1'] = fof.term_grounded_entity_id_dict['s2']
                    query2.term_grounded_entity_id_dict['f'] = fof.term_grounded_entity_id_dict['f']
                    query2.pred_grounded_relation_id_dict['r1'] = fof.pred_grounded_relation_id_dict['r2']
                    query2.easy_answer_list = fof.easy_answer_list
                    query2.hard_answer_list = fof.hard_answer_list
                    query2.noisy_answer_list = fof.noisy_answer_list
                    reasoner.initialize_with_query(query2)
                    reasoner.estimate_variable_embeddings()
                    batch_fvar_emb2 = reasoner.get_ent_emb('f')
                    batch_entity_rankings = nbp.get_all_entity_rankings_dnf(batch_fvar_emb1, batch_fvar_emb2, score="cos")
                elif fof.lstr == '((r1(s1,e1))|(r2(s2,e1)))&(r3(e1,f))' or fof.lstr == '((r1(s1,e1))&(r3(e1,f)))|((r2(s2,e1))&(r3(e1,f)))':
                    # query1
                    lformula1 = parse_lstr_to_lformula('(r1(s1,e1))&(r2(e1,f))')  # (r1(s1,e1))&(r3(e1,f))
                    query1 = EFO1Query(lformula1)
                    query1.term_grounded_entity_id_dict['s1'] = fof.term_grounded_entity_id_dict['s1']
                    query1.term_grounded_entity_id_dict['f'] = fof.term_grounded_entity_id_dict['f']
                    query1.term_grounded_entity_id_dict['e1'] = fof.term_grounded_entity_id_dict['e1']
                    query1.pred_grounded_relation_id_dict['r1'] = fof.pred_grounded_relation_id_dict['r1']
                    query1.pred_grounded_relation_id_dict['r2'] = fof.pred_grounded_relation_id_dict['r3']
                    query1.easy_answer_list = fof.easy_answer_list
                    query1.hard_answer_list = fof.hard_answer_list
                    query1.noisy_answer_list = fof.noisy_answer_list
                    reasoner.initialize_with_query(query1)
                    reasoner.estimate_variable_embeddings()
                    batch_fvar_emb1 = reasoner.get_ent_emb('f')
                    # query2
                    lformula2 = parse_lstr_to_lformula('(r1(s1,e1))&(r2(e1,f))')  # (r2(s2,e1))&(r3(e1,f))
                    query2 = EFO1Query(lformula2)
                    query2.term_grounded_entity_id_dict['s1'] = fof.term_grounded_entity_id_dict['s2']
                    query2.term_grounded_entity_id_dict['f'] = fof.term_grounded_entity_id_dict['f']
                    query2.pred_grounded_relation_id_dict['r1'] = fof.pred_grounded_relation_id_dict['r2']
                    query2.term_grounded_entity_id_dict['e1'] = fof.term_grounded_entity_id_dict['e1']
                    query2.pred_grounded_relation_id_dict['r2'] = fof.pred_grounded_relation_id_dict['r3']
                    query2.easy_answer_list = fof.easy_answer_list
                    query2.hard_answer_list = fof.hard_answer_list
                    query2.noisy_answer_list = fof.noisy_answer_list
                    reasoner.initialize_with_query(query2)
                    reasoner.estimate_variable_embeddings()
                    batch_fvar_emb2 = reasoner.get_ent_emb('f')
                    batch_entity_rankings = nbp.get_all_entity_rankings_dnf(batch_fvar_emb1, batch_fvar_emb2, score="cos")
                else:
                    reasoner.initialize_with_query(fof)
                    reasoner.estimate_variable_embeddings()
                    batch_fvar_emb = reasoner.get_ent_emb('f')
                    batch_entity_rankings = nbp.get_all_entity_rankings(batch_fvar_emb, score="cos")
            # [batch_size, num_entities]
            compute_evaluation_scores(
                fof, batch_entity_rankings, metric[fof.lstr])
            t.set_postfix({'lstr': fof.lstr})

            sum_metric = defaultdict(dict)
            for lstr in metric:
                for score_name in metric[lstr]:
                    sum_metric[lstr2name[lstr]][score_name] = float(
                        np.mean(metric[lstr][score_name]))

        postfix = {}
        for name in ['1p', '2p', '3p', '2i', 'inp']:
            if name in sum_metric:
                postfix[name + '_hit3'] = sum_metric[name]['hit3']
        torch.cuda.empty_cache()

    sum_metric['epoch'] = e
    logging.info(f"[{desc}][final] {json.dumps(sum_metric)}")



if __name__ == "__main__":
    # * parse argument
    args = parser.parse_args()
    set_global_seed(args.seed)

    # * prepare the logger
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(filename=osp.join(args.output_dir, 'output.log'),
                        format='%(asctime)s %(message)s',
                        level=logging.INFO,
                        filemode='wt')

    # * initialize the kgindex
    kgidx = KGIndex.load(
        osp.join(args.task_folder, "kgindex.json"))

    # * load neural binary predicate
    print(f"loading the nbp {args.model_name}")
    nbp = get_nbp_class(args.model_name)(
        num_entities=kgidx.num_entities,
        num_relations=kgidx.num_relations,
        embedding_dim=args.embedding_dim,
        p=args.p,
        margin=args.margin,
        scale=args.scale,
        device=args.device)

    if args.checkpoint_path:
        print("loading model from", args.checkpoint_path)
        nbp.load_state_dict(torch.load(args.checkpoint_path), strict=True)

    nbp.to(args.device)
    print(f"model loaded from {args.checkpoint_path}")

    if args.freeze_nlp:
        for param in nbp.parameters():  # frozen pre-trained neural link predictor
            param.requires_grad = False

    # * load the dataset, by default, we load the dataset to test
    print("loading dataset")
    if args.eval_queries:
        eval_queries = [name2lstr[tq] for tq in args.eval_queries]
    else:
        eval_queries = list(name2lstr.values())
    print("eval queries", eval_queries)

    valid_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'valid-qaa.json'),
        target_lstr=eval_queries,
        batch_size=args.batch_size_eval_dataloader,
        shuffle=False,
        num_workers=0)

    test_dataloader = QueryAnsweringSeqDataLoader(
        osp.join(args.task_folder, 'test-qaa.json'),
        target_lstr=eval_queries,
        batch_size=args.batch_size_eval_dataloader,
        shuffle=False,
        num_workers=0)

    if args.reasoner in ['gradient', 'beam']:
        # for those reasoners without training
        tnorm = Tnorm.get_tnorm(args.tnorm)

        reasoner = GradientEFOReasoner(
            nbp,
            tnorm,
            reasoning_rate=args.reasoning_rate,
            reasoning_steps=args.reasoning_steps,
            reasoning_optimizer=args.reasoning_optimizer)

        evaluate_by_search_emb_then_rank_truth_value(
            -1, f"valuate validate set", valid_dataloader, nbp, reasoner)
        evaluate_by_search_emb_then_rank_truth_value(
            -1, f"evaluate test set", test_dataloader, nbp, reasoner)
    elif args.reasoner == 'lmpt':
        if args.train_queries:
            train_queries = [name2lstr[tq] for tq in args.train_queries]
        else:
            train_queries = list(name2lstr.values())
        print("train queries", train_queries)

        train_dataloader = QueryAnsweringSeqDataLoader(
            osp.join(args.task_folder, 'train-qaa.json'),
            target_lstr=train_queries,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0)
        lgnn_layer = LMPTLayer(hidden_dim=args.hidden_dim,
                               nbp=nbp,
                               layers=args.num_layers,
                               agg_func=args.agg_func,
                               pre_norm=args.pre_norm
                               )

        if args.checkpoint_path_gnn:
            print("loading gnn model from", args.checkpoint_path_gnn)
            lgnn_layer.load_state_dict(torch.load(args.checkpoint_path_gnn), strict=True)

        lgnn_layer.to(nbp.device)

        reasoner = LMPTReasoner(nbp, lgnn_layer, depth_shift=args.depth_shift)
        print(lgnn_layer)
        print("Is entity embedding optimized ?",
              lgnn_layer.nbp._entity_embedding.weight.requires_grad)
        print("Is relation embedding optimized ?", lgnn_layer.nbp._relation_embedding.weight.requires_grad)
        optimizer_estimator = getattr(torch.optim, args.optimizer)(
            lgnn_layer.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_estimator, 50, 0.1)

        for e in range(1, 1 + args.epoch):
            reasoner.lgnn_layer.train()
            train_gnn(f"epoch {e}", train_dataloader, nbp, reasoner, optimizer_estimator, args)
            scheduler.step()
            if e % 5 == 0:
                reasoner.lgnn_layer.eval()
                evaluate_by_nearest_search_DNF(e, f"NN evaluate validate set epoch {e}",
                                           valid_dataloader, nbp, reasoner)
                evaluate_by_nearest_search_DNF(e, f"NN evaluate test set epoch {e}",
                                           test_dataloader, nbp, reasoner)

                save_name = os.path.join(args.output_dir,
                                         f'gnn-{e}.ckpt')
                torch.save(lgnn_layer.state_dict(), save_name)

                last_name = os.path.join(args.output_dir,
                                         f'gnn-last.ckpt')
                torch.save(lgnn_layer.state_dict(), last_name)

        if args.epoch == 0:
            reasoner.lgnn_layer.eval()
            evaluate_by_nearest_search_DNF(args.epoch, f"NN evaluate validate set",
                                       valid_dataloader, nbp, reasoner)
            evaluate_by_nearest_search_DNF(args.epoch, f"NN evaluate test set ",
                                       test_dataloader, nbp, reasoner)
    elif args.reasoner == 'ilin':
        if args.train_queries:
            train_queries = [name2lstr[tq] for tq in args.train_queries]
        else:
            train_queries = list(name2lstr.values())
        print("train queries", train_queries)

        train_dataloader = QueryAnsweringSeqDataLoader(
            osp.join(args.task_folder, 'train-qaa.json'),
            target_lstr=train_queries,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0)
        lgnn_layer = ILINLayer(hidden_dim=args.hidden_dim,
                                nbp=nbp,
                                layers=args.num_layers,
                                agg_func=args.agg_func
                                )

        if args.checkpoint_path_gnn:
            print("loading gnn model from", args.checkpoint_path_gnn)
            lgnn_layer.load_state_dict(torch.load(args.checkpoint_path_gnn), strict=True)

        lgnn_layer.to(nbp.device)

        reasoner = ILINReasoner(nbp, lgnn_layer, depth_shift=args.depth_shift)
        print(lgnn_layer)

        _lstr2name = {parse_lstr_to_lformula(k).lstr: v for k, v in beta_lstr2name.items()}
        all_query_names = list(_lstr2name.values())
        task_weight_manager = TaskWeightManager(all_query_names, args.device, alpha=args.task_weight_alpha)

        print("Is entity embedding optimized ?",
              lgnn_layer.nbp._entity_embedding.weight.requires_grad)
        print("Is relation embedding optimized ?", lgnn_layer.nbp._relation_embedding.weight.requires_grad)
        optimizer_estimator = getattr(torch.optim, args.optimizer)(
            lgnn_layer.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_estimator, gamma=0.96)

        for e in range(1, 1 + args.epoch):
            reasoner.lgnn_layer.train()
            train_gnn(f"epoch {e}", train_dataloader, nbp, reasoner, optimizer_estimator, task_weight_manager, args)
            scheduler.step()
            if e % 5 == 0:
                reasoner.lgnn_layer.eval()
                evaluate_by_nearest_search_DNF(e, f"NN evaluate validate set epoch {e}",
                                           valid_dataloader, nbp, reasoner)
                evaluate_by_nearest_search_DNF(e, f"NN evaluate test set epoch {e}",
                                           test_dataloader, nbp, reasoner)

                save_name = os.path.join(args.output_dir,
                                         f'gnn-{e}.ckpt')
                torch.save(lgnn_layer.state_dict(), save_name)

                last_name = os.path.join(args.output_dir,
                                         f'gnn-last.ckpt')
                torch.save(lgnn_layer.state_dict(), last_name)

        if args.epoch == 0:
            reasoner.lgnn_layer.eval()
            evaluate_by_nearest_search_DNF(args.epoch, f"NN evaluate validate set",
                                       valid_dataloader, nbp, reasoner)
            evaluate_by_nearest_search_DNF(args.epoch, f"NN evaluate test set ",
                                       test_dataloader, nbp, reasoner)
    elif args.reasoner == 'clmpt':
        if args.train_queries:
            train_queries = [name2lstr[tq] for tq in args.train_queries]
        else:
            train_queries = list(name2lstr.values())
        print("train queries", train_queries)

        train_dataloader = QueryAnsweringSeqDataLoader(
            osp.join(args.task_folder, 'train-qaa.json'),
            target_lstr=train_queries,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0)
        lgnn_layer = CLMPTLayer(hidden_dim=args.hidden_dim,
                                nbp=nbp,
                                layers=args.num_layers,
                                agg_func=args.agg_func,
                                pre_norm=args.pre_norm
                                )

        if args.checkpoint_path_gnn:
            print("loading gnn model from", args.checkpoint_path_gnn)
            lgnn_layer.load_state_dict(torch.load(args.checkpoint_path_gnn), strict=True)

        lgnn_layer.to(nbp.device)

        reasoner = CLMPTReasoner(nbp, lgnn_layer, depth_shift=args.depth_shift)
        print(lgnn_layer)
        print("Is entity embedding optimized ?",
              lgnn_layer.nbp._entity_embedding.weight.requires_grad)
        print("Is relation embedding optimized ?", lgnn_layer.nbp._relation_embedding.weight.requires_grad)
        optimizer_estimator = getattr(torch.optim, args.optimizer)(
            lgnn_layer.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_estimator, 50, 0.1)

        for e in range(1, 1 + args.epoch):
            reasoner.lgnn_layer.train()
            train_gnn(f"epoch {e}", train_dataloader, nbp, reasoner, optimizer_estimator, args)
            scheduler.step()
            if e % 5 == 0:
                reasoner.lgnn_layer.eval()
                evaluate_by_nearest_search_DNF(e, f"NN evaluate validate set epoch {e}",
                                           valid_dataloader, nbp, reasoner)
                evaluate_by_nearest_search_DNF(e, f"NN evaluate test set epoch {e}",
                                           test_dataloader, nbp, reasoner)

                save_name = os.path.join(args.output_dir,
                                         f'gnn-{e}.ckpt')
                torch.save(lgnn_layer.state_dict(), save_name)

                last_name = os.path.join(args.output_dir,
                                         f'gnn-last.ckpt')
                torch.save(lgnn_layer.state_dict(), last_name)

        if args.epoch == 0:
            reasoner.lgnn_layer.eval()
            evaluate_by_nearest_search_DNF(args.epoch, f"NN evaluate validate set",
                                       valid_dataloader, nbp, reasoner)
            evaluate_by_nearest_search_DNF(args.epoch, f"NN evaluate test set ",
                                       test_dataloader, nbp, reasoner)

    elif args.reasoner == 'lmpnn':
        # for those reasoners with training
        if args.train_queries:
            train_queries = [name2lstr[tq] for tq in args.train_queries]
        else:
            train_queries = list(name2lstr.values())
        print("train queries", train_queries)

        train_dataloader = QueryAnsweringSeqDataLoader(
            osp.join(args.task_folder, 'train-qaa.json'),
            target_lstr=train_queries,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0)
        lgnn_layer = LogicalMPNNLayer(hidden_dim=args.hidden_dim,
                                      nbp=nbp,
                                      layers=args.num_layers,
                                      eps=args.eps,
                                      agg_func=args.agg_func
                                      )

        if args.checkpoint_path_gnn:
            print("loading gnn model from", args.checkpoint_path_gnn)
            lgnn_layer.load_state_dict(torch.load(args.checkpoint_path_gnn), strict=True)

        lgnn_layer.to(nbp.device)

        reasoner = LMPNNReasoner(nbp, lgnn_layer, depth_shift=args.depth_shift)
        print(lgnn_layer)
        print("Is entity embedding optimized ?",
              lgnn_layer.nbp._entity_embedding.weight.requires_grad)
        print("Is relation embedding optimized ?", lgnn_layer.nbp._relation_embedding.weight.requires_grad)
        optimizer_estimator = getattr(torch.optim, args.optimizer)(
            lgnn_layer.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_estimator, 50, 0.1)

        for e in range(1, 1+args.epoch):
            train_gnn(f"epoch {e}", train_dataloader, nbp, reasoner, optimizer_estimator, args)
            scheduler.step()
            if e % 5 == 0:
                evaluate_by_nearest_search(e, f"NN evaluate validate set epoch {e}",
                                           valid_dataloader, nbp, reasoner)
                evaluate_by_nearest_search(e, f"NN evaluate test set epoch {e}",
                                           test_dataloader, nbp, reasoner)

                save_name = os.path.join(args.output_dir,
                                        f'gnn-{e}.ckpt')
                torch.save(lgnn_layer.state_dict(), save_name)

                last_name = os.path.join(args.output_dir,
                                        f'gnn-last.ckpt')
                torch.save(lgnn_layer.state_dict(), last_name)

        if args.epoch == 0:
            evaluate_by_nearest_search(args.epoch, f"NN evaluate validate set",
                                        valid_dataloader, nbp, reasoner)
            evaluate_by_nearest_search(args.epoch, f"NN evaluate test set ",
                                        test_dataloader, nbp, reasoner)
    elif args.reasoner == 'lmpnn_dnf':
        # for those reasoners with training
        if args.train_queries:
            train_queries = [name2lstr[tq] for tq in args.train_queries]
        else:
            train_queries = list(name2lstr.values())
        print("train queries", train_queries)

        train_dataloader = QueryAnsweringSeqDataLoader(
            osp.join(args.task_folder, 'train-qaa.json'),
            target_lstr=train_queries,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0)
        lgnn_layer = LogicalMPNNLayer(hidden_dim=args.hidden_dim,
                                      nbp=nbp,
                                      layers=args.num_layers,
                                      eps=args.eps,
                                      agg_func=args.agg_func
                                      )

        if args.checkpoint_path_gnn:
            print("loading gnn model from", args.checkpoint_path_gnn)
            lgnn_layer.load_state_dict(torch.load(args.checkpoint_path_gnn), strict=True)

        lgnn_layer.to(nbp.device)

        reasoner = LMPNNReasoner(nbp, lgnn_layer, depth_shift=args.depth_shift)
        print(lgnn_layer)
        print("Is entity embedding optimized ?",
              lgnn_layer.nbp._entity_embedding.weight.requires_grad)
        print("Is relation embedding optimized ?", lgnn_layer.nbp._relation_embedding.weight.requires_grad)
        optimizer_estimator = getattr(torch.optim, args.optimizer)(
            lgnn_layer.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_estimator, 50, 0.1)

        for e in range(1, 1+args.epoch):
            train_gnn(f"epoch {e}", train_dataloader, nbp, reasoner, optimizer_estimator, args)
            scheduler.step()
            if e % 5 == 0:
                evaluate_by_nearest_search_DNF(e, f"NN evaluate validate set epoch {e}",
                                           valid_dataloader, nbp, reasoner)
                evaluate_by_nearest_search_DNF(e, f"NN evaluate test set epoch {e}",
                                           test_dataloader, nbp, reasoner)

                save_name = os.path.join(args.output_dir,
                                        f'gnn-{e}.ckpt')
                torch.save(lgnn_layer.state_dict(), save_name)

                last_name = os.path.join(args.output_dir,
                                        f'gnn-last.ckpt')
                torch.save(lgnn_layer.state_dict(), last_name)

        if args.epoch == 0:
            evaluate_by_nearest_search_DNF(args.epoch, f"NN evaluate validate set",
                                        valid_dataloader, nbp, reasoner)
            evaluate_by_nearest_search_DNF(args.epoch, f"NN evaluate test set ",
                                        test_dataloader, nbp, reasoner)
