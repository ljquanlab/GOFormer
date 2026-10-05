"""
评估方式：使用 GO 本体祖先传播后处理（new_compute_performance_deepgoplus）
"""
import os
import warnings
import math
import pickle as pkl
from collections import deque, Counter

import numpy as np
import scipy.sparse as ssp
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import logging

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.metrics')



def compute_metrics(predictions, targets):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    target_sum = targets.sum(axis=1)
    fmax_val = 0.0
    best_threshold = 0.0
    precisions = []
    recalls = []
    for cut in [c / 100.0 for c in range(101)]:
        cut_sc = (predictions >= cut).astype(np.int32)
        correct = (cut_sc * targets).sum(axis=1)
        pred_sum = cut_sc.sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p = correct / pred_sum
            r = correct / target_sum
            valid_p = p[~np.isnan(p)]
            avg_p = np.average(valid_p) if len(valid_p) > 0 else 0.0
            avg_r = np.average(r)
        precisions.append(avg_p)
        recalls.append(avg_r)
        try:
            if avg_p + avg_r > 0.0:
                f_measure = 2 * avg_p * avg_r / (avg_p + avg_r)
            else:
                f_measure = 0.0
            if f_measure > fmax_val:
                fmax_val = f_measure
                best_threshold = cut
        except ZeroDivisionError:
            pass
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls_sorted = recalls[sorted_index]
    precisions_sorted = precisions[sorted_index]
    aupr_val = np.trapz(precisions_sorted, recalls_sorted)
    return {
        'fmax': fmax_val,
        'best_threshold': best_threshold,
        'aupr': aupr_val
    }
def print_metrics(metrics, prefix=""):
    """
    打印精简后的 CAFA 指标
    """
    logging.debug(f"\n{prefix} CAFA Metrics:")
    logging.debug(f"  Fmax: {metrics['fmax']:.4f} (threshold={metrics['threshold']:.2f})")
    logging.debug(f"  AUPR: {metrics['aupr']:.4f}")

ROOT_GO_TERMS = {'GO:0003674', 'GO:0008150', 'GO:0005575'}
BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {'cc': CELLULAR_COMPONENT, 'mf': MOLECULAR_FUNCTION, 'bp': BIOLOGICAL_PROCESS}
NAMESPACES = {'cc': 'cellular_component', 'mf': 'molecular_function', 'bp': 'biological_process'}


class Ontology:
    def __init__(self, filename='data/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None

    def has_term(self, term_id):
        return term_id in self.ont

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            min_n = n if len(parents) == 0 else min([cnt[x] for x in parents])
            self.ic[go_id] = math.log(min_n / n, 2)

    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('IC not yet calculated')
        return self.ic.get(go_id, 0.0)

    def load(self, filename, with_rels):
        ont = {}
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = {'is_a': [], 'part_of': [], 'regulates': [],
                           'alt_ids': [], 'is_obsolete': False}
                    continue
                elif line == '[Typedef]':
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(': ', 1)
                    if len(l) < 2:
                        continue
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        if it[0] == 'part_of':
                            obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
        if obj is not None:
            ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id].get('is_obsolete', False):
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    ont[p_id].setdefault('children', set()).add(term_id)
        return ont

    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque([term_id])
        while q:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set

    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        return {p for p in self.ont[term_id]['is_a'] if p in self.ont}

    def get_namespace_terms(self, namespace):
        return {go_id for go_id, obj in self.ont.items()
                if obj.get('namespace') == namespace}

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

def fmax(targets, scores):
    targets = ssp.csr_matrix(targets)
    fmax_ = 0.0, 0.0
    precisions, recalls = [], []
    for cut in (c / 100 for c in range(101)):
        cut_sc = ssp.csr_matrix((scores >= cut).astype(np.int32))
        correct = cut_sc.multiply(targets).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p = correct / cut_sc.sum(axis=1)
            r = correct / targets.sum(axis=1)
            p = np.average(p[np.invert(np.isnan(p))])
            r = np.average(r)
        if np.isnan(p):
            precisions.append(0.0)
            recalls.append(r)
            continue
        precisions.append(p)
        recalls.append(r)
        try:
            fmax_ = max(fmax_, (2 * p * r / (p + r) if p + r > 0.0 else 0.0, cut))
        except ZeroDivisionError:
            pass
    return fmax_[0], fmax_[1], precisions, recalls



def compute_performance(targets,scores):
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    else:
        targets = np.asarray(targets)

    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    else:
        scores = np.asarray(scores)
    result_fmax, result_t, precisions, recalls = fmax(targets, scores)
    
    # AUPR
    precisions_arr = np.array(precisions)
    recalls_arr    = np.array(recalls)
    sorted_idx     = np.argsort(recalls_arr)
    result_aupr    = np.trapz(precisions_arr[sorted_idx], recalls_arr[sorted_idx])

    return {'fmax': result_fmax, 'aupr': result_aupr, 'threshold': result_t}

def new_compute_performance_deepgoplus(test_df, go_file, ont, with_relations=True):
    go = Ontology(go_file, with_rels=with_relations)
    go_set = go.get_namespace_terms(NAMESPACES[ont])
    go_set.discard(FUNC_DICT[ont])

    labels = list(go_set)
    goid_idx = {goid: idx for idx, goid in enumerate(labels)}

    pred_scores, true_scores = [], []

    for row in test_df.itertuples():
        vals = [0] * len(labels)
        annots = set()
        for go_id in row.gos:
            if go.has_term(go_id):
                annots |= go.get_anchestors(go_id)
        for go_id in annots:
            if go_id in go_set:
                vals[goid_idx[go_id]] = 1
        true_scores.append(vals)
        vals = [-1.0] * len(labels)
        for item, score in row.predictions.items():
            if item in go_set:
                vals[goid_idx[item]] = max(score, vals[goid_idx[item]])
            for go_id in go.get_anchestors(item):
                if go_id in go_set:
                    vals[goid_idx[go_id]] = max(vals[goid_idx[go_id]], score)
        pred_scores.append(vals)

    pred_scores = np.array(pred_scores)
    true_scores = np.array(true_scores)

    result_fmax, result_t, precisions, recalls = fmax(true_scores, pred_scores)

    # AUPR
    precisions_arr = np.array(precisions)
    recalls_arr    = np.array(recalls)
    sorted_idx     = np.argsort(recalls_arr)
    result_aupr    = np.trapz(precisions_arr[sorted_idx], recalls_arr[sorted_idx])

    return result_fmax, result_aupr, result_t


def build_memory_bank(model, train_loader, device):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(train_loader, desc="Building Memory Bank", leave=False)
        for batch in pbar:
            embeddings = batch['embedding'].to(device)
            labels = batch['labels'].to(device)
            fixed_null_input = batch['fixed_null_input'].to(device).float()
            struct = batch['struct_embedding'].to(device)

            *_,fused_protein_features = model(embeddings, fixed_null_input=fixed_null_input, struct=struct, return_features=True)
            all_features.append(fused_protein_features.cpu())
            all_labels.append(labels.cpu()) 
            
    memory_features = torch.cat(all_features, dim=0) # [N_train, D]
    memory_labels = torch.cat(all_labels, dim=0)     # [N_train, NUM_LABELS]
    memory_features = F.normalize(memory_features, p=2, dim=-1)
    
    return memory_features.to(device), memory_labels.to(device)

def apply_knn_probabilities(p_model, test_features, memory_features, memory_labels, k=10, alpha=0.6, temperature=0.05):

    test_features = F.normalize(test_features, p=2, dim=-1)
    sim_matrix = torch.matmul(test_features, memory_features.t()) 
    topk_sim, topk_indices = torch.topk(sim_matrix, k, dim=-1) # [B, K]
    knn_weights = F.softmax(topk_sim / temperature, dim=-1) # [B, K]
    p_knn = torch.einsum('bk,bkl->bl', knn_weights, memory_labels[topk_indices]) # [B, NUM_LABELS]
    p_final = alpha * p_model + (1 - alpha) * p_knn
    return p_final


@torch.no_grad()
def evaluate_propagated(model, data_loader, device, train_loader):
    model.eval()
    all_probs       = []   # list of [B, N] tensor (cpu)
    all_labels      = []   # list of [B, N] tensor (cpu)
    all_protein_ids = []   # list of str        

    KNN_K = 10
    KNN_ALPHA = 0.7 
    KNN_TEMP = 0.05
    memory_features, memory_labels = build_memory_bank(model, train_loader, device)

    for batch in tqdm(data_loader, desc="Collecting predictions",leave=False):
        embeddings   = batch['embedding'].to(device)
        labels       = batch['labels'].to(device)
        fixed_null_input = batch['fixed_null_input'].to(device).float()
        struct       = batch['struct_embedding'].to(device)
        protein_ids  = batch['protein_id']          # list[str]
        logits, *_, fused_protein_features = model(
            embeddings, fixed_null_input=fixed_null_input, struct=struct, return_features=True
        )
        p_model = torch.sigmoid(logits)
        p_final = apply_knn_probabilities(
                p_model=p_model,
                test_features=fused_protein_features,
                memory_features=memory_features,
                memory_labels=memory_labels,
                k=KNN_K,
                alpha=KNN_ALPHA,
                temperature=KNN_TEMP
            )
        all_probs.append(p_final.cpu())
        all_labels.append(labels.cpu())
        all_protein_ids.extend(protein_ids)
    all_probs  = torch.cat(all_probs,  dim=0).numpy()   # [M, N]
    all_labels = torch.cat(all_labels, dim=0).numpy()   # [M, N]
    del memory_features
    del memory_labels
    torch.cuda.empty_cache()

    return all_probs, all_labels, all_protein_ids

def compute_fmax_and_aupr_propagated(all_probs, all_labels, all_protein_ids, idx_to_go, go_obo_path, ont='bp'):

    N = all_probs.shape[1]
    rows = []
    for i, pid in enumerate(all_protein_ids):
        true_indices = np.where(all_labels[i] > 0.5)[0]
        true_go_ids  = [idx_to_go[idx] for idx in true_indices if idx in idx_to_go]

        pred_dict = {
            idx_to_go[j]: float(all_probs[i, j])
            for j in range(N) if j in idx_to_go
        }
        rows.append({'protein_id': pid, 'gos': true_go_ids, 'predictions': pred_dict})

    test_df = pd.DataFrame(rows)
    print(f"  Built DataFrame: {len(test_df)} proteins, evaluating with GO ancestor propagation...")
    result_fmax, result_aupr, result_t = new_compute_performance_deepgoplus(
        test_df, go_obo_path, ont, with_relations=True
    )

    return {'fmax': result_fmax, 'aupr': result_aupr, 'threshold': result_t}

def save_results_to_pkl(all_probs, all_labels, all_protein_ids, idx_to_go, metric, output_path):
    results = {
        'protein_ids': all_protein_ids,
        'predictions': all_probs,
        'labels': all_labels,
        'threshold': metric['threshold'],
        'metric': metric,
        'idx_to_go': idx_to_go
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_pickle(results, output_path)
    print(f"Saved predictions to {output_path}")
    