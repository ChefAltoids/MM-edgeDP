"""
baselines.py — Train all 5 comparison methods + run privacy attacks for Phase 3.

Methods:
  Gaussian SGC   (ε,δ)-DP  — noisy 1-hop mean aggregation, MLP classifier
  Edge RR        ε-DP      — randomized response on edges, GCN classifier
  GAP-EDP        (ε,δ)-DP  — Sajadmanesh et al. multi-hop Gaussian PMA
  Public GCN     no DP     — GCN trained on public data only (no private edges)
  MM-EdgeDP      ε-DP      — exponential mechanism + prototype dictionary

Privacy attacks (per method × dataset):
  Threat Model B  white-box  — penultimate-layer embedding, Hadamard pairs
  Threat Model A  black-box  — softmax output, disagreement + co-activation

Usage:
  python baselines.py                                  # all datasets, defaults
  python baselines.py --datasets Cora                  # single dataset
  python baselines.py --epsilon 2.0 --seeds 0 1 2
  python baselines.py --dry-run                        # print configs, no training

Results are written to outputs/baseline_results.csv.
"""

import argparse
import csv
import math
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import coalesce, dropout_edge, subgraph, to_undirected

from experiments import (
    DEFAULT_CONFIG,
    PrototypeAssignmentDataset,
    StandardGCN,
    _get_device,
    build_candidate_pool,
    build_validation_loader,
    compute_public_query_features,
    diversify_and_cover,
    hard_filter_pool,
    load_dataset,
    one_hop_mean_aggregate,
    set_seed,
    split_public_pool_query,
    stratified_split_indices,
    synthesize_edge_dp_assignments,
    train_public_encoder,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DELTA = 1e-5
N_ATTACK_PAIRS = 1721
OUTPUT_CSV = Path('outputs/baseline_results.csv')

RESULT_KEYS = [
    'dataset', 'method', 'epsilon', 'public_frac', 'dict_per_class', 'seed',
    'val_acc', 'wb_auc', 'bb_auc', 'dp_type', 'status', 'error',
]

DEFAULT_PHASE3_CONFIG = {
    **DEFAULT_CONFIG,
    'epsilon': 1.0,
    'public_frac': 0.20,
    'dict_per_class': 16,
}

# ---------------------------------------------------------------------------
# Model definitions (baseline-specific)
# ---------------------------------------------------------------------------

class NodeMLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x):
        return self.net(x)


class NodeGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, num_classes)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GAPEncoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_classes, dropout=0.5):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )
        self.head = nn.Linear(out_dim, num_classes)
    def encode(self, x):
        return self.enc(x)
    def forward(self, x):
        return self.head(self.encode(x))


# ---------------------------------------------------------------------------
# Mechanism helpers
# ---------------------------------------------------------------------------

def _gauss_sigma(epsilon, delta, sensitivity=1.0):
    return math.sqrt(2.0 * math.log(1.25 / delta)) * sensitivity / epsilon


def _gap_sigma(epsilon, delta, K=1, sensitivity=1.0):
    c = math.sqrt(2.0 * K * math.log(1.0 / delta)) * sensitivity
    def eps_fn(s):
        return K * sensitivity**2 / (2.0 * s**2) + c / s
    lo, hi = 1e-8, 1e5
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if eps_fn(mid) > epsilon: lo = mid
        else: hi = mid
    return hi


_MAX_RR_NODES = 5000  # O(n²) — segfaults on large graphs


def _apply_edge_rr(edge_index, node_ids, epsilon, num_nodes, seed):
    n = node_ids.numel()
    if n > _MAX_RR_NODES:
        raise RuntimeError(
            f"Edge-RR requires O(n²) memory ({n} train nodes > limit {_MAX_RR_NODES}). "
            "Skip this method for large graphs."
        )
    torch.manual_seed(seed)
    p_keep = math.exp(epsilon) / (math.exp(epsilon) + 1.0)
    p_add  = 1.0 / (math.exp(epsilon) + 1.0)
    node_list = node_ids.tolist()
    local = {v: i for i, v in enumerate(node_list)}
    node_set = set(node_list)
    adj = torch.zeros(n, n, dtype=torch.bool)
    for r, c in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if r in node_set and c in node_set and r < c:
            adj[local[r], local[c]] = True
    ti, tj = torch.triu_indices(n, n, offset=1)
    exist = adj[ti, tj]
    probs = torch.where(exist, torch.full(exist.shape, p_keep), torch.full(exist.shape, p_add))
    keep = torch.rand(probs.shape) < probs
    ki, kj = ti[keep], tj[keep]
    src = torch.cat([node_ids[ki], node_ids[kj]])
    dst = torch.cat([node_ids[kj], node_ids[ki]])
    return torch.stack([src, dst])


@torch.no_grad()
def _gap_private_agg(edge_index, x_norm, node_ids, sigma, num_nodes, K, seed):
    torch.manual_seed(seed)
    eu = coalesce(to_undirected(edge_index, num_nodes=num_nodes), num_nodes=num_nodes)
    dst, src = eu
    d = x_norm.size(1)
    x_hat = x_norm.clone()
    hops = []
    for _ in range(K):
        X_agg = torch.zeros_like(x_hat)
        X_agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, d), x_hat[src])
        X_tilde = X_agg.clone()
        X_tilde[node_ids] = X_tilde[node_ids] + torch.randn(node_ids.numel(), d) * sigma
        norms = X_tilde.norm(dim=1, keepdim=True).clamp(min=1e-8)
        x_hat = X_tilde / norms
        hops.append(x_hat[node_ids].clone())
    return hops


# ---------------------------------------------------------------------------
# Training functions — each returns (model_obj, val_acc)
# ---------------------------------------------------------------------------

def _train_node_mlp(x_tr, y_tr, x_vl, y_vl, in_dim, hidden, num_classes,
                     epochs, lr, wd, dropout, seed, device):
    set_seed(seed)
    model = NodeMLP(in_dim, hidden, num_classes, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    xtr, ytr = x_tr.to(device), y_tr.to(device)
    xvl, yvl = x_vl.to(device), y_vl.to(device)
    best = 0.0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(xtr), ytr).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        best = float((model(xvl).argmax(1) == yvl).float().mean())
    return model, best


def _train_node_gcn(x_features, tr_ei, val_ei, labels, train_ids, val_ids,
                     num_classes, num_nodes, hidden, epochs, lr, dropout, seed, device):
    set_seed(seed)
    tr_sub, _ = subgraph(train_ids, tr_ei, relabel_nodes=True, num_nodes=num_nodes)
    tr_sub = coalesce(to_undirected(tr_sub, num_nodes=train_ids.numel()), num_nodes=train_ids.numel())
    vl_sub, _ = subgraph(val_ids, val_ei, relabel_nodes=True, num_nodes=num_nodes)
    vl_sub = coalesce(to_undirected(vl_sub, num_nodes=val_ids.numel()), num_nodes=val_ids.numel())
    xtr, ytr = x_features[train_ids].to(device), labels[train_ids].to(device)
    xvl, yvl = x_features[val_ids].to(device), labels[val_ids].to(device)
    tr_d, vl_d = tr_sub.to(device), vl_sub.to(device)
    model = NodeGCN(x_features.size(1), hidden, num_classes, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best = 0.0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(xtr, tr_d), ytr).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        best = float((model(xvl, vl_d).argmax(1) == yvl).float().mean())
    return model, best


def _train_mm_gcn(assignments, public_dict, val_loader, num_features, num_classes,
                   cfg, device):
    set_seed(cfg['seed'])
    ds = PrototypeAssignmentDataset(public_dict, assignments['proto_indices'], assignments['labels'])
    dl = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True)
    model = StandardGCN(32, num_features, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for _ in range(cfg['epochs']):
        model.train()
        for batch in dl:
            batch = batch.to(device); opt.zero_grad(set_to_none=True)
            xa = batch.x + torch.randn_like(batch.x) * cfg['feature_jitter_std']
            ea, _ = dropout_edge(batch.edge_index, p=cfg['edge_dropout_p'])
            if ea.numel() == 0: ea = batch.edge_index
            crit(model(xa, ea, batch.batch), batch.y).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            correct = sum(
                int((model(vb.x.to(device), vb.edge_index.to(device),
                           vb.batch.to(device)).argmax(1) == vb.y.to(device)).sum())
                for vb in val_loader
            )
        best = max(best, correct / len(val_loader.dataset))
    return model, best


# ---------------------------------------------------------------------------
# Per-method run functions (called from run_all_methods)
# ---------------------------------------------------------------------------

def _run_sgc(setup, cfg, device):
    sigma = _gauss_sigma(cfg['epsilon'], DELTA)
    x_enc, labels = setup['x_enc'], setup['labels']
    train_nodes, val_nodes = setup['train_nodes'], setup['val_nodes']
    priv_ei, val_ei = setup['priv_edge_index'], setup['val_edge_index']
    num_nodes = setup['data'].num_nodes

    set_seed(cfg['seed'])
    H = one_hop_mean_aggregate(priv_ei, x_enc, num_nodes=num_nodes)
    no_nbr = H.norm(dim=1) == 0; H[no_nbr] = x_enc[no_nbr]
    x_tr = (H[train_nodes] + torch.randn(train_nodes.numel(), x_enc.size(1)) * sigma).detach()

    H_v = one_hop_mean_aggregate(val_ei, x_enc, num_nodes=num_nodes)
    no_nbr_v = H_v.norm(dim=1) == 0; H_v[no_nbr_v] = x_enc[no_nbr_v]
    x_vl = (H_v[val_nodes] + torch.randn(val_nodes.numel(), x_enc.size(1)) * sigma).detach()

    model, val_acc = _train_node_mlp(
        x_tr, labels[train_nodes], x_vl, labels[val_nodes],
        in_dim=x_enc.size(1), hidden=64, num_classes=setup['num_classes'],
        epochs=300, lr=0.01, wd=5e-4, dropout=0.5, seed=cfg['seed'], device=device,
    )
    return {'type': 'sgc', 'model': model, 'x': x_enc}, val_acc


def _run_rr(setup, cfg, device):
    x_enc, labels = setup['x_enc'], setup['labels']
    train_nodes, val_nodes = setup['train_nodes'], setup['val_nodes']
    priv_ei, val_ei = setup['priv_edge_index'], setup['val_edge_index']
    num_nodes = setup['data'].num_nodes

    set_seed(cfg['seed'])
    perturbed_ei = _apply_edge_rr(priv_ei, train_nodes, cfg['epsilon'], num_nodes, cfg['seed'])
    model, val_acc = _train_node_gcn(
        x_enc, perturbed_ei, val_ei, labels, train_nodes, val_nodes,
        num_classes=setup['num_classes'], num_nodes=num_nodes,
        hidden=64, epochs=300, lr=0.01, dropout=0.5, seed=cfg['seed'], device=device,
    )
    return {'type': 'rr', 'model': model, 'x': x_enc}, val_acc


def _run_gap(setup, cfg, device):
    data, labels = setup['data'], setup['labels']
    train_nodes, val_nodes = setup['train_nodes'], setup['val_nodes']
    priv_ei, val_ei = setup['priv_edge_index'], setup['val_edge_index']
    num_nodes = data.num_nodes
    num_classes = setup['num_classes']
    GAP_K, GAP_DIM = 1, 32

    sigma = _gap_sigma(cfg['epsilon'], DELTA, K=GAP_K)

    # Encoder (MLP, no edges)
    set_seed(cfg['seed'])
    gap_enc = GAPEncoder(data.x.size(1), 64, GAP_DIM, num_classes, dropout=0.5).to(device)
    opt = torch.optim.Adam(gap_enc.parameters(), lr=0.01, weight_decay=5e-4)
    xr_tr = data.x[train_nodes].to(device); yr_tr = labels[train_nodes].to(device)
    for _ in range(200):
        gap_enc.train(); opt.zero_grad()
        F.cross_entropy(gap_enc(xr_tr), yr_tr).backward(); opt.step()
    gap_enc.eval()
    with torch.no_grad():
        x_gap_all = F.normalize(gap_enc.encode(data.x.to(device)).cpu(), p=2, dim=1)

    # Private multi-hop aggregation
    set_seed(cfg['seed'])
    tr_hops = _gap_private_agg(priv_ei, x_gap_all, train_nodes, sigma, num_nodes, GAP_K, cfg['seed'])
    x_gap_tr = torch.cat([x_gap_all[train_nodes]] + tr_hops, dim=1)
    vl_hops = _gap_private_agg(val_ei, x_gap_all, val_nodes, sigma, num_nodes, GAP_K, cfg['seed'] + 1)
    x_gap_vl = torch.cat([x_gap_all[val_nodes]] + vl_hops, dim=1)

    cls_model, val_acc = _train_node_mlp(
        x_gap_tr, labels[train_nodes], x_gap_vl, labels[val_nodes],
        in_dim=x_gap_tr.size(1), hidden=64, num_classes=num_classes,
        epochs=300, lr=0.01, wd=5e-4, dropout=0.5, seed=cfg['seed'], device=device,
    )
    return {'type': 'gap', 'enc': gap_enc, 'cls': cls_model, 'x_raw': data.x, 'x_gap_dim': GAP_DIM}, val_acc


def _run_public_gcn(setup, cfg, device):
    """GCN trained only on public data — no private edges, no DP mechanism."""
    x_enc, labels = setup['x_enc'], setup['labels']
    public_nodes, val_nodes = setup['public_nodes'], setup['val_nodes']
    pub_ei, val_ei = setup['pub_edge_index'], setup['val_edge_index']
    num_nodes = setup['data'].num_nodes

    model, val_acc = _train_node_gcn(
        x_enc, pub_ei, val_ei, labels, public_nodes, val_nodes,
        num_classes=setup['num_classes'], num_nodes=num_nodes,
        hidden=64, epochs=300, lr=0.01, dropout=0.5, seed=cfg['seed'], device=device,
    )
    return {'type': 'pub_gcn', 'model': model, 'x': x_enc}, val_acc


def _run_mm(setup, cfg, device):
    """Full MM-EdgeDP pipeline. Uses label_conditioning=False for Actor (heterophily)."""
    x_enc, labels = setup['x_enc'], setup['labels']
    num_classes = setup['num_classes']
    data = setup['data']

    # Disable label conditioning for heterophilic datasets
    lc = False if cfg['dataset'].lower() == 'actor' else cfg['label_conditioning']

    set_seed(cfg['seed'])
    candidate_pool = build_candidate_pool(
        data, x_enc, labels, setup['public_pool_nodes'], setup['pool_edge_index'],
        num_classes=num_classes,
        target_per_class=cfg['pool_mult'] * cfg['dict_per_class'],
        num_hops=cfg['walk_hops'], query_mode=cfg['query_mode'],
        max_proto_nodes=cfg['max_proto_nodes'],
        min_class_fraction=cfg['min_class_fraction'],
        rng_seed=cfg['seed'],
    )
    kept_pool, _ = hard_filter_pool(
        candidate_pool, setup['pub_edge_index'], data.num_nodes,
        min_nodes=cfg['min_proto_nodes'], min_edges=cfg['min_proto_edges'],
        purity_floor=cfg['purity_floor'], require_connected=cfg['require_connected'],
        max_overlap_frac=cfg['max_overlap_frac'],
    )
    query_features = compute_public_query_features(
        setup['public_query_nodes'], setup['pool_edge_index'], x_enc, labels,
        num_classes, cfg['query_mode'], cfg['query_hops'], data.num_nodes,
    )
    public_dict, class_to_proto = diversify_and_cover(
        kept_pool, candidate_pool, query_features,
        num_classes=num_classes, dict_per_class=cfg['dict_per_class'],
        kcenter_lambda=cfg['kcenter_lambda'], rng_seed=cfg['seed'],
    )
    proto_feats = torch.stack([p['proto_feat'] for p in public_dict])

    set_seed(cfg['seed'])
    assignments, _ = synthesize_edge_dp_assignments(
        setup['priv_edge_index'], x_enc, labels, setup['train_nodes'],
        proto_feats, class_to_proto,
        epsilon_total=cfg['epsilon'], utility_sensitivity=cfg['utility_sensitivity'],
        query_mode=cfg['query_mode'], label_conditioning=lc, tau=cfg['tau_soft_norm'],
    )

    model, val_acc = _train_mm_gcn(
        assignments, public_dict, setup['val_loader'],
        num_features=x_enc.size(1), num_classes=num_classes,
        cfg=cfg, device=device,
    )
    return {'type': 'mm', 'model': model, 'x': x_enc}, val_acc


# ---------------------------------------------------------------------------
# Privacy attack infrastructure
# ---------------------------------------------------------------------------

def _build_attack_pairs(edge_index, node_ids, n_max, seed):
    rng = np.random.default_rng(seed)
    node_list = node_ids.tolist()
    local = {v: i for i, v in enumerate(node_list)}
    n = len(node_list)
    pos_set = set()
    for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if u in local and v in local and u != v:
            a, b = local[u], local[v]
            pos_set.add((min(a, b), max(a, b)))
    pos_list = list(pos_set)
    rng.shuffle(pos_list)
    pos_list = pos_list[:n_max]
    neg_list, all_edges = [], set(pos_set)
    while len(neg_list) < len(pos_list):
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i != j:
            key = (min(i, j), max(i, j))
            if key not in all_edges:
                neg_list.append(key); all_edges.add(key)
    pos = torch.tensor(pos_list, dtype=torch.long)
    neg = torch.tensor(neg_list, dtype=torch.long)
    return pos, neg


def _run_lr_attack(feats_pos, feats_neg, seed):
    X = np.vstack([feats_pos, feats_neg])
    y = np.array([1] * len(feats_pos) + [0] * len(feats_neg), dtype=int)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y)); X, y = X[idx], y[idx]
    n_te = max(2, int(len(y) * 0.30))
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    clf.fit(X[n_te:], y[n_te:])
    return roc_auc_score(y[:n_te], clf.predict_proba(X[:n_te])[:, 1])


_EMPTY_EI = None  # initialized lazily on first use


def _empty_ei(device):
    return torch.zeros(2, 0, dtype=torch.long, device=device)


@torch.no_grad()
def _extract_wb(model_info, train_nodes, device):
    mt = model_info['type']
    ei = _empty_ei(device)
    if mt in ('sgc',):
        x = model_info['x'][train_nodes].to(device)
        h = model_info['model'].net[1](model_info['model'].net[0](x))
        return h.cpu()
    if mt in ('rr', 'pub_gcn'):
        x = model_info['x'][train_nodes].to(device)
        return model_info['model'].conv1(x, ei).relu().cpu()
    if mt == 'gap':
        x = model_info['x_raw'][train_nodes].to(device)
        return F.normalize(model_info['enc'].encode(x), p=2, dim=1).cpu()
    if mt == 'mm':
        x = model_info['x'][train_nodes].to(device)
        h = model_info['model'].conv1(x, ei).relu()
        return model_info['model'].conv2(h, ei).relu().cpu()
    raise ValueError(f'Unknown model type: {mt}')


@torch.no_grad()
def _extract_bb(model_info, train_nodes, device):
    mt = model_info['type']
    ei = _empty_ei(device)
    if mt == 'sgc':
        x = model_info['x'][train_nodes].to(device)
        return torch.softmax(model_info['model'](x), dim=1).cpu()
    if mt in ('rr', 'pub_gcn'):
        x = model_info['x'][train_nodes].to(device)
        return torch.softmax(model_info['model'](x, ei), dim=1).cpu()
    if mt == 'gap':
        x = model_info['x_raw'][train_nodes].to(device)
        h_enc = F.normalize(model_info['enc'].encode(x), p=2, dim=1)
        h_cat = torch.cat([h_enc, torch.zeros_like(h_enc)], dim=1)
        return torch.softmax(model_info['cls'](h_cat), dim=1).cpu()
    if mt == 'mm':
        x = model_info['x'][train_nodes].to(device)
        h = model_info['model'].conv1(x, ei).relu()
        h = model_info['model'].conv2(h, ei).relu()
        batch = torch.arange(train_nodes.numel(), device=device)
        h_pool = global_mean_pool(h, batch)
        return torch.softmax(model_info['model'].lin(h_pool), dim=1).cpu()
    raise ValueError(f'Unknown model type: {mt}')


def _run_attacks(model_info, setup, seed, device):
    pos_pairs = setup['pos_pairs']
    neg_pairs = setup['neg_pairs']
    train_nodes = setup['train_nodes']

    wb = _extract_wb(model_info, train_nodes, device)
    wb_pos = (wb[pos_pairs[:, 0]] * wb[pos_pairs[:, 1]]).numpy()
    wb_neg = (wb[neg_pairs[:, 0]] * wb[neg_pairs[:, 1]]).numpy()
    wb_auc = _run_lr_attack(wb_pos, wb_neg, seed)

    bb = _extract_bb(model_info, train_nodes, device)
    def bb_feats(pairs):
        pu, pv = bb[pairs[:, 0]], bb[pairs[:, 1]]
        return torch.cat([(pu - pv).abs(), pu * pv], dim=1).numpy()
    bb_auc = _run_lr_attack(bb_feats(pos_pairs), bb_feats(neg_pairs), seed)

    return wb_auc, bb_auc


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

METHODS = [
    ('Gaussian SGC', '(ε,δ)-DP', _run_sgc),
    ('Edge RR',      'ε-DP',      _run_rr),
    ('GAP-EDP',      '(ε,δ)-DP', _run_gap),
    ('Public GCN',   'none',      _run_public_gcn),
    ('MM-EdgeDP',    'ε-DP',      _run_mm),
]


def run_all_methods(config: dict, verbose: bool = True) -> list[dict]:
    """
    Run all 5 methods + attacks on one (dataset, epsilon, public_frac, seed) config.
    Returns a list of result dicts, one per method.
    """
    cfg = {**DEFAULT_PHASE3_CONFIG, **config}
    if 'utility_sensitivity' not in cfg:
        cfg['utility_sensitivity'] = 1.0 / math.sqrt(cfg['tau_soft_norm'])
    device = _get_device()
    set_seed(cfg['seed'])

    if verbose:
        print(f"\n[{cfg['dataset']}]  ε={cfg['epsilon']}  "
              f"pub={cfg['public_frac']}  seed={cfg['seed']}", flush=True)

    # --- Shared data setup ---
    try:
        data, num_classes = load_dataset(cfg['dataset'], cfg.get('data_root', '/tmp'))
        data.x = F.normalize(data.x.float(), p=2, dim=1)
        data.y = data.y.long()
        data.edge_index = coalesce(
            to_undirected(data.edge_index, num_nodes=data.num_nodes), num_nodes=data.num_nodes)
        labels = data.y

        public_nodes, train_nodes, val_nodes = stratified_split_indices(
            labels, cfg['public_frac'], cfg['val_frac'], cfg['seed'])
        public_pool_nodes, public_query_nodes = split_public_pool_query(
            public_nodes, labels, cfg['public_query_frac'], cfg['seed'])

        pub_ei,  _ = subgraph(public_nodes,      data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)
        priv_ei, _ = subgraph(train_nodes,        data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)
        val_ei,  _ = subgraph(val_nodes,          data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)
        pool_ei, _ = subgraph(public_pool_nodes,  data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)

        if verbose: print('  training shared encoder...', flush=True)
        x_enc = train_public_encoder(
            data.x, pub_ei, labels, public_nodes, num_classes,
            hidden=cfg['encoder_hidden'], out=cfg['encoder_out'],
            epochs=cfg['encoder_epochs'], lr=cfg['encoder_lr'],
            wd=cfg['encoder_wd'], dropout=cfg['encoder_dropout'],
            seed=cfg['seed'], device=device,
        )

        val_loader = build_validation_loader(
            x_enc, labels, val_nodes, val_ei, cfg['query_hops'], data.num_nodes)

        pos_pairs, neg_pairs = _build_attack_pairs(priv_ei, train_nodes, N_ATTACK_PAIRS, cfg['seed'])
        if verbose:
            print(f'  attack pairs: {len(pos_pairs)} pos + {len(neg_pairs)} neg', flush=True)

    except Exception:
        tb = traceback.format_exc().strip().splitlines()[-1]
        print(f'  SKIPPING {cfg["dataset"]} — dataset setup failed: {tb}', flush=True)
        base_row = dict(
            dataset=cfg['dataset'], epsilon=cfg['epsilon'],
            public_frac=cfg['public_frac'], dict_per_class=cfg['dict_per_class'],
            seed=cfg['seed'],
        )
        return [{**base_row, 'method': m, 'dp_type': dp, 'val_acc': float('nan'),
                 'wb_auc': float('nan'), 'bb_auc': float('nan'),
                 'status': 'error', 'error': tb}
                for m, dp, _ in METHODS]

    setup = dict(
        data=data, x_enc=x_enc, labels=labels, num_classes=num_classes,
        public_nodes=public_nodes, train_nodes=train_nodes, val_nodes=val_nodes,
        pub_edge_index=pub_ei, priv_edge_index=priv_ei,
        val_edge_index=val_ei, pool_edge_index=pool_ei,
        public_pool_nodes=public_pool_nodes, public_query_nodes=public_query_nodes,
        val_loader=val_loader, pos_pairs=pos_pairs, neg_pairs=neg_pairs,
    )

    base_row = dict(
        dataset=cfg['dataset'], epsilon=cfg['epsilon'],
        public_frac=cfg['public_frac'], dict_per_class=cfg['dict_per_class'],
        seed=cfg['seed'],
    )

    results = []
    for method_name, dp_type, train_fn in METHODS:
        if verbose: print(f'  [{method_name}]', flush=True)
        try:
            model_info, val_acc = train_fn(setup, cfg, device)
            wb_auc, bb_auc = _run_attacks(model_info, setup, cfg['seed'], device)
            results.append({**base_row,
                'method': method_name, 'dp_type': dp_type,
                'val_acc': round(val_acc, 4),
                'wb_auc': round(wb_auc, 4), 'bb_auc': round(bb_auc, 4),
                'status': 'ok', 'error': '',
            })
            if verbose:
                print(f'    val_acc={val_acc:.4f}  wb_auc={wb_auc:.4f}  bb_auc={bb_auc:.4f}', flush=True)
        except Exception:
            tb = traceback.format_exc().strip().splitlines()[-1]
            print(f'    FAILED: {tb}', flush=True)
            results.append({**base_row,
                'method': method_name, 'dp_type': dp_type,
                'val_acc': float('nan'), 'wb_auc': float('nan'), 'bb_auc': float('nan'),
                'status': 'error', 'error': tb,
            })
    return results


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _append_rows(csv_path: Path, rows: list[dict]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_KEYS, extrasaction='ignore')
        if write_header: writer.writeheader()
        writer.writerows(rows)


def _done_keys(csv_path: Path) -> set[tuple]:
    done = set()
    if not csv_path.exists(): return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                done.add((row['dataset'], float(row['epsilon']),
                          float(row['public_frac']), int(row['seed'])))
            except (KeyError, ValueError): pass
    return done


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='MM-EdgeDP baseline comparison + attacks')
    p.add_argument('--datasets', nargs='+', default=['Cora', 'AmazonPhoto', 'Actor'])
    p.add_argument('--epsilon', type=float, default=1.0)
    p.add_argument('--public_frac', type=float, default=0.20)
    p.add_argument('--dict_per_class', type=int, default=16)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--output', type=str, default=str(OUTPUT_CSV))
    p.add_argument('--resume', action='store_true', help='Skip already-completed rows')
    p.add_argument('--dry-run', action='store_true', help='Print configs without running')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def main():
    args = _parse_args()
    output_path = Path(args.output)
    done = _done_keys(output_path) if args.resume else set()

    configs = [
        {'dataset': ds, 'epsilon': args.epsilon,
         'public_frac': args.public_frac, 'dict_per_class': args.dict_per_class, 'seed': s}
        for ds in args.datasets for s in args.seeds
    ]
    total = len(configs)
    print(f'{total} configs  ({len(args.datasets)} datasets × {len(args.seeds)} seeds)', flush=True)

    if args.dry_run:
        for i, cfg in enumerate(configs, 1):
            print(f'  [{i:2d}/{total}] {cfg}')
        return

    skipped = 0
    for cfg in configs:
        key = (cfg['dataset'], float(cfg['epsilon']), float(cfg['public_frac']), int(cfg['seed']))
        if key in done:
            skipped += 1; continue
        rows = run_all_methods(cfg, verbose=args.verbose)
        _append_rows(output_path, rows)

    if skipped:
        print(f'Skipped {skipped} already-completed configs (--resume).', flush=True)
    print(f'Done. Results → {output_path}', flush=True)


if __name__ == '__main__':
    main()
