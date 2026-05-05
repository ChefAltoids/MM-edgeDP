"""
experiments.py — MM-EdgeDP pipeline as a single callable function.

Usage:
    from experiments import run_experiment, DEFAULT_CONFIG

    results = run_experiment({'dataset': 'Cora', 'epsilon': 1.0, 'dict_per_class': 8})
    print(results['val_acc'])
"""

import math
import random
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).parent

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import (
    coalesce, dropout_edge, k_hop_subgraph, subgraph, to_undirected,
)

# ---------------------------------------------------------------------------
# Default config — override any key in the dict you pass to run_experiment
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    'seed': 42,
    'dataset': 'Cora',        # 'Cora' | 'FacebookPagePage' | 'Actor'
    'data_root': '/tmp',
    # splits
    'public_frac': 0.20,
    'val_frac': 0.20,
    'public_query_frac': 0.25,
    # mechanism
    'epsilon': 1.0,
    'query_mode': 'one_hop',
    'walk_hops': 1,
    'query_hops': 1,
    'label_conditioning': True,
    'tau_soft_norm': 1e-3,
    # dictionary
    'dict_per_class': 8,
    'pool_mult': 6,
    'min_class_fraction': 1.0,
    'max_proto_nodes': 32,
    # Stage B filters
    'min_proto_nodes': 2,
    'min_proto_edges': 1,
    'purity_floor': 0.5,
    'require_connected': True,
    'max_overlap_frac': 0.85,
    # Stage C
    'kcenter_lambda': 0.5,
    'use_kcenter': True,
    # public encoder
    'encoder_hidden': 64,
    'encoder_out': 32,
    'encoder_epochs': 200,
    'encoder_lr': 0.01,
    'encoder_wd': 5e-4,
    'encoder_dropout': 0.5,
    # downstream GCN
    'epochs': 50,
    'batch_size': 32,
    'lr': 0.01,
    'feature_jitter_std': 0.05,
    'edge_dropout_p': 0.10,
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(name: str, root: str = '/tmp'):
    """
    Returns (data, num_classes).
    Features are NOT yet L2-normalized — run_experiment handles that.
    """
    name_lower = name.lower()
    if name_lower == 'cora':
        ds = Planetoid(root=f'{root}/Cora', name='Cora')
        data = ds[0]
        num_classes = int(ds.num_classes)
    elif name_lower == 'facebookpagepage':
        from torch_geometric.datasets import FacebookPagePage
        ds = FacebookPagePage(root=str(_REPO_ROOT / 'data' / 'facebookpagepage'))
        data = ds[0]
        num_classes = int(data.y.max().item()) + 1
    elif name_lower == 'actor':
        from torch_geometric.datasets import Actor
        ds = Actor(root=f'{root}/Actor')
        data = ds[0]
        if hasattr(data.x, 'to_dense'):
            data.x = data.x.to_dense()
        num_classes = int(data.y.max().item()) + 1
    elif name_lower == 'lastfmasia':
        from torch_geometric.datasets import LastFMAsia
        ds = LastFMAsia(root=f'{root}/LastFMAsia')
        data = ds[0]
        num_classes = int(data.y.max().item()) + 1
    elif name_lower == 'amazonphoto':
        from torch_geometric.datasets import Amazon
        ds = Amazon(root=f'{root}/AmazonPhoto', name='Photo')
        data = ds[0]
        num_classes = int(ds.num_classes)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: Cora, FacebookPagePage, LastFMAsia, Actor, AmazonPhoto"
        )
    return data, num_classes


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def stratified_split_indices(y, public_frac, val_frac, seed):
    g = torch.Generator().manual_seed(seed)
    pub, val, tr = [], [], []
    for c in torch.unique(y).tolist():
        idx = torch.where(y == c)[0]
        if idx.numel() == 0:
            continue
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        n_pub = max(1, int(public_frac * idx.numel()))
        n_val = max(1, int(val_frac * idx.numel()))
        pub.append(perm[:n_pub])
        val.append(perm[n_pub:n_pub + n_val])
        tr.append(perm[n_pub + n_val:])
    return torch.cat(pub), torch.cat(tr), torch.cat(val)


def split_public_pool_query(public_nodes, labels, query_frac, seed):
    g = torch.Generator().manual_seed(seed)
    pool, query = [], []
    for c in torch.unique(labels[public_nodes]).tolist():
        idx = public_nodes[labels[public_nodes] == c]
        if idx.numel() == 0:
            continue
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        n_q = max(1, min(int(query_frac * idx.numel()), idx.numel() - 1))
        query.append(perm[:n_q])
        pool.append(perm[n_q:])
    return torch.cat(pool), torch.cat(query)


# ---------------------------------------------------------------------------
# Public GCN encoder (Cell 3)
# ---------------------------------------------------------------------------

class PublicNodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden, out, num_classes, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out)
        self.head = nn.Linear(out, num_classes)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index).relu()

    def forward(self, x, edge_index):
        return self.head(self.encode(x, edge_index))


def train_public_encoder(x_features, pub_edges, labels, public_nodes, num_classes,
                          hidden, out, epochs, lr, wd, dropout, seed, device):
    set_seed(seed)
    x_d = x_features.to(device)
    e_d = pub_edges.to(device)
    y_d = labels.to(device)
    pub_mask = torch.zeros(x_d.size(0), dtype=torch.bool, device=device)
    pub_mask[public_nodes] = True

    model = PublicNodeEncoder(x_d.size(1), hidden, out, num_classes, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(x_d, e_d)[pub_mask], y_d[pub_mask])
        loss.backward(); opt.step()
        if epoch % 50 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                acc = float((model(x_d, e_d)[pub_mask].argmax(1) == y_d[pub_mask]).float().mean())
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        x_enc = F.normalize(model.encode(x_d, e_d), p=2, dim=1).cpu()
    return x_enc


# ---------------------------------------------------------------------------
# Mechanism helpers (Cell 4)
# ---------------------------------------------------------------------------

def one_hop_mean_aggregate(edge_index, x_features, num_nodes):
    eu = coalesce(to_undirected(edge_index, num_nodes=num_nodes), num_nodes=num_nodes)
    H = torch.zeros_like(x_features)
    if eu.numel() == 0:
        return H
    row, col = eu
    H.scatter_add_(0, row.unsqueeze(1).expand(-1, x_features.size(1)), x_features[col])
    counts = torch.zeros(num_nodes, dtype=x_features.dtype, device=x_features.device)
    counts.scatter_add_(0, row, torch.ones_like(row, dtype=x_features.dtype))
    out = torch.zeros_like(H)
    nz = counts > 0
    out[nz] = H[nz] / counts[nz].unsqueeze(1)
    return out


def soft_normalize(x, tau, dim=-1):
    sq = (x * x).sum(dim=dim, keepdim=True)
    return x / torch.sqrt(sq + tau)


def build_prototype_feature(edge_index_sub, x_sub, anchor_local_idx, query_mode='one_hop'):
    n = x_sub.size(0)
    H1 = one_hop_mean_aggregate(edge_index_sub, x_sub, num_nodes=n)
    z1 = H1[anchor_local_idx]
    if query_mode == 'one_hop':
        return F.normalize(z1, p=2, dim=0)
    if query_mode == 'two_hop_concat':
        H2 = one_hop_mean_aggregate(edge_index_sub, H1, num_nodes=n)
        z2 = H2[anchor_local_idx]
        return F.normalize(torch.cat([z1, z2], dim=0), p=2, dim=0)
    raise ValueError(f'Unsupported query_mode: {query_mode}')


def build_private_query_features(private_edges, x_features, query_mode='one_hop', tau=1e-3):
    n = x_features.size(0)
    H1 = one_hop_mean_aggregate(private_edges, x_features, num_nodes=n)
    if query_mode == 'one_hop':
        return soft_normalize(H1, tau=tau, dim=1)
    if query_mode == 'two_hop_concat':
        H2 = one_hop_mean_aggregate(private_edges, H1, num_nodes=n)
        return soft_normalize(torch.cat([H1, H2], dim=1), tau=tau, dim=1)
    raise ValueError(f'Unsupported query_mode: {query_mode}')


def _is_connected_subgraph(node_ids, edge_index, num_nodes):
    if node_ids.numel() <= 1:
        return True
    sub_edges, _ = subgraph(node_ids, edge_index, relabel_nodes=True, num_nodes=num_nodes)
    if sub_edges.numel() == 0:
        return False
    adj = defaultdict(list)
    for a, b in zip(sub_edges[0].tolist(), sub_edges[1].tolist()):
        adj[a].append(b)
    seen = {0}; stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); stack.append(v)
    return len(seen) == node_ids.numel()


# ---------------------------------------------------------------------------
# Dictionary construction — Stages A, B, C (Cell 5)
# ---------------------------------------------------------------------------

def build_candidate_pool(data_obj, x_features, labels, pool_nodes, pool_edges,
                          num_classes, target_per_class, num_hops, query_mode,
                          max_proto_nodes, min_class_fraction, rng_seed):
    pool_und = coalesce(
        to_undirected(pool_edges, num_nodes=data_obj.num_nodes),
        num_nodes=data_obj.num_nodes,
    )
    g = torch.Generator().manual_seed(int(rng_seed))
    pool_labels = labels[pool_nodes]
    candidates = []
    for c in range(num_classes):
        class_pool = pool_nodes[pool_labels == c]
        if class_pool.numel() == 0:
            continue
        for _ in range(target_per_class):
            pick = torch.randint(0, class_pool.numel(), (1,), generator=g).item()
            anchor = class_pool[pick]
            subset, _, _, _ = k_hop_subgraph(
                int(anchor.item()), num_hops=num_hops,
                edge_index=pool_und, relabel_nodes=False,
                num_nodes=data_obj.num_nodes,
            )
            proto_labels = labels[subset]
            class_mask = proto_labels == c
            class_count = int(class_mask.sum().item())
            frac = class_count / max(subset.numel(), 1)

            if frac >= min_class_fraction and class_count > 0:
                class_subset = subset
            else:
                class_subset = subset[class_mask]
                if class_subset.numel() == 0:
                    class_subset = anchor.view(1)

            if not torch.any(class_subset == anchor):
                class_subset = torch.unique(torch.cat([class_subset, anchor.view(1)]))

            if class_subset.numel() > max_proto_nodes:
                non_anchor = class_subset[class_subset != anchor]
                budget = max_proto_nodes - 1
                if budget <= 0:
                    class_subset = anchor.view(1)
                else:
                    if non_anchor.numel() > budget:
                        perm = torch.randperm(non_anchor.numel(), generator=g)[:budget]
                        non_anchor = non_anchor[perm]
                    class_subset = torch.cat([anchor.view(1), non_anchor])

            sub_edges, _ = subgraph(class_subset, pool_und, relabel_nodes=True,
                                     num_nodes=data_obj.num_nodes)
            x_sub = x_features[class_subset]
            anchor_local = int((class_subset == anchor).nonzero(as_tuple=True)[0].item())
            proto_feat = build_prototype_feature(sub_edges, x_sub, anchor_local, query_mode)
            candidates.append({
                'class': c,
                'anchor': int(anchor.item()),
                'node_ids': class_subset.clone(),
                'x': x_sub,
                'edge_index': sub_edges,
                'proto_feat': proto_feat,
                'n_nodes': int(class_subset.numel()),
                'n_edges': int(sub_edges.size(1)),
                'purity': frac,
            })
    return candidates


def hard_filter_pool(pool, pub_edges, num_nodes, min_nodes, min_edges,
                      purity_floor, require_connected, max_overlap_frac):
    pub_und = coalesce(to_undirected(pub_edges, num_nodes=num_nodes), num_nodes=num_nodes)
    kept_per_class = defaultdict(list); kept = []; reasons = Counter()
    for p in pool:
        if p['n_nodes'] < min_nodes:    reasons['min_nodes'] += 1;    continue
        if p['n_edges'] < min_edges:    reasons['min_edges'] += 1;    continue
        if p['purity']  < purity_floor: reasons['purity'] += 1;       continue
        if require_connected and not _is_connected_subgraph(p['node_ids'], pub_und, num_nodes):
            reasons['disconnected'] += 1; continue
        s = set(p['node_ids'].tolist()); overlap = False
        for prev_set, _ in kept_per_class[p['class']]:
            if len(s & prev_set) / max(len(s | prev_set), 1) >= max_overlap_frac:
                overlap = True; break
        if overlap:
            reasons['overlap'] += 1; continue
        kept_per_class[p['class']].append((s, p))
        kept.append(p)
    return kept, reasons


@torch.no_grad()
def compute_public_query_features(query_nodes, pool_edges, x_features, labels,
                                    num_classes, query_mode, query_hops, num_total_nodes):
    pool_und = coalesce(
        to_undirected(pool_edges, num_nodes=num_total_nodes),
        num_nodes=num_total_nodes,
    )
    per_class = {c: [] for c in range(num_classes)}
    for node in query_nodes.tolist():
        subset, _, _, _ = k_hop_subgraph(
            node, num_hops=query_hops, edge_index=pool_und,
            relabel_nodes=False, num_nodes=num_total_nodes,
        )
        if subset.numel() == 0:
            subset = torch.tensor([node], dtype=torch.long)
        if not torch.any(subset == node):
            subset = torch.unique(torch.cat([subset, torch.tensor([node])]))
        sub_edges, _ = subgraph(subset, pool_und, relabel_nodes=True, num_nodes=num_total_nodes)
        anchor_local = int((subset == node).nonzero(as_tuple=True)[0].item())
        q_feat = build_prototype_feature(sub_edges, x_features[subset], anchor_local, query_mode)
        per_class[int(labels[node].item())].append(q_feat)
    return {c: torch.stack(L, dim=0) if L else torch.empty(0) for c, L in per_class.items()}


@torch.no_grad()
def diversify_and_cover(kept_pool, candidate_pool, query_features_per_class,
                         num_classes, dict_per_class, kcenter_lambda, rng_seed,
                         use_kcenter=True):
    g = torch.Generator().manual_seed(int(rng_seed))
    by_kept = defaultdict(list)
    for p in kept_pool: by_kept[p['class']].append(p)
    by_all = defaultdict(list)
    for p in candidate_pool: by_all[p['class']].append(p)

    selected = []
    class_to_proto = {c: [] for c in range(num_classes)}
    for c in range(num_classes):
        cands = list(by_kept.get(c, []))
        if len(cands) < dict_per_class:
            cand_ids = {id(p) for p in cands}
            spares = [p for p in by_all.get(c, []) if id(p) not in cand_ids]
            random.Random(int(rng_seed) + c).shuffle(spares)
            cands += spares[:max(dict_per_class - len(cands), 0)]
        if not cands:
            continue
        budget = min(dict_per_class, len(cands))

        if not use_kcenter:
            # Random selection — no optimization
            random.Random(int(rng_seed) + c).shuffle(cands)
            chosen_items = cands[:budget]
            for p in chosen_items:
                class_to_proto[c].append(len(selected))
                selected.append(p)
            continue

        feats = torch.stack([p['proto_feat'] for p in cands], dim=0)
        Q = query_features_per_class.get(c, None)
        if Q is None or Q.numel() == 0:
            start_idx = int(torch.randint(0, len(cands), (1,), generator=g).item())
            running_cov = None
        else:
            start_idx = int(torch.cdist(Q, feats, p=2).mean(dim=0).argmin().item())
            running_cov = torch.linalg.norm(Q - feats[start_idx].unsqueeze(0), dim=1)
        chosen = [start_idx]
        dist_to_sel = torch.linalg.norm(feats - feats[start_idx].unsqueeze(0), dim=1)
        while len(chosen) < budget:
            if running_cov is not None:
                d_to_cands = torch.cdist(Q, feats, p=2)
                cov_score = -torch.minimum(running_cov.unsqueeze(1), d_to_cands).mean(dim=0)
            else:
                cov_score = torch.zeros(len(cands))
            score = kcenter_lambda * dist_to_sel + (1.0 - kcenter_lambda) * cov_score
            for ci in chosen: score[ci] = float('-inf')
            nxt = int(score.argmax().item())
            chosen.append(nxt)
            dist_to_sel = torch.minimum(
                dist_to_sel,
                torch.linalg.norm(feats - feats[nxt].unsqueeze(0), dim=1),
            )
            if running_cov is not None:
                running_cov = torch.minimum(
                    running_cov,
                    torch.linalg.norm(Q - feats[nxt].unsqueeze(0), dim=1),
                )
        for li in chosen:
            class_to_proto[c].append(len(selected))
            selected.append(cands[li])
    return selected, class_to_proto


# ---------------------------------------------------------------------------
# Exponential mechanism (Cell 6)
# ---------------------------------------------------------------------------

def gumbel_max_sample(logits):
    u = torch.rand_like(logits).clamp_(1e-8, 1 - 1e-8)
    return torch.argmax(logits + (-torch.log(-torch.log(u)))).item()


@torch.no_grad()
def synthesize_edge_dp_assignments(private_edges, x_features, labels, private_nodes,
                                     dict_features, class_to_proto_indices,
                                     epsilon_total, utility_sensitivity,
                                     query_mode, label_conditioning, tau):
    eps_per = epsilon_total / 2.0
    Q = build_private_query_features(private_edges, x_features, query_mode=query_mode, tau=tau)
    K = dict_features.size(0)
    nc = int(labels.max().item()) + 1
    all_idx = torch.arange(K, dtype=torch.long)
    class_idx = {
        c: (torch.tensor(class_to_proto_indices.get(c, []), dtype=torch.long)
            if class_to_proto_indices.get(c) else all_idx)
        for c in range(nc)
    }
    N = int(private_nodes.numel())
    sel = torch.empty(N, dtype=torch.long)
    priv_labels = labels[private_nodes].long().clone()
    counts = torch.zeros(K, dtype=torch.long)
    ent_sum = top_sum = 0.0
    for j, u in enumerate(private_nodes.tolist()):
        y_u = int(labels[u].item())
        cand_idx = class_idx[y_u] if label_conditioning else all_idx
        cand_feats = dict_features.index_select(0, cand_idx)
        d = torch.linalg.norm(cand_feats - Q[u].unsqueeze(0), dim=1)
        logits = (eps_per / (2.0 * utility_sensitivity)) * (-d)
        probs = torch.softmax(logits, dim=0)
        ent_sum += float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
        top_sum += float(probs.max().item())
        li = gumbel_max_sample(logits)
        sel[j] = int(cand_idx[li].item())
        counts[sel[j]] += 1
    return (
        {'proto_indices': sel, 'labels': priv_labels},
        {
            'epsilon_total': float(epsilon_total),
            'epsilon_per_node': float(eps_per),
            'mean_entropy': ent_sum / max(N, 1),
            'mean_top_probability': top_sum / max(N, 1),
            'unique_selected_ratio': float((counts > 0).sum().item()) / float(max(K, 1)),
        },
    )


# ---------------------------------------------------------------------------
# Downstream GCN training (Cell 7)
# ---------------------------------------------------------------------------

class PrototypeAssignmentDataset(torch.utils.data.Dataset):
    def __init__(self, public_dict, proto_indices, labels):
        self.public_dict = public_dict
        self.proto_indices = proto_indices.long().cpu()
        self.labels = labels.long().cpu()

    def __len__(self):
        return self.proto_indices.numel()

    def __getitem__(self, idx):
        p = self.public_dict[int(self.proto_indices[idx].item())]
        return Data(x=p['x'], edge_index=p['edge_index'], y=self.labels[idx].view(1))


class StandardGCN(nn.Module):
    def __init__(self, hidden, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(global_mean_pool(x, batch))


def build_validation_loader(x_features, labels, val_nodes, val_edges, query_hops, num_total_nodes):
    vu = coalesce(to_undirected(val_edges, num_nodes=num_total_nodes), num_nodes=num_total_nodes)
    items = []
    for n in val_nodes.tolist():
        subset, sei, _, _ = k_hop_subgraph(
            n, num_hops=query_hops, edge_index=vu,
            relabel_nodes=True, num_nodes=num_total_nodes,
        )
        items.append(Data(x=x_features[subset], edge_index=sei, y=labels[n].unsqueeze(0)))
    return DataLoader(items, batch_size=32, shuffle=False)


def train_gnn(assignments, public_dict, val_loader, num_features, num_classes,
               epochs, batch_size, lr, feature_jitter_std, edge_dropout_p, device, seed):
    set_seed(seed)
    ds = PrototypeAssignmentDataset(public_dict, assignments['proto_indices'], assignments['labels'])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = StandardGCN(32, num_features, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for _ in range(epochs):
        model.train()
        for batch in dl:
            batch = batch.to(device); opt.zero_grad(set_to_none=True)
            xa = batch.x + torch.randn_like(batch.x) * feature_jitter_std
            ea, _ = dropout_edge(batch.edge_index, p=edge_dropout_p)
            if ea.numel() == 0:
                ea = batch.edge_index
            crit(model(xa, ea, batch.batch), batch.y).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            correct = sum(
                int((model(vb.x.to(device), vb.edge_index.to(device), vb.batch.to(device))
                     .argmax(1) == vb.y.to(device)).sum())
                for vb in val_loader
            )
        best = max(best, correct / len(val_loader.dataset))
    return best


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(config: dict, verbose: bool = True) -> dict:
    """
    Run the full MM-EdgeDP pipeline.

    Any key in config overrides DEFAULT_CONFIG.
    Returns a flat dict of results suitable for appending to a DataFrame.
    """
    cfg = {**DEFAULT_CONFIG, **config}
    if 'utility_sensitivity' not in cfg:
        cfg['utility_sensitivity'] = 1.0 / math.sqrt(cfg['tau_soft_norm'])
    device = _get_device()
    set_seed(cfg['seed'])

    if verbose:
        print(f"[{cfg['dataset']}]  ε={cfg['epsilon']}  "
              f"pub={cfg['public_frac']}  dict/cls={cfg['dict_per_class']}")

    # --- Data ---
    data, num_classes = load_dataset(cfg['dataset'], cfg['data_root'])
    data.x = F.normalize(data.x.float(), p=2, dim=1)
    data.y = data.y.long()
    data.edge_index = coalesce(
        to_undirected(data.edge_index, num_nodes=data.num_nodes),
        num_nodes=data.num_nodes,
    )
    labels = data.y

    # --- Splits ---
    public_nodes, train_nodes, val_nodes = stratified_split_indices(
        labels, cfg['public_frac'], cfg['val_frac'], cfg['seed'])
    public_pool_nodes, public_query_nodes = split_public_pool_query(
        public_nodes, labels, cfg['public_query_frac'], cfg['seed'])

    pub_edge_index,  _ = subgraph(public_nodes,      data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)
    priv_edge_index, _ = subgraph(train_nodes,        data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)
    val_edge_index,  _ = subgraph(val_nodes,          data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)
    pool_edge_index, _ = subgraph(public_pool_nodes,  data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)

    # --- Encoder ---
    x_enc = train_public_encoder(
        data.x, pub_edge_index, labels, public_nodes, num_classes,
        hidden=cfg['encoder_hidden'], out=cfg['encoder_out'],
        epochs=cfg['encoder_epochs'], lr=cfg['encoder_lr'],
        wd=cfg['encoder_wd'], dropout=cfg['encoder_dropout'],
        seed=cfg['seed'], device=device,
    )

    # --- Dictionary (Stages A → B → C) ---
    set_seed(cfg['seed'])
    candidate_pool = build_candidate_pool(
        data, x_enc, labels, public_pool_nodes, pool_edge_index,
        num_classes=num_classes,
        target_per_class=cfg['pool_mult'] * cfg['dict_per_class'],
        num_hops=cfg['walk_hops'], query_mode=cfg['query_mode'],
        max_proto_nodes=cfg['max_proto_nodes'],
        min_class_fraction=cfg['min_class_fraction'],
        rng_seed=cfg['seed'],
    )
    kept_pool, _ = hard_filter_pool(
        candidate_pool, pub_edge_index, data.num_nodes,
        min_nodes=cfg['min_proto_nodes'], min_edges=cfg['min_proto_edges'],
        purity_floor=cfg['purity_floor'], require_connected=cfg['require_connected'],
        max_overlap_frac=cfg['max_overlap_frac'],
    )
    query_features_per_class = compute_public_query_features(
        public_query_nodes, pool_edge_index, x_enc, labels, num_classes,
        cfg['query_mode'], cfg['query_hops'], data.num_nodes,
    )
    public_dict, class_to_proto_indices = diversify_and_cover(
        kept_pool, candidate_pool, query_features_per_class,
        num_classes=num_classes,
        dict_per_class=cfg['dict_per_class'],
        kcenter_lambda=cfg['kcenter_lambda'],
        rng_seed=cfg['seed'],
        use_kcenter=cfg['use_kcenter'],
    )
    public_proto_feats = torch.stack([p['proto_feat'] for p in public_dict])

    # --- Exponential mechanism ---
    set_seed(cfg['seed'])
    assignments, syn_diag = synthesize_edge_dp_assignments(
        priv_edge_index, x_enc, labels, train_nodes,
        public_proto_feats, class_to_proto_indices,
        epsilon_total=cfg['epsilon'],
        utility_sensitivity=cfg['utility_sensitivity'],
        query_mode=cfg['query_mode'],
        label_conditioning=cfg['label_conditioning'],
        tau=cfg['tau_soft_norm'],
    )

    # --- Downstream GCN ---
    val_loader = build_validation_loader(
        x_enc, labels, val_nodes, val_edge_index, cfg['query_hops'], data.num_nodes)
    majority_acc = float(
        (labels[val_nodes] == int(torch.bincount(labels[val_nodes]).argmax())).float().mean()
    )
    best_val = train_gnn(
        assignments, public_dict, val_loader,
        num_features=x_enc.size(1), num_classes=num_classes,
        epochs=cfg['epochs'], batch_size=cfg['batch_size'], lr=cfg['lr'],
        feature_jitter_std=cfg['feature_jitter_std'],
        edge_dropout_p=cfg['edge_dropout_p'],
        device=device, seed=cfg['seed'],
    )

    results = {
        'dataset':            cfg['dataset'],
        'epsilon':            cfg['epsilon'],
        'public_frac':        cfg['public_frac'],
        'dict_per_class':     cfg['dict_per_class'],
        'seed':               cfg['seed'],
        'val_acc':            round(best_val, 4),
        'majority_acc':       round(majority_acc, 4),
        'dict_size':          len(public_dict),
        'n_train':            int(train_nodes.numel()),
        'n_val':              int(val_nodes.numel()),
        'n_public':           int(public_nodes.numel()),
        'mean_entropy':       round(syn_diag['mean_entropy'], 4),
        'mean_top_prob':      round(syn_diag['mean_top_probability'], 4),
        'unique_proto_ratio': round(syn_diag['unique_selected_ratio'], 4),
    }
    if verbose:
        print(f"  → val_acc={results['val_acc']:.4f}  majority={results['majority_acc']:.4f}")
    return results
