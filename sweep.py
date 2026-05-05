"""
sweep.py — MM-edgeDP experiment sweep runner

Phase 1 (default settings):
    python3 sweep.py --phase 1

Phase 2 (custom settings with --set overrides):
    python3 sweep.py --phase 2 --set public_frac=0.2 --set dict_per_class=128 \
        --output outputs/sweep_phase2_pf20.csv

Results are written as a CSV to --output (default: outputs/sweep_phase<N>.csv).
Each row = one (dataset, epsilon, seed) combination.
"""

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import (
    coalesce, dropout_edge, k_hop_subgraph, subgraph, to_undirected,
)

# ─────────────────────────────────────────────
# Default sweep configuration
# ─────────────────────────────────────────────
PHASE1_DEFAULTS = dict(
    datasets     = ["Cora", "CiteSeer", "PubMed", "ogbn-arxiv"],
    epsilons     = [0.5, 1.0, 2.0, 4.0],
    seeds        = [42, 43, 44],
    public_frac  = 0.20,
    val_frac     = 0.20,
    dict_per_class = 50,
    query_mode   = "one_hop",
    walk_hops    = 2,
    query_hops   = 1,
    label_conditioning = True,
    min_class_fraction = 1.0,
)

PHASE2_DEFAULTS = dict(
    datasets     = ["Cora", "CiteSeer", "PubMed", "ogbn-arxiv"],
    epsilons     = [0.5, 1.0, 2.0, 4.0],
    seeds        = [42, 43, 44],
    public_frac  = 0.20,
    val_frac     = 0.20,
    dict_per_class = 50,
    query_mode   = "one_hop",
    walk_hops    = 2,
    query_hops   = 1,
    label_conditioning = True,
    min_class_fraction = 1.0,
)

DATASET_PRESETS = {
    "cora":       dict(max_private=None, max_val=None,  epochs=50,  batch=32, hidden=16,  max_proto=None),
    "citeseer":   dict(max_private=None, max_val=None,  epochs=50,  batch=32, hidden=16,  max_proto=None),
    "pubmed":     dict(max_private=10000, max_val=2000, epochs=25,  batch=64, hidden=32,  max_proto=64),
    "ogbn-arxiv": dict(max_private=30000, max_val=3000, epochs=8,   batch=64, hidden=32,  max_proto=64),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cap_indices(indices, max_count=None, seed=0):
    if max_count is None or indices.numel() <= max_count:
        return indices
    g = torch.Generator().manual_seed(seed)
    return indices[torch.randperm(indices.numel(), generator=g)[:max_count]]


def stratified_split(y, public_frac, val_frac, seed=0):
    g = torch.Generator().manual_seed(seed)
    pub, val, train = [], [], []
    for c in torch.unique(y).tolist():
        idx = torch.where(y == c)[0]
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        n_pub = max(1, int(public_frac * idx.numel()))
        n_val = max(1, int(val_frac * idx.numel()))
        n_pub = min(n_pub, idx.numel())
        n_val = min(n_val, max(idx.numel() - n_pub, 0))
        pub.append(perm[:n_pub])
        val.append(perm[n_pub:n_pub + n_val])
        train.append(perm[n_pub + n_val:])
    pub   = torch.cat(pub)   if pub   else torch.tensor([], dtype=torch.long)
    val   = torch.cat(val)   if val   else torch.tensor([], dtype=torch.long)
    train = torch.cat(train) if train else torch.tensor([], dtype=torch.long)
    return pub, train, val


def load_dataset(name):
    import torch_geometric.transforms as T
    from torch_geometric.datasets import Planetoid
    key = name.lower()
    if key == "ogbn-arxiv":
        from ogb.nodeproppred import PygNodePropPredDataset
        ds   = PygNodePropPredDataset(name="ogbn-arxiv", root="data/ogb")
        data = ds[0]
        data.y = data.y.squeeze(-1).long()
    elif key in ("cora", "citeseer", "pubmed"):
        pretty = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[key]
        ds   = Planetoid(root=f"data/Planetoid/{pretty}", name=pretty,
                         transform=T.NormalizeFeatures())
        data = ds[0]
        data.y = data.y.long()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    data.x = F.normalize(data.x.float(), p=2, dim=1)
    data.edge_index = coalesce(to_undirected(data.edge_index, num_nodes=data.num_nodes),
                               num_nodes=data.num_nodes)
    num_classes = int(data.y.max().item()) + 1
    return data, num_classes


# ─────────────────────────────────────────────
# MM-edgeDP core (ported from notebook Cell 3)
# ─────────────────────────────────────────────
def one_hop_mean_agg(edge_index, x, n):
    ei  = coalesce(to_undirected(edge_index, num_nodes=n), num_nodes=n)
    out = torch.zeros_like(x)
    if ei.numel() == 0:
        return out
    row, col = ei
    out.scatter_add_(0, row.unsqueeze(1).expand(-1, x.size(1)), x[col])
    cnt = torch.zeros(n, dtype=x.dtype)
    cnt.scatter_add_(0, row, torch.ones(row.size(0), dtype=x.dtype))
    nz = cnt > 0
    out[nz] = out[nz] / cnt[nz].unsqueeze(1)
    return out


def degree_weighted_embed(x_sub, ei_sub):
    n = x_sub.size(0)
    if n == 0:
        return torch.zeros(x_sub.size(1))
    if ei_sub.numel() == 0:
        w = torch.ones(n, dtype=x_sub.dtype)
    else:
        row, col = ei_sub
        deg = torch.bincount(row, minlength=n).float() + torch.bincount(col, minlength=n).float()
        w   = deg + 1.0
    return (x_sub * w.unsqueeze(1)).sum(0) / w.sum().clamp_min(1e-12)


def build_proto_feat(ei_sub, x_sub, query_mode):
    n  = x_sub.size(0)
    H1 = one_hop_mean_agg(ei_sub, x_sub, n)
    z1 = degree_weighted_embed(H1, ei_sub)
    if query_mode == "one_hop":
        return F.normalize(z1, p=2, dim=0)
    H2 = one_hop_mean_agg(ei_sub, H1, n)
    z2 = degree_weighted_embed(H2, ei_sub)
    return F.normalize(torch.cat([z1, z2]), p=2, dim=0)


def build_query_features(private_ei, x, n, query_mode):
    H1 = one_hop_mean_agg(private_ei, x, n)
    if query_mode == "one_hop":
        return F.normalize(H1, p=2, dim=1)
    H2 = one_hop_mean_agg(private_ei, H1, n)
    return F.normalize(torch.cat([H1, H2], dim=1), p=2, dim=1)


def build_dictionary(data, x, labels, pub_nodes, pub_ei, num_classes,
                     dict_per_class, walk_hops, query_mode,
                     min_class_fraction, max_proto_nodes, verbose=True):
    pub_ei = coalesce(to_undirected(pub_ei, num_nodes=data.num_nodes), num_nodes=data.num_nodes)
    pub_labels = labels[pub_nodes]
    dictionary, class_to_idx = [], {c: [] for c in range(num_classes)}

    for c in range(num_classes):
        pool = pub_nodes[pub_labels == c]
        if pool.numel() == 0:
            pool = pub_nodes
        for _ in range(dict_per_class):
            anchor = pool[torch.randint(0, pool.numel(), (1,)).item()]
            subset, _, _, _ = k_hop_subgraph(int(anchor.item()), walk_hops,
                                             pub_ei, relabel_nodes=False,
                                             num_nodes=data.num_nodes)
            pl  = labels[subset]
            cm  = pl == c
            frac = float(cm.sum()) / max(subset.numel(), 1)
            use = subset[cm] if frac < min_class_fraction or cm.sum() == 0 else subset
            if use.numel() == 0:
                use = anchor.view(1)
            if max_proto_nodes and use.numel() > max_proto_nodes:
                use = use[torch.randperm(use.numel())[:max_proto_nodes]]
            ei_sub, _ = subgraph(use, pub_ei, relabel_nodes=True, num_nodes=data.num_nodes)
            pf = build_proto_feat(ei_sub, x[use], query_mode)
            idx = len(dictionary)
            dictionary.append({"edge_index": ei_sub, "x": x[use], "proto_feat": pf, "class": c})
            class_to_idx[c].append(idx)

    dict_feats = torch.stack([g["proto_feat"] for g in dictionary])
    if verbose:
        print(f"  Dictionary: {len(dictionary)} prototypes, feat_dim={dict_feats.size(1)}")
    return dictionary, dict_feats, class_to_idx


def gumbel_max(logits):
    u = torch.rand_like(logits).clamp_(1e-8, 1 - 1e-8)
    return torch.argmax(logits - torch.log(-torch.log(u))).item()


@torch.no_grad()
def sample_assignments(private_ei, x, labels, priv_nodes,
                        dict_feats, class_to_idx, epsilon,
                        utility_sensitivity, query_mode, label_conditioning):
    eps_node = epsilon / 2.0
    Q = build_query_features(private_ei, x, x.size(0), query_mode)
    K = dict_feats.size(0)
    all_idx = torch.arange(K)
    cls_tensors = {c: torch.tensor(v) if v else all_idx
                   for c, v in class_to_idx.items()}
    N   = priv_nodes.numel()
    sel = torch.empty(N, dtype=torch.long)
    lbl = labels[priv_nodes].long().clone()

    for j, u in enumerate(priv_nodes.tolist()):
        y_u  = int(labels[u].item())
        cand = cls_tensors[y_u] if label_conditioning else all_idx
        cf   = dict_feats.index_select(0, cand)
        dist = torch.linalg.norm(cf - Q[u].unsqueeze(0), dim=1)
        logits = -(eps_node / (2.0 * utility_sensitivity)) * dist
        sel[j] = int(cand[gumbel_max(logits)].item())

    return {"proto_indices": sel, "labels": lbl}


class ProtoDataset(torch.utils.data.Dataset):
    def __init__(self, pub_dict, indices, labels):
        self.d, self.i, self.l = pub_dict, indices.long(), labels.long()
    def __len__(self): return self.i.numel()
    def __getitem__(self, idx):
        p = self.d[int(self.i[idx])]
        return Data(x=p["x"], edge_index=p["edge_index"], y=self.l[idx].view(1))


class PrivGCN(nn.Module):
    def __init__(self, in_ch, hidden, out_ch):
        super().__init__()
        self.c1  = GCNConv(in_ch, hidden)
        self.c2  = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, out_ch)
    def forward(self, x, ei, batch):
        x = self.c1(x, ei).relu()
        x = self.c2(x, ei).relu()
        return self.lin(global_mean_pool(x, batch))


def build_val_loader(x, labels, val_nodes, val_ei, query_hops, max_val, seed, batch_size):
    val_nodes = cap_indices(val_nodes, max_val, seed)
    val_ei    = coalesce(to_undirected(val_ei, num_nodes=x.size(0)), num_nodes=x.size(0))
    items = []
    for v in val_nodes.tolist():
        sub, ei_sub, _, _ = k_hop_subgraph(v, query_hops, val_ei,
                                           relabel_nodes=True, num_nodes=x.size(0))
        items.append(Data(x=x[sub], edge_index=ei_sub, y=labels[v].unsqueeze(0)))
    return DataLoader(items, batch_size=batch_size, shuffle=False), val_nodes


def train_on_assignments(assignments, pub_dict, val_loader,
                          num_features, num_classes, hidden, epochs, batch_size):
    ds     = ProtoDataset(pub_dict, assignments["proto_indices"], assignments["labels"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model  = PrivGCN(num_features, hidden, num_classes).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=0.01)

    best_val, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for b in loader:
            b = b.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            x_aug  = b.x + torch.randn_like(b.x) * 0.05
            ei_aug, _ = dropout_edge(b.edge_index, p=0.10)
            if ei_aug.numel() == 0:
                ei_aug = b.edge_index
            loss = F.cross_entropy(model(x_aug, ei_aug, b.batch), b.y)
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for vb in val_loader:
                vb   = vb.to(device, non_blocking=True)
                pred = model(vb.x, vb.edge_index, vb.batch).argmax(1)
                correct += int((pred == vb.y).sum())
        val_acc = correct / max(len(val_loader.dataset), 1)
        if val_acc > best_val:
            best_val  = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val


# ─────────────────────────────────────────────
# Sweep runner
# ─────────────────────────────────────────────
def run_sweep(cfg, output_path, verbose=True):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    fieldnames = ["dataset", "epsilon", "seed", "public_frac", "dict_per_class",
                  "query_mode", "val_acc"]

    for ds_name in cfg["datasets"]:
        key = ds_name.lower()
        preset = DATASET_PRESETS.get(key, DATASET_PRESETS["cora"])
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        data, num_classes = load_dataset(ds_name)
        x = data.x

        pub_nodes, train_nodes, val_nodes = stratified_split(
            data.y, cfg["public_frac"], cfg.get("val_frac", 0.20), seed=42)
        train_nodes = cap_indices(train_nodes, preset["max_private"], seed=43)
        val_nodes   = cap_indices(val_nodes,   preset["max_val"],     seed=44)

        pub_ei,   _ = subgraph(pub_nodes,   data.edge_index, relabel_nodes=False,
                               num_nodes=data.num_nodes)
        priv_ei,  _ = subgraph(train_nodes, data.edge_index, relabel_nodes=False,
                               num_nodes=data.num_nodes)
        val_ei,   _ = subgraph(val_nodes,   data.edge_index, relabel_nodes=False,
                               num_nodes=data.num_nodes)

        pub_dict, dict_feats, class_to_idx = build_dictionary(
            data, x, data.y, pub_nodes, pub_ei, num_classes,
            dict_per_class    = cfg["dict_per_class"],
            walk_hops         = cfg.get("walk_hops", 2),
            query_mode        = cfg.get("query_mode", "one_hop"),
            min_class_fraction= cfg.get("min_class_fraction", 1.0),
            max_proto_nodes   = preset["max_proto"],
            verbose           = verbose,
        )

        val_loader, val_nodes_used = build_val_loader(
            x, data.y, val_nodes, val_ei,
            query_hops = cfg.get("query_hops", 1),
            max_val    = preset["max_val"],
            seed       = 47,
            batch_size = preset["batch"] * 4,
        )
        utility_sens = 2.0 if cfg.get("query_mode") == "two_hop_concat" else 1.0

        for eps in cfg["epsilons"]:
            for seed in cfg["seeds"]:
                set_seed(seed)
                print(f"\n  eps={eps}  seed={seed}")
                asgn = sample_assignments(
                    priv_ei, x, data.y, train_nodes,
                    dict_feats, class_to_idx, epsilon=eps,
                    utility_sensitivity = utility_sens,
                    query_mode          = cfg.get("query_mode", "one_hop"),
                    label_conditioning  = cfg.get("label_conditioning", True),
                )
                _, val_acc = train_on_assignments(
                    asgn, pub_dict, val_loader,
                    num_features = x.size(1),
                    num_classes  = num_classes,
                    hidden       = preset["hidden"],
                    epochs       = preset["epochs"],
                    batch_size   = preset["batch"],
                )
                print(f"  -> val acc = {val_acc:.4f}")
                rows.append({
                    "dataset":        ds_name,
                    "epsilon":        eps,
                    "seed":           seed,
                    "public_frac":    cfg["public_frac"],
                    "dict_per_class": cfg["dict_per_class"],
                    "query_mode":     cfg.get("query_mode", "one_hop"),
                    "val_acc":        round(val_acc, 6),
                })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {output_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_set(pairs):
    """Parse --set key=value overrides, coercing types."""
    out = {}
    for pair in pairs:
        k, _, v = pair.partition("=")
        k = k.strip()
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                if v.lower() == "true":
                    out[k] = True
                elif v.lower() == "false":
                    out[k] = False
                else:
                    out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(description="MM-edgeDP sweep runner")
    parser.add_argument("--phase",   type=int, required=True, choices=[1, 2],
                        help="1 = default settings, 2 = custom (use --set to override)")
    parser.add_argument("--set",     nargs="*", default=[],
                        metavar="key=value",
                        help="Override config values, e.g. --set public_frac=0.2")
    parser.add_argument("--output",  type=str, default=None,
                        help="Path for output CSV (default: outputs/sweep_phase<N>.csv)")
    parser.add_argument("--datasets",nargs="*", default=None,
                        help="Override dataset list, e.g. --datasets Cora CiteSeer")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = dict(PHASE1_DEFAULTS if args.phase == 1 else PHASE2_DEFAULTS)

    # Apply --set overrides
    overrides = parse_set(args.set)
    cfg.update(overrides)

    # Apply --datasets override
    if args.datasets:
        cfg["datasets"] = args.datasets

    output = args.output or f"outputs/sweep_phase{args.phase}.csv"

    print(f"Phase {args.phase} sweep")
    print(f"Datasets:      {cfg['datasets']}")
    print(f"Epsilons:      {cfg['epsilons']}")
    print(f"Seeds:         {cfg['seeds']}")
    print(f"public_frac:   {cfg['public_frac']}")
    print(f"dict_per_class:{cfg['dict_per_class']}")
    print(f"Output:        {output}")
    print(f"Device:        {device}")

    run_sweep(cfg, output_path=output, verbose=args.verbose)


if __name__ == "__main__":
    main()
