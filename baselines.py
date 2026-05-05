"""
baselines.py — Edge-DP baseline runner (Gaussian SGC, Edge RR, GAP-EDP)

Example:
    python3 baselines.py --datasets Cora AmazonPhoto Actor \
        --public_frac 0.2 --dict_per_class 128 --verbose --epsilon 1

Results are printed to stdout and saved to outputs/baselines_<timestamp>.csv.
Use --output to override the output path.
"""

import argparse
import csv
import math
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import coalesce, subgraph, to_undirected

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PRESETS = {
    "cora":          dict(max_private=None,  max_val=None,  epochs=50,  hidden=16,  rr_hidden=256, rr_epochs=200, enc_dim=16, enc_ep=50,  clf_ep=100, K=2),
    "citeseer":      dict(max_private=None,  max_val=None,  epochs=50,  hidden=16,  rr_hidden=256, rr_epochs=200, enc_dim=16, enc_ep=50,  clf_ep=100, K=2),
    "pubmed":        dict(max_private=10000, max_val=2000,  epochs=25,  hidden=32,  rr_hidden=256, rr_epochs=200, enc_dim=32, enc_ep=100, clf_ep=200, K=2),
    "ogbn-arxiv":    dict(max_private=30000, max_val=3000,  epochs=8,   hidden=32,  rr_hidden=128, rr_epochs=100, enc_dim=64, enc_ep=50,  clf_ep=100, K=2),
    "amazonphoto":   dict(max_private=None,  max_val=2000,  epochs=50,  hidden=32,  rr_hidden=256, rr_epochs=200, enc_dim=32, enc_ep=50,  clf_ep=100, K=2),
    "actor":         dict(max_private=None,  max_val=2000,  epochs=50,  hidden=32,  rr_hidden=256, rr_epochs=200, enc_dim=32, enc_ep=50,  clf_ep=100, K=2),
    "squirrel":      dict(max_private=None,  max_val=2000,  epochs=50,  hidden=32,  rr_hidden=256, rr_epochs=200, enc_dim=32, enc_ep=50,  clf_ep=100, K=2),
    "chameleon":     dict(max_private=None,  max_val=2000,  epochs=50,  hidden=32,  rr_hidden=256, rr_epochs=200, enc_dim=32, enc_ep=50,  clf_ep=100, K=2),
    "facebookpagepage": dict(max_private=None, max_val=3000, epochs=50, hidden=32, rr_hidden=256, rr_epochs=200, enc_dim=32, enc_ep=50, clf_ep=100, K=2),
    "lastfmasia":    dict(max_private=None,  max_val=2000,  epochs=50,  hidden=32,  rr_hidden=256, rr_epochs=200, enc_dim=32, enc_ep=50,  clf_ep=100, K=2),
}


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


def stratified_split(y, public_frac=0.20, val_frac=0.20, seed=0):
    g = torch.Generator().manual_seed(seed)
    pub, val, train = [], [], []
    for c in torch.unique(y).tolist():
        idx  = torch.where(y == c)[0]
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        n_pub = max(1, int(public_frac * idx.numel()))
        n_val = max(1, int(val_frac   * idx.numel()))
        n_pub = min(n_pub, idx.numel())
        n_val = min(n_val, max(idx.numel() - n_pub, 0))
        pub.append(perm[:n_pub])
        val.append(perm[n_pub:n_pub + n_val])
        train.append(perm[n_pub + n_val:])
    mk = lambda lst: torch.cat(lst) if lst else torch.tensor([], dtype=torch.long)
    return mk(pub), mk(train), mk(val)


def load_dataset(name):
    import torch_geometric.transforms as T
    from torch_geometric.utils import coalesce, to_undirected
    key = name.lower().replace("-", "").replace("_", "")

    if key == "ogbnarxiv":
        from ogb.nodeproppred import PygNodePropPredDataset
        ds   = PygNodePropPredDataset(name="ogbn-arxiv", root="data/ogb")
        data = ds[0]
        data.y = data.y.squeeze(-1).long()
    elif key in ("cora", "citeseer", "pubmed"):
        pretty = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[key]
        ds   = __import__("torch_geometric.datasets", fromlist=["Planetoid"]).Planetoid(
            root=f"data/Planetoid/{pretty}", name=pretty, transform=T.NormalizeFeatures())
        data = ds[0]
        data.y = data.y.long()
    elif key == "amazonphoto":
        from torch_geometric.datasets import Amazon
        ds   = Amazon(root="data/Amazon", name="Photo")
        data = ds[0]
        data.y = data.y.long()
    elif key == "actor":
        from torch_geometric.datasets import Actor
        ds   = Actor(root="data/Actor")
        data = ds[0]
        data.y = data.y.long()
    elif key == "squirrel":
        from torch_geometric.datasets import WikipediaNetwork
        ds   = WikipediaNetwork(root="data/Wikipedia", name="squirrel")
        data = ds[0]
        data.y = data.y.long()
    elif key == "chameleon":
        from torch_geometric.datasets import WikipediaNetwork
        ds   = WikipediaNetwork(root="data/Wikipedia", name="chameleon")
        data = ds[0]
        data.y = data.y.long()
    elif key == "facebookpagepage":
        from torch_geometric.datasets import FacebookPagePage
        ds   = FacebookPagePage(root="data/Facebook")
        data = ds[0]
        data.y = data.y.long()
    elif key == "lastfmasia":
        from torch_geometric.datasets import LastFMAsia
        ds   = LastFMAsia(root="data/LastFM")
        data = ds[0]
        data.y = data.y.long()
    else:
        raise ValueError(
            f"Unknown dataset: {name}\n"
            "Supported: Cora, CiteSeer, PubMed, ogbn-arxiv, AmazonPhoto, "
            "Actor, Squirrel, Chameleon, FacebookPagePage, LastFMAsia"
        )

    data.x = F.normalize(data.x.float(), p=2, dim=1)
    data.edge_index = coalesce(
        to_undirected(data.edge_index, num_nodes=data.num_nodes),
        num_nodes=data.num_nodes)
    num_classes = int(data.y.max().item()) + 1
    return data, num_classes


# ─────────────────────────────────────────────
# Baseline 1: Gaussian SGC
# ─────────────────────────────────────────────
def run_gaussian_sgc(data, priv_ei, val_ei, train_nodes, val_nodes,
                      epsilon, delta=1e-5, epochs=50, hidden=16, verbose=False):
    n, d       = data.num_nodes, data.x.size(1)
    x_norm     = F.normalize(data.x, p=2, dim=1)
    num_classes = int(data.y.max().item()) + 1

    def agg(ei):
        ei  = coalesce(to_undirected(ei, num_nodes=n), num_nodes=n)
        out = torch.zeros(n, d)
        if ei.numel() > 0:
            row, col = ei
            out.scatter_add_(0, row.unsqueeze(1).expand(-1, d), x_norm[col])
        return out

    x_priv = agg(priv_ei)
    sigma   = (2.0 / epsilon) * math.sqrt(2.0 * math.log(1.25 / delta))
    x_noisy = x_priv + torch.randn_like(x_priv) * sigma

    x_val_agg = agg(val_ei)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(d, hidden)
            self.l2 = nn.Linear(hidden, num_classes)
        def forward(self, x):
            return self.l2(F.relu(self.l1(x)))

    model = MLP().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01)

    tx = x_noisy[train_nodes].to(device)
    ty = data.y[train_nodes].to(device)
    vx = x_val_agg[val_nodes].to(device)
    vy = data.y[val_nodes].to(device)

    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(tx), ty).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(vx).argmax(1) == vy).float().mean().item()
        best = max(best, acc)
        if verbose and ep % max(1, epochs // 5) == 0:
            print(f"    [SGC] ep {ep}/{epochs}  val={acc:.4f}")

    return best


# ─────────────────────────────────────────────
# Baseline 2: Edge Randomized Response
# ─────────────────────────────────────────────
class GCN2L(nn.Module):
    def __init__(self, in_ch, hidden, out_ch):
        super().__init__()
        self.c1 = GCNConv(in_ch, hidden)
        self.c2 = GCNConv(hidden, out_ch)
    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.c2(x, ei)


def apply_edge_rr(edge_index, n, epsilon):
    p     = math.exp(epsilon) / (1.0 + math.exp(epsilon))
    ei    = coalesce(to_undirected(edge_index, num_nodes=n), num_nodes=n)
    m     = ei.size(1)
    keep  = torch.rand(m) < p
    kept  = ei[:, keep]
    n_add = int((~keep).sum())
    if n_add > 0:
        fs = torch.randint(0, n, (n_add,))
        fd = torch.randint(0, n, (n_add,))
        ok = fs != fd
        fake  = torch.stack([fs[ok], fd[ok]])
        out   = torch.cat([kept, fake], dim=1)
    else:
        out = kept
    return coalesce(to_undirected(out, num_nodes=n), num_nodes=n)


def run_edge_rr(data, train_nodes, val_nodes, epsilon,
                epochs=200, hidden=256, verbose=False):
    num_classes = int(data.y.max().item()) + 1
    ei_rr  = apply_edge_rr(data.edge_index, data.num_nodes, epsilon)
    x_d    = data.x.to(device)
    ei_d   = ei_rr.to(device)
    y_d    = data.y.squeeze().to(device)
    tn_d   = train_nodes.to(device)
    vn_d   = val_nodes.to(device)

    model = GCN2L(x_d.size(1), hidden, num_classes).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x_d, ei_d)[tn_d], y_d[tn_d]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(x_d, ei_d).argmax(1)[vn_d] == y_d[vn_d]).float().mean().item()
        best = max(best, acc)
        if verbose and ep % max(1, epochs // 5) == 0:
            print(f"    [RR]  ep {ep}/{epochs}  val={acc:.4f}")

    return best


# ─────────────────────────────────────────────
# Baseline 3: GAP-EDP
# ─────────────────────────────────────────────
def calibrate_sigma(epsilon, delta, K):
    if math.isinf(epsilon):
        return 0.0
    b = math.sqrt(2.0 * K * math.log(1.0 / delta))
    t = (-b + math.sqrt(b * b + 2.0 * K * epsilon)) / K
    return 1.0 / t


def pma_step(x_hat, ei, n, sigma):
    d     = x_hat.size(1)
    ei    = coalesce(to_undirected(ei, num_nodes=n), num_nodes=n)
    x_agg = torch.zeros(n, d)
    if ei.numel() > 0:
        row, col = ei
        x_agg.scatter_add_(0, row.unsqueeze(1).expand(-1, d), x_hat[col])
    if sigma > 0:
        x_agg = x_agg + torch.randn_like(x_agg) * sigma
    return F.normalize(x_agg, p=2, dim=1)


class GAPEncoder(nn.Module):
    def __init__(self, in_ch, enc_dim, out_ch):
        super().__init__()
        self.enc  = nn.Sequential(nn.Linear(in_ch, enc_dim * 2), nn.ReLU(),
                                  nn.Linear(enc_dim * 2, enc_dim))
        self.head = nn.Linear(enc_dim, out_ch)
    def forward(self, x):
        z = self.enc(x)
        return self.head(z), z


class GAPClassifier(nn.Module):
    def __init__(self, in_ch, hidden, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_ch, hidden), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_ch))
    def forward(self, x):
        return self.net(x)


def run_gap_edp(data, train_nodes, val_nodes, epsilon,
                enc_dim=16, enc_epochs=50, clf_epochs=100, K=2,
                delta=None, verbose=False):
    num_classes = int(data.y.max().item()) + 1
    n_edges     = int(data.edge_index.size(1))
    if delta is None:
        delta = 1.0 / (10 ** len(str(n_edges)))

    x_gpu = data.x.to(device)
    y_gpu = data.y.squeeze().to(device)
    tn_d  = train_nodes.to(device)
    vn_d  = val_nodes.to(device)

    # Step 1: encoder (no edges)
    enc     = GAPEncoder(x_gpu.size(1), enc_dim, num_classes).to(device)
    enc_opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    for ep in range(1, enc_epochs + 1):
        enc.train()
        enc_opt.zero_grad()
        logits, _ = enc(x_gpu[tn_d])
        F.cross_entropy(logits, y_gpu[tn_d]).backward()
        enc_opt.step()
        if verbose and ep % max(1, enc_epochs // 5) == 0:
            enc.eval()
            with torch.no_grad():
                _, z_all = enc(x_gpu)
                x0_tmp   = F.normalize(z_all, p=2, dim=1)
            print(f"    [GAP enc] ep {ep}/{enc_epochs}")

    enc.eval()
    with torch.no_grad():
        _, z_all = enc(x_gpu)
        x0 = F.normalize(z_all.cpu(), p=2, dim=1)

    # Step 2: PMA
    sigma  = calibrate_sigma(epsilon, delta, K)
    x_hat  = x0.clone()
    hops   = []
    for _ in range(K):
        x_hat = pma_step(x_hat, data.edge_index, data.num_nodes, sigma)
        hops.append(x_hat)
    x_comb = torch.cat([x0] + hops, dim=1).to(device)

    if verbose:
        print(f"    [GAP] sigma={sigma:.4f}  combined_dim={x_comb.size(1)}")

    # Step 3: classifier
    clf     = GAPClassifier(x_comb.size(1), enc_dim * 4, num_classes).to(device)
    clf_opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    best    = 0.0
    for ep in range(1, clf_epochs + 1):
        clf.train()
        clf_opt.zero_grad()
        F.cross_entropy(clf(x_comb[tn_d]), y_gpu[tn_d]).backward()
        clf_opt.step()
        clf.eval()
        with torch.no_grad():
            acc = (clf(x_comb).argmax(1)[vn_d] == y_gpu[vn_d]).float().mean().item()
        best = max(best, acc)
        if verbose and ep % max(1, clf_epochs // 5) == 0:
            print(f"    [GAP clf] ep {ep}/{clf_epochs}  val={acc:.4f}")

    return best


# ─────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────
def run_baselines(datasets, epsilons, public_frac, val_frac, seed,
                  dict_per_class, verbose, output_path):
    rows       = []
    fieldnames = ["dataset", "epsilon", "method", "val_acc",
                  "public_frac", "seed"]

    for ds_name in datasets:
        key    = ds_name.lower().replace("-", "").replace("_", "")
        preset = DATASET_PRESETS.get(key, DATASET_PRESETS["cora"])

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}  (device={device})")
        print(f"{'='*60}")
        set_seed(seed)
        data, num_classes = load_dataset(ds_name)
        print(f"  nodes={data.num_nodes}  edges={data.edge_index.size(1)}  classes={num_classes}")

        pub_nodes, train_nodes, val_nodes = stratified_split(
            data.y, public_frac, val_frac, seed=seed)
        train_nodes = cap_indices(train_nodes, preset["max_private"], seed=seed + 1)
        val_nodes   = cap_indices(val_nodes,   preset["max_val"],     seed=seed + 2)
        print(f"  split: pub={pub_nodes.numel()}  train={train_nodes.numel()}  val={val_nodes.numel()}")

        priv_ei, _ = subgraph(train_nodes, data.edge_index,
                               relabel_nodes=False, num_nodes=data.num_nodes)
        val_ei,  _ = subgraph(val_nodes,   data.edge_index,
                               relabel_nodes=False, num_nodes=data.num_nodes)

        for eps in epsilons:
            print(f"\n  ε = {eps}")

            # --- Gaussian SGC ---
            print("  Running Gaussian SGC...")
            sgc_acc = run_gaussian_sgc(
                data, priv_ei, val_ei, train_nodes, val_nodes,
                epsilon=eps, epochs=preset["epochs"],
                hidden=preset["hidden"], verbose=verbose,
            )
            print(f"    Gaussian SGC  -> {sgc_acc:.4f}")
            rows.append({"dataset": ds_name, "epsilon": eps, "method": "gaussian_sgc",
                         "val_acc": round(sgc_acc, 6),
                         "public_frac": public_frac, "seed": seed})

            # --- Edge RR ---
            print("  Running Edge RR...")
            rr_acc = run_edge_rr(
                data, train_nodes, val_nodes, epsilon=eps,
                epochs=preset["rr_epochs"], hidden=preset["rr_hidden"], verbose=verbose,
            )
            print(f"    Edge RR       -> {rr_acc:.4f}")
            rows.append({"dataset": ds_name, "epsilon": eps, "method": "edge_rr",
                         "val_acc": round(rr_acc, 6),
                         "public_frac": public_frac, "seed": seed})

            # --- GAP-EDP ---
            print("  Running GAP-EDP...")
            gap_acc = run_gap_edp(
                data, train_nodes, val_nodes, epsilon=eps,
                enc_dim    = preset["enc_dim"],
                enc_epochs = preset["enc_ep"],
                clf_epochs = preset["clf_ep"],
                K          = preset["K"],
                verbose    = verbose,
            )
            print(f"    GAP-EDP       -> {gap_acc:.4f}")
            rows.append({"dataset": ds_name, "epsilon": eps, "method": "gap_edp",
                         "val_acc": round(gap_acc, 6),
                         "public_frac": public_frac, "seed": seed})

        # Per-dataset summary
        print(f"\n  Summary for {ds_name}:")
        print(f"  {'epsilon':<10}{'Gaussian SGC':>14}{'Edge RR':>10}{'GAP-EDP':>10}")
        print(f"  {'-'*44}")
        for eps in epsilons:
            sgc = next(r["val_acc"] for r in rows
                       if r["dataset"] == ds_name and r["epsilon"] == eps
                       and r["method"] == "gaussian_sgc")
            rr  = next(r["val_acc"] for r in rows
                       if r["dataset"] == ds_name and r["epsilon"] == eps
                       and r["method"] == "edge_rr")
            gap = next(r["val_acc"] for r in rows
                       if r["dataset"] == ds_name and r["epsilon"] == eps
                       and r["method"] == "gap_edp")
            print(f"  {eps:<10.1f}{sgc:>14.4f}{rr:>10.4f}{gap:>10.4f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {output_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Edge-DP baseline runner")
    parser.add_argument("--datasets",      nargs="+", required=True,
                        help="Datasets to evaluate, e.g. Cora AmazonPhoto Actor")
    parser.add_argument("--epsilon",       nargs="+", type=float, default=[1.0],
                        help="Privacy budget(s), e.g. --epsilon 0.5 1 4")
    parser.add_argument("--public_frac",   type=float, default=0.20)
    parser.add_argument("--val_frac",      type=float, default=0.20)
    parser.add_argument("--dict_per_class",type=int,   default=50,
                        help="(Informational — not used by baselines, kept for parity with sweep.py)")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--output",        type=str,   default=None)
    parser.add_argument("--verbose",       action="store_true")
    args = parser.parse_args()

    stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or f"outputs/baselines_{stamp}.csv"

    print("Edge-DP Baseline Runner")
    print(f"  Datasets:    {args.datasets}")
    print(f"  Epsilons:    {args.epsilon}")
    print(f"  public_frac: {args.public_frac}")
    print(f"  seed:        {args.seed}")
    print(f"  output:      {output}")
    print(f"  device:      {device}")

    run_baselines(
        datasets    = args.datasets,
        epsilons    = args.epsilon,
        public_frac = args.public_frac,
        val_frac    = args.val_frac,
        seed        = args.seed,
        dict_per_class = args.dict_per_class,
        verbose     = args.verbose,
        output_path = output,
    )


if __name__ == "__main__":
    main()
