"""
Real-graph data loaders for DP-OT transfer experiments.

Supported setups
----------------
Twitch:
  Multiple language-specific user networks (EN, DE, FR, RU, PT, ES).
  Same node feature schema across graphs → natural cross-graph transfer.
  Usage: load_twitch_pair("EN", "DE", root="data/twitch")

OGB-arxiv temporal split:
  Use papers before year Y as source, papers from year Y onward as target.
  Induces temporal distribution shift.
  Usage: load_ogb_arxiv_temporal(source_year=2017, target_year=2018, root="data/ogb")

In both cases the function returns (G_source, G_target) as PyG Data objects
with x, edge_index, y already populated.

The target graph is treated as private: only x, edge_index are used during
DP adaptation; y is held out for final evaluation only.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, subgraph


# ---------------------------------------------------------------------------
# Twitch
# ---------------------------------------------------------------------------

TWITCH_LANGS = ("DE", "EN", "ES", "FR", "PT", "RU")


def load_twitch(lang: str, root: str = "data/twitch") -> Data:
    """
    Load one Twitch language graph via PyG's Twitch dataset.

    Returns PyG Data with x (float32), edge_index, y (int64: 0/1 mature).
    """
    from torch_geometric.datasets import Twitch
    dataset = Twitch(root=root, name=lang)
    data = dataset[0]
    data.x = data.x.float()
    data.y = data.y.long()
    return data


def load_twitch_pair(
    source_lang: str,
    target_lang: str,
    root: str = "data/twitch",
) -> tuple[Data, Data]:
    """
    Load a (source, target) Twitch pair.

    Example: load_twitch_pair("EN", "DE")
    """
    assert source_lang in TWITCH_LANGS, f"Unknown Twitch language: {source_lang}"
    assert target_lang in TWITCH_LANGS, f"Unknown Twitch language: {target_lang}"
    G_source = load_twitch(source_lang, root=root)
    G_target = load_twitch(target_lang, root=root)
    return G_source, G_target


# ---------------------------------------------------------------------------
# OGB-arxiv temporal split
# ---------------------------------------------------------------------------

def load_ogb_arxiv_temporal(
    source_before_year: int = 2018,
    target_from_year: int = 2018,
    root: str = "data/ogb",
) -> tuple[Data, Data]:
    """
    Load OGB-arxiv and split into source (older papers) and target (newer papers).

    The temporal split induces a topic-drift / distribution shift.
    source_before_year  : papers with year < source_before_year go to source
    target_from_year    : papers with year >= target_from_year go to target
    (gap between years is discarded if source_before_year < target_from_year)

    Returns (G_source, G_target) as induced subgraphs of the full arxiv graph.
    """
    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.transforms import ToUndirected

    # PyTorch >=2.6 defaults torch.load(weights_only=True), which cannot unpickle
    # OGB's cached PyG graph (custom classes such as DataEdgeAttr). The dataset
    # comes from the official OGB host (snap.stanford.edu), so loading with
    # weights_only=False is safe. Patch torch.load only around construction.
    _orig_load = torch.load

    def _compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _compat_load
    try:
        dataset = PygNodePropPredDataset("ogbn-arxiv", root=root, transform=ToUndirected())
    finally:
        torch.load = _orig_load
    data = dataset[0]

    # Node years are stored in data.node_year (shape: n×1)
    years = data.node_year.squeeze().numpy()

    src_mask = years < source_before_year
    tgt_mask = years >= target_from_year

    G_source = _induced_subgraph(data, src_mask)
    G_target = _induced_subgraph(data, tgt_mask)
    return G_source, G_target


def _induced_subgraph(data: Data, mask: np.ndarray) -> Data:
    """Extract a node-induced subgraph from a PyG Data object."""
    node_ids = torch.where(torch.tensor(mask))[0]
    edge_idx, _ = subgraph(node_ids, data.edge_index, relabel_nodes=True,
                           num_nodes=data.num_nodes)
    x = data.x[node_ids].float()
    y = data.y[node_ids].squeeze().long()
    return Data(x=x, edge_index=edge_idx, y=y, num_nodes=int(mask.sum()))


# ---------------------------------------------------------------------------
# Graph domain-adaptation citation networks (ACMv9 / DBLPv7 / Citationv1)
# ---------------------------------------------------------------------------
#
# The canonical *covariate-shift* graph-transfer benchmarks (ArnetMiner cross-
# domain citation networks, as used by AdaGCN / UDAGCN). All three share a common
# bag-of-words feature space and a common label space, so any (source, target)
# pair is a genuine domain-adaptation problem — unlike OGB-arxiv temporal, whose
# shift is mostly label-prior, not covariate.
#
# Distributed as MATLAB .mat files (acmv9.mat, dblpv7.mat, citationv1.mat) with
#     attrb   : (n, d) node features (sparse or dense)
#     network : (n, n) sparse adjacency
#     group   : (n, c) one-hot labels
# There is no pip-installable canonical source: download the .mat files once and
# place them under `root` (the notebook shows upload / gdown options).

GRAPH_DA_DATASETS = ("acmv9", "dblpv7", "citationv1")


def _first_present(mat: dict, keys: list[str]):
    for k in keys:
        if k in mat:
            return mat[k]
    present = [k for k in mat if not k.startswith("__")]
    raise KeyError(f"None of {keys} found in .mat; keys present: {present}")


def _to_dense(a) -> np.ndarray:
    import scipy.sparse as sp
    return np.asarray(a.todense()) if sp.issparse(a) else np.asarray(a)


def load_graph_da(name: str, root: str = "data/graph_da", url: str | None = None) -> Data:
    """
    Load one graph-DA citation network from a local .mat file.

    Looks for `{root}/{name}.mat` (keys attrb/network/group, with alternative
    names tried). If absent and `url` is given, downloads it first; otherwise
    raises with acquisition instructions.
    """
    import os
    import scipy.io as sio
    import scipy.sparse as sp

    name = name.lower()
    path = os.path.join(root, f"{name}.mat")
    if not os.path.exists(path) and url:
        os.makedirs(root, exist_ok=True)
        import urllib.request
        print(f"Downloading {name} from {url} ...", flush=True)
        urllib.request.urlretrieve(url, path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. The ACMv9/DBLPv7/Citationv1 graph-DA datasets "
            f"are not pip-installable. Download the .mat files (UDAGCN / AdaGCN "
            f"repositories, or the ArnetMiner cross-domain citation release) and "
            f"place '{name}.mat' under '{root}/'. In Colab you can upload via "
            f"google.colab.files.upload() or fetch with gdown, then re-run."
        )

    mat = sio.loadmat(path)
    X = _to_dense(_first_present(mat, ["attrb", "attr", "features", "X", "fea"])).astype(np.float32)
    A = _first_present(mat, ["network", "adj", "A", "W"])
    G = _to_dense(_first_present(mat, ["group", "label", "labels", "Y", "gnd"]))

    n = X.shape[0]
    y = G.argmax(axis=1) if (G.ndim == 2 and G.shape[1] > 1) else G.reshape(-1).astype(np.int64)

    coo = sp.coo_matrix(A)
    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=n)

    return Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        num_nodes=n,
    )


def load_graph_da_pair(
    source: str,
    target: str,
    root: str = "data/graph_da",
    source_url: str | None = None,
    target_url: str | None = None,
) -> tuple[Data, Data]:
    """Load a (source, target) graph-DA pair, e.g. ('acmv9', 'dblpv7')."""
    G_source = load_graph_da(source, root=root, url=source_url)
    G_target = load_graph_da(target, root=root, url=target_url)
    if G_source.x.shape[1] != G_target.x.shape[1]:
        raise ValueError(
            f"Feature-dim mismatch: source {G_source.x.shape[1]} vs target "
            f"{G_target.x.shape[1]}. A graph-DA pair must share a vocabulary — use "
            f"the aligned ArnetMiner release where all graphs have the same d."
        )
    return G_source, G_target


# ---------------------------------------------------------------------------
# Utility: compute basic shift diagnostics (non-private, for analysis only)
# ---------------------------------------------------------------------------

def compute_feature_shift(G_source: Data, G_target: Data) -> dict[str, float]:
    """
    Non-private summary of feature distribution shift (for paper diagnostics).

    Returns MMD^2 and mean-shift norm.
    """
    xs = G_source.x.numpy().astype(np.float64)
    xt = G_target.x.numpy().astype(np.float64)

    mean_s = xs.mean(axis=0)
    mean_t = xt.mean(axis=0)
    mean_shift = float(np.linalg.norm(mean_t - mean_s))

    # Linear-kernel MMD^2 (fast, no kernel trick needed for the mean statistic)
    mmd2 = float(np.sum((mean_t - mean_s) ** 2))

    return {"mean_shift_norm": mean_shift, "linear_mmd2": mmd2}
