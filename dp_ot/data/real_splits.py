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
