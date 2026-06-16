"""
Public prototype construction and private target summary computation.

Prototypes:
  - K-means on source GNN embeddings
  - Returns centroids, per-node assignments, source prototype mass vector

Target summaries (edge-DP safe, low sensitivity):
  s_j = [clip_B(x_j), (1/d_max) * sum_{k in N(j, d_max)} clip_B(x_k), log(1 + min(deg_j, d_max))]

Edge-DP sensitivity of the count histogram: 4
  One edge (u,v) changes u's and v's summaries; each can shift one prototype bin.
  L1 change of count vector <= 4.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.utils import degree


def embed_nodes(model: "GraphSAGEModel", data: Data, device: str = "cpu") -> np.ndarray:
    """Run model.embed on data, return numpy array of shape (n, d_embed)."""
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    with torch.no_grad():
        z = model.embed(data.x.to(dev), data.edge_index.to(dev))
    return z.cpu().numpy()


def fit_public_prototypes(
    Z_source: np.ndarray,
    K: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cluster source embeddings into K prototypes.

    Returns
    -------
    centroids       : (K, d) prototype centroids
    assignments     : (n_source,) int array, prototype index per source node
    alpha_source    : (K,) source prototype mass (fraction of nodes in each prototype)
    """
    km = KMeans(n_clusters=K, random_state=seed, n_init="auto")
    assignments = km.fit_predict(Z_source)
    centroids = km.cluster_centers_
    counts = np.bincount(assignments, minlength=K)
    alpha_source = counts / counts.sum()
    return centroids, assignments, alpha_source


def compute_target_summaries(
    data: Data,
    d_max: int,
    B: float,
) -> np.ndarray:
    """
    Compute bounded local summaries for each target node.

    Each summary is:
      [clip_B(x_j), (1/d_max) * sum_{k in N(j, d_max)} clip_B(x_k), log(1 + min(deg_j, d_max))]

    Using fixed denominator d_max (not actual degree) keeps the 1-hop mean
    sensitivity bounded: adding one edge changes the mean by at most B/d_max per dim.

    Parameters
    ----------
    data    : target graph Data (needs x, edge_index)
    d_max   : degree cap (both for averaging and for log-degree feature)
    B       : L-inf clip bound for node features

    Returns
    -------
    summaries : (n, d_feat + d_feat + 1) float array
    """
    x = data.x.cpu().numpy()
    n, d = x.shape
    edge_index = data.edge_index.cpu().numpy()  # (2, E)

    # Clip raw features
    x_clipped = np.clip(x, -B, B)

    # Compute capped 1-hop mean with fixed denominator d_max
    # For each node j: mean_j = (1/d_max) * sum_{k in N(j)} clip(x_k)
    #   where we sum over at most d_max neighbors (truncate if more)
    neighbor_sum = np.zeros_like(x_clipped)
    neighbor_count = np.zeros(n, dtype=np.int64)

    src, dst = edge_index[0], edge_index[1]
    # Accumulate: for each edge (src->dst), dst receives src's clipped features
    # But we need to cap at d_max neighbors per node
    # Process edges grouped by destination
    order = np.argsort(dst)
    src_sorted = src[order]
    dst_sorted = dst[order]

    i = 0
    while i < len(dst_sorted):
        j = dst_sorted[i]
        # collect all neighbors of j
        start = i
        while i < len(dst_sorted) and dst_sorted[i] == j:
            i += 1
        neighbors = src_sorted[start:i]
        # cap at d_max
        if len(neighbors) > d_max:
            neighbors = neighbors[:d_max]
        neighbor_sum[j] = x_clipped[neighbors].sum(axis=0)
        neighbor_count[j] = len(neighbors)

    # Use fixed denominator d_max (not actual count)
    hop_mean = neighbor_sum / d_max  # (n, d)

    # Degree feature (count out-degree from edge_index src side)
    deg = np.zeros(n, dtype=np.float32)
    unique, counts = np.unique(src, return_counts=True)
    deg[unique] = counts
    log_deg = np.log1p(np.minimum(deg, d_max)).reshape(-1, 1)  # (n, 1)

    summaries = np.concatenate([x_clipped, hop_mean, log_deg], axis=1)
    return summaries.astype(np.float32)
