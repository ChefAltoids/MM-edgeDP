"""
Mechanism A: DP histogram via Laplace noise on hard prototype assignments.

Privacy analysis (edge-DP):
  Adding/removing one edge (u,v) can change u's and v's summaries.
  Each changed summary can shift its prototype bin by 1.
  L1 sensitivity of the count vector = 4.

  => Add Lap(4/epsilon) to each count.
"""

from __future__ import annotations

import numpy as np


def dp_histogram_assign(
    summaries: np.ndarray,
    centroids: np.ndarray,
    epsilon: float,
    seed: int = 0,
) -> np.ndarray:
    """
    Privately estimate target prototype mass via Laplace-noised hard assignment.

    Parameters
    ----------
    summaries : (n_target, d_summary) target node summaries
    centroids : (K, d_summary) public source prototype centroids
    epsilon   : edge-DP privacy budget (pure DP, no delta)
    seed      : RNG seed for Laplace noise

    Returns
    -------
    alpha_dp : (K,) normalized target prototype masses (sum to 1)
    """
    K = len(centroids)
    rng = np.random.default_rng(seed)

    # Hard assignment: nearest centroid in L2
    dists = _sq_dists(summaries, centroids)  # (n, K)
    assignments = dists.argmin(axis=1)       # (n,)
    counts = np.bincount(assignments, minlength=K).astype(float)

    # Add Laplace noise with sensitivity 4
    sensitivity = 4.0
    if epsilon == float("inf") or epsilon <= 0:
        noisy_counts = counts
    else:
        noise = rng.laplace(loc=0.0, scale=sensitivity / epsilon, size=K)
        noisy_counts = counts + noise

    # Post-process: clip negatives, normalize
    noisy_counts = np.maximum(0.0, noisy_counts)
    total = noisy_counts.sum()
    if total == 0:
        return np.ones(K, dtype=np.float32) / K
    return (noisy_counts / total).astype(np.float32)


def nonprivate_histogram(
    summaries: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Non-private hard-assignment histogram (oracle baseline, epsilon=inf)."""
    return dp_histogram_assign(summaries, centroids, epsilon=float("inf"))


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared L2 distance matrix between rows of A (n,d) and B (K,d)."""
    # ||a - b||^2 = ||a||^2 - 2 a^T b + ||b||^2
    aa = (A ** 2).sum(axis=1, keepdims=True)   # (n, 1)
    bb = (B ** 2).sum(axis=1, keepdims=True).T  # (1, K)
    ab = A @ B.T                                # (n, K)
    return np.maximum(0.0, aa - 2 * ab + bb)
