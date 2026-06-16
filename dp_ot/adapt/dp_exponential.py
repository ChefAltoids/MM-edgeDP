"""
Mechanism B: DP exponential mechanism for soft prototype assignment.

Each target node j independently samples a prototype k with probability:
  Pr[a_j = k] ∝ exp(-epsilon_row * C[j,k] / (2 * Delta_C))

where:
  C[j,k] = ||s_j - c_k||^2  (squared L2 cost)
  Delta_C = per-row sensitivity of C[j,k] under edge-DP
  epsilon_row = epsilon / 2  (sequential composition over 2 affected rows per edge)

Sensitivity derivation:
  When edge (u,v) added/removed, u's summary s_u changes.
  The 1-hop mean component changes by at most B/d_max per dimension.
  Total L2 change of s_u: Delta_s <= sqrt(d * (B/d_max)^2 + (log_deg_change)^2)
  For features of dimension d_feat:
    Delta_s_feat  <= sqrt(d_feat) * B / d_max
    Delta_s_logdeg = |log(1+min(d+1,d_max)) - log(1+min(d,d_max))| <= 1/(1+d)
  Conservative bound: Delta_s <= sqrt(d_feat) * B / d_max + 1

  Sensitivity of C[j,k] = ||s_j - c_k||^2:
    |C_new - C_old| <= (|s_j_new| + |c_k| + |s_j_old| + |c_k|) * Delta_s
                    <= 4 * (B_summary + B_centroid) * Delta_s
  We use B_summary = sqrt(d_feat + d_feat + 1) * B as a loose bound on ||s_j||,
  and bound ||c_k|| similarly.

  In practice we compute Delta_C from B and d_max directly.
"""

from __future__ import annotations

import numpy as np


def dp_exponential_assign(
    summaries: np.ndarray,
    centroids: np.ndarray,
    epsilon: float,
    B: float,
    d_max: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Privately estimate target prototype mass via the exponential mechanism.

    Parameters
    ----------
    summaries : (n_target, d_summary) target node summaries
    centroids : (K, d_summary) public source prototype centroids
    epsilon   : edge-DP privacy budget
    B         : feature clip bound used in compute_target_summaries
    d_max     : degree cap used in compute_target_summaries
    seed      : RNG seed

    Returns
    -------
    alpha_dp : (K,) normalized target prototype masses
    """
    rng = np.random.default_rng(seed)
    n, d_summary = summaries.shape
    K = len(centroids)
    d_feat = (d_summary - 1) // 2  # d_summary = d_feat + d_feat + 1

    # Sensitivity of one summary under edge-DP
    delta_s = np.sqrt(d_feat) * B / max(d_max, 1) + 1.0
    # Sensitivity of squared-distance cost
    # ||s-c||^2 changes by at most 4 * (||s|| + ||c||) * Delta_s
    # Use 4 * 2 * B_bound * Delta_s as conservative upper bound
    B_bound = np.sqrt(d_summary) * B  # rough bound on ||s_j|| and ||c_k||
    delta_C = 4.0 * 2.0 * B_bound * delta_s

    # epsilon_row = epsilon/2 by sequential composition (2 affected rows per edge)
    if epsilon == float("inf") or epsilon <= 0:
        epsilon_row = float("inf")
    else:
        epsilon_row = epsilon / 2.0

    # Cost matrix C[j,k] = ||s_j - c_k||^2
    C = _sq_dists(summaries, centroids)  # (n, K)

    # Sample prototype for each node via exponential mechanism (Gumbel-max trick)
    if epsilon_row == float("inf"):
        assignments = C.argmin(axis=1)
    else:
        log_probs = -epsilon_row * C / (2.0 * delta_C + 1e-30)  # (n, K)
        # Gumbel-max: a_j = argmax_k (log_prob[j,k] + Gumbel(0,1))
        gumbel = rng.gumbel(size=(n, K))
        assignments = (log_probs + gumbel).argmax(axis=1)

    counts = np.bincount(assignments, minlength=K).astype(float)
    total = counts.sum()
    if total == 0:
        return np.ones(K, dtype=np.float32) / K
    return (counts / total).astype(np.float32)


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = (A ** 2).sum(axis=1, keepdims=True)
    bb = (B ** 2).sum(axis=1, keepdims=True).T
    ab = A @ B.T
    return np.maximum(0.0, aa - 2 * ab + bb)
