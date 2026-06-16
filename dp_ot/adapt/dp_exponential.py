"""
Mechanism B: DP exponential mechanism for soft prototype assignment.

Each target node j independently samples a prototype k with probability:
  Pr[a_j = k] ∝ exp(-epsilon_row * C[j,k] / (2 * Delta_C))

where:
  C[j,k] = ||s_j - c_k||   (L2 distance — NOT squared; see below)
  Delta_C = per-row sensitivity of C[j,k] under edge-DP
  epsilon_row = epsilon / 2  (sequential composition over 2 affected rows per edge)

Why L2 distance and not squared distance:
  The exponential-mechanism score is u(s, k) = -C[j,k]. Its sensitivity is the
  most one summary can move the score when one edge changes s_j by Delta_s.
  For C = ||s - c|| (linear), the triangle inequality gives a TIGHT bound:
    | ||s' - c|| - ||s - c|| | <= ||s' - s|| = Delta_s,
  independent of the embedding diameter. For C = ||s - c||^2 (squared), the
  sensitivity instead scales with ||s|| + ||c|| (the diameter), which for
  bounded summaries is ~100x larger and washes the mechanism out to near-uniform
  sampling. So we use linear distance: Delta_C = Delta_s.

Sensitivity of one summary under edge-DP (Delta_s):
  When edge (u,v) is added/removed, u's summary s_u changes.
  The 1-hop mean component changes by at most B/d_max per dimension.
    Delta_s_feat   <= sqrt(d_feat) * B / d_max
    Delta_s_logdeg = |log(1+min(d+1,d_max)) - log(1+min(d,d_max))| <= 1
  Conservative bound: Delta_s <= sqrt(d_feat) * B / d_max + 1
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
    # Sensitivity of the LINEAR-distance cost C[j,k] = ||s_j - c_k||.
    # By the triangle inequality this is exactly Delta_s, independent of the
    # embedding diameter (unlike squared distance — see module docstring).
    delta_C = delta_s

    # epsilon_row = epsilon/2 by sequential composition (2 affected rows per edge)
    if epsilon == float("inf") or epsilon <= 0:
        epsilon_row = float("inf")
    else:
        epsilon_row = epsilon / 2.0

    # Cost matrix C[j,k] = ||s_j - c_k|| (L2 distance)
    C = np.sqrt(_sq_dists(summaries, centroids))  # (n, K)

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
