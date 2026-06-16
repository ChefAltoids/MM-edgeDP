"""
Importance-reweighting: turn DP target prototype masses into per-source-node weights.

w_i = alpha_target[a(i)] / (alpha_source[a(i)] + rho)

Normalized so mean weight = 1 (preserves effective learning rate scale).
"""

from __future__ import annotations

import numpy as np


def reweight_source_nodes(
    source_assignments: np.ndarray,
    alpha_source: np.ndarray,
    alpha_target_dp: np.ndarray,
    rho: float = 1e-3,
) -> np.ndarray:
    """
    Compute per-source-node importance weights.

    Parameters
    ----------
    source_assignments : (n_source,) int array, prototype index per source node
    alpha_source       : (K,) source prototype mass fractions
    alpha_target_dp    : (K,) DP-estimated target prototype mass fractions
    rho                : small constant to prevent division by zero

    Returns
    -------
    weights : (n_source,) float array, mean-normalized importance weights
    """
    w = alpha_target_dp[source_assignments] / (alpha_source[source_assignments] + rho)
    mean_w = w.mean()
    if mean_w > 0:
        w = w / mean_w
    return w.astype(np.float32)


def uniform_weights(n: int) -> np.ndarray:
    """Unit weights for source-only baseline (no adaptation)."""
    return np.ones(n, dtype=np.float32)
