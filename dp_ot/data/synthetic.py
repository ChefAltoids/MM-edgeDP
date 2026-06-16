"""
Gaussian-mixture latent-variable graph generator for DP-OT experiments.

DGP:
  z_i ~ N(means[m_i], sigma^2 I),  m_i ~ Categorical(mixture_weights)
  x_i = z_i + N(0, sigma_x^2 I)
  y_i ~ Bernoulli(sigmoid(w^T z_i))
  Pr[(i,j) in E] = sigmoid(edge_alpha * z_i^T z_j + edge_beta)

Covariate shift: call twice with different mixture_weights.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def generate_synthetic_graph(
    n: int,
    mixture_weights: np.ndarray,
    means: np.ndarray,
    sigma: float,
    edge_alpha: float,
    edge_beta: float,
    seed: int,
    sigma_x: float | None = None,
    label_weight: np.ndarray | None = None,
) -> Data:
    """
    Generate one synthetic graph.

    Parameters
    ----------
    n               : number of nodes
    mixture_weights : (M,) array, must sum to 1
    means           : (M, d) array of component means
    sigma           : std of latent noise (and feature noise if sigma_x is None)
    edge_alpha      : coefficient of z_i^T z_j in edge logit
    edge_beta       : bias in edge logit
    seed            : RNG seed
    sigma_x         : feature noise std; defaults to sigma
    label_weight    : (d,) linear weight for labels; defaults to ones(d)/sqrt(d)

    Returns
    -------
    PyG Data with fields: x, edge_index, y, z, component
    """
    rng = np.random.default_rng(seed)
    M, d = means.shape
    if sigma_x is None:
        sigma_x = sigma

    # 1. Sample mixture components
    components = rng.choice(M, size=n, p=mixture_weights)

    # 2. Sample latent variables
    z = means[components] + sigma * rng.standard_normal((n, d))

    # 3. Sample labels via shared linear model
    if label_weight is None:
        w = np.ones(d) / np.sqrt(d)
    else:
        w = np.asarray(label_weight, dtype=float)
    logits = z @ w
    y = (rng.random(n) < _sigmoid(logits)).astype(np.int64)

    # 4. Sample observed features
    x = z + sigma_x * rng.standard_normal((n, d))

    # 5. Sample edges: upper triangle only, then symmetrize
    # Compute z_i^T z_j for all pairs efficiently
    gram = z @ z.T  # (n, n)
    edge_logits = edge_alpha * gram + edge_beta
    edge_probs = _sigmoid(edge_logits)

    # Sample upper triangle
    tri_rows, tri_cols = np.triu_indices(n, k=1)
    probs_upper = edge_probs[tri_rows, tri_cols]
    mask = rng.random(len(tri_rows)) < probs_upper
    src = tri_rows[mask]
    dst = tri_cols[mask]

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=n)

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        z=torch.tensor(z, dtype=torch.float32),
        component=torch.tensor(components, dtype=torch.long),
        num_nodes=n,
    )


def shift_weights(
    p_source: np.ndarray,
    p_shift: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Interpolate mixture weights: (1-gamma)*p_source + gamma*p_shift."""
    w = (1.0 - gamma) * np.asarray(p_source) + gamma * np.asarray(p_shift)
    return w / w.sum()


def make_source_target_pair(
    n_source: int,
    n_target: int,
    M: int,
    d_latent: int,
    gamma: float,
    sigma: float = 0.5,
    edge_alpha: float = 0.5,
    edge_beta: float = -2.0,
    seed: int = 0,
    mean_scale: float = 2.0,
) -> tuple[Data, Data, np.ndarray, np.ndarray]:
    """
    Convenience factory: makes source and target graphs with controlled covariate shift.

    Source uses uniform mixture weights. Target uses shifted weights
    (last component gets all the extra gamma mass, first component loses it).

    Returns (G_source, G_target, p_source, p_target).
    """
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((M, d_latent)) * mean_scale

    p_source = np.ones(M) / M
    # shift: move mass from component 0 to component M-1
    p_shift = np.zeros(M)
    p_shift[-1] = 1.0
    p_target = shift_weights(p_source, p_shift, gamma)

    G_source = generate_synthetic_graph(
        n=n_source,
        mixture_weights=p_source,
        means=means,
        sigma=sigma,
        edge_alpha=edge_alpha,
        edge_beta=edge_beta,
        seed=seed + 1,
    )
    G_target = generate_synthetic_graph(
        n=n_target,
        mixture_weights=p_target,
        means=means,
        sigma=sigma,
        edge_alpha=edge_alpha,
        edge_beta=edge_beta,
        seed=seed + 2,
    )
    return G_source, G_target, p_source, p_target


def make_regime(
    regime: str,
    n_source: int,
    n_target: int,
    M: int,
    d_latent: int,
    sigma: float = 0.5,
    edge_alpha: float = 0.5,
    edge_beta: float = -2.0,
    seed: int = 0,
    mean_scale: float = 2.0,
    gamma: float = 0.75,
    edge_alpha_T: float | None = None,
    edge_beta_T: float | None = None,
) -> tuple[Data, Data, np.ndarray, np.ndarray]:
    """
    Generate a source-target pair for one of four shift regimes (Experiment 2).

    Regimes
    -------
    'no_shift'          : same P(Z), same edge mechanism
    'covariate_shift'   : different P(Z), same edge mechanism
    'structural_shift'  : same P(Z), different edge mechanism
    'both'              : different P(Z) AND different edge mechanism

    For structural shift, the target uses edge_alpha_T / edge_beta_T
    (defaults: halved edge_alpha and raised edge_beta to change density+homophily).

    Returns (G_source, G_target, p_source, p_target).
    """
    assert regime in ("no_shift", "covariate_shift", "structural_shift", "both"), \
        f"Unknown regime: {regime!r}"

    rng = np.random.default_rng(seed)
    means = rng.standard_normal((M, d_latent)) * mean_scale

    p_source = np.ones(M) / M
    p_shift = np.zeros(M)
    p_shift[-1] = 1.0

    # Target mixture weights
    if regime in ("covariate_shift", "both"):
        p_target = shift_weights(p_source, p_shift, gamma)
    else:
        p_target = p_source.copy()

    # Target edge mechanism
    if edge_alpha_T is None:
        edge_alpha_T = edge_alpha * 0.25   # weaker homophily
    if edge_beta_T is None:
        edge_beta_T = edge_beta + 1.5      # denser graph

    if regime in ("structural_shift", "both"):
        tgt_alpha, tgt_beta = edge_alpha_T, edge_beta_T
    else:
        tgt_alpha, tgt_beta = edge_alpha, edge_beta

    # Fixed label weight (same for source and target — conditional P(Y|Z) shared)
    w = np.ones(d_latent) / np.sqrt(d_latent)

    G_source = generate_synthetic_graph(
        n=n_source,
        mixture_weights=p_source,
        means=means,
        sigma=sigma,
        edge_alpha=edge_alpha,
        edge_beta=edge_beta,
        seed=seed + 1,
        label_weight=w,
    )
    G_target = generate_synthetic_graph(
        n=n_target,
        mixture_weights=p_target,
        means=means,
        sigma=sigma,
        edge_alpha=tgt_alpha,
        edge_beta=tgt_beta,
        seed=seed + 2,
        label_weight=w,
    )
    return G_source, G_target, p_source, p_target


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
