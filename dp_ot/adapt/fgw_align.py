"""
Fused Gromov-Wasserstein (FGW) prototype alignment — NON-PRIVATE diagnostic (#1).

Purpose
-------
The default pipeline assumes source and target nodes live in the *same*
coordinate frame: target nodes are hard-assigned to prototypes fit on the
SOURCE summaries. If the same latent graph-roles exist in both domains but are
*misindexed* (rotated / relabelled / scaled differently), that shared-frame
assumption silently fails, and the recoverable-gap diagnostic
(`oracle - source_only`) reports "no gain" — indistinguishable from genuine
concept shift.

FGW removes the shared-frame assumption. It fits target prototypes in their
OWN frame, then couples source<->target prototypes using BOTH:
  - a feature term  (cross-domain centroid distance)            weight (1 - alpha)
  - a structure term (intra-domain relational geometry, GW)     weight  alpha
The GW structure term is invariant to isometry/relabelling, so FGW can match
roles even when coordinates differ.

This module is deliberately torch-free (numpy + sklearn + POT) so the alignment
machinery can be validated independently of the GNN. It is a DIAGNOSTIC: it
spends no privacy budget and is not a DP mechanism. It answers the scientific
question "is the residual gap misalignment or concept shift?" before any DP
cost is paid (see REPORT.md §8 and the FGW discussion).

Convention (POT): `fused_gromov_wasserstein(M, C1, C2, p, q, alpha=a)` minimises
    (1 - a) * <M, T>  +  a * GW(C1, C2, T)
so alpha=0 -> pure Wasserstein (feature), alpha=1 -> pure Gromov (structure).
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def fit_target_prototypes(
    target_summaries: np.ndarray,
    K: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster the TARGET summaries into K prototypes in the target's own frame.

    This is the key departure from the shared-frame pipeline, which assigns
    target nodes to SOURCE centroids. Here the target gets independent prototypes
    so FGW can discover the source<->target correspondence rather than assume it.

    Returns
    -------
    centroids_t  : (K, d) target prototype centroids
    alpha_target : (K,) target prototype mass (fraction of target nodes per proto)
    """
    km = KMeans(n_clusters=K, random_state=seed, n_init="auto")
    assignments = km.fit_predict(target_summaries)
    counts = np.bincount(assignments, minlength=K).astype(float)
    alpha_target = counts / counts.sum()
    return km.cluster_centers_, alpha_target


def _random_coupling(p, q, rng, n_iter=50):
    """A random valid coupling with marginals (p, q), via iterative proportional
    fitting on a random positive seed matrix."""
    G = rng.random((len(p), len(q))) + 1e-3
    for _ in range(n_iter):
        G *= (p / (G.sum(1) + 1e-12))[:, None]
        G *= (q / (G.sum(0) + 1e-12))[None, :]
    return G


def _pairwise_l2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix between rows of A (m,d) and B (n,d)."""
    aa = (A ** 2).sum(1, keepdims=True)
    bb = (B ** 2).sum(1, keepdims=True).T
    d2 = np.maximum(0.0, aa - 2.0 * A @ B.T + bb)
    return np.sqrt(d2)


def fgw_couple(
    centroids_s: np.ndarray,
    centroids_t: np.ndarray,
    alpha_source: np.ndarray,
    alpha_target: np.ndarray,
    fusion_alpha: float = 0.5,
    n_init: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """
    Fused Gromov-Wasserstein coupling between source and target prototypes.

    Feature cost = cross-domain centroid L2 (meaningful only when the two frames
    share features; for heterogeneous frames push `fusion_alpha` toward 1 so the
    structure term dominates). Structure cost = intra-domain centroid L2 (GW).

    FGW is a non-convex QAP whose conditional-gradient solver can get stuck, so we
    run `n_init` random initialisations and keep the lowest-cost coupling.

    Returns
    -------
    T          : (K_s, K_t) optimal coupling (row sums = alpha_source,
                 col sums = alpha_target)
    fgw_cost   : achieved FGW objective value (lower = better alignment)
    """
    import ot

    M = _pairwise_l2(centroids_s, centroids_t)           # feature cost
    M = M / (M.max() + 1e-12)                            # scale-normalise terms
    Cs = _pairwise_l2(centroids_s, centroids_s)
    Ct = _pairwise_l2(centroids_t, centroids_t)
    Cs = Cs / (Cs.max() + 1e-12)
    Ct = Ct / (Ct.max() + 1e-12)

    p = np.asarray(alpha_source, dtype=np.float64)
    q = np.asarray(alpha_target, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()

    rng = np.random.default_rng(seed)
    best_T, best_cost = None, np.inf
    for init_idx in range(max(1, n_init)):
        # POT requires G0 to be a valid coupling (row sums = p, col sums = q).
        # init 0 uses POT's deterministic product init; the rest are random valid
        # couplings (IPF-projected) to escape FGW's non-convex local minima.
        G0 = None if init_idx == 0 else _random_coupling(p, q, rng)
        try:
            T, log = ot.gromov.fused_gromov_wasserstein(
                M, Cs, Ct, p, q,
                loss_fun="square_loss",
                alpha=fusion_alpha,
                G0=G0,
                log=True,
            )
        except TypeError:
            # Older/newer POT may not accept G0; fall back to default init.
            T, log = ot.gromov.fused_gromov_wasserstein(
                M, Cs, Ct, p, q, loss_fun="square_loss", alpha=fusion_alpha, log=True,
            )
        cost = float(log.get("fgw_dist", log.get("loss", [np.nan])[-1]
                             if isinstance(log.get("loss"), (list, np.ndarray)) else np.nan))
        if not np.isfinite(cost):
            cost = float(np.sum(M * T))
        if cost < best_cost:
            best_T, best_cost = T, cost
    return best_T, best_cost


def align_target_mass(
    T: np.ndarray,
    alpha_source: np.ndarray,
    alpha_target: np.ndarray,
) -> np.ndarray:
    """
    Pull the target prototype mass back onto the SOURCE prototype index set
    through the FGW coupling, so it can drive the existing reweighting.

    For source prototype i, let R[i, .] = T[i, .] / p[i] be the conditional
    distribution over target prototypes it maps to. The target mass attributed
    to source prototype i is the expected target mass of its image:

        alpha_aligned[i] = sum_j (T[i, j] / p[i]) * q[j]

    In the perfect-isometry limit (T a mass-scaled permutation, T[i, sigma(i)] =
    p[i] = q[sigma(i)]), this reduces to alpha_aligned[i] = q[sigma(i)] — exactly
    the matched target prototype's mass, which is what a frame-aligned oracle
    should have used. Result is renormalised to a distribution.

    Returns
    -------
    alpha_aligned : (K_s,) normalised target-mass estimate indexed by SOURCE proto
    """
    p = np.asarray(alpha_source, dtype=np.float64)
    q = np.asarray(alpha_target, dtype=np.float64)
    p_safe = np.where(p > 0, p, 1.0)
    R = T / p_safe[:, None]                      # (K_s, K_t) rows ~ conditional
    alpha_aligned = R @ q                        # (K_s,)
    total = alpha_aligned.sum()
    if total <= 0:
        return np.full_like(p, 1.0 / len(p))
    return (alpha_aligned / total).astype(np.float32)


def fgw_aligned_target_mass(
    centroids_s: np.ndarray,
    centroids_t: np.ndarray,
    alpha_target: np.ndarray,
    fusion_alpha: float = 1.0,
    n_init: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Recommended end-to-end alignment: decouple *correspondence* from *mass*.

    FGW's marginal constraints are hard, so estimating the coupling with the true
    (shifted) target masses lets a large mass shift pull the match toward
    mass-matching rather than geometry (validated in tests/test_fgw_alignment.py,
    scenario 2). We therefore estimate the source<->target correspondence with
    UNIFORM marginals (geometry drives the match), then push the TRUE target
    prototype masses through that fixed correspondence. On the worst-case reversed
    shift this recovers the ground-truth aligned mass exactly.

    Returns
    -------
    alpha_aligned : (K_s,) target mass indexed by source prototype (normalised)
    T             : (K_s, K_t) the geometric correspondence coupling
    fgw_cost      : achieved FGW cost of the correspondence
    """
    K_s, K_t = len(centroids_s), len(centroids_t)
    u_s = np.full(K_s, 1.0 / K_s)
    u_t = np.full(K_t, 1.0 / K_t)
    T, cost = fgw_couple(centroids_s, centroids_t, u_s, u_t,
                         fusion_alpha=fusion_alpha, n_init=n_init, seed=seed)
    alpha_aligned = align_target_mass(T, u_s, np.asarray(alpha_target))
    return alpha_aligned, T, cost


def coupling_diagnostics(T: np.ndarray) -> dict[str, float]:
    """Cheap, torch-free descriptors of how concentrated/permutation-like the
    coupling is (a diffuse coupling means FGW found no clear correspondence)."""
    Tn = T / (T.sum() + 1e-12)
    row = Tn.sum(1, keepdims=True)
    cond = Tn / (row + 1e-12)                    # P(target | source)
    ent = -np.sum(cond * np.log(cond + 1e-12), axis=1)   # per-source entropy
    max_match = cond.max(axis=1)                 # peak mass of each row
    return {
        "coupling_mean_row_entropy": float(ent.mean()),
        "coupling_mean_peak": float(max_match.mean()),
    }
