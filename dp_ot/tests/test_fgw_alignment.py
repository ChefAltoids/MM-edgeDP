"""
Positive-control validation for the FGW alignment diagnostic (#1) — torch-free.

The scientific claim we must be able to trust before reading any ACM/DBLP result:
  When the same latent roles exist in source and target but the target frame is
  ROTATED, the shared-frame assignment (target -> nearest SOURCE centroid) breaks,
  but isometry-invariant FGW recovers the correct correspondence.

If this control passes, a NULL result on a shared-vocabulary benchmark means
"no misalignment" (-> concept shift). If it failed, a null would be uninterpretable
(maybe FGW is just underpowered). That is the whole point of the control.

Run:  .venv_fgw/bin/python dp_ot/tests/test_fgw_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dp_ot.adapt.fgw_align import (
    fit_target_prototypes,
    fgw_couple,
    align_target_mass,
    fgw_aligned_target_mass,
    coupling_diagnostics,
)
from dp_ot.adapt.dp_histogram import nonprivate_histogram


def _make_mixture(means, mass, n, sigma, rng):
    """Sample n points from a Gaussian mixture with given component means/mass."""
    comp = rng.choice(len(mass), size=n, p=mass)
    x = means[comp] + sigma * rng.standard_normal((n, means.shape[1]))
    return x.astype(np.float64)


def _l1(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def _scenario(source_mass, target_mass, sigma=0.30, n=4000, seed=0):
    """Build a source/target mixture, rotate the target frame, and compare the
    shared-frame (naive) vs FGW-aligned target-mass recovery. Returns a dict."""
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    M, d, K = len(source_mass), 8, len(source_mass)

    # Asymmetric, well-separated means -> a UNIQUE structural fingerprint.
    means = rng.standard_normal((M, d)) * np.array([2.0, 4.0, 7.0, 11.0])[:, None]

    Xs = _make_mixture(means, source_mass, n, sigma, rng)
    Xt = _make_mixture(means, target_mass, n, sigma, rng)

    km_s = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(Xs)
    centroids_s = km_s.cluster_centers_
    alpha_source = np.bincount(km_s.labels_, minlength=K) / len(Xs)

    # GROUND TRUTH: unrotated shared-frame histogram of target onto source
    # centroids. With matched frames this is exactly what the oracle would use.
    gt = nonprivate_histogram(Xt, centroids_s)

    # Rotate the target frame by a random orthogonal Q (an isometry).
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Xt_rot = Xt @ Q

    naive_rot = nonprivate_histogram(Xt_rot, centroids_s)      # should break
    centroids_t, alpha_target = fit_target_prototypes(Xt_rot, K, seed=0)

    # (a) coupled with TRUE marginals: a large mass shift can corrupt the match.
    T, fgw_cost = fgw_couple(
        centroids_s, centroids_t, alpha_source, alpha_target,
        fusion_alpha=1.0, n_init=10, seed=0,                   # pure structure
    )
    fgw_rot = align_target_mass(T, alpha_source, alpha_target)

    # (b) decoupled (recommended): correspondence via uniform marginals, then
    # push the true target mass through it.
    fgw_dec, _Td, _ = fgw_aligned_target_mass(
        centroids_s, centroids_t, alpha_target,
        fusion_alpha=1.0, n_init=10, seed=0,
    )
    return dict(gt=gt, naive=naive_rot, fgw=fgw_rot, fgw_dec=fgw_dec, fgw_cost=fgw_cost,
                diag=coupling_diagnostics(T),
                err_naive=_l1(naive_rot, gt), err_fgw=_l1(fgw_rot, gt),
                err_dec=_l1(fgw_dec, gt))


def _report(title, r):
    print(f"\n=== {title} ===")
    print(f"ground-truth aligned mass : {np.round(r['gt'], 3)}")
    print(f"naive  (rotated)          : {np.round(r['naive'], 3)}  L1={r['err_naive']:.3f}")
    print(f"FGW    true-marginals     : {np.round(r['fgw'], 3)}  L1={r['err_fgw']:.3f}")
    print(f"FGW    decoupled (rec.)   : {np.round(r['fgw_dec'], 3)}  L1={r['err_dec']:.3f}")
    print(f"FGW cost={r['fgw_cost']:.4f}  diag={ {k: round(v,3) for k,v in r['diag'].items()} }")


def main() -> int:
    # Scenario 1 (GATES): equal masses -> isolates isometry recovery from the
    # mass/geometry tension. FGW must recover gt where the naive frame breaks.
    r1 = _scenario(source_mass=np.array([0.40, 0.30, 0.20, 0.10]),
                   target_mass=np.array([0.40, 0.30, 0.20, 0.10]))
    _report("Scenario 1 — equal masses (positive control, GATES)", r1)
    broke = r1["err_naive"] > 0.3
    recovered = r1["err_fgw"] < 0.5 * r1["err_naive"] and r1["err_fgw"] < 0.2
    print(f"  rotation broke shared-frame: {broke} | FGW recovered: {recovered}")

    # Scenario 2 (INFORMATIVE): a real mass shift. Documents whether a large
    # prototype-mass shift pulls the coupling away from the geometric match
    # (a genuine caveat for using FGW as the reweighting mechanism, not a gate).
    r2 = _scenario(source_mass=np.array([0.40, 0.30, 0.20, 0.10]),
                   target_mass=np.array([0.10, 0.20, 0.30, 0.40]))
    _report("Scenario 2 — reversed mass shift (tension + decoupled fix)", r2)
    tension = r2["err_fgw"] > 0.3        # true-marginal coupling corrupted by shift
    dec_fixes = r2["err_dec"] < 0.1      # decoupled correspondence recovers it
    print(f"  mass shift corrupted true-marginal match: {tension} (L1={r2['err_fgw']:.3f})")
    print(f"  decoupled correspondence recovers it     : {dec_fixes} (L1={r2['err_dec']:.3f})")

    ok = broke and recovered and dec_fixes
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
