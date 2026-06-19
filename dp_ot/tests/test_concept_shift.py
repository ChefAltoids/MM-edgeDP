"""
Validation for the concept-shift probe (REPORT.md next-step #2) — torch-free.

Two controls that must both hold for the probe to be trustworthy:
  - POSITIVE: target P(Y|X) is the source one with the label rule FLIPPED -> the
    probe must report a large concept_gap (target-trained beats transferred).
  - NEGATIVE: source and target share P(Y|X) but P(X) differs (covariate shift
    only) -> concept_gap must be ~0 (the source conditional transfers).

Run:  .venv_fgw/bin/python dp_ot/tests/test_concept_shift.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dp_ot.eval.diagnostics import concept_shift_probe, print_concept_shift


class _G:
    """Minimal stand-in for a PyG Data object (just needs .x and .y)."""
    def __init__(self, x, y):
        self.x = x
        self.y = y


def _make(n, d, rng, w, mean=0.0, flip=False):
    X = rng.standard_normal((n, d)) + mean
    logits = X @ w
    y = (logits > 0).astype(int)
    if flip:
        y = 1 - y
    return _G(X.astype(np.float64), y)


def main() -> int:
    rng = np.random.default_rng(0)
    d = 20
    w = rng.standard_normal(d)
    kw = dict(target_test_frac=0.3, seed=0, standardize=True, mlp_hidden=64)

    # POSITIVE: same X-distribution, FLIPPED conditional -> strong concept shift.
    Gs = _make(4000, d, rng, w, mean=0.0, flip=False)
    Gt = _make(4000, d, rng, w, mean=0.0, flip=True)
    pos = concept_shift_probe(Gs, Gt, **kw)
    print("=== POSITIVE control: flipped P(Y|X) (expect large gap) ===")
    print_concept_shift(pos)

    # NEGATIVE: shared conditional, only P(X) shifts (mean offset) -> ~0 gap.
    Gs2 = _make(4000, d, rng, w, mean=0.0, flip=False)
    Gt2 = _make(4000, d, rng, w, mean=0.6, flip=False)
    neg = concept_shift_probe(Gs2, Gt2, **kw)
    print("\n=== NEGATIVE control: covariate-only shift (expect ~0 gap) ===")
    print_concept_shift(neg)

    # Gates: positive must show a clear gap on both probes; negative must not.
    pos_ok = pos["logistic"]["concept_gap_auroc"] > 0.2 and pos["mlp"]["concept_gap_auroc"] > 0.2
    neg_ok = abs(neg["logistic"]["concept_gap_auroc"]) < 0.05 and abs(neg["mlp"]["concept_gap_auroc"]) < 0.05
    print(f"\npositive shows concept gap : {pos_ok} "
          f"(logistic {pos['logistic']['concept_gap_auroc']:.3f}, mlp {pos['mlp']['concept_gap_auroc']:.3f})")
    print(f"negative shows ~no gap     : {neg_ok} "
          f"(logistic {neg['logistic']['concept_gap_auroc']:.3f}, mlp {neg['mlp']['concept_gap_auroc']:.3f})")
    ok = pos_ok and neg_ok
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
