"""
Concept-shift probe — NON-PRIVATE diagnostic (REPORT.md next-step #2), torch-free.

Question it answers
-------------------
The shared-frame oracle and the FGW alignment both recover almost none of the
ACM/DBLP target_oracle headroom, which *implies* concept shift (P(Y|X) differs).
This probe *measures* it directly, at the feature level, with no graph and no GNN.

Design
------
Train an X -> Y probe on SOURCE and a second on TARGET, then score BOTH on the
SAME held-out TARGET test set:

    transfer        = source-probe AUROC on target-test   (source conditional)
    within_target   = target-probe AUROC on target-test   (target ceiling)
    concept_gap     = within_target - transfer

Because both numbers are computed on identical target points, covariate shift
P(X) cancels out: the only thing that differs is which domain's P(Y|X) trained
the conditional. So concept_gap is the AUROC a target-trained conditional gains
over the source one on the same data — the component importance weighting (any
function of X) provably cannot recover. within_source (source-probe on source-
test) is a sanity ceiling.

Two probe families:
  - logistic : interpretable linear conditional. A large gap here = the linear
    feature->label map differs across domains.
  - mlp      : flexible conditional. If the logistic gap is large but the MLP gap
    is small, the apparent "concept shift" is mostly nonlinear-but-shared
    structure a linear probe can't fit (reweighting still won't help, but it is
    not true concept shift). If both gaps are large, P(Y|X) genuinely differs.

torch-free: inputs may be numpy arrays or torch tensors (converted via _to_np);
only numpy + scikit-learn are imported.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def _to_np(a) -> np.ndarray:
    """Accept a numpy array or a torch tensor without importing torch."""
    if hasattr(a, "detach"):              # torch.Tensor
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _train_test_mask(n: int, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    test = np.zeros(n, dtype=bool)
    test[rng.permutation(n)[: int(round(test_frac * n))]] = True
    return ~test, test


def _fit_probe(kind, X, y, *, seed, max_logistic_iter, mlp_hidden, standardize):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=max_logistic_iter, C=1.0)
    elif kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(mlp_hidden,), max_iter=300,
                            early_stopping=True, random_state=seed)
    else:
        raise ValueError(f"unknown probe kind: {kind}")

    steps = ([StandardScaler()] if standardize else []) + [clf]
    pipe = make_pipeline(*steps)
    pipe.fit(X, y)
    return pipe


def _full_proba(pipe, X, num_classes: int) -> np.ndarray:
    """predict_proba expanded to all `num_classes` columns (a probe may not have
    seen every class in training)."""
    p = pipe.predict_proba(X)
    full = np.zeros((X.shape[0], num_classes))
    full[:, pipe.classes_.astype(int)] = p
    return full


def _auroc(y, proba) -> float:
    C = proba.shape[1]
    try:
        if C == 2:
            return float(roc_auc_score(y, proba[:, 1]))
        return float(roc_auc_score(y, proba, multi_class="ovr", average="macro",
                                   labels=list(range(C))))
    except ValueError:
        return float("nan")


def concept_shift_probe(
    G_source,
    G_target,
    *,
    target_test_frac: float = 0.3,
    seed: int = 0,
    probes: tuple[str, ...] = ("logistic", "mlp"),
    standardize: bool = True,
    max_logistic_iter: int = 2000,
    mlp_hidden: int = 128,
    max_train: int | None = 20000,
) -> dict:
    """
    Measure the feature-level concept-shift component between two graphs.

    Parameters
    ----------
    G_source, G_target : objects exposing `.x` (n,d features) and `.y` (n,) labels
                         (PyG Data, or any object with those attributes).
    target_test_frac   : fraction of each domain held out for testing.
    max_train          : cap on training rows per probe (subsampled) for speed on
                         large graphs; None disables.

    Returns
    -------
    dict with per-probe {within_source, transfer, within_target, concept_gap}
    in both AUROC and accuracy, plus sizes.
    """
    Xs, ys = _to_np(G_source.x), _to_np(G_source.y).astype(int).ravel()
    Xt, yt = _to_np(G_target.x), _to_np(G_target.y).astype(int).ravel()
    C = int(max(ys.max(), yt.max())) + 1

    s_tr, s_te = _train_test_mask(len(ys), target_test_frac, seed)
    t_tr, t_te = _train_test_mask(len(yt), target_test_frac, seed)

    def _train_idx(mask):
        idx = np.where(mask)[0]
        if max_train and len(idx) > max_train:
            idx = np.random.default_rng(seed).choice(idx, max_train, replace=False)
        return idx

    si, ti = _train_idx(s_tr), _train_idx(t_tr)
    pkw = dict(seed=seed, max_logistic_iter=max_logistic_iter,
               mlp_hidden=mlp_hidden, standardize=standardize)

    out = {"num_classes": C, "n_source_train": int(len(si)),
           "n_target_train": int(len(ti)), "n_target_test": int(t_te.sum())}

    for kind in probes:
        src = _fit_probe(kind, Xs[si], ys[si], **pkw)
        tgt = _fit_probe(kind, Xt[ti], yt[ti], **pkw)

        within_source = _auroc(ys[s_te], _full_proba(src, Xs[s_te], C))
        transfer      = _auroc(yt[t_te], _full_proba(src, Xt[t_te], C))
        within_target = _auroc(yt[t_te], _full_proba(tgt, Xt[t_te], C))

        ws_acc = float(accuracy_score(ys[s_te], src.predict(Xs[s_te])))
        tr_acc = float(accuracy_score(yt[t_te], src.predict(Xt[t_te])))
        wt_acc = float(accuracy_score(yt[t_te], tgt.predict(Xt[t_te])))

        out[kind] = {
            "within_source_auroc": within_source,
            "transfer_auroc": transfer,
            "within_target_auroc": within_target,
            "concept_gap_auroc": within_target - transfer,
            "within_source_acc": ws_acc,
            "transfer_acc": tr_acc,
            "within_target_acc": wt_acc,
            "concept_gap_acc": wt_acc - tr_acc,
        }
    return out


def print_concept_shift(res: dict, gnn_source_auroc: float | None = None,
                        gnn_target_auroc: float | None = None) -> None:
    """Pretty-print the probe table and an interpretation."""
    print(f"Concept-shift probe (features only, no graph) — {res['num_classes']} classes")
    print(f"  n: source-train {res['n_source_train']}, target-train {res['n_target_train']}, "
          f"target-test {res['n_target_test']}")
    print(f"  {'probe':<10}{'within_src':>11}{'transfer':>10}{'within_tgt':>11}{'concept_gap':>13}")
    for kind in ("logistic", "mlp"):
        if kind not in res:
            continue
        r = res[kind]
        print(f"  {kind:<10}{r['within_source_auroc']:>11.4f}{r['transfer_auroc']:>10.4f}"
              f"{r['within_target_auroc']:>11.4f}{r['concept_gap_auroc']:>13.4f}")
    print("  (AUROC; concept_gap = within_target - transfer = irreducible-by-reweighting)")

    lg = res.get("logistic", {}).get("concept_gap_auroc")
    ml = res.get("mlp", {}).get("concept_gap_auroc")
    print()
    if lg is not None and ml is not None:
        big = 0.03
        if lg <= big and ml <= big:
            print("  -> both probes transfer: feature P(Y|X) is shared. Any residual GNN")
            print("     headroom is STRUCTURAL / optimization, not feature concept shift.")
        elif lg > big and ml <= big:
            print("  -> linear gap but no MLP gap: apparent shift is nonlinear-but-SHARED")
            print("     structure a linear map can't fit — not true concept shift (a flexible")
            print("     learner transfers). Reweighting still won't help.")
        else:
            print("  -> both probes show a gap: genuine CONCEPT shift, P(Y|X) differs across")
            print("     domains. Importance weighting (any function of X) provably cannot fix it.")

    if gnn_source_auroc is not None and gnn_target_auroc is not None:
        headroom = gnn_target_auroc - gnn_source_auroc
        print(f"\n  GNN headroom (target_oracle - source_only) = {headroom:+.4f}")
        print(f"  feature concept-gap (mlp) = {ml:+.4f} -- compare magnitudes: a concept-gap")
        print(f"  near 0 with positive headroom localizes the gap to graph STRUCTURE.")
