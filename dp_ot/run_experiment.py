"""
Full DP-OT pipeline: source GNN training → prototype construction →
DP target-mass estimation → importance-reweighted retraining → evaluation.

Usage:
  python dp_ot/run_experiment.py --config dp_ot/configs/synthetic_covariate_shift.yaml
  python dp_ot/run_experiment.py --config dp_ot/configs/synthetic_covariate_shift.yaml \\
      --set gamma=0.75 epsilon=1.0 K=32 seed=0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# Make dp_ot importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dp_ot.data.synthetic import make_source_target_pair, make_regime
from dp_ot.data.real_splits import load_twitch_pair, load_ogb_arxiv_temporal
from dp_ot.models.gnn import train_source_gnn, train_weighted_source_gnn
from dp_ot.adapt.prototypes import embed_nodes, fit_public_prototypes, compute_target_summaries
from dp_ot.adapt.dp_histogram import dp_histogram_assign, nonprivate_histogram
from dp_ot.adapt.dp_exponential import dp_exponential_assign
from dp_ot.adapt.reweighting import reweight_source_nodes, uniform_weights
from dp_ot.eval.metrics import evaluate, prototype_l1_error


DEFAULT_CONFIG = {
    # Data source: "synthetic" | "twitch" | "ogb_arxiv"
    "dataset": "synthetic",
    # Synthetic graph generation
    "n_source": 1000,
    "n_target": 500,
    "M": 4,           # number of mixture components
    "d_latent": 8,    # latent dimension = feature dimension
    "gamma": 0.5,     # covariate shift magnitude (synthetic only)
    "sigma": 0.5,
    "edge_alpha": 0.5,
    "edge_beta": -2.0,
    "mean_scale": 2.0,
    "seed": 0,
    # Regime for Experiment 2: "covariate_shift" | "structural_shift" | "both" | "no_shift"
    # When set, overrides gamma-based make_source_target_pair with make_regime.
    "regime": None,
    # Twitch-specific
    "twitch_source": "EN",
    "twitch_target": "DE",
    "twitch_root": "data/twitch",
    # OGB-arxiv-specific
    "ogb_source_before_year": 2018,
    "ogb_target_from_year": 2018,
    "ogb_root": "data/ogb",
    # Prototype construction
    "K": 32,
    "d_max": 10,      # degree cap for target summaries
    "B": 3.0,         # feature clip bound
    # Privacy
    "epsilon": 1.0,
    # GNN training
    "hidden": 64,
    "num_layers": 2,
    "epochs": 200,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "dropout": 0.5,
    "reweight_rho": 1e-3,
    "device": "cpu",
}


def _load_graphs(cfg: dict) -> tuple:
    """Return (G_source, G_target) based on cfg['dataset']."""
    dataset = cfg.get("dataset", "synthetic")

    if dataset == "twitch":
        return load_twitch_pair(
            source_lang=cfg["twitch_source"],
            target_lang=cfg["twitch_target"],
            root=cfg["twitch_root"],
        )

    if dataset == "ogb_arxiv":
        return load_ogb_arxiv_temporal(
            source_before_year=cfg["ogb_source_before_year"],
            target_from_year=cfg["ogb_target_from_year"],
            root=cfg["ogb_root"],
        )

    # Default: synthetic
    seed = cfg["seed"]
    regime = cfg.get("regime")
    syn_kwargs = dict(
        n_source=cfg["n_source"],
        n_target=cfg["n_target"],
        M=cfg["M"],
        d_latent=cfg["d_latent"],
        sigma=cfg["sigma"],
        edge_alpha=cfg["edge_alpha"],
        edge_beta=cfg["edge_beta"],
        seed=seed,
        mean_scale=cfg["mean_scale"],
    )

    if regime is not None:
        G_source, G_target, _, _ = make_regime(regime=regime, gamma=cfg["gamma"], **syn_kwargs)
    else:
        G_source, G_target, _, _ = make_source_target_pair(gamma=cfg["gamma"], **syn_kwargs)

    return G_source, G_target


def run_experiment(config: dict) -> dict[str, dict]:
    """
    Run all methods and return a dict of {method_name: metrics_dict}.

    Methods:
      source_only     - train on source, evaluate on target (no adaptation)
      oracle          - nonprivate target prototype mass → reweighted training
      dp_histogram    - Mechanism A (Laplace noise on counts)
      dp_exponential  - Mechanism B (exponential mechanism)
      target_oracle   - train directly on target graph (upper bound)
    """
    cfg = {**DEFAULT_CONFIG, **config}
    seed = cfg["seed"]

    # --- Load / generate graphs ---
    G_source, G_target = _load_graphs(cfg)

    gnn_kwargs = dict(
        hidden=cfg["hidden"],
        num_layers=cfg["num_layers"],
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        dropout=cfg["dropout"],
        device=cfg["device"],
    )

    # --- Train source GNN (public, zero privacy cost) ---
    print("  Training source GNN...", flush=True)
    source_model = train_source_gnn(G_source, seed=seed, **gnn_kwargs)

    # --- Build public prototypes ---
    print("  Building prototypes...", flush=True)
    Z_source = embed_nodes(source_model, G_source, device=cfg["device"])
    K = cfg["K"]
    centroids, source_assignments, alpha_source = fit_public_prototypes(Z_source, K, seed=seed)

    # --- Compute target summaries (DP-accessible representation of G_T) ---
    print("  Computing target summaries...", flush=True)
    summaries = compute_target_summaries(G_target, d_max=cfg["d_max"], B=cfg["B"])

    # True (non-private) target prototype mass for oracle and error tracking
    alpha_true = nonprivate_histogram(summaries, centroids)

    results = {}

    # === Method 1: Source-only ===
    print("  source_only...", flush=True)
    w = uniform_weights(G_source.num_nodes)
    model_so = train_weighted_source_gnn(G_source, w, seed=seed, **gnn_kwargs)
    metrics_so = evaluate(model_so, G_target, device=cfg["device"])
    metrics_so["proto_l1_error"] = float("nan")
    results["source_only"] = metrics_so

    # === Method 2: Oracle (nonprivate target mass) ===
    print("  oracle...", flush=True)
    w_oracle = reweight_source_nodes(source_assignments, alpha_source, alpha_true, rho=cfg["reweight_rho"])
    model_oracle = train_weighted_source_gnn(G_source, w_oracle, seed=seed, **gnn_kwargs)
    metrics_oracle = evaluate(model_oracle, G_target, device=cfg["device"])
    metrics_oracle["proto_l1_error"] = 0.0
    results["oracle"] = metrics_oracle

    # === Method 3: DP Histogram ===
    print("  dp_histogram...", flush=True)
    alpha_hist = dp_histogram_assign(summaries, centroids, epsilon=cfg["epsilon"], seed=seed)
    w_hist = reweight_source_nodes(source_assignments, alpha_source, alpha_hist, rho=cfg["reweight_rho"])
    model_hist = train_weighted_source_gnn(G_source, w_hist, seed=seed, **gnn_kwargs)
    metrics_hist = evaluate(model_hist, G_target, device=cfg["device"])
    metrics_hist["proto_l1_error"] = prototype_l1_error(alpha_hist, alpha_true)
    results["dp_histogram"] = metrics_hist

    # === Method 4: DP Exponential ===
    print("  dp_exponential...", flush=True)
    alpha_exp = dp_exponential_assign(
        summaries, centroids,
        epsilon=cfg["epsilon"],
        B=cfg["B"],
        d_max=cfg["d_max"],
        seed=seed,
    )
    w_exp = reweight_source_nodes(source_assignments, alpha_source, alpha_exp, rho=cfg["reweight_rho"])
    model_exp = train_weighted_source_gnn(G_source, w_exp, seed=seed, **gnn_kwargs)
    metrics_exp = evaluate(model_exp, G_target, device=cfg["device"])
    metrics_exp["proto_l1_error"] = prototype_l1_error(alpha_exp, alpha_true)
    results["dp_exponential"] = metrics_exp

    # === Method 5: Target oracle (train on G_T directly, upper bound) ===
    print("  target_oracle...", flush=True)
    model_tgt = train_source_gnn(G_target, seed=seed, **gnn_kwargs)
    metrics_tgt = evaluate(model_tgt, G_target, device=cfg["device"])
    metrics_tgt["proto_l1_error"] = float("nan")
    results["target_oracle"] = metrics_tgt

    return results


def _print_results(results: dict[str, dict]) -> None:
    header = f"{'method':<20} {'auroc':>8} {'acc':>8} {'f1':>8} {'proto_l1':>10}"
    print(header)
    print("-" * len(header))
    for method, m in results.items():
        l1 = f"{m['proto_l1_error']:.4f}" if not np.isnan(m.get("proto_l1_error", float("nan"))) else "   —"
        print(f"{method:<20} {m['auroc']:>8.4f} {m['acc']:>8.4f} {m['f1']:>8.4f} {l1:>10}")


def _load_config(path: str) -> dict:
    with open(path) as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.safe_load(f) or {}
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one DP-OT experiment.")
    parser.add_argument("--config", default=None, help="Path to YAML/JSON config file.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config values, e.g. --set gamma=0.5 epsilon=1.0")
    args = parser.parse_args()

    cfg: dict = {}
    if args.config:
        cfg = _load_config(args.config)

    for kv in args.set:
        k, v = kv.split("=", 1)
        # Try to parse as float, then int, then leave as string
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        cfg[k] = v

    print(f"Config: {cfg}")
    results = run_experiment(cfg)
    print()
    _print_results(results)


if __name__ == "__main__":
    main()
