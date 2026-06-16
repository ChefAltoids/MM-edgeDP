"""
Grid sweep runner for DP-OT experiments.

Produces a CSV with one row per (method, seed, epsilon, gamma, K) combination.
Supports resume: already-completed rows are skipped.

Usage:
  python dp_ot/sweep.py --config dp_ot/configs/synthetic_covariate_shift.yaml
  python dp_ot/sweep.py --config dp_ot/configs/synthetic_covariate_shift.yaml \\
      --out dp_ot/outputs/sweep.csv --set n_source=2000
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from dp_ot.run_experiment import run_experiment, DEFAULT_CONFIG

CSV_COLUMNS = [
    "method", "seed", "epsilon", "gamma", "K",
    "auroc", "acc", "f1", "proto_l1_error", "adaptation_gain",
]


def expand_grid(sweep_dict: dict) -> list[dict]:
    """Cartesian product of all sweep axes."""
    keys = list(sweep_dict.keys())
    values = [v if isinstance(v, list) else [v] for v in sweep_dict.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def load_done_keys(csv_path: str) -> set[tuple]:
    done = set()
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["method"], row["seed"], row["epsilon"], row["gamma"], row["K"])
                done.add(key)
    except FileNotFoundError:
        pass
    return done


def append_rows(csv_path: str, rows: list[dict]) -> None:
    path = Path(csv_path)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _load_config(path: str) -> dict:
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f) or {}
        return json.load(f)


def run_sweep(
    config: dict,
    out_path: str | None = None,
    progress: bool = True,
):
    """
    Run a grid sweep from a single config dict (notebook-friendly entry point).

    The config may contain a ``sweep:`` block listing the axes to expand
    (e.g. ``{"sweep": {"epsilon": [...], "seed": [...]}}``); every other key is
    treated as a fixed base setting. Results are appended to ``out_path`` as CSV
    with resume support, and also returned as a pandas DataFrame.

    Parameters
    ----------
    config    : merged base + ``sweep`` dict (e.g. a loaded YAML config).
    out_path  : CSV destination. Falls back to config['out'] or a default.
    progress  : if True, print per-combination progress.

    Returns
    -------
    pandas.DataFrame of all rows in ``out_path`` after the sweep.
    """
    import pandas as pd

    full_cfg = dict(config)
    sweep_axes: dict = full_cfg.pop("sweep", {})
    base_cfg: dict = full_cfg

    out_path = out_path or base_cfg.pop("out", "dp_ot/outputs/sweep.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    combos = expand_grid(sweep_axes)
    done_keys = load_done_keys(out_path)
    if progress:
        print(f"Sweep: {len(combos)} combinations × methods, output → {out_path}")

    for i, combo in enumerate(combos):
        cfg = {**DEFAULT_CONFIG, **base_cfg, **combo}
        epsilon = cfg.get("epsilon", 1.0)
        gamma = cfg.get("gamma", 0.5)
        K = cfg.get("K", 32)
        seed = cfg.get("seed", 0)

        # Skip a combination entirely if every method row already exists.
        combo_methods = ("source_only", "oracle", "dp_histogram",
                         "dp_exponential", "target_oracle")
        if all((m, str(seed), str(epsilon), str(gamma), str(K)) in done_keys
               for m in combo_methods):
            if progress:
                print(f"[{i+1}/{len(combos)}] eps={epsilon} gamma={gamma} "
                      f"K={K} seed={seed} — cached, skipping")
            continue

        if progress:
            print(f"[{i+1}/{len(combos)}] eps={epsilon} gamma={gamma} K={K} seed={seed}")
        t0 = time.time()

        results = run_experiment(cfg)
        elapsed = time.time() - t0
        if progress:
            print(f"  done in {elapsed:.1f}s")

        source_auroc = results["source_only"]["auroc"]

        rows = []
        for method, metrics in results.items():
            key = (method, str(seed), str(epsilon), str(gamma), str(K))
            if key in done_keys:
                continue
            adapt_gain = metrics["auroc"] - source_auroc
            rows.append({
                "method": method,
                "seed": seed,
                "epsilon": epsilon,
                "gamma": gamma,
                "K": K,
                "auroc": round(metrics["auroc"], 6),
                "acc": round(metrics["acc"], 6),
                "f1": round(metrics["f1"], 6),
                "proto_l1_error": round(metrics.get("proto_l1_error", float("nan")), 6),
                "adaptation_gain": round(adapt_gain, 6),
            })
            done_keys.add(key)

        if rows:
            append_rows(out_path, rows)

    if progress:
        print(f"\nSweep complete. Results at {out_path}")
    return pd.read_csv(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="DP-OT grid sweep.")
    parser.add_argument("--config", required=True, help="YAML/JSON sweep config.")
    parser.add_argument("--out", default=None, help="Output CSV path.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    full_cfg = _load_config(args.config)

    # CLI overrides
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        full_cfg[k] = v

    run_sweep(full_cfg, out_path=args.out)


if __name__ == "__main__":
    main()
