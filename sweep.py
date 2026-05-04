"""
sweep.py — run a grid of MM-EdgeDP experiments and save results to CSV.

Grids are defined in JSON files under configs/:
  configs/sweep_phase1.json   public_frac x dict_per_class
  configs/sweep_phase2.json   epsilon sweep

Phase 3 (dataset comparison + attacks) is handled by baselines.py.

Usage examples:

  # Run a phase by number (loads configs/sweep_phase{N}.json)
  python sweep.py --phase 1
  python sweep.py --phase 2

  # Run any custom JSON config
  python sweep.py --config configs/sweep_phase2.json

  # Override any grid value or base config key
  python sweep.py --phase 1 --set dict_per_class=16 --set epochs=5

  # Dry run — print all configs without running
  python sweep.py --phase 1 --dry-run

  # Resume — skip rows already written to the output CSV
  python sweep.py --phase 1 --resume
"""

import argparse
import csv
import json
import sys
import traceback
from itertools import product
from pathlib import Path

# ---------------------------------------------------------------------------
# Default phase → config file mapping
# ---------------------------------------------------------------------------

PHASE_CONFIGS = {
    1: 'configs/sweep_phase1.json',
    2: 'configs/sweep_phase2.json',
}

PHASE_OUTPUTS = {
    1: 'outputs/sweep_phase1.csv',
    2: 'outputs/sweep_phase2.csv',
}

OUTPUT_CSV = Path('outputs/sweep_results.csv')  # fallback for --config

RESULT_KEYS = [
    'dataset', 'epsilon', 'public_frac', 'dict_per_class', 'seed',
    'val_acc', 'majority_acc', 'dict_size',
    'n_train', 'n_val', 'n_public',
    'mean_entropy', 'mean_top_prob', 'unique_proto_ratio',
    'status', 'error',
]


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_sweep_config(path: str) -> tuple[dict, dict]:
    """
    Load a sweep JSON file.
    Returns (grid, base_config) where:
      grid        — dict of key → list of values (cartesian product will be run)
      base_config — dict of fixed overrides applied to every config in the grid
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sweep config not found: {path}")
    with open(p) as f:
        data = json.load(f)
    grid = data.get('grid', {})
    base_config = data.get('base_config', {})
    if not grid:
        raise ValueError(f"Sweep config has no 'grid' section: {path}")
    return grid, base_config


def expand_grid(grid: dict):
    keys = list(grid.keys())
    for combo in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_done_keys(csv_path: Path) -> set[tuple]:
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                done.add((
                    row['dataset'],
                    float(row['epsilon']),
                    float(row['public_frac']),
                    int(row['dict_per_class']),
                    int(row['seed']),
                ))
            except (KeyError, ValueError):
                pass
    return done


def append_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_KEYS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='MM-EdgeDP sweep runner')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--phase', type=int, choices=[1, 2],
                       help='Phase number — loads configs/sweep_phase{N}.json')
    group.add_argument('--config', type=str,
                       help='Path to a sweep JSON config file')
    p.add_argument('--output', type=str, default=None,
                   help='Output CSV path (default: outputs/sweep_phase{N}.csv)')
    p.add_argument('--dry-run', action='store_true',
                   help='Print configs without running experiments')
    p.add_argument('--resume', action='store_true',
                   help='Skip configs already present in the output CSV')
    p.add_argument('--set', dest='overrides', metavar='KEY=VALUE',
                   action='append', default=[],
                   help='Override any config key, e.g. --set epochs=5')
    p.add_argument('--verbose', action='store_true',
                   help='Pass verbose=True to run_experiment')
    return p.parse_args()


def parse_overrides(overrides: list[str]) -> dict:
    out = {}
    for kv in overrides:
        if '=' not in kv:
            raise ValueError(f"--set requires KEY=VALUE format, got: {kv!r}")
        k, v = kv.split('=', 1)
        for cast in (int, float):
            try:
                v = cast(v); break
            except ValueError:
                pass
        else:
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    config_path = PHASE_CONFIGS[args.phase] if args.phase else args.config
    grid, base_config = load_sweep_config(config_path)

    # CLI --set overrides win over both grid values and base_config
    cli_overrides = parse_overrides(args.overrides)

    # If --set targets a grid key, collapse it to a single value
    for k, v in cli_overrides.items():
        if k in grid:
            grid[k] = [v]

    configs = list(expand_grid(grid))
    total = len(configs)

    if args.output is not None:
        output_path = Path(args.output)
    elif args.phase:
        output_path = Path(PHASE_OUTPUTS[args.phase])
    else:
        output_path = OUTPUT_CSV
    done_keys = load_done_keys(output_path) if args.resume else set()

    label = f"Phase {args.phase}" if args.phase else config_path
    print(f"{label} ({config_path}): {total} configs", flush=True)

    if args.dry_run:
        for i, cfg in enumerate(configs, 1):
            merged = {**base_config, **cfg, **cli_overrides}
            print(f"  [{i:3d}/{total}] {merged}")
        return

    try:
        from experiments import run_experiment
    except ImportError as e:
        print(f"ERROR: could not import experiments.py — {e}", file=sys.stderr)
        sys.exit(1)

    skipped = 0
    for i, grid_cfg in enumerate(configs, 1):
        cfg = {**base_config, **grid_cfg, **cli_overrides}
        key = (cfg['dataset'], float(cfg['epsilon']),
               float(cfg['public_frac']), int(cfg['dict_per_class']), int(cfg['seed']))
        if key in done_keys:
            skipped += 1
            continue

        print(
            f"[{i:3d}/{total}] {cfg['dataset']}  "
            f"ε={cfg['epsilon']}  pub={cfg['public_frac']}  "
            f"dict={cfg['dict_per_class']}  seed={cfg['seed']}",
            flush=True,
        )
        try:
            result = run_experiment(cfg, verbose=args.verbose)
            result['status'] = 'ok'
            result['error'] = ''
        except Exception:
            tb = traceback.format_exc().strip().splitlines()[-1]
            print(f"  FAILED: {tb}", flush=True)
            result = {**cfg, 'status': 'error', 'error': tb}

        append_row(output_path, result)

    if skipped:
        print(f"Skipped {skipped} already-completed configs (--resume).", flush=True)
    print(f"Done. Results → {output_path}", flush=True)


if __name__ == '__main__':
    main()
