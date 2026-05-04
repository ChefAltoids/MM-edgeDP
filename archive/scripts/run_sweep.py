from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from mm_edgedp.config import apply_overrides, load_config
from mm_edgedp.runner import execute_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple seed/epsilon sweep.")
    parser.add_argument("--config", type=str, required=True, help="Path to sweep config JSON.")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _collect(values: Any, fallback: List[float | int]) -> List[float | int]:
    if isinstance(values, list) and values:
        return values
    return fallback


def main() -> None:
    args = parse_args()
    base = load_config(args.config, repo_root=args.repo_root)

    sweep = base.get("sweep", {})
    seeds = _collect(sweep.get("seeds"), [base.get("run", {}).get("seed", 42)])
    epsilons = _collect(sweep.get("epsilons"), [base.get("privacy", {}).get("epsilon", 1.0)])

    outcomes: List[Dict[str, Any]] = []
    for seed in seeds:
        for epsilon in epsilons:
            run_name = f"{base.get('run', {}).get('name', 'run')}_s{seed}_e{epsilon}"
            overrides = [
                f"run.seed={seed}",
                f"run.name={run_name}",
                f"privacy.epsilon={epsilon}",
            ]
            cfg = apply_overrides(base, overrides)
            outcome = execute_experiment(cfg, dry_run=args.dry_run)
            outcome["seed"] = seed
            outcome["epsilon"] = epsilon
            outcomes.append(outcome)

    print(json.dumps({"runs": outcomes}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
