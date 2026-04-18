from __future__ import annotations

import argparse
import json

from mm_edgedp.config import apply_overrides, load_config
from mm_edgedp.runner import execute_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a config-driven MM-edgeDP experiment.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config JSON.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Repository root used to resolve extended config paths.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Dot-path override in key=value form, e.g. train.epochs=20",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and save config without running training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, repo_root=args.repo_root)
    config = apply_overrides(config, args.overrides)

    result = execute_experiment(config, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
