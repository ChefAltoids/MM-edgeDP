from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from mm_edgedp.config import dump_config


def _build_output_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    run_cfg = config.get("run", {})
    output_root = Path(run_cfg.get("output_dir", "outputs"))
    run_name = run_cfg.get("name", "unnamed_run")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "resolved_config": run_dir / "resolved_config.json",
        "metrics": run_dir / "metrics.json",
    }


def execute_experiment(config: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    paths = _build_output_paths(config)
    dump_config(config, paths["resolved_config"])

    if dry_run:
        return {
            "status": "dry-run",
            "resolved_config_path": str(paths["resolved_config"]),
            "metrics_path": str(paths["metrics"]),
        }

    import torch
    import baseline

    run_cfg = config.get("run", {})
    dataset_cfg = config.get("dataset", {})
    tasks_cfg = config.get("tasks", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})

    dataset_name = dataset_cfg.get("name", "ogbn-arxiv")
    if dataset_name != "ogbn-arxiv":
        raise ValueError("Only ogbn-arxiv is supported by the current scripted runner.")

    baseline.set_seed(int(run_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, data, split_idx, evaluator = baseline.load_ogbn_arxiv(
        root=str(dataset_cfg.get("root", "data/ogb"))
    )

    results: Dict[str, Any] = {
        "run": {
            "name": run_cfg.get("name", "unnamed_run"),
            "seed": int(run_cfg.get("seed", 42)),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
        },
        "dataset": {
            "name": dataset_name,
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.size(1)),
            "num_features": int(data.x.size(1)),
            "num_classes": int(dataset.num_classes),
        },
        "metrics": {},
    }

    if tasks_cfg.get("run_logreg", True):
        results["metrics"]["logreg"] = baseline.run_logistic_regression(data, split_idx)

    if tasks_cfg.get("run_gcn", True):
        _, gcn_metrics = baseline.train_gcn(
            data=data,
            dataset=dataset,
            split_idx=split_idx,
            evaluator=evaluator,
            device=device,
            hidden_channels=int(model_cfg.get("hidden_channels", 256)),
            lr=float(train_cfg.get("lr", 0.01)),
            weight_decay=float(train_cfg.get("weight_decay", 5e-4)),
            epochs=int(train_cfg.get("epochs", 200)),
            dropout=float(model_cfg.get("dropout", 0.5)),
        )
        results["metrics"]["gcn"] = gcn_metrics

    with paths["metrics"].open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
        fh.write("\n")

    return {
        "status": "completed",
        "resolved_config_path": str(paths["resolved_config"]),
        "metrics_path": str(paths["metrics"]),
    }
