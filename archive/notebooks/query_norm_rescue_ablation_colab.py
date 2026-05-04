"""Colab-friendly ablation for query normalization rescue strategies.

Run this from MM-edgeDP repo root (or %run it from Colab after cloning repo).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import nbformat
import torch
import torch.nn.functional as F


def colab_bootstrap() -> None:
    repo_url = os.environ.get("MMEDGEDP_REPO_URL", "").strip()
    repo_dir = Path("/content/MM-edgeDP")

    if not repo_dir.exists():
        if not repo_url:
            raise RuntimeError("Set MMEDGEDP_REPO_URL or clone the repo to /content/MM-edgeDP first.")
        subprocess.check_call(["git", "clone", repo_url, str(repo_dir)])

    os.chdir(repo_dir)
    req_path = Path("requirements-colab.txt")
    if req_path.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_path)])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch-geometric", "ogb", "pandas"])


def bootstrap_core_definitions() -> None:
    source_notebook = Path("KL_perturb3.ipynb")
    if not source_notebook.exists():
        raise FileNotFoundError(f"Missing {source_notebook}. Run from MM-edgeDP root.")

    nb = nbformat.read(source_notebook.open("r", encoding="utf-8"), as_version=4)
    stop_markers = [
        "CONFIG = get_nesterov_sgd_ablation_config()",
        "EXECUTE_FIRST_FIVE = True",
    ]

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        if "```" in src:
            continue
        if any(marker in src for marker in stop_markers):
            break
        exec(src, globals())


def _resolve_tau(query_clip_norm: float = 1.0, query_clip_tau=None) -> float:
    return float(query_clip_tau if query_clip_tau is not None else query_clip_norm)


def install_query_normalization_patch() -> None:
    def apply_query_normalization(Q, mode="none", clip_norm=1.0, clip_tau=None):
        if mode is None or mode == "none":
            return Q
        if mode == "l2_row":
            return F.normalize(Q, p=2, dim=1)
        if mode == "clipped_l2":
            norms = torch.linalg.norm(Q, dim=1, keepdim=True).clamp_min(1e-12)
            scale = torch.clamp(float(clip_norm) / norms, max=1.0)
            return Q * scale

        tau = _resolve_tau(query_clip_norm=clip_norm, query_clip_tau=clip_tau)
        norms = torch.linalg.norm(Q, dim=1, keepdim=True).clamp_min(1e-12)

        if mode in {"clip_l2", "hard_l2"}:
            return Q / torch.clamp(norms, min=tau)
        if mode == "soft_l2":
            return Q / (norms + tau)

        raise ValueError(f"Unsupported query_normalization: {mode}")

    def build_private_query_features(
        private_edges,
        x_features,
        query_mode="one_hop",
        query_normalization="none",
        query_clip_norm=1.0,
        query_clip_tau=None,
        return_diagnostics=False,
    ):
        num_nodes = x_features.size(0)
        H1 = one_hop_mean_aggregate(private_edges, x_features, num_nodes=num_nodes)

        if query_mode == "one_hop":
            Q_raw = H1
        elif query_mode == "two_hop_concat":
            H2 = one_hop_mean_aggregate(private_edges, H1, num_nodes=num_nodes)
            Q_raw = torch.cat([H1, H2], dim=1)
        else:
            raise ValueError(f"Unsupported query_mode: {query_mode}")

        Q = apply_query_normalization(Q_raw, mode=query_normalization, clip_norm=query_clip_norm, clip_tau=query_clip_tau)

        if not return_diagnostics:
            return Q

        norms = torch.linalg.norm(Q_raw, dim=1)
        tau = _resolve_tau(query_clip_norm=query_clip_norm, query_clip_tau=query_clip_tau)
        if norms.numel() == 0:
            diag = {
                "query_normalization": str(query_normalization),
                "query_clip_tau": float(tau),
                "query_norm_mean_pre_norm": 0.0,
                "query_norm_median_pre_norm": 0.0,
                "query_small_norm_fraction": 0.0,
            }
        else:
            diag = {
                "query_normalization": str(query_normalization),
                "query_clip_tau": float(tau),
                "query_norm_mean_pre_norm": float(norms.mean().item()),
                "query_norm_median_pre_norm": float(norms.median().item()),
                "query_small_norm_fraction": float((norms < tau).float().mean().item()),
            }
        return Q, diag

    synthesize_original = globals()["synthesize_edge_dp_assignments"]

    @torch.no_grad()
    def synthesize_edge_dp_assignments(
        private_edges,
        x_features,
        labels,
        private_nodes_tensor,
        dict_features,
        class_to_proto_indices,
        epsilon_total,
        utility_sensitivity=1.0,
        query_mode="one_hop",
        label_conditioning=True,
        query_normalization="none",
        query_clip_norm=1.0,
        query_clip_tau=None,
        return_full_diagnostics=False,
    ):
        effective_tau = _resolve_tau(query_clip_norm=query_clip_norm, query_clip_tau=query_clip_tau)
        assignments, diagnostics = synthesize_original(
            private_edges=private_edges,
            x_features=x_features,
            labels=labels,
            private_nodes_tensor=private_nodes_tensor,
            dict_features=dict_features,
            class_to_proto_indices=class_to_proto_indices,
            epsilon_total=epsilon_total,
            utility_sensitivity=utility_sensitivity,
            query_mode=query_mode,
            label_conditioning=label_conditioning,
            query_normalization=query_normalization,
            query_clip_norm=effective_tau,
            return_full_diagnostics=return_full_diagnostics,
        )

        _, query_diag = build_private_query_features(
            private_edges=private_edges,
            x_features=x_features,
            query_mode=query_mode,
            query_normalization=query_normalization,
            query_clip_norm=query_clip_norm,
            query_clip_tau=query_clip_tau,
            return_diagnostics=True,
        )
        diagnostics = dict(diagnostics)
        diagnostics.update(query_diag)
        return assignments, diagnostics

    globals()["apply_query_normalization"] = apply_query_normalization
    globals()["build_private_query_features"] = build_private_query_features
    globals()["synthesize_edge_dp_assignments"] = synthesize_edge_dp_assignments


def prepare_state(cfg: dict) -> dict:
    set_seed(int(cfg["seed"]))
    dataset, data, num_classes = load_graph_dataset(cfg["dataset_name"])

    public_nodes, train_nodes, val_nodes = stratified_split_indices(
        data.y,
        public_frac=float(cfg["public_frac"]),
        val_frac=float(cfg["val_frac"]),
        seed=int(cfg["seed"]),
    )

    train_nodes = cap_indices(train_nodes, int(cfg["max_private_nodes"]), seed=int(cfg["seed"]) + 1)
    val_nodes = cap_indices(val_nodes, int(cfg["max_val_nodes"]), seed=int(cfg["seed"]) + 2)

    pub_edge_index, private_edge_index, val_edge_index = split_edges_by_nodes(
        data.edge_index, public_nodes, train_nodes, val_nodes, data.num_nodes
    )

    public_dict, public_proto_feats, class_to_proto_indices = build_class_conditional_public_dictionary(
        data_obj=data,
        x_features=data.x,
        labels_tensor=data.y,
        public_nodes_tensor=public_nodes,
        pub_edges=pub_edge_index,
        num_classes=num_classes,
        dict_per_class=int(cfg["dict_per_class"]),
        num_hops=int(cfg["walk_hops"]),
        query_mode=str(cfg["query_mode"]),
        min_class_fraction=float(cfg["min_class_fraction"]),
        max_proto_nodes=int(cfg["max_proto_nodes"]),
    )

    val_loader, _ = build_validation_loader(
        x_l2=data.x,
        labels=data.y,
        val_nodes=val_nodes,
        val_edge_index=val_edge_index,
        query_hops=int(cfg["query_hops"]),
        max_val_graphs=int(cfg["max_val_nodes"]),
        seed=int(cfg["seed"]) + 3,
    )

    return {
        "data": data,
        "num_classes": num_classes,
        "train_nodes": train_nodes,
        "private_edge_index": private_edge_index,
        "public_dict": public_dict,
        "public_proto_feats": public_proto_feats,
        "class_to_proto_indices": class_to_proto_indices,
        "val_loader": val_loader,
    }


def run_sweep(config_path: str = "configs/sweeps/query_norm_rescue_v1.json") -> Path:
    try:
        import pandas as pd
    except Exception:
        pd = None

    with open(config_path, "r", encoding="utf-8") as f:
        sweep_cfg = json.load(f)

    base_cfg = dict(sweep_cfg["base_config"])
    rows = []

    for seed in sweep_cfg["seeds"]:
        cfg_seed = dict(base_cfg)
        cfg_seed["seed"] = int(seed)
        state = prepare_state(cfg_seed)
        utility_sensitivity = 2.0 if cfg_seed["query_mode"] == "two_hop_concat" else 1.0

        for epsilon in sweep_cfg["epsilons"]:
            for arm in sweep_cfg["arms"]:
                t0 = time.time()
                mode = arm["query_normalization"]
                tau = arm.get("query_clip_tau", None)

                assignments, diag = synthesize_edge_dp_assignments(
                    private_edges=state["private_edge_index"],
                    x_features=state["data"].x,
                    labels=state["data"].y,
                    private_nodes_tensor=state["train_nodes"],
                    dict_features=state["public_proto_feats"],
                    class_to_proto_indices=state["class_to_proto_indices"],
                    epsilon_total=float(epsilon),
                    utility_sensitivity=utility_sensitivity,
                    query_mode=str(cfg_seed["query_mode"]),
                    label_conditioning=bool(cfg_seed["label_conditioning"]),
                    query_normalization=str(mode),
                    query_clip_tau=tau,
                )

                best_val_acc = train_gnn_from_assignments(
                    assignments=assignments,
                    public_dict=state["public_dict"],
                    val_loader=state["val_loader"],
                    num_features=state["data"].x.size(1),
                    num_classes=state["num_classes"],
                    epochs=int(cfg_seed["epochs"]),
                    batch_size=int(cfg_seed["batch_size"]),
                    lr=float(cfg_seed["lr"]),
                    feature_jitter_std=float(cfg_seed["feature_jitter_std"]),
                    edge_dropout_p=float(cfg_seed["edge_dropout_p"]),
                )

                rows.append(
                    {
                        "seed": int(seed),
                        "epsilon_total": float(epsilon),
                        "arm_name": str(arm["name"]),
                        "query_normalization": str(mode),
                        "query_clip_tau": None if tau is None else float(tau),
                        "best_val_acc": float(best_val_acc),
                        "mean_entropy": float(diag.get("mean_entropy", float("nan"))),
                        "mean_top_probability": float(diag.get("mean_top_probability", float("nan"))),
                        "mean_nearest_distance": float(diag.get("mean_nearest_distance", float("nan"))),
                        "mean_sampled_distance": float(diag.get("mean_sampled_distance", float("nan"))),
                        "query_norm_mean_pre_norm": float(diag.get("query_norm_mean_pre_norm", float("nan"))),
                        "query_norm_median_pre_norm": float(diag.get("query_norm_median_pre_norm", float("nan"))),
                        "query_small_norm_fraction": float(diag.get("query_small_norm_fraction", float("nan"))),
                        "elapsed_sec": float(time.time() - t0),
                    }
                )

    out_dir = Path("outputs/query_norm_rescue_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    if pd is not None:
        raw_df = pd.DataFrame(rows).sort_values(["epsilon_total", "seed", "arm_name"])
        summary_df = raw_df.groupby(["epsilon_total", "arm_name", "query_normalization", "query_clip_tau"], as_index=False).agg(
            mean_best_val_acc=("best_val_acc", "mean"),
            std_best_val_acc=("best_val_acc", "std"),
            mean_entropy=("mean_entropy", "mean"),
            mean_top_probability=("mean_top_probability", "mean"),
            mean_nearest_distance=("mean_nearest_distance", "mean"),
            mean_sampled_distance=("mean_sampled_distance", "mean"),
            mean_query_norm=("query_norm_mean_pre_norm", "mean"),
            mean_small_norm_fraction=("query_small_norm_fraction", "mean"),
            mean_elapsed_sec=("elapsed_sec", "mean"),
            seed_count=("seed", "nunique"),
        )
        summary_df["std_best_val_acc"] = summary_df["std_best_val_acc"].fillna(0.0)
        raw_df.to_csv(out_dir / "query_norm_rescue_raw.csv", index=False)
        summary_df.to_csv(out_dir / "query_norm_rescue_summary.csv", index=False)
        print(summary_df)
    else:
        with open(out_dir / "query_norm_rescue_raw.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

    return out_dir


if __name__ == "__main__":
    # Comment out colab_bootstrap() if you are already in repo root with deps installed.
    colab_bootstrap()
    bootstrap_core_definitions()
    install_query_normalization_patch()
    output_dir = run_sweep("configs/sweeps/query_norm_rescue_v1.json")
    print(f"Saved outputs under: {output_dir}")
