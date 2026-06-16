"""
Plotting utilities for DP-OT sweep results.

Three main plots:
  1. Target AUROC vs epsilon (at fixed gamma=0.5)
  2. Target AUROC vs gamma  (at fixed epsilon=1.0)
  3. Prototype L1 error vs epsilon
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["source_only", "dp_histogram", "dp_exponential", "oracle", "target_oracle"]
METHOD_LABELS = {
    "source_only": "Source-only",
    "dp_histogram": "DP Histogram (A)",
    "dp_exponential": "DP Exponential (B)",
    "oracle": "Oracle (nonprivate)",
    "target_oracle": "Target oracle",
}
COLORS = {
    "source_only": "#999999",
    "dp_histogram": "#1f77b4",
    "dp_exponential": "#ff7f0e",
    "oracle": "#2ca02c",
    "target_oracle": "#d62728",
}


def plot_auroc_vs_epsilon(
    df: pd.DataFrame,
    gamma: float = 0.5,
    K: int | None = None,
    out_path: str | None = None,
) -> plt.Figure:
    """Target AUROC vs epsilon, one line per method, at fixed gamma."""
    sub = df[np.isclose(df["gamma"], gamma)]
    if K is not None:
        sub = sub[sub["K"] == K]

    fig, ax = plt.subplots(figsize=(6, 4))
    for method in METHOD_ORDER:
        m = sub[sub["method"] == method]
        if m.empty:
            continue
        grouped = m.groupby("epsilon")["auroc"].agg(["mean", "sem"]).reset_index()
        label = METHOD_LABELS.get(method, method)
        color = COLORS.get(method, None)
        ls = "--" if method in ("source_only", "target_oracle") else "-"
        ax.errorbar(grouped["epsilon"], grouped["mean"], yerr=grouped["sem"],
                    label=label, color=color, linestyle=ls, marker="o", capsize=3)

    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget ε")
    ax.set_ylabel("Target AUROC")
    ax.set_title(f"AUROC vs ε  (γ={gamma})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_auroc_vs_gamma(
    df: pd.DataFrame,
    epsilon: float = 1.0,
    K: int | None = None,
    out_path: str | None = None,
) -> plt.Figure:
    """Target AUROC vs covariate-shift gamma, one line per method, at fixed epsilon."""
    sub = df[
        (df["method"] == "source_only") |
        np.isclose(df["epsilon"], epsilon)
    ]
    if K is not None:
        sub = sub[sub["K"] == K]

    fig, ax = plt.subplots(figsize=(6, 4))
    for method in METHOD_ORDER:
        m = sub[sub["method"] == method]
        if m.empty:
            continue
        grouped = m.groupby("gamma")["auroc"].agg(["mean", "sem"]).reset_index()
        label = METHOD_LABELS.get(method, method)
        color = COLORS.get(method, None)
        ls = "--" if method in ("source_only", "target_oracle") else "-"
        ax.errorbar(grouped["gamma"], grouped["mean"], yerr=grouped["sem"],
                    label=label, color=color, linestyle=ls, marker="o", capsize=3)

    ax.set_xlabel("Shift magnitude γ")
    ax.set_ylabel("Target AUROC")
    ax.set_title(f"AUROC vs γ  (ε={epsilon})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_l1_vs_epsilon(
    df: pd.DataFrame,
    gamma: float = 0.5,
    K: int | None = None,
    out_path: str | None = None,
) -> plt.Figure:
    """Prototype L1 error vs epsilon for DP methods."""
    sub = df[np.isclose(df["gamma"], gamma) & df["method"].isin(["dp_histogram", "dp_exponential"])]
    if K is not None:
        sub = sub[sub["K"] == K]

    fig, ax = plt.subplots(figsize=(6, 4))
    for method in ["dp_histogram", "dp_exponential"]:
        m = sub[sub["method"] == method]
        if m.empty:
            continue
        grouped = m.groupby("epsilon")["proto_l1_error"].agg(["mean", "sem"]).reset_index()
        label = METHOD_LABELS.get(method, method)
        color = COLORS.get(method, None)
        ax.errorbar(grouped["epsilon"], grouped["mean"], yerr=grouped["sem"],
                    label=label, color=color, marker="o", capsize=3)

    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget ε")
    ax.set_ylabel("Prototype L1 error")
    ax.set_title(f"Prototype mass recovery vs ε  (γ={gamma})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def make_all_plots(
    csv_path: str,
    out_dir: str = "dp_ot/outputs",
    gamma: float = 0.5,
    epsilon: float = 1.0,
) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    plot_auroc_vs_epsilon(df, gamma=gamma,
                          out_path=f"{out_dir}/auroc_vs_epsilon.pdf")
    plot_auroc_vs_gamma(df, epsilon=epsilon,
                        out_path=f"{out_dir}/auroc_vs_gamma.pdf")
    plot_l1_vs_epsilon(df, gamma=gamma,
                       out_path=f"{out_dir}/l1_vs_epsilon.pdf")
    print(f"Plots saved to {out_dir}/")
