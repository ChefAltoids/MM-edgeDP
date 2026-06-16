"""
Evaluation metrics for DP-OT adaptation experiments.

evaluate() runs the trained model inductively on the target graph and
returns accuracy, AUROC, and F1.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch_geometric.data import Data


def evaluate(
    model: "GraphSAGEModel",
    data: Data,
    device: str = "cpu",
) -> dict[str, float]:
    """
    Inductive evaluation: run model forward pass on data's graph structure.

    Returns dict with keys: acc, auroc, f1
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(dev), data.edge_index.to(dev))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

    y = data.y.cpu().numpy()
    acc = float(accuracy_score(y, preds))
    f1 = float(f1_score(y, preds, average="macro", zero_division=0))

    num_classes = probs.shape[1]
    if num_classes == 2:
        auroc = float(roc_auc_score(y, probs[:, 1]))
    else:
        try:
            auroc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
        except ValueError:
            auroc = float("nan")

    return {"acc": acc, "auroc": auroc, "f1": f1}


def prototype_l1_error(
    alpha_dp: np.ndarray,
    alpha_true: np.ndarray,
) -> float:
    """L1 distance between DP estimate and true (non-private) prototype masses."""
    return float(np.abs(alpha_dp - alpha_true).sum())
