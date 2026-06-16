"""
GraphSAGE model for source-graph training.

Supports standard cross-entropy training and per-node weighted loss
for importance-reweighted adaptation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GraphSAGEModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        sizes = [in_channels] + [hidden] * (num_layers - 1) + [out_channels]
        for i in range(num_layers):
            self.convs.append(SAGEConv(sizes[i], sizes[i + 1]))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def embed(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return penultimate-layer embeddings (before final linear)."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def _train_loop(
    model: GraphSAGEModel,
    data: Data,
    node_weights: Tensor | None,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> GraphSAGEModel:
    model = model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    if node_weights is not None:
        node_weights = node_weights.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x, edge_index)
        per_node_loss = F.cross_entropy(logits, y, reduction="none")
        if node_weights is not None:
            loss = (node_weights * per_node_loss).mean()
        else:
            loss = per_node_loss.mean()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def train_source_gnn(
    data: Data,
    *,
    hidden: int = 64,
    num_layers: int = 2,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    dropout: float = 0.5,
    seed: int = 0,
    device: str = "cpu",
) -> GraphSAGEModel:
    """Train a GraphSAGE model on source graph with uniform (unweighted) loss."""
    set_seed(seed)
    in_channels = data.x.shape[1]
    num_classes = int(data.y.max().item()) + 1
    model = GraphSAGEModel(in_channels, hidden, num_classes, num_layers, dropout)
    dev = torch.device(device)
    return _train_loop(model, data, node_weights=None, epochs=epochs, lr=lr,
                       weight_decay=weight_decay, device=dev)


def train_weighted_source_gnn(
    data: Data,
    node_weights: np.ndarray | Tensor,
    *,
    hidden: int = 64,
    num_layers: int = 2,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    dropout: float = 0.5,
    seed: int = 0,
    device: str = "cpu",
) -> GraphSAGEModel:
    """Train a GraphSAGE model on source graph with per-node importance weights."""
    set_seed(seed)
    in_channels = data.x.shape[1]
    num_classes = int(data.y.max().item()) + 1
    model = GraphSAGEModel(in_channels, hidden, num_classes, num_layers, dropout)
    if isinstance(node_weights, np.ndarray):
        node_weights = torch.tensor(node_weights, dtype=torch.float32)
    dev = torch.device(device)
    return _train_loop(model, data, node_weights=node_weights, epochs=epochs,
                       lr=lr, weight_decay=weight_decay, device=dev)
