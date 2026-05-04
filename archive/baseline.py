import argparse
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import ToUndirected


def set_seed(seed: int = 175) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_ogbn_arxiv(root: str = "data/ogb"):
    """
    Loads ogbn-arxiv with PyG wrapper.
    We make the graph undirected for the GCN baseline, which is common practice.
    """
    dataset = PygNodePropPredDataset(
        name="ogbn-arxiv",
        root=root,
        transform=ToUndirected()
    )
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name="ogbn-arxiv")
    return dataset, data, split_idx, evaluator


def prepare_numpy_splits(data, split_idx) -> SplitData:
    x = data.x.cpu().numpy()
    y = data.y.squeeze(1).cpu().numpy()

    train_idx = split_idx["train"].cpu().numpy()
    valid_idx = split_idx["valid"].cpu().numpy()
    test_idx = split_idx["test"].cpu().numpy()

    return SplitData(
        x_train=x[train_idx],
        y_train=y[train_idx],
        x_valid=x[valid_idx],
        y_valid=y[valid_idx],
        x_test=x[test_idx],
        y_test=y[test_idx],
    )


def run_logistic_regression(data, split_idx):
    splits = prepare_numpy_splits(data, split_idx)

    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
        verbose=1,
    )
    clf.fit(splits.x_train, splits.y_train)

    train_pred = clf.predict(splits.x_train)
    valid_pred = clf.predict(splits.x_valid)
    test_pred = clf.predict(splits.x_test)

    train_acc = accuracy_score(splits.y_train, train_pred)
    valid_acc = accuracy_score(splits.y_valid, valid_pred)
    test_acc = accuracy_score(splits.y_test, test_pred)

    print("\n=== Logistic Regression Results ===")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Valid Accuracy: {valid_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")

    return {
        "train_acc": train_acc,
        "valid_acc": valid_acc,
        "test_acc": test_acc,
    }


class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True, normalize=True)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


@torch.no_grad()
def evaluate_gnn(model, data, split_idx, evaluator, device):
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    y_pred = out.argmax(dim=-1, keepdim=True).cpu()

    results = {}
    for split in ["train", "valid", "test"]:
        acc = evaluator.eval({
            "y_true": data.y[split_idx[split]],
            "y_pred": y_pred[split_idx[split]],
        })["acc"]
        results[f"{split}_acc"] = acc
    return results


def train_gcn(
    data,
    dataset,
    split_idx,
    evaluator,
    device,
    hidden_channels=256,
    lr=0.01,
    weight_decay=5e-4,
    epochs=200,
    dropout=0.5,
):
    model = GCN(
        in_channels=data.x.size(-1),
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_idx = split_idx["train"].to(device)

    data = data.to(device)
    best_valid = 0.0
    best_test = 0.0
    best_train = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_idx], data.y.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()

        metrics = evaluate_gnn(model, data, split_idx, evaluator, device)

        if metrics["valid_acc"] > best_valid:
            best_valid = metrics["valid_acc"]
            best_train = metrics["train_acc"]
            best_test = metrics["test_acc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch: {epoch:03d}, "
                f"Loss: {loss.item():.4f}, "
                f"Train: {metrics['train_acc']:.4f}, "
                f"Valid: {metrics['valid_acc']:.4f}, "
                f"Test: {metrics['test_acc']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n=== Best GCN Results (selected by validation accuracy) ===")
    print(f"Train Accuracy: {best_train:.4f}")
    print(f"Valid Accuracy: {best_valid:.4f}")
    print(f"Test Accuracy:  {best_test:.4f}")

    return model, {
        "train_acc": best_train,
        "valid_acc": best_valid,
        "test_acc": best_test,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/ogb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--run_logreg", action="store_true")
    parser.add_argument("--run_gcn", action="store_true")
    args = parser.parse_args()

    if not args.run_logreg and not args.run_gcn:
        args.run_logreg = True
        args.run_gcn = True

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset, data, split_idx, evaluator = load_ogbn_arxiv(root=args.root)

    print("\n=== Dataset Info ===")
    print(data)
    print(f"Num nodes: {data.num_nodes}")
    print(f"Num edges: {data.edge_index.size(1)}")
    print(f"Num node features: {data.x.size(1)}")
    print(f"Num classes: {dataset.num_classes}")
    print(f"Train/Valid/Test sizes: "
          f"{split_idx['train'].numel()}/"
          f"{split_idx['valid'].numel()}/"
          f"{split_idx['test'].numel()}")

    if args.run_logreg:
        run_logistic_regression(data, split_idx)

    if args.run_gcn:
        train_gcn(
            data=data,
            dataset=dataset,
            split_idx=split_idx,
            evaluator=evaluator,
            device=device,
            hidden_channels=args.hidden_channels,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            dropout=args.dropout,
        )


if __name__ == "__main__":
    main()
