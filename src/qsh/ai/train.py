"""Training utilities for the small models (centralized baseline).

A minimal, GPU-aware training loop used both to produce the trained models the
public demo serves (inference only) and as the local "full implementation"
counterpart to the federated path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .models import build_model, device


@dataclass
class TrainResult:
    model: nn.Module
    train_acc: float
    test_acc: float
    losses: list[float]


def _to_loader(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(xt, yt), batch_size=batch, shuffle=shuffle)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    kind: str = "mlp",
    *,
    epochs: int = 15,
    batch: int = 64,
    lr: float = 1e-3,
    test_split: float = 0.2,
    seed: int = 0,
    model_kwargs: dict | None = None,
) -> TrainResult:
    """Train ``kind`` on (X, y); returns the model and train/test accuracy."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_split))
    tr, te = idx[:cut], idx[cut:]

    model_kwargs = dict(model_kwargs or {})
    if kind == "mlp":
        model_kwargs.setdefault("n_features", X.shape[1])
    model_kwargs.setdefault("n_classes", int(y.max()) + 1)
    dev = device()
    model = build_model(kind, **model_kwargs).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = _to_loader(X[tr], y[tr], batch, True)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        ep = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            ep += loss.item()
        losses.append(ep / max(1, len(loader)))

    return TrainResult(model, _accuracy(model, X[tr], y[tr], dev),
                       _accuracy(model, X[te], y[te], dev), losses)


@torch.no_grad()
def _accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray, dev) -> float:
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32).to(dev)
    pred = model(xb).argmax(1).cpu().numpy()
    return float((pred == y).mean())
