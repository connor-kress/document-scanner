"""Multi-layer perceptron for document corner regression.

Input pipeline (MLP):
    (n, 36, 64) uint8
    → flatten to (n, 2304) float32 in [0, 1]
    → optional PCA  →  (n, pca_components)
    → CornerMLP
    → (n, 8) normalized coordinates

FlatDataset wraps the already-preprocessed numpy arrays so PyTorch's
DataLoader can iterate over them.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class CornerMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: Sequence[int] = (256, 128),
        dropout: Sequence[float] = (0.3, 0.2),
    ) -> None:
        super().__init__()
        if len(hidden) != len(dropout):
            raise ValueError("hidden and dropout must have the same length")
        layers: list[nn.Module] = []
        in_d = input_dim
        for out_d, p in zip(hidden, dropout):
            layers.extend([nn.Linear(in_d, out_d), nn.ReLU(inplace=True), nn.Dropout(p)])
            in_d = out_d
        layers.extend([nn.Linear(in_d, 8), nn.Sigmoid()])
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FlatDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]
