"""Save and restore the best model weights during training."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class ModelCheckpoint:
    """Write a checkpoint whenever the tracked metric improves.

    Parameters
    ----------
    path : Path-like
        Where to save the .pt file.
    mode : str
        'max' if a higher metric is better (IoU), 'min' otherwise.
    """

    def __init__(self, path: str | Path, mode: str = "max") -> None:
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best: float | None = None

    def __call__(self, metric: float, model: nn.Module) -> bool:
        """Save model if metric improved. Returns True when a new file is written."""
        improved = (
            self.best is None
            or (self.mode == "max" and metric > self.best)
            or (self.mode == "min" and metric < self.best)
        )
        if improved:
            self.best = metric
            torch.save(model.state_dict(), self.path)
            return True
        return False

    def load_best(self, model: nn.Module, device: torch.device | str = "cpu") -> nn.Module:
        """Restore the best weights into model in-place and return it."""
        model.load_state_dict(torch.load(self.path, map_location=device, weights_only=True))
        return model
