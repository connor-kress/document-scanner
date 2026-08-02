"""Signal when validation performance stops improving."""
from __future__ import annotations


class EarlyStopping:
    """Count epochs with no improvement and set should_stop when patience runs out.

    Parameters
    ----------
    patience : int
        How many epochs without improvement before stopping.
    min_delta : float
        Minimum change to count as an improvement.
    mode : str
        'max' when higher is better (IoU), 'min' when lower is better (loss).
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max") -> None:
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, metric: float) -> bool:
        """Update state. Returns True when training should stop."""
        if self.best is None:
            self.best = metric
            return False
        improved = (
            metric > self.best + self.min_delta
            if self.mode == "max"
            else metric < self.best - self.min_delta
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
