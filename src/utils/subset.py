"""Deterministic, group-aware sampling for reduced-budget training runs."""
from __future__ import annotations

import numpy as np


def grouped_subset_indices(
    groups: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Sample the same fraction within every group and return sorted indices."""
    if not 0 < fraction <= 1:
        raise ValueError("train fraction must be in the interval (0, 1]")

    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    if fraction == 1 or len(groups) == 0:
        return np.arange(len(groups), dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected = []
    for group in np.unique(groups):
        candidates = np.flatnonzero(groups == group)
        count = max(1, round(len(candidates) * fraction))
        selected.append(rng.choice(candidates, size=count, replace=False))
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)
