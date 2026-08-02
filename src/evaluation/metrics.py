"""Extended evaluation metrics and timing helpers for model comparison."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics import FRAME_DIAG, corner_error_px, iou_batch


def full_evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: tuple[float, ...] = (0.5, 0.75, 0.9),
    iou_limit: int | None = None,
) -> dict:
    err = corner_error_px(y_true, y_pred)
    iou = iou_batch(y_true, y_pred, limit=iou_limit)
    out = {
        "corner_err_px": float(err.mean()),
        "corner_err_px_median": float(np.median(err)),
        "corner_err_norm": float(err.mean() / FRAME_DIAG),
        "iou_mean": float(iou.mean()),
        "iou_median": float(np.median(iou)),
        "iou_std": float(iou.std()),
    }
    for t in thresholds:
        out[f"iou>={t:.2f}"] = float((iou >= t).mean())
    return out


def evaluate_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    thresholds: tuple[float, ...] = (0.5, 0.75, 0.9),
) -> dict[str, dict]:
    return {
        str(group): full_evaluate(y_true[groups == group], y_pred[groups == group], thresholds)
        for group in np.unique(groups)
    }


def measure_inference_time_sklearn(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    n_warmup: int = 2,
) -> float:
    predict_fn(X[:n_warmup])
    t0 = time.perf_counter()
    predict_fn(X)
    return (time.perf_counter() - t0) / len(X)


def measure_inference_time_torch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    use_cuda = device.type == "cuda"
    model.eval()
    model.to(device)
    batches = [X_batch.to(device) for X_batch, _ in loader]
    with torch.no_grad():
        _ = model(batches[0])
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    total = 0
    with torch.no_grad():
        for X_batch in batches:
            _ = model(X_batch)
            total += len(X_batch)
    if use_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / total