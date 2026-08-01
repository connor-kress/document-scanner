""" Benchmark model metrics utils"""

from __future__ import annotations

import cv2
import numpy as np

from preprocess import FRAME_H, FRAME_W, denormalise

FRAME_DIAG = float(np.hypot(FRAME_W, FRAME_H))


def corner_error_px(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Average distance in pixels between guessed corners and real ones.
    """
    a, b = denormalise(y_true), denormalise(y_pred)
    return np.linalg.norm(a - b, axis=2).mean(axis=1)


def normalised_corner_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Same as corner_error_px, but as a fraction of the photo's diagonal, so
    it doesn't depend on the photo size."""
    return corner_error_px(y_true, y_pred) / FRAME_DIAG


def quad_iou(a: np.ndarray, b: np.ndarray, scale: int = 4) -> float:
    """How much two quads overlap, from 0 (nothing) to 1 (perfect match).
    """
    canvas = (FRAME_H // scale, FRAME_W // scale)
    ma, mb = np.zeros(canvas, np.uint8), np.zeros(canvas, np.uint8)
    cv2.fillConvexPoly(ma, (a / scale).astype(np.int32), 1)
    cv2.fillConvexPoly(mb, (b / scale).astype(np.int32), 1)
    union = np.count_nonzero(ma | mb)
    return np.count_nonzero(ma & mb) / union if union else 0.0


def iou_batch(y_true: np.ndarray, y_pred: np.ndarray,
              limit: int | None = None) -> np.ndarray:
    """IoU for every photo. `limit` checks only a sample of them, which is much
    faster while you are experimenting. Leave it out for final numbers."""
    a, b = denormalise(y_true), denormalise(y_pred)
    idx = (np.linspace(0, len(a) - 1, limit).astype(int)
           if limit and limit < len(a) else np.arange(len(a)))
    return np.array([quad_iou(a[i], b[i]) for i in idx])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             thresholds=(0.8, 0.9, 0.95), iou_limit: int | None = None) -> dict:
    """All the scores at once, in one dictionary. Use this for every model so
    the results can be compared fairly."""
    err = corner_error_px(y_true, y_pred)
    iou = iou_batch(y_true, y_pred, limit=iou_limit)
    out = {
        "corner_err_px": float(err.mean()),
        "corner_err_px_median": float(np.median(err)),
        "corner_err_norm": float(err.mean() / FRAME_DIAG),
        "iou": float(iou.mean()),
    }
    for t in thresholds:
        out[f"iou>{t}"] = float((iou > t).mean())
    return out


def constant_baseline(y_train: np.ndarray, n: int) -> np.ndarray:
    """Guess the same position for every photo and display model metrics
    """
    return np.repeat(y_train.mean(axis=0, keepdims=True), n, axis=0)


def format_row(name: str, metrics: dict) -> str:
    return (f"  {name:24s} corner err {metrics['corner_err_px']:7.1f} px"
            f"  (median {metrics['corner_err_px_median']:6.1f})"
            f"  normalised {metrics['corner_err_norm']:.4f}"
            f"  IoU {metrics['iou']:.3f}"
            f"  IoU>0.9 {100 * metrics.get('iou>0.9', float('nan')):5.1f}%")
