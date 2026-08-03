"""Evaluation helpers for the selected Ridge, MLP, and CNN models."""
from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import FrameSet, TorchFrameSet, load_arrays
from evaluation.metrics import evaluate_by_group, full_evaluate, measure_inference_time_torch
from models.cnn import DocumentCNN
from models.mlp import CornerMLP, FlatDataset
from models.ridge_model import RidgePipeline


def assert_same_split(reference_paths: np.ndarray, candidate_paths: np.ndarray, label: str) -> None:
    if len(reference_paths) != len(candidate_paths) or not np.array_equal(reference_paths, candidate_paths):
        raise ValueError(f"split mismatch for {label}; model comparisons must use identical rows")


def evaluate_ridge(
    model_path: str | Path,
    split: str = "test",
    scheme: str = "split_video",
    array_name: str = "tab_64x36_clahe",
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parts, _, raw = load_arrays(array_name, scheme=scheme)
    X_eval, y_eval = parts[split]
    mask = raw[scheme] == split
    image_paths = raw["image_path"][mask]
    bg_ids = raw["bg_id"][mask]
    model = RidgePipeline.load(model_path)
    y_pred = model.predict(X_eval)
    metrics = full_evaluate(y_eval, y_pred, iou_limit=None)
    metrics["inference_time_ms"] = model.measure_inference_time(X_eval) * 1000
    metrics["by_background"] = evaluate_by_group(y_eval, y_pred, bg_ids)
    return metrics, y_eval, y_pred, image_paths, bg_ids


def evaluate_mlp(
    model_path: str | Path,
    input_dim: int | None,
    hidden: list[int],
    dropout: list[float],
    split: str = "test",
    scheme: str = "split_video",
    array_name: str = "tab_64x36_clahe",
    pca_path: str | Path | None = None,
    sharpened: bool = False,
    preprocessing: str = "none",
    device: str | None = None,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parts, _, raw = load_arrays(array_name, scheme=scheme)
    X_eval_raw, y_eval = parts[split]
    mask = raw[scheme] == split
    image_paths = raw["image_path"][mask]
    bg_ids = raw["bg_id"][mask]

    if preprocessing == "rgb_to_gray":
        X_eval_raw = np.stack([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in X_eval_raw])
    elif preprocessing != "none":
        raise ValueError(f"unsupported MLP preprocessing: {preprocessing}")
    if sharpened:
        sharpened_images = np.empty_like(X_eval_raw)
        for idx, image in enumerate(X_eval_raw):
            blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
            sharpened_images[idx] = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        X_eval_raw = sharpened_images
    X_eval = X_eval_raw.reshape(len(X_eval_raw), -1).astype(np.float32) / 255.0
    if pca_path is not None:
        pca = joblib.load(pca_path)
        X_eval = pca.transform(X_eval)

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CornerMLP(input_dim=input_dim or X_eval.shape[1], hidden=hidden, dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location=device_t, weights_only=True))
    model = model.to(device_t)
    loader = DataLoader(FlatDataset(X_eval, y_eval), batch_size=512, shuffle=False, num_workers=0)

    preds = []
    model.eval()
    with torch.no_grad():
        for X_batch, _ in loader:
            preds.append(model(X_batch.to(device_t)).cpu().numpy())
    y_pred = np.concatenate(preds)

    metrics = full_evaluate(y_eval, y_pred, iou_limit=None)
    metrics["inference_time_ms"] = measure_inference_time_torch(model, loader, device_t) * 1000
    metrics["by_background"] = evaluate_by_group(y_eval, y_pred, bg_ids)
    return metrics, y_eval, y_pred, image_paths, bg_ids


def evaluate_cnn(
    model_path: str | Path,
    in_channels: int,
    split: str = "test",
    scheme: str = "split_video",
    size: tuple[int, int] = (384, 216),
    dropout: float = 0.3,
    device: str | None = None,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mode = "rgb" if in_channels == 3 else "gray"
    frame_set = FrameSet(split, scheme=scheme, size=size, mode=mode, augment=False)
    image_paths = frame_set.paths
    bg_ids = frame_set.df.bg_id.to_numpy()
    y_eval = frame_set.y

    model = DocumentCNN(in_channels=in_channels, dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location=device_t, weights_only=True))
    model = model.to(device_t)
    loader = DataLoader(TorchFrameSet(frame_set), batch_size=32, shuffle=False, num_workers=0)

    preds = []
    model.eval()
    with torch.no_grad():
        for X_batch, _ in loader:
            preds.append(model(X_batch.to(device_t)).cpu().numpy())
    y_pred = np.concatenate(preds)

    metrics = full_evaluate(y_eval, y_pred, iou_limit=None)
    metrics["inference_time_ms"] = measure_inference_time_torch(model, loader, device_t) * 1000
    metrics["by_background"] = evaluate_by_group(y_eval, y_pred, bg_ids)
    return metrics, y_eval, y_pred, image_paths, bg_ids


def build_comparison_table(results: dict[str, dict], out_path: str | Path | None = None) -> pd.DataFrame:
    rows = []
    for name, metrics in results.items():
        rows.append({
            "model": name,
            "corner_err_px": round(metrics["corner_err_px"], 2),
            "corner_err_norm": round(metrics["corner_err_norm"], 5),
            "iou_mean": round(metrics["iou_mean"], 4),
            "iou_median": round(metrics["iou_median"], 4),
            "iou>=0.50": round(metrics.get("iou>=0.50", float("nan")), 4),
            "iou>=0.75": round(metrics.get("iou>=0.75", float("nan")), 4),
            "iou>=0.90": round(metrics.get("iou>=0.90", float("nan")), 4),
            "inference_time_ms": round(metrics["inference_time_ms"], 5),
        })
    df = pd.DataFrame(rows).set_index("model")
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path)
    return df
