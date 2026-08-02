#!/usr/bin/env python
"""Run the MLP experiments required by the comparison plan."""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dataset import load_arrays
from evaluation.metrics import full_evaluate
from models.mlp import CornerMLP, FlatDataset
from training.trainer import train
from utils.config import MLPConfig, save_yaml
from utils.seed import seed_everything

RESULTS = PROJECT / "results" / "mlp"


def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1).astype(np.float32) / 255.0


def _sharpen_gray(X: np.ndarray) -> np.ndarray:
    out = np.empty_like(X)
    for idx, image in enumerate(X):
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
        out[idx] = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return out


def _rgb_to_gray(X: np.ndarray) -> np.ndarray:
    out = np.empty((len(X), X.shape[1], X.shape[2]), dtype=np.uint8)
    for idx, image in enumerate(X):
        out[idx] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return out


def _fit_pca(X_train: np.ndarray, n_components: int, seed: int = 42) -> PCA:
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(_flatten(X_train))
    return pca


def _prep(X: np.ndarray, pca: PCA | None = None, sharpened: bool = False) -> np.ndarray:
    if sharpened:
        X = _sharpen_gray(X)
    flat = _flatten(X)
    return pca.transform(flat) if pca is not None else flat


def _run_one(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden: list[int],
    dropout: list[float],
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    pca: PCA | None,
    sharpened: bool,
    out_dir: Path,
) -> dict:
    X_train_p = _prep(X_train, pca=pca, sharpened=sharpened)
    X_val_p = _prep(X_val, pca=pca, sharpened=sharpened)
    train_loader = DataLoader(FlatDataset(X_train_p, y_train), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(FlatDataset(X_val_p, y_val), batch_size=batch_size, shuffle=False, num_workers=0)
    model = CornerMLP(input_dim=X_train_p.shape[1], hidden=hidden, dropout=dropout)
    train(
        model,
        train_loader,
        val_loader,
        result_dir=out_dir,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = CornerMLP(input_dim=X_train_p.shape[1], hidden=hidden, dropout=dropout).to(device)
    best_model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=True))
    preds = []
    best_model.eval()
    with torch.no_grad():
        for X_batch, _ in DataLoader(FlatDataset(X_val_p, y_val), batch_size=512, shuffle=False, num_workers=0):
            preds.append(best_model(X_batch.to(device)).cpu().numpy())
    y_pred = np.concatenate(preds)
    metrics = full_evaluate(y_val, y_pred, iou_limit=None)
    return {
        "name": name,
        "hidden": str(hidden),
        "dropout": str(dropout),
        "learning_rate": learning_rate,
        "use_pca": pca is not None,
        "pca_components": 0 if pca is None else int(pca.n_components_),
        "sharpened": sharpened,
        "iou_mean": metrics["iou_mean"],
        "corner_err_px": metrics["corner_err_px"],
    }


def main(compare_color: bool = False, compare_sharpening: bool = False) -> None:
    seed_everything(42)
    RESULTS.mkdir(parents=True, exist_ok=True)
    parts, _, _ = load_arrays("tab_64x36_clahe", scheme="split_video")
    X_train, y_train = parts["train"]
    X_val, y_val = parts["val"]
    pca = _fit_pca(X_train, 256)
    joblib.dump(pca, RESULTS / "gray_pca256.joblib")

    rows = []
    base_settings = {
        "hidden": [256, 128],
        "dropout": [0.3, 0.2],
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "max_epochs": 100,
        "patience": 12,
    }

    rows.append(_run_one(
        "mlp_gray_pca256",
        X_train,
        y_train,
        X_val,
        y_val,
        pca=pca,
        sharpened=False,
        out_dir=RESULTS / "mlp_gray_pca256",
        **base_settings,
    ))
    rows.append(_run_one(
        "mlp_gray_nopca",
        X_train,
        y_train,
        X_val,
        y_val,
        pca=None,
        sharpened=False,
        out_dir=RESULTS / "mlp_gray_nopca",
        **base_settings,
    ))

    if compare_sharpening:
        rows.append(_run_one(
            "mlp_gray_pca256_sharp",
            X_train,
            y_train,
            X_val,
            y_val,
            pca=pca,
            sharpened=True,
            out_dir=RESULTS / "mlp_gray_pca256_sharp",
            **base_settings,
        ))

    if compare_color:
        rgb_parts, _, _ = load_arrays("img_128x72_rgb", scheme="split_video")
        X_train_rgb, y_train_rgb = rgb_parts["train"]
        X_val_rgb, y_val_rgb = rgb_parts["val"]
        rgb_pca = _fit_pca(X_train_rgb, 256)
        gray_from_rgb_train = _rgb_to_gray(X_train_rgb)
        gray_from_rgb_val = _rgb_to_gray(X_val_rgb)
        gray_from_rgb_pca = _fit_pca(gray_from_rgb_train, 256)
        joblib.dump(rgb_pca, RESULTS / "rgb_pca256.joblib")
        joblib.dump(gray_from_rgb_pca, RESULTS / "gray_from_rgb_pca256.joblib")
        rows.append(_run_one(
            "mlp_gray_from_rgb_pca256",
            gray_from_rgb_train,
            y_train_rgb,
            gray_from_rgb_val,
            y_val_rgb,
            hidden=[256, 128],
            dropout=[0.3, 0.2],
            learning_rate=3e-4,
            weight_decay=1e-4,
            batch_size=128,
            max_epochs=100,
            patience=12,
            pca=gray_from_rgb_pca,
            sharpened=False,
            out_dir=RESULTS / "mlp_gray_from_rgb_pca256",
        ))
        rows.append(_run_one(
            "mlp_rgb_pca256",
            X_train_rgb,
            y_train_rgb,
            X_val_rgb,
            y_val_rgb,
            hidden=[256, 128],
            dropout=[0.3, 0.2],
            learning_rate=3e-4,
            weight_decay=1e-4,
            batch_size=128,
            max_epochs=100,
            patience=12,
            pca=rgb_pca,
            sharpened=False,
            out_dir=RESULTS / "mlp_rgb_pca256",
        ))

    with open(RESULTS / "experiments.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    gray_rows = [
        row for row in rows
        if row["name"].startswith("mlp_gray") and "from_rgb" not in row["name"]
    ]
    best = max(gray_rows, key=lambda row: row["iou_mean"])
    best_dir = RESULTS / str(best["name"])
    shutil.copy(best_dir / "best.pt", RESULTS / "best.pt")
    if int(best["pca_components"]) > 0:
        shutil.copy(RESULTS / "gray_pca256.joblib", RESULTS / "best_pca.joblib")
    save_yaml(MLPConfig(
        name=str(best["name"]),
        pca_components=int(best["pca_components"]),
        use_pca=bool(best["use_pca"]),
        hidden=[256, 128],
        dropout=[0.3, 0.2],
        learning_rate=3e-4,
        weight_decay=1e-4,
    ), RESULTS / "config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-color", action="store_true")
    parser.add_argument("--compare-sharpening", action="store_true")
    args = parser.parse_args()
    main(compare_color=args.compare_color, compare_sharpening=args.compare_sharpening)