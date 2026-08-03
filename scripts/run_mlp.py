#!/usr/bin/env python
"""Run the MLP experiments required by the comparison plan."""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from dataclasses import replace
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
from utils.subset import grouped_subset_indices
from utils.config import MLPConfig, load_yaml, save_yaml
from utils.runs import run_directory
from utils.seed import seed_everything

RESULTS = PROJECT / "results"


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
    if n_components > len(X_train):
        raise ValueError(
            f"pca_components={n_components} exceeds the {len(X_train)} selected training samples; "
            "increase train_fraction or reduce pca_components"
        )
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(_flatten(X_train))
    return pca


def _prep(X: np.ndarray, pca: PCA | None = None, sharpened: bool = False) -> np.ndarray:
    if sharpened:
        X = _sharpen_gray(X)
    flat = _flatten(X)
    return pca.transform(flat) if pca is not None else flat


def _run_one(
    cfg: MLPConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pca: PCA | None,
    out_dir: Path,
) -> dict:
    started = time.perf_counter()
    cfg = replace(
        cfg,
        pca_components=0 if pca is None else int(pca.n_components_),
        use_pca=pca is not None,
    )
    save_yaml(cfg, out_dir / "config.yaml")
    X_train_p = _prep(X_train, pca=pca, sharpened=cfg.sharpened)
    X_val_p = _prep(X_val, pca=pca, sharpened=cfg.sharpened)
    train_loader = DataLoader(FlatDataset(X_train_p, y_train), batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(FlatDataset(X_val_p, y_val), batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    model = CornerMLP(input_dim=X_train_p.shape[1], hidden=cfg.hidden, dropout=cfg.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = train(
        model,
        train_loader,
        val_loader,
        result_dir=out_dir,
        max_epochs=cfg.max_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        loss=cfg.loss,
        device=device,
    )
    preds = []
    best_model.eval()
    with torch.no_grad():
        for X_batch, _ in DataLoader(FlatDataset(X_val_p, y_val), batch_size=512, shuffle=False, num_workers=0):
            preds.append(best_model(X_batch.to(device)).cpu().numpy())
    y_pred = np.concatenate(preds)
    metrics = full_evaluate(y_val, y_pred, iou_limit=None)
    return {
        "name": cfg.name,
        "hidden": str(cfg.hidden),
        "dropout": str(cfg.dropout),
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "max_epochs": cfg.max_epochs,
        "patience": cfg.patience,
        "use_pca": pca is not None,
        "pca_components": 0 if pca is None else int(pca.n_components_),
        "sharpened": cfg.sharpened,
        "iou_mean": metrics["iou_mean"],
        "corner_err_px": metrics["corner_err_px"],
        "elapsed_s": round(time.perf_counter() - started, 3),
    }


def _run(result_dir: Path, compare_color: bool = False, compare_sharpening: bool = False) -> None:
    seed_everything(42)
    parts, _, _ = load_arrays("tab_64x36_clahe", scheme="split_video")
    X_train, y_train = parts["train"]
    X_val, y_val = parts["val"]
    pca = _fit_pca(X_train, 256)
    joblib.dump(pca, result_dir / "gray_pca256.joblib")

    rows = []
    base_cfg = MLPConfig()

    rows.append(_run_one(
        replace(base_cfg, name="mlp_gray_pca256"),
        X_train,
        y_train,
        X_val,
        y_val,
        pca,
        result_dir / "trials" / "mlp_gray_pca256",
    ))
    rows.append(_run_one(
        replace(base_cfg, name="mlp_gray_nopca"),
        X_train,
        y_train,
        X_val,
        y_val,
        None,
        result_dir / "trials" / "mlp_gray_nopca",
    ))

    if compare_sharpening:
        rows.append(_run_one(
            replace(base_cfg, name="mlp_gray_pca256_sharp", sharpened=True),
            X_train,
            y_train,
            X_val,
            y_val,
            pca,
            result_dir / "trials" / "mlp_gray_pca256_sharp",
        ))

    if compare_color:
        rgb_parts, _, _ = load_arrays("img_128x72_rgb", scheme="split_video")
        X_train_rgb, y_train_rgb = rgb_parts["train"]
        X_val_rgb, y_val_rgb = rgb_parts["val"]
        rgb_pca = _fit_pca(X_train_rgb, 256)
        gray_from_rgb_train = _rgb_to_gray(X_train_rgb)
        gray_from_rgb_val = _rgb_to_gray(X_val_rgb)
        gray_from_rgb_pca = _fit_pca(gray_from_rgb_train, 256)
        joblib.dump(rgb_pca, result_dir / "rgb_pca256.joblib")
        joblib.dump(gray_from_rgb_pca, result_dir / "gray_from_rgb_pca256.joblib")
        rows.append(_run_one(
            replace(
                base_cfg,
                name="mlp_gray_from_rgb_pca256",
                input_source="img_128x72_rgb",
                preprocessing="rgb_to_gray",
                batch_size=128,
            ),
            gray_from_rgb_train,
            y_train_rgb,
            gray_from_rgb_val,
            y_val_rgb,
            gray_from_rgb_pca,
            result_dir / "trials" / "mlp_gray_from_rgb_pca256",
        ))
        rows.append(_run_one(
            replace(
                base_cfg,
                name="mlp_rgb_pca256",
                input_source="img_128x72_rgb",
                batch_size=128,
            ),
            X_train_rgb,
            y_train_rgb,
            X_val_rgb,
            y_val_rgb,
            rgb_pca,
            result_dir / "trials" / "mlp_rgb_pca256",
        ))

    with open(result_dir / "experiments.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    gray_rows = [
        row for row in rows
        if row["name"].startswith("mlp_gray") and "from_rgb" not in row["name"]
    ]
    best = max(gray_rows, key=lambda row: row["iou_mean"])
    best_dir = result_dir / "trials" / str(best["name"])
    shutil.copy(best_dir / "best.pt", result_dir / "best.pt")
    if int(best["pca_components"]) > 0:
        shutil.copy(result_dir / "gray_pca256.joblib", result_dir / "best_pca.joblib")
    shutil.copy(best_dir / "config.yaml", result_dir / "config.yaml")


def _run_config(result_dir: Path, config_path: Path) -> None:
    cfg = MLPConfig(**load_yaml(config_path))
    seed_everything(cfg.random_seed)
    parts, groups, _ = load_arrays(cfg.input_source, scheme=cfg.scheme)
    X_train, y_train = parts["train"]
    X_val, y_val = parts["val"]
    train_indices = grouped_subset_indices(groups["train"], cfg.train_fraction, cfg.subset_seed)
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    if cfg.preprocessing == "rgb_to_gray":
        X_train = _rgb_to_gray(X_train)
        X_val = _rgb_to_gray(X_val)
    elif cfg.preprocessing != "none":
        raise ValueError(f"unsupported MLP preprocessing: {cfg.preprocessing}")
    pca_input = _sharpen_gray(X_train) if cfg.sharpened else X_train
    pca = _fit_pca(pca_input, cfg.pca_components, cfg.random_seed) if cfg.use_pca else None
    if pca is not None:
        joblib.dump(pca, result_dir / "best_pca.joblib")
    row = _run_one(
        cfg,
        X_train,
        y_train,
        X_val,
        y_val,
        pca,
        result_dir,
    )
    with open(result_dir / "experiments.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main(
    compare_color: bool = False,
    compare_sharpening: bool = False,
    config_path: Path | None = None,
    result_model_type: str = "mlp",
) -> None:
    with run_directory(RESULTS, result_model_type) as result_dir:
        if config_path is not None:
            _run_config(result_dir, config_path)
        else:
            _run(result_dir, compare_color=compare_color, compare_sharpening=compare_sharpening)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--result-model-type", default="mlp", help=argparse.SUPPRESS)
    parser.add_argument("--compare-color", action="store_true")
    parser.add_argument("--compare-sharpening", action="store_true")
    args = parser.parse_args()
    if args.config and (args.compare_color or args.compare_sharpening):
        parser.error("--config cannot be combined with comparison flags")
    main(
        compare_color=args.compare_color,
        compare_sharpening=args.compare_sharpening,
        config_path=args.config,
        result_model_type=args.result_model_type,
    )
