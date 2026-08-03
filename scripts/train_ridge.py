#!/usr/bin/env python
"""Train the Ridge models required by the comparison plan."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dataset import load_arrays
from evaluation.metrics import full_evaluate
from models.ridge_model import RidgePipeline
from utils.config import RidgeConfig, save_yaml
from utils.runs import run_directory
from utils.seed import seed_everything

RESULTS = PROJECT / "results"
PCA_GRID = [64, 128, 256, 512]
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def _run(result_dir: Path) -> None:
    seed_everything(42)
    parts, _, _ = load_arrays("tab_64x36_clahe", scheme="split_video")
    X_train, y_train = parts["train"]
    X_val, y_val = parts["val"]
    rows = []

    for alpha in ALPHA_GRID:
        started = time.perf_counter()
        model = RidgePipeline(alpha=alpha, use_pca=False)
        model.fit(X_train, y_train)
        metrics = full_evaluate(y_val, model.predict(X_val), iou_limit=None)
        rows.append({
            "name": f"ridge_gray_nopca_a{alpha:g}",
            "use_pca": False,
            "pca_components": 0,
            "alpha": alpha,
            "sharpened": False,
            "iou_mean": metrics["iou_mean"],
            "corner_err_px": metrics["corner_err_px"],
            "elapsed_s": round(time.perf_counter() - started, 3),
        })

    for n_components in PCA_GRID:
        for alpha in ALPHA_GRID:
            started = time.perf_counter()
            model = RidgePipeline(n_components=n_components, alpha=alpha, use_pca=True)
            model.fit(X_train, y_train)
            metrics = full_evaluate(y_val, model.predict(X_val), iou_limit=None)
            rows.append({
                "name": f"ridge_gray_pca{n_components}_a{alpha:g}",
                "use_pca": True,
                "pca_components": n_components,
                "alpha": alpha,
                "sharpened": False,
                "iou_mean": metrics["iou_mean"],
                "corner_err_px": metrics["corner_err_px"],
                "elapsed_s": round(time.perf_counter() - started, 3),
            })

    stage1_best = max(rows, key=lambda row: row["iou_mean"])
    started = time.perf_counter()
    sharp_model = RidgePipeline(
        n_components=int(stage1_best["pca_components"]),
        alpha=float(stage1_best["alpha"]),
        use_pca=bool(stage1_best["use_pca"]),
        sharpened=True,
    )
    sharp_model.fit(X_train, y_train)
    sharp_metrics = full_evaluate(y_val, sharp_model.predict(X_val), iou_limit=None)
    rows.append({
        "name": f"{stage1_best['name']}_sharp",
        "use_pca": bool(stage1_best["use_pca"]),
        "pca_components": int(stage1_best["pca_components"]),
        "alpha": float(stage1_best["alpha"]),
        "sharpened": True,
        "iou_mean": sharp_metrics["iou_mean"],
        "corner_err_px": sharp_metrics["corner_err_px"],
        "elapsed_s": round(time.perf_counter() - started, 3),
    })

    with open(result_dir / "experiments.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: row["iou_mean"])
    final_model = RidgePipeline(
        n_components=int(best["pca_components"]),
        alpha=float(best["alpha"]),
        use_pca=bool(best["use_pca"]),
        sharpened=bool(best["sharpened"]),
    )
    final_model.fit(X_train, y_train)
    final_model.save(result_dir / "best_pipeline.joblib")
    save_yaml(RidgeConfig(
        name=str(best["name"]),
        pca_components=int(best["pca_components"]),
        alpha=float(best["alpha"]),
        use_pca=bool(best["use_pca"]),
        sharpened=bool(best["sharpened"]),
    ), result_dir / "config.yaml")


def main() -> None:
    with run_directory(RESULTS, "ridge") as result_dir:
        _run(result_dir)


if __name__ == "__main__":
    main()
