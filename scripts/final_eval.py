#!/usr/bin/env python
"""Evaluate the selected Ridge, MLP, grayscale CNN, and RGB CNN once on test."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from evaluation.evaluate import assert_same_split, build_comparison_table, evaluate_cnn, evaluate_mlp, evaluate_ridge
from utils.config import load_yaml
from utils.runs import latest_run, run_directory

RESULTS = PROJECT / "results"


def main() -> None:
    ridge_dir = latest_run(RESULTS, "ridge", required=("best_pipeline.joblib",))
    mlp_dir = latest_run(RESULTS, "mlp", required=("best.pt", "config.yaml"))
    gray_dir = latest_run(RESULTS, "cnn_gray", required=("best.pt", "config.yaml"))
    rgb_dir = latest_run(RESULTS, "cnn_rgb", required=("best.pt", "config.yaml"))
    print(
        "using runs:\n"
        f"  ridge: {ridge_dir}\n"
        f"  mlp: {mlp_dir}\n"
        f"  cnn_gray: {gray_dir}\n"
        f"  cnn_rgb: {rgb_dir}"
    )

    with run_directory(RESULTS, "evaluation") as evaluation_dir:
        results = {}
        ridge_cfg_path = ridge_dir / "config.yaml"
        ridge_cfg = load_yaml(ridge_cfg_path) if ridge_cfg_path.is_file() else {}
        ridge_metrics, ridge_y, ridge_pred, ridge_paths, ridge_bg = evaluate_ridge(
            ridge_dir / "best_pipeline.joblib",
            scheme=ridge_cfg.get("scheme", "split_video"),
            array_name=ridge_cfg.get("input_source", "tab_64x36_clahe"),
        )
        results["Ridge"] = ridge_metrics
        ref_paths = ridge_paths

        mlp_cfg = load_yaml(mlp_dir / "config.yaml")
        mlp_metrics, mlp_y, mlp_pred, mlp_paths, mlp_bg = evaluate_mlp(
            mlp_dir / "best.pt",
            input_dim=int(mlp_cfg.get("input_dim", 0) or mlp_cfg["pca_components"] or 0) or None,
            hidden=list(mlp_cfg["hidden"]),
            dropout=list(mlp_cfg["dropout"]),
            scheme=mlp_cfg.get("scheme", "split_video"),
            array_name=mlp_cfg.get("input_source", "tab_64x36_clahe"),
            pca_path=(mlp_dir / "best_pca.joblib") if mlp_cfg["use_pca"] else None,
            sharpened=mlp_cfg.get("sharpened", False),
            preprocessing=mlp_cfg.get("preprocessing", "none"),
        )
        assert_same_split(ref_paths, mlp_paths, "MLP")
        results["MLP"] = mlp_metrics

        gray_cfg = load_yaml(gray_dir / "config.yaml")
        gray_metrics, gray_y, gray_pred, gray_paths, gray_bg = evaluate_cnn(
            gray_dir / "best.pt",
            in_channels=1,
            scheme=gray_cfg.get("scheme", "split_video"),
            size=tuple(gray_cfg.get("image_size", (384, 216))),
            dropout=float(gray_cfg.get("dropout", 0.3)),
        )
        assert_same_split(ref_paths, gray_paths, "Grayscale CNN")
        results["Grayscale CNN"] = gray_metrics

        rgb_cfg = load_yaml(rgb_dir / "config.yaml")
        rgb_metrics, rgb_y, rgb_pred, rgb_paths, rgb_bg = evaluate_cnn(
            rgb_dir / "best.pt",
            in_channels=3,
            scheme=rgb_cfg.get("scheme", "split_video"),
            size=tuple(rgb_cfg.get("image_size", (384, 216))),
            dropout=float(rgb_cfg.get("dropout", 0.3)),
        )
        assert_same_split(ref_paths, rgb_paths, "RGB CNN")
        results["RGB CNN"] = rgb_metrics

        table = build_comparison_table(results, evaluation_dir / "final_comparison.csv")
        print(table.to_string())

        bg_rows = []
        for model_name, metrics in results.items():
            for bg_id, bg_metrics in metrics["by_background"].items():
                bg_rows.append({
                    "model": model_name,
                    "background": bg_id,
                    "iou_mean": bg_metrics["iou_mean"],
                    "corner_err_px": bg_metrics["corner_err_px"],
                })
        pd.DataFrame(bg_rows).to_csv(evaluation_dir / "by_background.csv", index=False)


if __name__ == "__main__":
    main()
