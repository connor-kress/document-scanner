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

RESULTS = PROJECT / "results"


def main() -> None:
    results = {}
    ref_paths = None

    ridge_metrics, ridge_y, ridge_pred, ridge_paths, ridge_bg = evaluate_ridge(RESULTS / "ridge" / "best_pipeline.joblib")
    results["Ridge"] = ridge_metrics
    ref_paths = ridge_paths

    mlp_cfg = load_yaml(RESULTS / "mlp" / "config.yaml")
    mlp_metrics, mlp_y, mlp_pred, mlp_paths, mlp_bg = evaluate_mlp(
        RESULTS / "mlp" / "best.pt",
        input_dim=int(mlp_cfg["pca_components"] or 36 * 64),
        hidden=list(mlp_cfg["hidden"]),
        dropout=list(mlp_cfg["dropout"]),
        pca_path=(RESULTS / "mlp" / "best_pca.joblib") if mlp_cfg["use_pca"] else None,
    )
    assert_same_split(ref_paths, mlp_paths, "MLP")
    results["MLP"] = mlp_metrics

    gray_metrics, gray_y, gray_pred, gray_paths, gray_bg = evaluate_cnn(RESULTS / "cnn_gray" / "best.pt", in_channels=1)
    assert_same_split(ref_paths, gray_paths, "Grayscale CNN")
    results["Grayscale CNN"] = gray_metrics

    rgb_metrics, rgb_y, rgb_pred, rgb_paths, rgb_bg = evaluate_cnn(RESULTS / "cnn_rgb" / "best.pt", in_channels=3)
    assert_same_split(ref_paths, rgb_paths, "RGB CNN")
    results["RGB CNN"] = rgb_metrics

    table = build_comparison_table(results, RESULTS / "final_comparison.csv")
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
    pd.DataFrame(bg_rows).to_csv(RESULTS / "by_background.csv", index=False)


if __name__ == "__main__":
    main()