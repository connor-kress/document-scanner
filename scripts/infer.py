#!/usr/bin/env python
"""Predict document corner pixel locations for one image."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
MODELS = PROJECT / "models"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from augment import to_tensor_chw
from models.cnn import DocumentCNN
from models.mlp import CornerMLP
from models.ridge_model import RidgePipeline
from scripts.download_models import download_models, models_ready
from utils.config import load_yaml
from utils.runs import latest_run


MODEL_TYPES = ("ridge", "mlp", "cnn-gray", "cnn-rgb")
RELEASE_DIRS = {
    "ridge": "ridge",
    "mlp": "mlp",
    "cnn-gray": "cnn_gray",
    "cnn-rgb": "cnn_rgb",
}
RESULT_DIRS = RELEASE_DIRS


@dataclass(frozen=True)
class Artifacts:
    weights: Path
    config: Path | None = None
    pca: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="document image to process")
    parser.add_argument("--model-type", choices=MODEL_TYPES, help="model family (default: cnn-rgb)")
    parser.add_argument("--weights", type=Path, help="custom model weights")
    parser.add_argument("--pca", type=Path, help="custom MLP PCA joblib file")
    parser.add_argument("--config", type=Path, help="custom MLP/CNN YAML configuration")
    parser.add_argument("--no-show", action="store_true", help="do not display the prediction figure")
    parser.add_argument("--output", type=Path, help="save the prediction figure to this path")
    parser.add_argument(
        "--latest-trained",
        action="store_true",
        help="use the newest completed run in results/ instead of released weights",
    )
    return parser


def validate_selection(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    custom = args.weights is not None or args.pca is not None or args.config is not None
    if (custom or args.latest_trained) and args.model_type is None:
        parser.error("--model-type is required with custom artifacts or --latest-trained")
    if custom and args.latest_trained:
        parser.error("--latest-trained cannot be combined with --weights, --pca, or --config")
    if args.pca is not None and args.model_type != "mlp":
        parser.error("--pca can only be used with --model-type mlp")
    if args.config is not None and args.model_type == "ridge":
        parser.error("--config is only used by MLP and CNN models")
    if custom and args.weights is None:
        parser.error("--weights is required when supplying custom artifacts")
    return args.model_type or "cnn-rgb"


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _config_for_weights(weights: Path, config: Path | None) -> Path:
    return _require_file(config or weights.with_name("config.yaml"), "model config")


def _mlp_pca(config: Path, candidate: Path | None) -> Path | None:
    cfg = load_yaml(config)
    if not cfg.get("use_pca", False):
        if candidate is not None:
            raise ValueError("--pca was supplied, but the MLP config has use_pca: false")
        return None
    if candidate is None:
        raise FileNotFoundError("the selected MLP config requires a PCA joblib file")
    return _require_file(candidate, "PCA model")


def released_artifacts(model_type: str) -> Artifacts:
    directory = MODELS / RELEASE_DIRS[model_type]
    weight_name = "model.joblib" if model_type == "ridge" else "model.pt"
    config = None if model_type == "ridge" else directory / "config.yaml"
    pca = directory / "pca.joblib" if model_type == "mlp" else None
    required = [directory / weight_name]
    if config is not None:
        required.append(config)
    if pca is not None:
        required.append(pca)
    if not all(path.is_file() for path in required):
        download_models(MODELS, force=MODELS.exists() and not models_ready(MODELS))
    return Artifacts(
        _require_file(directory / weight_name, "model weights"),
        _require_file(config, "model config") if config is not None else None,
        _require_file(pca, "PCA model") if pca is not None else None,
    )


def custom_artifacts(model_type: str, weights: Path, config: Path | None, pca: Path | None) -> Artifacts:
    weights = _require_file(weights, "model weights")
    if model_type == "ridge":
        return Artifacts(weights)
    config = _config_for_weights(weights, config)
    if model_type == "mlp":
        return Artifacts(weights, config, _mlp_pca(config, pca))
    return Artifacts(weights, config)


def trained_artifacts(model_type: str) -> Artifacts:
    result_type = RESULT_DIRS[model_type]
    weight_name = "best_pipeline.joblib" if model_type == "ridge" else "best.pt"
    required = (weight_name,) if model_type == "ridge" else (weight_name, "config.yaml")
    directory = latest_run(RESULTS, result_type, required=required)
    if not (directory / "COMPLETE").is_file():
        raise FileNotFoundError(f"latest {result_type!r} result is not a completed training run: {directory}")
    weights = _require_file(directory / weight_name, "trained model weights")
    if model_type == "ridge":
        return Artifacts(weights)
    config = _require_file(directory / "config.yaml", "trained model config")
    if model_type == "mlp":
        candidate = directory / "best_pca.joblib"
        return Artifacts(weights, config, _mlp_pca(config, candidate if candidate.is_file() else None))
    return Artifacts(weights, config)


def resolve_artifacts(model_type: str, args: argparse.Namespace) -> Artifacts:
    if args.latest_trained:
        return trained_artifacts(model_type)
    if args.weights is not None:
        return custom_artifacts(model_type, args.weights, args.config, args.pca)
    return released_artifacts(model_type)


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def preprocess_clahe(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def preprocess_mlp(image: np.ndarray, config: dict, pca_path: Path | None) -> np.ndarray:
    source = config.get("input_source", "tab_64x36_clahe")
    if source == "tab_64x36_clahe":
        prepared = preprocess_clahe(image)
    elif source == "img_128x72_rgb":
        resized = cv2.resize(image, (128, 72), interpolation=cv2.INTER_AREA)
        prepared = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"unsupported MLP input_source: {source}")

    preprocessing = config.get("preprocessing", "none")
    if preprocessing == "rgb_to_gray":
        if prepared.ndim != 3:
            raise ValueError("rgb_to_gray preprocessing requires an RGB input source")
        prepared = cv2.cvtColor(prepared, cv2.COLOR_RGB2GRAY)
    elif preprocessing != "none":
        raise ValueError(f"unsupported MLP preprocessing: {preprocessing}")
    if config.get("sharpened", False):
        blurred = cv2.GaussianBlur(prepared, (0, 0), sigmaX=1.5)
        prepared = cv2.addWeighted(prepared, 1.5, blurred, -0.5, 0)

    values = prepared.reshape(1, -1).astype(np.float32) / 255.0
    if pca_path is not None:
        values = joblib.load(pca_path).transform(values).astype(np.float32)
    return values


def preprocess_cnn(image: np.ndarray, model_type: str, config: dict) -> np.ndarray:
    expected_mode = "gray" if model_type == "cnn-gray" else "rgb"
    mode = config.get("color_mode", expected_mode)
    if mode != expected_mode:
        raise ValueError(f"config color_mode {mode!r} does not match --model-type {model_type}")
    size = tuple(config.get("image_size", (384, 216)))
    if len(size) != 2 or any(int(value) <= 0 for value in size):
        raise ValueError(f"invalid CNN image_size: {size}")
    resized = cv2.resize(image, tuple(map(int, size)), interpolation=cv2.INTER_AREA)
    prepared = (
        cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if mode == "gray"
        else cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    )
    return to_tensor_chw(prepared)[None].astype(np.float32)


def predict(image: np.ndarray, model_type: str, artifacts: Artifacts) -> np.ndarray:
    if model_type == "ridge":
        model = RidgePipeline.load(artifacts.weights)
        return model.predict(preprocess_clahe(image)[None])[0]

    if artifacts.config is None:
        raise ValueError(f"{model_type} inference requires a config")
    config = load_yaml(artifacts.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "mlp":
        values = preprocess_mlp(image, config, artifacts.pca)
        model = CornerMLP(
            input_dim=values.shape[1],
            hidden=list(config["hidden"]),
            dropout=list(config["dropout"]),
        )
    else:
        values = preprocess_cnn(image, model_type, config)
        model = DocumentCNN(
            in_channels=1 if model_type == "cnn-gray" else 3,
            dropout=float(config.get("dropout", 0.3)),
        )
    model.load_state_dict(torch.load(artifacts.weights, map_location=device, weights_only=True))
    model.to(device).eval()
    with torch.no_grad():
        tensor = torch.from_numpy(values).to(device)
        return model(tensor)[0].cpu().numpy()


def pixel_corners(prediction: np.ndarray, width: int, height: int) -> np.ndarray:
    corners = np.asarray(prediction, dtype=np.float64).reshape(4, 2).copy()
    corners *= np.array([width, height], dtype=np.float64)
    return corners


def display_corners(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    show: bool = True,
    output: Path | None = None,
) -> None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    closed = np.vstack([corners, corners[0]])
    color = "#2a78d6"
    height, width = image.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes((0, 0, 1, 1))
    fig.canvas.manager.set_window_title("Document Corner Prediction")
    ax.imshow(rgb)
    ax.fill(corners[:, 0], corners[:, 1], color=color, alpha=0.15)
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2, linestyle=":")
    ax.scatter(corners[:, 0], corners[:, 1], color=color, s=35)
    ax.axis("off")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    model_type = validate_selection(parser, args)
    try:
        image = load_image(args.image)
        artifacts = resolve_artifacts(model_type, args)
        prediction = predict(image, model_type, artifacts)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    height, width = image.shape[:2]
    corners = pixel_corners(prediction, width, height)
    for index, (x, y) in enumerate(corners):
        print(f"corner_{index}: ({x:.2f}, {y:.2f})")
    if not args.no_show or args.output is not None:
        display_corners(image, corners, show=not args.no_show, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
