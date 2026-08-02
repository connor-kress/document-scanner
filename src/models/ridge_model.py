"""Ridge regression with optional PCA and unsharp-mask sharpening.

Input pipeline:
    (n, 36, 64) uint8
    → optional sharpen
    → flatten to (n, 2304) float32 in [0, 1]
    → optional PCA
    → Ridge regression
    → (n, 8) normalized corner coordinates
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def _sharpen(X: np.ndarray) -> np.ndarray:
    out = np.empty_like(X)
    for i, img in enumerate(X):
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        out[i] = sharpened
    return out


class RidgePipeline:
    def __init__(
        self,
        n_components: int = 256,
        alpha: float = 1.0,
        use_pca: bool = True,
        sharpened: bool = False,
        random_seed: int = 42,
    ) -> None:
        self.n_components = n_components
        self.alpha = alpha
        self.use_pca = use_pca
        self.sharpened = sharpened
        self.random_seed = random_seed
        self.pca: Optional[PCA] = None
        self.model: Optional[Ridge] = None

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """(n, H, W) uint8 → (n, H*W) float32 in [0, 1], with optional sharpening."""
        if self.sharpened:
            X = _sharpen(X)
        return X.reshape(len(X), -1).astype(np.float32) / 255.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RidgePipeline":
        X = self._preprocess(X_train)
        if self.use_pca:
            self.pca = PCA(n_components=self.n_components, random_state=self.random_seed)
            X = self.pca.fit_transform(X)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._preprocess(X)
        if self.use_pca and self.pca is not None:
            X = self.pca.transform(X)
        preds = self.model.predict(X)
        # Ridge can predict slightly outside [0, 1]; clip to valid range.
        return np.clip(preds, 0.0, 1.0).astype(np.float32)

    def measure_inference_time(self, X: np.ndarray) -> float:
        """Mean seconds per sample, excluding preprocessing / model loading."""
        X_proc = self._preprocess(X)
        if self.use_pca and self.pca is not None:
            X_proc = self.pca.transform(X_proc)
        self.model.predict(X_proc[:2])          # warm-up
        t0 = time.perf_counter()
        self.model.predict(X_proc)
        return (time.perf_counter() - t0) / len(X_proc)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"saved Ridge pipeline → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "RidgePipeline":
        return joblib.load(path)
