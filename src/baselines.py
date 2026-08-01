"""Two simple models the CNN has to beat.

  constant  guess the same average quad for every photo..
  ridge     ridge regression on tiny 64x36 grayscale photos.

  Scored with metrics module

    python scripts/run_baseline.py
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from dataset import load_arrays
from metrics import constant_baseline, evaluate, format_row


def fit_ridge(X_train, y_train, X_eval, alpha: float = 100.0) -> np.ndarray:
    """Squash each photo into one long row of numbers, scale to 0-1, then fit
    ridge regression and predict."""
    def flat(a):
        return a.reshape(len(a), -1).astype(np.float32) / 255.0
    return Ridge(alpha=alpha).fit(flat(X_train), y_train).predict(flat(X_eval))


def main(array: str = "tab_64x36_clahe", scheme: str = "split_video",
         alphas=(1.0, 10.0, 100.0), split: str = "test") -> dict:
    parts, _, _ = load_arrays(array, scheme=scheme)
    X_tr, y_tr = parts["train"]
    X_ev, y_ev = parts[split]
    print(f"{array} | {scheme} | train {len(X_tr)}  {split} {len(X_ev)}\n")

    results = {"constant (train mean)": evaluate(y_ev, constant_baseline(y_tr, len(y_ev)),
                                                 iou_limit=400)}
    for a in alphas:
        results[f"ridge(alpha={a:g})"] = evaluate(
            y_ev, fit_ridge(X_tr, y_tr, X_ev, a), iou_limit=400)

    for name, m in results.items():
        print(format_row(name, m))
    return results


if __name__ == "__main__":
    main()
