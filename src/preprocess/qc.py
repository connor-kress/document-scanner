"""Step 1 of 3: open every photo once and check it. Measure conditions such as blur

    python scripts/build_data.py qc writes data/processed/qc.csv
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from .geometry import load_raw_quads
from .paths import FRAMES, OUT


def _qc_one(args: tuple[str, tuple[float, ...]]) -> dict:
    """Check one photo. Runs in a separate process so many go at once."""
    rel, bbox = args
    rec: dict = {"image_path": rel}
    path = FRAMES / rel
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        rec["decodable"] = False
        return rec

    h, w = img.shape[:2]
    rec.update(decodable=True, img_w=w, img_h=h, file_bytes=path.stat().st_size)

    # Laplacian finds edges, and sharp photos have more of them. Higher = sharper.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rec["blur_full"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    rec["gray_mean"] = float(gray.mean())
    rec["gray_std"] = float(gray.std())
    rec["clip_hi"] = float((gray >= 250).mean())
    rec["clip_lo"] = float((gray <= 5).mean())
    b, g, r = (float(img[:, :, i].mean()) for i in range(3))
    rec.update(mean_b=b, mean_g=g, mean_r=r)

    # Also measure blur on just the page. That is the number that matters --
    # a sharp desk with a smeared page is still useless to us.
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    if x1 - x0 > 8 and y1 - y0 > 8:
        crop = gray[y0:y1, x0:x1]
        rec["blur_doc"] = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        rec["doc_mean"] = float(crop.mean())
        rec["doc_std"] = float(crop.std())
    return rec


def run_qc(workers: int, limit: int | None = None) -> pd.DataFrame:
    """Check all 24,889 photos and save the results to qc.csv."""
    meta = pd.read_csv(FRAMES / "metadata.csv.gz")
    if limit:
        meta = meta.head(limit)
    # Box around the page: smallest and largest x and y of its 4 corners.
    quads = load_raw_quads(meta)
    boxes = np.stack(
        [quads[:, :, 0].min(1), quads[:, :, 1].min(1),
         quads[:, :, 0].max(1), quads[:, :, 1].max(1)], axis=1
    )
    jobs = [(p.replace("/", os.sep), tuple(bb)) for p, bb in zip(meta.image_path, boxes)]

    recs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rec in enumerate(pool.map(_qc_one, jobs, chunksize=64), 1):
            recs.append(rec)
            if i % 2000 == 0:
                print(f"  qc {i}/{len(jobs)}", flush=True)

    qc = pd.DataFrame(recs)
    qc["image_path"] = qc["image_path"].str.replace(os.sep, "/", regex=False)
    OUT.mkdir(parents=True, exist_ok=True)
    qc.to_csv(OUT / "qc.csv", index=False)
    print(f"wrote {OUT / 'qc.csv'}  ({len(qc)} rows)")
    return qc
