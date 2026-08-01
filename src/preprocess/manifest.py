"""Step 2 of 3: build the manifest, the main table everyone else uses.

One row per photo: fixed corner labels, measurements, checks, and splits.

    python scripts/build_data.py manifest    # ~10 s, writes manifest.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import (canonicalise_quad, is_convex, load_raw_quads, normalise,
                       polygon_area, rotation_deg, side_lengths)
from .paths import FRAME_H, FRAME_W, FRAMES, OUT, SEED


def build_manifest() -> pd.DataFrame:
    df = pd.read_csv(FRAMES / "metadata.csv.gz")
    raw = load_raw_quads(df)
    canon, shift = canonicalise_quad(raw)

    # Which of the 150 videos each photo came from. We split by video, so this
    # is the key column for keeping the test score honest.
    df["video_id"] = df.bg_name + "/" + df.model_name

    # ---- measurements about the page ---------------------------------------
    area = polygon_area(canon)
    sides = side_lengths(canon)
    df["rot_deg"] = rotation_deg(raw)
    df["rot_class"] = pd.cut(
        df.rot_deg, [-180, -135, -45, 45, 135, 180],
        labels=["flip", "ccw90", "upright", "cw90", "flip"], ordered=False,
    ).astype(str)
    df["doc_corner_at_slot0"] = np.array(["TL", "TR", "BR", "BL"])[shift]

    df["area_px"] = area
    df["area_frac"] = area / (FRAME_W * FRAME_H)
    df["center_x_n"] = canon[:, :, 0].mean(1) / FRAME_W
    df["center_y_n"] = canon[:, :, 1].mean(1) / FRAME_H
    df["min_side_px"] = sides.min(1)
    # Opposite edges are equal (ratio 1) when shot straight on, and drift
    # further from 1 the more the camera was tilted.
    df["edge_ratio_w"] = sides[:, 0] / sides[:, 2]
    df["edge_ratio_h"] = sides[:, 1] / sides[:, 3]
    df["perspective_skew"] = np.maximum(
        np.maximum(df.edge_ratio_w, 1 / df.edge_ratio_w),
        np.maximum(df.edge_ratio_h, 1 / df.edge_ratio_h),
    )
    df["doc_aspect"] = ((sides[:, 0] + sides[:, 2]) / (sides[:, 1] + sides[:, 3]))

    # ---- the labels ---------------------------------------------------------
    # Saved twice: in pixels to read, and as 0-1 fractions for the model.
    for i in range(4):
        df[f"c{i}_x"] = canon[:, i, 0]
        df[f"c{i}_y"] = canon[:, i, 1]
    norm = normalise(canon)
    for i in range(4):
        df[f"c{i}_x_n"] = norm[:, i, 0]
        df[f"c{i}_y_n"] = norm[:, i, 1]

    # ---- checks on the labels ----------------------------------------------
    # These all come out 0. They exist so that is proven, not assumed.
    xs, ys = canon[:, :, 0], canon[:, :, 1]
    df["flag_out_of_frame"] = (
        (xs < 0) | (xs > FRAME_W) | (ys < 0) | (ys > FRAME_H)
    ).any(1)
    df["flag_nonconvex"] = ~is_convex(canon)
    df["flag_degenerate"] = (df.min_side_px < 20) | (df.area_frac < 0.005)
    df["flag_label_bad"] = (
        df.flag_out_of_frame | df.flag_nonconvex | df.flag_degenerate
    )

    # ---- add the photo-quality numbers from step 1 -------------------------
    qc_path = OUT / "qc.csv"
    if qc_path.exists():
        qc = pd.read_csv(qc_path)
        df = df.merge(qc, on="image_path", how="left", validate="one_to_one")
        df["flag_missing"] = ~df.decodable.fillna(False).astype(bool)
        df["flag_wrong_size"] = (df.img_w != FRAME_W) | (df.img_h != FRAME_H)
        # Some videos are blurrier than others overall, so compare each photo
        # to its own video's middle photo instead of one fixed cutoff.
        med = df.groupby("video_id")["blur_doc"].transform("median")
        df["blur_rel"] = df.blur_doc / med
        df["flag_blurry"] = df.blur_rel < 0.35
    else:
        print("! qc.csv not found - run `qc` first for the image-quality columns")
        for c in ["flag_missing", "flag_wrong_size", "flag_blurry"]:
            df[c] = False

    df["usable"] = ~(df.flag_label_bad | df.flag_missing | df.flag_wrong_size)

    # ---- how far the page moved since the previous photo -------------------
    # Back to pixels first, since x and y use different scales as fractions.
    df = df.sort_values(["video_id", "frame_index"]).reset_index(drop=True)
    cen = df[["center_x_n", "center_y_n"]].to_numpy() * [FRAME_W, FRAME_H]
    step = np.r_[np.nan, np.linalg.norm(np.diff(cen, axis=0), axis=1)]
    step[df.groupby("video_id").cumcount().to_numpy() == 0] = np.nan
    df["center_step_px"] = step

    df = add_splits(df)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "manifest.csv", index=False)
    print(f"wrote {OUT / 'manifest.csv'}  ({len(df)} rows x {df.shape[1]} cols)")
    return df


def add_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Decide which photos go in train, val and test.

    Always split whole videos. Photos next to each other look nearly identical,
    so a random split would put near copies on both sides and fake the score.

    split_video  70/15/15 of the 150 videos. The normal one.
    split_doc    holds out whole documents. Harder, more honest.
    fold5        the videos in 5 groups, for cross-validation.
    """
    rng = np.random.default_rng(SEED)
    vids = df[["video_id", "bg_name", "modeltype_name", "model_id"]].drop_duplicates("video_id")

    assign = {}
    for _, grp in vids.groupby(["bg_name", "modeltype_name"]):
        ids = grp.video_id.to_numpy().copy()
        rng.shuffle(ids)
        n_test = max(1, round(0.15 * len(ids)))
        n_val = max(1, round(0.15 * len(ids)))
        for v in ids[:n_test]:
            assign[v] = "test"
        for v in ids[n_test:n_test + n_val]:
            assign[v] = "val"
        for v in ids[n_test + n_val:]:
            assign[v] = "train"
    df["split_video"] = df.video_id.map(assign)

    # One document of each of the 6 types goes to test, one more to val.
    doc_assign = {}
    for _, grp in vids.drop_duplicates("model_id").groupby("modeltype_name"):
        ids = grp.model_id.to_numpy().copy()
        rng.shuffle(ids)
        doc_assign.update({ids[0]: "test", ids[1]: "val"})
    df["split_doc"] = df.model_id.map(doc_assign).fillna("train")

    order = {v: i for i, v in enumerate(rng.permutation(vids.video_id.to_numpy()))}
    df["fold5"] = df.video_id.map(order) % 5
    return df


def subsample(df: pd.DataFrame, stride: int = 3, drop_blurry: bool = False) -> pd.DataFrame:
    """Keep every Nth photo of each video and drop the rest.

    Neighbouring photos are nearly the same, so stride=3 cuts 24,889 down to
    8,346 without losing much, and every video still runs start to finish.
    """
    d = df[df.usable].sort_values(["video_id", "frame_index"])
    keep = d.groupby("video_id").cumcount() % stride == 0
    d = d[keep]
    if drop_blurry and "flag_blurry" in d:
        d = d[~d.flag_blurry.fillna(False)]
    return d.reset_index(drop=True)
