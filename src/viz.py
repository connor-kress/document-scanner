"""Makes the charts in reports/figures/.

Each one shows the evidence behind a decision made in preprocess/, so they are
worth putting in the presentation.

    python scripts/make_figures.py
"""

from __future__ import annotations

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocess import (CANON_COLS, FIGURES, FRAMES, FRAME_H, FRAME_W, OUT,
                         dewarp, load_raw_quads, subsample)

FIG = FIGURES

# One colour per background. These 5 were checked to stay distinguishable for
# colour-blind readers. Every chart also labels its categories on an axis, so
# nobody has to rely on colour alone.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlesize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#c3c2b7",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": True,
    "axes.axisbelow": True, "legend.frameon": False,
})


def _title(ax, t, sub=None):
    """Put a bold title and a grey subtitle above a chart, aligned left."""
    ax.text(0, 1.12 if sub else 1.03, t, transform=ax.transAxes, fontsize=10,
            fontweight="600", color=INK, va="bottom")
    if sub:
        ax.text(0, 1.025, sub, transform=ax.transAxes, fontsize=8, color=MUTED,
                va="bottom")


def fig_overview(df: pd.DataFrame):
    bgs = sorted(df.bg_name.unique())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    # 1 -- how long each video is
    ax = axes[0, 0]
    n = df.groupby("video_id").size()
    ax.hist(n, bins=26, color=SERIES[0], edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(n.median(), color=INK, lw=2, ls="--")
    ax.text(n.median() + 5, ax.get_ylim()[1] * 0.9, f"median {int(n.median())}",
            color=INK, fontsize=8)
    _title(ax, "Frames per clip", "150 clips, 24,889 frames")
    ax.set_xlabel("frames in clip"); ax.set_ylabel("clips")

    # 2 -- the big finding: background04 was filmed sideways
    ax = axes[0, 1]
    for i, bg in enumerate(bgs):
        ax.hist(df[df.bg_name == bg].rot_deg, bins=90, range=(-120, 120),
                color=SERIES[i], label=bg, alpha=0.9)
    ax.set_yscale("log")
    _title(ax, "Page rotation in frame is a per-background constant",
           "background04 is shot 90° turned — 100% of its frames")
    ax.set_xlabel("rotation of page top edge (deg, 0 = upright)")
    ax.set_ylabel("frames (log)"); ax.legend(fontsize=7.5, loc="upper left")
    for x, lab in [(-88, "ccw90"), (0, "upright"), (88, "cw90")]:
        ax.annotate(lab, (x, ax.get_ylim()[0] * 2.2), ha="center", fontsize=8,
                    color=INK2)

    # 3 -- some videos are much blurrier than others, so we can't use one
    #      blur cutoff for all of them
    ax = axes[0, 2]
    data = [df[df.bg_name == bg].blur_doc.dropna() for bg in bgs]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color=INK, lw=2),
                    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    for patch, c in zip(bp["boxes"], SERIES):
        patch.set_facecolor(c); patch.set_edgecolor(SURFACE); patch.set_linewidth(2)
    ax.set_yscale("log")
    ax.set_xticklabels([b.replace("background", "bg") for b in bgs])
    _title(ax, "Page sharpness differs 7× between backgrounds",
           "an absolute blur cutoff would delete background03")
    ax.set_ylabel("Laplacian variance on page crop (log)")

    # 4 -- how big the page is in the photo
    ax = axes[1, 0]
    ax.hist(df.area_frac * 100, bins=60, color=SERIES[0], edgecolor=SURFACE,
            linewidth=0.6)
    _title(ax, "Page covers a small, consistent slice of the frame",
           f"median {df.area_frac.median() * 100:.1f}% of 1920×1080")
    ax.set_xlabel("page area (% of frame)"); ax.set_ylabel("frames")

    # 5 -- how tilted the camera was
    ax = axes[1, 1]
    ax.hist(df.perspective_skew, bins=60,
            range=(1, float(df.perspective_skew.quantile(0.999))),
            color=SERIES[1], edgecolor=SURFACE, linewidth=0.6)
    _title(ax, "Perspective distortion",
           "opposite-edge length ratio; 1.0 = fronto-parallel")
    ax.set_xlabel("max opposite-edge ratio"); ax.set_ylabel("frames")

    # 6 -- why we keep only every 3rd photo: they barely change
    ax = axes[1, 2]
    step = df.center_step_px.dropna()
    ax.hist(step, bins=70, range=(0, 60), color=SERIES[2], edgecolor=SURFACE,
            linewidth=0.6)
    _title(ax, "Consecutive frames are near-duplicates",
           f"{(step < 5).mean() * 100:.0f}% of frames move the page <5 px "
           f"from the previous one; median {step.median():.1f} px")
    ax.set_xlabel("page-centre shift from previous frame (px)")
    ax.set_ylabel("frames")

    fig.tight_layout()
    fig.savefig(FIG / "01_overview.png", dpi=140)
    plt.close(fig)


def fig_targets(df: pd.DataFrame):
    """The most important chart: what reordering the corners fixes.

    Top row is the original labels -- notice each corner has two separate
    blobs, because background04 was filmed sideways. Bottom row is after the
    fix: one blob each, which is something a model can actually learn.
    """
    raw = load_raw_quads(df)
    canon = df[CANON_COLS].to_numpy().reshape(-1, 4, 2)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2))
    names_raw = ["doc TL", "doc TR", "doc BR", "doc BL"]
    names_can = ["slot 0", "slot 1", "slot 2", "slot 3"]

    for row, (q, names, tag) in enumerate(
            [(raw, names_raw, "RAW  (document-relative)"),
             (canon, names_can, "CANONICAL  (image-relative)")]):
        for i in range(4):
            ax = axes[row, i]
            # Log scale, or the smaller blob (17% of photos) is too faint to see.
            ax.hexbin(q[:, i, 0], q[:, i, 1], gridsize=44, cmap="Blues",
                      extent=(0, FRAME_W, 0, FRAME_H), mincnt=1,
                      bins="log", linewidths=0)
            ax.set_xlim(0, FRAME_W); ax.set_ylim(FRAME_H, 0)
            ax.set_aspect("equal"); ax.grid(False)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{names[i]}", loc="left", fontsize=9,
                         fontweight="600", color=INK, pad=5)
            if i == 0:
                ax.set_ylabel(tag, fontsize=9.5, color=INK, fontweight="600",
                              labelpad=8)

    fig.suptitle("Corner target density over the frame — raw labels are multi-modal, "
                 "canonical labels are unimodal", x=0.012, y=0.985, ha="left",
                 fontsize=12, fontweight="600", color=INK)
    fig.text(0.012, 0.945, "Each panel is one of the 8 regression outputs, plotted over "
             "the 1920×1080 frame (log counts). Top row: the raw label scatters "
             "background04's corners into a second cluster on the far side of the frame, "
             "so the model is asked to fit a bimodal target.",
             fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.subplots_adjust(hspace=0.16)
    fig.savefig(FIG / "02_target_canonicalisation.png", dpi=140)
    plt.close(fig)


def fig_qualitative(df: pd.DataFrame, n: int = 5):
    """Three columns: the photo, the photo with corners drawn on, and the page
    flattened out. One row per background."""
    sel = (df.groupby("bg_name", group_keys=False)
             .apply(lambda g: g.sample(1, random_state=11), include_groups=False))
    sel = df.loc[sel.index]
    fig, axes = plt.subplots(len(sel), 3, figsize=(11, 3.0 * len(sel)))
    for r, (_, row) in enumerate(sel.iterrows()):
        img = cv2.cvtColor(cv2.imread(str(FRAMES / row.image_path.replace("/", "\\"))),
                           cv2.COLOR_BGR2RGB)
        raw_q = load_raw_quads(pd.DataFrame([row]))[0]
        can_q = np.asarray(row[CANON_COLS].to_numpy(), float).reshape(4, 2)

        axes[r, 0].imshow(img)
        axes[r, 0].set_ylabel(row.bg_name, fontsize=9, color=INK2, fontweight="600")

        ann = img.copy()
        cv2.polylines(ann, [can_q.astype(np.int32)], True, (43, 120, 214), 6)
        axes[r, 1].imshow(ann)
        for i, (px, py) in enumerate(can_q):
            axes[r, 1].plot(px, py, "o", ms=9, color=SERIES[i],
                            mec=SURFACE, mew=1.5)
            axes[r, 1].annotate(f"{i}", (px, py), xytext=(12, 0),
                                textcoords="offset points", fontsize=9,
                                color=INK, fontweight="600")

        axes[r, 2].imshow(dewarp(img, raw_q))
        if r == 0:
            for c, t in zip(range(3), ["frame (1920×1080)",
                                       "canonical corner order 0→3",
                                       "dewarped page (raw order)"]):
                axes[r, c].set_title(t, loc="left", fontsize=9,
                                     fontweight="600", color=INK)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([]); axes[r, c].grid(False)
            # Line the panels up at the top, since the flattened page is taller.
            axes[r, c].set_anchor("N")
    fig.tight_layout()
    fig.savefig(FIG / "03_qualitative.png", dpi=120)
    plt.close(fig)


def fig_colour(df: pd.DataFrame):
    bgs = sorted(df.bg_name.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    m = df.groupby("bg_name")[["mean_r", "mean_g", "mean_b"]].mean().loc[bgs]
    x = np.arange(len(bgs)); w = 0.26
    for i, (col, lab) in enumerate(zip(["mean_r", "mean_g", "mean_b"], "RGB")):
        ax.bar(x + (i - 1) * w, m[col], w * 0.92, label=lab,
               color=["#e34948", "#1baf7a", "#2a78d6"][i], edgecolor=SURFACE, lw=1.5)
    ax.set_xticks(x); ax.set_xticklabels([b.replace("background", "bg") for b in bgs])
    ax.set_ylim(100, 180); ax.legend(fontsize=8, ncol=3)
    _title(ax, "Every background has its own colour cast",
           "R runs 13–28 above G in all five — warm indoor light")
    ax.set_ylabel("channel mean (0–255)")

    ax = axes[1]
    for i, bg in enumerate(bgs):
        d = df[df.bg_name == bg]
        ax.scatter(d.mean_r - d.mean_g, d.mean_b - d.mean_g, s=3, alpha=0.25,
                   color=SERIES[i], label=bg, linewidths=0)
    _title(ax, "Backgrounds separate cleanly in chroma space",
           "a model can shortcut to background identity from colour alone")
    ax.set_xlabel("R − G"); ax.set_ylabel("B − G")
    leg = ax.legend(fontsize=8, markerscale=4)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.tight_layout()
    fig.savefig(FIG / "04_colour.png", dpi=140)
    plt.close(fig)


def fig_augmentation(df: pd.DataFrame, seed: int = 5):
    from augment import augment

    row = df[df.bg_name == "background05"].sample(1, random_state=2).iloc[0]
    img0 = cv2.cvtColor(
        cv2.imread(str(FRAMES / row.image_path.replace("/", "\\"))), cv2.COLOR_BGR2RGB)
    q0 = np.asarray(row[CANON_COLS].to_numpy(), float).reshape(4, 2)

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 4, figsize=(16, 5.8))
    for i, ax in enumerate(axes.ravel()):
        im, q = (img0, q0) if i == 0 else augment(img0.copy(), q0.copy(), rng)
        ax.imshow(im)
        ax.plot(*np.vstack([q, q[:1]]).T, color="#2a78d6", lw=3)
        for j, (px, py) in enumerate(q):
            ax.plot(px, py, "o", ms=8, color=SERIES[j], mec=SURFACE, mew=1.5)
        ax.set_title("original" if i == 0 else f"augmented #{i}", loc="left",
                     fontsize=9, fontweight="600", color=INK, pad=4)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlim(0, FRAME_W); ax.set_ylim(FRAME_H, 0)
    fig.suptitle("Geometric augmentation warps image and labels together; corner "
                 "order is re-canonicalised after the warp",
                 x=0.006, ha="left", fontsize=11, fontweight="600", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "05_augmentation.png", dpi=120)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT / "manifest.csv")
    sub = subsample(df, stride=3)
    fig_overview(df)
    fig_targets(df)
    fig_qualitative(sub)
    fig_colour(sub)
    fig_augmentation(sub)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
