"""Math on the four corner points.

A "quad" is one page's 4 corners, shaped (4, 2). These work on many at once,
so they take (N, 4, 2). `canonicalise_quad` is the important one.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .paths import DEWARP_SIZE, FRAME_H, FRAME_W, RAW_CORNER_COLS


def load_raw_quads(df: pd.DataFrame) -> np.ndarray:
    """Read the 8 corner columns from the metadata as (N, 4, 2).

    These are the original labels, in page order: tl, tr, br, bl.
    """
    return df[RAW_CORNER_COLS].to_numpy(dtype=np.float64).reshape(-1, 4, 2)


def canonicalise_quad(quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rename the corners by where they sit in the photo, not on the page.

    The original labels follow the page, so "top-left" lands somewhere else
    whenever the page is turned (all of background04). Corner 0 becomes
    whichever corner is nearest the photo's top-left, every time.

    Returns the reordered quads and `shift`, which records how far the list
    rotated so the original page order can be recovered.
    """
    # The corners always run the same way round, so we only need to change
    # which one is first. Top-left is the one with the smallest x + y.
    shift = (quad[:, :, 0] + quad[:, :, 1]).argmin(axis=1)
    rows = np.arange(quad.shape[0])[:, None]
    idx = (shift[:, None] + np.arange(4)[None, :]) % 4
    return quad[rows, idx], shift


def polygon_area(quad: np.ndarray) -> np.ndarray:
    """Area of each quad, in pixels squared."""
    x, y = quad[:, :, 0], quad[:, :, 1]
    return 0.5 * np.abs(
        (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)
    )


def side_lengths(quad: np.ndarray) -> np.ndarray:
    """Length of each of the 4 edges, in pixels. Returns (N, 4)."""
    return np.linalg.norm(np.roll(quad, -1, axis=1) - quad, axis=2)


def is_convex(quad: np.ndarray) -> np.ndarray:
    """True if the shape doesn't fold in on itself. Used to check the labels."""
    v = np.roll(quad, -1, axis=1) - quad
    w = np.roll(v, -1, axis=1)
    cross = v[:, :, 0] * w[:, :, 1] - v[:, :, 1] * w[:, :, 0]
    return (cross > 0).all(axis=1) | (cross < 0).all(axis=1)


def rotation_deg(quad: np.ndarray) -> np.ndarray:
    """How far the page is turned, in degrees: 0 upright, +90 clockwise.

    Give it the raw quads, not the reordered ones -- it reads the page's edge.
    """
    e = quad[:, 1] - quad[:, 0]
    return np.degrees(np.arctan2(e[:, 1], e[:, 0]))


def normalise(quad: np.ndarray, w: int = FRAME_W, h: int = FRAME_H) -> np.ndarray:
    """Turn pixel positions into fractions from 0 to 1.

    x is divided by 1920 and y by 1080, so the two axes are not the same
    scale. Never measure distance on these -- call denormalise first.
    """
    out = quad.copy()
    out[:, :, 0] /= w
    out[:, :, 1] /= h
    return out


def denormalise(flat: np.ndarray, w: int = FRAME_W, h: int = FRAME_H) -> np.ndarray:
    """Turn the 8 fractions back into pixels, (N, 8) -> (N, 4, 2).

    Undoes `normalise`. Use it before measuring any error.
    """
    q = np.asarray(flat, dtype=np.float64).reshape(-1, 4, 2).copy()
    q[:, :, 0] *= w
    q[:, :, 1] *= h
    return q


def dewarp(img: np.ndarray, quad: np.ndarray,
           size: tuple[int, int] = DEWARP_SIZE) -> np.ndarray:
    """Flatten the page out of a photo into a rectangle, like a scan.

    Needs the raw page-ordered quad, so it knows which edge is the top.
    """
    w, h = size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, m, (w, h))
