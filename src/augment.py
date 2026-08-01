"""Randomly tweak photos so the model sees more variety.

Two kinds of tweak:
  - brightness, blur, colour: change pixels only, labels stay put
  - rotate, zoom, tilt:       MOVE things, so the labels must move too
"""

from __future__ import annotations

import cv2
import numpy as np

from preprocess import FRAME_H, FRAME_W, canonicalise_quad

# average rgb values of our dataset that we can use for realistic augmentation
DATASET_MEAN = np.array([0.6346, 0.5556, 0.5995], np.float32)
DATASET_STD = np.array([0.1559, 0.1483, 0.1809], np.float32)


def photometric(img: np.ndarray, rng: np.random.Generator, strength: float = 1.0):
    """Brightness / contrast / gamma / colour balance / blur / sensor noise.

    The amounts are based on how much our real photos actually vary, so the
    results still look like plausible photos rather than nonsense.
    """
    out = img.astype(np.float32)

    alpha = 1.0 + rng.uniform(-0.25, 0.25) * strength      # contrast
    beta = rng.uniform(-28, 28) * strength                 # brightness
    # (multiply = contrast, add = brightness)
    out = out * alpha + beta

    if rng.random() < 0.5:                                 # tint the colours
        out *= (1.0 + rng.uniform(-0.08, 0.08, 3) * strength).astype(np.float32)

    out = np.clip(out, 0, 255)

    gamma = float(np.exp(rng.uniform(-0.30, 0.30) * strength))
    out = 255.0 * (out / 255.0) ** gamma

    if rng.random() < 0.35:                                # blur it
        k = int(rng.integers(3, 10)) | 1
        if rng.random() < 0.5:
            out = cv2.GaussianBlur(out, (k, k), 0)
        else:                                              # smear, like a shaky hand
            kern = np.zeros((k, k), np.float32)
            kern[k // 2, :] = 1.0 / k
            m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5),
                                        float(rng.uniform(0, 180)), 1.0)
            out = cv2.filter2D(out, -1, cv2.warpAffine(kern, m, (k, k)))

    if rng.random() < 0.3:
        out += rng.normal(0, rng.uniform(2, 7), out.shape).astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)



# shape changes
def random_homography(w: int, h: int, rng: np.random.Generator,
                      max_rot: float = 12.0, scale: tuple[float, float] = (0.85, 1.15),
                      max_shift: float = 0.06, max_warp: float = 0.04) -> np.ndarray:
    """Build one random turn + zoom + shift + tilt, as a 3x3 matrix.

    Packing it into a matrix is the trick that lets us apply the exact same
    change to the photo and to the 4 corner points.
    """
    m = cv2.getRotationMatrix2D((w / 2, h / 2),
                                float(rng.uniform(-max_rot, max_rot)),
                                float(rng.uniform(*scale)))
    m[0, 2] += rng.uniform(-max_shift, max_shift) * w
    m[1, 2] += rng.uniform(-max_shift, max_shift) * h
    H = np.vstack([m, [0, 0, 1]]).astype(np.float64)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = rng.uniform(-max_warp, max_warp, (4, 2)) * [w, h]
    H = cv2.getPerspectiveTransform(src, (src + jit).astype(np.float32)) @ H
    return H


def apply_homography(quad: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Move 4 corner points through the matrix from random_homography."""
    pts = np.hstack([quad, np.ones((len(quad), 1))]) @ H.T
    return pts[:, :2] / pts[:, 2:3]


def geometric(img: np.ndarray, quad: np.ndarray, rng: np.random.Generator,
              keep_inside: bool = True, tries: int = 6, **kw):
    """Turn/zoom/tilt the photo and its 4 corners together.

    If the random change pushes the page off the edge of the photo, we roll the
    dice again (up to `tries` times). Real pages are always fully in view, so a
    label outside the photo would be something the model can never predict.
    """
    h, w = img.shape[:2]
    for _ in range(tries):
        H = random_homography(w, h, rng, **kw)
        q = apply_homography(quad, H)
        if not keep_inside or (
            (q[:, 0] >= 0).all() and (q[:, 0] <= w).all()
            and (q[:, 1] >= 0).all() and (q[:, 1] <= h).all()
        ):
            warped = cv2.warpPerspective(img, H, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
            return warped, canonicalise_quad(q[None])[0][0]
    return img, quad 


def augment(img: np.ndarray, quad: np.ndarray, rng: np.random.Generator,
            geo: bool = True, photo: bool = True, strength: float = 1.0):
    if geo:
        img, quad = geometric(img, quad, rng)
    if photo:
        img = photometric(img, rng, strength)
    return img, quad


def to_tensor_chw(img: np.ndarray, mean=DATASET_MEAN, std=DATASET_STD) -> np.ndarray:
    """Get a photo ready for the model.

    Scales 0-255 down to 0-1, subtracts the average colour, and moves the
    colour channel to the front, which is the order PyTorch expects.
    """
    x = img.astype(np.float32) / 255.0
    if x.ndim == 2:
        x = x[:, :, None]
        return ((x - x.mean()) / (x.std() + 1e-6)).transpose(2, 0, 1)
    return ((x - mean) / std).transpose(2, 0, 1)
