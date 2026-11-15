"""Tumor zoom: square crops around a lesion box with margin (local) and with surrounding context."""
from __future__ import annotations

import cv2
import numpy as np


def square_window(box: tuple, side: float, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Square window (x0, y0, x1, y1) of `side` px centred on the box; may extend beyond the image (padded later)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    s = max(1, int(round(side)))
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    return x0, y0, x0 + s, y0 + s


def local_window(box: tuple, shape: tuple[int, int], margin_frac: float = 0.15, min_margin: int = 8) -> tuple[int, int, int, int]:
    """Box + margin on each side (max(margin_frac x box side, min_margin) px), made square."""
    w, h = box[2] - box[0], box[3] - box[1]
    side = max(w, h) + 2 * max(margin_frac * max(w, h), min_margin)
    return square_window(box, side, shape)


def context_window(box: tuple, shape: tuple[int, int], scale: float = 2.5, min_side: int = 128) -> tuple[int, int, int, int]:
    """Larger square (scale x longest box side, at least min_side px) to keep surrounding tissue."""
    return square_window(box, max(scale * max(box[2] - box[0], box[3] - box[1]), min_side), shape)


def crop(img: np.ndarray, win: tuple[int, int, int, int], out_size: int = 224) -> np.ndarray:
    """Crop a window (zero padding outside the image) and resize to out_size x out_size (uint8, same channels)."""
    x0, y0, x1, y1 = win
    h, w = img.shape[:2]
    pad = ((max(0, -y0), max(0, y1 - h)), (max(0, -x0), max(0, x1 - w))) + ((0, 0),) * (img.ndim - 2)
    src = np.pad(img, pad, mode="constant")
    patch = src[y0 + pad[0][0]:y1 + pad[0][0], x0 + pad[1][0]:x1 + pad[1][0]]
    interp = cv2.INTER_CUBIC if patch.shape[0] < out_size else cv2.INTER_AREA
    return cv2.resize(patch, (out_size, out_size), interpolation=interp)


def zoom(img: np.ndarray, box: tuple, out_size: int = 224, margin_frac: float = 0.15, min_margin: int = 8,
         context_scale: float = 2.5, context_min: int = 128) -> dict:
    """Local and context crops of one lesion box on one slice, plus the windows (to draw them on the original)."""
    lw = local_window(box, img.shape[:2], margin_frac, min_margin)
    cw = context_window(box, img.shape[:2], context_scale, context_min)
    return {"local": crop(img, lw, out_size), "context": crop(img, cw, out_size), "local_window": lw, "context_window": cw}
