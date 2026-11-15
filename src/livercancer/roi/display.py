"""Display-only Tumor Zoom for the website (the classifier's 224 px crops in roi.crop are NOT changed).

The classifier crop (box + 15 %, resized to 224) magnifies a 10 px lesion about 9x, which looks blurred on screen.
For display we cut a wider square from the original slice that keeps surrounding liver tissue, keep it inside the image
(shifted rather than padded with black where possible), and enlarge it at most MAX_UPSCALE times with bicubic
interpolation. No sharpening, no super-resolution: every pixel comes from the original slice.
"""
from __future__ import annotations

import cv2
import numpy as np

MAX_UPSCALE = 4.0            # never enlarge more than 4x
OUT_MAX = 480                # displayed side at most (px); larger windows are reduced with INTER_AREA
KINDS = {                    # window side = max(factor x longest box side, minimum px)
    "local": (2.2, 72),      # lesion + about half its size of tissue on each side
    "context": (3.0, 192),   # lesion in its surroundings
}
MARK = (59, 59, 229)         # BGR red corner marks around the lesion box


def window(box, shape: tuple[int, int], kind: str) -> tuple[int, int, int, int]:
    """Square window (x0, y0, x1, y1) around the box, inside the image when the image is large enough."""
    f, mn = KINDS[kind]
    h, w = shape[:2]
    side = int(round(max(f * max(box[2] - box[0], box[3] - box[1]), mn)))
    side = max(8, min(side, max(h, w)))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    if side <= w:
        x0 = min(max(0, x0), w - side)
    if side <= h:
        y0 = min(max(0, y0), h - side)
    return x0, y0, x0 + side, y0 + side


def _cut(img: np.ndarray, win) -> np.ndarray:
    x0, y0, x1, y1 = win
    h, w = img.shape[:2]
    pad = ((max(0, -y0), max(0, y1 - h)), (max(0, -x0), max(0, x1 - w))) + ((0, 0),) * (img.ndim - 2)
    src = np.pad(img, pad, mode="constant")
    return src[y0 + pad[0][0]:y1 + pad[0][0], x0 + pad[1][0]:x1 + pad[1][0]]


def _corners(im: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> None:
    """Corner brackets (not a full rectangle, so the lesion border stays visible)."""
    L = max(6, int(0.22 * min(x1 - x0, y1 - y0)))
    p = [int(round(v)) for v in (x0, y0, x1, y1)]
    for (cx, cy, dx, dy) in ((p[0], p[1], 1, 1), (p[2], p[1], -1, 1), (p[0], p[3], 1, -1), (p[2], p[3], -1, -1)):
        cv2.line(im, (cx, cy), (cx + dx * L, cy), MARK, 2, cv2.LINE_AA)
        cv2.line(im, (cx, cy), (cx, cy + dy * L), MARK, 2, cv2.LINE_AA)


def display_zoom(img: np.ndarray, box, kind: str = "local", marks: bool = True) -> dict:
    """{'image': BGR uint8, 'window': (x0, y0, x1, y1) in slice pixels, 'scale': enlargement factor}."""
    win = window(box, img.shape, kind)
    patch = _cut(img, win)
    side = patch.shape[0]
    scale = min(MAX_UPSCALE, OUT_MAX / side)
    out = int(round(side * scale))
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    z = cv2.resize(patch, (out, out), interpolation=interp)
    if z.ndim == 2:
        z = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)
    if marks:
        s = out / side
        _corners(z, (box[0] - win[0]) * s, (box[1] - win[1]) * s, (box[2] - win[0]) * s, (box[3] - win[1]) * s)
    return {"image": z, "window": tuple(int(v) for v in win), "scale": round(scale, 2)}
