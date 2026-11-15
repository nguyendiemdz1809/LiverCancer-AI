"""Lesion mask slice -> YOLO boxes, in the display convention used everywhere: image = slice.T (rows = j, cols = i)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def component_boxes(mask_ij: np.ndarray, min_area: int) -> tuple[list[tuple[int, int, int, int]], int]:
    """Boxes (i0, i1, j0, j1), exclusive upper bounds, one per 8-connected component with area >= min_area.

    Returns (boxes, n_filtered) where n_filtered counts components dropped as fragments.
    """
    lab, n = ndimage.label(mask_ij > 0, structure=EIGHT_CONNECTED)
    if n == 0:
        return [], 0
    areas = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    boxes, filtered = [], 0
    for idx, sl in enumerate(ndimage.find_objects(lab)):
        if areas[idx] < min_area:
            filtered += 1
            continue
        boxes.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    return boxes, filtered


def to_yolo(box: tuple[int, int, int, int], ni: int, nj: int) -> tuple[float, float, float, float]:
    """(i0, i1, j0, j1) on an (ni, nj) slice -> normalized (cx, cy, w, h) on the displayed image (width ni, height nj)."""
    i0, i1, j0, j1 = box
    return ((i0 + i1) / 2 / ni, (j0 + j1) / 2 / nj, (i1 - i0) / ni, (j1 - j0) / nj)


def from_yolo(cx: float, cy: float, w: float, h: float, width: int, height: int) -> tuple[float, float, float, float]:
    """Normalized YOLO box -> pixel (x0, x1, y0, y1) on the displayed image."""
    return ((cx - w / 2) * width, (cx + w / 2) * width, (cy - h / 2) * height, (cy + h / 2) * height)


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """IoU of (x0, x1, y0, y1) boxes."""
    ix = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = ix * iy
    union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / union if union > 0 else 0.0
