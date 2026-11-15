import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.data.nifti_io import to_display  # noqa: E402
from livercancer.data.yolo_boxes import box_iou, component_boxes, from_yolo, to_yolo  # noqa: E402


def test_component_boxes_and_fragment_filter():
    m = np.zeros((40, 30), np.uint8)  # (i, j)
    m[5:10, 2:8] = 1      # 30 px
    m[20:22, 20:21] = 1   # 2 px fragment
    boxes, filtered = component_boxes(m, min_area=10)
    assert boxes == [(5, 10, 2, 8)] and filtered == 1


def test_diagonal_pixels_are_one_component():
    m = np.zeros((10, 10), np.uint8)
    m[2, 2] = m[3, 3] = m[4, 4] = 1
    boxes, _ = component_boxes(m, min_area=1)
    assert boxes == [(2, 5, 2, 5)]


def test_yolo_box_matches_displayed_image():
    ni, nj = 64, 48                      # slice (i, j); displayed image is (nj rows, ni cols)
    m = np.zeros((ni, nj), np.uint8)
    m[10:20, 30:40] = 1
    (box,), _ = component_boxes(m, 1)
    cx, cy, w, h = to_yolo(box, ni, nj)
    img = to_display(m)
    assert img.shape == (nj, ni)
    x0, x1, y0, y1 = from_yolo(cx, cy, w, h, width=img.shape[1], height=img.shape[0])
    rows, cols = np.nonzero(img)
    assert (x0, x1, y0, y1) == pytest.approx((cols.min(), cols.max() + 1, rows.min(), rows.max() + 1))
    assert all(0 <= v <= 1 for v in (cx, cy, w, h))


def test_box_iou():
    assert box_iou((0, 2, 0, 2), (0, 2, 0, 2)) == 1
    assert box_iou((0, 2, 0, 2), (2, 4, 0, 2)) == 0
    assert box_iou((0, 2, 0, 2), (1, 3, 0, 2)) == pytest.approx(1 / 3)
