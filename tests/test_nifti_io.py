import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.data.nifti_io import as_3d, mask_bbox, to_display  # noqa: E402


def test_as_3d_passthrough():
    v = np.zeros((4, 5, 6), np.uint8)
    out, note = as_3d(v)
    assert out is v and note["converted"] is False


def test_as_3d_identical_channels():
    base = np.arange(4 * 5 * 6, dtype=np.uint8).reshape(4, 5, 6)
    v = np.repeat(base[..., None, None], 3, axis=-1)  # (X, Y, Z, 1, 3)
    out, note = as_3d(v)
    assert out.shape == (4, 5, 6) and np.array_equal(out, base)
    assert note["converted"] and note["channels_identical"] and note["n_channels"] == 3


def test_as_3d_rejects_differing_channels():
    v = np.zeros((4, 5, 6, 1, 3), np.uint8)
    v[0, 0, 0, 0, 2] = 1
    with pytest.raises(ValueError):
        as_3d(v)


def test_as_3d_tolerates_small_differences_only_when_allowed():
    v = np.zeros((10, 10, 10, 1, 3), np.uint8)
    v[0, 0, 0, 0, 1] = 5                     # 1 of 1000 voxels differs
    out, note = as_3d(v, max_diff_frac=0.01)
    assert out.shape == (10, 10, 10) and note["diff_mask"].sum() == 1 and not note["channels_identical"]
    with pytest.raises(ValueError):
        as_3d(v, max_diff_frac=0.0005)


def test_mask_bbox_inclusive_and_empty():
    m = np.zeros((10, 12, 8), bool)
    m[2:5, 3:9, 1:2] = True
    assert mask_bbox(m) == (2, 4, 3, 8, 1, 1)
    assert mask_bbox(np.zeros((3, 3, 3), bool)) is None


def test_to_display_rows_are_j_cols_are_i():
    s = np.zeros((7, 4))  # (i, j)
    s[6, 1] = 1
    d = to_display(s)
    assert d.shape == (4, 7) and d[1, 6] == 1
