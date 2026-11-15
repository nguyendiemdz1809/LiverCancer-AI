"""Read-only NIfTI helpers. Volumes are returned in native voxel order (i, j, k) without reorientation."""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def load_header(path: str | Path) -> dict:
    """Header-only metadata (does not decompress voxel data)."""
    img = nib.load(str(path))
    return {
        "shape": tuple(int(s) for s in img.shape),
        "zooms": tuple(round(float(z), 4) for z in img.header.get_zooms()),
        "orientation": "".join(nib.aff2axcodes(img.affine)),
        "dtype": str(img.get_data_dtype()),
        "affine": img.affine,
    }


def load_volume(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load voxel data keeping the on-disk dtype (avoids float64 blow-up of get_fdata)."""
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj), img.affine


def as_3d(vol: np.ndarray, max_diff_frac: float = 0.0) -> tuple[np.ndarray, dict]:
    """Reduce volumes stored as (X, Y, Z, 1, C) (e.g. RGB-encoded CT) to 3D by taking channel 0.

    By default all channels must be identical. With max_diff_frac > 0, channels may differ on at most that fraction
    of voxels (PLC-CECT: anti-aliased burned-in ruler text); `note["diff_mask"]` then marks the differing voxels so
    the caller can verify they lie outside the anatomy of interest. Otherwise raises — nothing is silently dropped.
    """
    note = {"original_shape": list(vol.shape), "converted": False}
    if vol.ndim == 3:
        return vol, note
    v = vol.reshape(vol.shape[:3] + (-1,))
    diff = np.zeros(v.shape[:3], bool)
    for c in range(1, v.shape[-1]):
        diff |= v[..., 0] != v[..., c]
    frac = float(diff.mean())
    note.update(converted=True, n_channels=int(v.shape[-1]), channels_identical=not diff.any(),
                channel_diff_frac=round(frac, 6))
    if diff.any():
        if frac > max_diff_frac:
            raise ValueError(f"Multi-channel volume with differing channels: shape {vol.shape}, "
                             f"differing voxel fraction {frac:.5f} > {max_diff_frac}")
        note["diff_mask"] = diff
    return v[..., 0], note


def mask_bbox(mask: np.ndarray) -> tuple[int, ...] | None:
    """Inclusive bbox (i_min, i_max, j_min, j_max, k_min, k_max) of non-zero voxels, or None if empty."""
    idx = np.nonzero(mask)
    if len(idx[0]) == 0:
        return None
    out = []
    for ax in idx:
        out += [int(ax.min()), int(ax.max())]
    return tuple(out)


def to_display(slice_ij: np.ndarray) -> np.ndarray:
    """Voxel slice (i, j) -> image (rows=j, cols=i). For LPS volumes this is radiological view."""
    return np.asarray(slice_ij).T
