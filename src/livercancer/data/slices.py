"""2D slices for inference, identical to the GĐ2a YOLO export (scripts/20_build_yolo_pilot.py, `export_volume`).

Training images were produced as:
  NIfTI (read-only) -> orientation must be LPS -> CT 5D (X,Y,Z,1,3) -> channel 0 (`as_3d`, <= 1 % differing voxels:
  burned-in ruler text) -> dtype must already be uint8 (no extra windowing / normalisation: CT is windowed and MRI is
  per-volume clipped to uint8 at the source) -> slice k shown as slice.T (rows = j, cols = i) -> grayscale PNG.
Ultralytics then read each PNG with cv2.IMREAD_COLOR, i.e. the gray value replicated into 3 identical channels.

The export also checked that differing RGB voxels do not overlap the GT liver/lesion masks. That check needs GT
masks, which must not be used at inference on Test, so it is not repeated here (all 62 such CT volumes were audited
in GĐ1/GĐ2a: ruler text only).
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from livercancer.data.nifti_io import as_3d, to_display

LPS = ("L", "P", "S")
CT_MAX_CHANNEL_DIFF = 0.01  # same value as `--max-channel-diff` of the export


def load_volume_uint8(path: str | Path, max_channel_diff: float = CT_MAX_CHANNEL_DIFF) -> np.ndarray:
    """Read one image volume (streamed, one at a time) exactly as the export did; returns (i, j, k) uint8."""
    img = nib.load(str(path))
    codes = nib.aff2axcodes(img.affine)
    if codes != LPS:
        raise ValueError(f"orientation {codes} != LPS: {path}")
    vol, _ = as_3d(np.asanyarray(img.dataobj), max_diff_frac=max_channel_diff)
    if vol.dtype != np.uint8:
        raise ValueError(f"unexpected dtype {vol.dtype} (training data was uint8 at the source): {path}")
    return vol


def slice_image(vol: np.ndarray, k: int) -> np.ndarray:
    """Slice k of an (i, j, k) volume as the 2D grayscale image used for training (H = j, W = i), uint8."""
    return np.ascontiguousarray(to_display(vol[:, :, k]))


def to_model_input(gray: np.ndarray) -> np.ndarray:
    """Grayscale HxW uint8 -> HxWx3 uint8, as cv2.IMREAD_COLOR returns for a grayscale PNG (identical channels)."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError(f"expected a 2D uint8 image, got {gray.shape} {gray.dtype}")
    return np.repeat(gray[:, :, None], 3, axis=2)
