"""Automatic Tumor Zoom: YOLO detections -> within-slice merge -> 3D tracks (volumes) -> filtered lesion candidates -> crops.

    det = LesionDetector.from_config("ct")
    res = analyze_volume(load_volume_uint8(path), det)           # NIfTI volume (streamed)
    res = analyze_image(gray_png, det)                            # one 2D slice (exported-format PNG)
    z = res.zoom(res.tracks[0])                                   # local / context crops on the track's best slice

Parameters come from configs/roi.yaml (track filter chosen on Validation only, scripts/71_eval_roi_val.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from livercancer.config import CONFIG_DIR, load_yaml
from livercancer.data.slices import slice_image
from livercancer.roi.boxes import Box, merge_boxes
from livercancer.roi.crop import zoom
from livercancer.roi.tracking import Track, filter_tracks, link_tracks


def load_roi_config(modality: str, path: str | Path = CONFIG_DIR / "roi.yaml") -> dict:
    cfg = load_yaml(path)
    out = dict(cfg["common"])
    out.update(cfg.get(modality, {}))
    return out


@dataclass
class RoiResult:
    modality: str
    images: list[np.ndarray]                    # displayed slices (H x W uint8), index = slice k
    raw: dict[int, list]                        # k -> detector boxes (Detection)
    merged: dict[int, list[Box]]                # k -> merged boxes
    tracks_all: list[Track]
    tracks: list[Track]                         # after the Val-chosen filter, sorted by max confidence
    params: dict
    device: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def n_slices(self) -> int:
        return len(self.images)

    def slices_with_boxes(self) -> list[int]:
        return sorted({k for t in self.tracks for k in t.ks})

    def boxes_on(self, k: int) -> list[tuple[Track, Box]]:
        return [(t, t.boxes[k]) for t in self.tracks if k in t.boxes]

    def zoom(self, track: Track, k: int | None = None) -> dict:
        k = track.best_slice if k is None else k
        p = self.params
        z = zoom(self.images[k], track.boxes[k].xyxy, p["crop_size"], p["margin_frac"], p["min_margin_px"],
                 p["context_scale"], p["context_min_px"])
        z["slice_k"] = k
        return z


def _build(images: list[np.ndarray], raw: dict[int, list], modality: str, params: dict, is_volume: bool, device: str) -> RoiResult:
    merged = {k: merge_boxes(d, params["merge_iou"], params["merge_containment"]) for k, d in raw.items() if d}
    tracks_all = link_tracks(merged, params["link_iou"], params["link_max_gap"])
    min_slices = params["min_track_slices"] if is_volume else 1
    kept = sorted(filter_tracks(tracks_all, min_slices, params["min_track_conf"]), key=lambda t: -t.max_conf)
    return RoiResult(modality, images, raw, merged, tracks_all, kept, params, device)


def analyze_volume(vol: np.ndarray, detector, params: dict | None = None, batch_size: int = 16) -> RoiResult:
    """All slices of an (i, j, k) uint8 volume (from livercancer.data.slices.load_volume_uint8)."""
    params = params or load_roi_config(detector.modality)
    images = [slice_image(vol, k) for k in range(vol.shape[2])]
    dets = detector.predict(images, batch_size=batch_size)
    return _build(images, dict(enumerate(dets)), detector.modality, params, True, detector.device)


def analyze_image(gray: np.ndarray, detector, params: dict | None = None) -> RoiResult:
    """One 2D grayscale uint8 slice in the training display convention (e.g. an exported PNG)."""
    params = params or load_roi_config(detector.modality)
    dets = detector.predict([gray])
    return _build([gray], {0: dets[0]}, detector.modality, params, False, detector.device)
