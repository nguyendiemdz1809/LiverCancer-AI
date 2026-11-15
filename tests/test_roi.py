"""Automatic Tumor Zoom: within-slice merge, 3D linking, crop windows, pipeline wiring (unit tests, no real data)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.detection.yolo_detector import Detection  # noqa: E402
from livercancer.roi.boxes import Box, containment, iou, merge_boxes  # noqa: E402
from livercancer.roi.crop import context_window, crop, local_window, zoom  # noqa: E402
from livercancer.roi.pipeline import analyze_image, analyze_volume, load_roi_config  # noqa: E402
from livercancer.roi.tracking import filter_tracks, link_tracks  # noqa: E402


def D(x0, y0, x1, y1, c):
    return Detection(x0, y0, x1, y1, c)


def test_iou_containment():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert containment((0, 0, 10, 10), (2, 2, 4, 4)) == 1.0 and iou((0, 0, 10, 10), (2, 2, 4, 4)) < 0.1


def test_merge_duplicates_and_nested():
    out = merge_boxes([D(10, 10, 50, 50, 0.9), D(12, 11, 52, 49, 0.4), D(20, 20, 30, 30, 0.3), D(200, 200, 230, 230, 0.5)])
    assert len(out) == 2
    a = out[0]
    assert a.conf == 0.9 and a.n_merged == 3 and (a.x0, a.y0, a.x1, a.y1) == (10, 10, 52, 50)
    assert (out[1].x0, out[1].conf) == (200, 0.5)


def test_link_tracks_gap_and_separation():
    b = lambda x, c=0.5: [Box(x, 100, x + 40, 140, c)]  # noqa: E731
    per = {5: b(100), 6: b(102), 8: b(104, 0.9), 11: b(105), 6.5: []}
    per.pop(6.5)
    per[6] = per[6] + [Box(300, 300, 320, 320, 0.2)]
    tr = link_tracks(per, iou_thr=0.3, max_gap=1)
    main = [t for t in tr if 5 in t.boxes][0]
    assert main.ks == [5, 6, 8] and main.max_conf == 0.9        # gap of 1 slice (7) bridged
    assert any(t.ks == [11] for t in tr)                         # gap of 2 slices (9, 10) -> new track
    assert any(t.ks == [6] and t.boxes[6].x0 == 300 for t in tr)  # second lesion on slice 6
    assert main.best_slice == 8
    assert [t.ks for t in filter_tracks(tr, min_slices=2)] == [[5, 6, 8]]
    assert all(t.max_conf >= 0.5 for t in filter_tracks(tr, min_max_conf=0.5))


def test_crop_windows_and_padding():
    box = (100.0, 120.0, 140.0, 150.0)
    lw = local_window(box, (512, 512), 0.15, 8)
    assert lw[2] - lw[0] == lw[3] - lw[1] >= 40 + 2 * 8
    assert lw[0] <= box[0] and lw[1] <= box[1] and lw[2] >= box[2] and lw[3] >= box[3]
    cw = context_window(box, (512, 512), 2.5, 128)
    assert cw[2] - cw[0] == 128                                 # 2.5 x 40 = 100 < min 128
    img = np.full((64, 64), 200, np.uint8)
    c = crop(img, (-10, -10, 30, 30), 224)                      # window outside the image -> zero padding
    assert c.shape == (224, 224) and c[0, 0] == 0 and c[-1, -1] == 200
    z = zoom(np.zeros((512, 512), np.uint8), box)
    assert z["local"].shape == (224, 224) and z["context"].shape == (224, 224)


class StubDetector:
    """Deterministic stand-in for LesionDetector to test the wiring only (never used outside tests)."""
    modality, device = "ct", "cpu"

    def __init__(self, per_slice):
        self.per_slice = per_slice

    def predict(self, images, batch_size=8):
        return [list(self.per_slice.get(i, [])) for i in range(len(images))]


def test_pipeline_volume_and_image():
    params = load_roi_config("ct")
    vol = np.zeros((64, 64, 6), np.uint8)
    det = StubDetector({1: [D(10, 10, 30, 30, 0.8), D(11, 11, 29, 31, 0.3)], 2: [D(11, 10, 31, 30, 0.6)], 4: [D(40, 40, 50, 50, 0.2)]})
    res = analyze_volume(vol, det, dict(params, min_track_slices=2, min_track_conf=0.12))
    assert res.n_slices == 6 and len(res.merged[1]) == 1
    assert len(res.tracks_all) == 2 and [t.ks for t in res.tracks] == [[1, 2]]
    z = res.zoom(res.tracks[0])
    assert z["slice_k"] == 1 and z["local"].shape == (224, 224)
    one = analyze_image(np.zeros((64, 64), np.uint8), StubDetector({0: [D(5, 5, 20, 20, 0.5)]}), dict(params, min_track_slices=5))
    assert len(one.tracks) == 1                                 # min_track_slices does not apply to a single 2D image


def test_roi_config_has_all_keys():
    for mod in ("ct", "mri"):
        p = load_roi_config(mod)
        for k in ("merge_iou", "merge_containment", "link_iou", "link_max_gap", "min_track_slices", "min_track_conf",
                  "crop_size", "margin_frac", "min_margin_px", "context_scale", "context_min_px"):
            assert k in p, (mod, k)
        assert p["min_track_conf"] >= 0.12  # never below the detector threshold chosen on Val


@pytest.mark.parametrize("mod", ["ct", "mri"])
def test_real_detector_zoom_on_val_slice(mod):
    """End-to-end on one real Val slice with the final checkpoint: preprocessing -> YOLO -> merge -> zoom runs."""
    import csv

    from livercancer.config import load_paths, load_yaml
    from livercancer.data.slices import load_volume_uint8, slice_image
    from livercancer.data.sources import image_path
    from livercancer.detection.yolo_detector import LesionDetector
    cfg = load_yaml(ROOT / "configs" / "yolo_inference.yaml")[mod]
    data = ROOT / cfg["dataset"]
    if not (ROOT / cfg["weights"]).exists() or not (data / "slices.csv").exists():
        pytest.skip("weights or exported Val set not available")
    with open(data / "slices.csv", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["split"] == "val" and r["kind"] == "pos")
    src = image_path(load_paths(), mod, row["patient_id"], row["series"])
    if not src.exists():
        pytest.skip("source dataset not available")
    img = slice_image(load_volume_uint8(src), int(row["slice_k"]))
    res = analyze_image(img, LesionDetector.from_config(mod, device="cpu"))
    for t in res.tracks:
        z = res.zoom(t)
        assert z["local"].shape == (224, 224) and z["local"].dtype == np.uint8
