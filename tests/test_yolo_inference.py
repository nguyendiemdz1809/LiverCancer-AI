"""Inference path: preprocessing identical to training export, device fallback, matching, detector loading."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import load_paths, load_yaml  # noqa: E402
from livercancer.data.slices import load_volume_uint8, slice_image, to_model_input  # noqa: E402
from livercancer.evaluation.matching import count_outcomes, match  # noqa: E402

CFG = load_yaml(ROOT / "configs" / "yolo_inference.yaml")


def test_to_model_input_replicates_gray():
    g = np.arange(12, dtype=np.uint8).reshape(3, 4)
    x = to_model_input(g)
    assert x.shape == (3, 4, 3) and all(np.array_equal(x[..., c], g) for c in range(3))
    with pytest.raises(ValueError):
        to_model_input(g.astype(np.float32))


def test_match_counts():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    pred = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]])
    tp, used = match(gt, pred, np.array([0.9, 0.8, 0.7]))
    assert tp.tolist() == [True, False, False] and used.tolist() == [True, False]
    oc = count_outcomes(gt, pred, np.array([0.9, 0.8, 0.7]))
    assert (oc["TP"], oc["FP"], oc["FN"]) == (1, 2, 1) and oc["gt_best_iou"] == [1.0, 0.0]
    assert count_outcomes(np.zeros((0, 4)), np.zeros((0, 4)), np.zeros(0)) == {"TP": 0, "FP": 0, "FN": 0, "gt_best_iou": []}


def test_device_fallback_to_cpu(monkeypatch):
    import torch

    from livercancer.detection.yolo_detector import resolve_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.warns(UserWarning):
        assert resolve_device("cuda") == "cpu"


@pytest.mark.parametrize("mod", ["ct", "mri"])
def test_nifti_slice_equals_training_png(mod):
    """The inference input rebuilt from the source NIfTI is pixel-identical to the PNG the model was trained on."""
    import cv2

    from livercancer.data.sources import image_path
    data = ROOT / CFG[mod]["dataset"]
    if not (data / "slices.csv").exists():
        pytest.skip(f"{data} not exported on this machine")
    paths = load_paths()
    with open(data / "slices.csv", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["split"] == "val" and r["kind"] == "pos")
    src = image_path(paths, mod, row["patient_id"], row["series"])
    if not src.exists():
        pytest.skip(f"source dataset not available: {src}")
    img = slice_image(load_volume_uint8(src), int(row["slice_k"]))
    png = cv2.imread(str(data / "images" / "val" / f"{row['file']}.png"), cv2.IMREAD_GRAYSCALE)
    assert img.dtype == np.uint8 and img.shape == (int(row["height"]), int(row["width"]))
    assert np.array_equal(img, png)


@pytest.mark.parametrize("mod", ["ct", "mri"])
def test_detector_loads_and_predicts_on_cpu(mod):
    from livercancer.detection.yolo_detector import LesionDetector
    if not (ROOT / CFG[mod]["weights"]).exists():
        pytest.skip("weights not downloaded")
    det = LesionDetector.from_config(mod, device="cpu")
    assert det.device == "cpu" and det.quantize is None and det.imgsz == 512
    assert det.sha256 == CFG[mod]["sha256"]
    out = det.predict([np.zeros((512, 512), np.uint8)])
    assert len(out) == 1 and all(d.label == "lesion" and 0 <= d.conf <= 1 for d in out[0])


def test_detector_rejects_wrong_checkpoint():
    from livercancer.detection.yolo_detector import LesionDetector
    if not (ROOT / CFG["ct"]["weights"]).exists():
        pytest.skip("weights not downloaded")
    with pytest.raises(ValueError, match="sha256"):
        LesionDetector("ct", ROOT / CFG["ct"]["weights"], device="cpu", expected_sha256=CFG["mri"]["sha256"])
    with pytest.raises(ValueError, match="modality"):
        LesionDetector("pet", ROOT / CFG["ct"]["weights"], device="cpu")


@pytest.mark.parametrize("mod", ["ct", "mri"])
def test_operating_threshold_matches_val_choice(mod):
    """configs/yolo_inference.yaml `conf` must be the Val F2-optimal threshold recorded by scripts/53_threshold_sweep.py."""
    import json
    src = ROOT / CFG[mod]["conf_source"]
    if not src.exists():
        pytest.skip("threshold sweep not run on this machine")
    choice = json.loads(src.read_text(encoding="utf-8"))
    assert "Val" in choice["rule"] and CFG[mod]["conf"] == choice["F2_optimal"]["conf"]
