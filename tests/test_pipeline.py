"""End-to-end pipeline (YOLO -> Tumor Zoom -> classifier): statuses, warnings, training-consistent ROI selection."""
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import load_yaml  # noqa: E402
from livercancer.detection.yolo_detector import Detection  # noqa: E402
from livercancer.roi.boxes import Box  # noqa: E402
from livercancer.roi.tracking import Track, spaced_key_slices  # noqa: E402

CLS = load_yaml(ROOT / "configs" / "cls.yaml")


def _have(mod):
    inf = CLS["inference"][mod]
    return (ROOT / inf["checkpoint"]).exists() and (ROOT / load_yaml(ROOT / "configs" / "yolo_inference.yaml")[mod]["weights"]).exists()


def test_key_slices_identical_to_dataset_builder():
    spec = importlib.util.spec_from_file_location("b80", ROOT / "scripts" / "80_build_cls_dataset.py")
    b80 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b80)
    rng = random.Random(0)
    for _ in range(200):
        ks = sorted(rng.sample(range(80), rng.randint(1, 25)))
        t = Track(0, {k: Box(0, 0, rng.uniform(5, 60), rng.uniform(5, 60), rng.uniform(0.12, 0.9)) for k in ks})
        for gap in (1, 2):
            assert spaced_key_slices(t, 5, gap) == b80.choose_slices(t, 5, gap, score=lambda b: b.area * b.conf)


class StubDetector:
    modality, device, conf = "ct", "cpu", 0.12

    def __init__(self, dets):
        self.dets = dets

    def predict(self, images, batch_size=8):
        return [list(self.dets.get(i, [])) for i in range(len(images))]


def _pipe(mod, monkeypatch, dets, checkpoint=None, gate="config"):
    import livercancer.detection.yolo_detector as yd
    from livercancer.pipeline import LiverCancerPipeline
    stub = StubDetector(dets)
    stub.modality = mod
    monkeypatch.setattr(yd.LesionDetector, "from_config", classmethod(lambda cls, m, device="auto", conf=None, config=None: stub))
    return LiverCancerPipeline(mod, "cpu", checkpoint, evidence_gate=gate)


def test_no_detection_never_means_no_cancer(monkeypatch):
    if not _have("ct"):
        pytest.skip("checkpoints not available")
    res = _pipe("ct", monkeypatch, {}).run([("C1", np.zeros((64, 64, 5), np.uint8))], "X")
    assert res.status == "no_lesion_detected" and res.label is None and not res.scores
    assert "KHÔNG có nghĩa" in res.message and "không mắc" in res.message
    json.dumps(res.as_dict())


def test_mri_warning_and_unavailable_classifier(monkeypatch):
    if not _have("mri"):
        pytest.skip("checkpoints not available")
    dets = {1: [Detection(10, 10, 40, 40, 0.8)], 2: [Detection(11, 10, 41, 40, 0.7)]}
    res = _pipe("mri", monkeypatch, dets).run([("T2WI", np.random.default_rng(0).integers(0, 255, (64, 64, 4), dtype=np.uint8))], "Y")
    assert res.status == "ok" and res.reliable is False and res.label in CLS["mri"]["classes"]
    assert any("CHƯA ĐẠT" in w and "KHÔNG phải chẩn đoán" in w for w in res.warnings)
    assert res.regions and res.regions[0]["key_slices"] == [1, 2]
    bad = _pipe("mri", monkeypatch, dets, checkpoint=ROOT / "outputs" / "does_not_exist.pt").run(
        [("T2WI", np.zeros((64, 64, 4), np.uint8))], "Y")
    assert bad.status == "classifier_unavailable" and bad.label is None


def test_ct_warns_about_three_class_design(monkeypatch):
    if not _have("ct"):
        pytest.skip("checkpoints not available")
    res = _pipe("ct", monkeypatch, {0: [Detection(5, 5, 30, 30, 0.5)]}).run([("C2", np.zeros((64, 64, 2), np.uint8))], "Z")
    assert res.status == "ok" and res.reliable is True
    assert any("không có lớp 'không tổn thương'" in w for w in res.warnings)


@pytest.mark.parametrize("mod", ["ct", "mri"])
def test_real_val_volume_end_to_end(mod):
    if not _have(mod):
        pytest.skip("checkpoints not available")
    from livercancer.config import load_paths
    from livercancer.data.slices import load_volume_uint8
    from livercancer.data.sources import image_path
    from livercancer.pipeline import LiverCancerPipeline
    man = ROOT / "data" / "processed" / "cls" / mod / "manifest.csv"
    if not man.exists():
        pytest.skip("classification dataset not built")
    with open(man, encoding="utf-8") as f:
        r = next(x for x in csv.DictReader(f) if x["split"] == "val" and x["source"] == "yolo" and x["matches_gt"] == "1")
    src = image_path(load_paths(), mod, r["patient_id"], r["series"])
    if not src.exists():
        pytest.skip("source dataset not available")
    pipe = LiverCancerPipeline(mod, "auto", evidence_gate="off")   # tests detection -> crop -> classifier, not the gate
    res = pipe.run([(r["series"], load_volume_uint8(src))], r["patient_id"])
    assert res.status == "ok" and res.label in CLS[mod]["classes"] and np.isclose(sum(res.scores.values()), 1, atol=1e-3)
    assert all(ro.local.shape == (224, 224) and ro.local.dtype == np.uint8 for ro in res.rois)
    assert res.reliable is (mod == "ct")
    json.dumps(res.as_dict())


def test_vietnamese_outputs(monkeypatch, tmp_path):
    if not _have("ct"):
        pytest.skip("checkpoints not available")
    from livercancer.pipeline_viz import render_case, result_vi, verdict
    nv = dict(zip(CLS["ct"]["classes"], CLS["ct"]["class_names_vi"]))
    ok = _pipe("ct", monkeypatch, {0: [Detection(5, 5, 30, 30, 0.5)]}).run([("C2", np.zeros((64, 64, 2), np.uint8))], "Z")
    none = _pipe("ct", monkeypatch, {}).run([("C2", np.zeros((64, 64, 2), np.uint8))], "Z")
    wrong = next(c for c in CLS["ct"]["classes"] if c != ok.label)
    assert verdict(ok, ok.label) == "ĐÚNG" and verdict(ok, wrong) == "SAI" and verdict(none, "HCC") == "KHÔNG PHÂN LOẠI"
    assert verdict(ok, None) is None
    cls3 = CLS["ct"]["classes"]
    assert verdict(ok, "CN", cls3) == "CN: DƯƠNG TÍNH GIẢ" and verdict(none, "CN", cls3) == "CN: KHÔNG GÁN NHÃN UNG THƯ"
    j = result_vi(ok, wrong, nv)
    assert j["Đánh giá"] == "SAI" and j["Kết luận của hệ thống"] == nv[ok.label] and "Chi tiết kỹ thuật" in j
    assert set(j["Xác suất từng lớp"]) == set(nv.values())
    jn = result_vi(none, "HCC", nv)
    assert jn["Kết luận của hệ thống"] == "Không đưa ra nhãn" and jn["Trạng thái"] == "Không phát hiện vùng nghi ngờ"
    json.dumps(j, ensure_ascii=False)
    for r, name in ((ok, "ok.png"), (none, "none.png")):
        render_case(r, tmp_path / name, "kiểm thử", "HCC", None, nv)
        assert (tmp_path / name).stat().st_size > 10_000


def test_evidence_rules():
    from livercancer.evidence import GateRule, features, passes, rule_from_config
    regs = [{"volume": "C1", "max_conf": 0.45, "n_slices": 6}, {"volume": "C2", "max_conf": 0.42, "n_slices": 3},
            {"volume": "P", "max_conf": 0.2, "n_slices": 20}]
    assert features(regs, 0.4) == {"max_conf": 0.45, "n_volumes_strong": 2, "longest_strong": 6}
    assert passes(regs, GateRule("n_volumes", 0.4, 2)) and not passes(regs, GateRule("n_volumes", 0.4, 3))
    assert passes(regs, GateRule("max_conf", 0.45)) and not passes(regs, GateRule("max_conf", 0.5))
    assert passes(regs, GateRule("longest", 0.4, 6)) and not passes(regs, GateRule("longest", 0.4, 7))
    assert not passes([], GateRule("max_conf", 0.1))
    assert rule_from_config({"enabled": False, "kind": "max_conf", "conf_min": 0.3}) is None
    # chosen on Val and enabled for CT (reports/pipeline_validation/so_sanh_evidence_gate.md); MRI: no controls, not enabled
    assert rule_from_config(CLS["inference"]["ct"]["evidence_gate"]) == GateRule("n_volumes", 0.4, 2)
    assert rule_from_config({**CLS["inference"]["ct"]["evidence_gate"]["single_volume"], "enabled": True}) == GateRule("max_conf", 0.35)
    assert rule_from_config(CLS["inference"]["mri"]["evidence_gate"]) is None


def test_pipeline_evidence_gate_rejects_without_healthy_claim(monkeypatch):
    if not _have("ct"):
        pytest.skip("checkpoints not available")
    weak = {0: [Detection(5, 5, 30, 30, 0.2)]}
    strong = {0: [Detection(5, 5, 30, 30, 0.6)], 1: [Detection(6, 5, 31, 30, 0.6)]}
    vol = np.zeros((64, 64, 3), np.uint8)
    res = _pipe("ct", monkeypatch, weak, gate="off").run([("C1", vol), ("C2", vol)], "G")   # gate off -> label from weak region
    assert res.status == "ok" and res.label is not None
    res = _pipe("ct", monkeypatch, weak).run([("C1", vol), ("C2", vol)], "G")                # CT default config: gate on
    assert res.status == "insufficient_evidence" and res.label is None
    import livercancer.detection.yolo_detector as yd
    from livercancer.pipeline import LiverCancerPipeline
    # 2 phases: conf >= 0.4 needed in >= 2 phases; 1 file: highest conf >= 0.35
    for dets, nvol, want in ((weak, 2, "insufficient_evidence"), (strong, 2, "ok"), (weak, 1, "insufficient_evidence"), (strong, 1, "ok")):
        stub = StubDetector(dets)
        monkeypatch.setattr(yd.LesionDetector, "from_config", classmethod(lambda cls, m, device="auto", conf=None, config=None, s=stub: s))
        pipe = LiverCancerPipeline("ct", "cpu", evidence_gate="on")
        r = pipe.run([(n, vol) for n in ("C1", "C2")[:nvol]], "G")
        assert r.status == want, (dets, nvol, r.status)
        if want != "ok":
            assert r.label is None and not r.scores and "KHÔNG kết luận bệnh nhân khỏe mạnh" in r.message and r.evidence["passed"] is False
