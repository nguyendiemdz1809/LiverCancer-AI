"""Lesion classification: metrics, aggregation, models, training loop (resume, OOM fallback), inference interface, notebook."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.classification.data import class_weights, read_manifest, to_tensor  # noqa: E402
from livercancer.classification.metrics import aggregate, choose_aggregation, compute_metrics, patient_level  # noqa: E402
from livercancer.classification.models import build_model  # noqa: E402

CLS_ROOT = ROOT / "data" / "processed" / "cls"


# ----------------------------------------------------------------------------- metrics
def test_metrics_known_values():
    y = [0, 0, 1, 1, 2, 2]
    prob = np.eye(3)[[0, 1, 1, 1, 2, 0]]
    m = compute_metrics(y, prob, ["a", "b", "c"])
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"]["a"]["recall_sensitivity"] == 0.5 and m["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["c"]["specificity"] == 1.0          # no false 'c'
    assert m["per_class"]["a"]["specificity"] == pytest.approx(3 / 4)
    assert m["roc_auc_macro_ovr"] is not None


def test_auc_absent_when_class_missing():
    m = compute_metrics([0, 0, 1], np.eye(3)[[0, 0, 1]], ["a", "b", "c"])
    assert m["roc_auc_macro_ovr"] is None and m["per_class"]["c"]["support"] == 0


def test_aggregation_and_patient_level():
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])
    assert aggregate(p, "mean_prob").argmax() == 1 and aggregate(p, "majority_vote").argmax() == 1
    assert aggregate(p, "top3_mean").shape == (2,) and np.isclose(aggregate(p, "conf_weighted").sum(), 1)
    pats, yl, pp = patient_level(["A", "A", "B"], [0, 0, 1], p, "mean_prob")
    assert pats == ["A", "B"] and yl.tolist() == [0, 1] and pp.shape == (2, 2)
    with pytest.raises(ValueError):
        patient_level(["A", "A"], [0, 1], p[:2], "mean_prob")      # one patient, two labels -> refused
    best, res = choose_aggregation(["A", "A", "B"], [0, 0, 1], p, ["x", "y"])
    assert best in res


def test_class_weights():
    w = class_weights([0, 0, 0, 1], 3, "inverse")
    assert w[2] == 0 and w[1] > w[0] and np.isclose(w[:2].mean(), 1)
    assert torch.allclose(class_weights([0, 1], 2, "none"), torch.ones(2))


def test_to_tensor_imagenet_norm():
    t = to_tensor(np.full((4, 4), 255, np.uint8))
    assert t.shape == (3, 4, 4) and torch.allclose(t[:, 0, 0], (1 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225]))


# ----------------------------------------------------------------------------- models
@pytest.mark.parametrize("name", ["resnet18", "swin_t", "hybrid"])
def test_models_forward(name):
    m = build_model(name, 3, pretrained=False, fusion="gated" if name == "hybrid" else "concat").eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert m(local=x, context=x).shape == (2, 3)


# ----------------------------------------------------------------------------- data + training (real ROI manifest)
def _manifest(mod):
    p = CLS_ROOT / mod / "manifest.csv"
    if not p.exists():
        pytest.skip(f"ROI dataset not built ({p})")
    return p


@pytest.mark.parametrize("mod", ["ct", "mri"])
def test_manifest_patient_split_and_labels(mod):
    from livercancer.config import load_yaml
    rows = list(csv.DictReader(open(_manifest(mod), encoding="utf-8")))
    split = {r["patient_id"]: r for r in csv.DictReader(open(ROOT / "data" / "splits" / f"{mod}_split.csv", encoding="utf-8"))}
    classes = load_yaml(ROOT / "configs" / "cls.yaml")[mod]["classes"]
    assert {r["split"] for r in rows} <= {"train", "val"}                          # no Test
    for r in rows:
        assert split[r["patient_id"]]["split"] == r["split"]                       # frozen patient-level split
        assert r["label"] == split[r["patient_id"]]["label"]                       # label from the dataset, not from boxes
        assert int(r["label_id"]) == (classes.index(r["label"]) if r["label"] in classes else -1)
    gt = [r for r in rows if r["source"] == "gt"]
    assert gt and all(int(r["label_id"]) >= 0 for r in gt)                       # CT controls never have GT lesions
    assert {r["split"] for r in gt} == {"train", "val"}


def _cfg(tmp_path, **kw):
    return {"modality": "ct", "model": "resnet18", "data_root": str(CLS_ROOT / "ct"), "out_dir": str(tmp_path / "run"),
            "epochs": 1, "patience": 3, "batch_size": 4, "max_train_rois": 12, "max_val_rois": 9, "pretrained": False,
            "num_workers": 0, "device": "cpu", "amp": False, **kw}


def test_train_resume_and_outputs(tmp_path):
    _manifest("ct")
    from livercancer.classification.train import train
    s1 = train(_cfg(tmp_path))
    out = tmp_path / "run"
    for f in ("best.pt", "last.pt", "history.csv", "config.json", "curves.png", "metrics_val_best.json", "cm_val_roi_best.png",
              "val_predictions_best.csv", "summary.json"):
        assert (out / f).exists(), f
    assert s1["epochs_run"] == 1
    s2 = train(_cfg(tmp_path, epochs=2))                                          # resumes from last.pt
    hist = list(csv.DictReader(open(out / "history.csv")))
    assert s2["epochs_run"] == 2 and [h["epoch"] for h in hist] == ["1", "2"]
    with pytest.raises(RuntimeError):
        train(_cfg(tmp_path, model="swin_t", epochs=3))                           # another model must not reuse the run


def test_train_halves_batch_on_cuda_oom(tmp_path, monkeypatch):
    _manifest("ct")
    import livercancer.classification.train as T
    real = T.build_model
    calls = {"n": 0}

    def build(*a, **k):
        m = real(*a, **k)
        fwd = m.forward

        def forward(*fa, **fk):
            if m.training and calls["n"] == 0:
                calls["n"] += 1
                raise torch.cuda.OutOfMemoryError("simulated")
            return fwd(*fa, **fk)
        m.forward = forward
        return m
    monkeypatch.setattr(T, "build_model", build)
    s = T.train(_cfg(tmp_path))
    assert s["oom_events"] == [{"epoch": 1, "new_batch_size": 2, "accum_steps": 2}] and s["epochs_run"] == 1


def test_read_manifest_filters():
    p = _manifest("ct")
    tr = read_manifest(p, "train", ("gt",))
    assert tr and all(r["split"] == "train" and r["source"] == "gt" for r in tr)
    yo = read_manifest(p, "val", ("yolo",), labelled_only=False)
    assert all(r["source"] == "yolo" for r in yo)


# ----------------------------------------------------------------------------- inference interface + Test lock
def test_inference_status_without_model():
    from livercancer.classification.inference import LesionClassifier, RoiInput
    clf = LesionClassifier.load("ct", checkpoint=ROOT / "outputs" / "no_such_checkpoint.pt")
    r = clf.classify([RoiInput(np.zeros((224, 224), np.uint8), np.zeros((224, 224), np.uint8))])
    assert r.status == "model_unavailable" and r.label is None and "chưa" in r.message


@pytest.mark.parametrize("mod,reliable", [("ct", True), ("mri", False)])
def test_configured_checkpoints_load_with_quality_flag(mod, reliable):
    from livercancer.classification.inference import LesionClassifier
    from livercancer.config import load_yaml
    inf = load_yaml(ROOT / "configs" / "cls.yaml")["inference"][mod]
    if not inf.get("checkpoint") or not (ROOT / inf["checkpoint"]).exists():
        pytest.skip("trained checkpoint not downloaded")
    clf = LesionClassifier.load(mod, device="cpu")            # configured checkpoint, sha256 verified
    assert clf.available and clf.reliable is reliable and clf.aggregation == inf["aggregation"]
    assert clf.metrics_path is not None and clf.metrics_path.exists()


def test_inference_with_checkpoint(tmp_path):
    _manifest("ct")
    from livercancer.classification.inference import LesionClassifier, RoiInput
    from livercancer.classification.train import train
    train(_cfg(tmp_path))
    clf = LesionClassifier.load("ct", checkpoint=tmp_path / "run" / "best.pt", device="cpu")
    assert clf.available and clf.classify([]).status == "no_roi"
    bad = clf.classify([RoiInput(np.zeros((100, 100), np.uint8), np.zeros((224, 224), np.uint8))])
    assert bad.status == "invalid_input" and bad.label is None
    rois = [RoiInput(np.full((224, 224), v, np.uint8), np.full((224, 224), v, np.uint8), f"r{v}", t, v, 0.5) for v, t in ((10, 0), (90, 0), (200, 1))]
    r = clf.classify(rois)
    assert r.status == "ok" and r.label in clf.classes and np.isclose(sum(r.scores.values()), 1, atol=1e-3)
    assert len(r.per_roi) == 3 and [t["track_id"] for t in r.per_track] == [0, 1]
    wrong = LesionClassifier.load("mri", checkpoint=tmp_path / "run" / "best.pt", device="cpu")   # CT checkpoint for MRI
    assert not wrong.available and wrong.classify(rois).status == "model_unavailable"


def test_test_split_locked():
    from livercancer.classification.evaluate import evaluate_test
    with pytest.raises(PermissionError):
        evaluate_test()


def test_cls_notebook_structure():
    import ast
    nb = json.loads((ROOT / "notebooks" / "05_train_classifier_colab.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("!", "%"))))
    for key in ("torch.cuda.is_available()", "drive.mount(", "SHA256SUMS_cls.txt", "MODEL", "'hybrid'", "'resnet18'", "'swin_t'",
                "from livercancer.classification.train import train", "evaluate_val", "SMOKE", "BATCH", "SEED", "experiments.csv",
                "summary.json", "Test must not be in the dataset"):
        assert key in src, key
    assert "split='test'" not in src and "evaluate_test" not in src
