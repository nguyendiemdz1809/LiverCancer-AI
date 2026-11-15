"""Scoring rules of the official Test evaluation (scripts/96_final_test.py): a case without a label is never correct."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ft():
    spec = importlib.util.spec_from_file_location("ft", ROOT / "scripts" / "96_final_test.py")
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:  # noqa: BLE001 - needs the local dataset configuration
        pytest.skip(f"final test script not importable here: {e}")
    return m


def _c(true, status, label=None):
    return {"true_label": true, "status": status, "pipeline_label": label}


def test_outcomes_and_rejected_never_correct(ft):
    assert ft.outcome(_c("HCC", "ok", "HCC")) == "correct"
    assert ft.outcome(_c("HCC", "ok", "ICC")) == "wrong"
    assert ft.outcome(_c("HCC", "insufficient_evidence")) == "rejected"
    assert ft.outcome(_c("HCC", "no_lesion_detected")) == "no_region"
    cls = ["HCC", "ICC", "CHCC"]
    cases = [_c("HCC", "ok", "HCC"), _c("HCC", "insufficient_evidence"), _c("ICC", "no_lesion_detected"), _c("CHCC", "ok", "HCC")]
    m = ft.cls_metrics(cases, cls)
    assert m["n"] == 4 and m["correct"] == 1 and m["labelled"] == 2
    assert m["accuracy_all"] == 0.25 and m["accuracy_labelled"] == 0.5 and m["coverage"] == 0.5
    assert m["confusion"][0] == [1, 0, 0, 1] and m["confusion"][1] == [0, 0, 0, 1]   # last column = no label
    assert m["per_class"]["HCC"]["recall"] == 0.5 and m["per_class"]["HCC"]["precision"] == 0.5
    lo, hi = ft.wilson(1, 4)
    assert 0 < lo < 0.25 < hi < 1
