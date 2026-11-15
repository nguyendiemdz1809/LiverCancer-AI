"""Export the measured numbers shown on the website's "Kết quả nghiên cứu" page -> web/src/data/research.json.

Every value is read from a result file (no number is typed by hand); each block carries its source path and its
denominator. Aggregates only: no patient identifier is exported (the site may be public).
  Validation : outputs/predictions/pipeline_validation_gate (locked configuration = CT evidence gate on), scored with
               the same functions as the Test report (scripts/96_final_test.py)
  Test       : reports/final_test/metrics_test.json (official run, once)
  YOLO       : outputs/runs/yolo_<mod>/<run>/metrics_val.json and analysis/threshold_choice.json
  Hybrid ROI : outputs/runs/cls/<run>/eval_val/metrics_val.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import CONFIG_DIR, load_yaml  # noqa: E402

spec = importlib.util.spec_from_file_location("ft", ROOT / "scripts" / "96_final_test.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)
OUT = ROOT / "web" / "src" / "data" / "research.json"


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def r(x, n=4):
    return round(float(x), n)


def cls_block(m: dict, classes: list[str], names: dict) -> dict:
    return {"n": m["n"], "correct": m["correct"], "labelled": m["labelled"], "accuracy_all": r(m["accuracy_all"]),
            "accuracy_labelled": r(m["accuracy_labelled"]), "macro_f1": r(m["macro_f1"]), "balanced_accuracy": r(m["balanced_accuracy"]),
            "per_class": [{"code": c, "name": names[c], "precision": r(m["per_class"][c]["precision"]), "recall": r(m["per_class"][c]["recall"]),
                           "f1": r(m["per_class"][c]["f1"]), "support": m["per_class"][c]["support"]} for c in classes]}


def det_block(d: dict) -> dict:
    keys = ("volumes_with_lesion", "volumes_lesion_detected", "lesions", "lesions_detected", "small", "small_detected",
            "patients_with_gt", "patients_no_true_lesion_found", "false_tracks_per_volume")
    return {k: (r(d[k], 2) if k == "false_tracks_per_volume" else d[k]) for k in keys}


def main() -> None:
    cfg = load_yaml(CONFIG_DIR / "cls.yaml")
    yc = load_yaml(CONFIG_DIR / "yolo_inference.yaml")
    test = json.loads((ROOT / "reports/final_test/metrics_test.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "reports/final_test/locked_config.json").read_text(encoding="utf-8"))
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "locked_commit": lock["git"]["commit"][:7],
           "test_run": {"started": test["run"]["started"], "finished": test["run"].get("finished")}, "modalities": {}}
    for mod in ("ct", "mri"):
        classes = cfg[mod]["classes"]
        from livercancer.pipeline_viz import short
        names = {c: short(n) for c, n in zip(classes, cfg[mod]["class_names_vi"])}
        val_cases = ft.load_cases(ft.VAL_CACHE, mod)
        vlab = [c for c in val_cases if c["true_label"] in classes]
        vctrl = [c for c in val_cases if c["true_label"] not in classes]
        t = test[mod]
        vstrict = ft.counts(vlab)["correct"] - sum(1 for c in vlab if ft.outcome(c) == "correct" and not any(x["matched"] for x in c["tracks"]))
        block = {
            "classes": [{"code": c, "name": names[c]} for c in classes],
            "val": {"n_patients": len(val_cases), "n_lesion_patients": len(vlab), "counts": ft.counts(vlab), "strict_correct": vstrict,
                    "classification": cls_block(ft.cls_metrics(vlab, classes), classes, names),
                    "detection": det_block(ft.detection(val_cases)), "source": rel(ft.VAL_CACHE / mod / "cases")},
            "test": {"n_patients": t["n_patients"], "n_lesion_patients": t["n_lesion_patients"], "counts": t["counts"],
                     "strict_correct": t["strict_correct"], "accuracy_ci95": [r(x) for x in t["pipeline"]["accuracy_all_ci95"]],
                     "macro_f1_ci95": [r(x) for x in t["bootstrap"]["macro_f1"]],
                     "classification": cls_block(t["pipeline"], classes, names), "oracle": {"n": t["oracle"]["n"], "correct": t["oracle"]["correct"]},
                     "detection": det_block(t["detection"]), "source": "reports/final_test/metrics_test.json"},
            "yolo": {}, "roi_classifier_val": None,
        }
        y = json.loads((ROOT / yc[mod]["run"] / "metrics_val.json").read_text(encoding="utf-8"))
        tc = json.loads((ROOT / yc[mod]["conf_source"]).read_text(encoding="utf-8"))
        block["yolo"] = {"model": "YOLO11n", "imgsz": y["imgsz"], "best_epoch": y["best_epoch"], "epochs_run": y["epochs_run"],
                         "precision": r(y["metrics_val"]["precision"]), "recall": r(y["metrics_val"]["recall"]),
                         "map50": r(y["metrics_val"]["mAP50"]), "map50_95": r(y["metrics_val"]["mAP50-95"]),
                         "conf": yc[mod]["conf"], "conf_rule": "F2 lớn nhất trên Validation", "n_val_slices": tc.get("n_val_slices"),
                         "source": rel(ROOT / yc[mod]["run"] / "metrics_val.json")}
        mv = json.loads((ROOT / cfg["inference"][mod]["metrics"]).read_text(encoding="utf-8"))
        roi = mv["gt_roi"]["roi"]
        block["roi_classifier_val"] = {"n_rois": roi["n"], "accuracy": r(roi["accuracy"]), "macro_f1": r(roi["macro_f1"]),
                                       "source": cfg["inference"][mod]["metrics"], "note": "Mức ROI, vùng cắt theo nhãn chuẩn (oracle)"}
        if mod == "ct":
            c = t["controls"]
            block["test"]["controls"] = {k: c[k] for k in ("n", "cancer_label", "rejected", "no_region", "ungated_cancer_label")}
            block["test"]["controls"]["cancer_label_ci95"] = [r(x) for x in c["cancer_label_ci95"]]
            block["val"]["controls"] = {"n": len(vctrl), "cancer_label": sum(x["status"] == "ok" for x in vctrl),
                                        "rejected": sum(x["status"] == "insufficient_evidence" for x in vctrl),
                                        "no_region": sum(x["status"] == "no_lesion_detected" for x in vctrl)}
            g = lock["modalities"]["ct"]["evidence_gate"]
            block["evidence_gate"] = {"conf_min": g["conf_min"], "k": g["k"], "single_conf_min": g["single_volume"]["conf_min"]}
            block["test"]["ungated_correct"] = t["ungated_ablation"]["correct"]
        else:
            block["test"]["groups"] = t["groups"]
        out["modalities"][mod] = block
    mm = ROOT / "outputs" / "runs" / "modality" / "modality_cnn_s42" / "metrics.json"
    if mm.exists():   # CT / MRI recognition of 2D inputs (separate intake model, Val-B report; not the locked pipeline)
        m = json.loads(mm.read_text(encoding="utf-8"))
        out["modality_recognition"] = {
            "threshold": m["threshold"], "n_patients_valB": m["n_patients_valB"],
            "valB": {v: {k: (r(x) if isinstance(x, float) else x) for k, x in b.items()} for v, b in m["valB"].items()},
            "valB_all": {k: (r(x) if isinstance(x, float) else x) for k, x in m["valB_all"].items()},
            "nifti_header_rule": m["nifti_header_rule"], "yolo_on_synthetic_screenshots": m.get("yolo_on_synthetic_screenshots"),
            "source": rel(mm)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(out, indent=1, ensure_ascii=False)
    import re
    assert not re.search(r"\b(P\d{4}|MR-?\d{4,})\b", txt), "patient identifier in web data"
    OUT.write_text(txt, encoding="utf-8")
    print("written", OUT)


if __name__ == "__main__":
    main()
