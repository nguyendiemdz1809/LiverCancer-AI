"""Evaluation of a trained classifier on Validation: ROI level and patient level, GT ROIs and YOLO (pipeline) ROIs.

The patient-level aggregation is chosen here, on Validation YOLO ROIs (the real pipeline input), by macro-F1.
Test: evaluate_test() refuses to run without explicit confirmation and writes a lock so it can run only once.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from livercancer.classification.data import RoiDataset, read_manifest
from livercancer.classification.metrics import AGGREGATIONS, choose_aggregation, compute_metrics, patient_level, plot_confusion
from livercancer.classification.models import build_model
from livercancer.classification.train import predict


def load_classifier(ckpt_path: str | Path, device: str = "auto") -> tuple[torch.nn.Module, dict, str]:
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck["cfg"]
    model = build_model(c["model"], len(ck["classes"]), pretrained=False, fusion=c.get("fusion", "concat"),
                        fusion_dim=c.get("fusion_dim", 256), dropout=c.get("dropout", 0.3))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck, device


def _probs(model, rows, root, device, amp, bs=32, workers=0) -> np.ndarray:
    if not rows:
        return np.zeros((0, 1))
    return predict(model, DataLoader(RoiDataset(rows, root), batch_size=bs, shuffle=False, num_workers=workers), device, amp)[0]


def _block(rows, prob, classes, method=None) -> dict:
    y = np.array([int(r["label_id"]) for r in rows])
    out = {"roi": compute_metrics(y, prob, classes)}
    pid = [r["patient_id"] for r in rows]
    if method:
        _, yl, pp = patient_level(pid, y, prob, method)
        out["patient"] = compute_metrics(yl, pp, classes)
    return out


def evaluate_val(run_dir: str | Path, data_root: str | Path, device: str = "auto", amp: bool = True, workers: int = 0,
                 out_subdir: str = "eval_val") -> dict:
    run_dir, root = Path(run_dir), Path(data_root)
    model, ck, device = load_classifier(run_dir / "best.pt", device)
    classes = ck["classes"]
    amp = amp and device.startswith("cuda")
    man = root / "manifest.csv"
    gt = read_manifest(man, "val", ("gt",))
    yo_all = read_manifest(man, "val", ("yolo",), labelled_only=False)
    yo = [r for r in yo_all if int(r["label_id"]) >= 0]
    p_gt, p_yo = _probs(model, gt, root, device, amp, workers=workers), _probs(model, yo, root, device, amp, workers=workers)

    res = {"run": str(run_dir), "checkpoint_epoch": ck["epoch"], "classes": classes, "split": "val",
           "time": datetime.now().isoformat(timespec="seconds")}
    y_yo = np.array([int(r["label_id"]) for r in yo])
    best, per_method = choose_aggregation([r["patient_id"] for r in yo], y_yo, p_yo, classes) if yo else ("mean_prob", {})
    res["aggregation"] = {"chosen": best, "rule": "max patient macro-F1 on Val YOLO ROIs", "candidates": list(AGGREGATIONS),
                          "macro_f1": {m: round(v["macro_f1"], 4) for m, v in per_method.items()}}
    res["gt_roi"] = _block(gt, p_gt, classes, best)
    res["yolo_roi_all"] = _block(yo, p_yo, classes, best) if yo else None
    matched = [i for i, r in enumerate(yo) if r["matches_gt"] == "1"]
    res["yolo_roi_matched"] = _block([yo[i] for i in matched], p_yo[matched], classes, best) if matched else None
    # pipeline coverage: labelled Val patients without any YOLO ROI cannot be classified -> counted as errors
    pats_all = {r["patient_id"]: int(r["label_id"]) for r in gt}
    covered = {r["patient_id"] for r in yo}
    missing = sorted(set(pats_all) - covered)
    if yo:
        pats, yl, pp = patient_level([r["patient_id"] for r in yo], y_yo, p_yo, best)
        correct = int((pp.argmax(1) == yl).sum())
        res["pipeline_patient"] = {"patients_with_label": len(pats_all), "patients_with_yolo_roi": len(pats),
                                   "patients_without_roi": missing, "accuracy_counting_missing_as_wrong":
                                   round(correct / max(1, len(pats_all)), 4)}
    ctrl = sorted({r["patient_id"] for r in yo_all if int(r["label_id"]) < 0})
    res["controls_with_yolo_roi"] = {"n": len(ctrl), "note": "CT controls (CN): any ROI here is a false lesion; the classifier "
                                     "has no 'no lesion' class, so these patients would receive a cancer label"}
    out = run_dir / out_subdir   # e.g. eval_val_local: never overwrite the results downloaded from Colab
    out.mkdir(exist_ok=True)
    (out / "metrics_val.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for key, title in (("gt_roi", "Val – ROI từ GT"), ("yolo_roi_all", "Val – ROI từ YOLO")):
        if res.get(key):
            for lvl in ("roi", "patient"):
                if lvl in res[key]:
                    m = res[key][lvl]
                    plot_confusion(m["confusion_matrix"], classes, f"{title} – mức {'ROI' if lvl == 'roi' else 'bệnh nhân'} "
                                   f"(macro-F1 {m['macro_f1']:.3f})", out / f"cm_{key}_{lvl}.png")
    return res


def evaluate_test(*_, confirm_final: bool = False, lock_path: str | Path = "", **__) -> None:
    """Final, single Test evaluation. Not implemented as an automatic step on purpose: it runs the full pipeline
    (YOLO -> Tumor Zoom -> classifier) on Test ONCE, after every model and threshold is frozen on Validation."""
    if not confirm_final:
        raise PermissionError("Test is locked: evaluate only once, after all choices are frozen on Validation "
                              "(call with confirm_final=True in the final-evaluation step).")
    if lock_path and Path(lock_path).exists():
        raise PermissionError(f"Test was already evaluated ({lock_path}); a second evaluation is not allowed.")
    raise NotImplementedError("final Test evaluation is the last step of the project (docs/PHASE5_CLASSIFICATION_PLAN.md)")
