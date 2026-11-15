"""GĐ3: series- and patient-level detection of an analysed YOLO run on Val. Never touches Test.

A series (MRI) / volume (CT phase) counts as detected at a threshold when at least one of its positive slices has a
prediction matched to GT (IoU >= --iou, same greedy matching as 50_analyze_yolo_run.py). This is the quantity that
matters for GĐ3 (automatic ROI zoom): a series without any correct box gives no correct ROI.
Inputs: <run>/analysis/val_predictions.csv (written by 50_analyze_yolo_run.py), GT labels of --data.
Output: <run>/analysis/series_level_detection.json
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import load_paths  # noqa: E402

_spec = importlib.util.spec_from_file_location("analyze", Path(__file__).with_name("50_analyze_yolo_run.py"))
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", choices=["ct", "mri"], required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="dataset dir of the run (default: pilot set)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.25, 0.4])
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    data = args.data or load_paths()["project"]["processed"] / "pilot" / f"yolo_{args.modality}"
    pred_csv = args.run / "analysis" / "val_predictions.csv"
    if not pred_csv.exists():
        print(f"missing {pred_csv} — run 50_analyze_yolo_run.py first")
        return 1

    with open(data / "slices.csv", encoding="utf-8") as f:
        slices = [r for r in csv.DictReader(f) if r["split"] == "val" and r["kind"] == "pos"]
    preds = defaultdict(list)
    with open(pred_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            preds[r["file"]].append([float(r[k]) for k in ("x0", "y0", "x1", "y1", "conf")])
    gts = {r["file"]: analyze.load_gt(data, r) for r in slices}

    out = {"run": str(args.run), "data": str(data), "iou_match": args.iou}
    for t in args.thresholds:
        hit, pat_hit, label = defaultdict(bool), defaultdict(bool), {}
        for r in slices:
            p = np.array([q for q in preds.get(r["file"], []) if q[4] >= t]).reshape(-1, 5)
            _, used = analyze.match(gts[r["file"]], p[:, :4], p[:, 4], args.iou)
            key = (r["patient_id"], r["series"])
            label[key] = r["patient_label"]
            hit[key] |= bool(used.any())
            pat_hit[r["patient_id"]] |= bool(used.any())
        by_seq, by_cls = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        for k, h in hit.items():
            by_seq[k[1]][0] += h
            by_seq[k[1]][1] += 1
            by_cls[label[k]][0] += h
            by_cls[label[k]][1] += 1
        out[str(t)] = {"series_detected": f"{sum(hit.values())}/{len(hit)}",
                       "patients_detected": f"{sum(pat_hit.values())}/{len(pat_hit)}",
                       "patients_missed": sorted(k for k, v in pat_hit.items() if not v),
                       "series_missed": sorted(f"{k[0]}_{k[1]}" for k, v in hit.items() if not v),
                       "by_series": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_seq.items())},
                       "by_class": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cls.items())}}
        print(f"conf>={t}: series {out[str(t)]['series_detected']}, patients {out[str(t)]['patients_detected']}, "
              f"missed {out[str(t)]['patients_missed']}")
    dst = args.run / "analysis" / "series_level_detection.json"
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("written", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
