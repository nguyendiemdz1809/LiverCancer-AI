"""GĐ3: finer FP/FN breakdown of a CT (or MRI) YOLO run on the pilot Val split (analysis only — never used for inference).

Uses <run>/analysis/val_predictions.csv from 50_analyze_yolo_run.py. For each confidence threshold:
  - FP split into: localisation errors (IoU with some GT in [0.1, 0.5)), FP inside liver ∪ lesion (box centre),
    FP outside the liver. The GT liver mask is read ONLY to describe errors on Val; the pipeline never uses it.
  - FN split into: near misses (best IoU with a prediction in [0.1, 0.5)) vs. no overlapping prediction.
  - recall at IoU 0.3 / 0.5 / 0.7.
Output: <run>/analysis/ct_fp_fn_breakdown.json (+ printed tables).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import load_paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

ana = import_module("50_analyze_yolo_run")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--modality", choices=["ct", "mri"], default="ct",
                    help="mri: no liver mask exists, FP are split by slice kind only")
    ap.add_argument("--data", type=Path, default=None, help="dataset dir of the run (default: pilot set)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.25, 0.4])
    args = ap.parse_args()

    paths = load_paths()
    data = args.data or paths["project"]["processed"] / "pilot" / f"yolo_{args.modality}"
    data_dir = Path(paths["ct"]["data_dir"])
    with open(data / "slices.csv", encoding="utf-8") as f:
        slices = {r["file"]: r for r in csv.DictReader(f) if r["split"] == "val"}
    preds = defaultdict(lambda: {"xyxy": [], "conf": []})
    with open(args.run / "analysis" / "val_predictions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            preds[r["file"]]["xyxy"].append([float(r[k]) for k in ("x0", "y0", "x1", "y1")])
            preds[r["file"]]["conf"].append(float(r["conf"]))
    preds = {k: {"xyxy": np.array(v["xyxy"]).reshape(-1, 4), "conf": np.array(v["conf"])} for k, v in preds.items()}
    gts = {k: ana.load_gt(data, r) for k, r in slices.items()}

    # liver ∪ lesion region per (patient, phase), loaded lazily for slices that need it
    region_cache: dict = {}

    def region(r: dict) -> np.ndarray:
        key = (r["patient_id"], r["series"])
        if key not in region_cache:
            region_cache.clear()  # keep one volume in memory
            liver = np.asanyarray(nib.load(str(data_dir / "liver_mask_files" /
                                                f"{r['patient_id']}-livermask_{r['series']}.nii.gz")).dataobj.get_unscaled()) > 0
            if r["patient_label"] != "CN":
                liver |= np.asanyarray(nib.load(str(data_dir / "mask_files" /
                                                     f"{r['patient_id']}_mask_{r['series']}.nii.gz")).dataobj.get_unscaled()) > 0
            region_cache[key] = liver
        return region_cache[key][:, :, int(r["slice_k"])]  # (i, j); image pixel (x=i, y=j)

    out = {}
    order = sorted(slices, key=lambda k: (slices[k]["patient_id"], slices[k]["series"], int(slices[k]["slice_k"])))
    for thr in args.thresholds:
        c = defaultdict(int)
        recall_iou = {0.3: [0, 0], 0.5: [0, 0], 0.7: [0, 0]}
        for name in order:
            r, gt = slices[name], gts[name]
            p = preds.get(name, {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0)})
            keep = p["conf"] >= thr
            pb, pc = p["xyxy"][keep], p["conf"][keep]
            is_tp, hit = ana.match(gt, pb, pc, 0.5)
            ious = ana.iou_matrix(pb, gt)  # (pred, gt)
            for t in recall_iou:
                recall_iou[t][0] += int(ana.match(gt, pb, pc, t)[1].sum())
                recall_iou[t][1] += len(gt)
            fps = np.where(~is_tp)[0]
            if len(fps):
                reg = region(r) if args.modality == "ct" else None
                for i in fps:
                    best = ious[i].max() if len(gt) else 0.0
                    if best >= 0.1:
                        c["FP_localisation_IoU_0.1-0.5"] += 1
                        continue
                    if reg is None:
                        c[f"FP_other_on_{r['kind']}_slices"] += 1
                        continue
                    cx = int(np.clip((pb[i, 0] + pb[i, 2]) / 2, 0, reg.shape[0] - 1))
                    cy = int(np.clip((pb[i, 1] + pb[i, 3]) / 2, 0, reg.shape[1] - 1))
                    side = "pos" if r["kind"] == "pos" else ("CN" if r["patient_label"] == "CN" else "neg_cancer")
                    c[f"FP_{'inside' if reg[cx, cy] else 'outside'}_liver_{side}"] += 1
            for g in np.where(~hit)[0]:
                best = ious[:, g].max() if len(pb) else 0.0
                c["FN_near_miss_IoU_0.1-0.5" if best >= 0.1 else "FN_no_overlapping_prediction"] += 1
        out[str(thr)] = {"counts": dict(sorted(c.items())),
                         "recall_by_IoU": {str(t): round(v[0] / v[1], 4) for t, v in recall_iou.items()}}
        print(f"conf>={thr}:", json.dumps(out[str(thr)], ensure_ascii=False))
    dst = args.run / "analysis" / f"{args.modality}_fp_fn_breakdown.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("written", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
