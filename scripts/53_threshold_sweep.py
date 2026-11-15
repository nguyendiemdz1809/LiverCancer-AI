"""GĐ3: confidence-threshold sweep of an analysed YOLO run on Val, to choose the operating threshold. Never touches Test.

Reads <run>/analysis/val_predictions.csv (50_analyze_yolo_run.py, conf >= 0.001, NMS IoU 0.7) and the GT labels of --data.
For every threshold t on a 0.01 grid (matching IoU >= --iou, greedy by confidence, one-to-one):
  box level   TP / FP / FN, precision, recall, F1, F2 (recall weighted 2x: missing a lesion costs more than a FP that the
              classifier can still reject)
  slice level FP per slice, % negative slices (no annotation) with >= 1 box
  series      % lesion volumes (CT patient x phase) / series (MRI) with >= 1 matched box  -> ROI available for GĐ3
  patient     lesion patients with >= 1 matched box; CT: CN patients (no lesion data) with >= 1 box
Selection rule (fixed before looking at Test, applied to Val only): t* = argmax F2 (ties -> higher t). The F1-optimal
threshold is reported next to it for reference.
Outputs: <run>/analysis/threshold_sweep.csv, threshold_choice.json; reports/figures/<tag>/threshold_sweep.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import load_paths  # noqa: E402
from livercancer.evaluation.matching import iou_matrix  # noqa: E402

GRID = np.round(np.arange(0.01, 0.81, 0.01), 2)


def load_gt(data: Path, row: dict) -> np.ndarray:
    w, h = int(row["width"]), int(row["height"])
    out = []
    for line in (data / "labels" / "val" / f"{row['file']}.txt").read_text().splitlines():
        if line.strip():
            _, cx, cy, bw, bh = map(float, line.split())
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out, float).reshape(-1, 4)


def greedy_tp(ious: np.ndarray, conf: np.ndarray, keep: np.ndarray, thr: float) -> tuple[int, int]:
    """(#matched preds, #matched GT) among preds with keep=True; greedy by descending confidence."""
    used = np.zeros(ious.shape[1], bool)
    tp = 0
    for p in np.where(keep)[0][np.argsort(-conf[keep])]:
        cand = np.where(~used & (ious[p] >= thr))[0]
        if len(cand):
            used[cand[np.argmax(ious[p, cand])]] = True
            tp += 1
    return tp, int(used.sum())


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["ct", "mri"], required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="dataset dir the run was validated on")
    ap.add_argument("--tag", required=True, help="figure folder under reports/figures/")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    paths = load_paths()
    with open(args.data / "slices.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    preds = defaultdict(lambda: ([], []))
    with open(args.run / "analysis" / "val_predictions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            preds[r["file"]][0].append([float(r[k]) for k in ("x0", "y0", "x1", "y1")])
            preds[r["file"]][1].append(float(r["conf"]))

    n_t = len(GRID)
    tp = np.zeros(n_t, int)
    fp = np.zeros(n_t, int)
    gt_hit = np.zeros(n_t, int)
    n_gt = 0
    neg_with_fp = np.zeros(n_t, int)
    n_neg = sum(r["kind"] == "neg" for r in rows)
    ser_hit = defaultdict(lambda: np.zeros(n_t, bool))
    pat_hit = defaultdict(lambda: np.zeros(n_t, bool))
    lesion_patients = {r["patient_id"] for r in rows if r["kind"] == "pos"}
    neg_only = {r["patient_id"] for r in rows} - lesion_patients  # CT: CN controls (no lesion data collected)
    negonly_fp = defaultdict(lambda: np.zeros(n_t, bool))
    for r in rows:
        gt = load_gt(args.data, r)
        n_gt += len(gt)
        b, c = preds.get(r["file"], ([], []))
        b, c = np.array(b, float).reshape(-1, 4), np.array(c, float)
        ious = iou_matrix(b, gt)
        for i, t in enumerate(GRID):
            keep = c >= t
            k = int(keep.sum())
            m, g = greedy_tp(ious, c, keep, args.iou) if len(gt) and k else (0, 0)
            tp[i] += m
            fp[i] += k - m
            gt_hit[i] += g
            if r["kind"] == "neg" and k:
                neg_with_fp[i] += 1
            if r["kind"] == "pos":
                ser_hit[(r["patient_id"], r["series"])][i] |= g > 0
                pat_hit[r["patient_id"]][i] |= g > 0
            if r["patient_id"] in neg_only and k:
                negonly_fp[r["patient_id"]][i] = True

    prec = tp / np.maximum(tp + fp, 1)
    rec = gt_hit / max(n_gt, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    f2 = 5 * prec * rec / np.maximum(4 * prec + rec, 1e-9)
    ser = np.mean(np.stack(list(ser_hit.values())), axis=0)
    pat = np.sum(np.stack([pat_hit[p] for p in lesion_patients]), axis=0)
    nof = np.sum(np.stack([negonly_fp[p] for p in neg_only]), axis=0) if neg_only else np.zeros(n_t, int)
    table = []
    for i, t in enumerate(GRID):
        table.append({"conf": float(t), "TP": int(tp[i]), "FP": int(fp[i]), "FN": int(n_gt - gt_hit[i]),
                      "precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                      "F1": round(float(f1[i]), 4), "F2": round(float(f2[i]), 4),
                      "FP_per_slice": round(float(fp[i]) / len(rows), 4),
                      "neg_slices_with_FP_pct": round(100 * float(neg_with_fp[i]) / max(n_neg, 1), 2),
                      "lesion_series_detected_pct": round(100 * float(ser[i]), 2),
                      "lesion_patients_detected": f"{int(pat[i])}/{len(lesion_patients)}",
                      "neg_only_patients_with_FP": f"{int(nof[i])}/{len(neg_only)}"})
    out = args.run / "analysis"
    with open(out / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    i2 = max(range(n_t), key=lambda i: (round(f2[i], 4), GRID[i]))
    i1 = max(range(n_t), key=lambda i: (round(f1[i], 4), GRID[i]))
    choice = {"rule": "argmax F2 on Val (box level, IoU>=%.2f); ties -> higher conf" % args.iou,
              "n_val_slices": len(rows), "n_gt_boxes": n_gt, "n_neg_slices": n_neg,
              "n_lesion_series": len(ser_hit), "n_lesion_patients": len(lesion_patients), "n_neg_only_patients": len(neg_only),
              "F2_optimal": table[i2], "F1_optimal": table[i1],
              "reference": {f"{t:g}": table[int(np.argmin(np.abs(GRID - t)))] for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.4)}}
    (out / "threshold_choice.json").write_text(json.dumps(choice, indent=1, ensure_ascii=False), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for y, lab in ((prec, "Precision"), (rec, "Recall"), (f1, "F1"), (f2, "F2")):
        ax[0].plot(GRID, y, label=lab, lw=1.8 if lab in ("Recall", "F2") else 1.2)
    ax[1].plot(GRID, 100 * ser, label="Volume/series có tổn thương được phát hiện (%)")
    ax[1].plot(GRID, 100 * pat / len(lesion_patients), label="BN có tổn thương được phát hiện (%)")
    ax[1].plot(GRID, 100 * neg_with_fp / max(n_neg, 1), label="Lát không annotation có FP (%)")
    if neg_only:
        ax[1].plot(GRID, 100 * nof / len(neg_only), label="BN CN có ≥1 FP (%)", ls="--")
    for a in ax:
        a.axvline(GRID[i2], color="k", ls="--", lw=1)
        a.axvline(GRID[i1], color="grey", ls=":", lw=1)
        a.set_xlabel("Ngưỡng confidence")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    ax[0].set_title(f"{args.modality.upper()} – mức box (IoU ≥ {args.iou:g}); -- F2 max {GRID[i2]:.2f}, ··· F1 max {GRID[i1]:.2f}", fontsize=9)
    ax[1].set_title(f"{args.modality.upper()} – mức volume/series, BN và lát âm (Val)", fontsize=9)
    fig.tight_layout()
    fig_dir = paths["project"]["reports"] / "figures" / args.tag
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "threshold_sweep.png", dpi=110)
    print(f"F2-optimal conf {GRID[i2]:.2f}: {table[i2]}")
    print(f"F1-optimal conf {GRID[i1]:.2f}: {table[i1]}")
    print("written", out / "threshold_sweep.csv", out / "threshold_choice.json", fig_dir / "threshold_sweep.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
