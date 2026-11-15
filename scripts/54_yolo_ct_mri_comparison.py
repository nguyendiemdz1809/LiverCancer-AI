"""GĐ3: side-by-side figure of the CT and MRI YOLO runs on their own Val sets + copy of the Ultralytics charts to reports/.

The two runs use different datasets (modality, patients, labels, sampling), so the figure shows each model's behaviour
on its own Val set; it is NOT a ranking of CT against MRI. Inputs: <run>/metrics_val.json, <run>/analysis/threshold_sweep.csv
(53_threshold_sweep.py), Ultralytics plots in <run>/ and <run>/analysis/val_conf{0.001,0.25}/.
Outputs: reports/figures/yolo_e0_ct_vs_mri.png, reports/figures/<tag>/ultralytics_*.png
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import load_paths  # noqa: E402

CHARTS = {"results.png": "ultralytics_results.png", "BoxPR_curve.png": "ultralytics_PR_curve.png",
          "BoxF1_curve.png": "ultralytics_F1_curve.png"}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ct-run", type=Path, default=Path("outputs/runs/yolo_ct/yolo11n_e50_img512_baseline"))
    ap.add_argument("--mri-run", type=Path, default=Path("outputs/runs/yolo_mri/yolo11n_e50_img512_baseline"))
    ap.add_argument("--ct-tag", default="ct_e0")
    ap.add_argument("--mri-tag", default="mri_e0")
    ap.add_argument("--mark", type=float, nargs="*", default=[0.12, 0.25], help="thresholds to mark")
    args = ap.parse_args()
    reports = load_paths()["project"]["reports"]

    for run, tag in ((args.ct_run, args.ct_tag), (args.mri_run, args.mri_tag)):
        dst = reports / "figures" / tag
        dst.mkdir(parents=True, exist_ok=True)
        for src, name in CHARTS.items():
            if (run / src).exists():
                shutil.copy2(run / src, dst / name)
        for c in ("0.001", "0.25"):
            for src in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
                p = run / "analysis" / f"val_conf{c}" / src
                if p.exists():
                    shutil.copy2(p, dst / f"ultralytics_{src.replace('.png', '')}_conf{c}.png")

    sweeps = {}
    for mod, run in (("CT", args.ct_run), ("MRI", args.mri_run)):
        with open(run / "analysis" / "threshold_sweep.csv", encoding="utf-8") as f:
            sweeps[mod] = list(csv.DictReader(f))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    style = {"CT": "-", "MRI": "--"}
    for mod, rows in sweeps.items():
        t = [float(r["conf"]) for r in rows]
        ax[0].plot(t, [float(r["recall"]) for r in rows], style[mod], label=f"{mod} recall")
        ax[0].plot(t, [float(r["precision"]) for r in rows], style[mod], label=f"{mod} precision", alpha=0.6)
        ax[1].plot(t, [float(r["lesion_series_detected_pct"]) for r in rows], style[mod], label=f"{mod}: volume/series có tổn thương được phát hiện")
        ax[2].plot(t, [float(r["neg_slices_with_FP_pct"]) for r in rows], style[mod], label=f"{mod}: lát không annotation có FP")
    for a, ttl, yl in zip(ax, ("Mức box (IoU ≥ 0.5)", "Mức volume (CT: BN × pha) / series (MRI)", "Lát không annotation có ≥ 1 box"),
                          ("", "%", "%")):
        for m in args.mark:
            a.axvline(m, color="grey", ls=":", lw=1)
        a.set_title(ttl, fontsize=10)
        a.set_xlabel("Ngưỡng confidence")
        a.set_ylabel(yl)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("YOLO11n E0 trên Val của từng modality. Hai bộ dữ liệu KHÁC NHAU (PLC-CECT CT ≠ LLD-MMRI MRI): "
                 "chỉ so sánh hành vi theo ngưỡng, không xếp hạng", fontsize=10)
    fig.tight_layout()
    out = reports / "figures" / "yolo_e0_ct_vs_mri.png"
    fig.savefig(out, dpi=110)
    print("written", out, "and Ultralytics charts in", reports / "figures" / args.ct_tag, reports / "figures" / args.mri_tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
