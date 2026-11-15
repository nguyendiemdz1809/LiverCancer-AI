"""Run the CT or MRI YOLO detector on one NIfTI volume (no ground truth needed) and save annotated slices.

  .venv\\Scripts\\python scripts\\61_yolo_predict_nifti.py --modality ct --patient P0267 --series C3
  .venv\\Scripts\\python scripts\\61_yolo_predict_nifti.py --modality mri --nifti E:\\...\\MR-xxx_..._T2WI_0000.nii.gz --slices 10 12

Without --slices, every slice is predicted (streamed volume, GPU fp16 when available) and the --top slices with the
highest box confidence are drawn. Display only: this is not an evaluation, and Test patients must not be used to
choose thresholds or models (a warning is printed when the patient belongs to Test).
Outputs: outputs/predictions/adhoc/<name>/slice_kXXX.png + detections.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import load_paths  # noqa: E402
from livercancer.data.slices import load_volume_uint8, slice_image  # noqa: E402
from livercancer.data.sources import image_path  # noqa: E402
from livercancer.detection.yolo_detector import LesionDetector  # noqa: E402


def patient_split(paths: dict, modality: str, patient_id: str) -> str:
    f = paths["project"]["splits"] / f"{modality}_split.csv"
    with open(f, encoding="utf-8") as fh:
        return next((r["split"] for r in csv.DictReader(fh) if r["patient_id"] == patient_id), "unknown")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["ct", "mri"], required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--nifti", type=Path, help="path to an image volume (.nii.gz)")
    src.add_argument("--patient", help="patient id from the dataset (with --series)")
    ap.add_argument("--series", help="CT phase (C1/C2/C3/P) or MRI sequence as in the JSON (e.g. 'T2WI', 'In Phase')")
    ap.add_argument("--slices", type=int, nargs="*", default=None, help="slice indices k (default: scan all)")
    ap.add_argument("--top", type=int, default=4, help="slices to draw when scanning the whole volume")
    ap.add_argument("--conf", type=float, default=None, help="override conf of configs/yolo_inference.yaml")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = ap.parse_args()

    paths = load_paths()
    if args.patient:
        if not args.series:
            ap.error("--patient needs --series")
        nifti, pid = image_path(paths, args.modality, args.patient, args.series), args.patient
    else:
        nifti, pid = args.nifti, args.nifti.name.split("_")[0]
    split = patient_split(paths, args.modality, pid)
    if split == "test":
        print(f"WARNING: {pid} belongs to the TEST split -> display only; never use it to choose thresholds/models.")

    det = LesionDetector.from_config(args.modality, device=args.device, conf=args.conf)
    print(f"{args.modality.upper()} detector on {det.device} (fp16={det.half}), conf={det.conf} | {nifti} | split={split}")
    vol = load_volume_uint8(nifti)
    ks = args.slices if args.slices else list(range(vol.shape[2]))
    imgs = [slice_image(vol, k) for k in ks]
    dets = det.predict(imgs)
    del vol
    if not args.slices:  # keep the most confident slices for drawing
        order = sorted(range(len(ks)), key=lambda i: -max((d.conf for d in dets[i]), default=0))[:args.top]
        order = sorted(i for i in order if dets[i]) or order[:1]
    else:
        order = list(range(len(ks)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out = paths["project"]["outputs"] / "predictions" / "adhoc" / nifti.name.replace(".nii.gz", "")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "detections.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slice_k", "x0", "y0", "x1", "y1", "conf", "label"])
        for k, d in zip(ks, dets):
            for x in d:
                w.writerow([k, round(x.x0, 1), round(x.y0, 1), round(x.x1, 1), round(x.y1, 1), round(x.conf, 4), x.label])
    for i in order:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
        for ax in axes:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]), ax.set_yticks([])
        axes[0].set_title("Ảnh gốc", fontsize=9)
        for x in dets[i]:
            axes[1].add_patch(Rectangle((x.x0, x.y0), x.x1 - x.x0, x.y1 - x.y0, fill=False, ec="#ff3d00", lw=1.6))
            axes[1].text(x.x0, max(x.y0 - 3, 8), f"{x.label} {x.conf:.2f}", fontsize=7,
                         bbox={"facecolor": "#ff3d00", "edgecolor": "none", "pad": 1})
        axes[1].set_title(f"YOLO {args.modality.upper()} (conf≥{det.conf:g}): {len(dets[i])} box", fontsize=9)
        fig.suptitle(f"{nifti.name} · lát k={ks[i]} · split={split} · chỉ để minh họa", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(out / f"slice_k{ks[i]:03d}.png", dpi=110)
        plt.close(fig)
        print(f"k={ks[i]:3d}: " + (", ".join(f"lesion {x.conf:.2f}" for x in dets[i]) or "no box"))
    print(f"{sum(map(len, dets))} boxes on {sum(1 for d in dets if d)}/{len(ks)} slices -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
