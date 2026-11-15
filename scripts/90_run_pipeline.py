"""Run the full AI pipeline (YOLO -> Tumor Zoom -> classification) on real CT / MRI images of ONE patient.

  .venv\\Scripts\\python scripts\\90_run_pipeline.py --modality ct --patient P0267                 # all available CT phases
  .venv\\Scripts\\python scripts\\90_run_pipeline.py --modality mri --patient MR179970 --series T2WI "C+V"
  .venv\\Scripts\\python scripts\\90_run_pipeline.py --modality ct --nifti E:\\...\\P0001_ct_C2.nii.gz
  .venv\\Scripts\\python scripts\\90_run_pipeline.py --modality mri --image slice.png               # one 2D slice (exported format)

Output (Vietnamese): outputs/predictions/pipeline/<case>/result.png + result.json, and a summary in the terminal.
Research use only. MRI labels are shown as NOT reliable. A Test patient is processed for display only (warning printed,
true label hidden); Test results must never be used to tune anything.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import CONFIG_DIR, load_paths, load_yaml  # noqa: E402
from livercancer.data.sources import image_path  # noqa: E402
from livercancer.pipeline import LiverCancerPipeline  # noqa: E402
from livercancer.pipeline_viz import STATUS_VI, render_case, result_vi, verdict  # noqa: E402

CT_PHASES = ("C1", "C2", "C3", "P")
MRI_SEQS = ("C-pre", "C+A", "C+V", "C+Delay", "DWI", "T2WI", "In Phase", "Out Phase")


def split_of(paths, mod, pid):
    with open(paths["project"]["splits"] / f"{mod}_split.csv", encoding="utf-8") as f:
        return next(((r["split"], r["label"]) for r in csv.DictReader(f) if r["patient_id"] == pid), ("unknown", None))


def names_vi_for(mod: str) -> dict:
    c = load_yaml(CONFIG_DIR / "cls.yaml")[mod]
    return dict(zip(c["classes"], c.get("class_names_vi", c["classes"])))


def print_summary(res, nv: dict, true_label: str | None) -> None:
    print("\n================ KẾT QUẢ PIPELINE AI ================")
    print(f"Ca: {res.case_id} · Loại ảnh: {res.modality.upper()} · Thiết bị: {res.device} · "
          f"Trạng thái: {STATUS_VI.get(res.status, res.status)}")
    for v in res.volumes:
        print(f"  - {v['name']}: {v['n_slices']} lát, {v['n_regions']} vùng nghi ngờ")
    if res.status == "ok":
        print(f"Kết luận của hệ thống (gộp mọi vùng): {nv.get(res.label, res.label)}")
        print("Xác suất: " + ", ".join(f"{nv.get(k, k)} {100 * v:.1f}%" for k, v in sorted(res.scores.items(), key=lambda x: -x[1])))
        for r in sorted(res.regions, key=lambda r: -r["max_conf"])[:5]:
            print(f"    vùng {r['track_id']} ({r['volume']}, lát {r['k_first']}–{r['k_last']}, YOLO {r['max_conf']:.2f}): "
                  f"{nv.get(r['label'], r['label'])}")
    print(res.message)
    if true_label:
        print(f"Nhãn thực tế (dataset): {nv.get(true_label, true_label)}  →  đánh giá: {verdict(res, true_label, list(nv))}")
    for w in res.warnings:
        print("! " + w)
    print("Thời gian (giây):", res.timings_s)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["ct", "mri"], required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--patient", help="patient id (all available phases / sequences, or --series)")
    src.add_argument("--nifti", nargs="+", type=Path, help="one or more .nii.gz volumes of the same patient")
    src.add_argument("--image", type=Path, help="one 2D grayscale slice (PNG, training display convention)")
    ap.add_argument("--series", nargs="*", default=None, help="CT phases (C1 C2 C3 P) or MRI sequences (JSON names)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cls-checkpoint", type=Path, default=None, help="other classifier checkpoint (default: configs/cls.yaml)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--evidence-gate", choices=["config", "on", "off"], default="config",
                    help="reject option (configs/cls.yaml inference.<mod>.evidence_gate); 'on' forces the proposed gate")
    args = ap.parse_args()

    paths = load_paths()
    pipe = LiverCancerPipeline(args.modality, args.device, args.cls_checkpoint, args.evidence_gate)
    true_label = None
    if args.image:
        import cv2
        img = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Không đọc được ảnh:", args.image)
            return 1
        case, res = args.image.stem, pipe.run_image(img, args.image.stem)
    else:
        if args.patient:
            pid = args.patient
            series = args.series or (CT_PHASES if args.modality == "ct" else MRI_SEQS)
            files, names = [], []
            for s in series:
                try:
                    p = image_path(paths, args.modality, pid, s)
                except KeyError:
                    continue
                if p.exists():
                    files.append(p)
                    names.append(s.replace(" ", ""))
            if not files:
                print(f"Không tìm thấy ảnh của {pid} ({args.modality.upper()})")
                return 1
        else:
            files, names = list(args.nifti), None
            pid = files[0].name.split("_")[0]
        sp, true_label = split_of(paths, args.modality, pid)
        if sp == "test":
            print(f"CẢNH BÁO: {pid} thuộc tập TEST → chỉ để minh họa; tuyệt đối không dùng để chọn mô hình / ngưỡng.")
            true_label = None   # do not display Test labels next to predictions during development
        case = pid
        res = pipe.run_nifti(files, pid, names)
    out = args.out or ROOT / "outputs" / "predictions" / "pipeline" / case
    out.mkdir(parents=True, exist_ok=True)
    nv = names_vi_for(args.modality)
    (out / "result.json").write_text(json.dumps(result_vi(res, true_label, nv), indent=1, ensure_ascii=False), encoding="utf-8")
    render_case(res, out / "result.png", f"Kết quả pipeline AI · ca {case} · {args.modality.upper()}", true_label, names_vi=nv)
    print_summary(res, nv, true_label)
    print("Đã lưu:", out / "result.json", "và", out / "result.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
