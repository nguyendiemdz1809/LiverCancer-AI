"""GĐ3: YOLO inference demo on a few Validation slices of CT and MRI (separate models). Never touches Test.

For each modality:
  1 load the detector from configs/yolo_inference.yaml (sha256-checked best.pt; GPU fp16 if CUDA works, else CPU)
  2 sample Val slices with a fixed seed, one slice per patient: --n-pos slices with an annotated lesion and
    --n-neg slices without annotation (a negative slice is NOT a healthy patient)
  3 rebuild every input from the source NIfTI with livercancer.data.slices (same preprocessing as training) and
    check it is pixel-identical to the exported training PNG
  4 predict, match to GT (IoU >= --iou, greedy by confidence) -> TP / FP / FN per slice, draw GT + predictions
GT boxes are used only to draw and to count outcomes, never as model input.
The counts describe this small illustrative sample only; they are NOT an evaluation of the Val set
(full-Val metrics: reports/<mod>_*_analysis.md).

Outputs: reports/figures/inference/<mod>/*.png (per slice + montage), outputs/predictions/inference_demo/
(<mod>_cases.csv, summary_<device>.json), reports/inference_demo.md (only for the run that draws figures).
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import CONFIG_DIR, PROJECT_ROOT, load_paths, load_yaml  # noqa: E402
from livercancer.data.slices import load_volume_uint8, slice_image  # noqa: E402
from livercancer.data.sources import image_path  # noqa: E402
from livercancer.detection.yolo_detector import LesionDetector  # noqa: E402
from livercancer.evaluation.matching import count_outcomes, match  # noqa: E402

COLORS = {"gt": "#00e676", "tp": "#ff3d00", "fp": "#ffea00"}


def read_val_rows(data: Path) -> list[dict]:
    with open(data / "slices.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == "val"]  # Test is not in the export at all


def sample_rows(rows: list[dict], n_pos: int, n_neg: int, seed: int) -> list[dict]:
    """One random slice per patient; positives first, negatives from patients not already used when possible."""
    rng = random.Random(seed)
    picked, used = [], set()
    for kind, n in (("pos", n_pos), ("neg", n_neg)):
        by_pat = defaultdict(list)
        for r in rows:
            if r["kind"] == kind:
                by_pat[r["patient_id"]].append(r)
        pats = sorted(by_pat)
        rng.shuffle(pats)
        pats = [p for p in pats if p not in used] + [p for p in pats if p in used]
        for p in pats[:n]:
            picked.append(rng.choice(sorted(by_pat[p], key=lambda r: r["file"])))
            used.add(p)
    return picked


def gt_boxes(data: Path, row: dict) -> np.ndarray:
    w, h = int(row["width"]), int(row["height"])
    out = []
    for line in (data / "labels" / "val" / f"{row['file']}.txt").read_text().splitlines():
        if line.strip():
            _, cx, cy, bw, bh = map(float, line.split())
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out, float).reshape(-1, 4)


def draw_case(ax_raw, ax_det, img, gt, dets, tp_mask, title):
    from matplotlib.patches import Rectangle

    for ax in (ax_raw, ax_det):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]), ax.set_yticks([])
    ax_raw.set_title("Ảnh gốc (lát Val, uint8 như khi train)", fontsize=9)
    for b in gt:
        ax_det.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=COLORS["gt"], lw=1.6))
    for d, is_tp in zip(dets, tp_mask):
        c = COLORS["tp"] if is_tp else COLORS["fp"]
        ax_det.add_patch(Rectangle((d.x0, d.y0), d.x1 - d.x0, d.y1 - d.y0, fill=False, ec=c, lw=1.6, ls="--"))
        ax_det.text(d.x0, max(d.y0 - 3, 8), f"{d.label} {d.conf:.2f}", color="black", fontsize=7,
                    bbox={"facecolor": c, "edgecolor": "none", "pad": 1, "alpha": 0.9})
    ax_det.set_title(title, fontsize=9)


def run_modality(mod: str, args, paths: dict, out_dir: Path, fig_root: Path) -> dict:
    import torch

    cfg = load_yaml(args.config)[mod]
    data = PROJECT_ROOT / cfg["dataset"]
    det = LesionDetector.from_config(mod, device=args.device, conf=args.conf, config=args.config)
    rows = sample_rows(read_val_rows(data), args.n_pos, args.n_neg, args.seed)

    imgs, gts, png_same, cache = [], [], [], {}
    for r in rows:  # stream volumes one at a time; a volume is reused if two sampled slices share it
        key = (r["patient_id"], r["series"])
        if key not in cache:
            cache.clear()
            cache[key] = load_volume_uint8(image_path(paths, mod, *key))
        img = slice_image(cache[key], int(r["slice_k"]))
        png = cv2.imread(str(data / "images" / "val" / f"{r['file']}.png"), cv2.IMREAD_GRAYSCALE)
        png_same.append(bool(png is not None and png.shape == img.shape and np.array_equal(png, img)))
        imgs.append(img)
        gts.append(gt_boxes(data, r))
    cache.clear()

    det.predict(imgs)  # warm-up with the same batch shapes (CUDA kernels, fp16 cast, fuse) excluded from timing
    if det.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    dets, ms = det.timed_predict(imgs)
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if det.device.startswith("cuda") else None

    # the same images given as the exported PNG files must give the same boxes (input pipeline == training)
    via_png = det.model.predict([str(data / "images" / "val" / f"{r['file']}.png") for r in rows], imgsz=det.imgsz,
                                conf=det.conf, iou=det.iou, max_det=det.max_det, device=det.device, quantize=det.quantize,
                                verbose=False)
    png_pred_diff = 0.0
    for d, p in zip(dets, via_png):
        a = np.array([[x.x0, x.y0, x.x1, x.y1] for x in d]).reshape(-1, 4)
        b = p.boxes.xyxy.cpu().numpy().reshape(-1, 4)
        png_pred_diff = max(png_pred_diff, float("inf") if a.shape != b.shape else float(np.abs(np.sort(a, 0) - np.sort(b, 0)).max(initial=0)))

    cases, fig_paths = [], []
    if args.figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mdir = fig_root / mod
        mdir.mkdir(parents=True, exist_ok=True)
        for old in mdir.glob("*.png"):
            old.unlink()
    for i, (r, img, gt, d) in enumerate(zip(rows, imgs, gts, dets)):
        pb = np.array([[x.x0, x.y0, x.x1, x.y1] for x in d]).reshape(-1, 4)
        pc = np.array([x.conf for x in d])
        oc = count_outcomes(gt, pb, pc, args.iou)
        tp_mask, _ = match(gt, pb, pc, args.iou)
        rec = {"modality": mod.upper(), "file": r["file"], "patient_id": r["patient_id"], "patient_label": r["patient_label"],
               "series": r["series"], "slice_k": r["slice_k"], "kind": r["kind"], "n_gt": len(gt), "n_pred": len(d),
               **{k: oc[k] for k in ("TP", "FP", "FN")}, "gt_best_iou": ";".join(map(str, oc["gt_best_iou"])),
               "pred_conf": ";".join(f"{x.conf:.3f}" for x in d), "nifti_equals_training_png": png_same[i]}
        cases.append(rec)
        if args.figures:
            fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
            title = f"YOLO {mod.upper()} conf≥{det.conf:g}: TP {oc['TP']} · FP {oc['FP']} · FN {oc['FN']}"
            draw_case(axes[0], axes[1], img, gt, d, tp_mask, title)
            fig.suptitle(f"{r['patient_id']} · {r['series']} · lát k={int(r['slice_k'])} · "
                         f"{'có tổn thương (GT)' if r['kind'] == 'pos' else 'không có annotation'} · "
                         f"chẩn đoán BN: {r['patient_label']}\nXanh lá = GT · Đỏ = dự đoán khớp GT (IoU≥{args.iou:g}) · "
                         f"Vàng = dự đoán không khớp", fontsize=8.5)
            fig.tight_layout()
            p = mdir / f"{i + 1:02d}_{r['kind']}_{r['file']}.png"
            fig.savefig(p, dpi=110)
            plt.close(fig)
            fig_paths.append(p)
    if args.figures:  # montage of the detection panels
        n, cols = len(rows), 4
        fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(3.2 * cols, 3.4 * ((n + cols - 1) // cols)))
        for ax in np.ravel(axes):
            ax.axis("off")
        for ax, r, img, gt, d, c in zip(np.ravel(axes), rows, imgs, gts, dets, cases):
            pb = np.array([[x.x0, x.y0, x.x1, x.y1] for x in d]).reshape(-1, 4)
            tp_mask, _ = match(gt, pb, np.array([x.conf for x in d]), args.iou)
            ax.axis("on")
            draw_case(ax, ax, img, gt, d, tp_mask, f"{r['file']}\nTP {c['TP']} · FP {c['FP']} · FN {c['FN']}")
            ax.title.set_fontsize(7)
        fig.suptitle(f"YOLO {mod.upper()} trên {n} lát Val (mẫu minh họa, seed {args.seed}; conf≥{det.conf:g}). "
                     "Xanh lá = GT, đỏ = khớp GT, vàng = không khớp", fontsize=9)
        fig.tight_layout()
        p = fig_root / f"{mod}_montage.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        fig_paths.append(p)

    with open(out_dir / f"{mod}_cases_{det.device.split(':')[0]}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cases[0]))
        w.writeheader()
        w.writerows(cases)
    tot = {k: sum(c[k] for c in cases) for k in ("TP", "FP", "FN", "n_gt", "n_pred")}
    pos = [c for c in cases if c["kind"] == "pos"]
    neg = [c for c in cases if c["kind"] == "neg"]
    return {"detector": det.info(), "dataset": cfg["dataset"], "n_slices": len(cases), "n_pos_slices": len(pos),
            "n_neg_slices": len(neg), "totals": tot,
            "pos_slices_with_TP": sum(c["TP"] > 0 for c in pos), "neg_slices_with_FP": sum(c["FP"] > 0 for c in neg),
            "pos_slices_fully_missed": sum(c["TP"] == 0 for c in pos),
            "nifti_equals_training_png": f"{sum(png_same)}/{len(png_same)}",
            "max_box_diff_px_ndarray_vs_png_input": round(png_pred_diff, 4),
            "ms_per_image": round(ms, 1), "batch_size": 8, "cuda_peak_allocated_MB": None if peak_mb is None else round(peak_mb, 1),
            "cases": cases, "figures": [str(p.relative_to(PROJECT_ROOT)) for p in fig_paths]}


def write_report(path: Path, summaries: dict, args, extra: dict, fig_name: str = "inference") -> None:
    L = ["# Demo suy luận YOLO CT/MRI trên ảnh Validation (mẫu minh họa)", "",
         f"*Sinh bởi `scripts/60_yolo_inference_demo.py` lúc {datetime.now():%Y-%m-%d %H:%M}. Số liệu: "
         f"`outputs/predictions/inference_demo{path.stem[len('inference_demo'):]}/`. Hình: `reports/figures/{fig_name}/`. **Tập Test không được dùng.***", "",
         "> **Đây không phải kết quả đánh giá.** Mỗi modality chỉ có "
         f"{args.n_pos} lát có tổn thương + {args.n_neg} lát không có annotation, chọn ngẫu nhiên (seed {args.seed}, mỗi lát một BN). "
         "Số TP/FP/FN dưới đây chỉ mô tả bộ ảnh minh họa này. Metrics trên toàn bộ Val nằm trong `reports/<mod>_*_analysis.md`. "
         + (f"Ngưỡng conf = {args.conf:g}, đặt tay bằng `--conf` (khác ngưỡng vận hành trong configs/yolo_inference.yaml)."
            if args.conf is not None else
            "Ngưỡng conf = ngưỡng vận hành đã chốt trong `configs/yolo_inference.yaml` ("
            + ", ".join(f"{m.upper()} {s['detector']['conf']:g}" for m, s in summaries.items())
            + "): F2 lớn nhất trên Val của **từng** modality (`scripts/53_threshold_sweep.py`), không chỉnh theo Test."), "",
         "## 1. Mô hình và thiết bị", "",
         "| | CT | MRI |", "|---|---|---|"]
    S = summaries
    get = lambda m, k: S[m]["detector"][k] if m in S else "—"  # noqa: E731
    for k, lab in (("weights", "Checkpoint"), ("sha256", "SHA256"), ("device", "Thiết bị"), ("half", "fp16"),
                   ("imgsz", "imgsz"), ("conf", "conf"), ("iou_nms", "NMS IoU")):
        L.append(f"| {lab} | `{get('ct', k)}` | `{get('mri', k)}` |")
    for k, lab in (("ms_per_image", "Thời gian suy luận (ms/ảnh, batch 8, sau warm-up)"),
                   ("cuda_peak_allocated_MB", "VRAM PyTorch cấp phát lớn nhất (MB)"),
                   ("nifti_equals_training_png", "Ảnh dựng lại từ NIfTI trùng từng pixel với PNG khi train"),
                   ("max_box_diff_px_ndarray_vs_png_input", "Sai khác box khi đưa ndarray so với file PNG (px)")):
        L.append(f"| {lab} | {S.get('ct', {}).get(k, '—')} | {S.get('mri', {}).get(k, '—')} |")
    if extra:
        L += ["", "**Chạy lại trên CPU** (cùng mẫu ảnh, `--device cpu`, fp32):", "",
              "| | CT | MRI |", "|---|---|---|",
              f"| ms/ảnh | {extra.get('ct', {}).get('ms_per_image', '—')} | {extra.get('mri', {}).get('ms_per_image', '—')} |",
              f"| TP / FP / FN (so với GPU) | {_tot(extra.get('ct'))} | {_tot(extra.get('mri'))} |"]
    L += ["", "## 2. Kết quả trên bộ ảnh minh họa", "",
          "| Modality | Lát (có tổn thương / không) | Box GT | Dự đoán | TP | FP | FN | Lát có tổn thương được phát hiện | Lát có tổn thương bị bỏ sót hoàn toàn | Lát không annotation có FP |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m, s in S.items():
        t = s["totals"]
        L.append(f"| {m.upper()} | {s['n_slices']} ({s['n_pos_slices']} / {s['n_neg_slices']}) | {t['n_gt']} | {t['n_pred']} | "
                 f"{t['TP']} | {t['FP']} | {t['FN']} | {s['pos_slices_with_TP']}/{s['n_pos_slices']} | "
                 f"{s['pos_slices_fully_missed']}/{s['n_pos_slices']} | {s['neg_slices_with_FP']}/{s['n_neg_slices']} |")
    L += ["", f"TP = dự đoán khớp một box GT với IoU ≥ {args.iou:g} (ghép tham lam theo confidence, một-một); FP = dự đoán không khớp; "
          "FN = box GT không được khớp. Lát \"không annotation\" nghĩa là lát không có tổn thương được gán nhãn, **không** có nghĩa BN khỏe mạnh.", ""]
    for m, s in S.items():
        L += [f"### {m.upper()}", "", f"![montage](figures/{fig_name}/{m}_montage.png)", "",
              "| # | Ảnh | BN (chẩn đoán) | Pha/chuỗi | Loại lát | GT | Dự đoán (conf) | TP | FP | FN | IoU tốt nhất của GT |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for i, c in enumerate(s["cases"], 1):
            fig = next((f for f in s["figures"] if f"{i:02d}_" in Path(f).name), "")
            link = f"[{c['file']}]({Path(fig).relative_to('reports').as_posix()})" if fig else c["file"]
            L.append(f"| {i} | {link} | {c['patient_id']} ({c['patient_label']}) | {c['series']} | "
                     f"{'có tổn thương' if c['kind'] == 'pos' else 'không annotation'} | {c['n_gt']} | {c['pred_conf'] or '—'} | "
                     f"{c['TP']} | {c['FP']} | {c['FN']} | {c['gt_best_iou'] or '—'} |")
        L.append("")
    L += ["## 3. Môi trường", "", f"- {extra.get('_env', '')}", "",
          "<!-- ANALYST-NOTES -->", "", "## 4. Nhận xét", "", "(người phân tích bổ sung sau khi xem hình)"]
    keep = ""
    if path.exists() and "<!-- ANALYST-NOTES -->" in path.read_text(encoding="utf-8"):
        keep = path.read_text(encoding="utf-8").split("<!-- ANALYST-NOTES -->", 1)[1]
    txt = "\n".join(L) + "\n"
    if keep.strip() and "(người phân tích bổ sung" not in keep:
        txt = txt.split("<!-- ANALYST-NOTES -->", 1)[0] + "<!-- ANALYST-NOTES -->" + keep
    path.write_text(txt, encoding="utf-8")


def _tot(s: dict | None) -> str:
    return "—" if not s else f"{s['totals']['TP']} / {s['totals']['FP']} / {s['totals']['FN']}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["ct", "mri", "both"], default="both")
    ap.add_argument("--device", default="auto", help="auto (CUDA if available, else CPU) | cpu | cuda")
    ap.add_argument("--conf", type=float, default=None, help="override conf of configs/yolo_inference.yaml")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU to count a prediction as TP")
    ap.add_argument("--n-pos", type=int, default=8)
    ap.add_argument("--n-neg", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=Path, default=CONFIG_DIR / "yolo_inference.yaml")
    ap.add_argument("--no-figures", dest="figures", action="store_false", help="only time/count (e.g. a CPU re-run)")
    ap.add_argument("--out-tag", default="", help="suffix for outputs, e.g. conf012 -> reports/inference_demo_conf012.md")
    args = ap.parse_args()

    import torch
    import ultralytics

    paths = load_paths()
    sfx = f"_{args.out_tag}" if args.out_tag else ""
    out_dir = paths["project"]["outputs"] / "predictions" / f"inference_demo{sfx}"
    fig_root = paths["project"]["reports"] / "figures" / f"inference{sfx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)
    mods = ["ct", "mri"] if args.modality == "both" else [args.modality]
    summaries = {}
    for m in mods:
        s = run_modality(m, args, paths, out_dir, fig_root)
        summaries[m] = s
        print(f"{m.upper()} [{s['detector']['device']}] TP {s['totals']['TP']} FP {s['totals']['FP']} FN {s['totals']['FN']} | "
              f"pos slices detected {s['pos_slices_with_TP']}/{s['n_pos_slices']} | neg slices with FP "
              f"{s['neg_slices_with_FP']}/{s['n_neg_slices']} | NIfTI==PNG {s['nifti_equals_training_png']} | "
              f"{s['ms_per_image']} ms/img | peak {s['cuda_peak_allocated_MB']} MB", flush=True)
    dev = next(iter(summaries.values()))["detector"]["device"].split(":")[0]
    env = (f"Python {platform.python_version()}, torch {torch.__version__}, Ultralytics {ultralytics.__version__}, "
           f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'không có CUDA'}")
    meta = {"time": datetime.now().isoformat(timespec="seconds"), "args": {k: str(v) for k, v in vars(args).items()},
            "env": env, "split": "val", "note": "illustrative sample only, not an evaluation"}
    (out_dir / f"summary_{dev}.json").write_text(json.dumps({**meta, **summaries}, indent=1, ensure_ascii=False, default=str),
                                                 encoding="utf-8")
    if args.figures:
        extra = {"_env": env}
        cpu = out_dir / "summary_cpu.json"
        if dev != "cpu" and cpu.exists():
            extra.update({m: v for m, v in json.loads(cpu.read_text(encoding="utf-8")).items() if m in ("ct", "mri")})
        write_report(paths["project"]["reports"] / f"inference_demo{sfx}.md", summaries, args, extra, fig_root.name)
        print(f"written reports/inference_demo{sfx}.md and", fig_root)
    print("written", out_dir / f"summary_{dev}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
