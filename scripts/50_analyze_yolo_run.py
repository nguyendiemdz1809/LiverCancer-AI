"""GĐ3: analyse a finished YOLO run (trained on Colab) against the local pilot Val set. Never touches Test.

Inputs (downloaded by the user from Drive into --run, e.g. outputs/runs/yolo_ct/pilot_yolo11n_e5/):
  results.csv, args.yaml, weights/best.pt  (required);  metrics_val.json (optional, written by the Colab notebook)
Steps:
  A  training curves from results.csv (losses, P, R, mAP, lr), best epoch by Ultralytics fitness (= mAP50-95)
  B  re-validate best.pt locally on the pilot Val split twice: conf=0.001 (Ultralytics default -> mAP, confusion
     matrix) and conf=0.25 (default predict threshold) -> shows why the confusion matrix 'background' row is huge
  C  predict once at conf=0.001, then match to GT (IoU >= 0.5, greedy by confidence) at several deployment
     thresholds: box-level TP/FP/FN, slice-level FP rate on negative slices, patient-level detection,
     FN by lesion size / phase / patient class
  D  GT-vs-prediction figures: highest-confidence FPs, largest FNs, random TPs
Outputs: <run>/analysis/*.json|csv, reports/figures/<tag>/*.png, reports/<tag>_analysis.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import load_paths  # noqa: E402

THRESHOLDS = (0.001, 0.05, 0.1, 0.25, 0.4, 0.5)
SIZE_BINS = (("nhỏ <32²px", 0, 32 ** 2), ("vừa 32²–96²px", 32 ** 2, 96 ** 2), ("lớn >96²px", 96 ** 2, math.inf))


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes a (N,4) and b (M,4)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(gt: np.ndarray, pred: np.ndarray, conf: np.ndarray, iou_thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one matching by descending confidence. Returns (pred_is_tp, gt_is_matched)."""
    order = np.argsort(-conf)
    tp, used = np.zeros(len(pred), bool), np.zeros(len(gt), bool)
    if len(gt) and len(pred):
        ious = iou_matrix(pred, gt)
        for p in order:
            cand = np.where(~used & (ious[p] >= iou_thr))[0]
            if len(cand):
                g = cand[np.argmax(ious[p, cand])]
                tp[p], used[g] = True, True
    return tp, used


def load_gt(data: Path, row: dict) -> np.ndarray:
    w, h = int(row["width"]), int(row["height"])
    out = []
    for line in (data / "labels" / row["split"] / f"{row['file']}.txt").read_text().splitlines():
        if line.strip():
            _, cx, cy, bw, bh = map(float, line.split())
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out, float).reshape(-1, 4)


# ----------------------------------------------------------------------------- A. curves
def analyse_curves(rows: list[dict], fig_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    ep = col("epoch")
    keys = [k for k in rows[0] if k != "epoch"]
    best = int(np.argmax(col("metrics/mAP50-95(B)")))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k in ("box_loss", "cls_loss", "dfl_loss"):
        ax[0].plot(ep, col(f"train/{k}"), "-o", label=f"train/{k}")
        ax[0].plot(ep, col(f"val/{k}"), "--s", label=f"val/{k}")
    ax[0].set_title("Loss"); ax[0].legend(fontsize=7)
    for k in ("precision(B)", "recall(B)", "mAP50(B)", "mAP50-95(B)"):
        ax[1].plot(ep, col(f"metrics/{k}"), "-o", label=k)
    ax[1].axvline(ep[best], color="gray", ls=":", label=f"best (mAP50-95) ep {int(ep[best])}")
    ax[1].set_title("Val metrics"); ax[1].legend(fontsize=7)
    for k in [k for k in keys if k.startswith("lr/")]:
        ax[2].plot(ep, col(k), "-o", label=k)
    ax[2].set_title("Learning rate"); ax[2].legend(fontsize=7)
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "curves.png", dpi=110); plt.close(fig)
    trend = lambda k: float(col(k)[-1] - col(k)[0])  # noqa: E731
    return {"epochs": len(rows), "best_epoch": int(ep[best]),
            "best_row": {k: float(v) for k, v in rows[best].items()},
            "last_row": {k: float(v) for k, v in rows[-1].items()},
            "delta_first_last": {k: round(trend(k), 4) for k in keys if k.startswith(("train/", "val/", "metrics/"))},
            "table": rows}


# ----------------------------------------------------------------------------- B. ultralytics val
def ultra_val(model, yaml_path: Path, conf: float, out: Path, name: str, imgsz: int, batch: int, device) -> dict:
    m = model.val(data=str(yaml_path), split="val", conf=conf, imgsz=imgsz, batch=batch, device=device, plots=True,
                  project=str(out), name=name, exist_ok=True, verbose=False,
                  workers=0 if os.name == "nt" else 2)
    cm = m.confusion_matrix.matrix  # [pred, true]; index 1 = background
    return {"conf": conf, "precision": float(m.box.mp), "recall": float(m.box.mr), "mAP50": float(m.box.map50),
            "mAP50-95": float(m.box.map), "confusion_matrix_[pred,true]": cm.astype(int).tolist(),
            "cm_TP": int(cm[0, 0]), "cm_FP_background_row": int(cm[0, 1]), "cm_FN_background_col": int(cm[1, 0]),
            "plots_dir": str(out / name)}


# ----------------------------------------------------------------------------- C. threshold analysis
def threshold_analysis(slices: list[dict], data: Path, preds: dict, iou_thr: float, modality: str) -> dict:
    lesion_pat = {r["patient_id"] for r in slices if r["kind"] == "pos"}
    neg_only_pat = {r["patient_id"] for r in slices} - lesion_pat
    gts = {r["file"]: load_gt(data, r) for r in slices}
    out = {}
    for thr in THRESHOLDS:
        tp = fp_pos = fp_neg = fn = 0
        neg_with_fp = pos_with_tp = 0
        pat_tp, pat_fp = set(), set()
        fn_size, gt_size = Counter(), Counter()
        by_group = defaultdict(lambda: [0, 0])  # (matched gt, total gt) per series / patient class
        for r in slices:
            gt = gts[r["file"]]
            p = preds.get(r["file"], {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0)})
            keep = p["conf"] >= thr
            pb, pc = p["xyxy"][keep], p["conf"][keep]
            is_tp, gt_hit = match(gt, pb, pc, iou_thr)
            tp += int(is_tp.sum())
            n_fp = int((~is_tp).sum())
            if r["kind"] == "pos":
                fp_pos += n_fp
                fn += int((~gt_hit).sum())
                pos_with_tp += bool(is_tp.any())
                if is_tp.any():
                    pat_tp.add(r["patient_id"])
                for g, hit in zip(gt, gt_hit):
                    area = (g[2] - g[0]) * (g[3] - g[1])
                    b = next(n for n, lo, hi in SIZE_BINS if lo <= area < hi)
                    gt_size[b] += 1
                    fn_size[b] += (not hit)
                for key in (f"series={r['series']}", f"class={r['patient_label']}"):
                    by_group[key][0] += int(gt_hit.sum())
                    by_group[key][1] += len(gt)
            else:
                fp_neg += n_fp
                neg_with_fp += n_fp > 0
            if n_fp:
                pat_fp.add(r["patient_id"])
        n_pos = sum(r["kind"] == "pos" for r in slices)
        n_neg = len(slices) - n_pos
        P = tp / (tp + fp_pos + fp_neg) if tp + fp_pos + fp_neg else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        out[str(thr)] = {
            "TP": tp, "FP_on_positive_slices": fp_pos, "FP_on_negative_slices": fp_neg, "FN": fn,
            "precision": round(P, 4), "recall": round(R, 4), "F1": round(2 * P * R / (P + R), 4) if P + R else 0.0,
            "FP_per_slice": round((fp_pos + fp_neg) / len(slices), 3),
            "negative_slices_with_FP": f"{neg_with_fp}/{n_neg} ({100 * neg_with_fp / max(n_neg, 1):.1f}%)",
            "positive_slices_with_TP": f"{pos_with_tp}/{n_pos} ({100 * pos_with_tp / max(n_pos, 1):.1f}%)",
            "lesion_patients_detected": f"{len(pat_tp)}/{len(lesion_pat)}",
            "negative_only_patients_with_FP": f"{len(pat_fp & neg_only_pat)}/{len(neg_only_pat)}"
                                              + (" (CN)" if modality == "ct" else ""),
            "FN_rate_by_size": {b: f"{fn_size[b]}/{gt_size[b]}" for b, _, _ in SIZE_BINS},
            "recall_by_group": {k: round(v[0] / v[1], 3) for k, v in sorted(by_group.items()) if v[1]},
        }
    return out


# ----------------------------------------------------------------------------- D. figures
def figures(slices: list[dict], data: Path, preds: dict, thr: float, iou_thr: float, fig_dir: Path, n: int, seed: int):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fps, fns, tps = [], [], []
    for r in slices:
        gt = load_gt(data, r)
        p = preds.get(r["file"], {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0)})
        keep = p["conf"] >= thr
        is_tp, hit = match(gt, p["xyxy"][keep], p["conf"][keep], iou_thr)
        if (~is_tp).any():
            fps.append((float(p["conf"][keep][~is_tp].max()), r))
        if len(gt) and (~hit).any():
            g = gt[~hit]
            fns.append((float(((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])).max()), r))
        if is_tp.any():
            tps.append((0.0, r))
    def top_per_patient(items: list) -> list:  # one case per patient, most severe first
        seen, out = set(), []
        for score, r in sorted(items, key=lambda x: -x[0]):
            if r["patient_id"] not in seen:
                seen.add(r["patient_id"])
                out.append((score, r))
        return out[:n]

    rng = random.Random(seed)
    groups = {"FP": top_per_patient(fps), "FN": top_per_patient(fns), "TP": rng.sample(tps, min(n, len(tps)))}
    for old in fig_dir.glob("case_*.png"):
        old.unlink()
    for kind, items in groups.items():
        for i, (_, r) in enumerate(items, 1):
            im = cv2.imread(str(data / "images" / r["split"] / f"{r['file']}.png"), cv2.IMREAD_UNCHANGED)
            p = preds.get(r["file"], {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0)})
            keep = p["conf"] >= thr
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            for g in load_gt(data, r):
                ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec="lime", lw=1.3))
            for b, c in zip(p["xyxy"][keep], p["conf"][keep]):
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="red", lw=1.0, ls="--"))
                ax.text(b[0], b[1] - 2, f"{c:.2f}", color="red", fontsize=6)
            ax.set_title(f"{kind} @conf≥{thr}: {r['patient_id']} {r['series']} k={r['slice_k']} ({r['patient_label']}, "
                         f"{r['kind']})\ngreen = GT, red dashed = prediction", fontsize=7)
            ax.axis("off"); fig.tight_layout()
            fig.savefig(fig_dir / f"case_{kind}_{i:02d}_{r['file']}.png", dpi=90); plt.close(fig)
    return {k: [r["file"] for _, r in v] for k, v in groups.items()}


# ----------------------------------------------------------------------------- report
def fmt_table(th: dict) -> list[str]:
    L = ["| conf ≥ | TP | FP (lát dương) | FP (lát âm) | FN | Precision | Recall | F1 | FP/lát | Lát âm có FP | "
         "Lát dương có TP | BN tổn thương được phát hiện | BN chỉ có lát âm mà có FP |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, v in th.items():
        L.append(f"| {t} | {v['TP']} | {v['FP_on_positive_slices']} | {v['FP_on_negative_slices']} | {v['FN']} | "
                 f"{v['precision']} | {v['recall']} | {v['F1']} | {v['FP_per_slice']} | {v['negative_slices_with_FP']} | "
                 f"{v['positive_slices_with_TP']} | {v['lesion_patients_detected']} | {v['negative_only_patients_with_FP']} |")
    return L


def write_report(path: Path, tag: str, run: Path, args_y: dict, curves: dict, colab: dict | None, val: list[dict],
                 th: dict, figs: dict, fig_rel: str, iou_thr: float, op_thr: float) -> None:
    b, l = curves["best_row"], curves["last_row"]
    g = lambda r, k: r.get(k, float("nan"))  # noqa: E731
    L = [f"# Phân tích YOLO run `{run.name}` ({tag.split('_')[0].upper()}), chỉ trên Validation\n",
         f"*Sinh bởi `scripts/50_analyze_yolo_run.py` lúc {datetime.now():%Y-%m-%d %H:%M}. Mọi số liệu đọc từ "
         f"`{run.as_posix()}` (tải về từ Colab) và từ lần chạy lại `best.pt` trên tập Val pilot ở local. "
         "**Tập Test không được dùng.***\n",
         "## 1. Cấu hình huấn luyện (args.yaml)\n",
         "| Tham số | Giá trị |", "|---|---|"]
    for k in ("model", "epochs", "imgsz", "batch", "optimizer", "lr0", "lrf", "cos_lr", "warmup_epochs", "close_mosaic",
              "mosaic", "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "scale", "translate", "degrees", "amp", "patience", "seed"):
        L.append(f"| {k} | {args_y.get(k)} |")
    L += ["", "## 2. Đường cong huấn luyện (results.csv)\n", f"![curves]({fig_rel}/curves.png)\n",
          "| Epoch | train box/cls/dfl | val box/cls/dfl | P | R | mAP50 | mAP50-95 |", "|---|---|---|---|---|---|---|"]
    for r in curves["table"]:
        f = lambda k: float(r[k])  # noqa: E731
        L.append(f"| {int(f('epoch'))} | {f('train/box_loss'):.3f} / {f('train/cls_loss'):.3f} / {f('train/dfl_loss'):.3f} | "
                 f"{f('val/box_loss'):.3f} / {f('val/cls_loss'):.3f} / {f('val/dfl_loss'):.3f} | {f('metrics/precision(B)'):.3f} | "
                 f"{f('metrics/recall(B)'):.3f} | {f('metrics/mAP50(B)'):.3f} | {f('metrics/mAP50-95(B)'):.3f} |")
    L += ["", f"- Epoch tốt nhất theo fitness của Ultralytics (= mAP50-95): **{curves['best_epoch']}** "
          f"(mAP50 {g(b, 'metrics/mAP50(B)'):.3f}, mAP50-95 {g(b, 'metrics/mAP50-95(B)'):.3f}); epoch cuối: "
          f"mAP50 {g(l, 'metrics/mAP50(B)'):.3f}, mAP50-95 {g(l, 'metrics/mAP50-95(B)'):.3f}.",
          f"- Thay đổi từ epoch đầu đến epoch cuối: {curves['delta_first_last']}\n"]
    L += ["## 3. Metrics của `best.pt` trên Val\n", "| Nguồn | conf | Precision | Recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|---|"]
    if colab:
        m = colab.get("metrics_val", {})
        L.append(f"| Colab `metrics_val.json` | 0.001 | {m.get('precision', float('nan')):.3f} | {m.get('recall', float('nan')):.3f} | "
                 f"{m.get('mAP50', float('nan')):.3f} | {m.get('mAP50-95', float('nan')):.3f} |")
    for v in val:
        L.append(f"| Chạy lại ở local | {v['conf']} | {v['precision']:.3f} | {v['recall']:.3f} | {v['mAP50']:.3f} | {v['mAP50-95']:.3f} |")
    L += ["", "Precision/Recall của Ultralytics được tính tại ngưỡng conf cho **F1 cao nhất** trên đường cong, "
          "không phải tại một ngưỡng cố định; mAP luôn tính với conf = 0.001.\n",
          "## 4. Vì sao Confusion Matrix có rất nhiều 'background' dự đoán sai\n",
          "| conf khi tính ma trận | TP | FP (hàng background) | FN (cột background) |", "|---|---|---|---|"]
    for v in val:
        L.append(f"| {v['conf']} | {v['cm_TP']} | {v['cm_FP_background_row']} | {v['cm_FN_background_col']} |")
    L += ["", "- Ultralytics 8.4.182 tính confusion matrix khi validate với `conf = args.conf`, mặc định **0.001** "
          "(`engine/validator.py`: `self.args.conf = 0.001`; `detect/val.py`: `process_batch(..., conf=self.args.conf)`), "
          "ghép box với GT ở IoU > 0.45. Vì vậy **mọi box có conf > 0.001** (kể cả box gần như chắc chắn là nhiễu) "
          "đều được tính; mọi box không khớp GT rơi vào hàng *background* (FP). Trên tập Val có rất nhiều lát âm tính, "
          "nên con số này rất lớn.",
          "- Ma trận được chuẩn hóa theo **cột** (cột GT), nên ô background→lesion trông như \"100%\" khi cột background chỉ "
          "chứa FP. Đó là cách hiển thị, không phải tỉ lệ lỗi thực.",
          "- Khi **inference** thực tế (`predict`), ngưỡng mặc định là **0.25** và ngưỡng vận hành sẽ được chọn trên Val. "
          "Bảng ở mục 5 cho thấy FP/FN tại từng ngưỡng.\n",
          f"## 5. FP / FN theo ngưỡng confidence (ghép GT ở IoU ≥ {iou_thr})\n", *fmt_table(th), ""]
    op = th[str(op_thr)]
    L += [f"Tại conf ≥ {op_thr}: FN theo kích thước GT {op['FN_rate_by_size']}; recall theo nhóm {op['recall_by_group']}\n",
          "## 6. Đối chiếu box dự đoán với ground truth\n",
          f"Xanh lá = GT, đỏ nét đứt = dự đoán (conf ≥ {op_thr}). Có {len(figs['FP'])} ca FP có confidence cao nhất, "
          f"{len(figs['FN'])} ca FN có tổn thương lớn nhất, và {len(figs['TP'])} ca TP ngẫu nhiên, lưu ở `{fig_rel}/case_*.png`.\n"]
    for kind in ("FP", "FN", "TP"):
        L.append(f"- **{kind}:** " + ", ".join(f"[{f}]({fig_rel}/case_{kind}_{i:02d}_{f}.png)" for i, f in enumerate(figs[kind], 1)))
    L += ["", "<!-- ANALYST-NOTES -->", "", "## 7. Nhận định và đề xuất\n",
          "(chưa có — người phân tích bổ sung sau khi xem số liệu)"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", choices=["ct", "mri"], required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="pilot dataset dir (default data/processed/pilot/yolo_<mod>)")
    ap.add_argument("--tag", default=None, help="report name (default <mod>_pilot)")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--op-thr", type=float, default=0.25, help="threshold used for the size/group breakdown and figures")
    ap.add_argument("--n-figures", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import yaml
    from ultralytics import YOLO

    paths = load_paths()
    run = args.run
    data = args.data or paths["project"]["processed"] / "pilot" / f"yolo_{args.modality}"
    tag = args.tag or f"{args.modality}_pilot"
    for req in ("results.csv", "args.yaml", "weights/best.pt"):
        if not (run / req).exists():
            print(f"missing {run / req} — download it from Drive first")
            return 1
    out = (run / "analysis").resolve()  # absolute: Ultralytics prefixes relative project dirs with runs/detect
    fig_dir = paths["project"]["reports"] / "figures" / tag
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    args_y = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
    imgsz = args.imgsz or int(args_y.get("imgsz", 512))
    colab = json.loads((run / "metrics_val.json").read_text(encoding="utf-8")) if (run / "metrics_val.json").exists() else None

    curves = analyse_curves(read_csv(run / "results.csv"), fig_dir)
    print(f"A curves: {curves['epochs']} epochs, best {curves['best_epoch']}", flush=True)

    local_yaml = out / "data_local.yaml"
    cfg = yaml.safe_load((data / "data.yaml").read_text(encoding="utf-8"))
    cfg["path"] = str(data.resolve())
    local_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    model = YOLO(str(run / "weights" / "best.pt"))
    device = int(args.device) if args.device.isdigit() else args.device
    val = [ultra_val(model, local_yaml, c, out, f"val_conf{c}", imgsz, args.batch, device) for c in (0.001, 0.25)]
    print("B val:", [(v["conf"], round(v["mAP50"], 3), v["cm_FP_background_row"]) for v in val], flush=True)

    with open(data / "slices.csv", encoding="utf-8") as f:
        slices = [r for r in csv.DictReader(f) if r["split"] == "val"]
    preds = {}
    files = [str(data / "images" / "val" / f"{r['file']}.png") for r in slices]
    for i in range(0, len(files), 256):
        for res in model.predict(files[i:i + 256], conf=0.001, iou=0.7, imgsz=imgsz, batch=args.batch, device=device,
                                 verbose=False, stream=True, max_det=300):
            preds[Path(res.path).stem] = {"xyxy": res.boxes.xyxy.cpu().numpy(), "conf": res.boxes.conf.cpu().numpy()}
    with open(out / "val_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "x0", "y0", "x1", "y1", "conf"])
        for k, p in preds.items():
            for b, c in zip(p["xyxy"], p["conf"]):
                w.writerow([k, *np.round(b, 2), round(float(c), 5)])
    th = threshold_analysis(slices, data, preds, args.iou, args.modality)
    print("C thresholds:", {t: (v["precision"], v["recall"]) for t, v in th.items()}, flush=True)
    figs = figures(slices, data, preds, args.op_thr, args.iou, fig_dir, args.n_figures, args.seed)

    summary = {"run": str(run), "data": str(data), "imgsz": imgsz, "iou_match": args.iou, "curves": {k: v for k, v in curves.items() if k != "table"},
               "colab_metrics_val": colab, "local_val": val, "thresholds": th, "figures": figs}
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    report = paths["project"]["reports"] / f"{tag}_analysis.md"
    rel = Path("figures") / tag
    keep_notes = ""
    if report.exists():  # preserve the analyst's discussion section on re-runs
        txt = report.read_text(encoding="utf-8")
        if "<!-- ANALYST-NOTES -->" in txt:
            keep_notes = txt.split("<!-- ANALYST-NOTES -->", 1)[1]
    write_report(report, tag, run, args_y, curves, colab, val, th, figs, rel.as_posix(), args.iou, args.op_thr)
    if keep_notes.strip():
        head = report.read_text(encoding="utf-8").split("<!-- ANALYST-NOTES -->", 1)[0]
        report.write_text(head + "<!-- ANALYST-NOTES -->" + keep_notes, encoding="utf-8")
    print("written", report, "and", out / "analysis_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
