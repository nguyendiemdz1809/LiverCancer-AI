"""YOLOv5n vs YOLO11n: control experiment of the detector (YOLO11n stays the model of the project; nothing is replaced).

Stages (all outputs in outputs/runs/comparison/, never in the YOLO11n run folders):
  smoke  --mod ct|mri   1 training epoch of yolov5nu.pt on 64 Train / 32 Val images of the SAME dataset with the SAME
                        hyper-parameters as YOLO11n (proves the notebook settings run; not a result)
  eval   --split val    both models, same local GPU, same code: Ultralytics val (conf 0.001, IoU-NMS 0.7, imgsz 512)
                        -> P, R, F1 (at the max-F1 point), mAP50, mAP50-95; plus our own matching at the LOCKED operating
                        threshold conf 0.12 (IoU >= 0.5): P, R, F1, recall of small boxes (area < 32 x 32 px)
  eval   --split test   same, ONCE (user decision 2026-11-09: deliberate exception to the "Test only once" rule, as an
                        independent check; YOLO11n is NOT replaced whatever the result). Refuses to run twice
                        (outputs/runs/comparison/test_done.json). Test = the GT export of scripts/96_final_test.py: lesion
                        slices only (no negative slice, no control patient) -> precision there ignores false alarms on
                        normal slices.
  bench                 speed on the local GPU (batch 1, imgsz 512, fp16 and fp32, 200 Val images, 20 warm-up),
                        parameters, GFLOPs, checkpoint size
  report                reports/yolov5n_vs_yolo11n_comparison.md + figures (only measured numbers)
YOLOv5n checkpoints: outputs/runs/comparison/yolov5n_<mod>/yolov5nu_e50_img512_comparison/ (Colab bundle, notebooks
04_train_yolov5n_<mod>_comparison.ipynb). YOLO11n: the locked weights of configs/yolo_inference.yaml (sha256 checked).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLO_OFFLINE", "1")

from livercancer.config import CONFIG_DIR, load_yaml  # noqa: E402

OUT = ROOT / "outputs" / "runs" / "comparison"
REPORT = ROOT / "reports" / "yolov5n_vs_yolo11n_comparison.md"
FIG = ROOT / "reports" / "figures" / "yolov5n_vs_yolo11n"
TEXT_DIR = ROOT / "reports" / "yolov5n_vs_yolo11n"  # hand-written intro / analysis merged by --stage report
V5_RUN = "yolov5nu_e50_img512_comparison"
PRETRAINED_V5 = OUT / "pretrained" / "yolov5nu.pt"
DATA = {  # the exact datasets of the locked YOLO11n runs (metrics_val.json: yolo_ct_full.zip / yolo_mri_pilot.zip)
    "ct": {"val": ROOT / "data/processed/full/yolo_ct", "test": ROOT / "data/processed/final_test/yolo_ct"},
    "mri": {"val": ROOT / "data/processed/pilot/yolo_mri", "test": ROOT / "data/processed/final_test/yolo_mri"},
}
TRAIN_KW = {"optimizer": "SGD", "lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005, "cos_lr": True,
            "warmup_epochs": 3.0, "close_mosaic": 10, "mosaic": 1.0, "mixup": 0.0, "fliplr": 0.5, "flipud": 0.0,
            "degrees": 0.0, "translate": 0.1, "scale": 0.5, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3}
IMGSZ, OP_CONF, SMALL = 512, 0.12, 32 * 32


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def models(mod: str) -> dict[str, dict]:
    cfg = load_yaml(CONFIG_DIR / "yolo_inference.yaml")[mod]
    y11 = ROOT / cfg["weights"]
    if sha256(y11) != cfg["sha256"]:
        raise SystemExit(f"YOLO11n {mod}: sha256 mismatch (locked weights changed?)")
    v5 = OUT / f"yolov5n_{mod}" / V5_RUN / "weights" / "best.pt"
    out = {"yolo11n": {"weights": y11, "sha256": cfg["sha256"], "run": ROOT / cfg["run"]}}
    if v5.exists():
        out["yolov5n"] = {"weights": v5, "sha256": sha256(v5), "run": v5.parents[1]}
    return out


def data_yaml(mod: str, split: str) -> Path:
    """Dataset yaml written in outputs/ (the dataset folders are only read)."""
    root = DATA[mod][split]
    d = OUT / "data"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{mod}_{split}.yaml"
    sub = "val" if split == "val" else "test"
    p.write_text(f"# written by scripts/78_yolov5n_comparison.py\npath: {root.as_posix()}\ntrain: images/{sub}\nval: images/{sub}\n"
                 f"names:\n  0: lesion\n", encoding="utf-8")
    return p


# ------------------------------------------------------------------------------------------------------------- smoke
def smoke(args) -> int:
    from ultralytics import YOLO
    import torch
    assert torch.cuda.is_available(), "no CUDA GPU: the smoke test is a GPU check"
    mod = args.mod
    root = DATA[mod]["val"]
    rng = random.Random(42)
    rows = list(csv.DictReader(open(root / "slices.csv", encoding="utf-8")))
    d = OUT / "smoke" / mod
    d.mkdir(parents=True, exist_ok=True)
    import shutil
    for split, n in (("train", 64), ("val", 32)):   # copies: Ultralytics writes its label cache next to the labels
        pick = rng.sample([r for r in rows if r["split"] == split], n)
        for sub, ext in (("images", ".png"), ("labels", ".txt")):
            (d / sub / split).mkdir(parents=True, exist_ok=True)
            for r in pick:
                shutil.copy2(root / sub / split / f"{r['file']}{ext}", d / sub / split / f"{r['file']}{ext}")
    y = d / "data.yaml"
    y.write_text(f"path: {d.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: lesion\n", encoding="utf-8")
    t0 = time.time()
    m = YOLO(str(PRETRAINED_V5))
    m.train(data=str(y), epochs=1, imgsz=IMGSZ, batch=8, device=0, amp=True, seed=42, deterministic=True, workers=0,
            project=str(d), name="run", exist_ok=True, plots=False, val=True, patience=20, **TRAIN_KW)
    res = {"model": "yolov5nu.pt", "sha256": sha256(PRETRAINED_V5), "mod": mod, "train_images": 64, "val_images": 32,
           "epochs": 1, "batch": 8, "train_kw": TRAIN_KW, "seconds": round(time.time() - t0, 1),
           "weights_written": sorted(p.name for p in (d / "run" / "weights").glob("*.pt")),
           "args_optimizer": load_yaml(d / "run" / "args.yaml")["optimizer"], "note": "pipeline check only, not a result"}
    (d / "smoke.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


# -------------------------------------------------------------------------------------------------------------- eval
def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def operating_point(model, img_dir: Path, lab_dir: Path) -> dict:
    """Greedy matching (highest confidence first, IoU >= 0.5) at the locked threshold conf 0.12."""
    files = sorted(img_dir.glob("*.png"))
    tp = fp = fn = 0
    small_tp = small_n = 0
    for i in range(0, len(files), 32):
        batch = files[i:i + 32]
        for f, r in zip(batch, model.predict([str(x) for x in batch], imgsz=IMGSZ, conf=OP_CONF, iou=0.7, max_det=300,
                                             device=0, verbose=False)):
            h, w = r.orig_shape
            pred = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
            conf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
            gt = []
            lp = lab_dir / f"{f.stem}.txt"
            for line in (lp.read_text().splitlines() if lp.exists() else []):
                if line.strip():
                    _, cx, cy, bw, bh = map(float, line.split())
                    gt.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            gt = np.array(gt).reshape(-1, 4)
            small = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) < SMALL
            matched = np.zeros(len(gt), bool)
            if len(pred) and len(gt):
                m = iou(pred[np.argsort(-conf)], gt)
                for row in m:
                    row = np.where(matched, -1, row)
                    j = int(row.argmax())
                    if row[j] >= 0.5:
                        matched[j] = True
            tp += int(matched.sum()); fn += int((~matched).sum()); fp += len(pred) - int(matched.sum())
            small_n += int(small.sum()); small_tp += int((matched & small).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"conf": OP_CONF, "iou_match": 0.5, "images": len(files), "gt_boxes": tp + fn, "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
            "small_boxes": small_n, "small_recall": small_tp / small_n if small_n else None}


def evaluate(args) -> int:
    from ultralytics import YOLO
    import torch
    assert torch.cuda.is_available(), "evaluation is done on the local GPU for both models"
    split = args.split
    lock = OUT / "test_done.json"
    mods = args.modality
    for mod in mods:
        ms = models(mod)
        if "yolov5n" not in ms:
            raise SystemExit(f"YOLOv5n {mod} checkpoint missing: outputs/runs/comparison/yolov5n_{mod}/{V5_RUN}/weights/best.pt "
                             "(train with notebooks/04_train_yolov5n_{mod}_comparison.ipynb, unzip the bundle there)")
    if split == "test":
        if lock.exists():
            raise SystemExit(f"Test already evaluated once ({rel(lock)}); use the saved numbers (stage report).")
        val_done = all((OUT / "eval" / f"{m}_val.json").exists() for m in mods)
        if not val_done:
            raise SystemExit("evaluate --split val first (both checkpoints final) before the single Test evaluation")
    res_all = {}
    for mod in mods:
        root = DATA[mod][split]
        sub = "val" if split == "val" else "test"
        y = data_yaml(mod, split)
        res = {"modality": mod, "split": split, "dataset": rel(root), "images": len(list((root / "images" / sub).glob("*.png"))),
               "time": datetime.now().isoformat(timespec="seconds"), "gpu": torch.cuda.get_device_name(0), "models": {}}
        for name, info in models(mod).items():
            print(f"== {mod} {split} {name}", flush=True)
            mdl = YOLO(str(info["weights"]))
            v = mdl.val(data=str(y), split="val", imgsz=IMGSZ, batch=16, conf=0.001, iou=0.7, device=0, workers=0,
                        plots=True, project=str(OUT / "eval" / "plots"), name=f"{mod}_{split}_{name}", exist_ok=True, verbose=False)
            p, r = float(v.box.mp), float(v.box.mr)
            res["models"][name] = {"weights": rel(info["weights"]), "sha256": info["sha256"],
                                   "ultralytics_val": {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                                                       "mAP50": float(v.box.map50), "mAP50-95": float(v.box.map),
                                                       "speed_ms": {k: round(float(x), 3) for k, x in v.speed.items()}},
                                   "operating_point": operating_point(mdl, root / "images" / sub, root / "labels" / sub)}
            print(json.dumps(res["models"][name], indent=1)[:900], flush=True)
        (OUT / "eval").mkdir(parents=True, exist_ok=True)
        (OUT / "eval" / f"{mod}_{split}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        res_all[mod] = res
    if split == "test":
        lock.write_text(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "modalities": mods,
                                    "note": "single Test evaluation of the YOLOv5n vs YOLO11n comparison (user decision 2026-11-09)"},
                                   indent=1), encoding="utf-8")
    return 0


# ------------------------------------------------------------------------------------------------------------- bench
def bench(args) -> int:
    from ultralytics import YOLO
    import cv2
    import torch
    assert torch.cuda.is_available()
    out = {"gpu": torch.cuda.get_device_name(0), "imgsz": IMGSZ, "batch": 1, "images": 200, "warmup": 20,
           "time": datetime.now().isoformat(timespec="seconds"), "models": {}}
    for mod in args.modality:
        files = sorted((DATA[mod]["val"] / "images" / "val").glob("*.png"))
        imgs = [cv2.imread(str(f)) for f in random.Random(0).sample(files, min(220, len(files)))]
        for name, info in models(mod).items():
            mdl = YOLO(str(info["weights"]))
            from ultralytics.utils.torch_utils import get_flops
            n_params = sum(p.numel() for p in mdl.model.parameters())
            row = {"params": int(n_params), "gflops": round(float(get_flops(mdl.model, IMGSZ)), 2),
                   "size_mb": round(info["weights"].stat().st_size / 1e6, 2), "blocks": len(mdl.model.model)}
            for prec, q in (("fp16", 16), ("fp32", None)):
                kw = {"quantize": q} if q else {}
                for im in imgs[:20]:
                    mdl.predict(im, imgsz=IMGSZ, conf=OP_CONF, device=0, verbose=False, **kw)
                tot, inf = [], []
                for im in imgs[20:]:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    r = mdl.predict(im, imgsz=IMGSZ, conf=OP_CONF, device=0, verbose=False, **kw)[0]
                    torch.cuda.synchronize(); tot.append((time.perf_counter() - t0) * 1000); inf.append(r.speed["inference"])
                row[prec] = {"end_to_end_ms": round(float(np.median(tot)), 2), "inference_ms": round(float(np.median(inf)), 2),
                             "fps": round(1000 / float(np.median(tot)), 1)}
            out["models"][f"{mod}/{name}"] = row
            print(mod, name, json.dumps(row), flush=True)
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    (OUT / "eval" / "bench.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


# ------------------------------------------------------------------------------------------------------------ report
def train_info(run: Path) -> dict:
    hist = list(csv.DictReader(open(run / "results.csv", encoding="utf-8")))
    hist = [{k.strip(): v.strip() for k, v in r.items()} for r in hist]
    best = max(hist, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    mv = json.loads((run / "metrics_val.json").read_text(encoding="utf-8")) if (run / "metrics_val.json").exists() else {}
    args = load_yaml(run / "args.yaml") if (run / "args.yaml").exists() else {}
    # Ultralytics restarts the "time" column at 0 after a resume: sum the last value of every segment
    hours, n_seg = None, 0
    if hist and "time" in hist[-1]:
        t = [float(r["time"]) for r in hist]
        ends = [t[i] for i in range(len(t) - 1) if t[i + 1] < t[i]] + [t[-1]]
        hours, n_seg = round(sum(ends) / 3600, 2), len(ends)
    return {"epochs_run": len(hist), "best_epoch": int(float(best["epoch"])),
            "train_hours": hours, "train_segments": n_seg,
            "dataset": mv.get("dataset"), "dataset_sha256": mv.get("dataset_sha256"), "gpu": mv.get("gpu"),
            "ultralytics": mv.get("ultralytics"), "optimizer": args.get("optimizer"), "lr0": args.get("lr0"),
            "batch": args.get("batch"), "imgsz": args.get("imgsz"), "epochs": args.get("epochs"), "patience": args.get("patience"),
            "seed": args.get("seed"), "hist": hist,
            "model_arg": (f"{mv.get('model')} (resume từ last.pt)" if "last.pt" in str(args.get("model")) else Path(str(args.get("model"))).name)}


def report(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    ev = {(m, s): json.loads((OUT / "eval" / f"{m}_{s}.json").read_text(encoding="utf-8"))
          for m in ("ct", "mri") for s in ("val", "test") if (OUT / "eval" / f"{m}_{s}.json").exists()}
    bench_j = json.loads((OUT / "eval" / "bench.json").read_text(encoding="utf-8")) if (OUT / "eval" / "bench.json").exists() else None
    if not ev:
        raise SystemExit("no evaluation yet (stage eval)")
    f3 = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
    L = ["# So sánh YOLOv5n và YOLO11n (thí nghiệm đối chứng)", "",
         f"_Sinh bởi `scripts/78_yolov5n_comparison.py --stage report` lúc {datetime.now():%Y-%m-%d %H:%M}. Chỉ số liệu đo được; "
         "YOLO11n vẫn là mô hình chính, không bị thay thế._", ""]
    L += ["## 6. Cấu hình huấn luyện", "",
          "- YOLOv5n = `yolov5nu.pt` (YOLOv5n chính thức của Ultralytics, COCO pretrained, chạy trong Ultralytics 8.4.182: backbone/neck YOLOv5, "
          "đầu dự đoán anchor-free tách nhánh của thư viện hiện tại). YOLO11n = checkpoint đã khóa của website (SHA256 kiểm tra).",
          "- Cùng dữ liệu (CT `yolo_ct_full.zip`, MRI `yolo_mri_pilot.zip`), cùng chia tập theo bệnh nhân, cùng siêu tham số "
          "(notebook sinh từ cùng generator), cùng GPU khi đánh giá, cùng mã đánh giá.", ""]
    rows = []
    for mod in ("ct", "mri"):
        ms = models(mod)
        for name in ("yolo11n", "yolov5n"):
            if name in ms and (ms[name]["run"] / "results.csv").exists():
                t = train_info(ms[name]["run"])
                rows.append(f"| {mod.upper()} | {name} | {t['model_arg']} | {t['dataset']} | {t['epochs']} / {t['patience']} | {t['batch']} | "
                            f"{t['optimizer']} {t['lr0']} | {t['seed']} | {t['epochs_run']} | {t['best_epoch']} | "
                            f"{t['train_hours']} h ({t['train_segments']} phiên; {t['train_hours'] * 60 / t['epochs_run']:.1f} phút/epoch) | {t['gpu']} |")
    L += ["| Ảnh | Mô hình | Khởi tạo | Dữ liệu | Epoch / patience | Batch | Optimizer, lr0 | Seed | Epoch đã chạy | Epoch tốt nhất | Thời gian train | GPU |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|", *rows, "",
          "Thời gian train = tổng các phiên Colab (Ultralytics đặt lại đồng hồ về 0 sau mỗi lần resume); Colab T4 miễn phí dùng chung, "
          "tốc độ đọc dữ liệu thay đổi giữa các phiên, nên thời gian train chỉ để tham khảo, không dùng để so tốc độ mô hình (xem mục 9b). "
          "Checkpoint dùng để đánh giá là `best.pt` (fitness = mAP50-95 trên Val).", ""]
    hdr = ["| Tập | Mô hình | Ảnh | P | R | F1 | mAP50 | mAP50-95 | P @0,12 | R @0,12 | F1 @0,12 | TP / FP / FN @0,12 | Recall box nhỏ @0,12 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    note = ["", "P, R, F1: điểm F1 lớn nhất của Ultralytics (`val`, conf 0,001, IoU-NMS 0,7). “@0,12”: ngưỡng vận hành đã khóa của dự án "
            "(chọn trên Val cho YOLO11n), ghép cặp IoU ≥ 0,5. Box nhỏ: diện tích < 32×32 px. "
            "Test = mọi lát **có tổn thương** của BN Test (xuất bởi `scripts/96_final_test.py`), không có lát âm tính và nhóm chứng, nên "
            "precision trên Test không phản ánh báo nhầm trên lát bình thường. Test chạy **một lần** (khóa `outputs/runs/comparison/test_done.json`).", ""]
    for sec, mod in ((7, "ct"), (8, "mri")):
        L += [f"## {sec}. Kết quả YOLOv5n-{mod.upper()} và YOLO11n-{mod.upper()}", "", *hdr]
        for split, title in (("val", "Validation"), ("test", "Test")):
            e = ev.get((mod, split))
            if not e:
                L.append(f"| {title} | (chưa đánh giá) | | | | | | | | | | | |")
                continue
            for name in ("yolo11n", "yolov5n"):
                if name not in e["models"]:
                    continue
                u, o = e["models"][name]["ultralytics_val"], e["models"][name]["operating_point"]
                sr = f"{o['small_recall']:.3f} ({o['small_boxes']} box)" if o["small_recall"] is not None else "–"
                L.append(f"| {title} | {name} | {e['images']:,} | {f3(u['precision'])} | {f3(u['recall'])} | {f3(u['f1'])} | {f3(u['mAP50'])} | "
                         f"{f3(u['mAP50-95'])} | {f3(o['precision'])} | {f3(o['recall'])} | {f3(o['f1'])} | {o['tp']:,} / {o['fp']:,} / {o['fn']:,} | {sr} |")
        L += note + [f"Nguồn: `outputs/runs/comparison/eval/{mod}_val.json`, `{mod}_test.json`; hình PR/confusion matrix: "
                     f"`outputs/runs/comparison/eval/plots/{mod}_<split>_<model>/`.", ""]
    # 9a: differences
    L += ["## 9. Bảng so sánh metrics", "", "### 9a. Chênh lệch YOLOv5n − YOLO11n (điểm phần trăm; dương = YOLOv5n cao hơn)", "",
          "| Ảnh | Tập | P | R | F1 | mAP50 | mAP50-95 | R @0,12 | F1 @0,12 | Recall box nhỏ @0,12 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for mod in ("ct", "mri"):
        for split, title in (("val", "Validation"), ("test", "Test")):
            e = ev.get((mod, split))
            if not e or "yolov5n" not in e["models"]:
                continue
            a, b = e["models"]["yolov5n"], e["models"]["yolo11n"]
            d = lambda s, k: 100 * (a[s][k] - b[s][k])  # noqa: E731
            cells = [d("ultralytics_val", k) for k in ("precision", "recall", "f1", "mAP50", "mAP50-95")] + \
                    [d("operating_point", k) for k in ("recall", "f1", "small_recall")]
            L.append(f"| {mod.upper()} | {title} | " + " | ".join(f"{c:+.1f}" for c in cells) + " |")
    L.append("")
    runs_b = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "eval").glob("bench_run*.json"))] or ([bench_j] if bench_j else [])
    if runs_b:
        b0 = runs_b[0]
        L += ["### 9b. Tốc độ và kích thước (cùng máy, batch 1, ảnh 512)", "",
              f"GPU: {b0['gpu']}; mỗi lần đo: {b0['images']} ảnh Val, {b0['warmup']} ảnh khởi động, lấy trung vị; đo {len(runs_b)} lần "
              f"({', '.join(r['time'] for r in runs_b)}). Bảng ghi trung vị của {len(runs_b)} lần và khoảng [min–max]. "
              "Thời gian end-to-end = tiền xử lý + suy luận + NMS cho một ảnh (gọi `predict`).", "",
              "| Ảnh / mô hình | Tham số | GFLOPs | Checkpoint (MB) | fp16: ms/ảnh [min–max] | FPS fp16 | fp32: ms/ảnh [min–max] | FPS fp32 | Chỉ suy luận fp32 (ms) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, r in b0["models"].items():
            def agg(prec, field):
                v = [rb["models"][k][prec][field] for rb in runs_b if k in rb["models"]]
                return float(np.median(v)), min(v), max(v)
            m16, a16, z16 = agg("fp16", "end_to_end_ms")
            m32, a32, z32 = agg("fp32", "end_to_end_ms")
            i32 = agg("fp32", "inference_ms")[0]
            L.append(f"| {k} | {r['params']:,} | {r['gflops']} | {r['size_mb']} | {m16:.2f} [{a16:.2f}–{z16:.2f}] | {1000 / m16:.0f} | "
                     f"{m32:.2f} [{a32:.2f}–{z32:.2f}] | {1000 / m32:.0f} | {i32:.2f} |")
        L += ["", "GFLOPs tính ở ảnh 512. Nguồn: `outputs/runs/comparison/eval/bench_run*.json`.", ""]
    L += ["## 10. Biểu đồ so sánh", ""]
    # figure: grouped bars of the main metrics
    for split in ("val", "test"):
        es = [(m, ev[(m, split)]) for m in ("ct", "mri") if (m, split) in ev and "yolov5n" in ev[(m, split)]["models"]]
        if not es:
            continue
        keys = [("mAP50", "u"), ("mAP50-95", "u"), ("f1", "u"), ("recall", "o"), ("small_recall", "o")]
        lab = ["mAP50", "mAP50-95", "F1", "Recall @0,12", "Recall box nhỏ"]
        fig, axs = plt.subplots(1, len(es), figsize=(6.2 * len(es), 3.6), squeeze=False)
        for ax, (mod, e) in zip(axs[0], es):
            x = np.arange(len(keys))
            for j, (name, col) in enumerate((("yolo11n", "#1d5fd6"), ("yolov5n", "#f59e0b"))):
                v = [(e["models"][name]["ultralytics_val"] if s == "u" else e["models"][name]["operating_point"])[k] or 0 for k, s in keys]
                b = ax.bar(x + (j - 0.5) * 0.38, v, 0.38, label=name, color=col)
                ax.bar_label(b, fmt="%.2f", fontsize=7)
            ax.set_xticks(x, lab, fontsize=8); ax.set_ylim(0, 1); ax.set_title(f"{mod.upper()} – {'Validation' if split == 'val' else 'Test'}")
            ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig(FIG / f"so_sanh_{split}.png", dpi=130); plt.close(fig)
        L += [f"**Hình 10.{1 if split == 'val' else 2}.** Chỉ số chính trên {'Validation' if split == 'val' else 'Test'}.", "",
              f"![so sánh {split}](figures/yolov5n_vs_yolo11n/so_sanh_{split}.png)", ""]
    if runs_b:
        ks = list(runs_b[0]["models"])
        fig, ax = plt.subplots(figsize=(7, 3.4))
        x = np.arange(len(ks))
        for j, prec in enumerate(("fp16", "fp32")):
            v = [float(np.median([rb["models"][k][prec]["end_to_end_ms"] for rb in runs_b])) for k in ks]
            b = ax.bar(x + (j - 0.5) * 0.38, v, 0.38, label=prec, color=("#64748b", "#0f766e")[j])
            ax.bar_label(b, fmt="%.1f", fontsize=7)
        ax.set_xticks(x, ks, fontsize=8); ax.set_ylabel("ms / ảnh (thấp hơn = nhanh hơn)")
        ax.set_title(f"Thời gian xử lý một ảnh, {runs_b[0]['gpu'].replace('NVIDIA GeForce ', '')}, batch 1"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig(FIG / "toc_do.png", dpi=130); plt.close(fig)
        L += ["**Hình 10.3.** Thời gian end-to-end (trung vị các lần đo).", "", "![tốc độ](figures/yolov5n_vs_yolo11n/toc_do.png)", ""]
    # training curves
    curves = [(mod, name, train_info(ms[name]["run"])["hist"]) for mod in ("ct", "mri") for ms in [models(mod)]
              for name in ("yolo11n", "yolov5n") if name in ms and (ms[name]["run"] / "results.csv").exists()]
    if any(n == "yolov5n" for _, n, _ in curves):
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
        for ax, mod in zip(axs, ("ct", "mri")):
            for m, name, h in curves:
                if m == mod:
                    ax.plot([float(r["epoch"]) for r in h], [float(r["metrics/mAP50-95(B)"]) for r in h], label=name,
                            color="#1d5fd6" if name == "yolo11n" else "#f59e0b")
            ax.set_title(f"{mod.upper()}: mAP50-95 trên Val theo epoch"); ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(FIG / "duong_huan_luyen.png", dpi=130); plt.close(fig)
        L += ["**Hình 10.4.** mAP50-95 trên Val theo epoch (từ `results.csv` của từng lần train).", "",
              "![đường huấn luyện](figures/yolov5n_vs_yolo11n/duong_huan_luyen.png)", ""]
    # hand-written parts (written AFTER the numbers, citing them): intro = sections before the tables, analysis = after
    intro, analysis = (TEXT_DIR / "phan_mo_dau.md"), (TEXT_DIR / "phan_nhan_xet.md")
    if intro.exists():
        head, body = L[:4], L[4:]
        L = head + [intro.read_text(encoding="utf-8").rstrip(), ""] + body
    if analysis.exists():
        L += [analysis.read_text(encoding="utf-8").rstrip(), ""]
    else:
        L += ["## 4. Nhận xét", "", "_Phần nhận xét được viết sau khi có số liệu, dựa trên các bảng trên (không viết trước kết quả)._", ""]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("->", rel(REPORT))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "eval", "bench", "report"], required=True)
    ap.add_argument("--mod", choices=["ct", "mri"], default="ct")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--modality", nargs="+", default=["ct", "mri"])
    a = ap.parse_args()
    return {"smoke": smoke, "eval": evaluate, "bench": bench, "report": report}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
