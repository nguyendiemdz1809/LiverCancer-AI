"""Faster R-CNN ResNet-50 FPN vs YOLO11n on CT: control experiment with a detector from another family.

User decision 2026-11-26 (after discussing with the supervisor): compare YOLO11n with a classic two-stage detector,
CT only, evaluation on Val and ONCE on Test (user decision 2026-11-26). YOLO11n stays the detector of
the project, whatever the result; nothing of the locked pipeline changes. Results are reported as measured.

Stages (outputs in outputs/runs/comparison/frcnn_ct/, never in the YOLO11n run folders):
  smoke              1 short epoch on 400 Train / 200 Val images on the local GPU (pipeline check, not a result)
  check-evaluator    YOLO11n (locked weights) on CT Val with OUR evaluator (livercancer.detection.frcnn.detection_metrics)
                     vs Ultralytics `val` (outputs/runs/comparison/frcnn_ct/yolo11n_ref/ct_val.json) -> proves the mAP of Faster R-CNN is
                     computed by the same algorithm as the YOLO numbers
  eval --split val   Faster R-CNN best.pt on CT Val: mAP etc. + operating point at 0.12 (YOLO11n's locked threshold) and at
                     its own threshold chosen on Val with the project rule (max F2, 53_threshold_sweep)
  eval --split test  same on the CT Test GT export (lesion slices only), ONCE (lock test_done.json); the threshold comes
                     from Val. YOLO11n Test numbers are read from outputs/runs/comparison/frcnn_ct/yolo11n_ref/ct_test.json (measured once 2026-11-13, not re-run)
  bench              params, GFLOPs (torch FlopCounterMode, both models, same counter), speed batch 1 fp32/fp16 on the
                     local GPU (200 Val images, 20 warm-up, median)
  notebook           notebooks/04_train_frcnn_ct_comparison.ipynb (Colab T4; trains with the module inlined)
  report             reports/frcnn_vs_yolo11n_comparison.md (+ figures), only measured numbers
"""
from __future__ import annotations

import argparse
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
os.environ.setdefault("TORCH_HOME", str(ROOT / "outputs/runs/comparison/pretrained/torch"))

from livercancer.config import CONFIG_DIR, load_yaml  # noqa: E402
from livercancer.detection import frcnn  # noqa: E402

OUT = ROOT / "outputs/runs/comparison/frcnn_ct"
RUN_NAME = "frcnn_r50fpn_img512_comparison"
RUN = OUT / RUN_NAME
YOLO_EVAL = ROOT / "outputs/runs/comparison/frcnn_ct/yolo11n_ref"  # YOLO11n Ultralytics val, Val + Test (Test measured once)
DATA = {"val": ROOT / "data/processed/full/yolo_ct", "test": ROOT / "data/processed/final_test/yolo_ct"}
REPORT = ROOT / "reports/frcnn_vs_yolo11n_comparison.md"
FIG = ROOT / "reports/figures/frcnn_vs_yolo11n"
TEXT_DIR = ROOT / "reports/frcnn_vs_yolo11n"           # hand-written parts merged by --stage report
NOTEBOOK = ROOT / "notebooks/04_train_frcnn_ct_comparison.ipynb"
OP_CONF = 0.12
RELEASE_URL = "https://github.com/nguyendiemdz1809/LiverCancer-AI/releases/download/du-lieu-ct-v1/"
CT_ZIP_SHA = "0a7ed974856f5634b2860b692b83fa717e820031d8974862607a78a5e281cfdc"   # yolo_ct_full.zip (YOLO11n run)
TRAIN = {"epochs": 12, "batch": 8, "lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4, "warmup_iters": 1000,
         "patience": 5, "seed": 42, "amp": True}                     # Colab T4 (16 GB): notebook stage
# Local RTX 3050 4 GB (user choice 2026-11-26): batch 8 overflows VRAM (5.0 GB reserved -> shared memory, half speed),
# so batch 4 with the linear-scaling rule lr = 0.01 * 4 / 8; warm-up in iterations doubled to cover the same images.
TRAIN_LOCAL = {**TRAIN, "batch": 4, "lr": 0.005, "warmup_iters": 2000}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT).as_posix()


def yolo11n() -> Path:
    cfg = load_yaml(CONFIG_DIR / "yolo_inference.yaml")["ct"]
    w = ROOT / cfg["weights"]
    if sha256(w) != cfg["sha256"]:
        raise SystemExit("YOLO11n CT: sha256 mismatch (locked weights changed?)")
    return w


def dump(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------ smoke
def smoke(_a) -> int:
    import torch
    assert torch.cuda.is_available()
    out = OUT / "smoke" / "run"
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    s = frcnn.train(DATA["val"], out, epochs=1, batch=4, warmup_iters=5, max_train=400, max_val=200, workers=0, save_every=50)
    res = {k: v for k, v in s.items() if k != "config"}
    res.update({"seconds": round(time.time() - t0, 1), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "note": "pipeline check only, not a result", "gpu": torch.cuda.get_device_name(0)})
    dump(OUT / "smoke" / "smoke.json", res)
    print(json.dumps(res, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------------------ train
def train_local(_a) -> int:
    """Full training on the local GPU (resumes from RUN/last.pt). Keeps Windows awake while it runs."""
    import ctypes
    import torch
    assert torch.cuda.is_available()
    if os.name == "nt":   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no system sleep while this process runs
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    RUN.mkdir(parents=True, exist_ok=True)
    s = frcnn.train(DATA["val"], RUN, workers=2, save_every=500, **TRAIN_LOCAL)
    env = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "torchvision": __import__("torchvision").__version__, "dataset": rel(DATA["val"]),
           "time": datetime.now().isoformat(timespec="seconds"), "train": TRAIN_LOCAL}
    dump(RUN / "env.json", env)
    print(json.dumps({k: v for k, v in s.items() if k not in ("history", "config")}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------- check-evaluator
def yolo_preds(weights: Path, root: Path, split: str) -> dict[str, dict]:
    from ultralytics import YOLO
    m = YOLO(str(weights))
    items = frcnn.list_split(root, split)
    out = {}
    for i in range(0, len(items), 32):
        b = items[i:i + 32]
        for (ip, lp), r in zip(b, m.predict([str(p) for p, _ in b], imgsz=frcnn.IMGSZ, conf=0.001, iou=0.7, max_det=300,
                                            device=0, verbose=False)):
            h, w = r.orig_shape
            out[ip.stem] = {"boxes": r.boxes.xyxy.cpu().numpy(), "conf": r.boxes.conf.cpu().numpy(),
                            "gt": frcnn.read_labels(lp, w, h)}
    return out


def check_evaluator(_a) -> int:
    ref = json.loads((YOLO_EVAL / "ct_val.json").read_text(encoding="utf-8"))["models"]["yolo11n"]
    t0 = time.time()
    p = yolo_preds(yolo11n(), DATA["val"], "val")
    ours = frcnn.detection_metrics(p)
    op = frcnn.operating_point(p, OP_CONF)
    thr = frcnn.choose_threshold(p)
    u = ref["ultralytics_val"]
    diff = {k: round(ours[k] - u[k], 4) for k in ("precision", "recall", "f1", "mAP50", "mAP50-95")}
    res = {"model": rel(yolo11n()), "split": "val", "ours": ours, "ultralytics_val": u, "diff": diff,
           "operating_point_ours": op, "operating_point_78": ref["operating_point"],
           "threshold_rule_on_yolo11n": {"conf": thr["conf"], "f2": thr["f2"], "locked": OP_CONF},
           "seconds": round(time.time() - t0, 1), "time": datetime.now().isoformat(timespec="seconds"),
           "ok": all(abs(diff[k]) <= 0.005 for k in ("mAP50", "mAP50-95"))}
    dump(OUT / "eval" / "evaluator_check.json", res)
    print(json.dumps({k: res[k] for k in ("diff", "threshold_rule_on_yolo11n", "ok")}, indent=1))
    print("operating point ours vs reference:", {k: (op[k], ref["operating_point"][k]) for k in ("tp", "fp", "fn")})
    return 0 if res["ok"] else 1


# ------------------------------------------------------------------------------------------------------------- eval
def evaluate(a) -> int:
    import torch
    assert torch.cuda.is_available()
    split = a.split
    lock = OUT / "test_done.json"
    best = RUN / "best.pt"
    if not best.exists():
        raise SystemExit(f"missing {rel(best)}: train with {rel(NOTEBOOK)} and unzip the bundle into {rel(OUT)}")
    if split == "test":
        if lock.exists():
            raise SystemExit(f"Test already evaluated once ({rel(lock)}); use the saved numbers (stage report).")
        if not (OUT / "eval" / "ct_val.json").exists():
            raise SystemExit("evaluate --split val first (threshold of Faster R-CNN is chosen there)")
    model, ck = frcnn.load_weights(best, "cuda")
    t0 = time.time()
    sub = "val" if split == "val" else "test"
    p = frcnn.predict_split(model, DATA[split], sub, "cuda", batch=8)
    res = {"model": "faster_rcnn_r50_fpn", "weights": rel(best), "sha256": sha256(best), "best_epoch": ck.get("epoch"),
           "split": split, "dataset": rel(DATA[split]), "images": len(p), "gpu": torch.cuda.get_device_name(0),
           "time": datetime.now().isoformat(timespec="seconds"), "metrics": frcnn.detection_metrics(p),
           "operating_point_0.12": frcnn.operating_point(p, OP_CONF)}
    if split == "val":
        res["threshold_choice"] = frcnn.choose_threshold(p)
        res["operating_point_own"] = res["threshold_choice"]
    else:
        thr = json.loads((OUT / "eval" / "ct_val.json").read_text(encoding="utf-8"))["threshold_choice"]["conf"]
        res["operating_point_own"] = frcnn.operating_point(p, thr)
    res["seconds"] = round(time.time() - t0, 1)
    dump(OUT / "eval" / f"ct_{split}.json", res)
    if split == "test":
        dump(lock, {"time": res["time"], "note": "single Test evaluation of the Faster R-CNN vs YOLO11n comparison "
                                                  "(user decision 2026-11-26)", "weights_sha256": res["sha256"]})
    print(json.dumps({k: res[k] for k in ("metrics", "operating_point_0.12", "operating_point_own")}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------------------ bench
def count_flops(model, x) -> float:
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    with torch.inference_mode(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9


def bench(a) -> int:
    import torch
    from ultralytics import YOLO
    assert torch.cuda.is_available()
    files = sorted((DATA["val"] / "images/val").glob("*.png"))
    pick = random.Random(0).sample(files, 220)
    out = {"gpu": torch.cuda.get_device_name(0), "imgsz": frcnn.IMGSZ, "batch": 1, "images": 200, "warmup": 20,
           "time": datetime.now().isoformat(timespec="seconds"), "models": {}}
    # YOLO11n: Ultralytics predict protocol (predict = preprocess + inference + NMS)
    y = YOLO(str(yolo11n()))
    import cv2
    imgs = [cv2.imread(str(f)) for f in pick]
    row = {"params": int(sum(p.numel() for p in y.model.parameters())), "size_mb": round(yolo11n().stat().st_size / 1e6, 2)}
    ym = y.model.float().eval().cuda()
    row["gflops_flopcounter"] = round(count_flops(ym, torch.zeros(1, 3, 512, 512, device="cuda")), 2)
    for prec, q in (("fp16", 16), ("fp32", None)):
        kw = {"quantize": q} if q else {}
        for im in imgs[:20]:
            y.predict(im, imgsz=512, conf=OP_CONF, device=0, verbose=False, **kw)
        tot = []
        for im in imgs[20:]:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            y.predict(im, imgsz=512, conf=OP_CONF, device=0, verbose=False, **kw)
            torch.cuda.synchronize(); tot.append((time.perf_counter() - t0) * 1000)
        row[prec] = {"end_to_end_ms": round(float(np.median(tot)), 2), "fps": round(1000 / float(np.median(tot)), 1)}
    out["models"]["yolo11n"] = row
    del y, ym
    torch.cuda.empty_cache()
    # Faster R-CNN: read PNG + forward (transform, RPN, RoI heads, NMS) for one image
    m, _ = frcnn.load_weights(RUN / "best.pt", "cuda")
    row = {"params": int(sum(p.numel() for p in m.parameters())), "size_mb": round((RUN / "best.pt").stat().st_size / 1e6, 2)}
    x0 = torch.from_numpy(frcnn.load_image(pick[0]).astype(np.float32) / 255)[None].expand(3, -1, -1).cuda()
    row["gflops_flopcounter"] = round(count_flops(m, [x0]), 2)
    for prec in ("fp16", "fp32"):
        def run(f):
            im = frcnn.load_image(f).astype(np.float32) / 255
            x = torch.from_numpy(im)[None].expand(3, -1, -1).cuda()
            with torch.inference_mode(), torch.autocast("cuda", enabled=prec == "fp16"):
                return m([x])
        for f in pick[:20]:
            run(f)
        tot = []
        for f in pick[20:]:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            run(f)
            torch.cuda.synchronize(); tot.append((time.perf_counter() - t0) * 1000)
        row[prec] = {"end_to_end_ms": round(float(np.median(tot)), 2), "fps": round(1000 / float(np.median(tot)), 1)}
    out["models"]["faster_rcnn"] = row
    n = len(list((OUT / "eval").glob("bench_run*.json"))) + 1
    dump(OUT / "eval" / f"bench_run{n}.json", out)
    print(json.dumps(out, indent=1))
    return 0


# --------------------------------------------------------------------------------------------------------- notebook
def notebook(_a) -> int:
    src = (ROOT / "src/livercancer/detection/frcnn.py").read_text(encoding="utf-8")
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}  # noqa: E731
    code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],  # noqa: E731
                      "source": s.strip("\n").splitlines(True)}
    cfg = json.dumps(TRAIN)
    cells = [
        md(f"""
# Huấn luyện Faster R-CNN ResNet-50 FPN trên CT – mô hình đối chứng của YOLO11n

Thí nghiệm đối chứng (người dùng quyết định 26/11/2026): so YOLO11n với một bộ phát hiện **khác họ YOLO**, loại hai giai đoạn kinh điển.
YOLO11n vẫn là mô hình chính của đồ án; notebook này **chỉ huấn luyện** Faster R-CNN và **chỉ đánh giá trên Validation**
(Test được đánh giá đúng một lần ở máy local bằng `scripts/79_frcnn_comparison.py --stage eval --split test`).

| | |
|---|---|
| Mô hình | `torchvision` `fasterrcnn_resnet50_fpn` (COCO pretrained), 2 lớp (nền, `lesion`) |
| Dữ liệu | **Cùng** `yolo_ct_full.zip` của YOLO11n-CT (Train 34.873 / Val 14.627 ảnh 512×512, gồm cả lát âm tính) |
| Ảnh | 512 × 512, không đổi kích thước |
| Huấn luyện | {TRAIN['epochs']} epoch tối đa, batch {TRAIN['batch']}, SGD lr {TRAIN['lr']} (momentum {TRAIN['momentum']}, weight decay {TRAIN['weight_decay']}), warm-up {TRAIN['warmup_iters']} vòng + cosine, AMP, seed {TRAIN['seed']} |
| Augmentation | lật ngang 0,5, độ sáng ±30 % (như `fliplr`, `hsv_v` của YOLO; mosaic là kỹ thuật riêng của YOLO nên không dùng) |
| Chọn checkpoint | `best.pt` = mAP50-95 cao nhất trên **Val** (giống `fitness` của YOLO); dừng sớm sau {TRAIN['patience']} epoch không cải thiện |
| mAP | Tính bằng cùng thuật toán của Ultralytics (`ap_per_class`, IoU 0,50:0,95) |

**Cách chạy:** *Thời gian chạy → Thay đổi loại thời gian chạy → GPU T4*, rồi *Chạy tất cả*. Checkpoint lưu trên Google Drive
(`MyDrive/LiverCancer_AI/runs/comparison/frcnn_ct/`) mỗi {500} vòng, nên khi Colab ngắt chỉ cần **chạy lại tất cả**: notebook tự tiếp tục.
Xong thì tải `{RUN_NAME}_bundle.zip` trên Drive về, giải nén vào `outputs/runs/comparison/frcnn_ct/`.
"""),
        code(f"""
import torch, subprocess, sys, json, os, shutil, zipfile, hashlib, time
from pathlib import Path
assert torch.cuda.is_available(), 'Không có GPU: Thời gian chạy → Thay đổi loại thời gian chạy → GPU T4'
print(torch.cuda.get_device_name(0), '|', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
TRAIN = {cfg}
RUN_NAME = {RUN_NAME!r}
"""),
        code("""
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = Path('/content/drive/MyDrive/LiverCancer_AI')
RUN_DIR = DRIVE_ROOT / 'runs' / 'comparison' / 'frcnn_ct' / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
print('Checkpoint:', RUN_DIR, '| đã có last.pt:', (RUN_DIR / 'last.pt').exists())
"""),
        code("""
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.182'], check=True)   # chỉ dùng ap_per_class
import torchvision, ultralytics
print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| ultralytics', ultralytics.__version__)
"""),
        code(f"""
# Dữ liệu: ưu tiên bản đã có trên Drive (đã dùng để train YOLO11n-CT), nếu không thì tải 3 phần từ GitHub Releases
RELEASE_URL = {RELEASE_URL!r}
ZIP_SHA = {CT_ZIP_SHA!r}
WORK = Path('/content/work'); WORK.mkdir(exist_ok=True)
ZIP = WORK / 'yolo_ct_full.zip'

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

if not ZIP.exists():
    d = DRIVE_ROOT / 'data' / 'yolo_ct_full.zip'
    if d.exists():
        print('Chép từ Drive:', d); shutil.copy(d, ZIP)
    else:
        import urllib.request
        with open(ZIP, 'wb') as out:
            for i in (1, 2, 3):
                print('Tải phần', i, '/ 3 ...', flush=True)
                with urllib.request.urlopen(RELEASE_URL + f'yolo_ct_full.zip.part{{i}}', timeout=600) as r:
                    shutil.copyfileobj(r, out, 64 << 20)
h = sha256(ZIP)
if h != ZIP_SHA:
    ZIP.unlink(); raise RuntimeError('SHA256 sai, tệp hỏng: chạy lại ô này')
print('SHA256 OK', ZIP.stat().st_size / 1e9, 'GB')
DATA = WORK / 'yolo_ct'
if not (DATA / 'data.yaml').exists():
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(WORK)
print('Train', len(list((DATA / 'images/train').glob('*.png'))), '| Val', len(list((DATA / 'images/val').glob('*.png'))))
"""),
        md("Mã mô hình (chép nguyên văn từ `src/livercancer/detection/frcnn.py` của dự án):"),
        code("Path('/content/frcnn.py').write_text(" + repr(src) + ", encoding='utf-8')\n"
             "sys.path.insert(0, '/content')\nimport frcnn\nprint('OK')"),
        code("""
# Huấn luyện (tự tiếp tục từ RUN_DIR/last.pt nếu đã có). Mỗi epoch: train rồi đánh giá toàn bộ Val.
summary = frcnn.train(DATA, RUN_DIR, workers=2, save_every=500, **TRAIN)
print({k: v for k, v in summary.items() if k not in ('history', 'config')})
"""),
        code("""
# Đóng gói kết quả (không gồm last.pt: chỉ dùng để resume) -> tải file này về máy
env = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'torchvision': torchvision.__version__,
       'ultralytics': ultralytics.__version__, 'dataset': 'yolo_ct_full.zip', 'dataset_sha256': ZIP_SHA,
       'time': time.strftime('%Y-%m-%d %H:%M:%S')}
(RUN_DIR / 'env.json').write_text(json.dumps(env, indent=1), encoding='utf-8')
bundle = RUN_DIR.parent / f'{RUN_NAME}_bundle.zip'
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_STORED) as z:
    for f in ('best.pt', 'results.csv', 'train_summary.json', 'env.json'):
        z.write(RUN_DIR / f, f'{RUN_NAME}/{f}')
print('Tải về:', bundle, round(bundle.stat().st_size / 1e6, 1), 'MB')
import pandas as pd
display(pd.read_csv(RUN_DIR / 'results.csv'))
"""),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("->", rel(NOTEBOOK))
    return 0


# ----------------------------------------------------------------------------------------------------------- report
def f3(x) -> str:
    return "–" if x is None else f"{x:.3f}".replace(".", ",")


def nn(x) -> str:
    return f"{x:,}".replace(",", ".")


def vn(x) -> str:
    return str(x).replace(".", ",")


def report(_a) -> int:
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)

    def rd(p: Path):
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    fr = {s: rd(OUT / "eval" / f"ct_{s}.json") for s in ("val", "test")}
    yo = {s: rd(YOLO_EVAL / f"ct_{s}.json") for s in ("val", "test")}
    chk = rd(OUT / "eval" / "evaluator_check.json")
    benches = [rd(b) for b in sorted((OUT / "eval").glob("bench_run*.json"))]
    summ = rd(RUN / "train_summary.json")
    if not fr["val"]:
        raise SystemExit("no Faster R-CNN evaluation yet (stage eval)")
    thr = fr["val"]["threshold_choice"]["conf"]
    yrun = ROOT / load_yaml(CONFIG_DIR / "yolo_inference.yaml")["ct"]["run"]
    yh = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(yrun / "results.csv", encoding="utf-8"))]
    ybest = max(yh, key=lambda r: float(r["metrics/mAP50-95(B)"]))

    L = ["# So sánh Faster R-CNN ResNet-50 FPN và YOLO11n trên CT (thí nghiệm đối chứng)", "",
         f"_Sinh bởi `scripts/79_frcnn_comparison.py --stage report` lúc {datetime.now():%Y-%m-%d %H:%M}. Chỉ số liệu đo được. "
         "YOLO11n vẫn là mô hình chính của đồ án, không bị thay thế._", ""]
    # 2. models
    L += ["## 2. Hai mô hình được so sánh", "", "| | YOLO11n | Faster R-CNN ResNet-50 FPN |", "|---|---|---|",
          "| Họ mô hình | YOLO (Ultralytics, 2024) | R-CNN (Ren và cộng sự, 2015; FPN: Lin và cộng sự, 2017) |",
          "| Số giai đoạn | Một giai đoạn: dự đoán hộp trực tiếp trên bản đồ đặc trưng | Hai giai đoạn: mạng đề xuất vùng (RPN), rồi phân loại và tinh chỉnh từng vùng (RoI head) |",
          "| Backbone / neck | C3k2, SPPF, C2PSA (chú ý theo vị trí) / PANet | ResNet-50 / Feature Pyramid Network |",
          "| Đầu dự đoán | Anchor-free, tách nhánh, DFL | RPN dùng anchor; RoIAlign 7×7 + 2 lớp kết nối đầy đủ |",
          "| Hậu xử lý | NMS IoU 0,7 | NMS IoU 0,5 (mặc định torchvision) |",
          "| Khởi tạo | COCO (`yolo11n.pt`) | COCO (torchvision `COCO_V1`) |",
          "| mAP50-95 công bố trên COCO | 39,5 | 37,0 |"]
    if benches:
        b = benches[0]["models"]
        L += [f"| Tham số (đo trên checkpoint 1 lớp) | {nn(b['yolo11n']['params'])} | {nn(b['faster_rcnn']['params'])} |",
              f"| GFLOPs ở ảnh 512 (cùng bộ đếm `torch` FlopCounterMode) | {vn(b['yolo11n']['gflops_flopcounter'])} | {vn(b['faster_rcnn']['gflops_flopcounter'])} |",
              f"| Dung lượng `best.pt` (MB) | {vn(b['yolo11n']['size_mb'])} | {vn(b['faster_rcnn']['size_mb'])} |"]
    L += ["", "Số trên COCO lấy từ trang công bố của Ultralytics và torchvision (ảnh tự nhiên, 80 lớp), không suy ra được cho CT một lớp tổn thương.", ""]
    # 3. data / evaluation
    L += ["## 3. Dữ liệu và cách đánh giá", "",
          "- **Cùng dữ liệu với YOLO11n-CT:** `yolo_ct_full.zip` (Train 34.873 / Val 14.627 lát 512×512, gồm cả lát không có tổn thương; chia theo bệnh nhân). "
          "Test = 7.163 lát có tổn thương của 42 BN ung thư Test. Test không có lát âm tính và nhóm chứng, nên precision trên Test "
          "không phản ánh báo nhầm trên lát bình thường.",
          "- **Cùng thuật toán tính chỉ số:** mAP của Faster R-CNN tính bằng mã viết lại đúng thuật toán của Ultralytics (ghép cặp tham lam theo độ tin cậy giảm dần "
          "tại 10 ngưỡng IoU 0,50:0,95, rồi `ap_per_class`). P, R, F1 là điểm F1 lớn nhất, như Ultralytics báo cáo."]
    if chk:
        d, o = chk["diff"], chk["operating_point_ours"]
        L += [f"- **Kiểm chứng bộ tính:** chạy lại YOLO11n trên Val bằng mã này và so với `val` của Ultralytics: mAP50 lệch {vn(d['mAP50'])}, "
              f"mAP50-95 lệch {vn(d['mAP50-95'])} (do fp16/fp32); TP / FP / FN ở ngưỡng 0,12 trùng khớp ({nn(o['tp'])} / {nn(o['fp'])} / {nn(o['fn'])}); "
              f"quy tắc chọn ngưỡng của dự án (F2 lớn nhất trên Val) áp lên YOLO11n cho lại đúng {vn(chk['threshold_rule_on_yolo11n']['conf'])}. "
              "Nguồn: `outputs/runs/comparison/frcnn_ct/eval/evaluator_check.json`."]
    L += ["- **Ngưỡng vận hành:** độ tin cậy của hai mô hình tính khác nhau (YOLO: sigmoid; Faster R-CNN: softmax 2 lớp), nên ngưỡng 0,12 của YOLO11n "
          f"không tương đương với Faster R-CNN. Báo cáo cả hai: ở 0,12 và ở ngưỡng riêng của Faster R-CNN chọn **trên Val** bằng cùng quy tắc F2 (= {vn(thr)}). "
          "Với YOLO11n, ngưỡng riêng chính là 0,12.",
          "- Ghép cặp ở ngưỡng vận hành: IoU ≥ 0,5, một-một, theo độ tin cậy giảm dần. Box nhỏ: diện tích < 32×32 px.", ""]
    # 4. training
    L += ["## 4. Cấu hình huấn luyện", "", "| | YOLO11n | Faster R-CNN |", "|---|---|---|",
          "| Nơi huấn luyện | Colab T4 | Máy local, RTX 3050 Laptop 4 GB |",
          f"| Epoch tối đa / dừng sớm | 50 / patience 20 | {TRAIN_LOCAL['epochs']} / patience {TRAIN_LOCAL['patience']} |",
          f"| Batch, optimizer | 32, SGD lr0 0,01, cosine | {TRAIN_LOCAL['batch']}, SGD lr 0,005 (= 0,01 × 4/8), momentum 0,9, cosine, warm-up {nn(TRAIN_LOCAL['warmup_iters'])} vòng |",
          "| Augmentation | mosaic, lật ngang, tịnh tiến, co giãn, độ sáng | lật ngang 0,5, độ sáng ±30 % |",
          "| Chọn checkpoint | mAP50-95 lớn nhất trên Val | mAP50-95 lớn nhất trên Val (cùng tiêu chí) |"]
    if summ:
        L += [f"| Epoch đã chạy / epoch tốt nhất | {len(yh)} / {int(float(ybest['epoch']))} | {summ['epochs_run']} / {summ['best_epoch']} |",
              f"| Thời gian huấn luyện (không gồm đánh giá Val) | – | {vn(summ['train_hours'])} giờ |"]
    L += ["| Seed | 42 | 42 |", "",
          "Faster R-CNN dùng batch 4 vì batch 8 vượt 4 GB VRAM (5,0 GB, tràn sang bộ nhớ chia sẻ, chậm một nửa); tốc độ học giảm theo quy tắc tỉ lệ tuyến tính. "
          "Mosaic là kỹ thuật riêng của YOLO; công thức chuẩn của torchvision cho Faster R-CNN chỉ dùng lật ngang.", ""]
    # 5-6. results
    hdr = ["| Mô hình | P | R | F1 | mAP50 | mAP50-95 | Ngưỡng | P | R | F1 | TP / FP / FN | Recall box nhỏ |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def op_cells(o):
        return f"{f3(o['precision'])} | {f3(o['recall'])} | {f3(o['f1'])} | {nn(o['tp'])} / {nn(o['fp'])} / {nn(o['fn'])} | {f3(o['small_recall'])}"
    for sec, split, title in ((5, "val", "Validation (14.627 lát, 7.631 box)"), (6, "test", "Test (7.163 lát có tổn thương)")):
        L += [f"## {sec}. Kết quả trên {title}", ""]
        if not fr[split]:
            L += ["_(chưa đánh giá)_", ""]
            continue
        u, yop = yo[split]["models"]["yolo11n"]["ultralytics_val"], yo[split]["models"]["yolo11n"]["operating_point"]
        m = fr[split]["metrics"]
        L += [*hdr,
              f"| YOLO11n | {f3(u['precision'])} | {f3(u['recall'])} | {f3(u['f1'])} | {f3(u['mAP50'])} | {f3(u['mAP50-95'])} | 0,12 | {op_cells(yop)} |",
              f"| Faster R-CNN | {f3(m['precision'])} | {f3(m['recall'])} | {f3(m['f1'])} | {f3(m['mAP50'])} | {f3(m['mAP50-95'])} | 0,12 | {op_cells(fr[split]['operating_point_0.12'])} |",
              f"| Faster R-CNN | | | | | | {vn(thr)} (riêng) | {op_cells(fr[split]['operating_point_own'])} |", ""]
    L += ["Năm cột chỉ số đầu không phụ thuộc ngưỡng (P, R, F1 ở điểm F1 lớn nhất); các cột sau tính tại ngưỡng vận hành. "
          "Nguồn: `outputs/runs/comparison/frcnn_ct/eval/ct_{val,test}.json` (Faster R-CNN), `outputs/runs/comparison/frcnn_ct/yolo11n_ref/ct_{val,test}.json` (YOLO11n, Ultralytics `val`).", "",
          "### Chênh lệch Faster R-CNN − YOLO11n (điểm phần trăm; dương = Faster R-CNN cao hơn)", "",
          "| Tập | P | R | F1 | mAP50 | mAP50-95 | Recall @ngưỡng riêng | F1 @ngưỡng riêng | Recall box nhỏ @ngưỡng riêng |", "|---|---|---|---|---|---|---|---|---|"]
    for split, title in (("val", "Validation"), ("test", "Test")):
        if fr[split]:
            m, u = fr[split]["metrics"], yo[split]["models"]["yolo11n"]["ultralytics_val"]
            o, yop = fr[split]["operating_point_own"], yo[split]["models"]["yolo11n"]["operating_point"]
            cells = [m[k] - u[k] for k in ("precision", "recall", "f1", "mAP50", "mAP50-95")] + \
                    [o[k] - yop[k] for k in ("recall", "f1", "small_recall")]
            L.append(f"| {title} | " + " | ".join(f"{100 * c:+.1f}".replace(".", ",") for c in cells) + " |")
    L.append("")
    # 7. speed
    if benches:
        L += ["## 7. Tốc độ và kích thước (cùng máy, batch 1, ảnh 512)", "",
              f"GPU: {benches[0]['gpu']}; {benches[0]['images']} ảnh Val, {benches[0]['warmup']} ảnh khởi động, lấy trung vị; {len(benches)} lần đo. "
              "Thời gian = tiền xử lý + suy luận + hậu xử lý (NMS) cho một ảnh.", "",
              "| Mô hình | fp16: ms/ảnh [min–max] | Ảnh/giây fp16 | fp32: ms/ảnh [min–max] | Ảnh/giây fp32 |", "|---|---|---|---|---|"]
        for k, lab in (("yolo11n", "YOLO11n"), ("faster_rcnn", "Faster R-CNN")):
            cells = []
            for prec in ("fp16", "fp32"):
                v = [bb["models"][k][prec]["end_to_end_ms"] for bb in benches]
                med = float(np.median(v))
                cells += [f"{med:.2f} [{min(v):.2f}–{max(v):.2f}]".replace(".", ","), f"{1000 / med:.0f}"]
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L += ["", "Nguồn: `outputs/runs/comparison/frcnn_ct/eval/bench_run*.json`.", ""]
    # 8. figures
    L += ["## 8. Biểu đồ", ""]
    splits = [s for s in ("val", "test") if fr[s]]
    fig, axs = plt.subplots(1, len(splits), figsize=(6 * len(splits), 3.6), squeeze=False)
    for ax, split in zip(axs[0], splits):
        u, yop = yo[split]["models"]["yolo11n"]["ultralytics_val"], yo[split]["models"]["yolo11n"]["operating_point"]
        m, o = fr[split]["metrics"], fr[split]["operating_point_own"]
        vals = (([u["mAP50"], u["mAP50-95"], u["f1"], yop["recall"], yop["small_recall"]], "YOLO11n", "#1d5fd6"),
                ([m["mAP50"], m["mAP50-95"], m["f1"], o["recall"], o["small_recall"]], "Faster R-CNN", "#b45309"))
        x = np.arange(5)
        for j, (v, lab, col) in enumerate(vals):
            bars = ax.bar(x + (j - 0.5) * 0.38, v, 0.38, label=lab, color=col)
            ax.bar_label(bars, fmt="%.2f", fontsize=7)
        ax.set_xticks(x, ["mAP50", "mAP50-95", "F1", "Recall\n@ngưỡng", "Recall\nbox nhỏ"], fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(f"CT – {'Validation' if split == 'val' else 'Test'}"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "so_sanh.png", dpi=130); plt.close(fig)
    L += ["**Hình 8.1.** Chỉ số chính (ngưỡng vận hành riêng của từng mô hình, chọn trên Val).", "",
          "![so sánh](figures/frcnn_vs_yolo11n/so_sanh.png)", ""]
    if summ:
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        ax.plot([float(r["epoch"]) for r in yh], [float(r["metrics/mAP50-95(B)"]) for r in yh], color="#1d5fd6", label="YOLO11n")
        ax.plot([h["epoch"] for h in summ["history"]], [h["mAP50-95"] for h in summ["history"]], "o-", color="#b45309", label="Faster R-CNN")
        ax.set_xlabel("epoch"); ax.set_ylabel("mAP50-95 trên Val"); ax.grid(alpha=.3); ax.legend()
        ax.set_title("mAP50-95 trên Val theo epoch")
        fig.tight_layout(); fig.savefig(FIG / "duong_huan_luyen.png", dpi=130); plt.close(fig)
        L += ["**Hình 8.2.** Đường huấn luyện (YOLO11n: `results.csv` của run đã khóa; Faster R-CNN: `results.csv` của run này). "
              "Một epoch = một lượt qua 34.873 ảnh Train cho cả hai mô hình.", "", "![đường huấn luyện](figures/frcnn_vs_yolo11n/duong_huan_luyen.png)", ""]
    intro, analysis = TEXT_DIR / "phan_mo_dau.md", TEXT_DIR / "phan_nhan_xet.md"
    if intro.exists():
        L = L[:4] + [intro.read_text(encoding="utf-8").rstrip(), ""] + L[4:]
    L += [analysis.read_text(encoding="utf-8").rstrip(), ""] if analysis.exists() else \
         ["## 9. Nhận xét", "", "_Viết sau khi có số liệu, dựa trên các bảng trên._", ""]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("->", rel(REPORT))
    return 0


def _roi71():
    """scripts/71_eval_roi_val.py as a module (same Val volumes, GT linking, track building and hit rule)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("roi71", ROOT / "scripts" / "71_eval_roi_val.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def volume_val(a) -> int:
    """Validation only (never Test): case-level comparison after Tumor Zoom, as shown on the website.
    Faster R-CNN scans every slice of the 210 usable CT Val volumes (kept boxes >= 0.05, cached); YOLO11n uses the
    cache of scripts/71_eval_roi_val.py (locked detector, conf 0.12, same volumes). Both go through the same merge ->
    link (configs/roi.yaml, no track filter = the website setting) and the same evaluation of 71_ (GT lesion hit when a
    track has IoU >= 0.3 with it on a shared slice; also 0.5). Faster R-CNN at its own threshold 0.28 and at 0.12."""
    from livercancer.config import load_paths
    from livercancer.data.slices import load_volume_uint8, slice_image
    from livercancer.data.sources import image_path, lesion_mask_path
    from livercancer.detection.frcnn_detector import FrcnnDetector
    from livercancer.roi.pipeline import load_roi_config
    r71 = _roi71()
    paths = load_paths()
    data = DATA["val"]
    vols = r71.val_volumes(data)
    keys = [(v["patient_id"], v["series"]) for v in vols]
    raw = OUT / "eval" / "roi_val_raw_dets_frcnn.csv"
    if a.rescan or not raw.exists():
        det = FrcnnDetector.from_config()
        print(f"scan Faster R-CNN: {len(vols)} CT Val volumes on {det.device}, boxes >= 0.05", flush=True)
        t0 = time.time()
        import csv
        tmp = raw.with_suffix(".part")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "series", "n_slices", "slice_k", "x0", "y0", "x1", "y1", "conf"])
            for i, v in enumerate(vols):
                vol = load_volume_uint8(image_path(paths, "ct", v["patient_id"], v["series"]))
                imgs = [slice_image(vol, k) for k in range(vol.shape[2])]
                dets = det.predict(imgs, conf=0.05)
                w.writerow([v["patient_id"], v["series"], len(imgs), -1, "", "", "", "", ""])
                for k, d in enumerate(dets):
                    for x in d:
                        w.writerow([v["patient_id"], v["series"], len(imgs), k, round(x.x0, 2), round(x.y0, 2), round(x.x1, 2), round(x.y1, 2), round(x.conf, 5)])
                if (i + 1) % 10 == 0:
                    el = time.time() - t0
                    print(f"  {i + 1}/{len(vols)}  {el / 60:.1f} min, ETA {el / (i + 1) * (len(vols) - i - 1) / 60:.1f} min", flush=True)
        tmp.replace(raw)
    params = load_roi_config("ct")
    labels = {v["patient_id"]: v["label"] for v in vols}
    gt, pos = r71.gt_by_volume(data)
    unknown = {k: r71.mask_slices(lesion_mask_path(paths, "ct", *k)) - pos.get(k, set()) for k in keys}

    def run(raw_csv: Path, thr: float) -> dict:
        dets, n = r71.load_scan(raw_csv)
        miss = [k for k in keys if k not in n]
        assert not miss, f"{len(miss)} volumes missing from {raw_csv}"
        dets = {k: {s: [d for d in ds if d.conf >= thr] for s, ds in v.items()} for k, v in dets.items()}
        pred, gtt = r71.build_tracks(keys, dets, gt, params)
        m, recs = r71.evaluate("ct", keys, pred, gtt, unknown, labels, 1, thr, 0.3)
        m5, _ = r71.evaluate("ct", keys, pred, gtt, unknown, labels, 1, thr, 0.5)
        cancer = [k for k in keys if labels.get(k[0]) != "CN"]
        cn = [k for k in keys if labels.get(k[0]) == "CN"]
        per = lambda ks: round(sum(1 for r in recs if r["key"] in set(ks)) / max(1, len(ks)), 3)  # noqa: E731
        fp_c = sum(1 for r in recs if r["status"] == "FP" and r["key"] in set(cancer))
        return {"threshold": thr, "hit_iou_0.3": m, "hit_iou_0.5": {k: m5[k] for k in ("lesion_recall", "small_lesion_recall", "volume_recall")},
                "regions_per_volume": round(len(recs) / len(keys), 3), "regions_per_cancer_volume": per(cancer),
                "regions_per_cn_volume": per(cn), "fp_regions_per_cancer_volume": round(fp_c / max(1, len(cancer)), 3),
                "n_volumes": len(keys), "n_cancer_volumes": len(cancer), "n_cn_volumes": len(cn)}

    yolo_raw = ROOT / "outputs/predictions/roi_val/ct/raw_dets.csv"
    thr_fr = json.loads((OUT / "eval" / "ct_val.json").read_text(encoding="utf-8"))["threshold_choice"]["conf"]
    out = {"split": "val", "note": "Validation only; same volumes, Tumor Zoom linking and hit rule as scripts/71_eval_roi_val.py; no track filter (website setting)",
           "roi_params": params, "yolo11n": run(yolo_raw, OP_CONF), "frcnn_own": run(raw, thr_fr), "frcnn_0.12": run(raw, OP_CONF),
           "frcnn_raw": rel(raw), "frcnn_raw_sha256": sha256(raw), "yolo11n_raw": rel(yolo_raw), "time": datetime.now().isoformat(timespec="seconds")}
    ref = json.loads((ROOT / "outputs/predictions/roi_val/ct/choice.json").read_text(encoding="utf-8"))["unfiltered"]
    same = all(out["yolo11n"]["hit_iou_0.3"][k] == ref[k] for k in ("lesion_recall", "volume_recall", "fp_tracks", "kept_tracks"))
    out["yolo11n_matches_71"] = same
    dump(OUT / "eval" / "ct_val_volume.json", out)
    for k in ("yolo11n", "frcnn_own", "frcnn_0.12"):
        m = out[k]["hit_iou_0.3"]
        print(f"{k:11s} thr {out[k]['threshold']}: volumes {m['volume_recall']} ({m['volumes_with_lesion']}), lesions {m['lesion_recall']}, "
              f"small {m['small_lesion_recall']}, FP/vol {m['fp_tracks_per_volume']}, regions/vol {out[k]['regions_per_volume']}, "
              f"CN vols with region {m.get('cn_volumes_with_track')}, CN patients {m.get('cn_patients_with_track')}", flush=True)
    print("YOLO11n identical to 71_ choice.json:", same)
    return 0 if same else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["smoke", "train", "check-evaluator", "eval", "bench", "volume-val", "notebook", "report"])
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--rescan", action="store_true", help="volume-val: re-run Faster R-CNN on all Val volumes")
    a = ap.parse_args()
    return {"smoke": smoke, "train": train_local, "check-evaluator": check_evaluator, "eval": evaluate, "bench": bench, "volume-val": volume_val,
            "notebook": notebook, "report": report}[a.stage](a)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
