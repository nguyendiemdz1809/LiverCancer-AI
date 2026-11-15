"""Generate the Colab notebooks.

  notebooks/03_train_yolo_{ct,mri}_colab.ipynb       pilot: YOLO11n pretrained, 5 epochs, Ultralytics defaults
  notebooks/04_train_yolo_{ct,mri}_full_colab.ipynb  official: explicit hyper-parameters, early stopping, registry

Both: GPU check, Drive mount, SHA256, unzip to /content, label check, checkpoints on Drive, resume, Val metrics at
conf=0.001 (mAP) and conf=0.25 (inference threshold, confusion matrix), analysis bundle for local analysis.
The Test split is never present or evaluated.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS = "8.4.182"  # same version as the local environment used for QA and analysis

FULL_DEFAULTS = {  # official-training baseline (docs/PHASE3_TRAINING_PLAN.md); change per experiment, keep a new RUN tag
    "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005, "cos_lr": True,
    "warmup_epochs": 3.0, "close_mosaic": 10, "mosaic": 1.0, "mixup": 0.0, "fliplr": 0.5, "flipud": 0.0,
    "degrees": 0.0, "translate": 0.1, "scale": 0.5, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
}
FULL_EPOCHS = 50  # E0/E1 budget agreed after MRI-E0 (best epoch 24, early stop at 44); patience 20 stays
# measured on MRI-E0 (T4, yolo11n, 512, batch 32): ~316 s/epoch for 28.6k train + 7.5k val images
SEC_PER_IMAGE_T4 = 316 / (28_595 + 7_491)
FULL_IMAGES = {"ct": 49_500, "mri": 28_595 + 7_491}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def config_cell(mod: str, full: bool) -> str:
    common = f"""
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
DRIVE_ROOT  = Path('/content/drive/MyDrive/LiverCancer_AI')
DATA_DRIVE  = DRIVE_ROOT / 'data'                       # yolo_{mod}_*.zip + SHA256SUMS.txt uploaded here
SHA_FILES   = sorted(DATA_DRIVE.glob('SHA256SUMS*.txt'))      # checksums of pilot/full ZIPs
RUN_PROJECT = DRIVE_ROOT / 'runs' / 'yolo_{mod}'        # checkpoints / plots / metrics are written here (persist)
DATA_DIR    = Path('/content/yolo_{mod}')               # local SSD of the Colab VM (fast I/O)
"""
    if not full:
        return common + f"""
ZIP_NAME = 'yolo_{mod}_pilot.zip'
RUN_NAME = 'pilot_yolo11n_e5'
MODEL, EPOCHS, IMGSZ, BATCH, SEED = 'yolo11n.pt', 5, 512, 16, 42   # pretrained COCO weights, T4-friendly batch
TRAIN_KW = {{}}                                          # pilot = Ultralytics defaults
RESUME = False   # set True after a disconnect to continue from RUN_PROJECT/RUN_NAME/weights/last.pt

DRIVE_ZIP = DATA_DRIVE / ZIP_NAME
assert DRIVE_ZIP.exists(), f"Chưa thấy {{DRIVE_ZIP}} — hãy upload ZIP lên Drive"
RUN_PROJECT.mkdir(parents=True, exist_ok=True)
print(DRIVE_ZIP, f"{{DRIVE_ZIP.stat().st_size / 1e9:.2f}} GB")
"""
    hp = ",\n    ".join(f"{k!r}: {v!r}" for k, v in FULL_DEFAULTS.items())
    return common + f"""
# ---- experiment definition: one row per run in docs/PHASE3_TRAINING_PLAN.md -------------------------------
ZIP_NAME  = '{"yolo_ct_full.zip" if mod == "ct" else "yolo_mri_pilot.zip"}'   # {"CT: full export (40_pack_pilot.py --set full)" if mod == "ct" else "MRI: the pilot ZIP already holds all eligible slices (plan §1)"}
MODEL     = 'yolo11n.pt'                # 'yolo11n.pt' | 'yolo11s.pt' (pretrained COCO)
EPOCHS, PATIENCE, IMGSZ, BATCH, SEED = {FULL_EPOCHS}, 20, 512, 32, 42   # BATCH=32 fits a T4 at 512 for n/s; use 16 if OOM
TRAIN_KW = {{
    {hp},
}}
TAG = 'baseline'                        # short name of what changes in this experiment (e.g. 'fliplr0', 'img640')
RUN_NAME = f"{{Path(MODEL).stem}}_e{{EPOCHS}}_img{{IMGSZ}}_{{TAG}}"
RESUME = False   # set True after a disconnect to continue from RUN_PROJECT/RUN_NAME/weights/last.pt

DRIVE_ZIP = DATA_DRIVE / ZIP_NAME
assert DRIVE_ZIP.exists(), f"Chưa thấy {{DRIVE_ZIP}} — hãy upload ZIP lên Drive"
RUN_PROJECT.mkdir(parents=True, exist_ok=True)
assert RESUME or not (RUN_PROJECT / RUN_NAME / 'weights' / 'last.pt').exists(), \\
    f"{{RUN_NAME}} already exists on Drive: set RESUME=True to continue it, or change TAG for a new experiment"
print(RUN_NAME, '|', DRIVE_ZIP, f"{{DRIVE_ZIP.stat().st_size / 1e9:.2f}} GB")
"""


def notebook(mod: str, full: bool) -> dict:
    M = mod.upper()
    license_note = ("CT PLC-CECT: CC BY 4.0." if mod == "ct" else
                    "MRI LLD-MMRI: **CC BY-NC 4.0**, chỉ dùng cho nghiên cứu phi thương mại; không chia sẻ lại dữ liệu.")
    if full:
        min_ep = FULL_IMAGES[mod] * SEC_PER_IMAGE_T4 / 60
        intro = f"""
# 04 · YOLO {M}: train chính thức (Google Colab)

Huấn luyện chính thức theo `docs/PHASE3_TRAINING_PLAN.md`: tối đa {FULL_EPOCHS} epoch, **early stopping** (patience 20),
hyper-parameter được ghi tường minh, kết quả được ghi vào `runs/yolo_{mod}/experiments.csv` trên Drive.
- Thời gian ước tính trên T4 (yolo11n, ngoại suy từ MRI-E0 đo được 316 s/epoch): khoảng **{min_ep:.0f} phút/epoch**
  ({FULL_IMAGES[mod]:,} ảnh train + val), tức ≤ khoảng {min_ep * FULL_EPOCHS / 60:.0f} giờ cho {FULL_EPOCHS} epoch. Early stopping có thể dừng sớm hơn.
  Một phiên Colab có thể bị ngắt giữa chừng, khi đó làm theo mục *Resume*.
- Mỗi thí nghiệm có một `TAG` riêng. **Không ghi đè lên run cũ.**
- Chỉ đánh giá trên **Val**. Tập Test không có trong ZIP và chỉ được đánh giá **một lần** cho cấu hình cuối, ở bước riêng.
- {license_note}
"""
    else:
        intro = f"""
# 03 · YOLO11n pilot: {M} (Google Colab)

Đồ án: *Ứng dụng mô hình học sâu trong phát hiện và phân loại ung thư gan từ ảnh CT/MRI*.

Notebook này **chạy thử (pilot)** YOLO11n pretrained trong **5 epoch** trên `yolo_{mod}_pilot.zip` (chỉ Train + Val).
- Detection 1 lớp: `0 = lesion`. Chẩn đoán bệnh **không** phải lớp YOLO.
- **Tập Test không có trong ZIP và không được dùng ở đây.**
- {license_note} Giữ ZIP trong thư mục Drive riêng tư.
"""
    intro += f"""
**Chuẩn bị:** upload `{'yolo_' + mod + '_pilot.zip' if not full else ('yolo_ct_full.zip' if mod == 'ct' else 'yolo_mri_pilot.zip')}` và `{'SHA256SUMS_full.txt' if full and mod == 'ct' else 'SHA256SUMS.txt'}` vào `MyDrive/LiverCancer_AI/data/`.
Chọn *Runtime → Change runtime type → GPU (T4)* rồi chạy các cell từ trên xuống.
**Nếu Colab bị ngắt:** xem mục *Resume* ở cuối notebook (`last.pt` được lưu trên Drive sau mỗi epoch).
"""
    train_call = ("model.train(data=str(DATA_YAML), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0, amp=True,\n"
                  "                seed=SEED, deterministic=True, workers=2, project=str(RUN_PROJECT), name=RUN_NAME,\n"
                  "                exist_ok=True, plots=True, val=True, " +
                  ("patience=PATIENCE, save_period=-1, **TRAIN_KW)" if full else "save_period=1, **TRAIN_KW)"))
    cells = [
        md(intro),
        md("## 1. Kiểm tra GPU CUDA"),
        code("""
!nvidia-smi
import torch
assert torch.cuda.is_available(), "Không có GPU: Runtime → Change runtime type → GPU"
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0),
      f"| {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
""" + ("""if 'T4' not in torch.cuda.get_device_name(0):   # batch 32 and the time estimate assume a T4; the GPU is logged
    print('WARNING: GPU is not a T4 -> epoch time differs from the plan; reduce BATCH to 16 if CUDA runs out of memory')
""" if full else "")),
        md("## 2. Mount Google Drive và cấu hình"),
        code(config_cell(mod, full)),
        md("## 3. Kiểm tra SHA256 và giải nén vào `/content`"),
        code("""
import hashlib, shutil, zipfile

def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

local_zip = Path('/content') / ZIP_NAME
ZIP_SHA = None
if not DATA_DIR.exists():
    shutil.copy(DRIVE_ZIP, local_zip)                 # copy once: reading many small files from Drive is slow
    ZIP_SHA = sha256(local_zip)
    expected = {}                                     # SHA256SUMS.txt (pilot) and SHA256SUMS_full.txt (full) may coexist
    for f in sorted(DATA_DRIVE.glob('SHA256SUMS*.txt')):
        for l in f.read_text().splitlines():
            if l.strip():
                h, n = l.split()
                expected[n] = h
    if ZIP_NAME in expected:
        assert ZIP_SHA == expected[ZIP_NAME], f"SHA256 mismatch: {ZIP_SHA}"
        print("SHA256 OK", ZIP_SHA)
    else:
        print(f"{ZIP_NAME} not listed in SHA256SUMS*.txt on Drive — checksum not verified:", ZIP_SHA)
    with zipfile.ZipFile(local_zip) as z:
        z.extractall('/content')
    local_zip.unlink()
print(sorted(p.name for p in DATA_DIR.iterdir()))
"""),
        md("## 4. Cài Ultralytics"),
        code(f"""
!pip -q install ultralytics=={ULTRALYTICS}
import ultralytics
ultralytics.checks()
"""),
        md("## 5. Kiểm tra `data.yaml` và nhãn (100% file nhãn)"),
        code("""
import csv, math, yaml
from collections import Counter

cfg = yaml.safe_load((DATA_DIR / 'data.yaml').read_text())
cfg['path'] = str(DATA_DIR)                          # point the dataset to the extracted location
DATA_YAML = DATA_DIR / 'data_colab.yaml'
DATA_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(DATA_YAML.read_text())
assert cfg['names'] == {0: 'lesion'} and 'test' not in cfg, "dataset must be single-class and without a test split"
assert not (DATA_DIR / 'images' / 'test').exists()

bad, n_box, stats = [], 0, {}
for split in ('train', 'val'):
    imgs = {p.stem for p in (DATA_DIR / 'images' / split).glob('*.png')}
    labs = {p.stem for p in (DATA_DIR / 'labels' / split).glob('*.txt')}
    assert imgs == labs, f"{split}: {len(imgs ^ labs)} image/label files without a pair"
    pos = 0
    for p in (DATA_DIR / 'labels' / split).glob('*.txt'):
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        pos += bool(lines)
        for l in lines:
            c, *v = l.split()
            ok = c == '0' and len(v) == 4 and all(math.isfinite(float(x)) for x in v)
            if ok:
                cx, cy, w, h = map(float, v)
                ok = w > 0 and h > 0 and cx - w / 2 >= -1e-6 and cx + w / 2 <= 1 + 1e-6 and cy - h / 2 >= -1e-6 and cy + h / 2 <= 1 + 1e-6
            n_box += 1
            if not ok:
                bad.append(f"{p.name}: {l}")
    stats[split] = {'images': len(imgs), 'positive': pos, 'negative': len(imgs) - pos}
assert not bad, bad[:10]
print('labels OK:', n_box, 'boxes', stats)

with open(DATA_DIR / 'slices.csv') as f:
    rows = list(csv.DictReader(f))
pat = {s: len({r['patient_id'] for r in rows if r['split'] == s}) for s in ('train', 'val')}
overlap = {r['patient_id'] for r in rows if r['split'] == 'train'} & {r['patient_id'] for r in rows if r['split'] == 'val'}
assert not overlap, f"patients in both train and val: {sorted(overlap)[:5]}"
print('patients per split:', pat)
print(Counter((r['split'], r['patient_label'], r['kind']) for r in rows))
"""),
        md("## 6. Huấn luyện" + (" (early stopping, hyper-parameter tường minh)" if full else " thử YOLO11n (5 epoch, GPU, AMP)") + """

- `amp=True`: Ultralytics tự kiểm tra AMP và tắt nếu GPU không tương thích.
- `project` nằm trên Drive, nên `last.pt` và `best.pt` được lưu sau mỗi epoch.
- Sau mỗi lần lưu, `last.pt` được chép thành `last_backup.pt`. Nếu Colab bị ngắt đúng lúc đang ghi `last.pt` lên Drive
  (file hỏng), resume sẽ dùng bản sao này (mất tối đa 1 epoch).
- `best.pt` được chọn theo fitness = **mAP50-95 trên Val**. **Không dùng Test.**
"""),
        code(f"""
import shutil
from ultralytics import YOLO

last = RUN_PROJECT / RUN_NAME / 'weights' / 'last.pt'
backup = last.with_name('last_backup.pt')

def resumable(p):
    # Ultralytics 8.4 strips the optimizer from last.pt when a run finishes; resuming such a file silently
    # starts a NEW training on the default dataset (coco8). Only resume unfinished checkpoints.
    ck = torch.load(str(p), map_location='cpu', weights_only=False)
    return ck.get('epoch', -1) >= 0 and ck.get('optimizer') is not None, ck.get('epoch', -1)

def backup_last(trainer):
    # on_model_save fires right after last.pt is written: keep a copy so a disconnect during the next write
    # (truncated last.pt on Drive) still leaves a loadable checkpoint
    if trainer.last.exists():
        shutil.copy2(trainer.last, trainer.last.with_name('last_backup.pt'))

def resume_source():
    # last.pt first; the backup only when last.pt cannot be loaded (never when the run is simply finished)
    for p in (last, backup):
        if p.exists():
            try:
                return (p, *resumable(p))
            except Exception as e:
                print(f'{{p.name}} cannot be loaded ({{type(e).__name__}}: {{e}}) -> trying the next checkpoint')
    return None, False, -1

if RESUME and (last.exists() or backup.exists()):
    src, ok, ep = resume_source()
    if ok:
        print(f'Resuming from {{src}} (last saved epoch {{ep + 1}})')
        model = YOLO(str(src))
        model.add_callback('on_model_save', backup_last)
        model.train(resume=True, data=str(DATA_YAML))
    elif src is not None:
        print(f'{{RUN_NAME}} is already finished (optimizer stripped) -> nothing to resume; go to section 7.')
    else:
        raise RuntimeError(f'no loadable checkpoint in {{last.parent}}')
elif last.exists():                                   # never overwrite a finished/partial run by accident
    print(f'{{RUN_NAME}} already exists on Drive -> training skipped. Set RESUME=True to continue it, '
          'or use a new RUN_NAME/TAG for a new experiment. Sections 7-9 still work on the existing run.')
else:
    model = YOLO(MODEL)
    model.add_callback('on_model_save', backup_last)
    {train_call}
"""),
        md("""
## 7. Đánh giá `best.pt` trên Val: Precision, Recall, mAP50, mAP50-95

- `conf=0.001` (mặc định của Ultralytics): dùng để tính **mAP**. Confusion matrix ở ngưỡng này đếm cả các box có
  confidence rất thấp, nên hàng *background* (FP) rất lớn. Đây là hiện tượng bình thường, không phải lỗi dữ liệu.
- `conf=0.25` (ngưỡng mặc định khi `predict`): confusion matrix và P/R gần với lúc **suy luận thực tế** hơn.
  Ngưỡng vận hành cuối cùng sẽ được chọn trên Val bằng `scripts/50_analyze_yolo_run.py`.
"""),
        code("""
import json, datetime

run = RUN_PROJECT / RUN_NAME
best = run / 'weights' / 'best.pt'
assert best.exists(), 'best.pt not found — training did not finish'
res = {}
for conf in (0.001, 0.25):
    m = YOLO(str(best)).val(data=str(DATA_YAML), split='val', conf=conf, imgsz=IMGSZ, batch=BATCH, device=0, workers=2,
                            project=str(RUN_PROJECT), name=f'{RUN_NAME}_val_conf{conf}', exist_ok=True, plots=True)
    cm = m.confusion_matrix.matrix
    res[f'conf{conf}'] = {'precision': float(m.box.mp), 'recall': float(m.box.mr), 'mAP50': float(m.box.map50),
                          'mAP50-95': float(m.box.map), 'cm_TP': int(cm[0, 0]), 'cm_FP_background': int(cm[0, 1]),
                          'cm_FN_background': int(cm[1, 0])}
print(json.dumps(res, indent=2))

import csv as _csv
with open(run / 'results.csv') as f:
    hist = [{k.strip(): v.strip() for k, v in r.items()} for r in _csv.DictReader(f)]
best_ep = max(hist, key=lambda r: float(r['metrics/mAP50-95(B)']))
record = {'run': RUN_NAME, 'metrics_val': res['conf0.001'], 'metrics_val_conf0.25': res['conf0.25'],
          'best_epoch': int(float(best_ep['epoch'])), 'epochs_run': len(hist), 'weights': str(best), 'model': MODEL,
          'epochs': EPOCHS, 'patience': globals().get('PATIENCE'), 'imgsz': IMGSZ, 'batch': BATCH, 'seed': SEED,
          'train_kw': TRAIN_KW,
          'dataset': ZIP_NAME, 'dataset_sha256': ZIP_SHA, 'ultralytics': ultralytics.__version__,
          'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
          'time': datetime.datetime.now().isoformat(timespec='seconds'), 'split': 'val'}
(run / 'metrics_val.json').write_text(json.dumps(record, indent=2))
print('saved', run / 'metrics_val.json')
"""),
        md("## 8. Biểu đồ và checkpoint đã lưu trên Drive"),
        code("""
from IPython.display import Image, display
for d, name in ((run, 'results.png'), (RUN_PROJECT / f'{RUN_NAME}_val_conf0.001', 'BoxPR_curve.png'),
                (RUN_PROJECT / f'{RUN_NAME}_val_conf0.001', 'confusion_matrix.png'),
                (RUN_PROJECT / f'{RUN_NAME}_val_conf0.25', 'confusion_matrix.png')):
    p = d / name
    if not p.exists() and name == 'BoxPR_curve.png':
        p = d / 'PR_curve.png'
    if p.exists():
        print(p); display(Image(filename=str(p), width=650))
for p in sorted((run / 'weights').glob('*.pt')):
    print(p, f"{p.stat().st_size / 1e6:.1f} MB")
assert (run / 'weights' / 'best.pt').exists() and (run / 'weights' / 'last.pt').exists()
"""),
        md("""
## 9. Gói kết quả để phân tích ở máy local

Tạo `RUN_NAME_bundle.zip` (vài MB) trên Drive, gồm `results.csv`, `args.yaml`, `weights/best.pt`, `metrics_val.json`
và các biểu đồ. Tải file này về, giải nén vào `outputs/runs/yolo_<mod>/` rồi chạy `scripts/50_analyze_yolo_run.py`.
"""),
        code("""
bundle = RUN_PROJECT / f'{RUN_NAME}_bundle.zip'
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in [run / 'results.csv', run / 'args.yaml', run / 'metrics_val.json', run / 'weights' / 'best.pt',
              *run.glob('*.png'), *run.glob('*.jpg')]:
        if p.exists():
            z.write(p, f'{RUN_NAME}/{p.relative_to(run).as_posix()}')
    for c in ('0.001', '0.25'):
        for p in (RUN_PROJECT / f'{RUN_NAME}_val_conf{c}').glob('*.png'):
            z.write(p, f'{RUN_NAME}/val_conf{c}/{p.name}')
print(bundle, f"{bundle.stat().st_size / 1e6:.1f} MB")
"""),
    ]
    if full:
        cells += [
            md("## 10. Ghi vào sổ thí nghiệm `experiments.csv` (Drive)"),
            code("""
reg = RUN_PROJECT / 'experiments.csv'
row = {'run': RUN_NAME, 'model': MODEL, 'tag': TAG, 'imgsz': IMGSZ, 'batch': BATCH, 'epochs_max': EPOCHS, 'patience': PATIENCE,
       'epochs_run': record['epochs_run'], 'best_epoch': record['best_epoch'], 'dataset': ZIP_NAME,
       'dataset_sha256': ZIP_SHA, **{f'val_{k}': round(v, 4) for k, v in record['metrics_val'].items()
                                     if k in ('precision', 'recall', 'mAP50', 'mAP50-95')},
       'train_kw': json.dumps(TRAIN_KW, sort_keys=True), 'time': record['time']}
rows = []
if reg.exists():                                      # one row per run: re-running sections 7-10 updates it in place
    with open(reg, newline='') as f:
        rows = list(_csv.DictReader(f))
old = next((r for r in rows if r['run'] == RUN_NAME), None)
if old and not row['dataset_sha256']:
    row['dataset_sha256'] = old.get('dataset_sha256', '')
rows = [r for r in rows if r['run'] != RUN_NAME] + [row]
with open(reg, 'w', newline='') as f:
    w = _csv.DictWriter(f, fieldnames=list(row))
    w.writeheader()
    w.writerows({k: r.get(k, '') for k in row} for r in rows)
print(reg.read_text())
"""),
        ]
    cells.append(md(f"""
## Resume khi Colab bị ngắt

1. *Runtime → Change runtime type → GPU*, rồi chạy lại các cell **1 → 5** (GPU, Drive + cấu hình, giải nén, cài Ultralytics, kiểm tra nhãn).
2. Ở cell cấu hình (mục 2), **giữ nguyên** `RUN_NAME` / `TAG`, đặt `RESUME = True`, rồi chạy lại cell đó.
3. Chạy cell huấn luyện (mục 6). Ultralytics đọc `{{RUN_PROJECT}}/{{RUN_NAME}}/weights/last.pt` trên Drive và
   **chạy tiếp từ epoch đã lưu**, giữ nguyên các tham số cũ. Nếu `last.pt` bị hỏng do ngắt giữa lúc ghi, cell tự dùng
   `last_backup.pt`. Run đã xong (optimizer bị strip) thì **không** resume (tránh lỗi âm thầm train lại trên coco8).
4. Chạy tiếp mục 7 trở đi (đánh giá Val, biểu đồ, bundle{', sổ thí nghiệm' if full else ''}).

Ghi chú:
- Muốn chạy một thí nghiệm mới, hãy đổi {'`TAG`' if full else '`RUN_NAME`'}; đừng ghi đè lên run cũ.
- {'Không đánh giá trên Test cho đến khi cấu hình cuối đã được chốt trên Val.' if full else 'Đây là **pilot 5 epoch** để kiểm tra pipeline, không phải kết quả cuối. Không đánh giá trên Test.'}
"""))
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    for mod in ("ct", "mri"):
        for full, name in ((False, f"03_train_yolo_{mod}_colab.ipynb"), (True, f"04_train_yolo_{mod}_full_colab.ipynb")):
            nb = notebook(mod, full)
            for i, c in enumerate(nb["cells"]):
                c["id"] = f"{mod}-{'full' if full else 'pilot'}-{i:02d}"
            p = ROOT / "notebooks" / name
            p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print("written", p)


if __name__ == "__main__":
    main()
