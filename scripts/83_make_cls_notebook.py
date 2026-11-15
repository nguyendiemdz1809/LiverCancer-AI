"""Generate notebooks/05_train_classifier_colab.ipynb: one notebook, model chosen in the config cell
(resnet18 | swin_t | hybrid), CT or MRI. It runs the project code (livercancer.classification) shipped in
livercancer_code.zip, on the ROI dataset cls_<mod>.zip. Training starts only when the user runs the cells on Colab.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)}


def code(t: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# 05 · Phân loại tổn thương gan: CNN, Transformer và Hybrid (Google Colab)

Huấn luyện **một** mô hình mỗi lần chạy, chọn ở mục 2:
- `resnet18`: CNN baseline, dùng ảnh zoom **local**;
- `swin_t`: Transformer baseline, dùng ảnh zoom **context**;
- `hybrid`: ResNet18 (local) + Swin-T (context), hợp nhất đặc trưng (feature fusion) rồi phân loại.

Mỗi lần chạy chọn **CT** (3 lớp HCC / ICC / CHCC) hoặc **MRI** (7 lớp).
- Dữ liệu: ROI từ GT (Train), Val gồm ROI từ GT và ROI do YOLO dự đoán (pipeline thật). **Tập Test không có trong ZIP và không được dùng.**
- Early stopping theo **macro-F1 trên Val**; `best.pt`, `last.pt` (resume), lịch sử, ma trận nhầm lẫn được lưu trên Drive.
- MRI LLD-MMRI: **CC BY-NC 4.0**, chỉ dùng cho nghiên cứu; giữ ZIP trong thư mục Drive riêng tư.

**Chuẩn bị:** upload `cls_ct.zip` và/hoặc `cls_mri.zip`, cùng `livercancer_code.zip` và `SHA256SUMS_cls.txt`, vào `MyDrive/LiverCancer_AI/data/`.
Chọn *Runtime → Change runtime type → T4 GPU*, rồi chạy từ trên xuống. Nên đặt `SMOKE = True` để chạy thử 1 lần trước (vài phút).
**Nếu Colab bị ngắt:** chạy lại từ đầu, giữ nguyên cấu hình. Huấn luyện tự tiếp tục từ `last.pt` trên Drive.
"""),
    md("## 1. Kiểm tra GPU"),
    code("""
!nvidia-smi
import torch
assert torch.cuda.is_available(), "Không có GPU: Runtime → Change runtime type → T4 GPU"
GPU = torch.cuda.get_device_name(0)
print("torch", torch.__version__, "|", GPU, f"| {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
if 'T4' not in GPU:
    print('WARNING: không phải T4 -> thời gian/bộ nhớ khác kế hoạch; giảm BATCH nếu thiếu bộ nhớ')
"""),
    md("## 2. Mount Drive và cấu hình thí nghiệm"),
    code("""
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/LiverCancer_AI')
DATA_DRIVE = DRIVE_ROOT / 'data'

# ---- experiment -------------------------------------------------------------------------------
MODALITY   = 'ct'        # 'ct' | 'mri'
MODEL      = 'hybrid'    # 'resnet18' | 'swin_t' | 'hybrid'  (also: 'resnet50', 'hybrid_r50_swin')
ROI_SOURCE = 'gt'        # 'gt' (baseline) | 'gt+yolo' (needs YOLO Train ROIs in the ZIP)
EPOCHS, PATIENCE, BATCH, SEED = 40, 8, 32, 42   # BATCH 32 fits a T4 for hybrid with AMP; 16 / 8 if CUDA OOM (also auto-halved)
OVERRIDES  = {}          # any key of configs/cls.yaml `train`, e.g. {'fusion': 'gated', 'lr_backbone': 5e-5}
TAG        = ''          # short suffix for a variant, e.g. '_gated'
SMOKE      = False       # True: 2 epochs on a tiny subset, written to /content (not Drive) -> pipeline check only

RUN_NAME = f"{MODALITY}_{MODEL}_{ROI_SOURCE.replace('+', 'plus')}_s{SEED}{TAG}"
RUN_DIR  = Path('/content/smoke') / RUN_NAME if SMOKE else DRIVE_ROOT / 'runs' / 'cls' / RUN_NAME
ZIP_DATA, ZIP_CODE = DATA_DRIVE / f'cls_{MODALITY}.zip', DATA_DRIVE / 'livercancer_code.zip'
for p in (ZIP_DATA, ZIP_CODE, DATA_DRIVE / 'SHA256SUMS_cls.txt'):
    assert p.exists(), f"Chưa thấy {p} — hãy upload lên Drive"
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(RUN_NAME, '->', RUN_DIR)
"""),
    md("## 3. Kiểm tra SHA256 và giải nén vào `/content`"),
    code("""
import hashlib, shutil, zipfile, sys

def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

expected = dict(l.split()[::-1] for l in (DATA_DRIVE / 'SHA256SUMS_cls.txt').read_text().splitlines() if l.strip())
DATA_SHA = {}
for z in (ZIP_CODE, ZIP_DATA):
    local = Path('/content') / z.name
    if not local.exists():
        shutil.copy(z, local)
    DATA_SHA[z.name] = sha256(local)
    assert DATA_SHA[z.name] == expected.get(z.name), f"SHA256 mismatch: {z.name}"
    print('SHA256 OK', z.name, DATA_SHA[z.name][:16])
CODE_DIR, DATA_ROOT = Path('/content/lc_code'), Path(f'/content/cls_{MODALITY}')
if not CODE_DIR.exists():
    zipfile.ZipFile(Path('/content') / ZIP_CODE.name).extractall(CODE_DIR)
if not DATA_ROOT.exists():
    zipfile.ZipFile(Path('/content') / ZIP_DATA.name).extractall('/content')
sys.path.insert(0, str(CODE_DIR / 'src'))
print(sorted(p.name for p in DATA_ROOT.iterdir()))
"""),
    md("## 4. Kiểm tra dữ liệu: lớp, số ROI, số bệnh nhân, độc lập Train/Val, không có Test"),
    code("""
import csv, json
from collections import Counter
rows = list(csv.DictReader(open(DATA_ROOT / 'manifest.csv', encoding='utf-8')))
assert {r['split'] for r in rows} <= {'train', 'val'}, 'Test must not be in the dataset'
pats = {}
for r in rows:
    pats.setdefault(r['patient_id'], set()).add(r['split'])
assert all(len(s) == 1 for s in pats.values()), 'a patient appears in Train and Val'
print(json.load(open(DATA_ROOT / 'class_map.json', encoding='utf-8'))['classes'])
print('ROI theo (split, nguồn, lớp):')
for k, v in sorted(Counter((r['split'], r['source'], r['label']) for r in rows).items()):
    print('  ', k, v)
print('Bệnh nhân theo (split, lớp):', sorted(Counter((r['split'], r['label']) for r in {r['patient_id']: r for r in rows}.values()).items()))
"""),
    md("""
## 5. Huấn luyện (transfer learning, AMP, class weighting, early stopping theo macro-F1 Val)

- Trọng số ImageNet của torchvision được tải tự động (lần đầu).
- `last.pt` được ghi sau mỗi epoch, nên khi chạy lại cell này với cùng `RUN_NAME`, huấn luyện **tiếp tục** từ epoch đã lưu.
  Run đã xong (có `summary.json`) thì cell sẽ bỏ qua.
- CUDA hết bộ nhớ: batch tự giảm một nửa, gradient accumulation tăng gấp đôi (batch hiệu dụng giữ nguyên).
"""),
    code("""
import yaml
from livercancer.classification.train import train
cfg = dict(yaml.safe_load(open(CODE_DIR / 'configs' / 'cls.yaml', encoding='utf-8'))['train'])
cfg.update(modality=MODALITY, model=MODEL, roi_source=ROI_SOURCE, epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH, seed=SEED,
           data_root=str(DATA_ROOT), out_dir=str(RUN_DIR), num_workers=2)
cfg.update(OVERRIDES)                     # user overrides win (e.g. {'fusion': 'gated', 'num_workers': 4})
if SMOKE:
    cfg.update(epochs=2, patience=2, max_train_rois=256, max_val_rois=128, resume=False)
if (RUN_DIR / 'summary.json').exists() and not SMOKE:
    print('Run đã hoàn thành -> bỏ qua huấn luyện:', json.load(open(RUN_DIR / 'summary.json')))
else:
    summary = train(cfg)
    print(summary)
"""),
    md("## 6. Đường cong huấn luyện, overfitting và ma trận nhầm lẫn (Val)"),
    code("""
from IPython.display import Image, display
import pandas as pd
hist = pd.read_csv(RUN_DIR / 'history.csv')
display(hist.tail(10))
print('epoch có cảnh báo overfitting (val loss tăng 3 epoch liên tiếp, train loss giảm):', hist.loc[hist.overfit_warning == 1, 'epoch'].tolist())
for name in ('curves.png', 'cm_val_roi_best.png', 'cm_val_patient_best.png'):
    if (RUN_DIR / name).exists():
        display(Image(filename=str(RUN_DIR / name), width=700))
"""),
    md("""
## 7. Đánh giá trên Val: mức ROI và mức bệnh nhân, ROI từ GT và từ YOLO (pipeline)

Cách gộp ROI thành dự đoán mức bệnh nhân được **chọn trên Val** (ROI YOLO), không dùng Test.
"""),
    code("""
from livercancer.classification.evaluate import evaluate_val
res = evaluate_val(RUN_DIR, DATA_ROOT, workers=2)
def show(tag, m):
    if m:
        for lvl in ('roi', 'patient'):
            if lvl in m:
                x = m[lvl]
                print(f"{tag:18s} {lvl:8s} n={x['n']:5d} Acc {x['accuracy']:.3f} BalAcc {x['balanced_accuracy']:.3f} "
                      f"macro-F1 {x['macro_f1']:.3f} Sens {x['macro_recall']:.3f} Spec {x['macro_specificity']:.3f} "
                      f"AUC {'—' if x['roc_auc_macro_ovr'] is None else round(x['roc_auc_macro_ovr'], 3)}")
print('Gộp mức bệnh nhân (chọn trên Val):', res['aggregation'])
show('ROI GT', res['gt_roi']); show('ROI YOLO (khớp GT)', res['yolo_roi_matched']); show('ROI YOLO (tất cả)', res['yolo_roi_all'])
print('Pipeline mức bệnh nhân:', res.get('pipeline_patient'))
for p in sorted((RUN_DIR / 'eval_val').glob('cm_*.png')):
    display(Image(filename=str(p), width=600))
"""),
    md("## 8. Sổ thí nghiệm (Drive) và gói kết quả để phân tích ở máy local"),
    code("""
import datetime
if not SMOKE:
    best = json.load(open(RUN_DIR / 'metrics_val_best.json'))
    reg = DRIVE_ROOT / 'runs' / 'cls' / 'experiments.csv'
    row = {'run': RUN_NAME, 'modality': MODALITY, 'model': MODEL, 'roi_source': ROI_SOURCE, 'seed': SEED, 'batch': BATCH,
           'best_epoch': best['epoch'], 'val_roi_acc': round(best['roi']['accuracy'], 4), 'val_roi_macro_f1': round(best['roi']['macro_f1'], 4),
           'val_patient_macro_f1_yolo': round(res['yolo_roi_all']['patient']['macro_f1'], 4) if res.get('yolo_roi_all') else '',
           'aggregation': res['aggregation']['chosen'], 'data_sha256': DATA_SHA[ZIP_DATA.name], 'gpu': GPU,
           'overrides': json.dumps(OVERRIDES), 'time': datetime.datetime.now().isoformat(timespec='seconds')}
    old = list(csv.DictReader(open(reg))) if reg.exists() else []
    old = [r for r in old if r['run'] != RUN_NAME] + [row]
    with open(reg, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows({k: r.get(k, '') for k in row} for r in old)
    bundle = DRIVE_ROOT / 'runs' / 'cls' / f'{RUN_NAME}_bundle.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in RUN_DIR.rglob('*'):
            if p.is_file() and p.name != 'last.pt':
                z.write(p, f'{RUN_NAME}/{p.relative_to(RUN_DIR).as_posix()}')
    print(reg.read_text()); print(bundle, f"{bundle.stat().st_size / 1e6:.1f} MB")
"""),
    md("""
## Ghi chú
- Thứ tự gợi ý: `SMOKE=True` (kiểm tra) → `resnet18` → `swin_t` → `hybrid` cho CT, rồi lặp lại cho MRI. So sánh **chỉ trên Val**.
- Muốn thử biến thể (ví dụ `fusion='gated'`), hãy đổi `OVERRIDES` **và** `TAG` để không ghi đè run cũ.
- **Không** đánh giá Test trong notebook này. Test chỉ được đánh giá một lần cho toàn bộ pipeline, sau khi mọi lựa chọn đã chốt trên Val.
"""),
]


def main() -> None:
    nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cls-{i:02d}"
    p = ROOT / "notebooks" / "05_train_classifier_colab.ipynb"
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("written", p)


if __name__ == "__main__":
    main()
