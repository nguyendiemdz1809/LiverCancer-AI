"""Sinh notebooks/06_tai_lap_huan_luyen_danh_gia_CT.ipynb: notebook Colab để NGƯỜI KHÁC (tài khoản Gmail bất kỳ) tự chạy
lại quy trình huấn luyện và đánh giá của đồ án trên CT, không cần Drive của tác giả.

- Mã nguồn: git clone từ GitHub công khai (REPO_URL).
- Dữ liệu đã tiền xử lý (PLC-CECT, CC BY 4.0) và trọng số đã huấn luyện: tải từ GitHub Releases của repo (tag du-lieu-ct-v1),
  kiểm tra SHA256 trước khi dùng. MRI (LLD-MMRI, CC BY-NC 4.0, cấm phân phối lại) KHÔNG được đưa vào.
- Ba phần: (A) huấn luyện + đánh giá YOLO11n, (B) huấn luyện + đánh giá bộ phân loại Hybrid, (C) đánh giá lại trọng số
  đã công bố để so với số liệu trong báo cáo. Chế độ 'nhanh' để kiểm tra quy trình, 'day_du' = đúng cấu hình báo cáo.
- Chỉ đánh giá trên Validation. Tập Test không có trong dữ liệu chia sẻ (đã chạy một lần với cấu hình khóa).

Mọi ô code là Python thuần (không dùng lệnh '!'), nên có thể chạy thử trên máy tác giả:
  python scripts/101_make_reproduce_notebook.py --py-out <file.py>
  LCAI_LOCAL_ZIPS=<thư mục có 3 ZIP> LCAI_LOCAL_REPO=<thư mục code> LCAI_WORK=<thư mục tạm> python <file.py>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "06_tai_lap_huan_luyen_danh_gia_CT.ipynb"

REPO_URL = "https://github.com/nguyendiemdz1809/LiverCancer-AI.git"
RELEASE_URL = "https://github.com/nguyendiemdz1809/LiverCancer-AI/releases/download/du-lieu-ct-v1/"
FILES = {  # tên tệp -> (các phần trên GitHub Releases, SHA256 của tệp ghép). GitHub giới hạn 2 GB mỗi tệp nên ZIP YOLO bị cắt 3 phần.
    "yolo_ct_full.zip": (["yolo_ct_full.zip.part1", "yolo_ct_full.zip.part2", "yolo_ct_full.zip.part3"],
                         "0a7ed974856f5634b2860b692b83fa717e820031d8974862607a78a5e281cfdc"),
    "cls_ct.zip": (["cls_ct.zip"], "2f2e52bac01186942ea688b42318a19c2097a3dc598e1a48a441c46e3bfb81d3"),
    "trong_so_CT_LiverCancerAI.zip": (["trong_so_CT_LiverCancerAI.zip"],
                                      "8d51f9f1b27d2d4ac8849c3d12e9af75caa42e2ffde5ccd4fe8cbcee02f4b820"),
}


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)}


def code(t: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(keepends=True)}


def files_literal() -> str:
    rows = [f"    {name!r}: {{'parts': {parts!r}, 'sha256': {sha!r}}}," for name, (parts, sha) in FILES.items()]
    return f"RELEASE_URL = {RELEASE_URL!r}\nFILES = {{\n" + "\n".join(rows) + "\n}"


CELLS = [
    md("""
# 06 · Tái lập huấn luyện và đánh giá mô hình CT (Google Colab)

**Đồ án:** Ứng dụng mô hình học sâu trong phát hiện và phân loại ung thư gan từ ảnh CT/MRI (LiverCancer AI).

Notebook này cho phép **bất kỳ ai** (tài khoản Gmail nào cũng được) tự chạy lại quy trình huấn luyện và đánh giá của đồ án
trên ảnh CT. Không cần quyền truy cập Google Drive của tác giả: mã nguồn được tải từ GitHub, dữ liệu và trọng số được tải
từ mục **Releases** của repo GitHub và được kiểm tra SHA256 trước khi dùng.

| Phần | Nội dung | Kết quả trong báo cáo (Validation) |
|---|---|---|
| A | Huấn luyện **YOLO11n** phát hiện tổn thương (1 lớp «lesion»), đánh giá trên Val | P 0,688 · R 0,481 · mAP50 0,551 · mAP50-95 0,282 |
| B | Huấn luyện bộ phân loại lai **Hybrid ResNet-18 + Swin-T** (HCC / ICC / CHCC), đánh giá trên Val | ROI: accuracy 0,862 · macro-F1 0,863 · AUC 0,942 |
| C | Đánh giá lại **trọng số đã huấn luyện của đồ án** trên cùng tập Val | Phải ra đúng các số ở hai dòng trên |

**Cách chạy:** *Runtime → Change runtime type → T4 GPU*, rồi *Runtime → Run all*.

- `MODE = 'nhanh'` (mặc định): huấn luyện rút gọn (ít epoch, một phần dữ liệu) để kiểm tra toàn bộ quy trình chạy được,
  khoảng **30–45 phút** gồm cả tải dữ liệu (khoảng 5 GB). Số liệu của phần A, B ở chế độ này **thấp hơn** báo cáo là bình thường;
  phần C mới là phần so với báo cáo.
- `MODE = 'day_du'`: đúng cấu hình trong báo cáo (YOLO tối đa 50 epoch, bộ phân loại tối đa 40 epoch), mất **vài giờ** trên T4.
  Nên đặt `LUU_VAO_DRIVE = True` để lưu checkpoint vào Drive **của chính bạn**: nếu Colab bị ngắt, chạy lại notebook sẽ tiếp tục.

**Dữ liệu:** PLC-CECT (Luo và cộng sự, *Scientific Data* 2025, giấy phép **CC BY 4.0**), đã tiền xử lý thành ảnh 2D
(chỉ tập Train và Validation). Tập **Test không được chia sẻ** vì đã được đánh giá đúng một lần với cấu hình khóa.
Dữ liệu MRI (LLD-MMRI, CC BY-NC 4.0) cấm phân phối lại nên không có trong notebook này (xem mục cuối).
"""),
    md("## 1. Kiểm tra GPU"),
    code("""
import os, sys, subprocess, json, hashlib, shutil, zipfile, time
from pathlib import Path
import torch

LOCAL_ZIPS = os.environ.get('LCAI_LOCAL_ZIPS')   # chỉ dùng khi tác giả chạy thử ngoài Colab; trên Colab để trống
WORK = Path(os.environ.get('LCAI_WORK', '/content'))
WORK.mkdir(parents=True, exist_ok=True)

assert torch.cuda.is_available(), 'Không có GPU: Runtime → Change runtime type → T4 GPU'
GPU = torch.cuda.get_device_name(0)
print('torch', torch.__version__, '|', GPU, f'| {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB')
if 'T4' not in GPU:
    print('Lưu ý: không phải T4 -> thời gian khác ước tính; nếu thiếu bộ nhớ GPU hãy giảm BATCH ở mục 2')
"""),
    md("""
## 2. Cấu hình

Chỉ cần đổi `MODE` (và `LUU_VAO_DRIVE` nếu chạy đầy đủ). Không cần sửa gì khác.
"""),
    code("""
MODE = 'nhanh'                       # 'nhanh' (kiểm tra quy trình, ~30–45 phút) | 'day_du' (đúng cấu hình báo cáo, vài giờ)
CHAY_PHAN_A_YOLO = True              # huấn luyện + đánh giá YOLO11n
CHAY_PHAN_B_PHAN_LOAI = True         # huấn luyện + đánh giá bộ phân loại Hybrid
CHAY_PHAN_C_TRONG_SO = True          # đánh giá trọng số đã công bố của đồ án
LUU_VAO_DRIVE = False                # True: lưu kết quả huấn luyện vào Google Drive của chính bạn (nên dùng khi MODE='day_du')
BATCH = 32                           # 32 vừa với GPU T4; giảm 16 nếu báo thiếu bộ nhớ

REPO_URL = '__REPO_URL__'
__FILES__

assert MODE in ('nhanh', 'day_du'), MODE
SEED = 42
if LUU_VAO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    RUNS = Path('/content/drive/MyDrive/LiverCancer_AI_tai_lap') / MODE
else:
    RUNS = WORK / 'runs' / MODE
RUNS.mkdir(parents=True, exist_ok=True)
KET_QUA = {}                         # tổng hợp ở mục cuối
print('MODE =', MODE, '| kết quả lưu ở', RUNS)
"""),
    md("""
## 3. Tải mã nguồn từ GitHub và cài thư viện

Mã nguồn của đồ án (gói `livercancer`, cấu hình, script) được tải bằng `git clone`. Phiên bản Ultralytics được ghim đúng
bản đã dùng khi huấn luyện (8.4.182).
"""),
    code("""
def run(cmd):
    print('$', ' '.join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True)

CODE = Path(os.environ.get('LCAI_LOCAL_REPO') or WORK / 'LiverCancer-AI')
if not CODE.exists():
    run(['git', 'clone', '--depth', '1', REPO_URL, CODE])
if not LOCAL_ZIPS:
    run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.182', 'timm==1.0.30', 'nibabel', 'pyyaml'])
sys.path.insert(0, str(CODE / 'src'))
commit = subprocess.run(['git', '-C', str(CODE), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
import ultralytics
print('Mã nguồn:', CODE, '| commit', commit or '(không phải git)', '| ultralytics', ultralytics.__version__)
"""),
    md("""
## 4. Tải dữ liệu và trọng số, kiểm tra SHA256

Mỗi tệp được tải một lần vào máy ảo Colab rồi kiểm tra mã SHA256. Sai mã thì dừng lại, không dùng tệp hỏng.
"""),
    code("""
def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

def tai_url(url, dst):
    import urllib.request
    tmp = dst.with_name(dst.name + '.tmp')
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, 'wb') as f:
        total, done, step = int(r.headers.get('Content-Length', 0)), 0, 0
        while chunk := r.read(8 << 20):
            f.write(chunk)
            done += len(chunk)
            if done >= step:
                print(f'  {dst.name}: {done / 1e9:.2f} / {total / 1e9:.2f} GB', flush=True)
                step += 500e6
    tmp.rename(dst)

def tai(name):
    info, dst = FILES[name], WORK / name
    if not dst.exists():
        if LOCAL_ZIPS:
            shutil.copy(Path(LOCAL_ZIPS) / name, dst)
        else:
            parts = [WORK / p for p in info['parts']]
            for p in parts:
                if not p.exists():
                    tai_url(RELEASE_URL + p.name, p)
            if parts != [dst]:                       # ghép các phần theo thứ tự thành tệp ZIP ban đầu
                with open(dst, 'wb') as out:
                    for p in parts:
                        with open(p, 'rb') as f:
                            shutil.copyfileobj(f, out, 64 << 20)
                        p.unlink()
    h = sha256(dst)
    if h != info['sha256']:
        dst.unlink()
        raise RuntimeError(f'{name}: SHA256 sai ({h[:16]}…), tệp tải về bị hỏng hoặc không đúng tệp. Chạy lại ô này.')
    print(f'SHA256 OK  {name}  {dst.stat().st_size / 1e9:.2f} GB')
    return dst

def giai_nen(z, dest, marker):
    if not (dest / marker).exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(dest)
    return dest

if CHAY_PHAN_A_YOLO or CHAY_PHAN_C_TRONG_SO:
    YOLO_DATA = giai_nen(tai('yolo_ct_full.zip'), WORK, 'yolo_ct/data.yaml') / 'yolo_ct'
if CHAY_PHAN_B_PHAN_LOAI or CHAY_PHAN_C_TRONG_SO:
    CLS_DATA = giai_nen(tai('cls_ct.zip'), WORK, 'cls_ct/manifest.csv') / 'cls_ct'
if CHAY_PHAN_C_TRONG_SO:
    TRONG_SO = giai_nen(tai('trong_so_CT_LiverCancerAI.zip'), WORK / 'trong_so', 'outputs')
print('Xong.')
"""),
    md("""
## 5. Kiểm tra dữ liệu

- Ảnh YOLO: mỗi ảnh có đúng một tệp nhãn, chỉ có tập Train và Val (không có Test), một lớp `lesion`.
- Dữ liệu phân loại: không có bệnh nhân nào nằm ở cả Train và Val (chia tập theo bệnh nhân).
"""),
    code("""
import csv, yaml
from collections import Counter

if CHAY_PHAN_A_YOLO or CHAY_PHAN_C_TRONG_SO:
    cfg = yaml.safe_load((YOLO_DATA / 'data.yaml').read_text())
    cfg['path'] = str(YOLO_DATA)
    DATA_YAML = YOLO_DATA / 'data_colab.yaml'
    DATA_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False))
    assert cfg['names'] == {0: 'lesion'} and 'test' not in cfg
    for split in ('train', 'val'):
        imgs = {p.stem for p in (YOLO_DATA / 'images' / split).glob('*.png')}
        labs = {p.stem for p in (YOLO_DATA / 'labels' / split).glob('*.txt')}
        assert imgs == labs, f'{split}: ảnh và nhãn không khớp'
        n_box = sum(sum(1 for l in p.read_text().splitlines() if l.strip()) for p in (YOLO_DATA / 'labels' / split).glob('*.txt'))
        print(f'YOLO {split}: {len(imgs):,} ảnh, khoảng {n_box:,} hộp')

if CHAY_PHAN_B_PHAN_LOAI or CHAY_PHAN_C_TRONG_SO:
    rows = list(csv.DictReader(open(CLS_DATA / 'manifest.csv', encoding='utf-8')))
    assert {r['split'] for r in rows} <= {'train', 'val'}, 'Không được có tập Test'
    pats = {}
    for r in rows:
        pats.setdefault(r['patient_id'], set()).add(r['split'])
    assert all(len(s) == 1 for s in pats.values()), 'Có bệnh nhân nằm ở cả Train và Val'
    print('Phân loại: ROI theo (tập, nguồn, lớp):')
    for k, v in sorted(Counter((r['split'], r['source'], r['label']) for r in rows).items()):
        print('   ', k, v)
"""),
    md("""
## 6. Phần A: huấn luyện YOLO11n (phát hiện tổn thương)

Siêu tham số giống hệt notebook huấn luyện chính thức của đồ án: SGD, lr 0,01 giảm theo cosine, ảnh 512, mosaic,
lật ngang, không lật dọc, không xoay. Checkpoint tốt nhất được chọn theo **mAP50-95 trên Val**.

| | `nhanh` | `day_du` (báo cáo) |
|---|---|---|
| Epoch | 3 | tối đa 50, dừng sớm sau 20 epoch không cải thiện |
| Dữ liệu Train | 10% ảnh (chọn ngẫu nhiên) | 100% (34.873 ảnh) |
"""),
    code("""
from ultralytics import YOLO

TRAIN_KW = dict(optimizer='SGD', lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, cos_lr=True, warmup_epochs=3.0,
                close_mosaic=10, mosaic=1.0, mixup=0.0, fliplr=0.5, flipud=0.0, degrees=0.0, translate=0.1, scale=0.5,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.3)
if MODE == 'nhanh':
    EPOCHS, PATIENCE, FRACTION = 3, 3, 0.1
    TRAIN_KW.update(warmup_epochs=1.0, close_mosaic=1)
else:
    EPOCHS, PATIENCE, FRACTION = 50, 20, 1.0
YOLO_PROJECT, YOLO_NAME = RUNS / 'yolo_ct', f'yolo11n_{MODE}'
yolo_last = YOLO_PROJECT / YOLO_NAME / 'weights' / 'last.pt'
yolo_best = YOLO_PROJECT / YOLO_NAME / 'weights' / 'best.pt'

def con_huan_luyen_duoc(p):
    # last.pt của run đã xong bị bỏ optimizer: resume tệp đó sẽ âm thầm huấn luyện mới trên coco8 -> không resume
    ck = torch.load(str(p), map_location='cpu', weights_only=False)
    return ck.get('epoch', -1) >= 0 and ck.get('optimizer') is not None

if CHAY_PHAN_A_YOLO:
    t0 = time.time()
    if yolo_last.exists() and con_huan_luyen_duoc(yolo_last):
        print('Tiếp tục huấn luyện từ', yolo_last)
        YOLO(str(yolo_last)).train(resume=True, data=str(DATA_YAML))
    elif yolo_best.exists():
        print('Run đã hoàn thành trước đó -> bỏ qua huấn luyện:', yolo_best)
    else:
        YOLO('yolo11n.pt').train(data=str(DATA_YAML), epochs=EPOCHS, patience=PATIENCE, fraction=FRACTION, imgsz=512,
                                 batch=BATCH, device=0, amp=True, seed=SEED, deterministic=True, workers=2,
                                 project=str(YOLO_PROJECT), name=YOLO_NAME, exist_ok=True, plots=True, val=True, **TRAIN_KW)
    print(f'Huấn luyện YOLO: {(time.time() - t0) / 60:.1f} phút')
"""),
    md("""
### Đánh giá YOLO11n trên Val

`conf = 0,001` là ngưỡng chuẩn của Ultralytics để tính mAP (giống báo cáo). Precision và recall được lấy tại điểm F1 lớn nhất.
"""),
    code("""
def danh_gia_yolo(weights, ten):
    m = YOLO(str(weights)).val(data=str(DATA_YAML), split='val', conf=0.001, imgsz=512, batch=BATCH, device=0, workers=2,
                               project=str(RUNS / 'yolo_ct'), name=f'val_{ten}', exist_ok=True, plots=True, verbose=False)
    r = {'precision': float(m.box.mp), 'recall': float(m.box.mr), 'mAP50': float(m.box.map50), 'mAP50-95': float(m.box.map)}
    print(ten, {k: round(v, 3) for k, v in r.items()})
    return r

if CHAY_PHAN_A_YOLO:
    KET_QUA['A. YOLO11n tự huấn luyện (' + MODE + ')'] = danh_gia_yolo(yolo_best, f'tu_huan_luyen_{MODE}')
    from IPython.display import Image, display
    for p in (YOLO_PROJECT / YOLO_NAME / 'results.png', RUNS / 'yolo_ct' / f'val_tu_huan_luyen_{MODE}' / 'BoxPR_curve.png'):
        if p.exists():
            display(Image(filename=str(p), width=700))
"""),
    md("""
## 7. Phần B: huấn luyện bộ phân loại lai Hybrid ResNet-18 + Swin-T

- Đầu vào: cặp ảnh 224×224 quanh tổn thương (ảnh cục bộ cho ResNet-18, ảnh ngữ cảnh cho Swin-T), ba lớp HCC / ICC / CHCC.
- AdamW, tốc độ học 1e-4 (backbone) và 1e-3 (phần đầu mới), làm mịn nhãn 0,05, dropout 0,3, trọng số lớp,
  dừng sớm theo **macro-F1 trên Val** (cấu hình đầy đủ trong `configs/cls.yaml`).

| | `nhanh` | `day_du` (báo cáo) |
|---|---|---|
| Epoch | 2 | tối đa 40, dừng sớm sau 8 epoch không cải thiện |
| Vùng Train | 1.024 vùng | toàn bộ 6.026 vùng |
"""),
    code("""
from livercancer.classification.train import train
from livercancer.classification.evaluate import evaluate_val

CLS_RUN = RUNS / 'cls' / f'ct_hybrid_{MODE}'
if CHAY_PHAN_B_PHAN_LOAI:
    cfg = dict(yaml.safe_load(open(CODE / 'configs' / 'cls.yaml', encoding='utf-8'))['train'])
    cfg.update(modality='ct', model='hybrid', roi_source='gt', epochs=40, patience=8, batch_size=BATCH, seed=SEED,
               data_root=str(CLS_DATA), out_dir=str(CLS_RUN), num_workers=2)
    if MODE == 'nhanh':
        cfg.update(epochs=2, patience=2, max_train_rois=1024, resume=False)
    t0 = time.time()
    if (CLS_RUN / 'summary.json').exists():
        print('Run đã hoàn thành trước đó -> bỏ qua huấn luyện:', CLS_RUN)
    else:
        print(train(cfg))
    print(f'Huấn luyện bộ phân loại: {(time.time() - t0) / 60:.1f} phút')
"""),
    md("""
### Đánh giá bộ phân loại trên Val

- **ROI theo nhãn chuẩn** (1.046 vùng của 42 bệnh nhân): accuracy, macro-F1, AUC (một-so-với-còn-lại).
- **ROI do YOLO tìm** (pipeline thật, gồm cả vùng giả): kết quả ở mức bệnh nhân sau khi gộp xác suất.
"""),
    code("""
from IPython.display import Image, display

def danh_gia_phan_loai(run_dir, ten, out_subdir='eval_val'):
    res = evaluate_val(run_dir, CLS_DATA, workers=2, out_subdir=out_subdir)
    g, y = res['gt_roi']['roi'], res['yolo_roi_all']['patient']
    r = {'ROI accuracy': g['accuracy'], 'ROI macro-F1': g['macro_f1'], 'ROI AUC': g['roc_auc_macro_ovr'],
         'BN macro-F1 (vùng YOLO)': y['macro_f1']}
    print(ten, {k: round(v, 3) for k, v in r.items()}, '| cách gộp:', res['aggregation']['chosen'])
    for p in sorted((Path(run_dir) / out_subdir).glob('cm_gt_roi_roi.png')):
        display(Image(filename=str(p), width=520))
    return r

if CHAY_PHAN_B_PHAN_LOAI:
    KET_QUA['B. Hybrid tự huấn luyện (' + MODE + ')'] = danh_gia_phan_loai(CLS_RUN, f'tu_huan_luyen_{MODE}')
    if (CLS_RUN / 'curves.png').exists():
        display(Image(filename=str(CLS_RUN / 'curves.png'), width=700))
"""),
    md("""
## 8. Phần C: đánh giá trọng số đã huấn luyện của đồ án

Dùng đúng hai checkpoint mà đồ án đã chọn trên Val (`best.pt` của YOLO11n CT và của Hybrid CT) để đánh giá lại trên cùng
tập Val. Kết quả phải trùng với báo cáo (có thể lệch ở chữ số thứ ba do khác GPU).
"""),
    code("""
if CHAY_PHAN_C_TRONG_SO:
    w_yolo = TRONG_SO / 'outputs/runs/yolo_ct/yolo11n_e50_img512_baseline/weights/best.pt'
    w_cls = TRONG_SO / 'outputs/runs/cls/ct_hybrid_gt_s42'
    KET_QUA['C. YOLO11n của đồ án'] = danh_gia_yolo(w_yolo, 'trong_so_do_an')
    KET_QUA['C. Hybrid của đồ án'] = danh_gia_phan_loai(w_cls, 'trong_so_do_an', out_subdir='eval_val_tai_lap')
"""),
    md("## 9. Tổng hợp và so sánh với báo cáo"),
    code("""
import pandas as pd
BAO_CAO = {
    'Báo cáo: YOLO11n (Val)': {'precision': 0.688, 'recall': 0.481, 'mAP50': 0.551, 'mAP50-95': 0.282},
    'Báo cáo: Hybrid (Val)': {'ROI accuracy': 0.862, 'ROI macro-F1': 0.863, 'ROI AUC': 0.942, 'BN macro-F1 (vùng YOLO)': 0.902},
}
bang = pd.DataFrame({**BAO_CAO, **KET_QUA}).T.round(3)
display(bang)
(RUNS / 'ket_qua_tai_lap.json').write_text(json.dumps({'mode': MODE, 'gpu': GPU, 'commit': commit, 'ket_qua': KET_QUA},
                                                      ensure_ascii=False, indent=2), encoding='utf-8')
print('Đã lưu', RUNS / 'ket_qua_tai_lap.json')
if 'C. YOLO11n của đồ án' in KET_QUA:
    lech = abs(KET_QUA['C. YOLO11n của đồ án']['mAP50'] - 0.551)
    print('Phần C – YOLO mAP50 lệch so với báo cáo:', round(lech, 4), '-> KHỚP' if lech < 0.005 else '-> KHÁC, cần kiểm tra')
if 'C. Hybrid của đồ án' in KET_QUA:
    lech = abs(KET_QUA['C. Hybrid của đồ án']['ROI macro-F1'] - 0.863)
    print('Phần C – Hybrid macro-F1 lệch so với báo cáo:', round(lech, 4), '-> KHỚP' if lech < 0.005 else '-> KHÁC, cần kiểm tra')
"""),
    md("""
## Ghi chú

**Vì sao không có tập Test?** Theo quy trình của đồ án, mọi lựa chọn (checkpoint, ngưỡng, cách gộp, cổng bằng chứng) chỉ dựa
trên Validation, sau đó cấu hình được khóa và tập Test được chạy **đúng một lần** (26/10/2026). Đánh giá Test cần ảnh NIfTI gốc
của 54 bệnh nhân Test và toàn bộ pipeline (YOLO → Tumor Zoom → Hybrid → cổng bằng chứng):
`scripts/95_final_test_lock.py` và `scripts/96_final_test.py` trong mã nguồn. Kết quả Test (38/42 bệnh nhân ung thư đúng,
macro-F1 0,927) nằm trong báo cáo.

**Tự tải dữ liệu gốc CT:** bộ PLC-CECT được công bố cùng bài báo Luo và cộng sự (*Scientific Data*, 2025). Từ dữ liệu gốc,
các script `10_make_splits.py` và `40_pack_pilot.py --set full` tạo lại đúng bộ ảnh 2D dùng ở đây (SHA256 trong
`data/processed/full/SHA256SUMS_full.txt`).

**MRI:** bộ LLD-MMRI-MedSAM2 có giấy phép CC BY-NC 4.0 kèm điều khoản không sao chép, công bố hay phân phối lại, nên dữ liệu MRI
không được chia sẻ ở đây. Người muốn chạy phần MRI cần tự tải bộ dữ liệu từ nguồn gốc của nhóm tác giả, tạo dữ liệu bằng
các script trong `scripts/`, rồi dùng các notebook `03_`, `04_` và `05_` với `MODALITY = 'mri'`.

**Lưu ý:** hệ thống là sản phẩm nghiên cứu, không thay thế chẩn đoán của bác sĩ.
"""),
]


def build_notebook() -> dict:
    cells = []
    for c in CELLS:
        c = json.loads(json.dumps(c))
        src = "".join(c["source"]).replace("__REPO_URL__", REPO_URL).replace("__FILES__", files_literal())
        c["source"] = src.splitlines(keepends=True)
        cells.append(c)
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--py-out", help="ghi thêm các ô code thành một tệp .py để chạy thử ngoài Colab")
    a = ap.parse_args()
    nb = build_notebook()
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Đã ghi", OUT)
    if a.py_out:
        body = ["from IPython.display import display  # noqa"]
        body += ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        Path(a.py_out).write_text("\n\n".join(body) + "\n", encoding="utf-8")
        print("Đã ghi", a.py_out)


if __name__ == "__main__":
    main()
