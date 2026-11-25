"""Sinh notebooks/06_tai_lap_huan_luyen_danh_gia_CT.ipynb: notebook Colab để NGƯỜI KHÁC (tài khoản Gmail bất kỳ) tự chạy
lại quy trình huấn luyện và đánh giá của đồ án trên CT, không cần Drive của tác giả.

- Mã nguồn: git clone từ GitHub công khai (REPO_URL).
- Dữ liệu đã tiền xử lý (PLC-CECT, CC BY 4.0) và trọng số đã huấn luyện: tải từ GitHub Releases của repo (tag du-lieu-ct-v1),
  kiểm tra SHA256 trước khi dùng. MRI (LLD-MMRI, CC BY-NC 4.0, cấm phân phối lại) KHÔNG được đưa vào.
- Bốn phần: (A) huấn luyện + đánh giá YOLO11n, (B) huấn luyện + đánh giá bộ phân loại Hybrid, (C) đánh giá lại trọng số
  đã công bố trên Val, (D) đánh giá lại trọng số đã khóa trên tập TEST (lát cắt + toàn bộ pipeline mức bệnh nhân) để so với
  báo cáo. Chế độ 'nhanh' để kiểm tra quy trình, 'day_du' = đúng cấu hình báo cáo.
- Phần D (người dùng quyết định 2026-11-25 theo yêu cầu GVHD): chỉ ĐÁNH GIÁ bằng cấu hình đã khóa (không huấn luyện, không
  chọn ngưỡng/checkpoint trên Test); dữ liệu Test CT (ảnh NIfTI gốc, CC BY 4.0) đóng gói bởi scripts/102_pack_test_ct_release.py.

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
    # Test CT (scripts/102_pack_test_ct_release.py): ảnh NIfTI gốc + lát có tổn thương + kết quả từng BN của lần chạy chính thức
    "test_ct.zip": ([f"test_ct.zip.part{i}" for i in range(1, 5)],
                    "46f767cf99f776409521af9322ff0ddda5bf6e3f5811860bee9de8f11ef319cd"),
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
| D | Đánh giá **trọng số đã khóa** trên tập **Test** (54 bệnh nhân, chưa từng dùng để huấn luyện hay chọn mô hình): D1 YOLO11n trên lát cắt có tổn thương, D2 toàn bộ pipeline YOLO → Tumor Zoom → Hybrid → cổng bằng chứng trên ảnh CT 3D gốc | D1: P 0,598 · R 0,515 · mAP50 0,523 · D2: 38/42 BN ung thư đúng, macro-F1 0,927, nhóm chứng bị gán ung thư 2/12 |

**Cách chạy:** *Runtime → Change runtime type → T4 GPU*, rồi *Runtime → Run all*.

- `MODE = 'nhanh'` (mặc định): huấn luyện rút gọn (ít epoch, một phần dữ liệu) để kiểm tra toàn bộ quy trình chạy được,
  khoảng **1–1,5 giờ** gồm cả tải dữ liệu (khoảng 11 GB). Số liệu của phần A, B ở chế độ này **thấp hơn** báo cáo là bình thường;
  phần C và D mới là phần so với báo cáo (không phụ thuộc `MODE`).
- `MODE = 'day_du'`: đúng cấu hình trong báo cáo (YOLO tối đa 50 epoch, bộ phân loại tối đa 40 epoch), mất **vài giờ** trên T4.
  Nên đặt `LUU_VAO_DRIVE = True` để lưu checkpoint vào Drive **của chính bạn**: nếu Colab bị ngắt, chạy lại notebook sẽ tiếp tục.

**Dữ liệu:** PLC-CECT (Luo và cộng sự, *Scientific Data* 2025, giấy phép **CC BY 4.0**): Train và Validation đã tiền xử lý
thành ảnh 2D; Test gồm **ảnh CT 3D gốc** (giữ nguyên từng byte) của 54 bệnh nhân và các lát có tổn thương.
Tập Test chỉ dùng để **đánh giá** với cấu hình đã khóa từ trước (ngưỡng, checkpoint, cách gộp, cổng bằng chứng đều chọn trên Val).
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
CHAY_PHAN_C_TRONG_SO = True          # đánh giá trọng số đã công bố của đồ án trên Val
CHAY_PHAN_D_TEST = True              # đánh giá trọng số đã khóa trên tập Test (tải thêm khoảng 6 GB ảnh CT gốc)
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
    run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.182', 'timm==1.0.30', 'nibabel', 'pyyaml', 'scikit-learn'])
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
        if LOCAL_ZIPS and (Path(LOCAL_ZIPS) / name).exists():
            shutil.copy(Path(LOCAL_ZIPS) / name, dst)
        else:
            srcs = []                                # (đường dẫn phần, có xóa sau khi ghép không)
            for p in info['parts']:
                if LOCAL_ZIPS and (Path(LOCAL_ZIPS) / p).exists():
                    srcs.append((Path(LOCAL_ZIPS) / p, False))
                else:
                    q = WORK / p
                    if not q.exists():
                        tai_url(RELEASE_URL + p, q)
                    srcs.append((q, True))
            if [s for s, _ in srcs] != [dst]:        # ghép các phần theo thứ tự thành tệp ZIP ban đầu
                with open(dst, 'wb') as out:
                    for p, xoa in srcs:
                        with open(p, 'rb') as f:
                            shutil.copyfileobj(f, out, 64 << 20)
                        if xoa:
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
if CHAY_PHAN_C_TRONG_SO or CHAY_PHAN_D_TEST:
    TRONG_SO = giai_nen(tai('trong_so_CT_LiverCancerAI.zip'), WORK / 'trong_so', 'outputs')
    W_YOLO = TRONG_SO / 'outputs/runs/yolo_ct/yolo11n_e50_img512_baseline/weights/best.pt'
    W_CLS = TRONG_SO / 'outputs/runs/cls/ct_hybrid_gt_s42'
if CHAY_PHAN_D_TEST:
    TEST_DATA = giai_nen(tai('test_ct.zip'), WORK, 'test_ct/CECT/test_volumes.csv') / 'test_ct'
    TEST_YOLO = TEST_DATA / 'yolo_ct'
print('Xong.')
"""),
    md("""
## 5. Kiểm tra dữ liệu

- Ảnh YOLO: mỗi ảnh có đúng một tệp nhãn, chỉ có tập Train và Val (không có Test), một lớp `lesion`.
- Dữ liệu phân loại: không có bệnh nhân nào nằm ở cả Train và Val (chia tập theo bệnh nhân).
- Tập Test: mọi bệnh nhân đều thuộc Test trong tệp chia tập đã đóng băng và không trùng với Train/Val.
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

if CHAY_PHAN_D_TEST:
    split_ct = {r['patient_id']: r for r in csv.DictReader(open(CODE / 'data/splits/ct_split.csv', encoding='utf-8-sig'))}
    vt = list(csv.DictReader(open(TEST_DATA / 'CECT/test_volumes.csv', encoding='utf-8')))
    bn_test = {r['patient_id'] for r in vt}
    assert all(split_ct[p]['split'] == 'test' for p in bn_test), 'Có BN không thuộc tập Test'
    assert all((TEST_DATA / 'CECT' / r['ct_path']).exists() for r in vt), 'Thiếu ảnh CT 3D'
    bn_train_val = {r['patient_id'] for r in rows} if (CHAY_PHAN_B_PHAN_LOAI or CHAY_PHAN_C_TRONG_SO) else set()
    assert not (bn_test & bn_train_val), 'BN Test lẫn vào Train/Val'
    print(f'Test: {len(vt)} volume CT 3D của {len(bn_test)} BN (ung thư + nhóm chứng CN);',
          'nhãn:', dict(Counter(split_ct[p]['label'] for p in bn_test)))
    print(f'Test lát có tổn thương: {len(list((TEST_YOLO / "images/test").glob("*.png"))):,} ảnh')
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
def danh_gia_yolo(weights, ten, data=None, batch=None):
    m = YOLO(str(weights)).val(data=str(data or DATA_YAML), split='val', conf=0.001, iou=0.7, imgsz=512, batch=batch or BATCH,
                               device=0, workers=2, project=str(RUNS / 'yolo_ct'), name=f'val_{ten}', exist_ok=True, plots=True,
                               verbose=False)
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
    KET_QUA['C. YOLO11n của đồ án'] = danh_gia_yolo(W_YOLO, 'trong_so_do_an')
    KET_QUA['C. Hybrid của đồ án'] = danh_gia_phan_loai(W_CLS, 'trong_so_do_an', out_subdir='eval_val_tai_lap')
"""),
    md("""
## 9. Phần D: đánh giá trên tập Test bằng cấu hình đã khóa

Tập Test gồm **54 bệnh nhân** (42 ung thư: HCC / ICC / CHCC, và 12 bệnh nhân nhóm chứng CN), tách theo bệnh nhân ngay từ đầu,
**không** dùng để huấn luyện hay chọn bất cứ thứ gì. Mọi tham số đã được chọn trên Val và khóa trước khi chạy Test
(ngưỡng YOLO 0,12, checkpoint, cách gộp xác suất, cổng bằng chứng); phần này chỉ **chạy lại** đúng cấu hình đó.

- **D1 – mức lát cắt:** YOLO11n trên 7.163 lát Test có tổn thương (chỉ lát có tổn thương, nên precision không so được với Val
  vốn có cả lát âm tính).
- **D2 – mức bệnh nhân (kết quả chính của báo cáo):** đưa toàn bộ ảnh CT 3D gốc của từng bệnh nhân qua pipeline thật
  YOLO → Tumor Zoom → Hybrid → cổng bằng chứng. Pipeline **chỉ dùng hộp do YOLO tìm**; nhãn chuẩn chỉ dùng **sau** khi chạy
  để chấm điểm. Bệnh nhân bị cổng từ chối hoặc không có vùng **không** được tính là đúng.
"""),
    code("""
if CHAY_PHAN_D_TEST:
    tcfg = yaml.safe_load((TEST_YOLO / 'data.yaml').read_text())
    tcfg['path'] = str(TEST_YOLO)
    TEST_YAML = TEST_YOLO / 'data_colab.yaml'
    TEST_YAML.write_text(yaml.safe_dump(tcfg, sort_keys=False))
    KET_QUA['D1. YOLO11n của đồ án – Test (lát có tổn thương)'] = danh_gia_yolo(W_YOLO, 'test_trong_so_do_an', data=TEST_YAML, batch=16)
"""),
    code("""
import importlib.util
import numpy as np
from sklearn.metrics import roc_auc_score

if CHAY_PHAN_D_TEST:
    # trọng số đặt đúng chỗ mà cấu hình của pipeline trỏ tới (configs/yolo_inference.yaml, configs/cls.yaml; SHA256 được kiểm tra khi nạp)
    shutil.copytree(TRONG_SO / 'outputs', CODE / 'outputs', dirs_exist_ok=True)
    spec = importlib.util.spec_from_file_location('b96', CODE / 'scripts' / '96_final_test.py')
    b96 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b96)                 # đúng mã đã dùng cho lần chạy Test chính thức (evaluate_case, chỉ số)
    from livercancer.pipeline import LiverCancerPipeline
    from livercancer.config import CONFIG_DIR, load_yaml

    CLASSES = load_yaml(CONFIG_DIR / 'cls.yaml')['ct']['classes']
    PATHS = {'ct': {'csv': str(TEST_DATA / 'CECT/test_volumes.csv'), 'data_dir': str(TEST_DATA / 'CECT')}}
    allv = b96.rows(TEST_YOLO / 'volumes.csv')
    vols = [v for v in allv if v['status'] == 'ok']
    pats = sorted({v['patient_id'] for v in allv})
    assert len(pats) == 54 and all(split_ct[p]['split'] == 'test' for p in pats)
    gt_les, files = b96.b80.gt_lesions(TEST_YOLO, 'test')
    pipe = LiverCancerPipeline('ct', 'auto', evidence_gate='config')
    assert pipe.gate is not None and pipe.detector.conf == 0.12 and pipe.classifier.aggregation == 'mean_prob', 'cấu hình khác bản đã khóa'
    print('Pipeline:', pipe.detector.device, '| YOLO conf', pipe.detector.conf, '| gộp', pipe.classifier.aggregation, '| cổng bằng chứng: bật')

    CASES = RUNS / 'test_ct' / 'cases'
    CASES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, pid in enumerate(pats):
        cf = CASES / f'{pid}.json'
        if cf.exists():
            continue
        series = [v['series'] for v in vols if v['patient_id'] == pid]
        true = split_ct[pid]['label']
        if not series:
            case = {'patient_id': pid, 'true_label': true, 'status': 'no_usable_volume', 'pipeline_label': None}
        else:
            case, res = b96.evaluate_case(pipe, PATHS, 'ct', pid, series, true, gt_les,
                                          b96.oracle_rois('ct', pid, series, gt_les, files), CLASSES)
        cf.write_text(json.dumps(case, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f"[{i + 1:2d}/{len(pats)}] {pid}  nhãn thật {true:5s} -> {case['status']:22s} nhãn hệ thống {case.get('pipeline_label')}",
              flush=True)
    print(f'Pipeline trên Test: {(time.time() - t0) / 60:.1f} phút')
"""),
    md("""
### Chỉ số Test mức bệnh nhân và so với báo cáo

Tính bằng đúng các hàm của `scripts/96_final_test.py` (script đã tạo số liệu Test trong báo cáo).
"""),
    code("""
import pandas as pd
from IPython.display import Image, display

if CHAY_PHAN_D_TEST:
    cases = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(CASES.glob('*.json'))]
    lab = [c for c in cases if c['true_label'] in CLASSES]
    ctrl = [c for c in cases if c['true_label'] not in CLASSES]
    cnt = b96.counts(lab)
    pm = b96.cls_metrics(lab, CLASSES)
    om = b96.cls_metrics([c for c in lab if c.get('volumes')], CLASSES, 'oracle_label')
    det = b96.detection([c for c in cases if c.get('volumes')])
    may = [c['patient_id'] for c in lab if b96.outcome(c) == 'correct' and not any(t['matched'] for t in c.get('tracks', []))]
    co_diem = [c for c in lab if c.get('pipeline_scores')]
    y = np.array([CLASSES.index(c['true_label']) for c in co_diem])
    p = np.array([[c['pipeline_scores'][k] for k in CLASSES] for c in co_diem])
    auc = float(np.mean([roc_auc_score(y == i, p[:, i]) for i in range(len(CLASSES))]))
    d2 = {'BN ung thư đúng': cnt['correct'], 'BN ung thư sai': cnt['wrong'], 'BN ung thư bị từ chối': cnt['rejected'],
          'BN ung thư không có vùng': cnt['no_region'], 'Accuracy (trên 42 BN)': round(pm['accuracy_all'], 3),
          'Macro-F1': round(pm['macro_f1'], 3), 'Đúng khi YOLO tìm thấy tổn thương thật': cnt['correct'] - len(may),
          f'AUC macro ({len(co_diem)} BN có nhãn)': round(auc, 3), 'Oracle đúng (vùng theo nhãn chuẩn)': om['correct'],
          'CN bị gán nhãn ung thư (/12)': sum(c['status'] == 'ok' for c in ctrl),
          'CN bị từ chối': sum(c['status'] == 'insufficient_evidence' for c in ctrl),
          'CN không có vùng': sum(c['status'] == 'no_lesion_detected' for c in ctrl),
          'Tổn thương được phát hiện': f"{det['lesions_detected']}/{det['lesions']}",
          'Tổn thương nhỏ (< 32×32 px) được phát hiện': f"{det['small_detected']}/{det['small']}"}
    BAO_CAO_D2 = {'BN ung thư đúng': 38, 'BN ung thư sai': 2, 'BN ung thư bị từ chối': 2, 'BN ung thư không có vùng': 0,
                  'Accuracy (trên 42 BN)': 0.905, 'Macro-F1': 0.927, 'Đúng khi YOLO tìm thấy tổn thương thật': 36,
                  'AUC macro (40 BN có nhãn)': 0.991, 'Oracle đúng (vùng theo nhãn chuẩn)': 41, 'CN bị gán nhãn ung thư (/12)': 2,
                  'CN bị từ chối': 8, 'CN không có vùng': 2, 'Tổn thương được phát hiện': '184/288',
                  'Tổn thương nhỏ (< 32×32 px) được phát hiện': '11/71'}
    bang_d2 = pd.DataFrame({'Báo cáo (Test chính thức 26/10/2026)': pd.Series(BAO_CAO_D2, dtype=object),
                            'Chạy lại trong notebook': pd.Series(d2, dtype=object)})
    bang_d2['Khớp'] = ['✓' if str(a) == str(b) else '✗' for a, b in zip(bang_d2.iloc[:, 0], bang_d2.iloc[:, 1])]
    display(bang_d2)
    KET_QUA['D2. Pipeline của đồ án – Test (mức BN)'] = {'BN ung thư đúng': cnt['correct'], 'Macro-F1 BN': pm['macro_f1'],
                                                        'CN bị gán ung thư': d2['CN bị gán nhãn ung thư (/12)']}

    # so từng bệnh nhân với kết quả của lần chạy Test chính thức
    ref = {r['patient_id']: r for r in csv.DictReader(open(TEST_DATA / 'bao_cao_test_ct_patients.csv', encoding='utf-8'))}
    khac = [(c['patient_id'], ref[c['patient_id']]['status'], ref[c['patient_id']]['pipeline_label'], c['status'], c.get('pipeline_label') or '')
            for c in cases if (ref[c['patient_id']]['status'], ref[c['patient_id']]['pipeline_label']) != (c['status'], c.get('pipeline_label') or '')]
    print(f'Từng bệnh nhân: {len(cases) - len(khac)}/{len(cases)} trùng với lần chạy Test chính thức (trạng thái và nhãn).')
    for k in khac:
        print('  khác:', k)

    # ma trận nhầm lẫn mức bệnh nhân (hàng cuối = nhóm chứng CN)
    cm = [list(r) for r in pm['confusion']]
    cn_ok = [c for c in ctrl if c['status'] == 'ok']
    cm.append([sum(1 for c in cn_ok if c['pipeline_label'] == k) for k in CLASSES] + [len(ctrl) - len(cn_ok)])
    hinh = RUNS / 'test_ct' / 'cm_test_ct_pipeline.png'
    b96.plot_cm(cm, CLASSES + ['CN (nhóm chứng)'], CLASSES + ['Từ chối / không có vùng'],
                'TEST CT – pipeline (mức bệnh nhân, cấu hình đã khóa)', hinh)
    display(Image(filename=str(hinh), width=560))
"""),
    md("## 10. Tổng hợp và so sánh với báo cáo"),
    code("""
import pandas as pd
BAO_CAO = {
    'Báo cáo: YOLO11n (Val)': {'precision': 0.688, 'recall': 0.481, 'mAP50': 0.551, 'mAP50-95': 0.282},
    'Báo cáo: Hybrid (Val)': {'ROI accuracy': 0.862, 'ROI macro-F1': 0.863, 'ROI AUC': 0.942, 'BN macro-F1 (vùng YOLO)': 0.902},
    'Báo cáo: YOLO11n (Test, lát có tổn thương)': {'precision': 0.598, 'recall': 0.515, 'mAP50': 0.523, 'mAP50-95': 0.261},
    'Báo cáo: Pipeline (Test, mức BN)': {'BN ung thư đúng': 38, 'Macro-F1 BN': 0.927, 'CN bị gán ung thư': 2},
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
if 'D1. YOLO11n của đồ án – Test (lát có tổn thương)' in KET_QUA:
    lech = abs(KET_QUA['D1. YOLO11n của đồ án – Test (lát có tổn thương)']['mAP50'] - 0.523)
    print('Phần D1 – YOLO mAP50 trên Test lệch so với báo cáo:', round(lech, 4), '-> KHỚP' if lech < 0.005 else '-> KHÁC, cần kiểm tra')
if 'D2. Pipeline của đồ án – Test (mức BN)' in KET_QUA:
    d = KET_QUA['D2. Pipeline của đồ án – Test (mức BN)']
    ok = d['BN ung thư đúng'] == 38 and abs(d['Macro-F1 BN'] - 0.927) < 0.005 and d['CN bị gán ung thư'] == 2
    print('Phần D2 – pipeline trên Test:', f"{d['BN ung thư đúng']}/42 đúng, macro-F1 {d['Macro-F1 BN']:.3f}, CN gán ung thư {d['CN bị gán ung thư']}/12",
          '-> KHỚP' if ok else '-> KHÁC (xem bảng ở phần D2)')
"""),
    md("""
## Ghi chú

**Về tập Test:** theo quy trình của đồ án, mọi lựa chọn (checkpoint, ngưỡng, cách gộp, cổng bằng chứng) chỉ dựa trên
Validation; cấu hình được khóa (`scripts/95_final_test_lock.py`) rồi tập Test được chạy chính thức một lần ngày 26/10/2026
(`scripts/96_final_test.py`). Phần D ở đây **chạy lại đúng cấu hình đã khóa** để người đọc tự kiểm chứng số liệu, không
huấn luyện hay điều chỉnh gì theo Test. Ảnh CT 3D của Test là tệp gốc của PLC-CECT (CC BY 4.0), giữ nguyên từng byte.
Khác GPU (T4 so với RTX 3050 của tác giả) có thể làm độ tin cậy của YOLO lệch rất nhỏ (fp16); ca nằm sát ngưỡng có thể đổi kết quả.

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
