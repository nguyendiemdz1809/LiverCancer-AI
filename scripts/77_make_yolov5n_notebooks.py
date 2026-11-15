"""Colab notebooks of the YOLOv5n vs YOLO11n comparison (control experiment; YOLO11n stays the model of the project).

  notebooks/04_train_yolov5n_ct_comparison.ipynb
  notebooks/04_train_yolov5n_mri_comparison.ipynb

Built from the SAME generator as the YOLO11n notebooks that produced the locked models (scripts/41_make_colab_notebooks.py,
function notebook(mod, full=True)), so data ZIP, SHA256 check, label check, epochs, patience, image size, batch, seed,
optimizer, learning rate and augmentation are identical. Only three things change:
  MODEL       'yolo11n.pt' -> 'yolov5nu.pt' (official Ultralytics YOLOv5n weights, COCO pretrained, Ultralytics 8.4.182;
              the "u" version keeps the YOLOv5 backbone/neck with the anchor-free split head of the current library)
  RUN_PROJECT runs/yolo_<mod>  -> runs/comparison/yolov5n_<mod>   (never the YOLO11n folder)
  TAG         'baseline'       -> 'comparison'  => RUN_NAME yolov5nu_e50_img512_comparison
The bundle also keeps last.pt. Test is NOT evaluated on Colab: scripts/78_yolov5n_comparison.py evaluates both models
locally, on the same GPU, once on Test.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nb41", ROOT / "scripts" / "41_make_colab_notebooks.py")
nb41 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb41)

MODEL = "yolov5nu.pt"
TAG = "comparison"


def replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, (text.count(old), old)
    return text.replace(old, new)


def notebook(mod: str) -> dict:
    nb = nb41.notebook(mod, True)
    cells = nb["cells"]
    src = lambda c: "".join(c["source"])  # noqa: E731
    put = lambda c, t: c.__setitem__("source", t.strip("\n").splitlines(keepends=True))  # noqa: E731
    M = mod.upper()
    zip_name = "yolo_ct_full.zip" if mod == "ct" else "yolo_mri_pilot.zip"
    sha_name = "SHA256SUMS_full.txt" if mod == "ct" else "SHA256SUMS.txt"
    n_img = "49.500 ảnh (Train 34.873 / Val 14.627)" if mod == "ct" else "36.086 ảnh (Train 28.595 / Val 7.491)"
    put(cells[0], f"""
# 04 · YOLOv5n {M}: thí nghiệm đối chứng với YOLO11n (Google Colab)

**Mục đích:** so sánh hai biến thể Nano của hai thế hệ YOLO trên **đúng dữ liệu {M}** mà YOLO11n-{M} đã dùng.
YOLO11n vẫn là mô hình chính của đồ án; notebook này **không** thay đổi mô hình, website hay kết quả đã khóa.

- Mô hình: `{MODEL}` (YOLOv5n chính thức của Ultralytics, pretrained COCO, chạy trong Ultralytics 8.4.182 giống YOLO11n).
  Bản "u" giữ backbone/neck YOLOv5 và dùng đầu dự đoán anchor-free của thư viện hiện tại; ghi rõ trong báo cáo.
- Dữ liệu: `{zip_name}` = {n_img}, **cùng ZIP và SHA256** với YOLO11n-{M}.
- Siêu tham số **giống hệt** YOLO11n: 50 epoch tối đa, patience 20, ảnh 512, batch 32, seed 42, SGD lr0 0.01,
  cosine, warmup 3, cùng augmentation (sinh từ cùng `scripts/41_make_colab_notebooks.py`).
- Kết quả lưu ở `MyDrive/LiverCancer_AI/runs/comparison/yolov5n_{mod}/` (**không** ghi vào thư mục YOLO11n).
- Chỉ đánh giá **Val** ở đây. **Test** được đánh giá **một lần** ở máy local cho cả hai mô hình
  (`scripts/78_yolov5n_comparison.py`), cùng GPU và cùng cách đo.
- Thời gian ước tính trên T4: tương tự YOLO11n-{M} (YOLO11n: {'khoảng 7 phút' if mod == 'ct' else 'khoảng 5 phút'}/epoch; early stopping có thể dừng sớm).
- {"CT PLC-CECT: CC BY 4.0." if mod == "ct" else "MRI LLD-MMRI: **CC BY-NC 4.0**, chỉ dùng cho nghiên cứu phi thương mại; không chia sẻ lại dữ liệu."}

**Chuẩn bị:** `{zip_name}` và `{sha_name}` đã có trong `MyDrive/LiverCancer_AI/data/` (đã dùng cho YOLO11n).
Chọn *Runtime → Change runtime type → GPU (T4)* rồi chạy các cell từ trên xuống. Nếu Colab bị ngắt: mục *Resume* ở cuối.
""")
    cfg = next(c for c in cells if "RUN_PROJECT = DRIVE_ROOT" in src(c))
    t = src(cfg)
    t = replace_once(t, f"RUN_PROJECT = DRIVE_ROOT / 'runs' / 'yolo_{mod}'",
                     f"RUN_PROJECT = DRIVE_ROOT / 'runs' / 'comparison' / 'yolov5n_{mod}'")
    t = replace_once(t, "MODEL     = 'yolo11n.pt'                # 'yolo11n.pt' | 'yolo11s.pt' (pretrained COCO)",
                     f"MODEL     = '{MODEL}'               # official Ultralytics YOLOv5n (COCO pretrained) - comparison with yolo11n.pt")
    t = replace_once(t, "TAG = 'baseline'", f"TAG = '{TAG}'")
    t += ("\n# the comparison must never write into the YOLO11n run folder\n"
          "assert 'comparison' in RUN_PROJECT.parts and 'yolo11n' not in RUN_NAME, RUN_PROJECT\n")
    put(cfg, t)
    bundle = next(c for c in cells if "_bundle.zip" in src(c) and "zipfile.ZipFile(bundle" in src(c))
    put(bundle, replace_once(src(bundle), "run / 'weights' / 'best.pt',", "run / 'weights' / 'best.pt', run / 'weights' / 'last.pt',"))
    bmd = next(c for c in cells if "Gói kết quả để phân tích" in src(c))
    put(bmd, f"""
## 9. Gói kết quả để so sánh ở máy local

Tạo `{{RUN_NAME}}_bundle.zip` trên Drive (`results.csv`, `args.yaml`, `metrics_val.json`, `best.pt`, `last.pt`, biểu đồ).
Tải về, giải nén vào `outputs/runs/comparison/yolov5n_{mod}/` (được `outputs/runs/comparison/yolov5n_{mod}/yolov5nu_e50_img512_comparison/`)
rồi chạy `scripts/78_yolov5n_comparison.py`.
""")
    for i, c in enumerate(cells):
        c["id"] = f"{mod}-v5n-{i:02d}"
    return nb


def main() -> None:
    for mod in ("ct", "mri"):
        p = ROOT / "notebooks" / f"04_train_yolov5n_{mod}_comparison.ipynb"
        p.write_text(json.dumps(notebook(mod), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("written", p)


if __name__ == "__main__":
    main()
