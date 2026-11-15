# Chạy demo suy luận YOLO CT/MRI trong VS Code

Demo dùng hai mô hình đã train trên Colab (**không train lại**), mỗi modality một mô hình, 1 lớp `lesion`:

| | Checkpoint | Run |
|---|---|---|
| CT | `outputs/runs/yolo_ct/yolo11n_e50_img512_baseline/weights/best.pt` | CT-E0: best epoch 15, early stop 35/50 |
| MRI | `outputs/runs/yolo_mri/yolo11n_e50_img512_baseline/weights/best.pt` | MRI-E0: best epoch 24, early stop 44/50 |

Đường dẫn, SHA256, `imgsz` và `conf` nằm trong `configs/yolo_inference.yaml`. Khi nạp mô hình, SHA256 được kiểm tra; nếu checkpoint bị thay
mà chưa cập nhật SHA thì chương trình dừng với thông báo lỗi.

**Ngưỡng confidence (đã chốt 2026-10-25):** `conf = 0.12` cho CT và cho MRI. Mỗi ngưỡng được chọn trên Val của **chính modality đó**
(F2 lớn nhất, ưu tiên hạn chế bỏ sót; `scripts/53_threshold_sweep.py` → `analysis/threshold_choice.json`); cả hai đều ra 0.12.
Ngưỡng 0.25 (mặc định Ultralytics) chỉ còn dùng cho bản so sánh. Giai đoạn YOLO Detection đã đóng: hai checkpoint `best.pt` là bản cuối.
Tổng kết: `reports/yolo_detection_summary.md`.

## 1. Chuẩn bị (một lần)
1. Mở thư mục `E:\LiverCancer_AI_Graduation` trong VS Code (*File → Open Folder*).
2. Cài extension **Python** (Microsoft); extension này kèm sẵn Python Debugger.
3. `Ctrl+Shift+P` → *Python: Select Interpreter* → chọn `.venv\Scripts\python.exe` của project.
4. Kiểm tra môi trường trong terminal của VS Code (`` Ctrl+` ``):
   ```powershell
   .venv\Scripts\python scripts\00_check_env.py
   $env:YOLO_OFFLINE = "1"; .venv\Scripts\python -m pytest tests\test_yolo_inference.py -v
   ```
   Kết quả mong đợi: mọi test PASS, gồm `test_nifti_slice_equals_training_png` (ảnh dựng từ NIfTI trùng từng pixel với ảnh train).

## 2. Chạy bằng F5 (Run and Debug)
`Ctrl+Shift+D` → chọn cấu hình trong danh sách → `F5`. Các cấu hình nằm trong `.vscode/launch.json`
(file này được commit; các file khác trong `.vscode/` vẫn bị bỏ qua).

| Cấu hình | Làm gì | Kết quả |
|---|---|---|
| **YOLO demo: CT + MRI trên Val, conf 0.12 (đã chốt)** | 12 lát Val mỗi modality (8 có tổn thương, 4 không annotation; seed 42, mỗi lát một BN), đọc thẳng từ NIfTI gốc, vẽ GT + dự đoán, đếm TP/FP/FN | `reports/figures/inference/{ct,mri}/*.png`, `*_montage.png`, `reports/inference_demo.md`, `outputs/predictions/inference_demo/` |
| YOLO demo: bản so sánh ở conf 0.25 | Cùng mẫu, ở conf 0.25 | `reports/inference_demo_conf025.md`, `reports/figures/inference_conf025/` |
| YOLO demo: chạy lại trên CPU | Cùng mẫu, fp32 trên CPU, chỉ đo thời gian và đếm | `outputs/predictions/inference_demo/summary_cpu.json` (báo cáo đọc file này ở lần chạy GPU sau) |
| YOLO CT: dự đoán một volume | Hỏi ngưỡng (mặc định 0.12), quét mọi lát của `P0267` pha C3, vẽ 4 lát có confidence cao nhất | `outputs/predictions/adhoc/P0267_ct_C3/` |
| YOLO MRI: dự đoán một volume | Tương tự với `MR179970` chuỗi C+V | `outputs/predictions/adhoc/MR179970_1_C+V_0000/` |
| YOLO: dự đoán file NIfTI tự chọn | Hỏi modality, đường dẫn `.nii.gz` và ngưỡng | `outputs/predictions/adhoc/<tên file>/` |
| pytest: kiểm tra đường suy luận | `tests/test_yolo_inference.py` | |

Mở ảnh PNG kết quả ngay trong VS Code (bấm vào file trong Explorer), hoặc mở `reports/inference_demo.md` rồi bấm `Ctrl+Shift+V` để xem dạng preview.
Đặt breakpoint (F9) trong `src/livercancer/detection/yolo_detector.py` (`LesionDetector.predict`) để xem box và confidence từng lát.

## 3. Chạy bằng terminal
```powershell
$env:YOLO_OFFLINE = "1"; $env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python scripts\60_yolo_inference_demo.py                       # CT + MRI, conf 0.12, tự chọn GPU/CPU
.venv\Scripts\python scripts\60_yolo_inference_demo.py --modality mri --n-pos 12 --n-neg 4 --seed 7
.venv\Scripts\python scripts\60_yolo_inference_demo.py --conf 0.25 --out-tag conf025     # bản so sánh ở 0.25, không ghi đè demo chính
.venv\Scripts\python scripts\60_yolo_inference_demo.py --device cpu --no-figures
.venv\Scripts\python scripts\61_yolo_predict_nifti.py --modality ct --patient P0267 --series C3
.venv\Scripts\python scripts\61_yolo_predict_nifti.py --modality mri --patient MR179970 --series "C+V" --slices 44 45 46
.venv\Scripts\python scripts\61_yolo_predict_nifti.py --modality mri --nifti "E:\LiverDataset\LLD-MMRI-MedSAM2\images\MR179970_1_T2WI_0000.nii.gz"
```
- `--series` của MRI dùng **tên trong JSON** (`In Phase`, `Out Phase`, `C+A`, …). CT dùng `C1/C2/C3/P`.
- Đổi seed hoặc số lát thì ra mẫu khác. Số TP/FP/FN của demo chỉ mô tả mẫu đó, **không** phải kết quả đánh giá.
- `YOLO_OFFLINE=1` để Ultralytics không kiểm tra mạng. Demo không cần tải gì.

## 4. GPU 4 GB và CPU
- `--device auto` (mặc định): dùng CUDA nếu có, với fp16 (`quantize=16`); không có CUDA thì chạy CPU fp32.
  Khi yêu cầu `--device cuda` mà máy không có CUDA, chương trình cảnh báo rồi chuyển sang CPU.
  Nếu hết VRAM giữa chừng, batch đó được chạy lại trên CPU.
- Đo trên máy này (RTX 3050 Laptop 4 GB, i5-12450H): GPU khoảng 5 ms/ảnh; nạp cả CT và MRI và chạy batch 8 dùng khoảng 235 MiB VRAM (`nvidia-smi`);
  CPU khoảng 30 ms/ảnh. Chi tiết: `reports/inference_demo.md` §1 và §4.4.
- Thử đường CPU trên máy có GPU: `$env:CUDA_VISIBLE_DEVICES = "-1"` trước khi chạy (nhớ xóa biến này sau đó: `Remove-Item Env:CUDA_VISIBLE_DEVICES`).

## 5. Tiền xử lý dùng khi suy luận (giống lúc train)
`livercancer.data.slices`: NIfTI → kiểm tra hướng **LPS** → CT 5D lấy kênh 0 (`as_3d`, ≤ 1 % voxel khác: chữ thước đo) → giữ nguyên **uint8**
(không windowing hay chuẩn hóa thêm) → lát `k` hiển thị dạng `slice.T` → lặp thành 3 kênh như `cv2.IMREAD_COLOR` khi train.
Toạ độ box là pixel của ảnh hiển thị: `x` = cột = trục i, `y` = hàng = trục j.

## 6. Quy tắc
- Demo chỉ lấy mẫu từ **Validation**. Không dùng Test để chọn ngưỡng hay mô hình.
  `61_yolo_predict_nifti.py` in cảnh báo khi BN thuộc Test (chỉ để minh họa).
- GT chỉ dùng để vẽ và đếm TP/FP/FN, **không** bao giờ là đầu vào của mô hình.
- Ảnh trong `reports/figures/` và `outputs/` chứa dữ liệu bệnh nhân, nên không commit (đã có trong `.gitignore`). Dữ liệu MRI theo CC BY-NC 4.0.
