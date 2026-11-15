# Hướng dẫn chạy YOLO MRI pilot trên Google Colab

Notebook: `notebooks/03_train_yolo_mri_colab.ipynb`. Pilot: YOLO11n pretrained (COCO), 5 epoch, chỉ Train + Val.
Tập Test không nằm trong ZIP và không được dùng.

## 0. Chuẩn bị trên Google Drive (đã làm ở GĐ2)
```
MyDrive/LiverCancer_AI/
├── data/
│   ├── yolo_mri_pilot.zip        (2.68 GB, SHA256 0088fa93…9f2f)
│   ├── yolo_ct_pilot.zip
│   └── SHA256SUMS.txt
└── runs/                         (notebook tự tạo runs/yolo_mri/)
```
- Thư mục phải **riêng tư, không chia sẻ**. MRI theo license CC BY-NC 4.0, với điều khoản "not to further copy, publish, or distribute".
- Kiểm tra lại SHA256 trong `SHA256SUMS.txt` khớp với file ở máy local (`data/processed/pilot/SHA256SUMS.txt`).

## 1. Mở notebook
1. Vào https://colab.research.google.com, chọn *File → Upload notebook*, chọn `notebooks/03_train_yolo_mri_colab.ipynb`.
   Hoặc copy file notebook lên Drive rồi mở bằng Colab.
2. Chọn *Runtime → Change runtime type → Hardware accelerator: GPU (T4)*, rồi *Save*.

## 2. Chạy lần lượt từng cell

| Mục | Làm gì | Kết quả mong đợi | Thời gian (ước lượng) |
|---|---|---|---|
| 1. GPU | `nvidia-smi`, kiểm tra CUDA | `Tesla T4 … 15.x GB` | vài giây |
| 2. Drive + cấu hình | Mount Drive (cấp quyền), đặt đường dẫn, `RESUME = False` | In đường dẫn ZIP và `2.68 GB` | ~30 giây |
| 3. SHA256 + giải nén | Copy ZIP vào `/content`, so SHA256, giải nén | `SHA256 OK 0088fa93…`; danh sách file trong `yolo_mri/` | 3–6 phút |
| 4. Cài Ultralytics | `pip install ultralytics==8.4.182` | `ultralytics.checks()` báo GPU | ~1 phút |
| 5. Kiểm tra nhãn | 100% nhãn hợp lệ, cặp ảnh–nhãn, không trùng BN train/val | `labels OK: 23541 boxes`; train 28595 ảnh (18619 dương / 9976 âm), val 7491 ảnh (4874 / 2617); BN 316 / 78 | ~1 phút |
| 6. Train | YOLO11n, 5 epoch, imgsz 512, batch 16, AMP | Log từng epoch; `last.pt` và `best.pt` trên Drive | phụ thuộc GPU, đo thực tế |
| 7. Đánh giá Val | `best.pt` ở conf 0.001 (mAP) và conf 0.25 (ngưỡng inference) | JSON có Precision, Recall, mAP50, mAP50-95 và số TP/FP/FN của confusion matrix ở mỗi ngưỡng; ghi `metrics_val.json` | vài phút |
| 8. Biểu đồ | `results.png`, PR curve, 2 confusion matrix | Hiển thị ảnh, liệt kê `best.pt`, `last.pt` | vài giây |
| 9. Bundle | `pilot_yolo11n_e5_bundle.zip` trên Drive | Kích thước vài MB | vài giây |

Các con số ở mục 5 lấy từ QA ở local (`data/processed/pilot/yolo_mri/qa_report.json`). Nếu Colab in ra số khác, **dừng lại** và báo lại.

## 3. Nếu Colab bị ngắt giữa lúc train
1. Kết nối lại runtime GPU, chạy lại mục **1 → 5**. Mục 3 sẽ giải nén lại vì `/content` đã bị xóa.
2. Ở mục 2, sửa `RESUME = True` rồi chạy lại cell đó.
3. Chạy mục 6: Ultralytics đọc `runs/yolo_mri/pilot_yolo11n_e5/weights/last.pt` và chạy tiếp.
4. Chạy tiếp mục 7–9.

Nếu run đã tồn tại mà `RESUME = False`, mục 6 sẽ **bỏ qua việc train**, không ghi đè. Mục 7–9 vẫn chạy được trên run cũ.

## 4. Sau khi chạy xong, gửi lại cho phân tích
1. Tải `MyDrive/LiverCancer_AI/runs/yolo_mri/pilot_yolo11n_e5_bundle.zip` về máy.
2. Giải nén vào `E:\LiverCancer_AI_Graduation\outputs\runs\yolo_mri\` (sẽ tạo thư mục `pilot_yolo11n_e5\`).
3. Chạy (máy local, GPU 4 GB):
   ```powershell
   .venv\Scripts\python scripts\50_analyze_yolo_run.py --modality mri --run outputs\runs\yolo_mri\pilot_yolo11n_e5
   ```
   Script sinh `reports/mri_pilot_analysis.md` với số liệu đọc từ bundle và từ lần chạy lại `best.pt` trên Val ở local.

## 5. Với CT pilot đã train (không train lại)
Mở `notebooks/03_train_yolo_ct_colab.ipynb` **bản mới**, chạy mục 1–5, giữ `RESUME = False`. Mục 6 sẽ tự bỏ qua vì run đã tồn tại.
Sau đó chạy mục 7–9 để có metrics của `best.pt` ở cả hai ngưỡng và tạo `pilot_yolo11n_e5_bundle.zip`.
Tải bundle về `outputs\runs\yolo_ct\`, rồi chạy `scripts\50_analyze_yolo_run.py --modality ct --run outputs\runs\yolo_ct\pilot_yolo11n_e5`.
Nếu không muốn chạy lại trên Colab, có thể tải trực tiếp `results.csv`, `args.yaml`, `weights/best.pt` (và `metrics_val.json` nếu có) vào cùng thư mục đó.
