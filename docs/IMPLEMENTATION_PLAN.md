# Kế hoạch triển khai

Mỗi giai đoạn chỉ được coi là xong khi **đạt toàn bộ tiêu chí hoàn thành (✅)** và người dùng đã kiểm tra.
Ở mọi giai đoạn: không sửa dữ liệu gốc, split ở mức bệnh nhân, không dùng test để tuning.

---

## GĐ0 — Thiết lập (lượt hiện tại)

**Công việc**
- Khảo sát project, môi trường, dataset (kết quả ở `docs/DATASET_NOTES.md`).
- Tạo khung thư mục, `CLAUDE.md`, `configs/paths.yaml`, `requirements*.txt`.
- `scripts/00_check_env.py`, `scripts/01_verify_datasets.py`.
- `.venv` Python 3.12 + PyTorch CUDA.

**✅ Hoàn thành khi**
- `00_check_env.py` báo `cuda_available = True`, VRAM khoảng 4 GB, đủ thư viện.
- `01_verify_datasets.py` xuất `reports/dataset_audit.json`, khớp với số liệu đã khảo sát (361 / 498 bệnh nhân, 40 CT thiếu, 0 file thừa).

---

## GĐ1 — Kiểm tra & chuẩn hóa dữ liệu

**1.1 Xác minh nguồn** (được phép đọc web/GitHub, không tải dataset)
- PLC-CECT: định nghĩa chính xác của `CN`, `CHCC`; ý nghĩa của các pha `C1/C2/C3/P`; nguồn gốc và bài báo của dataset.
- LLD-MMRI: split train/val/test chính thức (repo `LMMMEng/LLD-MMRI-Dataset`), license và điều khoản sử dụng, và việc mapping ID bệnh nhân giữa bản gốc và bản MedSAM2 có khớp không.
- Ghi kết quả kèm trích dẫn và link vào `docs/DATASET_NOTES.md`.

**1.2 Manifest** (`scripts/10_ct_manifest.py`, `scripts/11_mri_manifest.py`)
- Mỗi dòng ứng với một volume: patient_id, label, phase/sequence, đường dẫn ảnh/mask, file_exists, shape, spacing, orientation (đọc từ header).
- CT: loại 40 dòng thiếu file, ghi log vào `reports/ct_missing_files.csv`.
- MRI: map category id sang tên lớp theo `Category_info`; thêm cột benign/malignant.

**1.3 Audit hình học**
- Ảnh và mask có cùng shape/affine không. Mask tổn thương có nằm trong liver mask không (CT).
- Thống kê: số lát cắt có tổn thương, kích thước tổn thương, spacing.
- **Overlay trực quan khoảng 20 mẫu mỗi modality** (`reports/figures/overlay_*`) để chốt hệ trục `2D_box` của MRI và hướng lát cắt CT.

**1.4 Split** (`scripts/12_make_splits.py`)
- CT: `StratifiedShuffleSplit` theo bệnh nhân, 70/15/15, theo `cancer_type` 4 lớp (giữ CN cân bằng giữa các split), seed 42.
- MRI: split chính thức nếu xác minh được, nếu không thì 70/15/15 phân tầng theo 7 lớp.
- Lưu `data/splits/{ct,mri}_split.csv` và đóng băng.

**1.5 Xuất lát cắt 2D** (`scripts/20_export_slices.py`, streaming)
- CT: dữ liệu đã là `uint8` (không phải HU), nên **không dùng HU window**. Phân tích histogram từng pha trên train, rồi chọn giữ nguyên 0–255 hoặc chuẩn hóa percentile. Lấy mẫu lát cắt trong phạm vi z của gan, kèm lát cắt âm tính.
- MRI: `uint8`, chuẩn hóa percentile 0.5–99.5 cho từng volume (chốt sau khi xem histogram).
- Định dạng PNG 8-bit (cho YOLO) và/hoặc NPZ. Kèm manifest lát cắt (patient, slice_idx, split, có/không tổn thương).

**✅ Hoàn thành khi**
- Mọi mismatch trong audit đều có giải thích.
- `pytest tests/test_splits_no_leakage.py` pass.
- Người dùng đã duyệt ảnh overlay.
- Có bảng phân bố lớp × split (bệnh nhân và lát cắt) trong `reports/tables/`.

---

## GĐ2 — YOLO detection

- Sinh nhãn YOLO (`scripts/30_build_yolo_ds.py`):
  - CT: connected components của mask tổn thương cho ra bbox lớp `lesion`; liver mask cho ra bbox lớp `liver`.
  - MRI: `2D_box` cho ra bbox lớp `lesion` (sau khi đã chốt hệ trục ở GĐ1).
- Lát cắt không có tổn thương và lát cắt CN được dùng làm background (không có nhãn lesion).
- Local: smoke test (`yolo11n`, imgsz 512, 1–2 epoch, một phần nhỏ dữ liệu).
- Colab: train chính, chọn model size / imgsz / conf threshold **trên val**.

**✅ Hoàn thành khi**
- Có mAP@0.5, mAP@0.5:0.95, Precision, Recall trên val và test (test chạy 1 lần).
- Có sensitivity ở mức tổn thương và mức bệnh nhân; tỉ lệ bệnh nhân CN bị phát hiện nhầm có tổn thương.
- Weights, config và log được lưu trong `outputs/runs/yolo_{ct,mri}/`.

---

## GĐ3 — Automatic Lesion ROI Zoom

- Từ bbox **dự đoán**: lọc theo conf, NMS, gộp qua các lát cắt liền kề, chọn top-k lát cắt theo diện tích × confidence.
- Crop local: bbox + margin 10–20%, resize 224. Crop global: context ×2–3 quanh tâm bbox, resize 224.
- Train/val classifier: crop từ GT (có thể trộn thêm crop từ YOLO để giảm domain gap). **Test: chỉ dùng crop từ YOLO.**

**✅ Hoàn thành khi**
- Có dataset ROI kèm manifest.
- Có thống kê số ca YOLO bỏ sót và IoU trung bình giữa ROI dự đoán và GT trên val.

---

## GĐ4 — Hybrid CNN + Transformer

- Nhánh local: ResNet18/34 (timm, pretrained ImageNet) trên crop local.
- Nhánh global: Swin-T / ViT-S trên crop global.
- Fusion: concat + MLP (mặc định); gated fusion / cross-attention làm ablation.
- Loss: CE có class weights (hoặc focal). AMP, gradient accumulation, early stopping theo macro-F1 trên val.
- Số lớp: CT 3 lớp (HCC/ICC/CHCC); MRI 7 lớp, kèm báo cáo phụ benign/malignant.
- Dự đoán mức bệnh nhân: tổng hợp (trung bình xác suất) qua lát cắt, pha và chuỗi.

**✅ Hoàn thành khi**
- Loss hội tụ, có checkpoint tốt nhất theo val, đầy đủ log, config và seed.

---

## GĐ5 — Đánh giá & so sánh

- Baseline: ResNet50 đơn, ViT/Swin đơn, Hybrid.
- Ablation: có/không ROI zoom, kiểu fusion, một nhánh so với hai nhánh.
- Hai chế độ báo cáo **tách riêng**: end-to-end (YOLO → ROI → cls) và oracle-ROI (gắn nhãn "oracle").
- Metric: accuracy, balanced accuracy, macro-F1, P/R/F1 từng lớp, AUC OvR, confusion matrix, 95% CI bằng bootstrap mức bệnh nhân.
- XAI: Grad-CAM (nhánh CNN), attention rollout (nhánh Transformer).

**✅ Hoàn thành khi**
- Có bảng so sánh kèm CI trong `reports/tables/`, hình XAI trong `reports/figures/`.
- Mỗi cấu hình cuối chỉ chạy test một lần.

---

## GĐ6 — Streamlit demo

- Upload NIfTI (CT hoặc MRI), chọn lát cắt, chạy YOLO, xem ROI zoom, dự đoán kèm xác suất, xem Grad-CAM.
- Chạy local trên GPU 4 GB, fp16, fallback sang CPU.
- Disclaimer: "Chỉ phục vụ nghiên cứu, không dùng cho chẩn đoán."

**✅ Hoàn thành khi**
- Chạy end-to-end được trên ca thuộc test set; độ trễ được đo và ghi lại.

---

## GĐ7 — Tổng hợp báo cáo

- Bảng, hình, sơ đồ kiến trúc, mô tả dữ liệu, quy trình thí nghiệm.
- Phần hạn chế: 40 CT thiếu, CN không có mask, annotation MRI bán tự động (MedSAM2 human-in-the-loop), dữ liệu đơn trung tâm, ...

**✅ Hoàn thành khi**
- Mọi con số trong báo cáo đều truy được về file trong `outputs/`.
