# Giai đoạn 5: phân loại tổn thương gan bằng CNN, Transformer và Hybrid

> Trạng thái (2026-10-26): **4 run đã huấn luyện trên Colab** (CT resnet18 / swin_t / hybrid, MRI hybrid; `outputs/runs/cls/`), đã kiểm tra
> và tích hợp vào pipeline YOLO → Tumor Zoom → phân loại (`reports/pipeline_val_report.md`, `docs/PIPELINE_VSCODE.md`). MRI chưa đạt chất lượng.
> **Chưa huấn luyện chính thức.** Người dùng tự chạy trên Colab. Không cam kết accuracy; chỉ báo cáo số đo được. Test bị khóa.

## 1. Dữ liệu và nhãn (chi tiết: `reports/cls_data_report.md`)
| | CT — PLC-CECT | MRI — LLD-MMRI |
|---|---|---|
| Lớp | HCC · ICC · CHCC (3) | HCC · ICC · Di căn · U máu · Nang · Áp xe · FNH (7), kèm Lành tính / Ác tính |
| Nhãn | chẩn đoán **mức bệnh nhân** (mô học), áp cho mọi ROI của BN | **mức tổn thương** (annotation gốc), 1 tổn thương/series |
| CN | nhóm chứng, không có tổn thương → **không** phải lớp; chỉ xuất hiện như ROI-YOLO giả trên Val | — |
| Split | split BN đã khóa (`data/splits/ct_split.csv`) | split BN đã khóa (Val = 78 ID chính thức) |
| Hạn chế | nhãn ROI là nhãn yếu (theo BN); Val chỉ 13–15 BN/lớp | lớp nhỏ: Val chỉ 7–9 BN cho FNH, áp xe, nang, di căn, ICC → chỉ số theo lớp dao động lớn |

ROI (`scripts/80_build_cls_dataset.py`, dùng lại Tumor Zoom ở `src/livercancer/roi/`):
- Train: ROI-GT (baseline). Tùy chọn thêm ROI-YOLO khớp GT (`roi_source: gt+yolo`).
- Val: ROI-GT **và** ROI-YOLO (pipeline thật, gồm cả vùng giả và BN không có ROI).
- Mỗi ROI gồm hai ảnh 224×224: `local` (CNN) và `context` (Transformer), lưu PNG uint8. Chuẩn hóa ImageNet khi nạp.

## 2. Mô hình (`src/livercancer/classification/models/`)
| Tên | Kiến trúc | Đầu vào | Tham số |
|---|---|---|---|
| `resnet18` | ResNet18 ImageNet → Dropout → Linear | local | 11.2 M |
| `swin_t` | Swin-T ImageNet → Dropout → Linear | context | 27.5 M |
| `hybrid` | ResNet18 (local) + Swin-T (context) → chiếu về 256 chiều + LayerNorm → **fusion** (concat + MLP, hoặc gated) → head | cả hai | khoảng 39 M |

Huấn luyện (`src/livercancer/classification/train.py`):
- AdamW, LR backbone 1e-4, LR head 1e-3, warmup 1 epoch rồi cosine.
- AMP; class weight 1/√n; label smoothing 0.05.
- Early stopping theo **macro-F1 ROI trên Val** (patience 8).
- Lưu `last.pt` mỗi epoch (resume); khi CUDA hết bộ nhớ thì tự giảm batch một nửa.

## 3. File upload lên Google Drive (`MyDrive/LiverCancer_AI/data/`)
| File local | Dung lượng | Nội dung | SHA256 |
|---|---|---|---|
| `E:\LiverCancer_AI_Graduation\data\processed\cls\cls_ct.zip` | 0.86 GB (858.980.399 byte), 30.114 file | 15.055 ROI CT (Train + Val), manifest, class map | `2f2e52ba…b81d3` |
| `E:\LiverCancer_AI_Graduation\data\processed\cls\cls_mri.zip` | 1.54 GB (1.535.290.729 byte), 62.512 file | 31.254 ROI MRI (Train + Val). **CC BY-NC 4.0**: thư mục Drive riêng tư | `9bf16658…35a0` |
| `E:\LiverCancer_AI_Graduation\data\processed\cls\livercancer_code.zip` | 37 KB | code `src/livercancer` + `configs/cls.yaml`, `roi.yaml`, `paths.yaml` | `d580bcbc…63c7` |
| `E:\LiverCancer_AI_Graduation\data\processed\cls\SHA256SUMS_cls.txt` | < 1 KB | SHA256 đầy đủ của 3 ZIP (notebook kiểm tra) | |

Đã kiểm tra bằng `scripts/84_verify_cls_zip.py` (2026-10-26): **ALL OK** (`data/processed/cls/verify_cls.json`):
- SHA256 khớp; CRC mọi entry hợp lệ; số entry bằng số file nguồn;
- 0 ảnh thiếu so với manifest; 300 file mẫu trùng từng byte; không có Test;
- 0 BN nằm ở hai split; nhãn khớp 100% split đã khóa;
- ZIP code import và dựng được cả 3 mô hình khi giải nén riêng.

Nếu sửa `src/` sau ngày này, phải chạy lại `80_build_cls_dataset.py --stages pack` rồi `84_verify_cls_zip.py` và upload lại `livercancer_code.zip` cùng `SHA256SUMS_cls.txt`.

## 3b. Cấu hình đề xuất cho GPU T4 (16 GB)
| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `BATCH` | **32** | Hybrid batch 32 + AMP chạy được cả trên GPU 4 GB local (đã thử), nên T4 dư bộ nhớ. Nếu OOM: trainer tự giảm 16 → 8 và tăng accumulation |
| `EPOCHS` / `PATIENCE` | 40 / 8 | early stopping theo macro-F1 ROI trên Val |
| AMP | bật | fp16 autocast + GradScaler |
| LR backbone / head | 1e-4 / 1e-3 | AdamW, weight decay 0.05, warmup 1 epoch + cosine |
| `num_workers` | 2 | Colab có 2 CPU |
| `ROI_SOURCE` | `gt` | baseline; `gt+yolo` là biến thể (thêm ROI-YOLO khớp GT ở Train) |
| `SEED` | 42 | các seed khác chỉ dùng cho cấu hình cuối |
| Thời gian (ước lượng thô từ GPU local, **chưa đo trên T4**) | CT khoảng 2 phút/epoch, MRI khoảng 4 phút/epoch cho hybrid | ResNet18 nhanh hơn; Swin-T ở giữa |

Thứ tự gợi ý mỗi modality: `SMOKE=True` → `resnet18` → `swin_t` → `hybrid`, tất cả `ROI_SOURCE='gt'`, seed 42.

## 4. Chạy trên Colab: `notebooks/05_train_classifier_colab.ipynb`
1. Mở notebook, chọn *Runtime → T4 GPU*.
2. Mục 2: `MODALITY = 'ct'`, `MODEL = 'resnet18'`, `SMOKE = True` → chạy hết một lượt để kiểm tra (vài phút, ghi vào `/content`).
3. Đặt `SMOKE = False`, chạy lần lượt `MODEL = 'resnet18'` → `'swin_t'` → `'hybrid'` (mỗi run một `RUN_NAME`, lưu ở `MyDrive/LiverCancer_AI/runs/cls/`).
4. Lặp lại cho `MODALITY = 'mri'`.
5. Bị ngắt: chạy lại notebook với cùng cấu hình; huấn luyện tiếp tục từ `last.pt`.
6. Tải `<RUN_NAME>_bundle.zip` về `outputs\runs\cls\` để phân tích ở local (`scripts/82_eval_classifier.py`).

## 5. Đánh giá (`src/livercancer/classification/evaluate.py`, `metrics.py`)
- Chỉ số: Accuracy, Balanced accuracy, Precision, Recall/Sensitivity, Specificity (one-vs-rest), macro-F1,
  confusion matrix, ROC-AUC (OvR, khi đủ lớp).
- **Mức ROI** và **mức bệnh nhân**. Cách gộp ROI → BN (`mean_prob`, `conf_weighted`, `majority_vote`, `top3_mean`)
  được **chọn trên Val bằng ROI-YOLO** (đầu vào thật của pipeline).
- Pipeline: BN không có ROI-YOLO nào bị tính là sai (`accuracy_counting_missing_as_wrong`).
  BN CN có ROI-YOLO sẽ nhận một nhãn ung thư, vì mô hình không có lớp "không tổn thương". Đây là hạn chế cần ghi trong báo cáo.
- Chọn mô hình cuối (CNN, Transformer hay Hybrid) **chỉ trên Val**.
- **Test:** chỉ chạy **một lần**, cho toàn bộ pipeline YOLO → Tumor Zoom → phân loại, sau khi chốt mọi lựa chọn.
  `evaluate_test()` từ chối nếu thiếu xác nhận, và có lock để không chạy lần hai.

## 6. Kết nối website (chưa sửa website)
`livercancer.classification.inference`:
- `LesionClassifier.load(modality)` đọc `configs/cls.yaml → inference.<mod>.checkpoint`.
- `rois_from_roi_result(RoiResult)` lấy crop từ Tumor Zoom.
- `classify()` trả về `ClassificationResult(status, message, label, label_vi, scores, per_track, per_roi)`.
- Các trạng thái: `ok` · `no_roi` · `model_unavailable` (chưa có checkpoint, như hiện tại) · `invalid_input` · `error`. Không có nhãn khi trạng thái khác `ok`.
- Sau khi train xong: điền `checkpoint` và `aggregation` (lấy từ `eval_val/metrics_val.json`) vào `configs/cls.yaml`, rồi nối trang "Phân loại ung thư" của website.
