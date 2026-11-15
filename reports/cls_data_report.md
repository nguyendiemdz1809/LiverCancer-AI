# Dữ liệu phân loại tổn thương (ROI) — CT và MRI

*Sinh bởi `scripts/80_build_cls_dataset.py --stages report` lúc 2026-10-25 22:05. Dữ liệu: `data/processed/cls/<mod>/` (`manifest.csv`, `patients.csv`, `summary.json`, `class_map.json`). **Chỉ Train và Val; Test không được xuất.***

## 1. Nhãn đã xác minh

| | CT — PLC-CECT | MRI — LLD-MMRI |
|---|---|---|
| Nguồn nhãn | `patient_data.csv` → `cancer_type` (chẩn đoán mô học/tế bào học, **mức bệnh nhân**) | `LLD_MMRI_Annotation.json` → `lesion.category` (**mức tổn thương**, 1 tổn thương/series) |
| Kiểm tra | 361 BN, mỗi BN đúng 1 nhãn ở cả 4 pha; khớp 100% `data/splits/ct_split.csv` | 498 BN × 8 series; nhãn giống nhau ở cả 8 series của mỗi BN; khớp 100% `mri_split.csv`; `num_targets = 1` ở cả 3.984 series |
| Lớp phân loại | HCC, ICC, CHCC (CN = nhóm chứng không có tổn thương → **không** phải lớp) | Hepatocellular_carcinoma, Intrahepatic_cholangiocarcinoma, Hepatic_metastasis, Hepatic_hemangioma, Hepatic_cyst, Hepatic_abscess, FOCAL_NODULAR_HYPERPLASIA (+ nhóm Lành tính / Ác tính) |
| Gán nhãn cho ROI | mọi tổn thương của BN nhận nhãn chẩn đoán của BN (nhãn yếu ở mức tổn thương) | nhãn của chính tổn thương |
| Không làm | không suy nhãn từ bounding box; không tạo nhãn cho BN CN; không dùng Test | như CT; series bị cách ly (`mri_bbox_quarantine.csv`) không tạo ROI-GT |

## 2. CT — 15055 ROI

| Lớp | Train: BN / ROI-GT | Train: ROI-YOLO khớp GT | Val: BN / ROI-GT | Val: ROI-YOLO (tất cả / khớp GT) |
|---|---|---|---|---|
| HCC | 66 / 2014 | 1743 | 14 / 390 | 862 / 367 |
| ICC | 69 / 2090 | 1953 | 15 / 332 | 959 / 357 |
| CHCC | 59 / 1922 | 1547 | 13 / 324 | 619 / 300 |
| CN | 0 / 0 | 0 | 0 / 0 | 300 / 0 |

## 3. MRI — 31254 ROI

| Lớp | Train: BN / ROI-GT | Train: ROI-YOLO khớp GT | Val: BN / ROI-GT | Val: ROI-YOLO (tất cả / khớp GT) |
|---|---|---|---|---|
| Hepatocellular_carcinoma | 99 / 3833 | 3638 | 25 / 919 | 2136 / 878 |
| Intrahepatic_cholangiocarcinoma | 37 / 1467 | 1385 | 9 / 332 | 937 / 291 |
| Hepatic_metastasis | 32 / 1100 | 1028 | 8 / 241 | 1170 / 199 |
| Hepatic_hemangioma | 50 / 1380 | 1296 | 13 / 338 | 817 / 250 |
| Hepatic_cyst | 34 / 842 | 649 | 8 / 190 | 479 / 98 |
| Hepatic_abscess | 35 / 1532 | 1583 | 8 / 310 | 731 / 315 |
| FOCAL_NODULAR_HYPERPLASIA | 29 / 1076 | 992 | 7 / 245 | 608 / 189 |

## Cách tạo ROI

- **ROI-GT:** box GT đã xuất ở GĐ2 (đã QA, ảnh trùng từng pixel với NIfTI) được nối thành tổn thương 3D bằng cùng thuật toán của Tumor Zoom; mỗi tổn thương lấy tối đa 6 (Train) / 5 (Val) lát có diện tích lớn nhất (CT: các lát cách nhau ≥ 2). Hai ảnh 224×224 uint8: **local** (box + lề 15%, ≥ 8 px) và **context** (2.5× cạnh box, ≥ 128 px).
- **ROI-YOLO (Val):** mô phỏng pipeline thật: YOLO (conf 0.12) → gộp box → track 3D (configs/roi.yaml) → mọi track được giữ, kể cả track giả và track trên BN CN; cột `matches_gt` (IoU ≥ 0.3 với một tổn thương GT) chỉ dùng để phân tích.
- **ROI-YOLO (Train, tùy chọn):** chỉ giữ track khớp GT, nên nhãn của BN/tổn thương áp đúng vào tổn thương thật. YOLO đã thấy các ảnh này khi train, nên đây là dữ liệu lạc quan.
- **Không có ROI:** BN/volume mà YOLO không tạo track nào sẽ không có dòng ROI-YOLO. Khi đánh giá pipeline, những BN này bị tính là **sai** (`pipeline_patient` trong đánh giá).
- **Chuẩn hóa cho backbone pretrained:** ảnh xám → 3 kênh giống nhau → [0, 1] → chuẩn hóa mean/std ImageNet (`livercancer.classification.data.to_tensor`).

