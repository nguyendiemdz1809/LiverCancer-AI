# Pilot dataset report — GĐ2a

*Sinh tự động bởi `scripts/40_pack_pilot.py` lúc 2026-10-19 16:02.* Pilot chỉ gồm **Train + Val**. **Test được giữ riêng**, không nằm trong ZIP.

## 1. Bệnh nhân theo split (toàn bộ dataset, split đóng băng)

| Modality | Train | Val | Test | Tổng |
|---|---|---|---|---|
| CT | 253 | 54 | 54 | 361 |
| MRI | 316 | 78 | 104 | 498 |

Chi tiết theo lớp: `reports/split_report.md`. Kiểm tra rò rỉ: `tests/test_splits_no_leakage.py` và QA check C4 (không BN nào thuộc hai split, không có BN test trong pilot).

## 2a. CT — `yolo_ct_pilot.zip`

- Ảnh: **25196**, file nhãn: **25196**, box: **13247**; lát cắt theo split và loại: {'train_neg': 8897, 'train_pos': 8917, 'val_neg': 3691, 'val_pos': 3691}
- BN có trong pilot: {'train': 253, 'val': 54}
- Lấy mẫu: lấy 1/4 số lát dương tính ở train, 1/2 ở val; tỉ lệ âm:dương = 1.0; lát âm cách lát có annotation ≥ 2 lát; mảnh mask < 10 px bị bỏ (19 mảnh)
- Trạng thái volume (train+val): {'ok': 1194, 'excluded:ct_file_missing': 31, 'excluded:lesion_mask_empty': 1, 'excluded:lesion_mask_shape_mismatch': 1, 'excluded:source_file_corrupt': 1}
- Lát bị bỏ (không là dương tính, cũng không là âm tính): skipped_all_fragments = 2, skipped_no_json_box = 0, skipped_low_json_iou = 0
- Volume RGB 5D có kênh khác nhau chỉ ở chữ thước đo in chồng (ngoài gan/tổn thương, dùng kênh 0): 62
- Dung lượng ảnh+nhãn: 2.11 GB; ZIP: **2.12 GB** (50399 entries)
- QA (C1–C6): C1 0 lỗi, C2 0 lỗi, C3 0 lỗi, C4 0 lỗi, C5 0 lỗi, C6 0 lỗi; đối chiếu NIfTI gốc trên 40 volume / 813 lát

Phân bố (lớp là chẩn đoán mức bệnh nhân, chỉ dùng để truy vết; YOLO chỉ có 1 lớp `lesion`):

| Split | Lớp (BN) | BN | Lát dương | Lát âm | Box |
|---|---|---|---|---|---|
| train | CHCC | 59 | 2116 | 1474 | 2350 |
| train | CN | 59 | 0 | 3219 | 0 |
| train | HCC | 66 | 3605 | 1027 | 3760 |
| train | ICC | 69 | 3196 | 3177 | 3287 |
| val | CHCC | 13 | 841 | 704 | 904 |
| val | CN | 12 | 0 | 1444 | 0 |
| val | HCC | 14 | 1425 | 321 | 1496 |
| val | ICC | 15 | 1425 | 1222 | 1450 |

## 2b. MRI — `yolo_mri_pilot.zip`

- Ảnh: **36086**, file nhãn: **36086**, box: **23541**; lát cắt theo split và loại: {'train_neg': 9976, 'train_pos': 18619, 'val_neg': 2617, 'val_pos': 4874}
- BN có trong pilot: {'train': 316, 'val': 78}
- Lấy mẫu: lấy 1/1 số lát dương tính ở train, 1/1 ở val; tỉ lệ âm:dương = 0.5; lát âm cách lát có annotation ≥ 2 lát; mảnh mask < 10 px bị bỏ (148 mảnh)
- Trạng thái volume (train+val): {'ok': 3125, 'quarantined:json_box_vs_mask_mismatch': 27}
- Lát bị bỏ (không là dương tính, cũng không là âm tính): skipped_all_fragments = 4, skipped_no_json_box = 1586, skipped_low_json_iou = 121

- Dung lượng ảnh+nhãn: 2.66 GB; ZIP: **2.68 GB** (72179 entries)
- QA (C1–C6): C1 0 lỗi, C2 0 lỗi, C3 0 lỗi, C4 0 lỗi, C5 0 lỗi, C6 0 lỗi; đối chiếu NIfTI gốc trên 40 volume / 412 lát

- Mỗi lát MRI dương tính đều phải có JSON 2D_box gốc đồng thuận với box từ mask (IoU ≥ 0.3). Lát không đồng thuận bị bỏ (xem dòng trên), không sửa nhãn. Trên các lát được giữ: IoU p5/p25/p50 = [0.596, 0.791, 0.852] (23493 lát).

Phân bố (lớp là chẩn đoán mức bệnh nhân, chỉ dùng để truy vết; YOLO chỉ có 1 lớp `lesion`):

| Split | Lớp (BN) | BN | Lát dương | Lát âm | Box |
|---|---|---|---|---|---|
| train | FOCAL_NODULAR_HYPERPLASIA | 29 | 1775 | 1030 | 1779 |
| train | Hepatic_abscess | 35 | 3184 | 1016 | 3195 |
| train | Hepatic_cyst | 34 | 948 | 1156 | 948 |
| train | Hepatic_hemangioma | 50 | 1732 | 1549 | 1732 |
| train | Hepatic_metastasis | 32 | 1530 | 1263 | 1531 |
| train | Hepatocellular_carcinoma | 99 | 7010 | 2892 | 7019 |
| train | Intrahepatic_cholangiocarcinoma | 37 | 2440 | 1070 | 2457 |
| val | FOCAL_NODULAR_HYPERPLASIA | 7 | 429 | 301 | 429 |
| val | Hepatic_abscess | 8 | 609 | 216 | 610 |
| val | Hepatic_cyst | 8 | 210 | 297 | 210 |
| val | Hepatic_hemangioma | 13 | 443 | 453 | 444 |
| val | Hepatic_metastasis | 8 | 361 | 319 | 361 |
| val | Hepatocellular_carcinoma | 25 | 2183 | 731 | 2187 |
| val | Intrahepatic_cholangiocarcinoma | 9 | 639 | 300 | 639 |

## 3. Dữ liệu lỗi / bị cách ly (không vào training)

| Modality | Loại | Số volume/series (train+val) |
|---|---|---|
| CT | excluded:ct_file_missing | 31 |
| CT | excluded:lesion_mask_empty | 1 |
| CT | excluded:lesion_mask_shape_mismatch | 1 |
| CT | excluded:source_file_corrupt | 1 |
| MRI | quarantined:json_box_vs_mask_mismatch | 27 |

**Tập Test (không đóng gói), kiểm tra gzip toàn vẹn** (`scripts/21_check_source_integrity.py`): 2264 file; CT missing: 9, CT ok: 591, MRI ok: 1664

Danh sách đầy đủ: `volumes.csv` trong mỗi ZIP; MRI cách ly: `data/quarantine/mri_bbox_quarantine.csv`.

## 4. File đầu ra

| File | Dung lượng | SHA256 |
|---|---|---|
| `data/processed/pilot/yolo_ct_pilot.zip` | 2.117 GB | `0f4c59c5e5d9375dfe994a7a81a3884d7a5e6d6a9ca929970436e24bf89c6fa3` |
| `data/processed/pilot/yolo_mri_pilot.zip` | 2.677 GB | `0088fa93b294887aa71f226790aee0b5a96ac3cbd9567a57c8a774f045cd9f2f` |
| `data/processed/pilot/split_manifest.csv` | 0.000 GB | `dd9f1fa1d6175d130dfc59a789226600cda37f486eaa7f1ff39970c20dc10dc1` |

## 5. Cách dùng

1. Upload `yolo_ct_pilot.zip`, `yolo_mri_pilot.zip`, `SHA256SUMS.txt` vào `MyDrive/LiverCancer_AI/data/` (thư mục riêng tư).
2. Mở `notebooks/03_train_yolo_ct_colab.ipynb` / `03_train_yolo_mri_colab.ipynb` trên Colab (Runtime → GPU) và chạy từ trên xuống.
3. Kết quả (best.pt, last.pt, biểu đồ, metrics JSON) lưu ở `MyDrive/LiverCancer_AI/runs/`.

