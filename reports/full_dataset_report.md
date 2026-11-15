# Full dataset report — GĐ3 (train chính thức)

*Sinh tự động bởi `scripts/40_pack_pilot.py --set full` lúc 2026-10-20 21:07.* Bộ dữ liệu chỉ gồm **Train + Val**. **Test được giữ riêng**, không nằm trong ZIP.

## 1. Bệnh nhân theo split (toàn bộ dataset, split đóng băng)

| Modality | Train | Val | Test | Tổng |
|---|---|---|---|---|
| CT | 253 | 54 | 54 | 361 |
| MRI | 316 | 78 | 104 | 498 |

Chi tiết theo lớp: `reports/split_report.md`. Kiểm tra rò rỉ: `tests/test_splits_no_leakage.py` và QA check C4 (không BN nào thuộc hai split, không có BN test trong pilot).

## 2a. CT — `yolo_ct_full.zip`

- Ảnh: **49500**, file nhãn: **49500**, box: **26057**; lát cắt theo split và loại: {'train_neg': 17408, 'train_pos': 17465, 'val_neg': 7313, 'val_pos': 7314}
- BN có trong dataset: {'train': 253, 'val': 54}
- Lấy mẫu: lấy 1/2 số lát dương tính ở train, 1/1 ở val; tỉ lệ âm:dương = 1.0; lát âm cách lát có annotation ≥ 2 lát; mảnh mask < 10 px bị bỏ (45 mảnh)
- Trạng thái volume (train+val): {'ok': 1194, 'excluded:ct_file_missing': 31, 'excluded:lesion_mask_empty': 1, 'excluded:lesion_mask_shape_mismatch': 1, 'excluded:source_file_corrupt': 1}
- Lát bị bỏ (không là dương tính, cũng không là âm tính): skipped_all_fragments = 4, skipped_no_json_box = 0, skipped_low_json_iou = 0
- Volume RGB 5D có kênh khác nhau chỉ ở chữ thước đo in chồng (ngoài gan/tổn thương, dùng kênh 0): 62
- Dung lượng ảnh+nhãn: 4.15 GB; ZIP: **4.16 GB** (99007 entries)
- QA (C1–C7): C1 0 lỗi, C2 0 lỗi, C3 0 lỗi, C4 0 lỗi, C5 0 lỗi, C6 0 lỗi, C7 0 lỗi; đối chiếu NIfTI gốc trên 60 volume / 2550 lát; C3 đọc 49500 ảnh

Độ phủ bệnh nhân (QA C7: không BN Train/Val nào có volume dùng được bị mất do lấy mẫu lát):

| Split | Lớp | BN trong split | BN có volume dùng được | BN có trong dataset |
|---|---|---|---|---|
| train | CHCC | 59 | 59 | 59 |
| train | CN | 59 | 59 | 59 |
| train | HCC | 66 | 66 | 66 |
| train | ICC | 69 | 69 | 69 |
| val | CHCC | 13 | 13 | 13 |
| val | CN | 12 | 12 | 12 |
| val | HCC | 14 | 14 | 14 |
| val | ICC | 15 | 15 | 15 |

Phân bố (lớp là chẩn đoán mức bệnh nhân, chỉ dùng để truy vết; YOLO chỉ có 1 lớp `lesion`):

| Split | Lớp (BN) | BN | Lát dương | Lát âm | Box |
|---|---|---|---|---|---|
| train | CHCC | 59 | 4113 | 2888 | 4581 |
| train | CN | 59 | 0 | 6289 | 0 |
| train | HCC | 66 | 7093 | 1994 | 7395 |
| train | ICC | 69 | 6259 | 6237 | 6450 |
| val | CHCC | 13 | 1658 | 1398 | 1781 |
| val | CN | 12 | 0 | 2856 | 0 |
| val | HCC | 14 | 2831 | 641 | 2970 |
| val | ICC | 15 | 2825 | 2418 | 2880 |

## 3. Dữ liệu lỗi / bị cách ly (không vào training)

| Modality | Loại | Số volume/series (train+val) |
|---|---|---|
| CT | excluded:ct_file_missing | 31 |
| CT | excluded:lesion_mask_empty | 1 |
| CT | excluded:lesion_mask_shape_mismatch | 1 |
| CT | excluded:source_file_corrupt | 1 |

**Tập Test (không đóng gói), kiểm tra gzip toàn vẹn** (`scripts/21_check_source_integrity.py`): 2264 file; CT missing: 9, CT ok: 591, MRI ok: 1664

Danh sách đầy đủ: `volumes.csv` trong mỗi ZIP; MRI cách ly: `data/quarantine/mri_bbox_quarantine.csv`.

## 4. File đầu ra

| File | Dung lượng | SHA256 |
|---|---|---|
| `data/processed/full/yolo_ct_full.zip` | 4.163 GB | `0a7ed974856f5634b2860b692b83fa717e820031d8974862607a78a5e281cfdc` |
| `data/processed/full/split_manifest.csv` | 0.000 GB | `dd9f1fa1d6175d130dfc59a789226600cda37f486eaa7f1ff39970c20dc10dc1` |

## 5. Cách dùng

1. Upload `yolo_ct_full.zip`, `SHA256SUMS_full.txt` vào `MyDrive/LiverCancer_AI/data/` (thư mục riêng tư).
2. Mở `notebooks/04_train_yolo_ct_full_colab.ipynb` trên Colab (Runtime → GPU) (kiểm tra `ZIP_NAME` ở mục 2: CT mặc định `yolo_ct_full.zip`), rồi chạy từ trên xuống.
3. Kết quả (best.pt, last.pt, biểu đồ, metrics JSON) lưu ở `MyDrive/LiverCancer_AI/runs/`.

## 6. Kiểm tra ZIP trước khi upload (`scripts/42_verify_zip.py`)

| Kiểm tra | Kết quả |
|---|---|
| SHA256 của ZIP = dòng trong `SHA256SUMS_full.txt` | ✅ `0a7ed974856f5634b2860b692b83fa717e820031d8974862607a78a5e281cfdc` |
| `testzip` (CRC mọi entry) | ✅ |
| Giải nén toàn bộ; số file giải nén = số file nguồn = số entry | ✅ 99007 / 99007 / 99007 |
| Thiếu / thừa / lệch kích thước so với nguồn | 0 / 0 / 0 |
| So từng byte trên mẫu ngẫu nhiên | 500 file, 0 khác |
| Ảnh / nhãn trong ZIP | 49500 / 49500 |
| Có `images/test` hoặc `labels/test` | ✅ không |
| Thư mục giải nén tạm đã xóa | ✅ |
| **Kết luận** | **PASS** (`data/processed/full/yolo_ct_full.verify.json`) |

## 7. Tóm tắt tổn thương
- **Box tổn thương** (thành phần liên thông 2D trên mỗi lát; một khối u xuất hiện trên nhiều lát và nhiều pha):
  train 18.426 (CHCC 4.581 · HCC 7.395 · ICC 6.450), val 7.631 (CHCC 1.781 · HCC 2.970 · ICC 2.880); tổng **26.057**.
- **BN có tổn thương**: train 194 (CHCC 59 · HCC 66 · ICC 69), val 42 (CHCC 13 · HCC 14 · ICC 15). BN CN (train 59, val 12)
  chỉ đóng góp lát âm tính; lát âm nghĩa là "không có annotation tổn thương", **không** phải bệnh nhân khỏe mạnh.
- Val CT đầy đủ (7.314 lát dương tính = 100% số lát có tổn thương của 54 BN Val), nên metrics của các run full **không so trực tiếp** với CT pilot
  (Val pilot chỉ có 1/2).

