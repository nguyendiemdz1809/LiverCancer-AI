# Bộ minh họa Roboflow chỉ gồm ảnh có tổn thương: CT + MRI

*Sinh tự động bởi `scripts/89_build_roboflow_positive.py` lúc 2026-10-28 15:06 (seed 42). Số liệu: `data/processed/roboflow_positive_20k/summary.json`, `manifest.csv`.*

> **Đây chỉ là bộ minh họa để xem nhãn trên Roboflow, không phải tập Train mới.** Không train lại, không đổi YOLO, Hybrid, checkpoint, Evidence Gate hay split Train/Val/Test. Nhãn **chuyển đổi tự động** từ mask của dataset gốc, không gán tay.

## 1. Tổng quan

| | CT | MRI | Tổng |
|---|---|---|---|
| Ảnh (đều có ≥ 1 box) | 10.000 | 10.000 | 20.000 |
| Bounding box `lesion` | 10.542 | 10.019 | 20.561 |
| Bệnh nhân | 194 | 316 | 510 |
| Volume (pha CT / chuỗi MRI) | 753 | 2.490 | 3.243 |
| Chỉ tiêu | 10.000 | 10.000 | 20.000 |
| Thiếu so với chỉ tiêu | 0 | 0 | 0 |
| Lát có tổn thương sẵn có trong Train (nguồn) | 17.465 | 18.619 | |
| Dung lượng dataset (ảnh + nhãn) | 874.8 MB | 743.3 MB | 1.618.1 MB |
| Dung lượng ZIP | 877.6 MB | 746.2 MB | 1.623.8 MB |


## 2. Nhãn và nguồn dữ liệu

- Danh sách nhãn YOLO: **một lớp duy nhất `0 = lesion`** (tổn thương gan khu trú). File `data.yaml`: `nc: 1`, `names: ['lesion']`.
- Chẩn đoán mức bệnh nhân (HCC, ICC, ...) **không** phải lớp YOLO; chỉ ghi trong `manifest.csv` (local) để thống kê.
- CT: PLC-CECT (Luo et al., Scientific Data 2025) - CC BY 4.0 (attribution required). Ảnh lấy từ `data/processed/full/yolo_ct` (Train).
- MRI: LLD-MMRI (Lou et al.) + MedSAM2 masks (wanglab/LLD-MMRI-MedSAM2) - CC BY-NC 4.0, terms: do not further copy, publish or distribute. Do NOT upload without the owner's decision.. Ảnh lấy từ `data/processed/pilot/yolo_mri` (Train; đã loại series bị cách ly do bbox JSON lệch mask).
- Quy trình gốc: NIfTI 3D → lát 2D PNG xám độ phân giải gốc (`slice.T`, cường độ uint8 như nguồn) → mỗi thành phần liên thông 8 hướng của mask tổn thương (≥ 10 px) thành 1 box YOLO chuẩn hóa. MRI chỉ giữ lát mà mask và `2D_box` JSON gốc đồng thuận (IoU ≥ 0.3). Chi tiết: `reports/roboflow_dataset_report.md` mục 3–4.

| Chẩn đoán (mức BN) | CT: BN | CT: ảnh | CT: box | | Chẩn đoán (mức BN) | MRI: BN | MRI: ảnh | MRI: box |
|---|---|---|---|---|---|---|---|---|
| CHCC | 59 | 2.941 | 3.273 | | FOCAL_NODULAR_HYPERPLASIA | 29 | 963 | 964 |
| HCC | 66 | 4.028 | 4.183 | | Hepatic_abscess | 35 | 1.498 | 1.506 |
| ICC | 69 | 3.031 | 3.086 | | Hepatic_cyst | 34 | 596 | 596 |
| | | | | | Hepatic_hemangioma | 50 | 1.172 | 1.172 |
| | | | | | Hepatic_metastasis | 32 | 917 | 918 |
| | | | | | Hepatocellular_carcinoma | 99 | 3.532 | 3.535 |
| | | | | | Intrahepatic_cholangiocarcinoma | 37 | 1.322 | 1.328 |

## 3. Cách chọn ảnh

Chỉ lấy lát có ≥ 1 box (`kind = pos`) của BN thuộc Train đã đóng băng. Ba mức ưu tiên; mức sau chỉ dùng khi mức trước đã cạn:

| Mức | Điều kiện | CT | MRI |
|---|---|---|---|
| 1 – đa dạng chặt | cách ≥ 4 lát (CT) / ≥ 3 lát (MRI) trong một volume; không gần trùng ảnh khác của cùng BN (dHash ≤ 16/256); ≤ 40 ảnh/BN | 3.924 | 5.291 |
| 2 – nới vừa | cách ≥ 2 lát (CT) / ≥ 1 lát (MRI); chỉ loại ảnh gần như y hệt (dHash ≤ 6/256); ≤ 80 ảnh/BN | 6.076 | 4.709 |
| 3 – bổ sung | mọi lát có tổn thương còn lại (khác lát, không trùng tuyệt đối) | 0 | 0 |

Trong mỗi mức: lần lượt mỗi BN một ảnh mỗi vòng, xoay vòng qua các volume; lát có tổn thương rõ (cạnh ngắn box lớn nhất ≥ 16 px) trước, tổn thương lớn trước trong cùng volume; bỏ ảnh gần như trống (độ lệch chuẩn < 8). Cột `pass` trong `manifest.csv` ghi mức của từng ảnh.

Lưu ý: CT gốc đã lấy mẫu 1/2 lát (2 mm) ở Train, nên ảnh mức 2–3 của CT là các lát cách nhau 2–4 mm của cùng tổn thương; chúng hợp lệ nhưng giống nhau hơn ảnh mức 1.

| Phân bố | CT | MRI |
|---|---|---|
| Tổn thương rõ / nhỏ | 9.939 / 61 | 9.664 / 336 |
| Ảnh theo cạnh ngắn box lớn nhất | <8 px: 0; 8-15 px: 61; 16-31 px: 1.602; 32-63 px: 3.647; >=64 px: 4.690 | <8 px: 16; 8-15 px: 320; 16-31 px: 3.508; 32-63 px: 4.009; >=64 px: 2.147 |
| Ảnh theo số box trong ảnh | 1 box: 9.567; 2 box: 381; 3 box: 27; 4 box: 11; 5 box: 4; 6 box: 4; 7 box: 5; 9 box: 1 | 1 box: 9.981; 2 box: 19 |
| Ảnh mỗi BN (min / trung vị / max) | 7 / 53 / 80 | 3 / 32 / 59 |
| Ảnh mỗi volume (min / trung vị / max) | 1 / 13 / 32 | 1 / 4 / 11 |
| Lượt bị loại khi chọn | gần trùng ảnh khác của cùng BN (mức 1): 7.160; gần như y hệt (mức 2): 2.792 | gần trùng ảnh khác của cùng BN (mức 1): 2.079; gần như y hệt (mức 2): 738 |

Theo pha CT: C1 2.775, C2 2.541, C3 2.573, P 2.111. Theo chuỗi MRI: C+A 1.579, C+Delay 1.496, C+V 1.450, C-pre 1.522, DWI 902, In Phase 1.034, Out Phase 1.012, T2WI 1.005.

## 4. Kiểm tra sau khi xuất

| Kiểm tra | CT | MRI |
|---|---|---|
| Q11. **Mọi ảnh có ≥ 1 box `lesion`** (file nhãn rỗng: CT 0, MRI 0) | PASS | PASS |
| Q1. Ảnh mở được, đúng kích thước | PASS | PASS |
| Q2. Mỗi ảnh có đúng một `.txt` cùng tên, không thừa/thiếu | PASS | PASS |
| Q3. Đúng 5 cột, class ID = 0, box trong [0, 1], w, h > 0 | PASS | PASS |
| Q4. Số box khớp manifest nguồn | PASS | PASS |
| Q5a. Ảnh trùng SHA256 với file nguồn | PASS | PASS |
| Q5b. Nhãn trùng SHA256 với file nguồn | PASS | PASS |
| Q6. Không có ảnh trùng tuyệt đối | PASS | PASS |
| Q9. PNG không chứa metadata văn bản | PASS | PASS |
| Q12. Đối chiếu NIfTI gốc: pixel PNG và box tính lại từ mask trùng khớp (30 / 30 volume, 480 / 122 ảnh) | PASS | PASS |
| Q7. Mọi BN thuộc Train (BN ngoài Train: 0) | PASS | (chung) |
| Q8. Không trùng CT–MRI (SHA256 ảnh 0, ID BN 0) | PASS | (chung) |
| Q10. ZIP: CRC, đủ cặp ảnh–nhãn, các phần không trùng, gộp lại đủ bộ | PASS | PASS |

Thông tin thêm: cặp ảnh gần như y hệt (dHash ≤ 6/256) trong toàn bộ: CT 0, MRI 0 (lát liền nhau của cùng một tổn thương; không phải ảnh trùng).

## 5. File đầu ra

Thư mục: `data/processed/roboflow_positive_20k/` (không đưa vào git).

| ZIP | Ảnh | Box | BN | Dung lượng | SHA256 |
|---|---|---|---|---|---|
| `zips/CT/Roboflow_CT_Positive_10000_part01of10.zip` | 1.000 | 1.028 | 18 | 87.7 MB | `6bb0129c12e1db64…` |
| `zips/CT/Roboflow_CT_Positive_10000_part02of10.zip` | 1.000 | 1.030 | 15 | 88.7 MB | `dc86b4f6816d95f4…` |
| `zips/CT/Roboflow_CT_Positive_10000_part03of10.zip` | 1.000 | 1.070 | 19 | 83.8 MB | `bd8192ca38c643ff…` |
| `zips/CT/Roboflow_CT_Positive_10000_part04of10.zip` | 1.000 | 1.027 | 17 | 93.1 MB | `5f9ba4ebefc3ede6…` |
| `zips/CT/Roboflow_CT_Positive_10000_part05of10.zip` | 1.000 | 1.004 | 22 | 84.1 MB | `f1df8bf57e34ca0e…` |
| `zips/CT/Roboflow_CT_Positive_10000_part06of10.zip` | 1.000 | 1.008 | 22 | 82.4 MB | `6bfb1fc9c596b08a…` |
| `zips/CT/Roboflow_CT_Positive_10000_part07of10.zip` | 1.000 | 1.042 | 27 | 74.8 MB | `d1ef7a05bde4f739…` |
| `zips/CT/Roboflow_CT_Positive_10000_part08of10.zip` | 1.000 | 1.043 | 21 | 90.8 MB | `f7eec06cf0da9cb1…` |
| `zips/CT/Roboflow_CT_Positive_10000_part09of10.zip` | 1.000 | 1.097 | 20 | 95.7 MB | `7125f4421b4ff993…` |
| `zips/CT/Roboflow_CT_Positive_10000_part10of10.zip` | 1.000 | 1.193 | 22 | 96.5 MB | `bf82d4e2e086fd05…` |
| `zips/MRI_offline/Roboflow_MRI_Positive_10000_part01of10.zip` | 1.000 | 1.002 | 28 | 79.4 MB | `d2b7fc0d09dc4142…` |
| `zips/MRI_offline/Roboflow_MRI_Positive_10000_part02of10.zip` | 1.000 | 1.004 | 31 | 78.4 MB | `fabe8003c905ade1…` |
| `zips/MRI_offline/Roboflow_MRI_Positive_10000_part03of10.zip` | 1.000 | 1.000 | 33 | 84.8 MB | `e5f56b1cbbdea7fe…` |
| `zips/MRI_offline/Roboflow_MRI_Positive_10000_part04of10.zip` | 1.000 | 1.001 | 39 | 77.5 MB | `fc0d49ace92f318e…` |
| `zips/MRI_offline/Roboflow_MRI_Positive_10000_part05of10.zip` | 1.000 | 1.003 | 39 | 74.1 MB | `cc7455e76c451dde…` |
| `zips/MRI_offline/Roboflow_MRI_Positive_10000_part06of10.zip` | 1.000 | 1.001 | 34 | 81.7 MB | `ee56f5ab284d70f7…` |
| `zips/MRI_offline/Roboflow_MRI_Positive_10000_part07of10.zip` | 1.000 | 1.000 | 32 | 68.8 MB | `6c33299df255f6df…` |
| `zips/MRI_offline/Roboflow_MRI_Positive_10000_part08of10.zip` | 1.000 | 1.002 | 31 | 63.0 MB | `630cff9c916262cb…` |
| `zips/MRI_offline/Roboflow_MRI_Positive_10000_part09of10.zip` | 1.000 | 1.003 | 28 | 64.4 MB | `dc9c578c5e1a7676…` |
| `zips/MRI_offline/Roboflow_MRI_Positive_10000_part10of10.zip` | 1.000 | 1.003 | 30 | 74.1 MB | `430e2c3fd2ae7396…` |

- Dataset chưa nén: `data/processed/roboflow_positive_20k/Roboflow_CT_Positive_10000/`, `Roboflow_MRI_Positive_10000/` (`data.yaml`, `train/images/`, `train/labels/`).
- `data/processed/roboflow_positive_20k/manifest.csv`: từng ảnh (tên mới ↔ ID BN, pha/chuỗi, lát, số box, mức chọn, ZIP, file nguồn, SHA256). Chỉ lưu local.
- `data/processed/roboflow_positive_20k/upload_manifest.csv`: theo dõi từng ZIP đã upload (cột `uploaded_to_roboflow`, `upload_date`, `roboflow_project`).
- `data/processed/roboflow_positive_20k/previews/`: 100 ảnh vẽ box thật (50 CT + 50 MRI, mỗi ảnh một BN khác nhau) + `contact_sheet_CT.jpg`, `contact_sheet_MRI.jpg`.

## 6. Upload

- **CT** (CC BY 4.0): có thể upload lên Roboflow, ghi nguồn PLC-CECT trong mô tả project. Upload từng ZIP 1.000 ảnh trong `zips/` (`{rel(out)}/zips/CT/`).
- **MRI** (CC BY-NC 4.0, *"not to further copy, publish, or distribute"*): **giữ offline** (`{rel(out)}/zips/MRI_offline/`); không upload khi chưa có quyền phân phối lại.
- Chưa có file nào được upload. Chỉnh sửa box trên Roboflow (nếu có) là hiệu chỉnh bổ sung, phải ghi lại riêng.
