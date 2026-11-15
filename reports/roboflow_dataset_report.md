# Bộ dữ liệu YOLO cho Roboflow: CT 5.000 + MRI 5.000 ảnh

*Sinh tự động bởi `scripts/88_build_roboflow_export.py` lúc 2026-10-27 23:21 (seed 42). Mọi con số lấy từ `data/processed/roboflow/meta/summary.json`.*

## 1. Mục đích và nguồn gốc nhãn

Hai bộ dữ liệu này dùng để **nhập vào Roboflow nhằm trực quan hóa, kiểm tra và trình bày nhãn** trước hội đồng. Chúng không dùng để huấn luyện lại mô hình.

- **Nhãn không được gán thủ công trên Roboflow.** Mọi bounding box được **chuyển đổi tự động** từ mask tổn thương của dataset gốc (CT: mask tổn thương của PLC-CECT; MRI: mask MedSAM2 của LLD-MMRI-MedSAM2, có đối chiếu với bbox JSON gốc của LLD-MMRI). Roboflow chỉ dùng để xem, rà soát và nếu cần thì hiệu chỉnh bổ sung; mọi hiệu chỉnh như vậy phải được ghi lại riêng.
- Ảnh và file nhãn được **sao chép nguyên byte** từ bộ YOLO đã dùng ở GĐ2b (`data/processed/full/yolo_ct`, `data/processed/pilot/yolo_mri`); SHA256 của từng ảnh và từng nhãn được so khớp với file nguồn (kiểm tra Q5).
- Chỉ có **1 lớp: `0 = lesion`** (tổn thương gan khu trú). Chẩn đoán mức bệnh nhân (HCC, ICC, ...) **không** phải lớp YOLO; chỉ ghi trong manifest local để thống kê.
- File nhãn rỗng nghĩa là **lát đó không có tổn thương được gán nhãn**, không có nghĩa là bệnh nhân khỏe mạnh.

## 2. Giấy phép và quyền riêng tư (kiểm tra trước khi upload)

| Bộ | Nguồn | Giấy phép | Upload lên Roboflow |
|---|---|---|---|
| CT | PLC-CECT (Luo et al., *Scientific Data* 2025) | CC BY 4.0 (`CECT/LICENSE`) | Được phép, **phải ghi nguồn**. |
| MRI | LLD-MMRI (Lou et al.) + mask MedSAM2 (`wanglab/LLD-MMRI-MedSAM2`) | CC BY-NC 4.0; điều khoản LLD-MMRI: *"solely for noncommercial research purposes"*, *"not to further copy, publish, or distribute any part"* (`reports/audit_report.md`) | **Chưa nên upload.** Upload lên dịch vụ bên ngoài là sao chép dữ liệu ra ngoài; workspace công khai (gói miễn phí của Roboflow) là công bố dữ liệu. Chỉ làm khi bạn đã cân nhắc điều khoản và quyết định rõ; an toàn nhất là chỉ trình bày MRI bằng ảnh minh họa local. |

Quyền riêng tư:

- Ảnh là PNG xám cắt ra từ NIfTI (không có header DICOM). Kiểm tra Q9: không PNG nào chứa chunk văn bản/metadata (`tEXt`, `zTXt`, `iTXt`, `eXIf`, `tIME`).
- Tên file trong ZIP dùng **mã trung tính** (`CT_BN001_C1_k045`, `MRI_BN001_T2WI_k012`); bảng nối mã BN với ID của dataset chỉ nằm ở máy local (`data/processed/roboflow/meta/images_manifest_*.csv`), không có trong ZIP.
- Một số volume CT có **chữ thước đo in chồng** ("15cm", "20cm") ở rìa ảnh (GĐ1); đây không phải thông tin định danh. 100 ảnh minh họa đã được xem để phát hiện chữ in chồng khác.

## 3. Từ NIfTI 3D sang ảnh 2D (`scripts/20_build_yolo_pilot.py`, GĐ2a)

1. Đọc từng volume (streaming, không nạp cả dataset), kiểm tra hướng LPS, shape và affine của ảnh so với mask.
2. CT lưu dạng 5D `(X,Y,Z,1,3)` được đưa về 3D bằng `livercancer.data.nifti_io.as_3d` (chỉ chấp nhận khi các kênh khác nhau < 1% voxel và phần khác không chồng lên gan/tổn thương). Mask đọc giá trị raw (`get_unscaled()`) rồi so sánh `> 0`.
3. Cường độ giữ nguyên như nguồn (`uint8`): CT đã được cửa sổ hóa [−200, 200] HU tại nguồn; MRI đã chuẩn hóa theo từng volume. Không áp thêm cửa sổ hay tăng cường ảnh.
4. Mỗi lát `k` được lưu thành PNG xám ở **độ phân giải gốc** với `slice.T` (hàng = trục j, cột = trục i).
5. Lát dương tính = lát có mask tổn thương; lát âm tính = lát không có annotation và cách mọi lát có annotation ≥ 2 lát (CT: chỉ lấy lát có gan; MRI: cũng nằm ngoài các lát có `2D_box` trong JSON).
6. Volume lỗi bị loại và ghi log (`volumes.csv`): CT thiếu file, mask rỗng/lệch shape, file nén hỏng; MRI: 26 series Train bị cách ly do bbox JSON lệch mask (`data/quarantine/mri_bbox_quarantine.csv`).

## 4. Từ mask sang nhãn YOLO (`src/livercancer/data/yolo_boxes.py`)

- Mỗi **thành phần liên thông 8 hướng** của mask trên lát → 1 box (hộp bao nhỏ nhất của thành phần).
- Mảnh < 10 pixel bị bỏ; lát mà mọi thành phần đều là mảnh nhỏ bị bỏ hẳn (không coi là âm tính).
- Tọa độ chuẩn hóa theo kích thước ảnh: `0 x_center y_center width height` (∈ [0, 1]).
- MRI: lát dương tính chỉ được giữ khi hộp bao của mask và `2D_box` gốc trong JSON đồng thuận (IoU ≥ 0.3).

## 5. Cách chọn 10.000 ảnh

- Nguồn: **chỉ tập Train** đã đóng băng (`data/splits/*_split.csv`); không có ảnh Val hay Test. Bộ YOLO gốc không bị sửa.
- Mỗi loại 5.000 ảnh: mục tiêu 4.000 ảnh có tổn thương + 1.000 ảnh không có box (20%), để Roboflow có cả ảnh nền.
- Chia đều theo bệnh nhân: lần lượt mỗi bệnh nhân một ảnh mỗi vòng, xoay vòng qua các volume (pha CT / chuỗi MRI) của bệnh nhân đó.
- Mỗi bệnh nhân tối đa 40 ảnh. Trong cùng một volume, hai lát được chọn cách nhau ≥ 4 lát (CT) / ≥ 3 lát (MRI).
- Nếu không đủ ảnh có tổn thương thỏa các điều kiện trên, phần thiếu được bù bằng ảnh không box (và ngược lại); **không nhân bản ảnh** và không nới điều kiện.
- Loại ảnh gần giống hệt ảnh đã chọn của cùng bệnh nhân (dHash 256 bit, khoảng cách Hamming ≤ 16/256) và ảnh gần như trống (độ lệch chuẩn cường độ < 8).
- Ưu tiên **tổn thương rõ**: lát có box lớn nhất với cạnh ngắn ≥ 16 px được chọn trước (trong mỗi volume, lát có box lớn hơn được xét trước). Chỉ khi thiếu mới lấy lát có tổn thương nhỏ hơn.

## 6. Kết quả

| | CT | MRI | Tổng |
|---|---|---|---|
| Ảnh | 5.000 | 5.000 | 10.000 |
| Ảnh có box tổn thương | 3.924 | 4.000 | 7.924 |
| Ảnh không có box (nhãn rỗng) | 1.076 | 1.000 | 2.076 |
| Bounding box | 4.147 | 4.007 | 8.154 |
| Bệnh nhân | 253 | 316 | 569 |
| Volume (pha CT / chuỗi MRI) | 977 | 2.324 | 3.301 |
| Thiếu so với chỉ tiêu | 0 | 0 | 0 |
| BN Train có trong nguồn | 253 | 316 | |

- **CT:** chỉ có 3.924 ảnh có tổn thương thỏa các điều kiện đa dạng (mục tiêu 4.000); 76 suất còn lại được bù bằng ảnh không box, nên tỷ lệ là 78% có box / 22% không box. Nguyên nhân: nhiều BN có ít lát chứa tổn thương, và lát cùng mức cắt ở các pha khác nhau của một BN gần như giống hệt nhau nên bị loại.

Chi tiết theo mức độ rõ của tổn thương và phân bố:

| | CT | MRI |
|---|---|---|
| Ảnh tổn thương rõ / nhỏ | 3.863 / 61 | 4.000 / 0 |
| Số ảnh theo số box trong ảnh | 0 box: 1.076; 1 box: 3.738; 2 box: 166; 3 box: 13; 4 box: 3; 5 box: 1; 6 box: 2; 9 box: 1 | 0 box: 1.000; 1 box: 3.993; 2 box: 7 |
| Ảnh có box theo cạnh ngắn của box lớn nhất | <8 px: 0; 8-15 px: 61; 16-31 px: 558; 32-63 px: 1.318; >=64 px: 1.987 | <8 px: 0; 8-15 px: 0; 16-31 px: 1.444; 32-63 px: 1.716; >=64 px: 840 |
| Ảnh mỗi BN (min / trung vị / max) | 1 / 18 / 40 | 3 / 16 / 24 |
| Ảnh mỗi volume (min / trung vị / max) | 1 / 4 / 15 | 1 / 2 / 6 |
| Ảnh bị loại khi chọn | gần trùng ảnh khác của cùng BN: 7.487 | gần trùng ảnh khác của cùng BN: 1.557; gần như trống: 2 |

### CT: theo chẩn đoán mức bệnh nhân (chỉ để thống kê, không phải lớp YOLO)

| Chẩn đoán | BN | Volume | Ảnh | Có box | Không box | Box |
|---|---|---|---|---|---|---|
| CHCC | 59 | 222 | 1.444 | 1.175 | 269 | 1.300 |
| CN | 59 | 230 | 297 | 0 | 297 | 0 |
| HCC | 66 | 261 | 1.991 | 1.784 | 207 | 1.857 |
| ICC | 69 | 264 | 1.268 | 965 | 303 | 990 |

Theo pha: C1 1.374 ảnh (1.099 có box), C2 1.206 ảnh (945 có box), C3 1.260 ảnh (987 có box), P 1.160 ảnh (893 có box).

### MRI: theo chẩn đoán mức bệnh nhân (chỉ để thống kê, không phải lớp YOLO)

| Chẩn đoán | BN | Volume | Ảnh | Có box | Không box | Box |
|---|---|---|---|---|---|---|
| FOCAL_NODULAR_HYPERPLASIA | 29 | 220 | 512 | 422 | 90 | 423 |
| Hepatic_abscess | 35 | 275 | 707 | 593 | 114 | 596 |
| Hepatic_cyst | 34 | 191 | 300 | 193 | 107 | 193 |
| Hepatic_hemangioma | 50 | 360 | 634 | 478 | 156 | 478 |
| Hepatic_metastasis | 32 | 232 | 469 | 368 | 101 | 369 |
| Hepatocellular_carcinoma | 99 | 766 | 1.702 | 1.388 | 314 | 1.388 |
| Intrahepatic_cholangiocarcinoma | 37 | 280 | 676 | 558 | 118 | 560 |

Theo chuỗi: C+A 760 ảnh (633 có box), C+Delay 702 ảnh (580 có box), C+V 673 ảnh (561 có box), C-pre 750 ảnh (613 có box), DWI 463 ảnh (341 có box), In Phase 569 ảnh (433 có box), Out Phase 539 ảnh (419 có box), T2WI 544 ảnh (420 có box).

Theo nhãn YOLO: chỉ có lớp `0 = lesion`, gồm 4.147 box ở CT và 4.007 box ở MRI.

Bảng theo từng bệnh nhân: `data/processed/roboflow/meta/patient_stats_CT.csv`, `patient_stats_MRI.csv` (local, có ID dataset).

## 7. Kiểm tra chất lượng

| Kiểm tra | CT | MRI |
|---|---|---|
| Q1. Ảnh mở được, đúng kích thước | PASS | PASS |
| Q2. Mỗi ảnh có đúng một file nhãn cùng tên, không thừa/thiếu | PASS | PASS |
| Q3. Nhãn đúng định dạng (5 cột, class 0), box trong [0, 1], w, h > 0 | PASS | PASS |
| Q4. Số box khớp manifest nguồn; ảnh âm tính có nhãn rỗng | PASS | PASS |
| Q5a. Ảnh trùng SHA256 với file nguồn | PASS | PASS |
| Q5b. Nhãn trùng SHA256 với file nguồn (giữ nguyên class và tọa độ) | PASS | PASS |
| Q6. Không có ảnh trùng SHA256 trong cùng loại | PASS | PASS |
| Q9. PNG không chứa metadata văn bản | PASS | PASS |
| Q7. Mọi BN thuộc Train đã đóng băng (BN ngoài Train: 0) | PASS | (chung) |
| Q8. Không trùng giữa CT và MRI: SHA256 ảnh 0, ID BN 0, tên file 0 | PASS | (chung) |
| Q10. ZIP: CRC đúng, ảnh-nhãn đủ cặp, các phần không trùng ảnh, hợp lại đủ bộ | PASS | PASS |

Thông tin thêm (không phải lỗi): cặp ảnh gần giống nhau giữa **các BN khác nhau** (dHash ≤ 16/256): CT 0, MRI 1. Lát cắt bụng của hai người có thể có bố cục rất giống nhau; đây không phải ảnh trùng (Q6 đã kiểm tra trùng tuyệt đối).

## 8. File đầu ra

Thư mục: `data/processed/roboflow/` (không đưa vào git).

| File | Ảnh | Có box | Không box | Box | BN | Dung lượng | SHA256 |
|---|---|---|---|---|---|---|---|
| `zips/Roboflow_CT_5000_part01of05.zip` | 1.000 | 915 | 85 | 943 | 32 | 87.7 MB | `9e68c918798c8cb8…` |
| `zips/Roboflow_CT_5000_part02of05.zip` | 1.000 | 874 | 126 | 919 | 36 | 87.5 MB | `889199b047bd4d3e…` |
| `zips/Roboflow_CT_5000_part03of05.zip` | 1.000 | 769 | 231 | 782 | 53 | 81.3 MB | `bc94f271a9a68f7d…` |
| `zips/Roboflow_CT_5000_part04of05.zip` | 1.000 | 798 | 202 | 827 | 46 | 88.5 MB | `c5b2e372b35a5cd9…` |
| `zips/Roboflow_CT_5000_part05of05.zip` | 1.000 | 568 | 432 | 676 | 90 | 91.2 MB | `ca1f8d7287113c78…` |
| `zips/Roboflow_MRI_5000_part01of05.zip` | 1.000 | 812 | 188 | 815 | 60 | 79.2 MB | `1f0c2bcb51d9ba02…` |
| `zips/Roboflow_MRI_5000_part02of05.zip` | 1.000 | 775 | 225 | 775 | 72 | 80.2 MB | `e681a21520d48071…` |
| `zips/Roboflow_MRI_5000_part03of05.zip` | 1.000 | 789 | 211 | 791 | 68 | 77.0 MB | `2fa860c182dc07f3…` |
| `zips/Roboflow_MRI_5000_part04of05.zip` | 1.000 | 803 | 197 | 804 | 63 | 66.0 MB | `1b09347b7cee7847…` |
| `zips/Roboflow_MRI_5000_part05of05.zip` | 1.000 | 821 | 179 | 822 | 57 | 69.2 MB | `4add9f87dacb6d91…` |
| **Tổng CT** | 5.000 | 3.924 | 1.076 | 4.147 | | 436.2 MB | |
| **Tổng MRI** | 5.000 | 4.000 | 1.000 | 4.007 | | 371.6 MB | |

- Thư mục dataset đầy đủ (không nén): `data/processed/roboflow/Roboflow_CT_5000/`, `Roboflow_MRI_5000/` (mỗi bộ: `data.yaml`, `train/images/`, `train/labels/`).
- Theo dõi upload: `data/processed/roboflow/upload_manifest.csv` (cột `uploaded_to_roboflow`, `upload_date`, `roboflow_project` để bạn điền).
- Manifest từng ảnh (local): `data/processed/roboflow/meta/images_manifest_{CT,MRI}.csv` (tên file mới ↔ ID BN, pha/chuỗi, lát, file nguồn, SHA256).
- 100 ảnh minh họa đã vẽ box (50 CT + 50 MRI; mỗi loại 40 có box + 10 không box, mỗi ảnh một BN khác nhau): `data/processed/roboflow/previews/{CT,MRI}/`, tổng quan: `previews/contact_sheet_CT.jpg`, `contact_sheet_MRI.jpg`.

## 9. Nhập vào Roboflow

1. Tạo project loại **Object Detection**, đặt tên lớp `lesion`. Với CT, ghi nguồn PLC-CECT (CC BY 4.0) trong mô tả project. Gói miễn phí của Roboflow thường để workspace **công khai**: không upload MRI vào đó.
2. Chọn **Upload Data**, kéo thả một file ZIP (ví dụ `Roboflow_CT_5000_part01of05.zip`). Roboflow nhận dạng định dạng YOLO (`data.yaml` + `train/images` + `train/labels`) và hiện số ảnh, số box trước khi xác nhận.
3. Khi được hỏi chia tập, giữ nguyên **Train** (mọi ảnh đều từ tập Train của đồ án). Không dùng Roboflow để tạo Val/Test thay cho split đã đóng băng.
4. Bấm **Save and Continue**. Ảnh có file nhãn rỗng có thể hiện là *null* (ảnh nền) hoặc *unannotated* tùy cách Roboflow đọc; nếu là *unannotated*, chọn các ảnh đó rồi **Mark Null**. Danh sách ảnh không box: cột `kind = neg` trong manifest.
5. Sau mỗi phần, ghi `da upload`, ngày và tên project vào `upload_manifest.csv`. Các phần không trùng ảnh nên có thể upload theo thứ tự bất kỳ.
6. Xem và chỉnh box: mở ảnh trong **Annotate** → kéo cạnh box để sửa, `Delete` để xóa, vẽ box mới bằng công cụ Bounding Box. Mọi chỉnh sửa trên Roboflow là **hiệu chỉnh bổ sung** so với nhãn chuyển đổi từ dataset gốc và phải được ghi lại (ai sửa, ảnh nào, lý do) nếu muốn đưa vào báo cáo; mô hình và kết quả đã báo cáo **không** dùng các chỉnh sửa này.

## 10. Hạn chế

- Nhãn là hộp bao của mask tự động/bán tự động của dataset gốc (MRI: MedSAM2 có người kiểm tra trong vòng lặp), không phải bbox do bác sĩ vẽ trực tiếp trên Roboflow.
- Một thành phần liên thông = một box: tổn thương bị mask tách làm nhiều mảnh sẽ có nhiều box; hai tổn thương sát nhau có thể chung một box.
- Ảnh không có box chỉ có nghĩa là không có tổn thương được gán nhãn trên lát đó.
- Không train lại mô hình, không đổi checkpoint, ngưỡng, Evidence Gate, split hay dữ liệu Test.
