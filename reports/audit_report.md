# Báo cáo Audit Dataset — Giai đoạn 1

*Sinh tự động bởi `scripts/04_build_audit_report.py` lúc 2026-10-19 10:42. Mọi số liệu lấy từ các file trong `reports/` (không chứa ảnh gốc).*

Các lệnh đã chạy (đều chỉ đọc dataset gốc):

```
.venv\Scripts\python scripts\00_check_env.py
.venv\Scripts\python scripts\01_verify_datasets.py --headers 3   # đối chiếu CSV/JSON <-> file
.venv\Scripts\python scripts\02_audit_samples.py                 # đọc thử 5 CT + 5 MRI, xuất overlay
.venv\Scripts\python scripts\03_audit_full.py --workers 2        # toàn bộ mask + header ảnh
.venv\Scripts\python scripts\04_build_audit_report.py
```

## 1. Môi trường

Python 3.12.10 (.venv) · torch 2.11.0+cu128 · CUDA available = True · RTX 3050 Laptop 4.00 GB (compute capability 8.6) · đủ thư viện (`00_check_env.py` → READY).

## 2. Xác minh từ tài liệu gốc

| Nội dung | Kết quả | Nguồn |
|---|---|---|
| **CN** (CT) | Nhóm chứng: *"83 individuals without hepatobiliary disease as a control group"*; *"confirmed to have no hepatobiliary disease, and no lesion data were collected"*. CN = **không có bệnh gan mật**. Bài báo **không** nói đây là "người khỏe mạnh hoàn toàn" và không nêu tiêu chí xác nhận. Có liver mask, không có lesion mask. | [P1] |
| **HCC / ICC / CHCC** | Hepatocellular Carcinoma / Intrahepatic Cholangiocarcinoma / Combined Hepatocellular-Cholangiocarcinoma (bài báo viết *cHCC-CCA*; CSV viết `CHCC`). Mã hình thái ICD-O: HCC M81700/3, ICC M81600/3, cHCC-CCA M81800/3. Chẩn đoán: *"confirmed as liver cancer through histological and/or cytological examination"* trên mẫu phẫu thuật hoặc sinh thiết. | [P1] |
| **Pha CT** | Bài báo: *Plain (Non-Contrast), Arterial, Venous, Delayed*. **Bài báo không nhắc đến tên C1/C2/C3/P.** Bằng chứng duy nhất là README của repo tác giả: demo dùng *stage "C1"* để hiển thị *"arterial-phase slices"* (ca P0059). Trên overlay: P0089, P0110, P0172, P0198 cho thấy P không tăng quang, C1 có động mạch chủ sáng, C2/C3 ở pha muộn hơn. Nhưng **P0320 (CN) có C1 trông không tăng quang**. Kết luận: giả thuyết P = Plain, C1 = Arterial, C2 = Venous, C3 = Delayed **chưa được xác nhận chính thức và có thể không nhất quán**. Pipeline xử lý từng pha theo tên file, không gán tên lâm sàng. | [P1], [G1], overlay |
| **Tiền xử lý CT** | Bài báo: *"clipping the pixel intensity to a specified range (−200 to 200)"*, WL 0 / WW 400. Repo có `window_adjust.py` clip về [−200, 200]. Bài báo không nói về việc chuyển sang uint8, nhưng dữ liệu tải về là `uint8` 0–255. **Không còn HU gốc**; giá trị 0 tương ứng ≤ −200 HU (nền và khí). | [P1], [G1] |
| **Số pha / BN** | Bài báo: *"All individuals ... had complete 3D CECT imaging data across the Plain, Arterial, Venous, and Delayed phases"*. Vậy 40 file thiếu là do **bản tải về không đầy đủ**, không phải do thiết kế dataset. | [P1] |
| License CT | CC BY 4.0 (Science Data Bank, doi 10.57760/sciencedb.12207) | [P1], `CECT/LICENSE` |
| **Split CT** | Tác giả **không** công bố split. | [P1] |
| **Split MRI** | Challenge LLD-MMRI2023: train 316 / val 78 / test 104. Danh sách chính thức (`labels.txt`, `labels_val_inaccessible.txt`) chỉ có trong **gói dữ liệu gốc**, không có trong bản HF MedSAM2. File `main/output/LLDBaseline.json` (repo challenge) là *"prediction result file format"* của baseline, gồm đúng **78 image_id**, bằng kích thước tập val. → **Suy luận** đây là tập val (độ tin cậy cao nhưng không phải tuyên bố trực tiếp). Đã lưu ở `data/external/lld_mmri2023_official_val_ids.csv`; khớp 78/78 với web, và cả 78 ID có trong dữ liệu. **Danh sách train và test không công khai.** Tác giả khuyến nghị *5-fold cross-validation* trên phần dữ liệu có nhãn. | [G2] |
| Nhãn MRI | `Category_info` trong JSON: 0 Hemangioma, 1 ICC, 2 Abscess, 3 Metastasis, 4 Cyst, 5 FNH, 6 HCC; Benign [0,2,4,5], Malignant [1,3,6]. Khớp với danh sách 7 loại tổn thương trong README HF. 8 chuỗi khớp với README. | README HF, [G2] |
| License MRI | **CC BY-NC 4.0**, *"solely for noncommercial research purposes"*; *"You agree not to further copy, publish, or distribute any part of the LLD-MMRI dataset"* (ngoại trừ dùng nội bộ). **Upload lên Google Drive/Colab cần người dùng tự cân nhắc theo điều khoản này và cho phép rõ ràng.** | [G3] |

Nguồn: [P1] Luo et al., *Comprehensive multi-phase 3D contrast-enhanced CT imaging for primary liver cancer*, Sci Data 12:768 (2025), doi 10.1038/s41597-025-05125-2 (PMC12065907). [G1] github.com/ljwa2323/PLC_CECT. [G2] github.com/LMMMEng/LLD-MMRI2023 (README, main/README.md, main/output/LLDBaseline.json). [G3] github.com/LMMMEng/LLD-MMRI-Dataset. Đọc ngày 2026-10-19.

## 3. Thống kê

- **CT:** 361 bệnh nhân · 1444 volume trong CSV · 1404 volume có file.
- **MRI:** 498 bệnh nhân · 3984 series ảnh · 3984 label. Mỗi bệnh nhân có 8 chuỗi và 1 tổn thương.

### CT — PLC-CECT

| Lớp | Bệnh nhân | Volume trong CSV | Volume có file | Volume thiếu | BN đủ 4 pha | BN thiếu ≥1 pha | Tuổi TB (min–max) | Nam/Nữ/NA |
|---|---|---|---|---|---|---|---|---|
| HCC | 94 | 376 | 368 | 8 | 86 | 8 | 57.3 (39–75) | 75/19/0 |
| ICC | 99 | 396 | 387 | 9 | 90 | 9 | 63.6 (39–86) | 53/43/3 |
| CHCC | 85 | 340 | 324 | 16 | 70 | 15 | 56.3 (39–76) | 57/28/0 |
| CN | 83 | 332 | 325 | 7 | 76 | 7 | 45.1 (21–76) | 15/68/0 |
| Tổng | 361 | 1444 | 1404 | 40 | 322 | 39 | 55.9 (21–86) | 200/158/3 |

Volume CT thiếu theo pha: {'C3': 11, 'C1': 8, 'C2': 10, 'P': 11}
Bệnh nhân có age/gender không nhất quán giữa các dòng: 0 []
Bệnh nhân thiếu ≥1 pha: ['P0004', 'P0005', 'P0037', 'P0046', 'P0057', 'P0071', 'P0074', 'P0092', 'P0096', 'P0122', 'P0144', 'P0146', 'P0149', 'P0150', 'P0157', 'P0172', 'P0185', 'P0194', 'P0201', 'P0209', 'P0211', 'P0212', 'P0216', 'P0227', 'P0231', 'P0238', 'P0239', 'P0240', 'P0247', 'P0254', 'P0255', 'P0259', 'P0286', 'P0291', 'P0293', 'P0299', 'P0311', 'P0317', 'P0336']

### MRI — LLD-MMRI-MedSAM2

| id | Lớp | Nhóm | Bệnh nhân | Series | Trong val chính thức | Còn lại (train+test) |
|---|---|---|---|---|---|---|
| 0 | Hepatic_hemangioma | Benign | 79 | 632 | 13 | 66 |
| 1 | Intrahepatic_cholangiocarcinoma | Malignant | 58 | 464 | 9 | 49 |
| 2 | Hepatic_abscess | Benign | 54 | 432 | 8 | 46 |
| 3 | Hepatic_metastasis | Malignant | 51 | 408 | 8 | 43 |
| 4 | Hepatic_cyst | Benign | 53 | 424 | 8 | 45 |
| 5 | FOCAL_NODULAR_HYPERPLASIA | Benign | 46 | 368 | 7 | 39 |
| 6 | Hepatocellular_carcinoma | Malignant | 157 | 1256 | 25 | 132 |
| | **Tổng** | | 498 | 3984 | 78 | 420 |

## 4. Tương ứng ảnh – mask – nhãn (toàn bộ dataset)

### CT

| Kiểm tra | Kết quả |
|---|---|
| Volume CT có file | 1404/1444 (thiếu 40) |
| Số chiều volume CT | {'3': 728, '5': 676} |
| dtype CT / spacing header | {'uint8': 1404} / {'1x1x1': 1404} |
| Số lát cắt (trục k) | 20–587 |
| Liver mask rỗng | 13 |
| Liver mask ≠ CT (shape / affine) | 0 / 0 |
| Lesion mask (dòng có mask) | 1112 (CN: `mask_path = "NA"`) |
| Lesion mask rỗng | 2 ['P0051_P', 'P0165_C1'] |
| Lesion ≠ CT shape / ≠ liver shape / ≠ liver affine | 1 / 1 / 0 |
| Giá trị raw của lesion mask | {'0;65535': 109, '0;1': 1001, '0': 2} |
| Số thành phần liên thông tổn thương / volume | {'1': 982, '3': 23, '2': 93, '5': 2, '4': 7, '22': 1, '9': 1, '27': 1} |
| Tổn thương nằm trong liver mask < 50% | 149 |
| Tổn thương nằm trong bbox gan < 90% | 77 |

### MRI

| Kiểm tra | Kết quả |
|---|---|
| Series thiếu file ảnh/label | 0 |
| Số chiều ảnh | {'3': 3984} |
| Label ≠ ảnh (shape / affine) | 0 / 0 |
| Spacing header ≠ JSON | 0 |
| Label rỗng | 0 [] |
| Giá trị label (raw) | {'0;1': 3984} |
| JSON slice_idx ngoài phạm vi / bbox ngoài ảnh | 0 / 0 |
| Khoảng z của JSON và mask không giao nhau | 2 |
| IoU(JSON 3D_box, bbox của mask) — min/p5/p25/median/p75 | [0.034, 0.537, 0.772, 0.847, 0.894] |
| IoU < 0.3 | 31 |
| Số thành phần liên thông của mask | {'1': 3913, '2': 61, '3': 6, '4': 4} |

**Hệ trục đã chốt:** JSON `x` ↔ trục i, `y` ↔ trục j, `slice_idx` ↔ trục k của mảng nibabel. Trên 40 series mẫu, IoU theo cách hiểu này là 0.58–0.96, còn cách hiểu đảo x/y cho IoU ≈ 0 (xác nhận bằng overlay). Trên toàn bộ 3984 series: median IoU 0.847, p25 0.772; các ca IoU thấp xem E9. Hiển thị: `slice.T` (hàng = j, cột = i).

### 4b. Điều tra 31 series MRI có JSON box lệch với mask (`scripts/05_check_mri_bbox_mismatch.py`)

Phạm vi: toàn bộ 8 chuỗi của 18 BN bị ảnh hưởng (144 series), trong đó 31 series bị gắn cờ; 113 series còn lại dùng làm mốc đối chiếu.

| Giả thuyết | Kiểm tra | Kết quả | Kết luận |
|---|---|---|---|
| Orientation / affine / spacing | affine ảnh = affine label; spacing JSON = header; origin JSON (LPS) = affine (RAS, đảo dấu x, y) | 144/144 series nhất quán | Loại trừ |
| Quy ước trục / thứ tự lát cắt | IoU 3D khi giữ nguyên, lật i, lật j, lật k, hoán đổi i/j | biến đổi tốt nhất: {'identity': 31}; không biến đổi nào cho IoU ≥ 0.5 | Loại trừ |
| 2D_box ↔ 3D_box | z của các 2D_box = z của 3D_box; slice_idx liên tục | z khớp 31/31; không liên tục 3 (MR160207 DWI, MR19145 C-pre, MR19145 C+V) | Không giải thích được |
| Vị trí trong tọa độ thế giới (mm) | khoảng cách từ tâm box/mask đến median của các chuỗi tốt cùng BN | series lệch: JSON 15.8 mm, mask 16.5 mm; series tốt: JSON 5.5 mm, mask 5.4 mm | **Cả hai nguồn đều lệch**, không xác định được nguồn nào đúng |
| Kích thước | thể tích mask / thể tích JSON box | series lệch 0.073, series tốt 0.435 | Mask thường nhỏ hơn nhiều so với box |

**Kết luận:** không xác minh được nguyên nhân từ orientation, affine, spacing, thứ tự lát cắt hay hệ tọa độ. **Cách ly 31 series / 18 BN** khỏi huấn luyện YOLO và ROI classification. Bệnh nhân vẫn được giữ với các chuỗi còn lại. Không sửa bbox, không xóa file gốc.

- Series bị cách ly theo lớp: {'FOCAL_NODULAR_HYPERPLASIA': 6, 'Hepatic_abscess': 4, 'Hepatic_hemangioma': 5, 'Intrahepatic_cholangiocarcinoma': 12, 'Hepatic_metastasis': 2, 'Hepatocellular_carcinoma': 2}
- BN bị ảnh hưởng theo lớp: {'FOCAL_NODULAR_HYPERPLASIA': 4, 'Hepatic_abscess': 2, 'Hepatic_hemangioma': 4, 'Intrahepatic_cholangiocarcinoma': 4, 'Hepatic_metastasis': 2, 'Hepatocellular_carcinoma': 2}
- Số series bị cách ly / BN: {1: 12, 2: 2, 3: 2, 4: 1, 5: 1} (không BN nào mất cả 8 chuỗi)
- Overlay minh họa: `figures/audit/mri_lowiou_check.png`.

### Volume CT dùng được

Một volume được tính là dùng được nếu có file CT và (với lớp ung thư) lesion mask không rỗng, đúng shape. Kết quả: **1401/1444** ({'HCC': 367, 'ICC': 386, 'CHCC': 323, 'CN': 325}). Số BN theo số pha dùng được: {4: 320, 3: 39, 2: 2}; BN có 0 pha dùng được: 0. Liver mask rỗng (E14) không làm volume mất tác dụng cho lesion detection.

## 5. Đọc thử 5 CT + 5 MRI và ảnh overlay

| Modality | Bệnh nhân | Nhãn (từ CSV/JSON) | Ghi chú | Ảnh |
|---|---|---|---|---|
| CT | P0089 | HCC | OK | [png](figures/audit/ct_P0089.png) |
| CT | P0110 | ICC | C1: 5D→3D, C2: 5D→3D, C3: 5D→3D, P: 5D→3D | [png](figures/audit/ct_P0110.png) |
| CT | P0198 | CHCC | OK | [png](figures/audit/ct_P0198.png) |
| CT | P0320 | CN | C1: 5D→3D, C2: 5D→3D, C3: 5D→3D, P: 5D→3D | [png](figures/audit/ct_P0320.png) |
| CT | P0172 | ICC | C1: 5D→3D, C2: 5D→3D, C3: thiếu file, P: 5D→3D | [png](figures/audit/ct_P0172.png) |
| MRI | MR191390 | Intrahepatic_cholangiocarcinoma | IoU JSON–mask 0.78–0.94 | [png](figures/audit/mri_MR191390.png) |
| MRI | MR111774 | Hepatocellular_carcinoma | IoU JSON–mask 0.75–0.91 | [png](figures/audit/mri_MR111774.png) |
| MRI | MR170186 | Hepatic_hemangioma | IoU JSON–mask 0.58–0.92 | [png](figures/audit/mri_MR170186.png) |
| MRI | MR110046 | Hepatic_cyst | IoU JSON–mask 0.85–0.96 | [png](figures/audit/mri_MR110046.png) |
| MRI | MR135466 | Hepatic_abscess | IoU JSON–mask 0.88–0.93 | [png](figures/audit/mri_MR135466.png) |

CT: xanh lá = liver mask, đỏ = lesion mask. MRI: đỏ = label mask, vàng = JSON 2D_box. Mask khớp giải phẫu trên toàn bộ mẫu đã xem.

## 6. Lỗi và dữ liệu không hợp lệ

| # | Vấn đề | Quy mô | Mức độ | Cách xử lý (không sửa dữ liệu gốc) |
|---|---|---|---|---|
| E1 | File CT có trong CSV nhưng không tồn tại | 40 volume / 39 BN | Trung bình | Bài báo nói mọi BN đủ 4 pha, nên đây là **bản tải về không đầy đủ** (HF cache chỉ có 1404 metadata). Loại các volume này khỏi manifest; bệnh nhân vẫn được giữ với các pha còn lại. Không tải lại (quy tắc 4). |
| E2 | CT lưu dạng 5D (X,Y,Z,1,3) | {'3': 728, '5': 676} · 173 BN | Thấp | Khi đọc: kiểm tra 3 kênh giống nhau rồi lấy kênh 0 (`nifti_io.as_3d`); nếu các kênh khác nhau thì báo lỗi. Audit toàn bộ chỉ đọc header; việc 3 kênh giống nhau mới được xác nhận trên các ca mẫu (P0110, P0172, P0320), phần còn lại sẽ được kiểm tra khi xuất lát cắt. |
| E3 | Lesion mask lưu raw 0/65535 kèm slope 1/65535 (đọc ra ≈ 0.9999999998) | 109 | Thấp | Đọc giá trị raw (`get_unscaled`) rồi nhị phân hóa `> 0`. |
| E4 | Liver mask không bao trọn tổn thương (tổn thương nằm trong liver mask < 50%) | 149 volume | Trung bình | Liver mask là nhu mô gan và có thể loại trừ khối u. Khi cần vùng gan: dùng **hợp liver ∪ lesion** (chỉ trên train/val). |
| E5 | Tổn thương nằm ngoài bbox gan (< 90%) | 77 volume | Thấp/TB | Như E4. Kiểm tra trực quan khi sinh nhãn YOLO. |
| E6 | Lesion mask rỗng (BN ung thư nhưng mask trống) | 2: P0051_P, P0165_C1 | TB | **Không coi là âm tính** (BN có ung thư); loại volume khỏi detection và classification. |
| E7 | Spacing header CT luôn là {'1x1x1': 1404} trong khi số lát cắt dao động 20–587 | toàn bộ | Trung bình | **Không tin được spacing thật**, nên không dùng phép đo thể tích theo mm. Pipeline 2D không bị ảnh hưởng. |
| E8 | CT đã windowing sẵn [-200, 200] HU về uint8 | toàn bộ | Thông tin | Không áp HU window; giữ 0–255 và chuẩn hóa theo thống kê của train. |
| E9 | MRI: JSON 3D_box lệch với mask MedSAM2 (IoU < 0.3) | 31 series / 18 BN; theo chuỗi {'C-pre': 6, 'T2WI': 2, 'C+A': 2, 'C+Delay': 1, 'In Phase': 5, 'C+V': 4, 'Out Phase': 5, 'DWI': 6} | TB | Điều tra ở mục 4b: **không xác minh được nguyên nhân**. → **Cách ly toàn bộ** khỏi huấn luyện YOLO và ROI classification (`data/quarantine/mri_bbox_quarantine.csv`). Không sửa bbox, không xóa file. Ví dụ: MR209135 C-pre 0.034, MR-467988 C-pre 0.0494, MR22508 T2WI 0.0555, MR209135 C+A 0.0712, MR209135 C+Delay 0.0713 |
| E10 | MRI: mask có nhiều thành phần liên thông | 71 series | Thấp | num_targets = 1 nên có thể là mảnh vụn của mask. Khi tạo bbox, lấy thành phần lớn nhất hoặc bbox hợp, quyết định ở GĐ2. |
| E11 | **Confounder nhân khẩu học**: CN 15 nam/68 nữ, tuổi TB 45.1; nhóm ung thư đa số nam, tuổi 56–64 | 83 BN | **Cao** (bias) | Không dùng tuổi/giới tính làm đặc trưng; nêu trong phần hạn chế; split phân tầng theo nhãn. |
| E12 | Không có mapping chính thức C1/C2/C3/P → pha lâm sàng; overlay P0320 mâu thuẫn với giả thuyết C1 = arterial | 4 pha | Thấp/TB | Xử lý từng pha độc lập theo tên file; không gán tên pha lâm sàng khi chưa có nguồn. Nếu cần, kiểm tra thêm bằng cường độ động mạch chủ (GĐ1b). |
| E13 | MRI: không có danh sách tách train/test chính thức | 420 BN | Trung bình | Xem mục 7. |
| E14 | Liver mask rỗng | 13 volume / 4 BN: P0112_P, P0112_C2, P0112_C3, P0112_C1, P0161_P, P0175_P, P0175_C3, P0175_C2, P0175_C1, P0193_C3, P0193_C1, P0193_P, P0193_C2 (đều là ICC và dạng 5D; P0112/P0193 có tổn thương rất lớn, khoảng 0.4–0.55 triệu voxel) | Trung bình | Không sinh nhãn YOLO lớp `liver` cho các volume này. Vẫn dùng được lesion mask (và lát cắt trong z-range của tổn thương). |
| E15 | Lesion mask khác shape với CT (lệch số lát cắt) | 1: P0231_C3 | Cao (cho volume đó) | P0231: mask C3 có 132 lát, bằng CT C2, nhưng không trùng voxel với mask C2; mức chồng lên liver C3 và C2 là 0.91 và 0.93 → **không xác định được pha thật**. **Loại lesion mask của volume này**; các pha khác vẫn dùng được. |
| E16 | MRI: khoảng z của JSON và mask không giao nhau | 2: MR209135 C-pre (JSON z 27-33, mask z 34-39); MR206627 In Phase (JSON z 34-36, mask z 32-32) | TB | Thuộc nhóm E9 (overlay `mri_lowiou_check.png`). Xử lý theo phương án E9 mà người dùng chốt. |
| E17 | CT: lesion mask có nhiều thành phần liên thông (≥ 5) | 5: P0017_C3 (5), P0063_C2 (5), P0268_C3 (22), P0268_C1 (9), P0268_C2 (27) | Thấp | Có thể là đa ổ hoặc mảnh vụn của annotation. Khi sinh bbox YOLO: mỗi thành phần đủ lớn là một box, mảnh vụn thì lọc theo diện tích (ngưỡng chốt ở GĐ2 trên train/val). |

## 7. Đánh giá điều kiện chia Train/Validation/Test

**CT (361 BN):** Lớp nhỏ nhất là CN với 83 BN. Split 70/15/15 phân tầng cho khoảng 58/12–13/12–13 BN cho lớp nhỏ nhất. → **Đủ điều kiện.** Split ở mức bệnh nhân; volume thiếu được loại trước khi split nhưng bệnh nhân vẫn được giữ.

**MRI (498 BN):** Hướng người dùng chọn là ưu tiên split chính thức nếu ánh xạ được chính xác. Mức ánh xạ được:

- **Validation chính thức (78 BN): ánh xạ được.** 78/78 ID có trong MedSAM2; mỗi ID là đúng một BN với 8 series; không trùng lặp. Mức tin cậy: suy luận (78 ID trong file dự đoán baseline = kích thước tập val).
- **Train (316) / Test (104) chính thức: không ánh xạ được**, vì danh sách không được công bố công khai (chỉ có trong gói dữ liệu gốc mà project không có). 316 + 104 = 420 = đúng số ID còn lại.
- Vì vậy chỉ có thể giữ phần val chính thức. 420 BN còn lại phải tự chia Train/Test (phân tầng theo lớp, **mức bệnh nhân**, seed cố định, kích thước 316/104 như challenge). Tập Test này **không phải** test chính thức của challenge, và kết quả không so sánh trực tiếp được với leaderboard. Chi tiết trong `docs/PHASE2_PLAN.md`.

