# Phân tích hạn chế và cơ chế từ chối phân loại (Validation)

*Sinh bởi `scripts/93_limitations_analysis.py` lúc 2026-10-26 18:20. Nguồn: cache pipeline `outputs/predictions/pipeline_validation/` (scripts/92_), quét YOLO Val `outputs/predictions/roi_val/`, bộ ROI phân loại, run `mri_hybrid_gt_s42`. Số liệu tổng hợp: `outputs/predictions/pipeline_validation/limitations.json`. **Chỉ Validation; không dùng Test; không huấn luyện lại; checkpoint và dữ liệu gốc không đổi.***

## 1. Kết quả CT phải đọc theo từng nhóm bệnh nhân

Con số 90,5% (38/42) là **chỉ trên bệnh nhân ung thư**, **không** phải accuracy trên toàn bộ CT Validation (54 BN).

| CT Validation | Không có cổng bằng chứng (hiện tại) | **Có cổng bằng chứng (đề xuất)** | Cổng 'không mất ca ung thư' |
|---|---|---|---|
| BN ung thư: phân loại đúng | 38/42 (90.5%) | **36/42 (85.7%)** | 38/42 (90.5%) |
| BN ung thư: phân loại sai | 4/42 (9.5%) | **4/42 (9.5%)** | 4/42 (9.5%) |
| BN ung thư: không đưa nhãn (từ chối / không có vùng) | 0/42 (0.0%) | **2/42 (4.8%)** | 0/42 (0.0%) |
| Nhóm chứng CN: không đưa nhãn (đúng mong muốn) | 1/12 (8.3%) | **10/12 (83.3%)** | 8/12 (66.7%) |
| Nhóm chứng CN: bị gán nhãn ung thư (dương tính giả) | 11/12 (91.7%) | **2/12 (16.7%)** | 4/12 (33.3%) |
| Toàn bộ CT: kết quả chấp nhận được (ung thư đúng + CN không nhãn) | 39/54 (72.2%) | **46/54 (85.2%)** | 46/54 (85.2%) |
| Toàn bộ CT: đưa ra nhãn sai (ung thư sai + CN bị gán ung thư) | 15/54 (27.8%) | **6/54 (11.1%)** | 8/54 (14.8%) |

- Cổng đề xuất: **chỉ phân loại khi có vùng với độ tin cậy ≥ 0.4 ở ít nhất 2 pha CT**; nếu không đạt, trạng thái `insufficient_evidence`, **không** đưa nhãn và **không** kết luận bệnh nhân khỏe mạnh.
- Phương án 'không mất ca ung thư nào' trên Val: có vùng với độ tin cậy ≥ 0.25 ở ít nhất 3 pha CT.
- Cổng khi chỉ có **một file** (một pha) tải lên: độ tin cậy YOLO cao nhất ≥ 0.35 (mức volume Val: giữ 155/163 volume có tổn thương, từ chối 32/47 volume CN).

### Cách chọn cổng và độ lạc quan

- Quy tắc chọn (khai báo trước, chỉ trên Val): trong 65 luật (độ tin cậy cao nhất; số pha có vùng mạnh; độ dài vùng mạnh), chọn luật **từ chối được nhiều BN CN nhất** mà vẫn **phân loại ≥ 95% BN ung thư** (≥ 40/42); hòa thì giữ nhiều BN ung thư hơn, luật đơn giản hơn.
- Ngưỡng được chọn trên chính 54 BN này (chỉ 12 CN) nên **lạc quan**. Ước lượng bằng leave-one-out (chọn luật trên 53 BN, áp cho BN còn lại): giữ 40/42 BN ung thư, từ chối 9/12 BN CN.
- Xác suất của bộ phân loại **không** tách được CN (BN CN vẫn có xác suất lớn nhất tới 0,96), nên cổng dựa trên **bằng chứng phát hiện**, không dựa trên xác suất lớp.

| BN | Nhãn thực tế | Pipeline | YOLO cao nhất | Số pha có vùng mạnh | Số vùng | Xác suất lớp lớn nhất | Qua cổng |
|---|---|---|---|---|---|---|---|
| P0304 | CN | None | 0.0 | 0 | 0 |  | **không** |
| P0351 | CN | CHCC | 0.123 | 0 | 1 | 0.961 | **không** |
| P0301 | CN | ICC | 0.189 | 0 | 3 | 0.506 | **không** |
| P0341 | CN | ICC | 0.266 | 0 | 9 | 0.549 | **không** |
| P0358 | CN | ICC | 0.286 | 0 | 8 | 0.885 | **không** |
| P0344 | CN | ICC | 0.411 | 1 | 14 | 0.597 | **không** |
| P0340 | CN | HCC | 0.418 | 1 | 13 | 0.607 | **không** |
| P0300 | CN | ICC | 0.428 | 1 | 14 | 0.666 | **không** |
| P0336 | CN | ICC | 0.557 | 1 | 6 | 0.505 | **không** |
| P0329 | CN | ICC | 0.574 | 1 | 10 | 0.696 | **không** |
| P0288 | CN | ICC | 0.596 | 3 | 15 | 0.528 | có |
| P0281 | CN | HCC | 0.732 | 4 | 61 | 0.733 | có |
| P0133 | ICC | ICC | 0.341 | 0 | 10 | 0.588 | **không** |
| P0132 | ICC | ICC | 0.42 | 1 | 28 | 0.623 | **không** |

(Bảng trên liệt kê mọi BN CN và các BN ung thư bị cổng từ chối; đầy đủ ở `gate_per_patient.csv`.)

## 2. Vì sao nhóm chứng CT bị gán nhãn ung thư

- **Mô hình phân loại chỉ có 3 lớp ung thư.** Không có lớp 'không phải tổn thương', nên mọi vùng YOLO (kể cả vùng giả) đều nhận HCC / ICC / CHCC.
- **YOLO ở ngưỡng 0.12 (chọn để hạn chế bỏ sót) tạo vùng giả ở 11/12 BN CN**: tổng 154 vùng trên 36 volume CN; tâm vùng nằm **trong gan** 98, **ngoài gan** 56 (liver mask chỉ dùng để phân tích). 31 vùng có độ tin cậy ≥ 0.4.
- Độ tin cậy vùng giả ở CN: trung vị 0.21, lớn nhất 0.73; độ dài trung vị 2 lát. Vùng thật ở BN ung thư thường lặp lại ở 3–4 pha với độ tin cậy cao.
- Một vài BN CN có vùng giả mạnh và lặp lại qua các pha (ví dụ P0281, P0288): cổng không loại được; cần lớp 'không tổn thương' hoặc YOLO tốt hơn.
- ROI và tiền xử lý **không** phải nguyên nhân: crop của pipeline trùng từng pixel với dữ liệu đánh giá (`reports/pipeline_val_report.md` §3).

![cn_vung_gia_P0281_C3_k055.png](cn_vung_gia_P0281_C3_k055.png)
![cn_vung_gia_P0288_C1_k004.png](cn_vung_gia_P0288_C1_k004.png)
![cn_vung_gia_P0329_P_k140.png](cn_vung_gia_P0329_P_k140.png)
![cn_vung_gia_P0336_C1_k052.png](cn_vung_gia_P0336_C1_k052.png)

## 3. MRI: vì sao chỉ đạt 47,4% (37/78 BN)

| Lớp | ROI Train (GT) | ROI Val thật | ROI Val bị dự đoán là lớp này | Diện tích trung vị Train / Val (px²) | Tỉ lệ tổn thương nhỏ Train / Val |
|---|---|---|---|---|---|
| HCC (ung thư biểu mô tế bào gan) | 3833 | 919 | 1100 | 1720 / 2385 | 32% / 24% |
| ICC (ung thư đường mật trong gan) | 1467 | 332 | 268 | 2178 / 1600 | 26% / 24% |
| Di căn gan | 1100 | 241 | 144 | 780 / 598 | 67% / 56% |
| U máu gan | 1380 | 338 | 412 | 900 / 905 | 54% / 55% |
| Nang gan | 842 | 190 | 180 | 400 / 437 | 86% / 75% |
| Áp xe gan | 1532 | 310 | 186 | 4618 / 3688 | 9% / 13% |
| FNH (tăng sản nốt khu trú) | 1076 | 245 | 285 | 1802 / 806 | 35% / 58% |

- **Dự đoán bị hút về HCC:** 1100 ROI Val bị đoán là HCC so với 919 ROI HCC thật; HCC chiếm nhiều ROI Train nhất.
- **Quá tự tin khi sai:** độ tin cậy trung bình khi đúng 0.832, khi **sai** vẫn 0.709; 43% số dự đoán sai có xác suất ≥ 0.8. Đây là dấu hiệu overfit (train acc 0.97 so với Val 0.48; val loss nhỏ nhất ở epoch 1).
- **Không phải lỗi nhãn:** nhãn MRI khớp 100% giữa JSON, 8 chuỗi và split (`reports/cls_data_report.md`). **Không phải lỗi tiền xử lý:** crop trùng từng pixel; cường độ trung bình crop local Train 131.9 ± 37.5, Val 130.1 ± 38.0.
- **ROI do YOLO làm giảm thêm:** ROI acc GT 47.8% → YOLO khớp GT 49.0% → mọi vùng YOLO 35.8%; 83% vùng YOLO MRI là vùng giả.
- **Không có chuỗi MRI nào tốt:** theo chuỗi, oracle chỉ 40–56% (`bao_cao_pipeline_validation.md` §2), nên không phải do một chuỗi kém.
- **Giả thuyết chưa kiểm chứng:** mô hình xử lý từng chuỗi riêng lẻ, trong khi phân biệt u gan trên MRI dựa vào mẫu ngấm thuốc qua nhiều pha; Val nhỏ (7–25 BN/lớp) và là tập val chính thức LLD-MMRI2023 (có thể khác phân bố so với Train).

Phân loại có chọn lọc (chỉ trả nhãn khi xác suất lớn nhất ≥ ngưỡng), mức BN, pipeline:

| Xác suất tối thiểu | Tỉ lệ BN được trả nhãn | Accuracy trên BN được trả nhãn |
|---|---|---|
| 0.0 | 100% (78) | 47.4% |
| 0.4 | 62% (48) | 58.3% |
| 0.5 | 42% (33) | 72.7% |
| 0.6 | 24% (19) | 84.2% |
| 0.7 | 15% (12) | 83.3% |
| 0.8 | 5% (4) | 100.0% |

- Chỉ giữ ca tự tin làm tăng accuracy trên phần được giữ (ví dụ ngưỡng 0.6: 84.2% trên 19/78 BN, tức chỉ 24% số BN), nhưng ngưỡng này được nhìn trên chính Val nhỏ (khoảng 20 BN) nên **chưa đủ tin cậy**, và phần lớn BN sẽ không có nhãn.
- **Phân bố kích thước khác nhau giữa Train và Val** ở: FNH (tăng sản nốt khu trú) (tổn thương nhỏ 35% Train / 58% Val). Đây là một nguồn chênh lệch giữa Train và Val có thể đo được.
- Kết luận: **không đưa nhãn MRI vào website như kết quả chẩn đoán.** Nếu cần minh họa, chỉ hiện ở chế độ nghiên cứu, kèm cảnh báo đỏ.

## 4. Các trường hợp YOLO bỏ sót

### CT: 177/275 tổn thương 3D được tìm thấy (64.4%)

| Kích thước tổn thương | Số tổn thương | Được tìm thấy |
|---|---|---|
| nhỏ < 32² | 69 | 14 (20.3%) |
| vừa 32²–96² | 166 | 123 (74.1%) |
| lớn > 96² | 40 | 40 (100.0%) |

- Tổn thương chỉ xuất hiện trên **1 lát**: tìm thấy 4/24.
- Theo lớp: CHCC 63/88, HCC 59/113, ICC 55/74.
- Theo pha: C1 41/64, C2 46/69, C3 47/74, P 43/68.
- Volume có tổn thương mà YOLO không tìm thấy tổn thương nào: **11/163**; BN bị bỏ sót **mọi** tổn thương: **0**.


![bo_sot_ct_P0133_C1_k090.png](bo_sot_ct_P0133_C1_k090.png)
![bo_sot_ct_P0133_C3_k084.png](bo_sot_ct_P0133_C3_k084.png)
![bo_sot_ct_P0266_C3_k082.png](bo_sot_ct_P0266_C3_k082.png)

### MRI: 554/648 tổn thương 3D được tìm thấy (85.5%)

| Kích thước tổn thương | Số tổn thương | Được tìm thấy |
|---|---|---|
| nhỏ < 32² | 264 | 186 (70.5%) |
| vừa 32²–96² | 307 | 291 (94.8%) |
| lớn > 96² | 77 | 77 (100.0%) |

- Tổn thương chỉ xuất hiện trên **1 lát**: tìm thấy 33/66.
- Theo lớp: FOCAL_NODULAR_HYPERPLASIA 47/58, Hepatic_abscess 67/68, Hepatic_cyst 36/64, Hepatic_hemangioma 82/104, Hepatic_metastasis 57/63, Hepatocellular_carcinoma 196/215, Intrahepatic_cholangiocarcinoma 69/76.
- Theo chuỗi: C+A 71/80, C+Delay 66/79, C+V 74/81, C-pre 66/77, DWI 74/86, In Phase 66/81, Out Phase 66/80, T2WI 71/84.
- Volume có tổn thương mà YOLO không tìm thấy tổn thương nào: **82/622**; BN bị bỏ sót **mọi** tổn thương: **2** (MR140232, MR142442).

  - MR140232: C+A 112 px² (3 lát); C+Delay 99 px² (3 lát); C+V 154 px² (3 lát); C-pre 132 px² (3 lát); DWI 90 px² (1 lát); In Phase 117 px² (1 lát); Out Phase 240 px² (1 lát); T2WI 169 px² (1 lát)
  - MR142442: C+A 600 px² (5 lát); C+Delay 660 px² (5 lát); C+V 720 px² (5 lát); C-pre 702 px² (5 lát); DWI 380 px² (1 lát); In Phase 572 px² (1 lát); Out Phase 621 px² (1 lát); T2WI 550 px² (3 lát)

![bo_sot_mri_MR175358_OutPhase_k011.png](bo_sot_mri_MR175358_OutPhase_k011.png)
![bo_sot_mri_MR146022_InPhase_k010.png](bo_sot_mri_MR146022_InPhase_k010.png)
![bo_sot_mri_MR127280_C-pre_k027.png](bo_sot_mri_MR127280_C-pre_k027.png)

- YOLO bỏ sót chủ yếu **tổn thương nhỏ và chỉ trên 1–2 lát**. Ở mức BN, CT không bỏ sót BN ung thư nào; MRI có 2 BN bị bỏ sót hoàn toàn. Theo quy tắc của dự án, các ca này là **lỗi của pipeline** và vẫn được tính khi đánh giá.

## 5. Cấu hình cuối đề xuất cho website (chưa triển khai)

| Thành phần | CT | MRI |
|---|---|---|
| Detection | YOLO11n CT, conf 0.12 | YOLO11n MRI, conf 0.12 |
| Tumor Zoom | local + context, không lọc vùng | như CT |
| Phân loại | Hybrid CT (`ct_hybrid_gt_s42`), gộp trung bình xác suất | **Tắt mặc định.** Chỉ hiển thị phát hiện và Tumor Zoom; nhãn MRI chỉ ở chế độ 'nghiên cứu' kèm cảnh báo đỏ |
| Cổng bằng chứng | **Bật:** có vùng với độ tin cậy ≥ 0.4 ở ít nhất 2 pha CT; một file: độ tin cậy YOLO cao nhất ≥ 0.35 | không áp dụng (không phân loại) |
| Khi không có vùng / không qua cổng | 'Không phát hiện vùng nghi ngờ' / 'Chưa đủ bằng chứng', **không** kết luận khỏe mạnh | 'Không phát hiện vùng nghi ngờ' |
| Cách trình bày kết quả CT | Ghi rõ: kết quả Val **trên BN ung thư**; mô hình không nhận biết người không có bệnh gan | — |

Các giá trị đã ghi vào `configs/cls.yaml → inference.ct.evidence_gate` với `enabled: false` (**chờ người dùng duyệt**). Bật bằng `enabled: true`; pipeline sẽ trả `insufficient_evidence` khi không đạt. Không đánh giá Test trước khi chốt cấu hình này.

