# Đánh giá chính thức trên tập Test (chạy một lần)

_Sinh bởi `scripts/96_final_test.py --stage report` lúc 2026-10-26 22:52. Lần chạy Test: bắt đầu 2026-10-26T22:23:36, kết thúc 2026-10-26T22:50:19. Cấu hình khóa trước khi chạy: commit `2e42271f5b`, `reports/final_test/locked_config.json`; kiểm tra độc lập: `reports/final_test/kiem_tra_doc_lap.md`. Không huấn luyện lại, không đổi checkpoint, ngưỡng, cách gộp hay cổng._

**Cách đếm (mỗi BN thuộc đúng một nhóm):** *Đúng* / *Sai* (hệ thống trả nhãn) · *Từ chối* (Evidence Gate không cho nhãn) · *YOLO không có vùng* (không phát hiện vùng nào). Ca bị từ chối hoặc không có vùng **không** được tính là đúng. Pipeline chỉ dùng box của YOLO; nhãn chuẩn (GT) chỉ dùng **sau** khi chạy để chấm điểm. Kết quả *oracle* (phân loại trên vùng cắt theo GT) chỉ để đối chiếu, **không** phải kết quả của hệ thống.

## 1. CT (PLC-CECT) — HCC / ICC / CHCC, Evidence Gate bật

Test CT: 54 BN = 42 BN ung thư (CHCC 13, HCC 14, ICC 15) + 12 BN nhóm chứng CN. Volume bị loại theo đúng quy tắc của Val: excluded:ct_file_missing 9, excluded:lesion_mask_empty 1. Lỗi khi chạy: 0.

### 1a. Nhóm ung thư

| | Đúng | Sai | Từ chối phân loại | YOLO không có vùng | Accuracy trên toàn bộ BN (KTC 95%) | Accuracy trên ca có nhãn | Macro-F1 (bỏ sót = sai) |
|---|---|---|---|---|---|---|---|
| **Test** | 38 | 2 | 2 | 0 | 38/42 (90.5%) (77.9–96.2%) | 38/40 (95.0%) | 0.927 (bootstrap 0.839–0.988) |
| Val (tham chiếu) | 36 | 4 | 2 | 0 | 36/42 (85.7%) | 36/40 (90.0%) | 0.877 |

| Lớp | Precision | Recall | F1 | Số BN |
|---|---|---|---|---|
| HCC | 0.875 | 1.000 | 0.933 | 14 |
| ICC | 1.000 | 0.800 | 0.889 | 15 |
| CHCC | 1.000 | 0.923 | 0.960 | 13 |

- Sai: P0120 (ICC→HCC; oracle ICC), P0237 (CHCC→HCC; oracle ICC).
- Bị từ chối: P0129 (ICC; conf cao nhất 0.48, 1 pha đạt), P0183 (ICC; conf cao nhất 0.54, 1 pha đạt).
- **YOLO bỏ sót:** BN ung thư không có vùng nào 0; BN ung thư mà không vùng nào trùng tổn thương thật 2/42 (P0141, P0221; nhãn của các BN này, nếu có, chỉ đến từ vùng giả).
- **Đúng nhưng không tìm thấy tổn thương thật** (nhãn đến từ vùng giả, đúng do may): P0141, P0221. Nếu chỉ tính đúng khi YOLO tìm thấy tổn thương thật: Test 36/42 (85.7%), Val 36/42 (85.7%).

### 1b. Nhóm chứng CN (không có lớp đúng trong mô hình 3 lớp)

| | Bị gán nhãn ung thư (cảnh báo sai, KTC 95%) | Từ chối phân loại | YOLO không có vùng | Không bị gán nhãn ung thư |
|---|---|---|---|---|
| **Test** | 2/12 (16.7%) (4.7–44.8%) | 8 | 2 | 10/12 (83.3%) |
| Val (tham chiếu) | 2/12 (16.7%) | | | |

CN bị gán nhãn: P0287 (ICC), P0342 (ICC).

### 1c. Toàn bộ Test CT

- Ung thư phân loại đúng + CN không bị gán nhãn ung thư: 48/54 (88.9%). Nhãn sai (ung thư sai lớp + CN bị gán nhãn): 4/54 (7.4%). BN ung thư không có nhãn: 2/54 (3.7%).
  Đây không phải accuracy 4 lớp (hệ thống không có lớp CN).

**Phát hiện (YOLO, conf 0.12):** volume có tổn thương được tìm thấy 146/159 (91.8%); tổn thương 3D 184/288 (63.9%); tổn thương nhỏ (< 32×32 px) 11/71 (15.5%); vùng giả trung bình 4.3/volume. Val (tham chiếu): volume 152/163 (93.3%); tổn thương 3D 177/275 (64.4%); tổn thương nhỏ 14/69 (20.3%); vùng giả 5.0/volume.

**Oracle (vùng cắt theo GT — chỉ để đối chiếu, không phải kết quả hệ thống):** 41/42 (97.6%), macro-F1 0.976.

**Ablation tắt cổng (chỉ để mô tả; cấu hình đã khóa trước, không dùng để chọn):** BN ung thư đúng 40/42 (95.2%); CN bị gán nhãn ung thư 10/12 (83.3%).

![CM pipeline](figures/cm_test_ct_pipeline.png) ![CM oracle](figures/cm_test_ct_oracle.png)

## 2. MRI (LLD-MMRI) — THÍ NGHIỆM NGHIÊN CỨU, CHƯA ĐẠT YÊU CẦU

> Phân loại MRI 7 lớp **không** đạt yêu cầu trên Validation (ROI accuracy 0.478) và **không** được tích hợp làm chức năng chẩn đoán của website. Kết quả dưới đây chỉ công bố như một thí nghiệm nghiên cứu. Evidence Gate không bật cho MRI.

Test MRI: 104 BN (FNH 10, Áp xe gan 11, Nang gan 11, U máu gan 16, Di căn gan 11, HCC 33, ICC 12). Volume bị loại (cách ly bbox, như Val): quarantined:json_box_vs_mask_mismatch 4. Lỗi khi chạy: 0.

| | Đúng | Sai | Từ chối | YOLO không có vùng | Accuracy trên toàn bộ BN (KTC 95%) | Macro-F1 | Balanced accuracy |
|---|---|---|---|---|---|---|---|
| **Test** | 52 | 52 | 0 | 0 | 52/104 (50.0%) (40.6–59.4%) | 0.385 (bootstrap 0.273–0.485) | 0.375 |
| Val (tham chiếu) | 37 | 41 | 0 | 0 | 37/78 (47.4%) | 0.358 | 0.370 |

| Lớp | Precision | Recall | F1 | Số BN |
|---|---|---|---|---|
| HCC | 0.492 | 0.939 | 0.646 | 33 |
| ICC | 0.444 | 0.333 | 0.381 | 12 |
| Di căn gan | 0.667 | 0.182 | 0.286 | 11 |
| U máu gan | 0.471 | 0.500 | 0.485 | 16 |
| Nang gan | 0.667 | 0.182 | 0.286 | 11 |
| Áp xe gan | 1.000 | 0.091 | 0.167 | 11 |
| FNH | 0.500 | 0.400 | 0.444 | 10 |

- Nhóm lành / ác tính (tham khảo): đúng nhóm 75/104 (72.1%); tổn thương lành bị gán lớp ác tính (cảnh báo ung thư sai) 24/48; tổn thương ác tính bị gán lớp lành 5/56.
- **YOLO bỏ sót:** BN không có vùng nào 0; BN mà không vùng nào trùng tổn thương thật 0/104.
- **Đúng nhưng không tìm thấy tổn thương thật** (nhãn đến từ vùng giả, đúng do may): không có. Nếu chỉ tính đúng khi YOLO tìm thấy tổn thương thật: Test 52/104 (50.0%), Val 37/78 (47.4%).

**Phát hiện (YOLO, conf 0.12):** volume có tổn thương được tìm thấy 729/828 (88.0%); tổn thương 3D 741/873 (84.9%); tổn thương nhỏ (< 32×32 px) 231/334 (69.2%); vùng giả trung bình 4.2/volume. Val (tham chiếu): volume 540/622 (86.8%); tổn thương 3D 554/648 (85.5%); tổn thương nhỏ 188/266 (70.7%); vùng giả 4.5/volume.

**Oracle (vùng cắt theo GT — chỉ để đối chiếu, không phải kết quả hệ thống):** 52/104 (50.0%), macro-F1 0.453.

![CM pipeline](figures/cm_test_mri_pipeline.png) ![CM oracle](figures/cm_test_mri_oracle.png)

## 3. Mức sẵn sàng cho giai đoạn website

Tiêu chí kỹ thuật (đặt trước khi chạy Test): (1) pipeline chạy hết mọi BN Test không lỗi; (2) chạy đúng cấu hình đã khóa; (3) mọi trạng thái không nhãn đều có thông điệp không kết luận khỏe mạnh.

- (1) Đạt: lỗi CT 0, MRI 0.
- (2) Đạt: khóa được kiểm tra lại trước khi chạy (commit `2e42271f5b`).
- (3) Đạt: trạng thái `insufficient_evidence` / `no_lesion_detected` luôn kèm câu "không kết luận khỏe mạnh" (`tests/test_pipeline.py`).
- CT: tích hợp **phát hiện + Tumor Zoom + phân loại 3 lớp có Evidence Gate**, luôn kèm cảnh báo "không phải chẩn đoán" và "không có lớp không tổn thương".
- MRI: chỉ tích hợp **phát hiện + Tumor Zoom**; phân loại MRI không đưa vào chức năng chẩn đoán.
- Đây là mức sẵn sàng **kỹ thuật cho một hệ thống nghiên cứu**, không phải đánh giá lâm sàng.

## 4. Hạn chế

- Tập Test nhỏ (khoảng tin cậy rộng, đặc biệt nhóm CN). Một lần chạy, một seed huấn luyện.
- CN là nhóm "không có bệnh gan mật" theo nguồn dữ liệu, không phải "khỏe mạnh hoàn toàn". Từ chối phân loại không phải kết luận âm tính.
- Kiểm tra ảnh trùng chỉ phát hiện bản sao y hệt giữa các split.
- YOLO CT chỉ tìm thấy 11/71 (15.5%) tổn thương nhỏ (< 32×32 px); nhãn mức BN được gộp từ mọi vùng YOLO của BN, nên có thể đúng hoặc sai ngay cả khi tổn thương thật bị bỏ sót.
- CT Test (accuracy 90.5%) cao hơn Val (85.7%) nằm trong dao động do mẫu nhỏ (khoảng tin cậy chồng lấn); không nên hiểu là mô hình tốt hơn trên Test. Nếu chỉ tính đúng khi tổn thương thật được tìm thấy: Test 36/42, Val 36/42.

Tệp: `reports/final_test/metrics_test.json` (chỉ số tổng hợp), `patients.csv`, `volumes.csv` (từng BN / volume; không commit), `figures/` (confusion matrix, hình từng ca), dữ liệu từng ca `outputs/final_test/<ct|mri>/cases/`.
