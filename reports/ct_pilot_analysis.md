# Phân tích YOLO run `pilot_yolo11n_e5` (CT), chỉ trên Validation

*Sinh bởi `scripts/50_analyze_yolo_run.py` lúc 2026-10-20 09:18. Mọi số liệu đọc từ `outputs/runs/yolo_ct/pilot_yolo11n_e5` (tải về từ Colab) và từ lần chạy lại `best.pt` trên tập Val pilot ở local. **Tập Test không được dùng.***

## 1. Cấu hình huấn luyện (args.yaml)

| Tham số | Giá trị |
|---|---|
| model | yolo11n.pt |
| epochs | 5 |
| imgsz | 512 |
| batch | 16 |
| optimizer | auto |
| lr0 | 0.01 |
| lrf | 0.01 |
| cos_lr | False |
| warmup_epochs | 3.0 |
| close_mosaic | 10 |
| mosaic | 1.0 |
| fliplr | 0.5 |
| flipud | 0.0 |
| hsv_h | 0.015 |
| hsv_s | 0.7 |
| hsv_v | 0.4 |
| scale | 0.5 |
| translate | 0.1 |
| degrees | 0.0 |
| amp | True |
| patience | 100 |
| seed | 42 |

## 2. Đường cong huấn luyện (results.csv)

![curves](figures/ct_pilot/curves.png)

| Epoch | train box/cls/dfl | val box/cls/dfl | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| 1 | 1.962 / 4.484 / 1.782 | 1.789 / 5.416 / 1.460 | 0.343 | 0.259 | 0.208 | 0.081 |
| 2 | 1.902 / 2.181 / 1.728 | 1.834 / 5.991 / 1.462 | 0.334 | 0.243 | 0.212 | 0.090 |
| 3 | 1.818 / 1.943 / 1.647 | 1.655 / 3.897 / 1.408 | 0.454 | 0.368 | 0.360 | 0.167 |
| 4 | 1.729 / 1.723 / 1.566 | 1.719 / 5.409 / 1.406 | 0.432 | 0.353 | 0.319 | 0.145 |
| 5 | 1.613 / 1.473 / 1.476 | 1.629 / 4.608 / 1.352 | 0.428 | 0.357 | 0.333 | 0.170 |

- Epoch tốt nhất theo fitness của Ultralytics (= mAP50-95): **5** (mAP50 0.333, mAP50-95 0.170); epoch cuối: mAP50 0.333, mAP50-95 0.170.
- Thay đổi từ epoch đầu đến epoch cuối: {'train/box_loss': -0.3492, 'train/cls_loss': -3.0106, 'train/dfl_loss': -0.3065, 'metrics/precision(B)': 0.0843, 'metrics/recall(B)': 0.0979, 'metrics/mAP50(B)': 0.1251, 'metrics/mAP50-95(B)': 0.0895, 'val/box_loss': -0.1598, 'val/cls_loss': -0.808, 'val/dfl_loss': -0.1082}

## 3. Metrics của `best.pt` trên Val

| Nguồn | conf | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Colab `metrics_val.json` | 0.001 | 0.428 | 0.355 | 0.333 | 0.170 |
| Chạy lại ở local | 0.001 | 0.429 | 0.355 | 0.333 | 0.170 |
| Chạy lại ở local | 0.25 | 0.434 | 0.353 | 0.255 | 0.136 |

Precision/Recall của Ultralytics được tính tại ngưỡng conf cho **F1 cao nhất** trên đường cong, không phải tại một ngưỡng cố định; mAP luôn tính với conf = 0.001.

## 4. Vì sao Confusion Matrix có rất nhiều 'background' dự đoán sai

| conf khi tính ma trận | TP | FP (hàng background) | FN (cột background) |
|---|---|---|---|
| 0.001 | 3365 | 288513 | 485 |
| 0.25 | 1421 | 1707 | 2429 |

- Ultralytics 8.4.182 tính confusion matrix khi validate với `conf = args.conf`, mặc định **0.001** (`engine/validator.py`: `self.args.conf = 0.001`; `detect/val.py`: `process_batch(..., conf=self.args.conf)`), ghép box với GT ở IoU > 0.45. Vì vậy **mọi box có conf > 0.001** (kể cả box gần như chắc chắn là nhiễu) đều được tính; mọi box không khớp GT rơi vào hàng *background* (FP). Trên tập Val có rất nhiều lát âm tính, nên con số này rất lớn.
- Ma trận được chuẩn hóa theo **cột** (cột GT), nên ô background→lesion trông như "100%" khi cột background chỉ chứa FP. Đó là cách hiển thị, không phải tỉ lệ lỗi thực.
- Khi **inference** thực tế (`predict`), ngưỡng mặc định là **0.25** và ngưỡng vận hành sẽ được chọn trên Val. Bảng ở mục 5 cho thấy FP/FN tại từng ngưỡng.

## 5. FP / FN theo ngưỡng confidence (ghép GT ở IoU ≥ 0.5)

| conf ≥ | TP | FP (lát dương) | FP (lát âm) | FN | Precision | Recall | F1 | FP/lát | Lát âm có FP | Lát dương có TP | BN tổn thương được phát hiện | BN chỉ có lát âm mà có FP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | 3260 | 166594 | 114453 | 590 | 0.0115 | 0.8468 | 0.0226 | 38.072 | 3632/3691 (98.4%) | 3172/3691 (85.9%) | 41/42 | 12/12 (CN) |
| 0.05 | 2097 | 6779 | 2790 | 1753 | 0.1798 | 0.5447 | 0.2703 | 1.296 | 1471/3691 (39.9%) | 2078/3691 (56.3%) | 38/42 | 12/12 (CN) |
| 0.1 | 1832 | 3554 | 1369 | 2018 | 0.2712 | 0.4758 | 0.3455 | 0.667 | 941/3691 (25.5%) | 1821/3691 (49.3%) | 36/42 | 10/12 (CN) |
| 0.25 | 1398 | 1139 | 485 | 2452 | 0.4626 | 0.3631 | 0.4069 | 0.22 | 411/3691 (11.1%) | 1398/3691 (37.9%) | 30/42 | 8/12 (CN) |
| 0.4 | 1046 | 467 | 210 | 2804 | 0.6071 | 0.2717 | 0.3754 | 0.092 | 194/3691 (5.3%) | 1046/3691 (28.3%) | 27/42 | 7/12 (CN) |
| 0.5 | 826 | 247 | 107 | 3024 | 0.7 | 0.2145 | 0.3284 | 0.048 | 106/3691 (2.9%) | 826/3691 (22.4%) | 25/42 | 3/12 (CN) |

Tại conf ≥ 0.25: FN theo kích thước GT {'nhỏ <32²px': '563/582', 'vừa 32²–96²px': '1793/2434', 'lớn >96²px': '96/834'}; recall theo nhóm {'class=CHCC': 0.374, 'class=HCC': 0.518, 'class=ICC': 0.197, 'series=C1': 0.307, 'series=C2': 0.417, 'series=C3': 0.386, 'series=P': 0.328}

## 6. Đối chiếu box dự đoán với ground truth

Xanh lá = GT, đỏ nét đứt = dự đoán (conf ≥ 0.25). Có 8 ca FP có confidence cao nhất, 8 ca FN có tổn thương lớn nhất, và 8 ca TP ngẫu nhiên, lưu ở `figures/ct_pilot/case_*.png`.

- **FP:** [P0011_C1_k008](figures/ct_pilot/case_FP_01_P0011_C1_k008.png), [P0035_C1_k081](figures/ct_pilot/case_FP_02_P0035_C1_k081.png), [P0014_P_k029](figures/ct_pilot/case_FP_03_P0014_P_k029.png), [P0077_P_k012](figures/ct_pilot/case_FP_04_P0077_P_k012.png), [P0047_P_k061](figures/ct_pilot/case_FP_05_P0047_P_k061.png), [P0148_P_k112](figures/ct_pilot/case_FP_06_P0148_P_k112.png), [P0091_P_k045](figures/ct_pilot/case_FP_07_P0091_P_k045.png), [P0026_C1_k044](figures/ct_pilot/case_FP_08_P0026_C1_k044.png)
- **FN:** [P0035_C3_k068](figures/ct_pilot/case_FN_01_P0035_C3_k068.png), [P0011_P_k047](figures/ct_pilot/case_FN_02_P0011_P_k047.png), [P0053_P_k038](figures/ct_pilot/case_FN_03_P0053_P_k038.png), [P0018_C2_k018](figures/ct_pilot/case_FN_04_P0018_C2_k018.png), [P0267_C2_k031](figures/ct_pilot/case_FN_05_P0267_C2_k031.png), [P0044_C3_k114](figures/ct_pilot/case_FN_06_P0044_C3_k114.png), [P0027_P_k032](figures/ct_pilot/case_FN_07_P0027_P_k032.png), [P0014_C1_k009](figures/ct_pilot/case_FN_08_P0014_C1_k009.png)
- **TP:** [P0106_C1_k188](figures/ct_pilot/case_TP_01_P0106_C1_k188.png), [P0106_P_k171](figures/ct_pilot/case_TP_02_P0106_P_k171.png), [P0011_C3_k044](figures/ct_pilot/case_TP_03_P0011_C3_k044.png), [P0044_C3_k050](figures/ct_pilot/case_TP_04_P0044_C3_k050.png), [P0171_P_k027](figures/ct_pilot/case_TP_05_P0171_P_k027.png), [P0158_C2_k071](figures/ct_pilot/case_TP_06_P0158_C2_k071.png), [P0106_C2_k214](figures/ct_pilot/case_TP_07_P0106_C2_k214.png), [P0047_C2_k051](figures/ct_pilot/case_TP_08_P0047_C2_k051.png)

<!-- ANALYST-NOTES -->

## 7. Nhận định và đề xuất

*Mọi số liệu dưới đây lấy từ các file của run: `results.csv`, `metrics_val.json`, `args.yaml`, biểu đồ của Ultralytics, và
`analysis/analysis_summary.json` cùng `analysis/ct_fp_fn_breakdown.json` (`scripts/50_analyze_yolo_run.py` và `scripts/51_ct_fp_fn_breakdown.py`).
Đây là **pilot 5 epoch**, chỉ dùng để kiểm tra pipeline. Các con số này **không** phải năng lực cuối cùng của mô hình.*

### 7.1 Precision, Recall, mAP50, mAP50-95
- `best.pt` = **epoch 5** (fitness = mAP50-95 = 0.170, cao nhất; epoch 3 có 0.167). Trên Val pilot (54 BN; 7.382 lát; 3.850 box GT):
  **P 0.428 · R 0.355 · mAP50 0.333 · mAP50-95 0.170**. Chạy lại `best.pt` ở local cho cùng kết quả
  (P 0.429 · R 0.355 · mAP50 0.333 · mAP50-95 0.170), nên kết quả **tái lập được**.
- P/R ở trên được lấy tại ngưỡng **F1 lớn nhất ≈ 0.39, conf ≈ 0.24** (`BoxF1_curve.png`). Bảng ở mục 5 dùng phép ghép một-một
  ở IoU ≥ 0.5 và cho F1 = 0.407 tại conf 0.25, nhất quán với đường F1.
- mAP50-95 (0.170) chỉ khoảng một nửa mAP50 (0.333): box đã đặt **đúng vùng nhưng chưa khít**. Recall theo ngưỡng IoU
  (conf ≥ 0.25) là **0.417 @IoU0.3 → 0.363 @0.5 → 0.269 @0.7**.
- Đường PR **dừng ở recall khoảng 0.85**. Ở conf 0.001, khoảng **15% box GT** không có dự đoán nào với IoU ≥ 0.5 (590/3.850).

### 7.2 Loss và độ ổn định khi huấn luyện
- Loss train giảm đều qua 5 epoch: box 1.96→1.61, cls 4.48→1.47, dfl 1.78→1.48. **Chưa bão hòa**, mô hình còn học tiếp được.
- Loss val giảm nhưng **dao động**, đặc biệt `val/cls_loss`: 5.42 → 5.99 → 3.90 → 5.41 → 4.61. mAP cũng dao động
  (mAP50 0.360 ở epoch 3 → 0.319 → 0.333). Nguyên nhân phù hợp với cấu hình đã xác minh:
  - **Chỉ 5 epoch, trong đó 3 epoch là warmup.** Cột `lr/pg0` xác nhận lr0 thực tế là **0.002 (AdamW do `optimizer=auto`)**:
    epoch 4 có lr = 0.000812 = 0.002 × 0.406, epoch 5 có lr = 0.000416 = 0.002 × 0.208 (lr giảm tuyến tính, `lrf = 0.01`).
  - Lr đạt đỉnh khoảng 0.0012 ở epoch 3 rồi giảm ngay; mô hình gần như không có giai đoạn học ổn định.
  - `close_mosaic = 10 ≥ 5` nên mosaic tắt suốt run.
- **Không có dấu hiệu overfitting** (loss val không tăng theo xu hướng). Vấn đề chính là **chưa huấn luyện đủ**.
- Thời gian: khoảng **200 giây/epoch** trên Colab T4 (cột `time`: 210 / 202 / 201 / 202 / 199 giây), gồm train 17.814 ảnh và val 7.382 ảnh.

### 7.3 Confusion Matrix: vì sao "background" lớn
- Ma trận của Colab (`confusion_matrix.png`): **TP 3.366 · FP 287.448 · FN 484**, được tính ở **conf 0.001** (IoU > 0.45).
  TP + FN = 3.850 = đúng số box GT của Val, nên ma trận không bị lỗi dữ liệu.
- Lý do (đã kiểm chứng trong source Ultralytics 8.4.182): khi validate, `conf` mặc định là 0.001 và confusion matrix dùng
  chính ngưỡng đó. Ở mức này có trung bình **38 FP/lát** (mục 5, conf 0.001), và mọi box không khớp GT bị tính vào hàng
  background. Ô "1.00" trong ma trận chuẩn hóa là do chuẩn hóa theo cột: cột background chỉ chứa FP.
- Chạy lại ở local: conf 0.001 → **TP 3.365 · FP 288.513 · FN 485** (gần như trùng Colab); conf **0.25** (ngưỡng mặc định của `predict`)
  → **TP 1.421 · FP 1.707 · FN 2.429**. FP giảm khoảng 169 lần, đổi lại FN tăng.
  Đây mới là bức tranh gần với lúc suy luận. Ngưỡng vận hành sẽ được chọn trên Val sau khi train chính thức.

### 7.4 False positive (conf ≥ 0.25, IoU ≥ 0.5; tổng FP = 1.624)

| Loại FP | Số lượng | Tỉ lệ |
|---|---|---|
| **Lỗi định vị**: box chồng lên GT nhưng IoU 0.1–0.5 (box lệch, quá to, hoặc chỉ bao một phần khối u) | 763 | 47% |
| Trong gan (liver ∪ lesion), trên lát dương tính | 229 | 14% |
| Trong gan, trên lát âm tính của BN ung thư (cách lát có annotation ≥ 2 lát) | 336 | 21% |
| Trong gan, BN CN | 65 | 4% |
| Ngoài gan (lát dương / lát âm của BN ung thư / BN CN) | 147 / 72 / 12 | 14% |

- Liver mask GT chỉ được dùng để **mô tả lỗi trên Val**. Pipeline suy luận không dùng mask này.
- Gần một nửa số FP thực chất là **cùng một tổn thương nhưng box chưa khít**. Loại FP này giảm khi định vị tốt lên
  (đây cũng là lý do mAP50-95 thấp).
- FP ngoài gan (14%): trên ảnh minh họa, đó là các box trùm lên cơ quan đặc, đồng nhất ở phía đối diện gan
  (`case_FP_07_P0091_P_k045.png`, `case_FP_02_P0035_C1_k081.png`). Có thể giảm loại này bằng cách giới hạn vùng tìm kiếm trong gan,
  nhưng vùng gan lúc suy luận phải lấy từ một mô hình phân đoạn gan riêng, **không** dùng GT.
- 336 FP nằm trong gan trên lát âm tính của BN ung thư. Tôi **không** suy luận đây là tổn thương chưa được annotate. Theo quy tắc
  của đồ án, chúng được tính là FP. Có thể rà soát trực quan một mẫu ca ở bước phân tích run chính thức.
- Mức lát: ở conf 0.25 có **11.1% lát âm tính** có ít nhất một FP; **8/12 BN CN** có ít nhất một lát có FP.

### 7.5 False negative (conf ≥ 0.25; tổng FN = 2.452 / 3.850 box GT)
- **Phụ thuộc mạnh vào kích thước tổn thương.** Tỉ lệ bỏ sót: nhỏ (< 32² px) **563/582 (97%)**, vừa (32²–96²) **1.793/2.434 (74%)**,
  lớn (> 96²) **96/834 (12%)**. Ở conf 0.1, tỉ lệ này vẫn là 91% / 59% / 5%.
- **86% FN (2.103) không có box nào chồng lên.** Chỉ 14% (349) là "bắt gần đúng" (IoU 0.1–0.5). Kết hợp với recall 0.847 ở conf 0.001,
  có thể thấy phần lớn tổn thương *có* được mô hình đề xuất, nhưng với **confidence rất thấp**.
- **Theo lớp bệnh** (chỉ là mô tả, YOLO không phân biệt lớp): recall HCC 0.518, CHCC 0.374, **ICC 0.197**.
  **Theo pha**: C2 0.417, C3 0.386, P 0.328, C1 0.307.
- Mức bệnh nhân: ở conf 0.25, **30/42 BN có tổn thương** có ít nhất một tổn thương được phát hiện (IoU ≥ 0.5); con số này là 36/42 ở conf 0.1.
- Ảnh FN (`case_FN_*`): tổn thương lớn thường chỉ được bao **một phần** (P0035_C3_k068, P0044_C3_k114), hoặc box rơi vào
  vùng khác của gan (P0018_C2_k018). Trên các ảnh đã xem, tôi **không thấy dấu hiệu lỗi nhãn có hệ thống**.

### 7.6 Đề xuất cho training chính thức (đã đưa vào `docs/PHASE3_TRAINING_PLAN.md`)

| Ưu tiên | Đề xuất | Căn cứ trong run này |
|---|---|---|
| 1 | **Train lâu hơn**: 100 epoch, early stopping patience 20 | Loss train chưa bão hòa; 3/5 epoch là warmup; recall 0.847 ở conf 0.001 nhưng chỉ 0.363 ở conf 0.25 (confidence thấp) |
| 2 | **Optimizer tường minh**: SGD lr0 0.01, `cos_lr`, warmup 3 epoch | `auto` đã chọn AdamW lr 0.002; ở run dài, `auto` lại chọn MuSGD, nên các run không so sánh được |
| 3 | **Dùng nhiều dữ liệu hơn**: CT train toàn bộ lát dương tính (hoặc 1/2), **Val đầy đủ** | Pilot chỉ dùng 26% số lát dương tính của train |
| 4 | Bật **mosaic** (tắt ở 10 epoch cuối) và scale augmentation | Mosaic không hoạt động trong pilot; mosaic và scale giúp đa dạng kích thước tổn thương |
| 5 | **Tổn thương nhỏ**: thử `imgsz = 640` (E3) và so sánh `yolo11s` với `yolo11n` (E1) | 97% tổn thương < 32² px bị bỏ sót ở conf 0.25 |
| 6 | **Chọn ngưỡng vận hành trên Val**, ưu tiên recall (ví dụ conf 0.1–0.25), vì FP còn được nhánh classifier và bước ROI lọc tiếp | F1 lớn nhất ở conf ≈ 0.24; ở conf 0.1 recall 0.476, BN được phát hiện 36/42 |
| 7 | (Sau GĐ này) Giới hạn detection trong vùng gan dự đoán (mô hình liver segmentation riêng) | 14% FP nằm ngoài gan |
| 8 | Theo dõi riêng **ICC** và **tổn thương nhỏ** trong mọi báo cáo phân tích | Recall ICC 0.197, thấp nhất |

- Ước tính thời gian (ngoại suy tuyến tính từ 200 giây/epoch, ước lượng thô): CT train 1/2 + Val đầy đủ (khoảng 49k lát) ≈ **6–7 phút/epoch**;
  toàn bộ (khoảng 84k lát) ≈ **11 phút/epoch**. Early stopping thường dừng trước 100 epoch. Resume trên Drive đã được kiểm thử.
- **Không** đề xuất sửa nhãn. Không dùng Test ở bất kỳ bước nào trên.
