# Phân tích YOLO run `pilot_yolo11n_e5` (MRI), chỉ trên Validation

*Sinh bởi `scripts/50_analyze_yolo_run.py` lúc 2026-10-20 18:27. Mọi số liệu đọc từ `outputs/runs/yolo_mri/pilot_yolo11n_e5` (tải về từ Colab) và từ lần chạy lại `best.pt` trên tập Val pilot ở local. **Tập Test không được dùng.***

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

![curves](figures/mri_pilot/curves.png)

| Epoch | train box/cls/dfl | val box/cls/dfl | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| 1 | 1.509 / 2.960 / 1.365 | 1.699 / 2.220 / 1.397 | 0.588 | 0.442 | 0.475 | 0.263 |
| 2 | 1.457 / 1.608 / 1.329 | 1.545 / 1.802 / 1.295 | 0.663 | 0.548 | 0.612 | 0.387 |
| 3 | 1.386 / 1.428 / 1.283 | 1.550 / 1.587 / 1.319 | 0.713 | 0.559 | 0.635 | 0.381 |
| 4 | 1.311 / 1.237 / 1.228 | 1.406 / 1.487 / 1.210 | 0.744 | 0.591 | 0.685 | 0.453 |
| 5 | 1.212 / 1.036 / 1.170 | 1.365 / 1.356 / 1.190 | 0.766 | 0.630 | 0.712 | 0.472 |

- Epoch tốt nhất theo fitness của Ultralytics (= mAP50-95): **5** (mAP50 0.712, mAP50-95 0.472); epoch cuối: mAP50 0.712, mAP50-95 0.472.
- Thay đổi từ epoch đầu đến epoch cuối: {'train/box_loss': -0.2972, 'train/cls_loss': -1.9235, 'train/dfl_loss': -0.1955, 'metrics/precision(B)': 0.1786, 'metrics/recall(B)': 0.1877, 'metrics/mAP50(B)': 0.2376, 'metrics/mAP50-95(B)': 0.2088, 'val/box_loss': -0.3337, 'val/cls_loss': -0.8646, 'val/dfl_loss': -0.2065}

## 3. Metrics của `best.pt` trên Val

| Nguồn | conf | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Colab `metrics_val.json` | 0.001 | 0.766 | 0.630 | 0.712 | 0.471 |
| Chạy lại ở local | 0.001 | 0.766 | 0.630 | 0.712 | 0.471 |
| Chạy lại ở local | 0.25 | 0.766 | 0.630 | 0.625 | 0.423 |

Precision/Recall của Ultralytics được tính tại ngưỡng conf cho **F1 cao nhất** trên đường cong, không phải tại một ngưỡng cố định; mAP luôn tính với conf = 0.001.

## 4. Vì sao Confusion Matrix có rất nhiều 'background' dự đoán sai

| conf khi tính ma trận | TP | FP (hàng background) | FN (cột background) |
|---|---|---|---|
| 0.001 | 4654 | 176932 | 226 |
| 0.25 | 3339 | 1405 | 1541 |

- Ultralytics 8.4.182 tính confusion matrix khi validate với `conf = args.conf`, mặc định **0.001** (`engine/validator.py`: `self.args.conf = 0.001`; `detect/val.py`: `process_batch(..., conf=self.args.conf)`), ghép box với GT ở IoU > 0.45. Vì vậy **mọi box có conf > 0.001** (kể cả box gần như chắc chắn là nhiễu) đều được tính; mọi box không khớp GT rơi vào hàng *background* (FP). Trên tập Val có rất nhiều lát âm tính, nên con số này rất lớn.
- Ma trận được chuẩn hóa theo **cột** (cột GT), nên ô background→lesion trông như "100%" khi cột background chỉ chứa FP. Đó là cách hiển thị, không phải tỉ lệ lỗi thực.
- Khi **inference** thực tế (`predict`), ngưỡng mặc định là **0.25** và ngưỡng vận hành sẽ được chọn trên Val. Bảng ở mục 5 cho thấy FP/FN tại từng ngưỡng.

## 5. FP / FN theo ngưỡng confidence (ghép GT ở IoU ≥ 0.5)

| conf ≥ | TP | FP (lát dương) | FP (lát âm) | FN | Precision | Recall | F1 | FP/lát | Lát âm có FP | Lát dương có TP | BN tổn thương được phát hiện | BN chỉ có lát âm mà có FP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | 4615 | 107685 | 56901 | 265 | 0.0273 | 0.9457 | 0.053 | 21.971 | 2281/2617 (87.2%) | 4613/4874 (94.6%) | 78/78 | 0/0 |
| 0.05 | 3899 | 5352 | 1950 | 981 | 0.3481 | 0.799 | 0.4849 | 0.975 | 953/2617 (36.4%) | 3898/4874 (80.0%) | 77/78 | 0/0 |
| 0.1 | 3708 | 2964 | 1004 | 1172 | 0.4831 | 0.7598 | 0.5906 | 0.53 | 659/2617 (25.2%) | 3708/4874 (76.1%) | 77/78 | 0/0 |
| 0.25 | 3299 | 1058 | 339 | 1581 | 0.7025 | 0.676 | 0.689 | 0.186 | 298/2617 (11.4%) | 3299/4874 (67.7%) | 75/78 | 0/0 |
| 0.4 | 2872 | 461 | 138 | 2008 | 0.8274 | 0.5885 | 0.6878 | 0.08 | 131/2617 (5.0%) | 2872/4874 (58.9%) | 72/78 | 0/0 |
| 0.5 | 2544 | 237 | 83 | 2336 | 0.8883 | 0.5213 | 0.657 | 0.043 | 81/2617 (3.1%) | 2544/4874 (52.2%) | 68/78 | 0/0 |

Tại conf ≥ 0.25: FN theo kích thước GT {'nhỏ <32²px': '657/1275', 'vừa 32²–96²px': '756/2615', 'lớn >96²px': '168/990'}; recall theo nhóm {'class=FOCAL_NODULAR_HYPERPLASIA': 0.562, 'class=Hepatic_abscess': 0.779, 'class=Hepatic_cyst': 0.457, 'class=Hepatic_hemangioma': 0.613, 'class=Hepatic_metastasis': 0.526, 'class=Hepatocellular_carcinoma': 0.724, 'class=Intrahepatic_cholangiocarcinoma': 0.69, 'series=C+A': 0.633, 'series=C+Delay': 0.685, 'series=C+V': 0.668, 'series=C-pre': 0.695, 'series=DWI': 0.744, 'series=In Phase': 0.678, 'series=Out Phase': 0.734, 'series=T2WI': 0.612}

## 6. Đối chiếu box dự đoán với ground truth

Xanh lá = GT, đỏ nét đứt = dự đoán (conf ≥ 0.25). Có 8 ca FP có confidence cao nhất, 8 ca FN có tổn thương lớn nhất, và 8 ca TP ngẫu nhiên, lưu ở `figures/mri_pilot/case_*.png`.

- **FP:** [MR209281_C-pre_k020](figures/mri_pilot/case_FP_01_MR209281_C-pre_k020.png), [MR80569_InPhase_k010](figures/mri_pilot/case_FP_02_MR80569_InPhase_k010.png), [MR49502_InPhase_k010](figures/mri_pilot/case_FP_03_MR49502_InPhase_k010.png), [MR207755_OutPhase_k018](figures/mri_pilot/case_FP_04_MR207755_OutPhase_k018.png), [MR32504_C+V_k034](figures/mri_pilot/case_FP_05_MR32504_C+V_k034.png), [MR132628_InPhase_k021](figures/mri_pilot/case_FP_06_MR132628_InPhase_k021.png), [MR145114_C+Delay_k024](figures/mri_pilot/case_FP_07_MR145114_C+Delay_k024.png), [MR137627_DWI_k008](figures/mri_pilot/case_FP_08_MR137627_DWI_k008.png)
- **FN:** [MR49502_C+Delay_k027](figures/mri_pilot/case_FN_01_MR49502_C+Delay_k027.png), [MR78507_T2WI_k003](figures/mri_pilot/case_FN_02_MR78507_T2WI_k003.png), [MR35505_C+A_k045](figures/mri_pilot/case_FN_03_MR35505_C+A_k045.png), [MR82133_C+V_k008](figures/mri_pilot/case_FN_04_MR82133_C+V_k008.png), [MR164939_C-pre_k008](figures/mri_pilot/case_FN_05_MR164939_C-pre_k008.png), [MR209281_C+V_k026](figures/mri_pilot/case_FN_06_MR209281_C+V_k026.png), [MR179970_C-pre_k051](figures/mri_pilot/case_FN_07_MR179970_C-pre_k051.png), [MR162257_C+A_k038](figures/mri_pilot/case_FN_08_MR162257_C+A_k038.png)
- **TP:** [MR189873_InPhase_k005](figures/mri_pilot/case_TP_01_MR189873_InPhase_k005.png), [MR82133_C-pre_k016](figures/mri_pilot/case_TP_02_MR82133_C-pre_k016.png), [MR199345_C+V_k032](figures/mri_pilot/case_TP_03_MR199345_C+V_k032.png), [MR104280_C+V_k045](figures/mri_pilot/case_TP_04_MR104280_C+V_k045.png), [MR172409_C-pre_k018](figures/mri_pilot/case_TP_05_MR172409_C-pre_k018.png), [MR236955_C+V_k034](figures/mri_pilot/case_TP_06_MR236955_C+V_k034.png), [MR210196_C-pre_k032](figures/mri_pilot/case_TP_07_MR210196_C-pre_k032.png), [MR192701_OutPhase_k019](figures/mri_pilot/case_TP_08_MR192701_OutPhase_k019.png)

<!-- ANALYST-NOTES -->

## 7. Nhận định và đề xuất

*Mọi số liệu lấy từ `outputs/runs/yolo_mri/pilot_yolo11n_e5/`: `results.csv`, `metrics_val.json` (dataset SHA256 `0088fa93…` khớp với
`yolo_mri_pilot.zip`), các biểu đồ, `analysis/analysis_summary.json` và `analysis/mri_fp_fn_breakdown.json`.
Đây là **pilot 5 epoch**. Các con số không phải năng lực cuối cùng của mô hình.*

### 7.1 Precision, Recall, mAP50, mAP50-95
- `best.pt` = **epoch 5**. Trên Val (78 BN, val chính thức của LLD-MMRI2023; 7.491 lát; 4.880 box GT):
  **P 0.766 · R 0.630 · mAP50 0.712 · mAP50-95 0.471**. Chạy lại ở local cho cùng kết quả.
- Ngưỡng F1 lớn nhất là **0.69 tại conf ≈ 0.33** (`BoxF1_curve.png`). Vì ngưỡng này > 0.25, lần val ở conf 0.25 cho **cùng P/R**
  (0.766 / 0.630). Chỉ mAP giảm (0.625 / 0.423), vì đường PR bị cắt ở 0.25.
- Recall theo IoU (conf ≥ 0.25): **0.704 @0.3 → 0.676 @0.5 → 0.595 @0.7**. Box khít hơn CT (mAP50-95/mAP50 = 0.66 so với 0.51 ở CT).
- Đường PR dừng ở recall khoảng **0.94**. Ở conf 0.001, 265/4.880 box GT (5%) không có dự đoán nào với IoU ≥ 0.5.

### 7.2 Loss và độ ổn định
- **Mọi loss train và val đều giảm** qua 5 epoch (val box 1.70→1.37, cls 2.22→1.36, dfl 1.40→1.19). **Mọi metric đều tăng**
  (mAP50-95 0.263→0.387→0.381→0.453→0.472). Chỉ có dao động nhỏ ở epoch 3.
- **Chưa bão hòa và không overfit.** Như CT, đây là run chưa huấn luyện đủ: `optimizer=auto` → AdamW lr0 0.002
  (lr epoch 4 = 0.000812, epoch 5 = 0.000416); 3/5 epoch là warmup; mosaic tắt (`close_mosaic = 10`).
- Thời gian khoảng **340 giây/epoch** trên T4 (28.595 ảnh train + 7.491 ảnh val).

### 7.3 Confusion Matrix
- Colab, conf 0.001: **TP 4.654 · FP 176.950 · FN 226**; TP + FN = 4.880 = đúng số box GT. Chạy lại ở local cho TP 4.654 · FP 176.932 · FN 226.
- conf 0.25: **TP 3.339 · FP 1.405 · FN 1.541**. FP ở conf 0.001 lớn vì cùng cơ chế với CT (ma trận tính ở conf 0.001;
  trung bình 22 FP/lát ở mức này).

### 7.4 False positive (conf ≥ 0.25; tổng FP = 1.397)

| Loại FP | Số lượng | Tỉ lệ |
|---|---|---|
| Lỗi định vị (IoU với GT 0.1–0.5) | 589 | 42% |
| Không chồng lên GT, trên lát dương tính | 469 | 34% |
| Trên lát âm tính (cách lát có annotation ≥ 2 lát) | 339 | 24% |

- Ở conf 0.25: **11.4% lát âm tính** có FP. MRI không có liver mask nên không chia được FP trong hay ngoài gan.
  Trên ảnh minh họa có FP ở phía cơ quan đặc đối diện gan (`case_FP_01_MR209281_C-pre_k020.png`).
- Mọi BN MRI đều có tổn thương, nên **không có nhóm "BN chỉ có lát âm"** như CN của CT (cột 0/0 ở mục 5).
- Lỗi định vị chiếm phần lớn. Trên ảnh: box quá lớn so với tổn thương nhỏ (MR137627 DWI), hoặc chỉ bao một phần (MR49502).

### 7.5 False negative (conf ≥ 0.25; tổng FN = 1.581)
- **Theo kích thước:** nhỏ (< 32²) **657/1.275 (52%)**, vừa **756/2.615 (29%)**, lớn **168/990 (17%)**.
  Tổn thương nhỏ vẫn là nhóm khó nhất. Tỉ lệ box nhỏ ở MRI val là 26%, so với 15% ở CT val.
- 87% FN (1.378) không có box nào chồng lên; 13% (203) là bắt gần đúng.
- **Theo lớp** (mô tả, YOLO không phân biệt lớp): abscess 0.779, HCC 0.724, ICC 0.690, hemangioma 0.613, FNH 0.562,
  metastasis 0.526, **cyst 0.457** (thấp nhất).
- **Theo chuỗi:** DWI 0.744, Out Phase 0.734, C-pre 0.695, C+Delay 0.685, In Phase 0.678, C+V 0.668, C+A 0.633, **T2WI 0.612**.
- Mức bệnh nhân: **75/78** BN có ít nhất một tổn thương được phát hiện ở conf 0.25 (77/78 ở conf 0.1).
- Ảnh FN: nhiều tổn thương **lớn** bị bỏ sót hoặc chỉ được bao một phần (MR78507 T2WI, MR35505 C+A, MR164939 C-pre).
  Tôi không thấy dấu hiệu lỗi nhãn có hệ thống.

### 7.6 Đề xuất cho training chính thức MRI
1. **100 epoch + early stopping (patience 20)**, optimizer **SGD tường minh** (lr0 0.01, `cos_lr`), mosaic bật: run còn đang cải thiện ở epoch 5.
2. **Dữ liệu:** dùng lại `yolo_mri_pilot.zip`. Nó đã chứa 93% số lát dương tính hợp lệ và Val đầy đủ; 7% còn lại bị loại vì
   mask và JSON không đồng thuận hoặc là mảnh vụn, và việc loại này đã được người dùng duyệt.
3. **So sánh yolo11n với yolo11s (E0/E1)** ở imgsz 512. Nếu FN ở box nhỏ vẫn cao, thêm E3 `imgsz = 640`.
4. **Ngưỡng vận hành** chọn trên Val. Ở pilot, F1 lớn nhất tại khoảng 0.33; nếu ưu tiên recall, có thể chọn khoảng 0.1–0.25.
5. Theo dõi riêng **cyst, metastasis, FNH** và **T2WI, C+A** trong các phân tích sau.
