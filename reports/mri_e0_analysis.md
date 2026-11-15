# Phân tích YOLO run `yolo11n_e50_img512_baseline` (MRI), chỉ trên Validation

*Sinh bởi `scripts/50_analyze_yolo_run.py` lúc 2026-10-24 18:21. Mọi số liệu đọc từ `outputs/runs/yolo_mri/yolo11n_e50_img512_baseline` (tải về từ Colab) và từ lần chạy lại `best.pt` trên tập Val pilot ở local. **Tập Test không được dùng.***

## 1. Cấu hình huấn luyện (args.yaml)

| Tham số | Giá trị |
|---|---|
| model | /content/drive/MyDrive/LiverCancer_AI/runs/yolo_mri/yolo11n_e50_img512_baseline/weights/last.pt |
| epochs | 50 |
| imgsz | 512 |
| batch | 32 |
| optimizer | SGD |
| lr0 | 0.01 |
| lrf | 0.01 |
| cos_lr | True |
| warmup_epochs | 3.0 |
| close_mosaic | 10 |
| mosaic | 1.0 |
| fliplr | 0.5 |
| flipud | 0.0 |
| hsv_h | 0.0 |
| hsv_s | 0.0 |
| hsv_v | 0.3 |
| scale | 0.5 |
| translate | 0.1 |
| degrees | 0.0 |
| amp | True |
| patience | 20 |
| seed | 42 |

## 2. Đường cong huấn luyện (results.csv)

![curves](figures/mri_e0/curves.png)

| Epoch | train box/cls/dfl | val box/cls/dfl | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| 1 | 1.489 / 2.487 / 1.328 | 1.426 / 1.954 / 1.227 | 0.617 | 0.459 | 0.514 | 0.333 |
| 2 | 1.426 / 1.577 / 1.223 | 1.536 / 1.894 / 1.253 | 0.573 | 0.450 | 0.474 | 0.273 |
| 3 | 1.511 / 1.644 / 1.281 | 1.616 / 2.161 / 1.331 | 0.500 | 0.372 | 0.380 | 0.209 |
| 4 | 1.544 / 1.653 / 1.313 | 1.742 / 2.035 / 1.407 | 0.594 | 0.402 | 0.454 | 0.263 |
| 5 | 1.465 / 1.491 / 1.265 | 1.620 / 1.923 / 1.352 | 0.593 | 0.456 | 0.490 | 0.281 |
| 6 | 1.415 / 1.387 / 1.235 | 1.418 / 1.573 / 1.213 | 0.684 | 0.564 | 0.629 | 0.397 |
| 7 | 1.378 / 1.304 / 1.216 | 1.397 / 1.491 / 1.216 | 0.706 | 0.557 | 0.636 | 0.401 |
| 8 | 1.352 / 1.249 / 1.202 | 1.363 / 1.414 / 1.195 | 0.730 | 0.577 | 0.659 | 0.420 |
| 9 | 1.327 / 1.201 / 1.186 | 1.378 / 1.431 / 1.198 | 0.720 | 0.595 | 0.671 | 0.431 |
| 10 | 1.305 / 1.163 / 1.172 | 1.422 / 1.480 / 1.211 | 0.726 | 0.574 | 0.649 | 0.415 |
| 11 | 1.297 / 1.135 / 1.164 | 1.337 / 1.320 / 1.169 | 0.766 | 0.603 | 0.696 | 0.455 |
| 12 | 1.280 / 1.104 / 1.152 | 1.350 / 1.368 / 1.171 | 0.743 | 0.610 | 0.688 | 0.450 |
| 13 | 1.267 / 1.077 / 1.140 | 1.322 / 1.284 / 1.157 | 0.768 | 0.616 | 0.709 | 0.465 |
| 14 | 1.255 / 1.052 / 1.135 | 1.328 / 1.315 / 1.159 | 0.760 | 0.626 | 0.706 | 0.462 |
| 15 | 1.241 / 1.024 / 1.127 | 1.302 / 1.266 / 1.150 | 0.789 | 0.627 | 0.727 | 0.483 |
| 16 | 1.236 / 1.009 / 1.125 | 1.289 / 1.215 / 1.140 | 0.766 | 0.652 | 0.732 | 0.487 |
| 17 | 1.233 / 0.995 / 1.115 | 1.286 / 1.204 / 1.128 | 0.794 | 0.640 | 0.736 | 0.488 |
| 18 | 1.212 / 0.966 / 1.105 | 1.277 / 1.213 / 1.130 | 0.784 | 0.644 | 0.734 | 0.495 |
| 19 | 1.204 / 0.952 / 1.102 | 1.269 / 1.190 / 1.127 | 0.771 | 0.651 | 0.741 | 0.501 |
| 20 | 1.189 / 0.925 / 1.100 | 1.272 / 1.191 / 1.127 | 0.790 | 0.651 | 0.741 | 0.500 |
| 21 | 1.193 / 0.910 / 1.094 | 1.284 / 1.174 / 1.134 | 0.794 | 0.651 | 0.739 | 0.496 |
| 22 | 1.177 / 0.894 / 1.091 | 1.277 / 1.190 / 1.132 | 0.788 | 0.650 | 0.739 | 0.497 |
| 23 | 1.168 / 0.876 / 1.085 | 1.282 / 1.183 / 1.129 | 0.781 | 0.668 | 0.748 | 0.497 |
| 24 | 1.156 / 0.856 / 1.074 | 1.277 / 1.167 / 1.123 | 0.799 | 0.659 | 0.749 | 0.504 |
| 25 | 1.147 / 0.841 / 1.069 | 1.280 / 1.168 / 1.127 | 0.807 | 0.655 | 0.743 | 0.502 |
| 26 | 1.136 / 0.826 / 1.064 | 1.289 / 1.183 / 1.130 | 0.780 | 0.658 | 0.736 | 0.496 |
| 27 | 1.128 / 0.804 / 1.062 | 1.294 / 1.202 / 1.128 | 0.773 | 0.647 | 0.730 | 0.491 |
| 28 | 1.122 / 0.796 / 1.060 | 1.292 / 1.180 / 1.129 | 0.779 | 0.656 | 0.737 | 0.496 |
| 29 | 1.109 / 0.781 / 1.049 | 1.295 / 1.187 / 1.130 | 0.791 | 0.651 | 0.735 | 0.496 |
| 30 | 1.100 / 0.756 / 1.042 | 1.293 / 1.193 / 1.128 | 0.807 | 0.642 | 0.735 | 0.497 |
| 31 | 1.100 / 0.760 / 1.047 | 1.296 / 1.203 / 1.129 | 0.803 | 0.645 | 0.733 | 0.494 |
| 32 | 1.084 / 0.737 / 1.038 | 1.297 / 1.194 / 1.127 | 0.809 | 0.646 | 0.736 | 0.495 |
| 33 | 1.077 / 0.717 / 1.034 | 1.296 / 1.196 / 1.126 | 0.789 | 0.661 | 0.735 | 0.495 |
| 34 | 1.066 / 0.707 / 1.027 | 1.298 / 1.199 / 1.128 | 0.794 | 0.660 | 0.735 | 0.495 |
| 35 | 1.059 / 0.697 / 1.024 | 1.299 / 1.203 / 1.128 | 0.793 | 0.660 | 0.734 | 0.495 |
| 36 | 1.048 / 0.680 / 1.023 | 1.299 / 1.208 / 1.128 | 0.799 | 0.655 | 0.733 | 0.495 |
| 37 | 1.039 / 0.669 / 1.019 | 1.300 / 1.210 / 1.129 | 0.801 | 0.653 | 0.732 | 0.494 |
| 38 | 1.034 / 0.655 / 1.018 | 1.302 / 1.214 / 1.130 | 0.796 | 0.655 | 0.732 | 0.493 |
| 39 | 1.016 / 0.640 / 1.011 | 1.305 / 1.219 / 1.132 | 0.795 | 0.653 | 0.730 | 0.492 |
| 40 | 1.013 / 0.631 / 1.009 | 1.308 / 1.221 / 1.134 | 0.799 | 0.651 | 0.731 | 0.492 |
| 41 | 0.956 / 0.540 / 0.981 | 1.309 / 1.221 / 1.136 | 0.799 | 0.650 | 0.730 | 0.492 |
| 42 | 0.936 / 0.515 / 0.973 | 1.312 / 1.226 / 1.137 | 0.792 | 0.656 | 0.729 | 0.491 |
| 43 | 0.933 / 0.505 / 0.969 | 1.315 / 1.231 / 1.139 | 0.786 | 0.658 | 0.729 | 0.490 |
| 44 | 0.919 / 0.490 / 0.964 | 1.317 / 1.235 / 1.140 | 0.789 | 0.657 | 0.728 | 0.490 |

- Epoch tốt nhất theo fitness của Ultralytics (= mAP50-95): **24** (mAP50 0.749, mAP50-95 0.504); epoch cuối: mAP50 0.728, mAP50-95 0.490.
- Thay đổi từ epoch đầu đến epoch cuối: {'train/box_loss': -0.5701, 'train/cls_loss': -1.9969, 'train/dfl_loss': -0.3639, 'metrics/precision(B)': 0.172, 'metrics/recall(B)': 0.1986, 'metrics/mAP50(B)': 0.2145, 'metrics/mAP50-95(B)': 0.1564, 'val/box_loss': -0.1091, 'val/cls_loss': -0.719, 'val/dfl_loss': -0.0871}

## 3. Metrics của `best.pt` trên Val

| Nguồn | conf | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Colab `metrics_val.json` | 0.001 | 0.797 | 0.660 | 0.749 | 0.504 |
| Chạy lại ở local | 0.001 | 0.797 | 0.660 | 0.749 | 0.504 |
| Chạy lại ở local | 0.25 | 0.797 | 0.660 | 0.650 | 0.449 |

Precision/Recall của Ultralytics được tính tại ngưỡng conf cho **F1 cao nhất** trên đường cong, không phải tại một ngưỡng cố định; mAP luôn tính với conf = 0.001.

## 4. Vì sao Confusion Matrix có rất nhiều 'background' dự đoán sai

| conf khi tính ma trận | TP | FP (hàng background) | FN (cột background) |
|---|---|---|---|
| 0.001 | 4571 | 84809 | 309 |
| 0.25 | 3422 | 1078 | 1458 |

- Ultralytics 8.4.182 tính confusion matrix khi validate với `conf = args.conf`, mặc định **0.001** (`engine/validator.py`: `self.args.conf = 0.001`; `detect/val.py`: `process_batch(..., conf=self.args.conf)`), ghép box với GT ở IoU > 0.45. Vì vậy **mọi box có conf > 0.001** (kể cả box gần như chắc chắn là nhiễu) đều được tính; mọi box không khớp GT rơi vào hàng *background* (FP). Trên tập Val có rất nhiều lát âm tính, nên con số này rất lớn.
- Ma trận được chuẩn hóa theo **cột** (cột GT), nên ô background→lesion trông như "100%" khi cột background chỉ chứa FP. Đó là cách hiển thị, không phải tỉ lệ lỗi thực.
- Khi **inference** thực tế (`predict`), ngưỡng mặc định là **0.25** và ngưỡng vận hành sẽ được chọn trên Val. Bảng ở mục 5 cho thấy FP/FN tại từng ngưỡng.

## 5. FP / FN theo ngưỡng confidence (ghép GT ở IoU ≥ 0.5)

| conf ≥ | TP | FP (lát dương) | FP (lát âm) | FN | Precision | Recall | F1 | FP/lát | Lát âm có FP | Lát dương có TP | BN tổn thương được phát hiện | BN chỉ có lát âm mà có FP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | 4509 | 52788 | 25807 | 371 | 0.0543 | 0.924 | 0.1025 | 10.492 | 1886/2617 (72.1%) | 4507/4874 (92.5%) | 78/78 | 0/0 |
| 0.05 | 3864 | 3187 | 999 | 1016 | 0.48 | 0.7918 | 0.5977 | 0.559 | 624/2617 (23.8%) | 3863/4874 (79.3%) | 77/78 | 0/0 |
| 0.1 | 3681 | 1848 | 572 | 1199 | 0.6033 | 0.7543 | 0.6704 | 0.323 | 433/2617 (16.5%) | 3680/4874 (75.5%) | 77/78 | 0/0 |
| 0.25 | 3348 | 796 | 256 | 1532 | 0.7609 | 0.6861 | 0.7216 | 0.14 | 226/2617 (8.6%) | 3348/4874 (68.7%) | 75/78 | 0/0 |
| 0.4 | 3042 | 429 | 140 | 1838 | 0.8424 | 0.6234 | 0.7165 | 0.076 | 135/2617 (5.2%) | 3042/4874 (62.4%) | 74/78 | 0/0 |
| 0.5 | 2809 | 270 | 95 | 2071 | 0.885 | 0.5756 | 0.6975 | 0.049 | 93/2617 (3.6%) | 2809/4874 (57.6%) | 71/78 | 0/0 |

Tại conf ≥ 0.25: FN theo kích thước GT {'nhỏ <32²px': '627/1275', 'vừa 32²–96²px': '699/2615', 'lớn >96²px': '206/990'}; recall theo nhóm {'class=FOCAL_NODULAR_HYPERPLASIA': 0.55, 'class=Hepatic_abscess': 0.811, 'class=Hepatic_cyst': 0.443, 'class=Hepatic_hemangioma': 0.622, 'class=Hepatic_metastasis': 0.554, 'class=Hepatocellular_carcinoma': 0.733, 'class=Intrahepatic_cholangiocarcinoma': 0.696, 'series=C+A': 0.643, 'series=C+Delay': 0.676, 'series=C+V': 0.688, 'series=C-pre': 0.663, 'series=DWI': 0.846, 'series=In Phase': 0.688, 'series=Out Phase': 0.681, 'series=T2WI': 0.707}

## 6. Đối chiếu box dự đoán với ground truth

Xanh lá = GT, đỏ nét đứt = dự đoán (conf ≥ 0.25). Có 8 ca FP có confidence cao nhất, 8 ca FN có tổn thương lớn nhất, và 8 ca TP ngẫu nhiên, lưu ở `figures/mri_e0/case_*.png`.

- **FP:** [MR174815_C-pre_k021](figures/mri_e0/case_FP_01_MR174815_C-pre_k021.png), [MR49502_InPhase_k010](figures/mri_e0/case_FP_02_MR49502_InPhase_k010.png), [MR35505_C+V_k019](figures/mri_e0/case_FP_03_MR35505_C+V_k019.png), [MR137627_C-pre_k025](figures/mri_e0/case_FP_04_MR137627_C-pre_k025.png), [MR132628_InPhase_k021](figures/mri_e0/case_FP_05_MR132628_InPhase_k021.png), [MR78507_C+V_k009](figures/mri_e0/case_FP_06_MR78507_C+V_k009.png), [MR209281_T2WI_k011](figures/mri_e0/case_FP_07_MR209281_T2WI_k011.png), [MR34162_DWI_k007](figures/mri_e0/case_FP_08_MR34162_DWI_k007.png)
- **FN:** [MR49502_C+Delay_k027](figures/mri_e0/case_FN_01_MR49502_C+Delay_k027.png), [MR164939_InPhase_k005](figures/mri_e0/case_FN_02_MR164939_InPhase_k005.png), [MR35505_C-pre_k041](figures/mri_e0/case_FN_03_MR35505_C-pre_k041.png), [MR78507_C-pre_k013](figures/mri_e0/case_FN_04_MR78507_C-pre_k013.png), [MR82133_C+V_k008](figures/mri_e0/case_FN_05_MR82133_C+V_k008.png), [MR179970_C+A_k051](figures/mri_e0/case_FN_06_MR179970_C+A_k051.png), [MR209281_C+Delay_k028](figures/mri_e0/case_FN_07_MR209281_C+Delay_k028.png), [MR106372_T2WI_k012](figures/mri_e0/case_FN_08_MR106372_T2WI_k012.png)
- **TP:** [MR189873_C-pre_k019](figures/mri_e0/case_TP_01_MR189873_C-pre_k019.png), [MR80569_C+V_k008](figures/mri_e0/case_TP_02_MR80569_C+V_k008.png), [MR198925_C+Delay_k060](figures/mri_e0/case_TP_03_MR198925_C+Delay_k060.png), [MR104280_C+V_k048](figures/mri_e0/case_TP_04_MR104280_C+V_k048.png), [MR172409_C-pre_k017](figures/mri_e0/case_TP_05_MR172409_C-pre_k017.png), [MR236008_InPhase_k057](figures/mri_e0/case_TP_06_MR236008_InPhase_k057.png), [MR209281_OutPhase_k009](figures/mri_e0/case_TP_07_MR209281_OutPhase_k009.png), [MR192701_InPhase_k017](figures/mri_e0/case_TP_08_MR192701_InPhase_k017.png)

<!-- ANALYST-NOTES -->

## 7. Nhận định và đề xuất (MRI-E0)

*Nguồn số liệu, đều nằm trong `outputs/runs/yolo_mri/yolo11n_e50_img512_baseline/`: `results.csv`, `args.yaml`,
`metrics_val.json` (Colab), các biểu đồ, `analysis/analysis_summary.json` (chạy lại ở local),
`analysis/mri_fp_fn_breakdown.json` (`scripts/51_ct_fp_fn_breakdown.py --modality mri`) và
`analysis/series_level_detection.json` (mức series và BN, `scripts/52_series_level_detection.py`).
Dataset SHA256 `0088fa93…` = `yolo_mri_pilot.zip`, nên **Val E0 trùng hoàn toàn với Val pilot** (78 BN, 7.491 lát, 4.880 box GT)
và hai run so sánh trực tiếp được. Chỉ dùng Val, **không dùng Test**.*

### 7.1 Cấu hình thực tế và quá trình train
- yolo11n (COCO pretrained), **EPOCHS = 50**, patience 20, imgsz 512, batch 32, SGD lr0 0.01 + `cos_lr`, mosaic 1.0 (`close_mosaic = 10`),
  seed 42, AMP, T4, Ultralytics 8.4.182 (`metrics_val.json`). Kế hoạch §8 ban đầu ghi 100 epoch; run thực tế dùng **50** (quyết định của người dùng).
  Early stopping dừng ở epoch 44, trước khi chạm trần 50, nên trần epoch không cắt ngắn run. Tuy nhiên lịch `cos_lr` phụ thuộc vào `EPOCHS`,
  nên không suy ra được kết quả của cấu hình 100 epoch.
- Run bị ngắt sau **epoch 29** và **resume** từ epoch 30: cột `time` trong `results.csv` bắt đầu lại từ 384 s.
  Vì vậy `args.yaml` ghi `model = …/last.pt` và `resume = …/last.pt`; mô hình gốc là `yolo11n.pt`.
  Thời gian khoảng **316 s/epoch** (epoch 1–29: 9.155 s) và khoảng 325 s/epoch sau resume; tổng 44 epoch ≈ **3.9 giờ** T4.
- `weights/best.pt` = **epoch 24** (đã kiểm tra checkpoint: đã strip optimizer, `best_fitness` 0.50374 khớp dòng 24 của `results.csv`).
  **Lưu ý:** `weights/last.pt` (epoch 29, còn optimizer) và `weights/best_backup.pt` (epoch 24) trong thư mục local là **bản sao trước khi resume**,
  không phải checkpoint cuối của run. Bundle của notebook chỉ chứa `best.pt`. Phân tích này chỉ dùng `best.pt`.

### 7.2 Precision, Recall, mAP (best.pt, Val)
| | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Pilot (5 epoch, AdamW auto, batch 16) | 0.766 | 0.630 | 0.712 | 0.471 |
| **E0** (Colab `metrics_val.json`) | **0.797** | **0.660** | **0.749** | **0.504** |
| E0 chạy lại ở local | 0.797 | 0.660 | 0.749 | 0.504 |
| Chênh lệch E0 − pilot | +0.031 | +0.030 | +0.037 | +0.033 |

- P/R của Ultralytics lấy tại ngưỡng F1 lớn nhất: **F1 = 0.72 tại conf 0.326** (`BoxF1_curve.png`), so với 0.69 ở pilot.
  Vì ngưỡng này lớn hơn 0.25, lần val ở conf 0.25 cho cùng P/R; chỉ mAP giảm (0.650 / 0.449) vì đường PR bị cắt.
- Chạy lại ở local cho P/R/mAP trùng đến 3 chữ số. Confusion matrix ở conf 0.001 lệch nhẹ (FP 84.809 ở local so với 84.838 trên Colab;
  TP và FN trùng), do khác GPU (RTX 3050 / T4).
- mAP50-95/mAP50 = 0.67, box khá khít. Recall theo IoU (conf ≥ 0.25): **0.715 @0.3 → 0.686 @0.5 → 0.605 @0.7**.

### 7.3 Loss: hội tụ và overfitting
- Epoch 1–5 có dao động mạnh (mAP50-95 0.333 → 0.209 ở epoch 3), trùng với giai đoạn warmup 3 epoch,
  sau đó metrics tăng đều đến khoảng epoch 19.
- **Val loss nhỏ nhất ở epoch 19–24** (box 1.269 ở ep19; cls 1.167 và dfl 1.123 ở ep24), sau đó **tăng đều** đến epoch 44
  (1.317 / 1.235 / 1.140), trong khi train loss **vẫn giảm** (box 1.156 → 0.919, cls 0.856 → 0.490).
  Đây là dấu hiệu **overfit nhẹ** sau khoảng epoch 24. mAP50-95 giảm từ 0.504 (ep24) xuống 0.490 (ep44).
- Tắt mosaic ở epoch 41 làm train loss giảm thêm (box 1.013 → 0.956), nhưng không cải thiện metrics Val.
- Kết luận: với dữ liệu MRI hiện tại, yolo11n đạt **trần ở khoảng epoch 20–25**, và early stopping (patience 20) hoạt động đúng như thiết kế.

### 7.4 Confusion matrix
- conf 0.001: TP 4.571 · FP 84.838 · FN 309 (TP + FN = 4.880 box GT). FP lớn vì ma trận đếm mọi box có conf > 0.001 (giải thích ở mục 4).
  So với pilot (FP 176.950), E0 sinh **ít box rác hơn một nửa**.
- conf 0.25: **TP 3.422 · FP 1.078 · FN 1.458** (pilot: 3.339 / 1.405 / 1.541).

### 7.5 FP / FN theo ngưỡng (ghép GT ở IoU ≥ 0.5, mục 5)
| conf ≥ | E0 P / R / F1 | Pilot P / R / F1 | E0: lát âm có FP | E0: BN được phát hiện |
|---|---|---|---|---|
| 0.1 | 0.603 / 0.754 / 0.670 | 0.483 / 0.760 / 0.591 | 16.5% | 77/78 |
| 0.25 | 0.761 / 0.686 / 0.722 | 0.703 / 0.676 / 0.689 | 8.6% | 75/78 |
| 0.4 | 0.842 / 0.623 / 0.717 | 0.827 / 0.589 / 0.688 | 5.2% | 74/78 |

- **Phần cải thiện chủ yếu nằm ở precision**: số FP giảm (conf 0.25: 1.052 so với 1.397 FP; lát âm có FP 8.6% so với 11.4%).
  **Recall ở ngưỡng thấp không tăng**: conf 0.1 cho 0.754 so với 0.760; conf 0.001 cho 0.924 so với 0.946.
  Đường PR của E0 dừng ở recall khoảng 0.92. Như vậy khoảng 8% box GT không có dự đoán nào với IoU ≥ 0.5, ở bất kỳ ngưỡng nào.
- **FP (conf ≥ 0.25, tổng 1.052):** lỗi định vị (IoU 0.1–0.5) 504 (**48%**), không chồng GT trên lát dương 292 (28%), trên lát âm 256 (24%).
  Trên ảnh minh họa, box dự đoán lớn hơn tổn thương nhỏ (`case_FP_03_MR35505_C+V_k019.png`, `case_FP_01_MR174815_C-pre_k021.png`).
- **FN (conf ≥ 0.25, tổng 1.532):** 85% không có box nào chồng lên; 15% (225) là bắt gần đúng.
  - Theo kích thước: nhỏ (< 32²) **627/1.275 (49%)**, vừa 699/2.615 (27%), lớn **206/990 (21%)**. Pilot: 52% / 29% / 17%.
    Box nhỏ vẫn là nhóm khó nhất. **Box lớn bị bỏ sót nhiều hơn pilot**: ảnh FN cho thấy tổn thương lớn chỉ được bao một phần
    (`case_FN_01_MR49502_C+Delay_k027.png`, `case_FN_02_MR164939_InPhase_k005.png`), nên IoU < 0.5.
  - Theo lớp (recall mức box, mô tả; YOLO không phân lớp bệnh): abscess 0.811, HCC 0.733, ICC 0.696, hemangioma 0.622,
    metastasis 0.554, FNH 0.550, **cyst 0.443** (thấp nhất, như pilot 0.457).
  - Theo chuỗi: **DWI 0.846** (pilot 0.744), **T2WI 0.707** (pilot 0.612), C+V 0.688, In Phase 0.688, Out Phase 0.681,
    C+Delay 0.676, C-pre 0.663, **C+A 0.643** (thấp nhất).
- Không thấy dấu hiệu lỗi nhãn có hệ thống trên 24 ảnh minh họa.

### 7.6 Mức series và bệnh nhân (cơ sở cho GĐ3 ROI zoom)
Series "được phát hiện" nghĩa là có ít nhất một lát của series có box dự đoán khớp GT (IoU ≥ 0.5). Nguồn: `analysis/series_level_detection.json`.

| conf ≥ | Series được phát hiện (E0) | Pilot | BN được phát hiện (E0) | BN bị bỏ sót (E0) |
|---|---|---|---|---|
| 0.1 | **537/622 (86.3%)** | 546/622 | 77/78 | MR140232 (cyst) |
| 0.25 | **501/622 (80.5%)** | 507/622 | 75/78 | MR109260 (HCC), MR140232 (cyst), MR142442 (cyst) |
| 0.4 | 465/622 (74.8%) | 457/622 | 74/78 | + MR174862 (hemangioma) |

- Ở mức series, E0 **không tốt hơn pilot** (khác biệt ±1–2%), dù metrics mức box tăng.
  Ở conf 0.25, khoảng **1/5 số series** không có ROI đúng nào. Theo quy tắc của dự án, đây là lỗi của pipeline và phải được tính khi đánh giá end-to-end.
- Theo lớp, ở conf 0.25: abscess 64/64, HCC 184/200, ICC 64/72, metastasis 50/63, FNH 41/56, hemangioma 70/103, **cyst 28/64 (44%)**.
  Các BN bị bỏ sót có tổn thương xuất hiện trên ít lát (1–7 lát mỗi chuỗi).

### 7.7 Kết luận và đề xuất
1. **MRI-E0 là baseline MRI chính thức** (best.pt epoch 24; Val: P 0.797, R 0.660, mAP50 0.749, mAP50-95 0.504). **Không train lại.**
2. **E1 (yolo11s)** với cùng cấu hình (50 epoch, patience 20, SGD, batch 32) để so sánh n với s theo §3 của kế hoạch.
   Không đổi cấu hình giữa E0 và E1.
3. **Ngưỡng vận hành:** chưa chốt. Chỉ chốt sau E1, và chỉ dựa trên Val. Ứng viên: **0.1** (ưu tiên recall: series 86%, BN 77/78,
   lát âm có FP 16.5%) và **0.25–0.33** (F1 lớn nhất). Vì nhánh phân loại có thể loại FP, còn FN thì mất hẳn ROI, nên nghiêng về ngưỡng thấp,
   kèm giới hạn số box mỗi lát ở GĐ3.
   **Cập nhật 2026-10-25:** quét ngưỡng bước 0.01 trên Val (`scripts/53_threshold_sweep.py` →
   `analysis/threshold_sweep.csv`, `analysis/threshold_choice.json`, `figures/mri_e0/threshold_sweep.png`).
   Ngưỡng có **F2 lớn nhất là 0.12**: P 0.634 · R 0.745 · series 85.9% · BN 77/78 · lát âm có FP 14.7%.
   Ngưỡng có F1 lớn nhất là 0.34 (F1 0.724). **Đề xuất conf = 0.12**, chờ xác nhận. So sánh với CT: `reports/ct_e0_analysis.md` §7.8.
4. **Tổn thương nhỏ và cyst:** cân nhắc E3 `imgsz = 640` (sau E1). **Tổn thương lớn bị bao thiếu:** GĐ3 nên nới crop context
   (padding tỉ lệ theo kích thước box) và gộp box qua các lát liền kề, thay vì dựa vào một box đơn lẻ.
5. Overfit sau khoảng epoch 24: giữ patience 20. Nếu E1 cũng có best epoch ≤ 30 thì trần 50 epoch là đủ.
