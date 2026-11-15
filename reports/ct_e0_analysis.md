# Phân tích YOLO run `yolo11n_e50_img512_baseline` (CT), chỉ trên Validation

*Sinh bởi `scripts/50_analyze_yolo_run.py` lúc 2026-10-25 10:09. Mọi số liệu đọc từ `outputs/runs/yolo_ct/yolo11n_e50_img512_baseline` (tải về từ Colab) và từ lần chạy lại `best.pt` trên tập Val pilot ở local. **Tập Test không được dùng.***

## 1. Cấu hình huấn luyện (args.yaml)

| Tham số | Giá trị |
|---|---|
| model | yolo11n.pt |
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

![curves](figures/ct_e0/curves.png)

| Epoch | train box/cls/dfl | val box/cls/dfl | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| 1 | 1.815 / 2.781 / 1.554 | 1.625 / 20.533 / 1.298 | 0.401 | 0.325 | 0.298 | 0.141 |
| 2 | 1.719 / 1.748 / 1.394 | 1.636 / 18.047 / 1.228 | 0.340 | 0.311 | 0.233 | 0.104 |
| 3 | 1.789 / 1.789 / 1.446 | 1.799 / 7.337 / 1.402 | 0.320 | 0.282 | 0.180 | 0.072 |
| 4 | 1.804 / 1.761 / 1.470 | 1.665 / 5.791 / 1.319 | 0.501 | 0.396 | 0.372 | 0.169 |
| 5 | 1.741 / 1.610 / 1.424 | 2.137 / 46.213 / 1.683 | 0.253 | 0.169 | 0.099 | 0.048 |
| 6 | 1.678 / 1.487 / 1.384 | 1.769 / 25.269 / 1.403 | 0.320 | 0.273 | 0.218 | 0.108 |
| 7 | 1.629 / 1.410 / 1.352 | 1.738 / 19.761 / 1.353 | 0.380 | 0.303 | 0.284 | 0.136 |
| 8 | 1.601 / 1.342 / 1.330 | 1.738 / 5.964 / 1.317 | 0.474 | 0.338 | 0.335 | 0.140 |
| 9 | 1.576 / 1.291 / 1.312 | 1.544 / 4.602 / 1.173 | 0.638 | 0.460 | 0.514 | 0.256 |
| 10 | 1.550 / 1.245 / 1.292 | 1.574 / 6.618 / 1.204 | 0.590 | 0.426 | 0.462 | 0.237 |
| 11 | 1.526 / 1.197 / 1.275 | 1.556 / 4.372 / 1.184 | 0.662 | 0.470 | 0.519 | 0.264 |
| 12 | 1.510 / 1.165 / 1.266 | 1.587 / 6.004 / 1.208 | 0.633 | 0.469 | 0.507 | 0.247 |
| 13 | 1.505 / 1.140 / 1.258 | 1.603 / 6.225 / 1.216 | 0.589 | 0.454 | 0.487 | 0.240 |
| 14 | 1.477 / 1.110 / 1.243 | 1.548 / 3.908 / 1.172 | 0.650 | 0.488 | 0.540 | 0.269 |
| 15 | 1.476 / 1.077 / 1.239 | 1.545 / 3.924 / 1.181 | 0.685 | 0.483 | 0.551 | 0.282 |
| 16 | 1.463 / 1.054 / 1.228 | 1.552 / 4.329 / 1.167 | 0.660 | 0.478 | 0.530 | 0.272 |
| 17 | 1.438 / 1.030 / 1.219 | 1.542 / 3.640 / 1.157 | 0.686 | 0.483 | 0.548 | 0.282 |
| 18 | 1.436 / 1.013 / 1.212 | 1.561 / 4.112 / 1.173 | 0.661 | 0.488 | 0.542 | 0.278 |
| 19 | 1.427 / 0.992 / 1.207 | 1.546 / 3.581 / 1.165 | 0.682 | 0.482 | 0.547 | 0.282 |
| 20 | 1.407 / 0.967 / 1.200 | 1.550 / 3.280 / 1.166 | 0.666 | 0.480 | 0.537 | 0.278 |
| 21 | 1.407 / 0.949 / 1.192 | 1.571 / 3.382 / 1.182 | 0.667 | 0.486 | 0.541 | 0.275 |
| 22 | 1.388 / 0.930 / 1.180 | 1.566 / 3.245 / 1.180 | 0.671 | 0.489 | 0.540 | 0.278 |
| 23 | 1.375 / 0.915 / 1.176 | 1.584 / 3.319 / 1.193 | 0.671 | 0.470 | 0.526 | 0.270 |
| 24 | 1.374 / 0.897 / 1.172 | 1.586 / 3.390 / 1.192 | 0.672 | 0.473 | 0.523 | 0.269 |
| 25 | 1.356 / 0.881 / 1.164 | 1.578 / 3.260 / 1.184 | 0.659 | 0.480 | 0.524 | 0.270 |
| 26 | 1.341 / 0.861 / 1.148 | 1.583 / 3.253 / 1.187 | 0.663 | 0.472 | 0.521 | 0.270 |
| 27 | 1.332 / 0.840 / 1.147 | 1.590 / 3.279 / 1.191 | 0.670 | 0.467 | 0.519 | 0.268 |
| 28 | 1.320 / 0.822 / 1.143 | 1.595 / 3.299 / 1.195 | 0.677 | 0.466 | 0.517 | 0.268 |
| 29 | 1.308 / 0.807 / 1.137 | 1.597 / 3.302 / 1.196 | 0.682 | 0.462 | 0.519 | 0.269 |
| 30 | 1.305 / 0.796 / 1.133 | 1.602 / 3.319 / 1.200 | 0.676 | 0.463 | 0.515 | 0.268 |
| 31 | 1.284 / 0.773 / 1.125 | 1.606 / 3.315 / 1.203 | 0.674 | 0.461 | 0.514 | 0.267 |
| 32 | 1.274 / 0.762 / 1.120 | 1.613 / 3.297 / 1.208 | 0.675 | 0.461 | 0.512 | 0.265 |
| 33 | 1.270 / 0.744 / 1.114 | 1.617 / 3.282 / 1.212 | 0.676 | 0.459 | 0.510 | 0.264 |
| 34 | 1.267 / 0.737 / 1.110 | 1.621 / 3.278 / 1.215 | 0.677 | 0.458 | 0.509 | 0.263 |
| 35 | 1.242 / 0.719 / 1.099 | 1.623 / 3.272 / 1.217 | 0.673 | 0.457 | 0.508 | 0.262 |

- Epoch tốt nhất theo fitness của Ultralytics (= mAP50-95): **15** (mAP50 0.551, mAP50-95 0.282); epoch cuối: mAP50 0.508, mAP50-95 0.262.
- Thay đổi từ epoch đầu đến epoch cuối: {'train/box_loss': -0.5726, 'train/cls_loss': -2.0617, 'train/dfl_loss': -0.4543, 'metrics/precision(B)': 0.2722, 'metrics/recall(B)': 0.1321, 'metrics/mAP50(B)': 0.2095, 'metrics/mAP50-95(B)': 0.1214, 'val/box_loss': -0.0019, 'val/cls_loss': -17.2612, 'val/dfl_loss': -0.0812}

## 3. Metrics của `best.pt` trên Val

| Nguồn | conf | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Colab `metrics_val.json` | 0.001 | 0.688 | 0.481 | 0.551 | 0.282 |
| Chạy lại ở local | 0.001 | 0.688 | 0.481 | 0.551 | 0.282 |
| Chạy lại ở local | 0.25 | 0.665 | 0.493 | 0.444 | 0.236 |

Precision/Recall của Ultralytics được tính tại ngưỡng conf cho **F1 cao nhất** trên đường cong, không phải tại một ngưỡng cố định; mAP luôn tính với conf = 0.001.

## 4. Vì sao Confusion Matrix có rất nhiều 'background' dự đoán sai

| conf khi tính ma trận | TP | FP (hàng background) | FN (cột background) |
|---|---|---|---|
| 0.001 | 6988 | 201758 | 643 |
| 0.25 | 3922 | 1735 | 3709 |

- Ultralytics 8.4.182 tính confusion matrix khi validate với `conf = args.conf`, mặc định **0.001** (`engine/validator.py`: `self.args.conf = 0.001`; `detect/val.py`: `process_batch(..., conf=self.args.conf)`), ghép box với GT ở IoU > 0.45. Vì vậy **mọi box có conf > 0.001** (kể cả box gần như chắc chắn là nhiễu) đều được tính; mọi box không khớp GT rơi vào hàng *background* (FP). Trên tập Val có rất nhiều lát âm tính, nên con số này rất lớn.
- Ma trận được chuẩn hóa theo **cột** (cột GT), nên ô background→lesion trông như "100%" khi cột background chỉ chứa FP. Đó là cách hiển thị, không phải tỉ lệ lỗi thực.
- Khi **inference** thực tế (`predict`), ngưỡng mặc định là **0.25** và ngưỡng vận hành sẽ được chọn trên Val. Bảng ở mục 5 cho thấy FP/FN tại từng ngưỡng.

## 5. FP / FN theo ngưỡng confidence (ghép GT ở IoU ≥ 0.5)

| conf ≥ | TP | FP (lát dương) | FP (lát âm) | FN | Precision | Recall | F1 | FP/lát | Lát âm có FP | Lát dương có TP | BN tổn thương được phát hiện | BN chỉ có lát âm mà có FP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | 6810 | 133839 | 69605 | 821 | 0.0324 | 0.8924 | 0.0625 | 13.909 | 5676/7313 (77.6%) | 6581/7314 (90.0%) | 42/42 | 12/12 (CN) |
| 0.05 | 4937 | 8942 | 2682 | 2694 | 0.2981 | 0.647 | 0.4082 | 0.795 | 1531/7313 (20.9%) | 4847/7314 (66.3%) | 42/42 | 12/12 (CN) |
| 0.1 | 4521 | 4662 | 1297 | 3110 | 0.4314 | 0.5925 | 0.4993 | 0.407 | 960/7313 (13.1%) | 4440/7314 (60.7%) | 42/42 | 11/12 (CN) |
| 0.25 | 3780 | 1555 | 407 | 3851 | 0.6583 | 0.4953 | 0.5653 | 0.134 | 372/7313 (5.1%) | 3726/7314 (50.9%) | 40/42 | 7/12 (CN) |
| 0.4 | 3208 | 716 | 172 | 4423 | 0.7832 | 0.4204 | 0.5471 | 0.061 | 165/7313 (2.3%) | 3184/7314 (43.5%) | 36/42 | 4/12 (CN) |
| 0.5 | 2792 | 437 | 88 | 4839 | 0.8417 | 0.3659 | 0.51 | 0.036 | 86/7313 (1.2%) | 2779/7314 (38.0%) | 32/42 | 2/12 (CN) |

Tại conf ≥ 0.25: FN theo kích thước GT {'nhỏ <32²px': '914/1133', 'vừa 32²–96²px': '2667/4835', 'lớn >96²px': '270/1663'}; recall theo nhóm {'class=CHCC': 0.505, 'class=HCC': 0.586, 'class=ICC': 0.397, 'series=C1': 0.447, 'series=C2': 0.581, 'series=C3': 0.529, 'series=P': 0.395}

## 6. Đối chiếu box dự đoán với ground truth

Xanh lá = GT, đỏ nét đứt = dự đoán (conf ≥ 0.25). Có 8 ca FP có confidence cao nhất, 8 ca FN có tổn thương lớn nhất, và 8 ca TP ngẫu nhiên, lưu ở `figures/ct_e0/case_*.png`.

- **FP:** [P0148_C3_k109](figures/ct_e0/case_FP_01_P0148_C3_k109.png), [P0011_C1_k010](figures/ct_e0/case_FP_02_P0011_C1_k010.png), [P0053_C3_k064](figures/ct_e0/case_FP_03_P0053_C3_k064.png), [P0047_C1_k019](figures/ct_e0/case_FP_04_P0047_C1_k019.png), [P0018_C2_k002](figures/ct_e0/case_FP_05_P0018_C2_k002.png), [P0218_C2_k030](figures/ct_e0/case_FP_06_P0218_C2_k030.png), [P0035_C3_k071](figures/ct_e0/case_FP_07_P0035_C3_k071.png), [P0266_C2_k087](figures/ct_e0/case_FP_08_P0266_C2_k087.png)
- **FN:** [P0035_P_k075](figures/ct_e0/case_FN_01_P0035_P_k075.png), [P0011_P_k045](figures/ct_e0/case_FN_02_P0011_P_k045.png), [P0262_C3_k041](figures/ct_e0/case_FN_03_P0262_C3_k041.png), [P0053_C3_k048](figures/ct_e0/case_FN_04_P0053_C3_k048.png), [P0018_C2_k013](figures/ct_e0/case_FN_05_P0018_C2_k013.png), [P0267_C2_k034](figures/ct_e0/case_FN_06_P0267_C2_k034.png), [P0044_P_k095](figures/ct_e0/case_FN_07_P0044_P_k095.png), [P0047_C2_k031](figures/ct_e0/case_FN_08_P0047_C2_k031.png)
- **TP:** [P0262_C2_k059](figures/ct_e0/case_TP_01_P0262_C2_k059.png), [P0053_C2_k053](figures/ct_e0/case_TP_02_P0053_C2_k053.png), [P0224_C2_k050](figures/ct_e0/case_TP_03_P0224_C2_k050.png), [P0269_C1_k085](figures/ct_e0/case_TP_04_P0269_C1_k085.png), [P0088_P_k014](figures/ct_e0/case_TP_05_P0088_P_k014.png), [P0011_C3_k025](figures/ct_e0/case_TP_06_P0011_C3_k025.png), [P0035_C3_k067](figures/ct_e0/case_TP_07_P0035_C3_k067.png), [P0148_C1_k071](figures/ct_e0/case_TP_08_P0148_C1_k071.png)

<!-- ANALYST-NOTES -->

## 7. Nhận định và đề xuất (CT-E0)

*Nguồn số liệu, đều nằm trong `outputs/runs/yolo_ct/yolo11n_e50_img512_baseline/`: `results.csv`, `args.yaml`, `metrics_val.json` (Colab),
và trong `analysis/`: `analysis_summary.json` (chạy lại `best.pt` ở local, GPU RTX 3050), `ct_fp_fn_breakdown.json`,
`series_level_detection.json`, `threshold_sweep.csv`, `threshold_choice.json`. Biểu đồ: `reports/figures/ct_e0/`.
Val = bộ full (`data/processed/full/yolo_ct`, SHA256 ZIP `0a7ed974…`, khớp `metrics_val.json`): **54 BN** (42 có tổn thương + 12 CN),
**14.627 lát** (7.314 lát dương tính = 100% số lát có tổn thương; 7.313 lát âm), **7.631 box GT**, 163 volume (BN × pha) có tổn thương.
**Tập Test không được dùng. Không train lại; checkpoint và dataset gốc không bị thay đổi.***

### 7.1 Huấn luyện
- yolo11n (COCO pretrained), 50 epoch, patience 20, imgsz 512, batch 32, SGD lr0 0.01 + `cos_lr`, seed 42, T4.
  **Early stopping ở epoch 35; `best.pt` = epoch 15** (checkpoint đã strip optimizer, `train_args.imgsz = 512`, SHA256 `65c126f3…`).
  Thời gian khoảng 393 s/epoch, tổng 35 epoch ≈ 3,8 giờ.
- **Epoch 1–8 dao động mạnh:** mAP50-95 0.141 → 0.048 (ep5) → 0.256 (ep9); val cls_loss vọt lên 20,5 (ep1) và **46,2 (ep5)**.
  Giai đoạn này trùng với warmup 3 epoch và lr ở mức cao nhất (≈ 0,0099 ở ep3–5). Từ epoch 9 trở đi metrics ổn định.
- **Overfit sau khoảng epoch 15–17:** val box/dfl loss nhỏ nhất ở ep17 (1.542 / 1.157), sau đó tăng (ep35: 1.623 / 1.217),
  trong khi train loss vẫn giảm (box 1.476 → 1.242, cls 1.077 → 0.719). mAP50-95 giảm từ 0.282 (ep15) xuống 0.262 (ep35).
- **Val cls_loss cao hơn train nhiều:** CT ở best epoch 3,92 so với 1,08 (gấp 3,6 lần), ở epoch cuối 3,27 so với 0,72 (gấp 4,5 lần).
  MRI-E0 ở best epoch là 1,17 so với 0,86 (gấp 1,4 lần), ở epoch cuối 1,24 so với 0,49 (gấp 2,5 lần).
  Nguyên nhân **chưa được xác minh**; chỉ ghi nhận như một khác biệt giữa hai run.

### 7.2 Precision, Recall, mAP
| | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Colab `metrics_val.json` (conf 0.001) | 0.688 | 0.481 | 0.551 | 0.282 |
| Chạy lại ở local (conf 0.001) | **0.688** | **0.481** | **0.551** | **0.282** |
| Chạy lại ở local (conf 0.25) | 0.665 | 0.493 | 0.444 | 0.236 |

- P/R của Ultralytics lấy tại ngưỡng F1 lớn nhất: **F1 = 0.57 tại conf 0.266** (`figures/ct_e0/ultralytics_F1_curve.png`).
- **Định vị chưa khít:** mAP50-95/mAP50 = **0.51** (MRI-E0: 0.67). Recall theo IoU (conf ≥ 0.25): 0.56 @0.3 → 0.50 @0.5 → 0.37 @0.7.
  Hạ tiêu chí IoU từ 0.5 xuống 0.3 làm recall tăng thêm 6 điểm.
- Không so trực tiếp với CT pilot: Val của pilot chỉ gồm 1/2 số lát dương tính.

### 7.3 Confusion matrix (`figures/ct_e0/ultralytics_confusion_matrix_conf{0.001,0.25}.png`)
| conf khi tính ma trận | TP | FP (hàng background) | FN (cột background) |
|---|---|---|---|
| 0.001 (mặc định khi val) | 6.988 | 201.758 | 643 |
| 0.25 | **3.922** | **1.735** | **3.709** |

- TP + FN = 7.631 = số box GT. FP ở conf 0.001 rất lớn vì ma trận đếm mọi box có conf > 0.001 (giải thích ở mục 4).
- Ở conf 0.25, **số FN gần bằng số TP**: khoảng một nửa số box GT bị bỏ sót.

### 7.4 False negative
| | conf ≥ 0.1 | conf ≥ 0.25 |
|---|---|---|
| FN (box) | 3.109 / 7.631 (41%) | 3.850 / 7.631 (50%) |
| trong đó near-miss (IoU 0.1–0.5) | 1.086 (35%) | 780 (20%) |
| không có box nào chồng lên | 2.023 (65%) | 3.070 (80%) |
| FN box **nhỏ** < 32² | 817 / 1.133 (**72%**) | 914 / 1.133 (**81%**) |
| FN box vừa 32²–96² | 2.099 / 4.835 (43%) | 2.667 / 4.835 (55%) |
| FN box lớn > 96² | 194 / 1.663 (12%) | 270 / 1.663 (16%) |
| Recall box theo lớp: HCC / CHCC / ICC | 0.660 / 0.591 / **0.524** | 0.586 / 0.505 / **0.397** |
| Recall box theo pha: C1 / C2 / C3 / P | 0.551 / 0.660 / 0.638 / **0.493** | 0.447 / 0.581 / 0.529 / **0.395** |

- **Tổn thương nhỏ gần như bị bỏ sót hoàn toàn**, và nhóm vừa (32²–96², chiếm 63% số box) bị bỏ sót một nửa. Đây là nguồn FN chính.
- **ICC** và **pha P** có recall thấp nhất, giống với nhận định từ CT pilot. Tên pha C1/C2/C3/P chưa có mapping lâm sàng chính thức (CLAUDE.md).
- Ảnh FN (`figures/ct_e0/case_FN_*.png`): tổn thương lớn và thâm nhiễm chỉ được **bao một phần** (P0035 P, IoU < 0.5).
  Không thấy dấu hiệu lỗi nhãn có hệ thống.

### 7.5 False positive (FP chia theo vị trí bằng GT liver mask, **chỉ để mô tả lỗi trên Val**)
| | conf ≥ 0.1 | conf ≥ 0.25 |
|---|---|---|
| FP tổng | 5.959 | 1.962 |
| lỗi định vị (IoU với GT 0.1–0.5) | 3.696 (**62%**) | 1.313 (**67%**) |
| lát dương tính, không chồng GT: trong / ngoài gan | 665 / 300 | 153 / 88 |
| lát âm của BN ung thư: trong / ngoài gan | 683 / 273 | 187 / 83 |
| lát của BN **CN**: trong / ngoài gan | 264 / 77 | 111 / 26 |
| **FP ngoài gan** (tổng) | 650 (11%) | 197 (10%) |
| Lát không annotation có ≥ 1 box | 13.1% | 5.1% |
| BN CN có ≥ 1 box | 11/12 | 7/12 |

- **Phần lớn FP là box lệch hoặc quá rộng quanh một tổn thương thật**: box rộng hơn tổn thương (`case_FP_01_P0148_C3_k109.png`).
  Loại FP này vẫn cho một ROI có chứa tổn thương ở GĐ3.
- FP ngoài gan chiếm khoảng 10%, ví dụ box với confidence cao trên **lách** (`case_FP_08_P0266_C2_k087.png`).
- Trên BN CN (nhóm chứng không có bệnh gan mật, *không* đồng nghĩa "khỏe mạnh hoàn toàn"), FP chủ yếu nằm **trong gan**.

### 7.6 Mức volume (BN × pha) và BN
| conf ≥ | Volume có tổn thương được phát hiện (163) | BN có tổn thương được phát hiện (42) | BN bị bỏ sót |
|---|---|---|---|
| 0.05 | 154 (94.5%) | 42 | — |
| 0.1 | 150 (92.0%) | 42 | — |
| 0.15 | 142 (87.1%) | 42 | — |
| 0.2 / 0.25 | 136 (83.4%) / 132 (81.0%) | 40 | P0133, P0266 |
| 0.4 | 118 (72.4%) | 36 | + P0058, P0096, P0132, P0147 |

Theo lớp ở conf 0.1: CHCC 47/50, HCC 53/56, ICC 50/57; theo pha: C1 36/40, C2 39/41, C3 38/41, P 37/41.
Mỗi volume CT có trung bình khoảng 45 lát dương tính (7.314 / 163), so với khoảng 8 ở MRI. Vì vậy chỉ cần bắt được tổn thương ở một vài lát
là volume đã được tính là "phát hiện", dù recall mức box thấp.

### 7.7 Đề xuất ngưỡng confidence (chọn trên Val, ưu tiên hạn chế bỏ sót)
**Tiêu chí (khai báo trước, chỉ dùng Val):** chọn ngưỡng có **F2** lớn nhất ở mức box (IoU ≥ 0.5; recall có trọng số gấp đôi precision),
quét từ 0.01 đến 0.80, bước 0.01 (`scripts/53_threshold_sweep.py`). Ngưỡng có F1 lớn nhất được ghi kèm để tham khảo.

| conf | P | R | F1 | F2 | Lát âm có FP | Volume phát hiện | BN phát hiện | BN CN có FP |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.298 | 0.647 | 0.408 | 0.524 | 20.9% | 94.5% | 42/42 | 12/12 |
| 0.10 | 0.432 | 0.593 | 0.499 | 0.551 | 13.1% | 92.0% | 42/42 | 11/12 |
| **0.12 (F2 max)** | **0.474** | **0.578** | 0.521 | **0.554** | **11.1%** | **91.4%** | **42/42** | 11/12 |
| 0.15 | 0.530 | 0.554 | 0.541 | 0.549 | 8.9% | 87.1% | 42/42 | 10/12 |
| 0.25 (mặc định) | 0.659 | 0.496 | 0.566 | 0.521 | 5.1% | 81.0% | 40/42 | 7/12 |
| 0.28 (F1 max) | 0.690 | 0.482 | 0.568 | 0.513 | 4.2% | 80.4% | 40/42 | 6/12 |

- **Đề xuất cho CT: conf = 0.12.** So với 0.25: recall box tăng từ 0.496 lên 0.578; volume có ROI đúng tăng từ 81% lên **91%**;
  không còn BN nào bị bỏ sót (42/42). Đổi lại, precision giảm từ 0.66 xuống 0.47, và **11/12 BN CN có ít nhất 1 box**.
- **Hệ quả:** ở ngưỡng này, YOLO **không** dùng được để kết luận "BN có tổn thương hay không". Nếu cần đánh giá mức BN có tính nhóm CN,
  phải có quy tắc gộp riêng (ví dụ số lát liên tiếp, confidence lớn nhất), quy tắc này **chọn trên Val** ở bước sau.
  Đây không phải việc của detector.
- F2 khá phẳng trong khoảng 0.08–0.15 (0.547–0.554), nên một ngưỡng bất kỳ trong khoảng này đều cho kết quả gần như nhau.
  Ngưỡng **0.15** là lựa chọn thay thế nếu muốn ít FP hơn (lát âm có FP 8.9%, volume phát hiện 87%).
- Đây là **đề xuất, chờ người dùng xác nhận**. Ngưỡng chỉ được chốt sau khi cân nhắc E1, và sau đó **không chỉnh theo Test**.

### 7.8 So sánh với MRI-E0 (hai bộ dữ liệu khác nhau)
| | CT-E0 (PLC-CECT) | MRI-E0 (LLD-MMRI) |
|---|---|---|
| Ảnh | CT có thuốc cản quang, 4 pha (C1/C2/C3/P), 512×512, uint8 đã window sẵn | MRI 8 chuỗi, kích thước khác nhau (512² 40%, 448² 25%, 384² 8%, …), uint8 |
| Nhãn box | thành phần liên thông của lesion mask gốc (≥ 10 px) | mask MedSAM2, chỉ giữ lát **đồng thuận với 2D_box JSON gốc** (IoU ≥ 0.3); 31 series lệch bị cách ly |
| Lớp BN | HCC / ICC / CHCC + **CN** (nhóm chứng, không có mask) | 7 lớp (lành và ác), mọi BN đều có tổn thương |
| Train | 34.873 ảnh (1/2 lát dương tính, âm:dương 1:1) | 28.595 ảnh (âm:dương 0.5) |
| Val | 54 BN · 14.627 lát (50% âm) · 7.631 box | 78 BN (val chính thức LLD-MMRI2023) · 7.491 lát (35% âm) · 4.880 box |
| Box nhỏ < 32² / vừa / lớn (Val) | 14.8% / 63.4% / 21.8% | 26.1% / 53.6% / 20.3% |
| Lát dương tính trung bình mỗi volume/series | khoảng 45 | khoảng 8 |
| Cấu hình | yolo11n, 512, SGD, 50 ep, patience 20, seed 42 | giống CT |
| Best epoch / dừng | 15 / 35 | 24 / 44 |
| P / R / mAP50 / mAP50-95 (Val riêng) | 0.688 / 0.481 / 0.551 / 0.282 | 0.797 / 0.660 / 0.749 / 0.504 |
| mAP50-95 / mAP50 | 0.51 | 0.67 |
| F2 max (ngưỡng) | 0.554 (0.12) | 0.719 (0.12) |
| Ở conf 0.12: recall box / volume phát hiện / BN phát hiện | 0.578 / 91.4% / 42/42 | 0.745 / 85.9% / 77/78 |
| Ở conf 0.12: lát không annotation có FP | 11.1% | 14.7% |

- **Không xếp hạng CT với MRI bằng mAP** (CLAUDE.md): hai bộ khác nhau về modality, BN, nguồn nhãn, cách lấy mẫu lát âm và phân bố kích thước.
  Bảng trên chỉ mô tả mỗi mô hình trên Val của chính nó.
- Quan sát trên từng bộ dữ liệu, không phải kết luận nhân quả:
  - CT có metrics mức box thấp hơn, và box kém khít hơn. Điều này xảy ra dù Val CT có **ít** box nhỏ hơn MRI (15% so với 26%).
  - Ở mức volume, CT lại cao hơn ở cùng ngưỡng (91% so với 86%), phù hợp với việc mỗi volume CT có nhiều lát dương tính hơn.
  - Theo đồ thị `figures/yolo_e0_ct_vs_mri.png`, cả hai modality đều có F2 lớn nhất ở cùng ngưỡng 0.12.

### 7.9 Kết luận cho Detection
1. **Pipeline detection CT hoạt động ổn định và tái lập được:** chạy lại ở local trùng metrics Colab; ảnh suy luận từ NIfTI trùng từng pixel với ảnh train
   (`reports/inference_demo.md`). Không phát hiện lỗi nhãn có hệ thống.
2. **Hiệu năng CT ở mức trung bình:** khoảng một nửa số box GT bị bỏ sót ở conf 0.25; điểm yếu là tổn thương nhỏ và vừa, ICC, pha P, và định vị chưa khít.
   Ở ngưỡng đề xuất 0.12, 91% volume có tổn thương vẫn có ít nhất một ROI đúng.
3. Các thí nghiệm có thể cải thiện, **chưa chạy**, cần người dùng quyết định; đều chỉ chọn trên Val:
   - E1 `yolo11s`;
   - E3 `imgsz = 640` cho tổn thương nhỏ;
   - bộ CT đầy đủ (`--pos-stride-train 1`), vì overfit sớm ở epoch 15 cho thấy có thể cần thêm dữ liệu hoặc augmentation hơn là thêm epoch.
