# Demo suy luận YOLO CT/MRI trên ảnh Validation (mẫu minh họa)

*Sinh bởi `scripts/60_yolo_inference_demo.py` lúc 2026-10-25 10:27. Số liệu: `outputs/predictions/inference_demo_conf025/`. Hình: `reports/figures/inference_conf025/`. **Tập Test không được dùng.***

> **Đây không phải kết quả đánh giá.** Mỗi modality chỉ có 8 lát có tổn thương + 4 lát không có annotation, chọn ngẫu nhiên (seed 42, mỗi lát một BN). Số TP/FP/FN dưới đây chỉ mô tả bộ ảnh minh họa này. Metrics trên toàn bộ Val nằm trong `reports/<mod>_*_analysis.md`. Ngưỡng conf = 0.25, đặt tay bằng `--conf` (khác ngưỡng vận hành trong configs/yolo_inference.yaml).

## 1. Mô hình và thiết bị

| | CT | MRI |
|---|---|---|
| Checkpoint | `outputs/runs/yolo_ct/yolo11n_e50_img512_baseline/weights/best.pt` | `outputs/runs/yolo_mri/yolo11n_e50_img512_baseline/weights/best.pt` |
| SHA256 | `65c126f310d8b7751f67625d917252ff5b8984ff741795d2e92b496a831f5707` | `cc6fef44446514d598e6f4a862c78e0eb214d28f2f7a4dac82ce4b060b157377` |
| Thiết bị | `cuda:0` | `cuda:0` |
| fp16 | `True` | `True` |
| imgsz | `512` | `512` |
| conf | `0.25` | `0.25` |
| NMS IoU | `0.7` | `0.7` |
| Thời gian suy luận (ms/ảnh, batch 8, sau warm-up) | 5.4 | 4.9 |
| VRAM PyTorch cấp phát lớn nhất (MB) | 97.0 | 102.0 |
| Ảnh dựng lại từ NIfTI trùng từng pixel với PNG khi train | 12/12 | 12/12 |
| Sai khác box khi đưa ndarray so với file PNG (px) | 0.125 | 0.25 |

**Chạy lại trên CPU** (cùng mẫu ảnh, `--device cpu`, fp32):

| | CT | MRI |
|---|---|---|
| ms/ảnh | 31.2 | 30.5 |
| TP / FP / FN (so với GPU) | 3 / 2 / 6 | 4 / 1 / 4 |

## 2. Kết quả trên bộ ảnh minh họa

| Modality | Lát (có tổn thương / không) | Box GT | Dự đoán | TP | FP | FN | Lát có tổn thương được phát hiện | Lát có tổn thương bị bỏ sót hoàn toàn | Lát không annotation có FP |
|---|---|---|---|---|---|---|---|---|---|
| CT | 12 (8 / 4) | 9 | 5 | 3 | 2 | 6 | 3/8 | 5/8 | 0/4 |
| MRI | 12 (8 / 4) | 8 | 5 | 4 | 1 | 4 | 4/8 | 4/8 | 0/4 |

TP = dự đoán khớp một box GT với IoU ≥ 0.5 (ghép tham lam theo confidence, một-một); FP = dự đoán không khớp; FN = box GT không được khớp. Lát "không annotation" nghĩa là lát không có tổn thương được gán nhãn, **không** có nghĩa BN khỏe mạnh.

### CT

![montage](figures/inference_conf025/ct_montage.png)

| # | Ảnh | BN (chẩn đoán) | Pha/chuỗi | Loại lát | GT | Dự đoán (conf) | TP | FP | FN | IoU tốt nhất của GT |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [P0053_C2_k074](figures/inference_conf025/ct/01_pos_P0053_C2_k074.png) | P0053 (HCC) | C2 | có tổn thương | 1 | 0.318 | 1 | 0 | 0 | 0.506 |
| 2 | [P0018_C2_k053](figures/inference_conf025/ct/02_pos_P0018_C2_k053.png) | P0018 (HCC) | C2 | có tổn thương | 1 | 0.710 | 1 | 0 | 0 | 0.888 |
| 3 | [P0058_C2_k050](figures/inference_conf025/ct/03_pos_P0058_C2_k050.png) | P0058 (HCC) | C2 | có tổn thương | 2 | — | 0 | 0 | 2 | 0.0;0.0 |
| 4 | [P0170_P_k036](figures/inference_conf025/ct/04_pos_P0170_P_k036.png) | P0170 (ICC) | P | có tổn thương | 1 | 0.327 | 0 | 1 | 1 | 0.0 |
| 5 | [P0262_C1_k023](figures/inference_conf025/ct/05_pos_P0262_C1_k023.png) | P0262 (CHCC) | C1 | có tổn thương | 1 | 0.781 | 1 | 0 | 0 | 0.836 |
| 6 | [P0132_P_k050](figures/inference_conf025/ct/06_pos_P0132_P_k050.png) | P0132 (ICC) | P | có tổn thương | 1 | — | 0 | 0 | 1 | 0.0 |
| 7 | [P0267_C3_k034](figures/inference_conf025/ct/07_pos_P0267_C3_k034.png) | P0267 (CHCC) | C3 | có tổn thương | 1 | 0.417 | 0 | 1 | 1 | 0.466 |
| 8 | [P0168_C1_k153](figures/inference_conf025/ct/08_pos_P0168_C1_k153.png) | P0168 (ICC) | C1 | có tổn thương | 1 | — | 0 | 0 | 1 | 0.0 |
| 9 | [P0098_C2_k076](figures/inference_conf025/ct/09_neg_P0098_C2_k076.png) | P0098 (ICC) | C2 | không annotation | 0 | — | 0 | 0 | 0 | — |
| 10 | [P0301_C1_k386](figures/inference_conf025/ct/10_neg_P0301_C1_k386.png) | P0301 (CN) | C1 | không annotation | 0 | — | 0 | 0 | 0 | — |
| 11 | [P0124_C2_k157](figures/inference_conf025/ct/11_neg_P0124_C2_k157.png) | P0124 (ICC) | C2 | không annotation | 0 | — | 0 | 0 | 0 | — |
| 12 | [P0077_C3_k021](figures/inference_conf025/ct/12_neg_P0077_C3_k021.png) | P0077 (HCC) | C3 | không annotation | 0 | — | 0 | 0 | 0 | — |

### MRI

![montage](figures/inference_conf025/mri_montage.png)

| # | Ảnh | BN (chẩn đoán) | Pha/chuỗi | Loại lát | GT | Dự đoán (conf) | TP | FP | FN | IoU tốt nhất của GT |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [MR206734_C+V_k047](figures/inference_conf025/mri/01_pos_MR206734_C+V_k047.png) | MR206734 (FOCAL_NODULAR_HYPERPLASIA) | C+V | có tổn thương | 1 | — | 0 | 0 | 1 | 0.0 |
| 2 | [MR207755_T2WI_k007](figures/inference_conf025/mri/02_pos_MR207755_T2WI_k007.png) | MR207755 (Hepatic_metastasis) | T2WI | có tổn thương | 1 | 0.824 | 1 | 0 | 0 | 0.935 |
| 3 | [MR222125_InPhase_k022](figures/inference_conf025/mri/03_pos_MR222125_InPhase_k022.png) | MR222125 (Hepatic_metastasis) | In Phase | có tổn thương | 1 | — | 0 | 0 | 1 | 0.0 |
| 4 | [MR124193_C+Delay_k031](figures/inference_conf025/mri/04_pos_MR124193_C+Delay_k031.png) | MR124193 (Hepatocellular_carcinoma) | C+Delay | có tổn thương | 1 | 0.789 | 1 | 0 | 0 | 0.876 |
| 5 | [MR210196_C+A_k030](figures/inference_conf025/mri/05_pos_MR210196_C+A_k030.png) | MR210196 (Hepatic_hemangioma) | C+A | có tổn thương | 1 | — | 0 | 0 | 1 | 0.0 |
| 6 | [MR69046_InPhase_k003](figures/inference_conf025/mri/06_pos_MR69046_InPhase_k003.png) | MR69046 (Hepatic_abscess) | In Phase | có tổn thương | 1 | 0.524;0.405 | 1 | 1 | 0 | 0.664 |
| 7 | [MR179970_C+V_k045](figures/inference_conf025/mri/07_pos_MR179970_C+V_k045.png) | MR179970 (Intrahepatic_cholangiocarcinoma) | C+V | có tổn thương | 1 | 0.884 | 1 | 0 | 0 | 0.916 |
| 8 | [MR13762_C+V_k023](figures/inference_conf025/mri/08_pos_MR13762_C+V_k023.png) | MR13762 (Hepatocellular_carcinoma) | C+V | có tổn thương | 1 | — | 0 | 0 | 1 | 0.0 |
| 9 | [MR142442_DWI_k004](figures/inference_conf025/mri/09_neg_MR142442_DWI_k004.png) | MR142442 (Hepatic_cyst) | DWI | không annotation | 0 | — | 0 | 0 | 0 | — |
| 10 | [MR140376_C+Delay_k045](figures/inference_conf025/mri/10_neg_MR140376_C+Delay_k045.png) | MR140376 (Hepatic_cyst) | C+Delay | không annotation | 0 | — | 0 | 0 | 0 | — |
| 11 | [MR175358_C-pre_k001](figures/inference_conf025/mri/11_neg_MR175358_C-pre_k001.png) | MR175358 (Hepatic_hemangioma) | C-pre | không annotation | 0 | — | 0 | 0 | 0 | — |
| 12 | [MR200076_InPhase_k026](figures/inference_conf025/mri/12_neg_MR200076_InPhase_k026.png) | MR200076 (Hepatic_hemangioma) | In Phase | không annotation | 0 | — | 0 | 0 | 0 | — |

## 3. Môi trường

- Python 3.12.10, torch 2.11.0+cu128, Ultralytics 8.4.182, GPU: NVIDIA GeForce RTX 3050 Laptop GPU

<!-- ANALYST-NOTES -->

## 4. Nhận xét (bản lưu ở conf 0.25; ngưỡng vận hành đã chốt là 0.12: `reports/inference_demo.md`)

*Nguồn: `outputs/predictions/inference_demo_conf025/summary_cuda.json`, `summary_cpu.json`, `{ct,mri}_cases_*.csv` và các hình trong `reports/figures/inference_conf025/`.
Mẫu gồm 12 lát mỗi modality, nên **không** suy ra được recall hay precision của mô hình.
Metrics trên toàn bộ Val: CT `outputs/runs/yolo_ct/yolo11n_e50_img512_baseline/metrics_val.json` (ở conf 0.25: P 0.665 · R 0.493);
MRI `reports/mri_e0_analysis.md`.*

### 4.1 Đường suy luận khớp với lúc train
- **24/24 lát** dựng lại từ NIfTI gốc (`livercancer.data.slices`) **trùng từng pixel** với PNG đã dùng khi train.
  Test `tests/test_yolo_inference.py::test_nifti_slice_equals_training_png` kiểm tra lại điều này mỗi lần chạy pytest.
- Đưa ảnh vào dạng ndarray hay đọc từ file PNG cho box lệch tối đa **0.25 px** (sai số fp16); số TP/FP/FN giống hệt nhau.
- GPU (fp16) và CPU (fp32) cho **cùng** TP/FP/FN trên cả hai modality.
- Box GT nằm đúng trên tổn thương trong mọi hình, nên quy ước hiển thị `slice.T` khớp với nhãn.

### 4.2 CT (8 lát có tổn thương, 9 box GT; 4 lát không annotation, trong đó 1 lát của BN CN)
- **Phát hiện đúng 3 lát:** P0018 HCC (conf 0.71, IoU 0.89), P0262 CHCC khối lớn (0.78, IoU 0.84) và P0053 HCC (0.32, IoU 0.51, sát ngưỡng IoU).
- **Bỏ sót 5 lát:**
  - Tổn thương **nhỏ** trên 4 lát: P0058 có 2 ổ nhỏ; P0132, P0168 và P0170 là ICC.
  - Ở P0170 có 1 box conf 0.33 nằm gần nhưng không chồng lên GT (IoU 0), nên tính là FP.
  - P0267 CHCC là **near-miss**: box nằm trong vùng GT lớn nhưng IoU 0.466 < 0.5, nên tính 1 FP + 1 FN.
- 4 lát không annotation (gồm P0301 CN): **không có FP**.
- Mẫu này phù hợp với các nhận định trước đó (`reports/ct_pilot_analysis.md`, `metrics_val.json` của CT-E0): tổn thương nhỏ và ICC là nhóm khó;
  box dự đoán hay bao **thiếu** tổn thương lớn. Chữ thước đo in chồng thấy rõ trên P0132 và P0170, nhưng không có box nào rơi vào đó trong mẫu này.

### 4.3 MRI (8 lát có tổn thương, 8 box GT; 4 lát không annotation)
- **Phát hiện đúng 4 lát:** MR207755 metastasis T2WI (0.82, IoU 0.94), MR179970 ICC C+V (0.88, IoU 0.92), MR124193 HCC C+Delay (0.79, IoU 0.88)
  và MR69046 abscess In Phase (0.52, IoU 0.66). Ở MR69046 còn **1 box trùng lặp** (0.40) nằm bên trong cùng tổn thương, nên tính là FP.
- **Bỏ sót 4 lát:** MR206734 FNH, MR222125 metastasis và MR210196 hemangioma đều là tổn thương **nhỏ**; MR13762 HCC (C+V) có kích thước vừa.
- 4 lát không annotation: **không có FP**.

### 4.4 Khả năng chạy trên RTX 3050 4 GB
- GPU fp16: khoảng **5 ms/ảnh** (batch 8, sau warm-up); PyTorch cấp phát lớn nhất khoảng 100 MB mỗi modality.
  `nvidia-smi` báo **235 MiB / 4096 MiB** khi nạp **cả hai** detector và chạy batch 8.
- CPU fp32 (i5-12450H): khoảng **30 ms/ảnh**. Chạy được, nhưng quét toàn bộ volume (khoảng 100–200 lát) mất vài giây.
- Thiết bị `auto` dùng CUDA nếu có, nếu không thì CPU. Khi yêu cầu `cuda` mà không có CUDA, chương trình cảnh báo rồi chuyển sang CPU
  (đã thử với `CUDA_VISIBLE_DEVICES=-1`). Nếu hết VRAM giữa chừng, batch đó được chạy lại trên CPU (`LesionDetector.predict`).

### 4.5 Hệ quả cho GĐ3 (ROI zoom)
- **Box trùng lặp** trong cùng một tổn thương (MR69046) và **box bao thiếu** tổn thương lớn (P0267): cần gộp box (NMS theo IoU hoặc chứa nhau,
  và nối qua các lát liền kề) và **mở rộng margin** trước khi crop.
- **Tổn thương nhỏ bị bỏ sót** ở cả hai modality: đây là lỗi của pipeline, phải được tính khi đánh giá end-to-end. Ngưỡng vận hành (đang là 0.25)
  sẽ được chọn trên Val. Ngưỡng thấp hơn tăng recall nhưng cũng tăng FP (MRI-E0: conf 0.1 → series phát hiện 86%, lát âm có FP 16.5%).
- Việc còn thiếu trước khi chốt CT: **phân tích CT-E0 trên toàn bộ Val** (`50_` / `51_` / `52_` với `--data data\processed\full\yolo_ct`),
  tương tự `reports/mri_e0_analysis.md`.
