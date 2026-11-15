# Automatic Tumor Zoom trên Validation (CT và MRI)

*Sinh bởi `scripts/71_eval_roi_val.py` lúc 2026-10-25 11:24. Detector và ngưỡng đã chốt (`configs/yolo_inference.yaml`, conf 0.12). Số liệu: `outputs/predictions/roi_val/<mod>/` (`raw_dets.csv`, `grid.csv`, `choice.json`, `tracks.csv`). Hình: `reports/figures/roi_<mod>/`. **Chỉ dùng Validation; Test không được dùng.***

## 1. Cách làm

1. Quét **mọi lát** của mỗi volume Val dùng được (CT: BN × pha; MRI: series) bằng YOLO ở conf 0.12.
2. Gộp box trong cùng lát (IoU > 0.5 hoặc chứa nhau > 80%) → nối box qua các lát liên tiếp thành **track** (IoU ≥ 0.3, cho phép hở 1 lát).
3. Lọc track theo số lát tối thiểu và confidence lớn nhất tối thiểu. **Quy tắc chọn (khai báo trước, chỉ trên Val):** trong các tổ hợp giữ recall tổn thương ≥ (không lọc − 0.02) và recall volume ≥ (không lọc − 0.01), chọn tổ hợp có ít track giả nhất trên mỗi volume.
4. Mỗi track được zoom trên lát đại diện (diện tích × conf lớn nhất): **local** = box + margin max(15%, 8 px), vuông, 224×224; **context** = 2.5 × cạnh box (≥ 128 px), vuông, 224×224.
5. GT (chỉ để đánh giá): nhãn Val đã xuất, nối thành tổn thương 3D bằng cùng thuật toán. Một tổn thương GT được tính là **có ROI** khi có track trùng với nó (IoU ≥ 0.3) trên ít nhất một lát chung. Lát có mask tổn thương nhưng không được xuất nhãn (MRI: mask và JSON không đồng thuận) là vùng *không xác định*: track chỉ nằm trên những lát đó không bị tính là track giả.

## 2. CT (210 volume Val, 275 tổn thương GT 3D)

| | Không lọc (1 lát, conf 0.12) | **Đã chọn** (1 lát, conf 0.12) | Đã chọn, tiêu chí IoU 0.5 |
|---|---|---|---|
| Tổn thương GT có ROI (recall) | 0.6436 | **0.6436** | 0.6036 |
| Tổn thương nhỏ < 32² px có ROI | 0.2029 | **0.2029** | 0.2029 |
| Volume có tổn thương có ≥ 1 ROI đúng | 0.9325 | **0.9325** | 0.9018 |
| BN có tổn thương có ROI | 42/42 | **42/42** | 41/42 |
| Track giả / volume | 5.038 | **5.038** | 5.3 |
| Tổng track giữ lại | 1368 | **1368** | 1368 |
| Track vùng không xác định | 0 | **0** | 0 |
| Volume BN CN có ≥ 1 track | 36/47 | **36/47** | 36/47 |
| BN CN có ≥ 1 track | 11/12 | **11/12** | 11/12 |

**Độ phủ của crop** (track khớp GT, trên lát đại diện; 310 track, trong đó 11 có lát đại diện không có box GT): phần box GT nằm trong khung local, trung vị **1.0** (83.3% track ≥ 0.95); trong khung context, trung vị **1.0** (98.7% ≥ 0.95).

Hình trước/sau (ảnh gốc | box + khung zoom | zoom local | zoom context):

- [01_TP_P0015_C1_k018.png](figures/roi_ct/01_TP_P0015_C1_k018.png)
- [02_TP_P0106_C3_k237.png](figures/roi_ct/02_TP_P0106_C3_k237.png)
- [03_FP_P0168_P_k053.png](figures/roi_ct/03_FP_P0168_P_k053.png)
- [04_TP_P0171_C2_k056.png](figures/roi_ct/04_TP_P0171_C2_k056.png)
- [05_TP_P0185_C2_k201.png](figures/roi_ct/05_TP_P0185_C2_k201.png)
- [06_TP_P0226_P_k032.png](figures/roi_ct/06_TP_P0226_P_k032.png)
- [07_TP_P0262_C2_k058.png](figures/roi_ct/07_TP_P0262_C2_k058.png)
- [08_FP_P0266_C2_k067.png](figures/roi_ct/08_FP_P0266_C2_k067.png)

## 3. MRI (623 volume Val, 648 tổn thương GT 3D)

| | Không lọc (1 lát, conf 0.12) | **Đã chọn** (1 lát, conf 0.12) | Đã chọn, tiêu chí IoU 0.5 |
|---|---|---|---|
| Tổn thương GT có ROI (recall) | 0.8549 | **0.8549** | 0.838 |
| Tổn thương nhỏ < 32² px có ROI | 0.7068 | **0.7068** | 0.6842 |
| Volume có tổn thương có ≥ 1 ROI đúng | 0.8682 | **0.8682** | 0.8537 |
| BN có tổn thương có ROI | 76/78 | **76/78** | 76/78 |
| Track giả / volume | 4.478 | **4.478** | 4.509 |
| Tổng track giữ lại | 3413 | **3413** | 3413 |
| Track vùng không xác định | 44 | **44** | 44 |

**Độ phủ của crop** (track khớp GT, trên lát đại diện; 579 track, trong đó 15 có lát đại diện không có box GT): phần box GT nằm trong khung local, trung vị **1.0** (97.5% track ≥ 0.95); trong khung context, trung vị **1.0** (100.0% ≥ 0.95).

Hình trước/sau (ảnh gốc | box + khung zoom | zoom local | zoom context):

- [01_FP_MR145114_C+A_k040.png](figures/roi_mri/01_FP_MR145114_C+A_k040.png)
- [02_TP_MR173412_C+Delay_k035.png](figures/roi_mri/02_TP_MR173412_C+Delay_k035.png)
- [03_TP_MR179970_T2WI_k014.png](figures/roi_mri/03_TP_MR179970_T2WI_k014.png)
- [04_TP_MR184623_T2WI_k014.png](figures/roi_mri/04_TP_MR184623_T2WI_k014.png)
- [05_FP_MR192701_T2WI_k016.png](figures/roi_mri/05_FP_MR192701_T2WI_k016.png)
- [06_TP_MR207755_C+Delay_k024.png](figures/roi_mri/06_TP_MR207755_C+Delay_k024.png)
- [07_TP_MR78507_InPhase_k007.png](figures/roi_mri/07_TP_MR78507_InPhase_k007.png)
- [08_TP_MR82133_C+A_k014.png](figures/roi_mri/08_TP_MR82133_C+A_k014.png)

<!-- ANALYST-NOTES -->

## 4. Nhận xét

*Nguồn: `outputs/predictions/roi_val/{ct,mri}/grid.csv` (30 tổ hợp lọc), `choice.json`, `tracks.csv`; hình `reports/figures/roi_{ct,mri}/`.
Chỉ dùng Validation.*

### 4.1 Bộ lọc vùng: giữ nguyên (không lọc)
- Theo quy tắc khai báo trước, **không tổ hợp lọc nào giữ được recall** trong giới hạn cho phép (−2 điểm recall tổn thương, −1 điểm recall volume).
  Vì vậy cấu hình chốt là **không lọc thêm**: track 1 lát, conf ≥ 0.12 (`configs/roi.yaml`).
- Ví dụ về cái giá của việc lọc:
  - **CT**, yêu cầu ≥ 2 lát: track giả giảm từ 5.0 xuống 2.6 / volume, nhưng recall tổn thương giảm 64.4% → 58.2%,
    tổn thương nhỏ 20.3% → 13.0%, BN 42 → 40.
  - **MRI**, yêu cầu ≥ 2 lát: track giả giảm từ 4.5 xuống 1.5, nhưng recall tổn thương giảm 85.5% → 76.4%.
    Tổn thương MRI chỉ xuất hiện trên khoảng 8 lát, nên nhiều tổn thương chỉ được bắt ở 1 lát.
- **Hệ quả:** trung bình khoảng **5 vùng giả mỗi volume** (CT 5.0, MRI 4.5); 36/47 volume CT của BN CN có ít nhất 1 vùng.
  Website sắp xếp vùng theo độ tin cậy giảm dần. Các vùng giả cần được **nhánh phân loại (GĐ4)** hoặc người đọc loại bỏ.
  Đây là lựa chọn có chủ đích: ưu tiên không bỏ sót.

### 4.2 Phát hiện ở mức tổn thương 3D và volume (cấu hình đã chốt)
| | CT | MRI |
|---|---|---|
| Volume có tổn thương có ≥ 1 vùng zoom đúng (IoU ≥ 0.3) | **93.3%** (163 volume) | **86.8%** (622 series) |
| Tổn thương GT 3D có vùng zoom đúng | 64.4% (275 tổn thương) | 85.5% (648) |
| Tổn thương nhỏ < 32² px có vùng zoom | **20.3%** (69) | 70.7% (266) |
| BN có tổn thương có vùng zoom đúng | 42/42 | 76/78 |
| Theo tiêu chí chặt hơn (IoU ≥ 0.5) | xem bảng mục 2 | xem bảng mục 3 |

- Volume-level CT (93.3%) cao hơn recall mức tổn thương (64.4%): khối chính thường được bắt, nhưng các ổ nhỏ hoặc vệ tinh bị bỏ sót.
  Điều này khớp với phân tích detection (FN tập trung ở box nhỏ).
- Số tổn thương GT 3D được dựng bằng cách nối box GT qua các lát. Một tổn thương có thể bị tách làm hai nếu có khoảng trống > 1 lát
  (MRI: một số lát bị loại vì mask và JSON không đồng thuận), nên số tổn thương MRI (648) lớn hơn số series (622).
  Chỉ số mức volume/series ổn định hơn chỉ số mức tổn thương.

### 4.3 Chất lượng ảnh zoom
- Trên lát đại diện của các vùng đúng, phần tổn thương GT nằm trong crop **local** có trung vị 100%.
  CT: **83.3%** vùng có ≥ 95% tổn thương nằm trong crop local, **98.7%** trong crop context. MRI: 97.5% và 100%.
- CT thấp hơn ở crop local vì YOLO hay **bao thiếu tổn thương lớn** (đã thấy ở phân tích CT-E0). Crop context (2.5× cạnh box, ≥ 128 px)
  bù lại được phần lớn. Đây là lý do GĐ4 dùng **cả hai** crop: CNN cho local, Transformer cho context.
- Ví dụ P0262 C2 lát 58 (`roi_ct/07_TP_…`): vùng khớp với một tổn thương GT trên các lát 59–62. Trên chính lát 58, khung xanh lá là một
  tổn thương khác (khối lớn bên trái) mà vùng này không bao. Độ phủ được tính với đúng tổn thương mà vùng đã khớp.
- Không thấy lỗi lệch tọa độ giữa ảnh gốc, khung và ảnh zoom (8 + 8 hình minh họa).

### 4.4 Việc chưa làm ở GĐ3
- Chưa xuất bộ crop đầy đủ cho train/val (và Test để đánh giá cuối) để huấn luyện Hybrid. Việc này là bước đầu của GĐ4 (`docs/PHASE4_ROI_WEBSITE_PLAN.md` §A.2 bước 7–8).
- Margin local (15%) và tỉ lệ context (2.5×) là giá trị thiết kế cố định, **chưa** tối ưu. Nếu cần tối ưu thì chỉ làm trên Val.
