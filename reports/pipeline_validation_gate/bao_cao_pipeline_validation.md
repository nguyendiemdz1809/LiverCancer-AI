# Đánh giá pipeline AI trên toàn bộ Validation (CT và MRI)

*Sinh bởi `scripts/92_pipeline_validation.py --stage report` lúc 2026-10-26 19:31. Dữ liệu từng ca: `outputs/predictions/pipeline_validation/<ct|mri>/cases/*.json`; tổng hợp: `summary.json`; bảng: `reports/pipeline_validation/*.csv`. **Chỉ Validation, không dùng Test, không huấn luyện lại, không đổi ngưỡng hay mô hình** (YOLO conf 0.12, Tumor Zoom không lọc, CT Hybrid / MRI Hybrid và cách gộp đã chọn trên Val trước đó).*

**Cách đọc:**
- *Pipeline* = hệ thống thật: YOLO tự tìm vùng → Tumor Zoom → Hybrid, gộp mọi vùng của bệnh nhân.
- *Oracle (ROI-GT)* = cùng mô hình phân loại nhưng trên vùng cắt theo nhãn chuẩn. Chỉ để đối chiếu, **không** phải kết quả của hệ thống.
- *Chỉ vùng khớp GT* = pipeline nhưng bỏ các vùng YOLO không trùng tổn thương thật. Đây là phân tích chẩn đoán (cần GT), không dùng được khi chạy thật.
- Bệnh nhân không được phân loại (YOLO không thấy vùng nào) bị tính là **sai** khi tính độ chính xác.

## Tóm tắt

- **CT** (54 BN Val, 42 có ung thư + 12 CN): trên **BN ung thư** (không phải toàn bộ CT) pipeline đúng **36/42 (85.7%)**, macro-F1 89.9%; oracle (vùng GT) 37/42 (88.1%). YOLO tìm thấy tổn thương ở 152/163 volume; bỏ sót mọi tổn thương ở 0 BN. Lỗi: Mô hình phân loại sai ngay cả với vùng chuẩn: 4; Bị cổng bằng chứng từ chối: 2. **Nhóm chứng CN: 2/12 BN bị gán nhãn ung thư** (dương tính giả mức BN). **Toàn bộ 54 BN CT:** kết quả chấp nhận được (ung thư đúng + CN không bị gán nhãn) 46/54 (85.2%). Cổng bằng chứng đề xuất: `han_che/bao_cao_han_che.md`.
- **MRI** (78 BN Val): pipeline đúng **37/78 (47.4%)**, macro-F1 35.8%; oracle (vùng GT) 42/78 (53.8%). YOLO tìm thấy tổn thương ở 540/622 volume; bỏ sót mọi tổn thương ở 2 BN. Lỗi: Mô hình phân loại sai ngay cả với vùng chuẩn: 27; Vùng giả của YOLO làm lệch kết quả gộp: 10; Vùng YOLO cắt khác vùng chuẩn: 2; YOLO bỏ sót toàn bộ tổn thương: 2. **Mô hình MRI chưa đạt chất lượng**; nhãn MRI không đáng tin cậy.
- Ca **P0267** (CHCC → HCC): bộ phân loại đoán HCC ngay cả trên vùng GT ở cả 4 pha; YOLO tìm đúng khối chính. Xem §3.

## 1. CT: 54 bệnh nhân Validation (42 có ung thư + 12 nhóm chứng CN)

### Phân loại mức bệnh nhân

| Cách lấy vùng | BN có nhãn | Được phân loại | Đúng | Accuracy (tính BN không phân loại là sai) | Macro-Precision | Macro-Recall | Macro-F1 | Balanced acc |
|---|---|---|---|---|---|---|---|---|
| **Pipeline (vùng YOLO, hệ thống thật)** | 42 | 40 | 36 | **85.7%** | 91.6% | 89.7% | 89.9% | 89.7% |
| Chỉ vùng YOLO khớp GT (chẩn đoán) | 42 | 42 | 36 | **85.7%** | 86.3% | 85.5% | 85.6% | 85.5% |
| Oracle: vùng cắt theo nhãn chuẩn | 42 | 42 | 37 | **88.1%** | 89.6% | 87.7% | 88.0% | 87.7% |

Precision / Recall / F1 tính trên các BN được phân loại; accuracy tính trên mọi BN có nhãn. Ma trận nhầm lẫn: [pipeline](cm_ct_pipeline.png), [oracle](cm_ct_oracle.png).

| Lớp | Số BN | Pipeline đúng | Oracle đúng | Pipeline dự đoán thành |
|---|---|---|---|---|
| HCC (ung thư biểu mô tế bào gan) | 14 | 14 | 13 | HCC: 14 |
| ICC (ung thư biểu mô đường mật trong gan) | 15 | 12 | 14 | ICC: 12, không phân loại: 2, HCC: 1 |
| CHCC (ung thư hỗn hợp tế bào gan – đường mật) | 13 | 10 | 10 | CHCC: 10, HCC: 2, ICC: 1 |

| Lớp (pipeline, mức BN) | Precision | Recall (Sensitivity) | Specificity | F1 |
|---|---|---|---|---|
| HCC | 82.4% | 100.0% | 88.5% | 90.3% |
| ICC | 92.3% | 92.3% | 96.3% | 92.3% |
| CHCC | 100.0% | 76.9% | 100.0% | 87.0% |

**Nhóm chứng CN (12 BN, không có tổn thương được thu thập):** 1 BN không có vùng YOLO nào; 9 BN bị cổng bằng chứng từ chối (không đưa nhãn); 2 BN bị gán nhãn ung thư từ vùng giả: HCC 1, ICC 1. Đây là dương tính giả ở mức bệnh nhân, do mô hình 3 lớp không có lớp 'không tổn thương'.

### Phát hiện (YOLO) và Tumor Zoom

| Chỉ số | Giá trị |
|---|---|
| Volume có tổn thương được YOLO tìm thấy | 152/163 (93.3%) |
| Tổn thương 3D được tìm thấy | 177/275 (64.4%) |
| Tổn thương nhỏ (< 32×32 px) được tìm thấy | 14/69 (20.3%) |
| BN có tổn thương mà YOLO bỏ sót **mọi** tổn thương | 0/42 |
| BN không có vùng YOLO nào (trạng thái `no_lesion_detected`) | 1/54 |
| Vùng YOLO giả (không trùng tổn thương thật) | 1057/1367 (77.3%), trung bình 5.0 mỗi volume |

### Theo pha CT (mức volume, BN có tổn thương)

| Pha | Volume | Tổn thương được YOLO tìm thấy | Pipeline đúng | Oracle đúng |
|---|---|---|---|---|
| C1 | 40 | 36/40 | 35/40 (87.5%) | 34/40 (85.0%) |
| C2 | 41 | 39/41 | 35/41 (85.4%) | 37/41 (90.2%) |
| C3 | 41 | 38/41 | 35/41 (85.4%) | 35/41 (85.4%) |
| P | 41 | 39/41 | 35/41 (85.4%) | 35/41 (85.4%) |

### Phân tích lỗi (BN có tổn thương bị pipeline phân loại sai hoặc không phân loại)

| BN | Nhãn thực tế | Pipeline | Oracle | Chỉ vùng khớp GT | Vùng (khớp GT / tổng) | Tỉ lệ ROI từ vùng giả | Tổn thương tìm thấy | Toàn tổn thương nhỏ | Nhiều tổn thương | Nguyên nhân | Hình |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P0132 | ICC | không phân loại | ICC | ICC | 2/28 | 94% | 2/4 | không | không | Bị cổng bằng chứng từ chối (không đưa nhãn) | [hình](figures/ct/P0132_khong_phan_loai.png) |
| P0133 | ICC | không phân loại | ICC | ICC | 2/10 | 83% | 1/7 | không | có | Bị cổng bằng chứng từ chối (không đưa nhãn) | [hình](figures/ct/P0133_khong_phan_loai.png) |
| P0148 | ICC | HCC | HCC | HCC | 5/28 | 64% | 4/5 | không | có | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/ct/P0148_sai.png) |
| P0200 | CHCC | ICC | ICC | ICC | 4/16 | 44% | 4/4 | không | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/ct/P0200_sai.png) |
| P0266 | CHCC | HCC | HCC | HCC | 1/44 | 99% | 1/4 | không | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/ct/P0266_sai.png) |
| P0267 | CHCC | HCC | HCC | HCC | 12/29 | 38% | 4/11 | không | có | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/ct/P0267_sai.png) |

Tổng hợp nguyên nhân: Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai): **4**; Bị cổng bằng chứng từ chối (không đưa nhãn): **2**.

Ví dụ ca đúng: [P0011](figures/ct/P0011_dung.png), [P0014](figures/ct/P0014_dung.png), [P0015](figures/ct/P0015_dung.png), [P0018](figures/ct/P0018_dung.png). Mọi ca có hình trong `figures/ct/`.

## 2. MRI: 78 bệnh nhân Validation

### Phân loại mức bệnh nhân

| Cách lấy vùng | BN có nhãn | Được phân loại | Đúng | Accuracy (tính BN không phân loại là sai) | Macro-Precision | Macro-Recall | Macro-F1 | Balanced acc |
|---|---|---|---|---|---|---|---|---|
| **Pipeline (vùng YOLO, hệ thống thật)** | 78 | 78 | 37 | **47.4%** | 40.3% | 37.0% | 35.8% | 37.0% |
| Chỉ vùng YOLO khớp GT (chẩn đoán) | 78 | 76 | 39 | **50.0%** | 50.8% | 48.9% | 47.8% | 48.9% |
| Oracle: vùng cắt theo nhãn chuẩn | 78 | 78 | 42 | **53.8%** | 55.0% | 51.7% | 52.1% | 51.7% |

Precision / Recall / F1 tính trên các BN được phân loại; accuracy tính trên mọi BN có nhãn. Ma trận nhầm lẫn: [pipeline](cm_mri_pipeline.png), [oracle](cm_mri_oracle.png).

| Lớp | Số BN | Pipeline đúng | Oracle đúng | Pipeline dự đoán thành |
|---|---|---|---|---|
| HCC (ung thư biểu mô tế bào gan) | 25 | 21 | 16 | HCC: 21, FNH: 2, U máu gan: 2 |
| ICC (ung thư đường mật trong gan) | 9 | 0 | 3 | HCC: 6, U máu gan: 1, Di căn gan: 1, Nang gan: 1 |
| Di căn gan | 8 | 2 | 1 | HCC: 6, Di căn gan: 2 |
| U máu gan | 13 | 7 | 7 | U máu gan: 7, HCC: 5, Nang gan: 1 |
| Nang gan | 8 | 2 | 5 | HCC: 3, Nang gan: 2, U máu gan: 2, ICC: 1 |
| Áp xe gan | 8 | 0 | 4 | HCC: 6, ICC: 1, U máu gan: 1 |
| FNH (tăng sản nốt khu trú) | 7 | 5 | 6 | FNH: 5, HCC: 1, U máu gan: 1 |

| Lớp (pipeline, mức BN) | Precision | Recall (Sensitivity) | Specificity | F1 |
|---|---|---|---|---|
| HCC | 43.8% | 84.0% | 49.1% | 57.5% |
| ICC | 0.0% | 0.0% | 97.1% | 0.0% |
| Di căn gan | 66.7% | 25.0% | 98.6% | 36.4% |
| U máu gan | 50.0% | 53.8% | 89.2% | 51.9% |
| Nang gan | 50.0% | 25.0% | 97.1% | 33.3% |
| Áp xe gan | 0.0% | 0.0% | 100.0% | 0.0% |
| FNH | 71.4% | 71.4% | 97.2% | 71.4% |
- Gộp **lành tính / ác tính** từ dự đoán 7 lớp (Pipeline): đúng 55/78 (70.5%).
- Gộp **lành tính / ác tính** từ dự đoán 7 lớp (Oracle): đúng 57/78 (73.1%).

### Phát hiện (YOLO) và Tumor Zoom

| Chỉ số | Giá trị |
|---|---|
| Volume có tổn thương được YOLO tìm thấy | 540/622 (86.8%) |
| Tổn thương 3D được tìm thấy | 554/648 (85.5%) |
| Tổn thương nhỏ (< 32×32 px) được tìm thấy | 188/266 (70.7%) |
| BN có tổn thương mà YOLO bỏ sót **mọi** tổn thương | 2/78 |
| BN không có vùng YOLO nào (trạng thái `no_lesion_detected`) | 0/78 |
| Vùng YOLO giả (không trùng tổn thương thật) | 2834/3413 (83.0%), trung bình 4.5 mỗi volume |

### Theo chuỗi MRI (mức volume, BN có tổn thương)

| Chuỗi | Volume | Tổn thương được YOLO tìm thấy | Pipeline đúng | Oracle đúng |
|---|---|---|---|---|
| C+A | 78 | 70/78 | 39/78 (50.0%) | 36/78 (46.2%) |
| C+Delay | 78 | 66/78 | 35/78 (44.9%) | 44/78 (56.4%) |
| C+V | 78 | 72/78 | 35/78 (44.9%) | 44/78 (56.4%) |
| C-pre | 77 | 66/77 | 34/77 (44.2%) | 38/77 (49.4%) |
| DWI | 78 | 67/77 | 31/78 (39.7%) | 35/77 (45.5%) |
| In Phase | 78 | 64/78 | 35/78 (44.9%) | 32/78 (41.0%) |
| Out Phase | 78 | 66/78 | 32/78 (41.0%) | 31/78 (39.7%) |
| T2WI | 78 | 69/78 | 36/78 (46.2%) | 32/78 (41.0%) |

### Phân tích lỗi (BN có tổn thương bị pipeline phân loại sai hoặc không phân loại)

| BN | Nhãn thực tế | Pipeline | Oracle | Chỉ vùng khớp GT | Vùng (khớp GT / tổng) | Tỉ lệ ROI từ vùng giả | Tổn thương tìm thấy | Toàn tổn thương nhỏ | Nhiều tổn thương | Nguyên nhân | Hình |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MR-400851 | HCC | FNH | FNH | FNH | 8/49 | 68% | 8/8 | không | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR-400851_sai.png) |
| MR104280 | Áp xe gan | HCC | Áp xe gan | HCC | 8/23 | 34% | 8/8 | không | không | Vùng YOLO cắt khác vùng chuẩn (oracle đúng, vùng YOLO thật vẫn sai) | [hình](figures/mri/MR104280_sai.png) |
| MR106372 | Áp xe gan | HCC | HCC | HCC | 10/53 | 55% | 8/8 | không | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR106372_sai.png) |
| MR124193 | HCC | U máu gan | U máu gan | U máu gan | 8/31 | 52% | 8/8 | không | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR124193_sai.png) |
| MR125176 | Nang gan | HCC | HCC | HCC | 7/55 | 85% | 7/8 | có | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR125176_sai.png) |
| MR127280 | Di căn gan | HCC | HCC | HCC | 7/72 | 82% | 7/8 | không | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR127280_sai.png) |
| MR137627 | Áp xe gan | HCC | HCC | HCC | 8/28 | 57% | 8/8 | không | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR137627_sai.png) |
| MR140232 | Nang gan | U máu gan | Nang gan | không phân loại | 0/35 | 100% | 0/8 | có | không | YOLO bỏ sót toàn bộ tổn thương (không có vùng nào trùng tổn thương thật) | [hình](figures/mri/MR140232_sai.png) |
| MR140376 | Nang gan | U máu gan | U máu gan | U máu gan | 6/35 | 82% | 6/8 | có | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR140376_sai.png) |
| MR142442 | Nang gan | HCC | Nang gan | không phân loại | 0/46 | 100% | 0/8 | có | không | YOLO bỏ sót toàn bộ tổn thương (không có vùng nào trùng tổn thương thật) | [hình](figures/mri/MR142442_sai.png) |
| MR145114 | Áp xe gan | ICC | Áp xe gan | ICC | 9/47 | 59% | 8/8 | không | không | Vùng YOLO cắt khác vùng chuẩn (oracle đúng, vùng YOLO thật vẫn sai) | [hình](figures/mri/MR145114_sai.png) |
| MR146022 | ICC | U máu gan | U máu gan | U máu gan | 6/39 | 64% | 6/8 | không | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR146022_sai.png) |
| MR159752 | Nang gan | HCC | Di căn gan | Di căn gan | 5/36 | 81% | 5/8 | có | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR159752_sai.png) |
| MR162257 | Áp xe gan | U máu gan | U máu gan | U máu gan | 8/37 | 49% | 9/9 | không | có | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR162257_sai.png) |
| MR173412 | U máu gan | Nang gan | Nang gan | Nang gan | 6/23 | 62% | 6/8 | có | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR173412_sai.png) |
| MR174815 | ICC | HCC | HCC | HCC | 10/50 | 54% | 8/8 | không | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR174815_sai.png) |
| MR174870 | U máu gan | HCC | U máu gan | U máu gan | 8/40 | 62% | 8/8 | không | không | Vùng giả của YOLO làm lệch kết quả gộp (chỉ vùng thật thì đúng; oracle đúng) | [hình](figures/mri/MR174870_sai.png) |
| MR175358 | U máu gan | HCC | HCC | HCC | 7/24 | 38% | 7/9 | không | có | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR175358_sai.png) |
| MR179548 | U máu gan | HCC | Nang gan | Nang gan | 8/33 | 68% | 8/8 | có | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR179548_sai.png) |
| MR179970 | ICC | HCC | HCC | HCC | 8/25 | 32% | 8/8 | không | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR179970_sai.png) |
| MR184663 | U máu gan | HCC | U máu gan | U máu gan | 2/34 | 95% | 2/8 | có | không | Vùng giả của YOLO làm lệch kết quả gộp (chỉ vùng thật thì đúng; oracle đúng) | [hình](figures/mri/MR184663_sai.png) |
| MR192701 | Áp xe gan | HCC | Áp xe gan | Áp xe gan | 8/63 | 74% | 8/9 | không | có | Vùng giả của YOLO làm lệch kết quả gộp (chỉ vùng thật thì đúng; oracle đúng) | [hình](figures/mri/MR192701_sai.png) |
| MR193842 | Di căn gan | HCC | HCC | Nang gan | 6/47 | 77% | 6/8 | có | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR193842_sai.png) |
| MR198925 | FNH | HCC | FNH | FNH | 4/33 | 74% | 4/8 | không | không | Vùng giả của YOLO làm lệch kết quả gộp (chỉ vùng thật thì đúng; oracle đúng) | [hình](figures/mri/MR198925_sai.png) |
| MR199345 | ICC | Di căn gan | ICC | ICC | 8/83 | 82% | 8/9 | không | có | Vùng giả của YOLO làm lệch kết quả gộp (chỉ vùng thật thì đúng; oracle đúng) | [hình](figures/mri/MR199345_sai.png) |
| MR200076 | U máu gan | HCC | HCC | HCC | 8/79 | 78% | 8/8 | không | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR200076_sai.png) |
| MR201013 | Nang gan | ICC | Nang gan | Nang gan | 2/28 | 91% | 2/8 | có | không | Vùng giả của YOLO làm lệch kết quả gộp (chỉ vùng thật thì đúng; oracle đúng) | [hình](figures/mri/MR201013_sai.png) |
| MR210193 | Di căn gan | HCC | U máu gan | HCC | 6/55 | 87% | 6/8 | có | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR210193_sai.png) |
| MR222125 | Di căn gan | HCC | FNH | ICC | 8/39 | 67% | 7/7 | không | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR222125_sai.png) |
| MR222216 | ICC | HCC | ICC | ICC | 5/42 | 75% | 5/8 | không | không | Vùng giả của YOLO làm lệch kết quả gộp (chỉ vùng thật thì đúng; oracle đúng) | [hình](figures/mri/MR222216_sai.png) |
| MR226356 | FNH | U máu gan | FNH | FNH | 5/73 | 93% | 5/8 | có | không | Vùng giả của YOLO làm lệch kết quả gộp (chỉ vùng thật thì đúng; oracle đúng) | [hình](figures/mri/MR226356_sai.png) |
| MR229934 | Di căn gan | HCC | U máu gan | U máu gan | 7/62 | 90% | 7/8 | có | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR229934_sai.png) |
| MR236955 | Di căn gan | HCC | ICC | ICC | 10/66 | 71% | 8/8 | không | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR236955_sai.png) |
| MR27202 | ICC | Nang gan | HCC | HCC | 8/85 | 81% | 8/8 | không | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR27202_sai.png) |
| MR32504 | ICC | HCC | ICC | ICC | 8/68 | 78% | 8/8 | không | không | Vùng giả của YOLO làm lệch kết quả gộp (chỉ vùng thật thì đúng; oracle đúng) | [hình](figures/mri/MR32504_sai.png) |
| MR45055 | HCC | U máu gan | U máu gan | U máu gan | 8/28 | 49% | 8/8 | có | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR45055_sai.png) |
| MR61313 | HCC | FNH | FNH | FNH | 8/32 | 42% | 8/8 | không | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR61313_sai.png) |
| MR69046 | Áp xe gan | HCC | Áp xe gan | Áp xe gan | 11/50 | 57% | 10/10 | không | có | Vùng giả của YOLO làm lệch kết quả gộp (chỉ vùng thật thì đúng; oracle đúng) | [hình](figures/mri/MR69046_sai.png) |
| MR88293 | ICC | HCC | HCC | HCC | 9/41 | 57% | 8/9 | không | có | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR88293_sai.png) |
| MR94389 | ICC | HCC | HCC | HCC | 8/21 | 36% | 10/10 | không | có | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR94389_sai.png) |
| MR96745 | Áp xe gan | HCC | HCC | HCC | 8/34 | 44% | 8/8 | không | không | Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai) | [hình](figures/mri/MR96745_sai.png) |

Tổng hợp nguyên nhân: Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai): **27**; Vùng giả của YOLO làm lệch kết quả gộp (chỉ vùng thật thì đúng; oracle đúng): **10**; Vùng YOLO cắt khác vùng chuẩn (oracle đúng, vùng YOLO thật vẫn sai): **2**; YOLO bỏ sót toàn bộ tổn thương (không có vùng nào trùng tổn thương thật): **2**.

Ví dụ ca đúng: [MR-451771](figures/mri/MR-451771_dung.png), [MR-470019](figures/mri/MR-470019_dung.png), [MR109260](figures/mri/MR109260_dung.png), [MR110203](figures/mri/MR110203_dung.png). Mọi ca có hình trong `figures/mri/`.

## 3. Ca P0267 (nhãn thực tế CHCC, pipeline dự đoán HCC)

![P0267](figures/ct/P0267_sai.png)

Xác suất pipeline: HCC 70.1%, ICC 19.2%, CHCC 10.7%. Oracle (vùng GT): **HCC** (HCC 57.6%, ICC 34.4%, CHCC 8.1%). Chỉ vùng YOLO khớp GT: **HCC**.

| Pha | Lát | Vùng YOLO | Tổn thương GT | GT được tìm thấy | Diện tích lớn nhất của tổn thương GT (px) | Pipeline (pha) | Oracle (pha) |
|---|---|---|---|---|---|---|---|
| C1 | 140 | 9 | 3 | 1 | [15872, 598, 1558] | HCC | HCC |
| C3 | 136 | 7 | 2 | 1 | [14231, 1188] | HCC | HCC |
| P | 141 | 4 | 3 | 1 | [12700, 1312, 1258] | HCC | HCC |
| C2 | 138 | 9 | 3 | 1 | [19110, 806, 1287] | HCC | HCC |

- 29 vùng YOLO trên 4 pha, trong đó 12 vùng trùng tổn thương thật; 38% số ROI đưa vào phân loại đến từ vùng giả.
- **Bộ phân loại sai ngay cả trên vùng chuẩn (oracle → HCC)** và cả khi chỉ dùng vùng YOLO khớp GT (→ HCC): lỗi nằm ở **bộ phân loại**, không phải do YOLO bỏ sót hay Tumor Zoom cắt sai.
- YOLO tìm thấy 4/11 tổn thương GT 3D trên 4 pha; P0267 có nhiều ổ tổn thương trong cùng pha. Các ổ không được tìm thấy không làm đổi kết luận, vì oracle (đủ mọi ổ) cũng đoán HCC.
- CHCC (ung thư hỗn hợp tế bào gan – đường mật) mang đặc điểm của cả HCC và ICC, nên dễ nhầm sang một trong hai lớp này. Đây là giải thích hợp lý về mặt bệnh học, nhưng *chưa được kiểm chứng* bằng thực nghiệm riêng.
- Đã kiểm tra tiền xử lý: crop của pipeline trùng từng pixel với dữ liệu đánh giá (`reports/pipeline_val_report.md` §3), nên không phải lỗi tiền xử lý.

## 4. Đề xuất cải thiện (có căn cứ từ số liệu trên; mọi thử nghiệm chỉ chọn trên Validation)

1. **Vùng giả chi phối kết quả gộp.** Tỉ lệ ROI đến từ vùng giả rất cao; với các BN mà oracle đúng nhưng pipeline sai, nên thử *gộp có trọng số theo độ tin cậy YOLO*, hoặc chỉ dùng k vùng tin cậy nhất (chọn trên Val). Không dùng GT lúc suy luận.
2. **Thêm lớp 'không phải tổn thương' cho bộ phân loại** (huấn luyện bằng các vùng YOLO giả trên Train) để loại vùng giả và tránh gán nhãn ung thư cho BN nhóm chứng CN.
3. **CT: CHCC** là lớp khó nhất về bản chất (hỗn hợp HCC và ICC). Cần xem ma trận nhầm lẫn theo lớp, có thể dùng nhiều pha cùng lúc (fusion C1/C2/C3/P ở mức tổn thương).
4. **MRI:** mô hình overfit nặng (train acc ≈ 0.97, Val ≈ 0.48). Hướng thử: fusion đa chuỗi ở mức tổn thương (8 chuỗi của cùng tổn thương cùng vào một mẫu), mô hình nhỏ hơn hoặc đóng băng một phần backbone, augmentation mạnh hơn, và bài toán lành / ác tính 2 lớp.
5. **YOLO bỏ sót tổn thương nhỏ:** đã ghi nhận ở giai đoạn detection. Nếu được phép huấn luyện thêm: imgsz 640 hoặc bộ CT đầy đủ.
6. Mọi số trên là Validation (đã dùng để chọn mô hình và cách gộp), nên còn lạc quan. Kết luận cuối phải chờ đánh giá Test **một lần**.

