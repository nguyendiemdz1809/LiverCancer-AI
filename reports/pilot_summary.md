# Tổng kết giai đoạn Pilot YOLO: CT và MRI

*Mọi số liệu lấy từ `outputs/runs/yolo_{ct,mri}/pilot_yolo11n_e5/` (`results.csv`, `metrics_val.json`, `analysis/*.json`),
do `scripts/50_analyze_yolo_run.py` và `scripts/51_ct_fp_fn_breakdown.py` sinh ra. Phân tích chi tiết:
[`ct_pilot_analysis.md`](ct_pilot_analysis.md) và [`mri_pilot_analysis.md`](mri_pilot_analysis.md).
**Chỉ dùng Validation.** Test chưa từng được dùng. Hai run đều là **pilot 5 epoch** để kiểm tra pipeline, không phải kết quả cuối.*

## 1. Cấu hình chung của hai pilot
YOLO11n pretrained (COCO), 5 epoch, imgsz 512, batch 16, seed 42, Colab T4, Ultralytics 8.4.182. Ultralytics mặc định:
`optimizer=auto`, và cột lr trong `results.csv` của cả hai run xác nhận **AdamW lr0 = 0.002**; 3/5 epoch là warmup; mosaic tắt
vì `close_mosaic = 10 ≥ 5`. Hai run có **cùng công thức huấn luyện**; khác biệt nằm ở dữ liệu.

## 2. Kết quả trên Validation

| | CT (PLC-CECT) | MRI (LLD-MMRI-MedSAM2) |
|---|---|---|
| BN Val / lát Val / box GT | 54 / 7.382 / 3.850 | 78 / 7.491 / 4.880 |
| `best.pt` (fitness = mAP50-95) | epoch 5 | epoch 5 |
| **Precision / Recall** (tại F1 max) | **0.428 / 0.355** | **0.766 / 0.630** |
| **mAP50 / mAP50-95** | **0.333 / 0.170** | **0.712 / 0.471** |
| F1 max, conf tương ứng | 0.39 @ 0.24 | 0.69 @ 0.33 |
| Recall tại conf 0.001 (trần, IoU ≥ 0.5) | 0.847 | 0.946 |
| Ở conf 0.25: P / R / F1 | 0.463 / 0.363 / 0.407 | 0.703 / 0.676 / 0.689 |
| Ở conf 0.25: FP/lát; lát âm có FP | 0.22; 11.1% | 0.19; 11.4% |
| Ở conf 0.25: BN có tổn thương được phát hiện | 30/42 | 75/78 |
| Ở conf 0.25: FN theo kích thước (nhỏ / vừa / lớn) | 97% / 74% / 12% | 52% / 29% / 17% |
| Ở conf 0.25: FP là lỗi định vị | 47% | 42% |
| Xu hướng loss val | giảm nhưng dao động (cls 5.4→6.0→3.9→5.4→4.6) | giảm đều |
| Thời gian/epoch (T4) | ~200 s | ~340 s |

Confusion matrix của Ultralytics (conf 0.001) có hàng background rất lớn (CT 287.448; MRI 176.950) vì ma trận tính ở conf 0.001.
Ở conf 0.25: CT TP 1.421 / FP 1.707 / FN 2.429; MRI TP 3.339 / FP 1.405 / FN 1.541.

## 3. Vì sao **không** kết luận mô hình nào tốt hơn chỉ từ mAP
1. **Hai bài toán khác nhau:** khác modality, khác máy chụp và quy trình; CT là cường độ đã windowing, MRI là chuẩn hóa theo từng volume.
2. **Nguồn nhãn khác nhau:**
   - CT: mask do bác sĩ vẽ; mọi ca có chẩn đoán mô học hoặc tế bào học.
   - MRI: mask dẫn xuất từ MedSAM2, và **chỉ giữ những lát mà mask đồng thuận với JSON box gốc** (IoU ≥ 0.3).
     Bộ lọc này loại các lát "khó" hoặc mơ hồ, nên tập nhãn MRI "sạch" hơn theo cách xây dựng.
3. **Loại tổn thương khác nhau:**
   - CT chỉ có ung thư gan nguyên phát (HCC/ICC/CHCC).
   - MRI có cả tổn thương lành tính (nang, áp xe, u máu…), vốn có độ tương phản khác.
4. **Mỗi tổn thương MRI xuất hiện trên 8 chuỗi**, nên các mẫu có tương quan và số tổn thương độc lập ít hơn số box.
5. **Dữ liệu train khác nhau:** CT pilot chỉ lấy **1/4** số lát dương tính ở train (8.917 lát), còn MRI lấy **toàn bộ** lát hợp lệ (18.619).
   Vì vậy MRI có 1.6× số iteration (8.940 so với 5.570) và gấp đôi số lát dương tính.
6. **Thành phần Val khác nhau:** CT val chỉ lấy 1/2 số lát dương tính và 50% lát âm (có 12 BN CN); MRI có 35% lát âm và không có BN âm tính.
   Tỉ lệ âm khác nhau làm Precision và mAP khác nhau ngay cả khi mô hình như nhau.
7. **Kích thước tổn thương và ảnh khác nhau:** CT luôn 512², còn MRI có từ 176² đến 512². Tỉ lệ box nhỏ trong Val: CT 15%, MRI 26%.
8. **Chỉ 1 seed, 5 epoch, cả hai run đều chưa hội tụ.**

→ Kết luận hợp lệ: (a) pipeline dữ liệu, nhãn và Colab của **cả hai modality hoạt động đúng** (kết quả tái lập được ở local,
TP + FN = số box GT); (b) **cả hai mô hình chưa được huấn luyện đủ**; (c) điểm yếu chung là **tổn thương nhỏ** và **box chưa khít**.
Không xếp hạng CT so với MRI.

## 4. Những điều rút ra cho training chính thức
- Tăng epoch (100 + early stopping) và dùng **optimizer tường minh**. Với `auto`, run dài sẽ chuyển sang MuSGD và không còn so sánh được.
- CT: tăng dữ liệu train (pilot mới dùng 26% số lát dương tính) và dùng **Val đầy đủ** để chọn mô hình.
- MRI: dữ liệu pilot đã đủ (93% số lát hợp lệ), dùng lại ZIP pilot.
- So sánh yolo11n với yolo11s; cân nhắc imgsz 640 nếu FN ở tổn thương nhỏ vẫn cao.
- Chọn ngưỡng vận hành trên Val, ưu tiên recall. Báo cáo theo kích thước, lớp và chuỗi/pha.
- Chưa thấy lỗi nhãn có hệ thống; không đề xuất sửa nhãn.

## 5. Trạng thái
- [x] CT pilot: train (Colab), phân tích (local), báo cáo.
- [x] MRI pilot: train (Colab), phân tích (local), báo cáo.
- [ ] Train chính thức: E0 (yolo11n) và E1 (yolo11s) cho CT và MRI. Cấu hình: `docs/PHASE3_TRAINING_PLAN.md` §3–4 và §8.
