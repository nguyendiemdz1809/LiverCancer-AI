# Kiểm toán kết quả 3D CT của website (lát nghi ngờ, vùng sau gộp)

Tạo bởi `scripts/74_case3d_audit.py`; số liệu: `outputs/case3d_audit/` (volumes.csv, P0002_C1.json, ảnh P0002_C1_lat_nghi_ngo.jpg).
Chạy đúng đường xử lý của website / pipeline đã khóa (YOLO CT ngưỡng 0,12, gộp box trong lát, nối vùng 3D theo configs/roi.yaml, không lọc vùng). Mask gốc chỉ dùng để kiểm toán, website không dùng. Không dùng Test.

Định nghĩa: *lát nghi ngờ* = lát có ít nhất một box của các vùng được giữ; *vùng* = chuỗi box nối qua các lát liên tiếp; vùng *trúng tổn thương* = IoU ≥ 0.3 với khung tổn thương thật trên một lát chung (cùng định nghĩa `scripts/71_eval_roi_val.py`); *trong gan* = tâm box nằm trong mask gan ∪ mask tổn thương.

## Ca 92 lát trên website: P0002, pha C1 (bệnh nhân thuộc tập **Train**)

Ca này thuộc tập huấn luyện: YOLO và Hybrid đã học trên chính ca này, nên kết quả (HCC 94,6%) **không** phản ánh khả năng của mô hình trên ca mới. Để trình diễn trung thực nên dùng ca Validation.

| Chỉ số | Giá trị |
|---|---|
| Số lát | 92 |
| Box YOLO thô (≥ 0,12) | 76 |
| Lát có box YOLO thô | 65 |
| Box sau gộp trong lát | 69 |
| Lát nghi ngờ | 65 |
| Vùng nghi ngờ (sau nối 3D) | 7 |
| Lát có gan (mask) | 90 |
| Lát nghi ngờ không có gan | 0 |
| Box có tâm ngoài gan | 1 |
| Vùng nằm hoàn toàn ngoài gan | 1 |
| Tổn thương thật (khối 3D trong mask) | 1 |
| … được ít nhất một vùng trúng | 1 |
| Lát có tổn thương thật | 64 |
| Lát nghi ngờ có tổn thương thật | 61 |
| Vùng trúng tổn thương (IoU ≥ 0,3) | 1 |
| Vùng chỉ chạm tổn thương (IoU < 0,3) | 2 |
| Vùng không chạm tổn thương | 4 |
| Tổn thương bị tách thành nhiều vùng | 0 |
| Vùng gộp nhầm nhiều tổn thương | 0 |

Từng vùng của ca này:

| Vùng (theo độ tin cậy) | Lát | Số lát | Độ tin cậy cao nhất | Lát đại diện | Trúng tổn thương | Chạm tổn thương |
|---|---|---|---|---|---|---|
| 1 | 28-88 | 61 | 0.86 | 40 | có | có |
| 2 | 7-7 | 1 | 0.455 | 7 | không | không |
| 3 | 20-20 | 1 | 0.345 | 20 | không | không |
| 4 | 1-2 | 2 | 0.245 | 2 | không | không |
| 5 | 29-30 | 2 | 0.188 | 30 | không | có |
| 6 | 20-20 | 1 | 0.164 | 20 | không | không |
| 7 | 59-59 | 1 | 0.131 | 59 | không | có |

## Toàn bộ CT Validation (211 volume; 163 có tổn thương, 47 nhóm chứng CN)

| Chỉ số | Tổng |
|---|---|
| Lát | 27560 |
| Lát nghi ngờ | 8091 |
| Vùng nghi ngờ | 1374 (trung bình 6.5/volume) |
| Volume CN (không có tổn thương) có vùng nghi ngờ | 36/47 |
| Box có tâm ngoài gan (volume có mask gan) | 1246/8738 |
| Lát nghi ngờ không có gan | 243/8091 |
| Vùng nằm hoàn toàn ngoài gan | 361/1374 |
| Tổn thương thật được trúng | 156/187 |
| Vùng trúng / chỉ chạm / không chạm tổn thương (volume có tổn thương) | 295 / 111 / 807 |
| Tổn thương bị tách thành nhiều vùng | 75 |
| Vùng gộp nhầm nhiều tổn thương | 0 |

**Kích thước khung YOLO so với tổn thương thật** (5554 khung chạm tổn thương, Validation): diện tích khung / diện tích khung tổn thương thật: trung vị 1.05, P75 1.44, P90 2.32; 4% khung lớn hơn 4 lần. Đa số khung gần kích thước tổn thương, nhưng một phần nhỏ lớn hơn nhiều lần; khối bao 3D (bao các khung qua các lát) vì vậy không đo kích thước khối u.

Lỗi đọc: không có.


## Ca người dùng hỏi: 55 lát, 4 vùng, vùng 152 × 207 × 12 mm (P0030 C1, bệnh nhân **Test**)

Chạy lại đúng đường xử lý website để chẩn đoán hiển thị; không thay đổi tham số nào theo ca này. Vùng 1 (lát 2–13, 12 lát, độ tin cậy cao nhất 0,80): mỗi lát chỉ có **một** tổn thương thật và khung của vùng chứa nó; khung hai lát liền nhau chồng lấp IoU 0,55–0,93 (> 0,3 của quy tắc nối). **Không phải lỗi gộp hai tổn thương.** Nhưng khung YOLO rộng hơn tổn thương thật rất nhiều (tổn thương khoảng 25–35 × 35–50 px; IoU khung/tổn thương 0,03–0,14); từ lát 9 YOLO cho 2–4 khung chồng nhau và bước gộp trong lát (đã khóa) lấy hợp của chúng, nên khung lát 13 là 152 × 207 px. Đây là ca thuộc nhóm hiếm có khung lớn hơn 4 lần tổn thương.
