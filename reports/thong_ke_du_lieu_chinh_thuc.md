# Thống kê dữ liệu chính thức – LiverCancer AI

*Sinh bởi `scripts/85_build_roboflow_full.py` lúc 2026-10-30 11:50; bản CSV: `reports/thong_ke_du_lieu_chinh_thuc.csv`. Dùng thống nhất trong báo cáo và slide.*

## Bộ dữ liệu YOLO (ảnh 2D) đã dùng để huấn luyện và chọn mô hình trên Google Colab

| Loại ảnh | Tập | Ảnh | Ảnh có tổn thương | Ảnh không có box | Bounding box | Bệnh nhân | Volume |
|---|---|---|---|---|---|---|---|
| CT | Train | 34.873 | 17.465 | 17.408 | 18.426 | 253 | 984 |
| CT | Validation | 14.627 | 7.314 | 7.313 | 7.631 | 54 | 210 |
| MRI | Train | 28.595 | 18.619 | 9.976 | 18.661 | 316 | 2.502 |
| MRI | Validation | 7.491 | 4.874 | 2.617 | 4.880 | 78 | 623 |
| **CT** | **Train + Validation** | **49.500** | 24.779 | 24.721 | 26.057 | 307 | 1.194 |
| **MRI** | **Train + Validation** | **36.086** | 23.493 | 12.593 | 23.541 | 394 | 3.125 |
| **Tổng** | | **85.586** | 48.272 | 37.314 | 49.598 | 701 | 4.319 |

## Tập Test (khóa, chỉ chạy một lần; **không** nằm trong 85.586 ảnh)

Test được đánh giá trên **volume NIfTI gốc** của 54 BN CT / 104 BN MRI. Bảng dưới là bộ ảnh GT chỉ gồm **lát có tổn thương** (xuất bởi `scripts/96_final_test.py` để chấm điểm phát hiện), nên không có ảnh không box và không có BN CN; **không** so sánh trực tiếp với số ảnh Train/Validation.

| Loại ảnh | Ảnh | Ảnh có tổn thương | Ảnh không có box | Bounding box | Bệnh nhân | Volume |
|---|---|---|---|---|---|---|
| CT | 7.163 | 7.163 | 0 | 7.776 | 42 | 159 |
| MRI | 6.945 | 6.945 | 0 | 6.965 | 104 | 828 |

## Cách diễn đạt thống nhất

- "Mô hình YOLO được huấn luyện **trên Google Colab** với **85.586 ảnh 2D** (CT 49.500, MRI 36.086), gồm tập Train và Validation; tập Test được giữ riêng."
- Nhãn bounding box được **chuyển đổi tự động** từ mask tổn thương của dataset gốc (PLC-CECT; LLD-MMRI-MedSAM2), không gán tay. Roboflow (nếu dùng) chỉ để **trực quan hóa và rà soát nhãn**; **không** train mô hình trên Roboflow.
- CT Train lấy 1/2 số lát có tổn thương (cách 1 lát); CT Validation lấy đủ. Ảnh không có box chỉ nghĩa là lát đó không có tổn thương được gán nhãn, không phải bệnh nhân khỏe mạnh.
- Số bệnh nhân: CT 253 Train / 54 Validation / 54 Test (42 BN ung thư + 12 BN CN); MRI 316 / 78 / 104 (chia theo bệnh nhân, seed 42).
