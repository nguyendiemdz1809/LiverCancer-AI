## 1. Mục tiêu thực nghiệm

Kiểm tra lựa chọn bộ phát hiện tổn thương của dự án bằng một **thí nghiệm đối chứng**: huấn luyện YOLOv5n trên đúng dữ liệu, đúng cách chia tập và đúng siêu tham số mà YOLO11n đã dùng, rồi đánh giá hai mô hình bằng cùng mã, cùng GPU, trên cùng tập Validation và Test. Hai thí nghiệm riêng: **YOLOv5n-CT** đối chứng **YOLO11n-CT**, **YOLOv5n-MRI** đối chứng **YOLO11n-MRI** (CT và MRI không gộp, giống cách dự án huấn luyện YOLO11n).

Thí nghiệm **không** thay mô hình chính: YOLO11n vẫn là bộ phát hiện của pipeline và website (checkpoint, ngưỡng, Evidence Gate, Test chính thức của pipeline đều giữ nguyên). Kết quả mới nằm riêng trong `outputs/runs/comparison/`.

## 2. Lý do lựa chọn YOLOv5n

- **Cùng nhóm kích thước Nano** (khoảng 2,6 triệu tham số, checkpoint khoảng 5,5 MB), nên so sánh là giữa hai **thế hệ kiến trúc** chứ không phải giữa hai cỡ mô hình. YOLOv5s (khoảng 9 triệu tham số) sẽ lẫn hiệu ứng "mô hình to hơn" vào kết quả.
- YOLOv5 là phiên bản YOLO được dùng rộng rãi, thường làm mốc so sánh trong các nghiên cứu phát hiện đối tượng, nên là đối chứng tự nhiên cho câu hỏi "tại sao không dùng YOLOv5".
- Dùng bản `yolov5nu.pt` (YOLOv5n chính thức của Ultralytics, COCO pretrained) để chạy **trong cùng thư viện Ultralytics 8.4.182** với YOLO11n: cùng bộ nạp dữ liệu, augmentation, hàm loss, NMS và cách tính mAP. Nhờ vậy khác biệt đo được đến từ kiến trúc mạng, không đến từ hai bộ mã huấn luyện khác nhau. Đánh đổi: đầu dự đoán của `yolov5nu` là đầu anchor-free tách nhánh của thư viện hiện tại, không phải đầu anchor-based của repo YOLOv5 gốc (xem mục 3).

## 3. Giới thiệu ngắn gọn YOLOv5n và YOLO11n

| | YOLOv5n (`yolov5nu.pt`) | YOLO11n (`yolo11n.pt`) |
|---|---|---|
| Ra đời | YOLOv5: Ultralytics, 2020; bản "u" cập nhật sau đó | Ultralytics, 2024 |
| Backbone | CSPDarknet với khối **C3**, SPPF | Khối **C3k2** (CSP với nút cổ chai kernel nhỏ), SPPF, thêm **C2PSA** (khối chú ý theo vị trí) sau SPPF |
| Neck | PANet (FPN + PAN) với khối C3 | PANet với khối C3k2 |
| Đầu dự đoán | Anchor-free, tách nhánh box/lớp, DFL (bản "u"); YOLOv5 gốc dùng anchor | Anchor-free, tách nhánh, DFL; nhánh phân lớp dùng tích chập tách chiều sâu (depthwise) |
| Tham số (đo trên checkpoint của dự án, 1 lớp) | 2.635.619 | 2.604.883 |
| GFLOPs (đo ở ảnh 512) | 4,95 | 4,19 |
| Checkpoint `best.pt` | 5,51 MB | 5,48 MB |

Theo bảng công bố của Ultralytics trên COCO (ảnh 640), YOLO11n đạt mAP50-95 cao hơn YOLOv5nu (khoảng 39,5 so với 34,3) với ít FLOPs hơn (6,5 so với 7,7 GFLOPs). Đây là kết quả trên ảnh tự nhiên (80 lớp), **không** suy ra được cho ảnh CT/MRI một lớp tổn thương; chính vì vậy cần thí nghiệm này. Tỷ lệ GFLOPs đo ở ảnh 512 trong dự án (4,95 / 4,19 = 1,18) khớp với tỷ lệ công bố ở 640 (7,7 / 6,5 = 1,18).

## 4. Mô tả dataset CT/MRI

Cả hai mô hình dùng **đúng hai tệp dữ liệu YOLO đã huấn luyện YOLO11n** trên Drive (MRI: SHA256 `0088fa93…` ghi trong `metrics_val.json` khớp `data/processed/pilot/SHA256SUMS.txt`; CT: lượt resume của YOLOv5n-CT không ghi SHA256, cột để trống, tên tệp `yolo_ct_full.zip` trùng), tổng **85.586 ảnh** Train + Validation; Test giữ riêng.

| Ảnh | Nguồn | Bộ YOLO | Train | Validation | Test (chỉ lát có tổn thương) |
|---|---|---|---|---|---|
| CT | PLC-CECT (361 BN, 4 pha, ảnh `uint8`) | `yolo_ct_full.zip` | 34.873 ảnh (253 BN) | 14.627 ảnh (54 BN) | 7.163 ảnh (42 BN ung thư trong 54 BN Test) |
| MRI | LLD-MMRI-MedSAM2 (498 BN, 8 chuỗi) | `yolo_mri_pilot.zip` | 28.595 ảnh (316 BN) | 7.491 ảnh (78 BN) | 6.945 ảnh (104 BN) |

- Một lớp duy nhất `lesion`; bounding box **chuyển tự động từ mask tổn thương** của dataset gốc (không gán tay), lát cắt 2D grayscale 512 px.
- CT Train lấy một nửa số lát có tổn thương (cách 1 lát) cùng lát không tổn thương; Validation lấy đủ. Nhóm CN (không có bệnh gan mật) chỉ đóng vai trò ảnh âm tính.
- MRI: lát dương tính chỉ được giữ khi mask MedSAM2 và bbox gốc của JSON đồng thuận (IoU ≥ 0,3); 31 series lệch nặng bị cách ly. Test MRI dùng cùng quy tắc.
- Test là bộ GT do `scripts/96_final_test.py` xuất từ volume NIfTI gốc: chỉ gồm lát **có tổn thương**, không có lát âm tính hay BN nhóm chứng. Bảng thống kê đầy đủ: `reports/thong_ke_du_lieu_chinh_thuc.md`.

## 5. Phương pháp chia dữ liệu

- Chia **theo bệnh nhân**: mọi pha/chuỗi/lát của một BN nằm trong cùng một tập; danh sách đã đóng băng ở `data/splits/ct_split.csv`, `data/splits/mri_split.csv` và được `tests/test_splits_no_leakage.py` kiểm tra không rò rỉ.
- **CT:** 70/15/15 phân tầng theo `cancer_type`, seed 42: 253 / 54 / 54 BN.
- **MRI:** Validation = 78 ID validation chính thức của LLD-MMRI2023; 420 BN còn lại chia phân tầng theo lớp thành Train 316 / Test 104 (seed 42).
- Thí nghiệm này **không** tạo split mới và không đổi nhãn. Checkpoint được chọn theo Validation (`best.pt`, fitness = mAP50-95); Test chỉ chạy **một lần** cho cả hai mô hình sau khi đã chọn xong checkpoint (ngoại lệ có chủ đích do người dùng quyết định ngày 2026-11-09, khóa bằng `outputs/runs/comparison/test_done.json`), không dùng để chọn bất kỳ tham số nào.
