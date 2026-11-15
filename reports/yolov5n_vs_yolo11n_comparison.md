# So sánh YOLOv5n và YOLO11n (thí nghiệm đối chứng)

_Sinh bởi `scripts/78_yolov5n_comparison.py --stage report` lúc 2026-11-13 20:52. Chỉ số liệu đo được; YOLO11n vẫn là mô hình chính, không bị thay thế._

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

## 6. Cấu hình huấn luyện

- YOLOv5n = `yolov5nu.pt` (YOLOv5n chính thức của Ultralytics, COCO pretrained, chạy trong Ultralytics 8.4.182: backbone/neck YOLOv5, đầu dự đoán anchor-free tách nhánh của thư viện hiện tại). YOLO11n = checkpoint đã khóa của website (SHA256 kiểm tra).
- Cùng dữ liệu (CT `yolo_ct_full.zip`, MRI `yolo_mri_pilot.zip`), cùng chia tập theo bệnh nhân, cùng siêu tham số (notebook sinh từ cùng generator), cùng GPU khi đánh giá, cùng mã đánh giá.

| Ảnh | Mô hình | Khởi tạo | Dữ liệu | Epoch / patience | Batch | Optimizer, lr0 | Seed | Epoch đã chạy | Epoch tốt nhất | Thời gian train | GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CT | yolo11n | yolo11n.pt | yolo_ct_full.zip | 50 / 20 | 32 | SGD 0.01 | 42 | 35 | 15 | 3.82 h (1 phiên; 6.5 phút/epoch) | Tesla T4 |
| CT | yolov5n | yolov5nu.pt (resume từ last.pt) | yolo_ct_full.zip | 50 / 20 | 32 | SGD 0.01 | 42 | 40 | 20 | 4.77 h (2 phiên; 7.2 phút/epoch) | Tesla T4 |
| MRI | yolo11n | yolo11n.pt (resume từ last.pt) | yolo_mri_pilot.zip | 50 / 20 | 32 | SGD 0.01 | 42 | 44 | 24 | 3.9 h (2 phiên; 5.3 phút/epoch) | Tesla T4 |
| MRI | yolov5n | yolov5nu.pt (resume từ last.pt) | yolo_mri_pilot.zip | 50 / 20 | 32 | SGD 0.01 | 42 | 46 | 26 | 4.4 h (3 phiên; 5.7 phút/epoch) | Tesla T4 |

Thời gian train = tổng các phiên Colab (Ultralytics đặt lại đồng hồ về 0 sau mỗi lần resume); Colab T4 miễn phí dùng chung, tốc độ đọc dữ liệu thay đổi giữa các phiên, nên thời gian train chỉ để tham khảo, không dùng để so tốc độ mô hình (xem mục 9b). Checkpoint dùng để đánh giá là `best.pt` (fitness = mAP50-95 trên Val).

## 7. Kết quả YOLOv5n-CT và YOLO11n-CT

| Tập | Mô hình | Ảnh | P | R | F1 | mAP50 | mAP50-95 | P @0,12 | R @0,12 | F1 @0,12 | TP / FP / FN @0,12 | Recall box nhỏ @0,12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Validation | yolo11n | 14,627 | 0.688 | 0.480 | 0.566 | 0.551 | 0.282 | 0.474 | 0.578 | 0.521 | 4,409 / 4,889 / 3,222 | 0.263 (1133 box) |
| Validation | yolov5n | 14,627 | 0.726 | 0.498 | 0.590 | 0.578 | 0.293 | 0.491 | 0.597 | 0.539 | 4,557 / 4,729 / 3,074 | 0.265 (1133 box) |
| Test | yolo11n | 7,163 | 0.598 | 0.515 | 0.553 | 0.523 | 0.261 | 0.498 | 0.569 | 0.531 | 4,427 / 4,466 / 3,349 | 0.195 (1210 box) |
| Test | yolov5n | 7,163 | 0.671 | 0.543 | 0.600 | 0.589 | 0.293 | 0.550 | 0.601 | 0.575 | 4,676 / 3,826 / 3,100 | 0.227 (1210 box) |

P, R, F1: điểm F1 lớn nhất của Ultralytics (`val`, conf 0,001, IoU-NMS 0,7). “@0,12”: ngưỡng vận hành đã khóa của dự án (chọn trên Val cho YOLO11n), ghép cặp IoU ≥ 0,5. Box nhỏ: diện tích < 32×32 px. Test = mọi lát **có tổn thương** của BN Test (xuất bởi `scripts/96_final_test.py`), không có lát âm tính và nhóm chứng, nên precision trên Test không phản ánh báo nhầm trên lát bình thường. Test chạy **một lần** (khóa `outputs/runs/comparison/test_done.json`).

Nguồn: `outputs/runs/comparison/eval/ct_val.json`, `ct_test.json`; hình PR/confusion matrix: `outputs/runs/comparison/eval/plots/ct_<split>_<model>/`.

## 8. Kết quả YOLOv5n-MRI và YOLO11n-MRI

| Tập | Mô hình | Ảnh | P | R | F1 | mAP50 | mAP50-95 | P @0,12 | R @0,12 | F1 @0,12 | TP / FP / FN @0,12 | Recall box nhỏ @0,12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Validation | yolo11n | 7,491 | 0.796 | 0.660 | 0.722 | 0.749 | 0.504 | 0.634 | 0.744 | 0.685 | 3,633 / 2,099 / 1,247 | 0.598 (1275 box) |
| Validation | yolov5n | 7,491 | 0.802 | 0.656 | 0.722 | 0.739 | 0.504 | 0.638 | 0.734 | 0.683 | 3,583 / 2,034 / 1,297 | 0.605 (1275 box) |
| Test | yolo11n | 6,945 | 0.843 | 0.685 | 0.756 | 0.776 | 0.508 | 0.702 | 0.748 | 0.725 | 5,213 / 2,210 / 1,752 | 0.623 (1543 box) |
| Test | yolov5n | 6,945 | 0.829 | 0.669 | 0.741 | 0.757 | 0.499 | 0.691 | 0.731 | 0.711 | 5,094 / 2,279 / 1,871 | 0.591 (1543 box) |

P, R, F1: điểm F1 lớn nhất của Ultralytics (`val`, conf 0,001, IoU-NMS 0,7). “@0,12”: ngưỡng vận hành đã khóa của dự án (chọn trên Val cho YOLO11n), ghép cặp IoU ≥ 0,5. Box nhỏ: diện tích < 32×32 px. Test = mọi lát **có tổn thương** của BN Test (xuất bởi `scripts/96_final_test.py`), không có lát âm tính và nhóm chứng, nên precision trên Test không phản ánh báo nhầm trên lát bình thường. Test chạy **một lần** (khóa `outputs/runs/comparison/test_done.json`).

Nguồn: `outputs/runs/comparison/eval/mri_val.json`, `mri_test.json`; hình PR/confusion matrix: `outputs/runs/comparison/eval/plots/mri_<split>_<model>/`.

## 9. Bảng so sánh metrics

### 9a. Chênh lệch YOLOv5n − YOLO11n (điểm phần trăm; dương = YOLOv5n cao hơn)

| Ảnh | Tập | P | R | F1 | mAP50 | mAP50-95 | R @0,12 | F1 @0,12 | Recall box nhỏ @0,12 |
|---|---|---|---|---|---|---|---|---|---|
| CT | Validation | +3.8 | +1.7 | +2.5 | +2.7 | +1.1 | +1.9 | +1.8 | +0.2 |
| CT | Test | +7.4 | +2.8 | +4.7 | +6.6 | +3.2 | +3.2 | +4.3 | +3.2 |
| MRI | Validation | +0.6 | -0.4 | +0.0 | -0.9 | +0.0 | -1.0 | -0.2 | +0.7 |
| MRI | Test | -1.4 | -1.6 | -1.5 | -1.9 | -0.9 | -1.7 | -1.4 | -3.2 |

### 9b. Tốc độ và kích thước (cùng máy, batch 1, ảnh 512)

GPU: NVIDIA GeForce RTX 3050 Laptop GPU; mỗi lần đo: 200 ảnh Val, 20 ảnh khởi động, lấy trung vị; đo 3 lần (2026-11-13T20:34:38, 2026-11-13T20:46:36, 2026-11-13T20:47:04). Bảng ghi trung vị của 3 lần và khoảng [min–max]. Thời gian end-to-end = tiền xử lý + suy luận + NMS cho một ảnh (gọi `predict`).

| Ảnh / mô hình | Tham số | GFLOPs | Checkpoint (MB) | fp16: ms/ảnh [min–max] | FPS fp16 | fp32: ms/ảnh [min–max] | FPS fp32 | Chỉ suy luận fp32 (ms) |
|---|---|---|---|---|---|---|---|---|
| ct/yolo11n | 2,604,883 | 4.19 | 5.48 | 10.50 [8.89–10.80] | 95 | 9.33 [9.10–9.79] | 107 | 7.44 |
| ct/yolov5n | 2,635,619 | 4.95 | 5.51 | 8.45 [8.03–8.53] | 118 | 7.84 [7.15–8.02] | 128 | 6.04 |
| mri/yolo11n | 2,604,883 | 4.19 | 5.49 | 10.28 [9.70–10.44] | 97 | 9.46 [8.89–9.95] | 106 | 7.37 |
| mri/yolov5n | 2,635,619 | 4.95 | 5.51 | 9.01 [8.93–9.07] | 111 | 8.19 [7.89–8.42] | 122 | 6.00 |

GFLOPs tính ở ảnh 512. Nguồn: `outputs/runs/comparison/eval/bench_run*.json`.

## 10. Biểu đồ so sánh

**Hình 10.1.** Chỉ số chính trên Validation.

![so sánh val](figures/yolov5n_vs_yolo11n/so_sanh_val.png)

**Hình 10.2.** Chỉ số chính trên Test.

![so sánh test](figures/yolov5n_vs_yolo11n/so_sanh_test.png)

**Hình 10.3.** Thời gian end-to-end (trung vị các lần đo).

![tốc độ](figures/yolov5n_vs_yolo11n/toc_do.png)

**Hình 10.4.** mAP50-95 trên Val theo epoch (từ `results.csv` của từng lần train).

![đường huấn luyện](figures/yolov5n_vs_yolo11n/duong_huan_luyen.png)

## 11. Phân tích ưu điểm và hạn chế

### 11a. Khả năng phát hiện tổn thương

**CT: YOLOv5n tốt hơn YOLO11n ở mọi chỉ số, trên cả Validation và Test (bảng 9a).**
- Validation: mAP50 0,578 so với 0,551 (+2,7 điểm), mAP50-95 0,293 so với 0,282, F1 0,590 so với 0,566. Ở ngưỡng vận hành 0,12, YOLOv5n tìm đúng thêm 148 box (4.557 so với 4.409 trên 7.631 box GT) và báo nhầm ít hơn 160 box (4.729 so với 4.889). Validation có cả lát âm tính, nên đây là nơi đánh giá báo nhầm đáng tin hơn Test.
- Test: khoảng cách lớn hơn: mAP50 0,589 so với 0,523 (+6,6 điểm), mAP50-95 +3,2 điểm, F1 +4,7 điểm; ở ngưỡng 0,12 thêm 249 box đúng (4.676 so với 4.427 trên 7.776), ít hơn 640 box sai. Recall tổn thương nhỏ (< 32×32 px) 0,227 so với 0,195.
- Ngưỡng 0,12 được chọn trên Validation **cho YOLO11n**, tức là thuận lợi cho YOLO11n; YOLOv5n vẫn cao hơn ở ngưỡng này. Các chỉ số mAP không phụ thuộc ngưỡng cho cùng kết luận.

**MRI: hai mô hình gần như ngang nhau; YOLO11n nhỉnh hơn trên Test.**
- Validation: mAP50-95 bằng nhau (0,504), F1 bằng nhau (0,722), mAP50 0,749 (YOLO11n) so với 0,739.
- Test: YOLO11n cao hơn 1–2 điểm ở mọi chỉ số (mAP50 0,776 so với 0,757; F1 0,756 so với 0,741; ở ngưỡng 0,12 thêm 119 box đúng trên 6.965). Recall tổn thương nhỏ 0,623 so với 0,591 (+3,2 điểm).

**Cả hai mô hình chung một điểm yếu:** tổn thương nhỏ trên CT. Recall box nhỏ chỉ khoảng 0,20–0,27 với cả hai kiến trúc. Thay thế hệ YOLO không giải quyết được giới hạn này; nó đến từ dữ liệu (tổn thương nhỏ trên ảnh 512 px, ảnh CT đã lượng tử hóa `uint8`).

**Đường huấn luyện (Hình 10.4):** hai mô hình hội tụ gần như trùng nhau. CT đạt đỉnh mAP50-95 trên Val ở epoch 15 (YOLO11n) và 20 (YOLOv5n), sau đó giảm nhẹ (overfit). MRI đạt đỉnh ở epoch 24 và 26. Không mô hình nào cần nhiều epoch hơn hẳn mô hình kia.

### 11b. Tốc độ và tài nguyên phần cứng

- **Kích thước gần như bằng nhau:** 2,60 so với 2,64 triệu tham số, checkpoint 5,48 so với 5,51 MB. Cả hai chạy thoải mái trên GPU 4 GB của dự án.
- **YOLO11n ít phép tính hơn** (4,19 so với 4,95 GFLOPs ở ảnh 512, ít hơn khoảng 15 %).
- **Nhưng trên RTX 3050 Laptop với batch 1, YOLOv5n chạy nhanh hơn trong cả 3 lần đo.** fp32: 7,84 so với 9,33 ms/ảnh (CT), 8,19 so với 9,46 ms/ảnh (MRI), tức nhanh hơn khoảng 13–16 %. fp16 cho cùng thứ tự. Khoảng [min–max] của hai mô hình không chồng lên nhau.
- Cách giải thích hợp lý (chưa đo riêng): YOLO11n có khối chú ý C2PSA và tích chập depthwise. Các phép này ít FLOPs nhưng gồm nhiều thao tác nhỏ, tốn băng thông bộ nhớ. Với batch 1 trên GPU, thời gian bị chi phối bởi số thao tác hơn là số FLOPs. GFLOPs thấp hơn **không** đồng nghĩa với chạy nhanh hơn trên mọi phần cứng.
- Tính theo trung vị, fp16 không nhanh hơn fp32 ở batch 1 với cả hai mô hình. Mô hình quá nhỏ nên chi phí chuyển đổi và khởi chạy lấn át lợi ích.
- Về thực tế sử dụng, cả hai đều trên 95 ảnh/giây. Một volume CT khoảng 150 lát mất khoảng 1,2–1,3 s (YOLOv5n) hoặc 1,4–1,6 s (YOLO11n) cho phần phát hiện. Chênh lệch này không đáng kể so với thời gian đọc tệp và các bước khác của pipeline.
- Thời gian **huấn luyện** trên Colab (bảng mục 6) không dùng để so sánh: T4 miễn phí dùng chung, các phiên bị ngắt và resume, tốc độ đọc dữ liệu thay đổi.

### 11c. Hạn chế của thí nghiệm

- **Mỗi mô hình chỉ huấn luyện một lần (seed 42).** Chưa đo được dao động giữa các lần huấn luyện, nên chênh lệch 1–2 điểm (MRI) có thể nằm trong mức dao động đó. Chênh lệch trên CT cùng chiều ở cả Validation lẫn Test, nhưng vẫn chỉ từ một lần chạy.
- Chưa có khoảng tin cậy theo bệnh nhân. Các lát của cùng một BN tương quan với nhau, nên số lát (hàng nghìn) đánh giá quá cao số mẫu độc lập (42 BN CT, 104 BN MRI trên Test).
- Test chỉ gồm lát có tổn thương, nên precision trên Test không phản ánh báo nhầm trên lát bình thường hay BN nhóm chứng.
- Một số lần train bị ngắt do hết hạn mức GPU Colab và được resume từ `last.pt`: YOLOv5n-CT 2 phiên, YOLOv5n-MRI 3 phiên, YOLO11n-MRI 2 phiên. Resume khôi phục trọng số, optimizer và epoch, lịch học tiếp tục đúng như cũ. Thứ tự xáo trộn dữ liệu sau resume không trùng tuyệt đối với một lần chạy liền mạch.
- `yolov5nu` có đầu dự đoán anchor-free của Ultralytics hiện tại. Kết luận áp dụng cho "backbone/neck YOLOv5 trong cùng khung huấn luyện", không phải cho repo YOLOv5 gốc với đầu anchor-based.
- Tốc độ chỉ đo trên một GPU laptop, batch 1, PyTorch. Chưa đo CPU, TensorRT hay batch lớn.
- Thí nghiệm đo **phát hiện ở mức lát**. Chưa đo ảnh hưởng của việc đổi bộ phát hiện tới Tumor Zoom, Evidence Gate và phân loại mức bệnh nhân.

## 12. Kết luận: mô hình nào phù hợp hơn với bài toán

**Kết quả đo được không xác nhận rằng YOLO11n tốt hơn YOLOv5n trên dữ liệu của đề tài.**

| Câu hỏi | Kết quả đo được |
|---|---|
| Phát hiện tổn thương CT | **YOLOv5n tốt hơn** (Test: mAP50 +6,6 điểm, F1 +4,7; Val: mAP50 +2,7) |
| Phát hiện tổn thương MRI | **Ngang nhau trên Val, YOLO11n nhỉnh hơn trên Test** (mAP50 +1,9 điểm, F1 +1,5) |
| Tổn thương nhỏ | Cả hai đều yếu trên CT (recall 0,20–0,27); MRI YOLO11n nhỉnh hơn trên Test |
| Tốc độ (RTX 3050 Laptop, batch 1) | **YOLOv5n nhanh hơn khoảng 13–16 %**, dù nhiều FLOPs hơn |
| Tài nguyên (tham số, dung lượng, VRAM) | Ngang nhau (2,6 triệu tham số, 5,5 MB); YOLO11n ít FLOPs hơn 15 % |

- **Riêng cho bước phát hiện CT**, theo số đo của thí nghiệm này, YOLOv5n là lựa chọn phù hợp hơn: chính xác hơn và nhanh hơn trên phần cứng của dự án.
- **Với MRI**, hai mô hình tương đương; YOLO11n nhỉnh hơn một chút trên Test.
- **YOLO11n vẫn là mô hình chính của dự án.** Không phải vì nó thắng trong thí nghiệm này, mà vì:
  1. Lựa chọn được đưa ra **trước** thí nghiệm, dựa trên thông tin có lúc đó: thế hệ mới nhất của Ultralytics, kết quả công bố trên COCO tốt hơn, ít FLOPs hơn.
  2. Toàn bộ phần sau của pipeline đã được chọn trên Validation và **khóa theo đầu ra của YOLO11n**: ngưỡng 0,12, quy tắc nối vùng Tumor Zoom, Evidence Gate (độ tin cậy ≥ 0,4 ở ≥ 2 pha), dữ liệu ROI-YOLO dùng cho Hybrid. Test chính thức của pipeline cũng đã chạy một lần với cấu hình đó.
  3. Đổi bộ phát hiện đòi hỏi chọn lại các tham số trên trên Validation và cần một tập kiểm tra độc lập mới. Dùng lại Test hiện tại để quyết định việc thay mô hình là dùng Test để chọn mô hình, trái với nguyên tắc của dự án.
- **Hướng phát triển:** thay YOLOv5n cho nhánh CT là ứng viên rõ ràng. Cần (a) huấn luyện lại nhiều seed để xác nhận chênh lệch, (b) chọn lại ngưỡng, Tumor Zoom và Evidence Gate trên Validation, (c) đánh giá trên dữ liệu kiểm tra mới.

### Gợi ý trả lời hội đồng (dựa trên số liệu trên)

- **Tại sao em chọn YOLO11n thay vì YOLOv5?** Lúc xây dựng pipeline, em chọn YOLO11n vì là thế hệ mới nhất của Ultralytics, có kết quả công bố trên COCO tốt hơn và ít phép tính hơn YOLOv5n. Sau đó em làm thí nghiệm đối chứng để kiểm tra lại lựa chọn này: cùng dữ liệu, cùng siêu tham số, cùng cách đánh giá. Kết quả cho thấy trên dữ liệu của em, YOLO11n không vượt trội. Trên CT, YOLOv5n còn tốt hơn; trên MRI, hai mô hình tương đương.
- **YOLO11n cải thiện gì so với YOLOv5n?** Về kiến trúc, YOLO11n có khối C3k2, khối chú ý C2PSA và đầu phân lớp depthwise, nên ít hơn khoảng 15 % FLOPs với cùng số tham số. Trên dữ liệu của đề tài, cải thiện chỉ thấy ở MRI Test (+1,9 điểm mAP50, +3,2 điểm recall tổn thương nhỏ). Trên CT không có cải thiện. Trên GPU của em, YOLO11n cũng không chạy nhanh hơn.
- **Mô hình nào phát hiện tổn thương tốt hơn?** CT: YOLOv5n. MRI: tương đương, YOLO11n nhỉnh hơn trên Test.
- **Mô hình nào xử lý nhanh hơn?** YOLOv5n, nhanh hơn khoảng 13–16 % trên RTX 3050 Laptop ở batch 1. Cả hai đều trên 95 ảnh/giây.
- **Mô hình nào phù hợp hơn với tài nguyên phần cứng?** Cả hai đều 2,6 triệu tham số và 5,5 MB, chạy được trên GPU 4 GB. YOLO11n ít FLOPs hơn nhưng trên GPU này không nhanh hơn.
- **Tại sao em vẫn dùng YOLO11n làm mô hình chính?** Vì ngưỡng vận hành, Tumor Zoom, Evidence Gate và dữ liệu huấn luyện Hybrid đều đã được chọn và khóa theo YOLO11n, và Test chính thức đã chạy một lần với cấu hình đó. Đổi bộ phát hiện phải chọn lại toàn bộ trên Validation và cần tập kiểm tra mới. Em ghi nhận YOLOv5n là hướng cải thiện cho nhánh CT.

