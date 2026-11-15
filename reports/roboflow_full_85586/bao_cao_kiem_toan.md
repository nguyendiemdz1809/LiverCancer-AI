# Kiểm toán 85.586 ảnh và chuẩn bị bản xuất Roboflow

*`scripts/85_build_roboflow_full.py`, 2026-10-30 11:50. Chi tiết máy đọc: `data/processed/roboflow_full_85586/audit.json`.*

## 1. Kết luận

- **Xác nhận đủ 85.586 ảnh hợp lệ**, đúng bằng hai bộ đã huấn luyện trên Google Colab: CT 49.500 + MRI 36.086. Mọi kiểm tra bên dưới đạt.
- Bản xuất giữ nguyên ảnh, nhãn, tên file và phân chia Train/Validation gốc; không thêm, bớt, sửa hay chia lại. Dữ liệu huấn luyện chính không bị ghi.
- **Chưa upload, chưa xóa gì trên Roboflow.**

## 2. Nguồn gốc (đối chiếu log Google Colab)

| | CT | MRI |
|---|---|---|
| Archive | `yolo_ct_full.zip` | `yolo_mri_pilot.zip` |
| Dataset ghi trong bundle Colab | `yolo_ct_full.zip` | `yolo_mri_pilot.zip` |
| Thời điểm ghi kết quả Colab | `2026-10-24T16:19:06` | `2026-10-21T02:15:59` |
| Số epoch đã chạy | `35` | `44` |
| Best epoch | `15` | `24` |
| SHA256 archive = `SHA256SUMS` = SHA256 ghi trong Colab | `0a7ed974856f5634…` PASS | `0088fa93b294887a…` PASS |

Bundle Colab: `outputs/runs/yolo_{ct,mri}/yolo11n_e50_img512_baseline/metrics_val.json`. Log Colab lưu ở local chỉ gồm `args.yaml`, `results.csv`, metrics; dòng "Scanning … images" của lúc train không được lưu, nên số ảnh được xác minh qua SHA256 của chính archive đã train và đếm trực tiếp trong archive. Lần đánh giá lại ở local (`outputs/runs/yolo_*/analyze_*_e0.log`) quét đúng CT Val 14.627 ảnh, MRI Val 7.491 ảnh, 0 ảnh hỏng.

## 3. Kiểm tra

| Kiểm tra | CT | MRI |
|---|---|---|
| A2. Số ảnh mỗi tập đúng kỳ vọng (CT 34.873 / 14.627; MRI 28.595 / 7.491) | PASS | PASS |
| A2. Mỗi ảnh có đúng một file nhãn | PASS | PASS |
| A2. Khớp 1–1 với `slices.csv` trong archive | PASS | PASS |
| A3. Mọi ảnh giải mã được, đúng kích thước | PASS | PASS |
| A4. Nhãn đúng định dạng (5 cột, class 0, box trong [0, 1]) | PASS | PASS |
| A4. Số box khớp metadata; ảnh âm tính có nhãn rỗng | PASS | PASS |
| A5. Mỗi BN chỉ ở một tập và đúng split đóng băng | PASS | PASS |
| A5. Không có BN Test | PASS | PASS |
| A6. ZIP mới: CRC đúng | PASS | PASS |
| A6. ZIP mới: đủ cặp ảnh–nhãn | PASS | PASS |
| A6. Các ZIP không trùng ảnh | PASS | PASS |
| A6. Gộp các ZIP = đủ bộ | PASS | PASS |
| A6. Từng ảnh/nhãn trùng byte (CRC32) với archive Colab | PASS | PASS |
| A7. Không trùng ảnh giữa CT và MRI (SHA256 chung: 0) | PASS | (chung) |

Ảnh trùng byte **trong cùng** bộ: CT 111 cặp, MRI 6 cặp. Mọi cặp thuộc cùng một BN và cùng một tập (không vắt qua Train/Validation), khác pha/chuỗi: dataset gốc có một số pha giống hệt nhau ở vài lát (ví dụ `P0069_C2_k047` = `P0069_C3_k047`). Giữ nguyên vì đây là đúng dữ liệu đã train. Roboflow có thể bỏ qua ảnh trùng khi upload, nên số ảnh hiển thị trên Roboflow có thể ít hơn tương ứng (CT tối thiểu 49.389).

## 4. Thống kê

Xem `reports/thong_ke_du_lieu_chinh_thuc.md` (bảng chính thức).

## 5. Bản xuất

| Bộ | Thư mục | ZIP | Ảnh | Dung lượng | Được upload? |
|---|---|---|---|---|---|
| CT Train | `data/processed/roboflow_full_85586/CT_49500/zips/` | 35 (`Roboflow_CT_train_partXXofYY.zip`) | 34.873 | 2.909,9 MB | Có (CC BY 4.0, ghi nguồn) |
| CT Validation | `data/processed/roboflow_full_85586/CT_49500/zips/` | 15 (`Roboflow_CT_valid_partXXofYY.zip`) | 14.627 | 1.250,8 MB | Có (CC BY 4.0, ghi nguồn) |
| MRI Train | `data/processed/roboflow_full_85586/MRI_36086_offline/zips/` | 29 (`Roboflow_MRI_train_partXXofYY.zip`) | 28.595 | 2.101,6 MB | **Không** – giữ offline |
| MRI Validation | `data/processed/roboflow_full_85586/MRI_36086_offline/zips/` | 8 (`Roboflow_MRI_valid_partXXofYY.zip`) | 7.491 | 573,5 MB | **Không** – giữ offline |
| **Tổng** | | 87 | 85.586 | 6.835,7 MB | |

- Mỗi ZIP ≤ 1.000 ảnh, chứa `data.yaml` + `train/` hoặc `valid/` (`images/`, `labels/`) – Roboflow nhận đúng tập Train/Valid.
- `data/processed/roboflow_full_85586/upload_manifest.csv`: một dòng mỗi ZIP (số ảnh, box, SHA256, cột `uploaded_to_roboflow` / `upload_date` / `roboflow_project` để đánh dấu, tránh upload trùng).
- `data/processed/roboflow_full_85586/manifest_images.csv`: một dòng mỗi ảnh (ZIP chứa nó, BN, pha/chuỗi, lát, số box, SHA256, CRC32).

## 6. Quyền phân phối

- **CT – PLC-CECT (Luo et al., Scientific Data 2025) - CC BY 4.0 (attribution required).** Được upload; ghi nguồn trong mô tả project Roboflow.
- **MRI – LLD-MMRI, CC BY-NC 4.0** kèm điều khoản *"not to further copy, publish, or distribute any part"* (`reports/audit_report.md`). **Không upload** lên Roboflow (kể cả project riêng tư) khi chưa có văn bản cho phép của tác giả.

## 7. Khả năng upload (giới hạn Roboflow)

Không truy cập được tài khoản Roboflow của bạn; số dưới đây theo bảng giá công khai đọc ngày 2026-10-30 ([roboflow.com/pricing](https://roboflow.com/pricing), [roboflow.com/credits](https://roboflow.com/credits)): gói **Free Tier = 10 credit/tháng**, project riêng tư, không có flex billing; upload **10.000 ảnh/credit**; lưu trữ **5.000 ảnh/credit/tháng** (tính theo tháng). Hết credit thì nhiều chức năng bị dừng tới khi credit tháng sau được cộng hoặc mua thêm.

| Phương án | Ảnh | Credit upload (1 lần) | Credit lưu trữ/tháng | Tháng đầu | So với Free 10 credit |
|---|---|---|---|---|---|
| Chỉ CT (Train + Valid) | 49.500 | 4,95 | 9,90 | 14,85 | vượt |
| Chỉ CT Train | 34.873 | 3,49 | 6,97 | 10,46 | vượt |
| CT + MRI (không được phép với MRI) | 85.586 | 8,56 | 17,12 | 25,68 | vượt |

- Với Free Tier, tháng đầu chỉ chứa được tối đa khoảng **33.333 ảnh** (upload + lưu trữ = 10 credit) và chưa tính credit project cũ đang dùng. **Không thể đưa đủ 49.500 ảnh CT lên gói miễn phí.**
- Đủ 49.500 ảnh CT cần khoảng 14,9 credit tháng đầu và 9,9 credit/tháng sau đó → cần gói trả phí hoặc credit mua thêm. **Tôi không mua credit.** Hãy kiểm tra số credit còn lại tại Roboflow → Settings → *Billing / Usage* trước khi upload.
- Project cũ vẫn tốn credit lưu trữ; **chưa xóa** project cũ cho tới khi bộ thay thế được upload và kiểm tra xong.
- Phương án vừa gói miễn phí: dùng các bộ minh họa đã có (`data/processed/roboflow/` 5.000 ảnh CT, `data/processed/roboflow_positive_20k/zips/CT/` 10.000 ảnh CT), còn con số 85.586 trình bày bằng bảng thống kê chính thức.

## 8. Thứ tự đề xuất (chờ bạn xác nhận)

1. Xem số credit còn lại trong tài khoản Roboflow.
2. Nếu đủ: tạo project Object Detection mới (lớp `lesion`), upload lần lượt `Roboflow_CT_train_part01of35.zip` … rồi `Roboflow_CT_valid_part01of15.zip` …, đánh dấu từng dòng trong `upload_manifest.csv`.
3. Kiểm tra trên Roboflow: tổng ảnh = 49.500 (Train 34.873, Valid 14.627), hoặc ít hơn tối đa 111 nếu Roboflow bỏ ảnh trùng; ảnh nhãn rỗng hiện là *null* (hoặc *unannotated* → *Mark Null*).
4. Chỉ sau khi xác minh xong mới cân nhắc xóa project cũ.
5. MRI: giữ offline.
