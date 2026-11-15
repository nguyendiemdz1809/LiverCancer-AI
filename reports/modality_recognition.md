# Nhận dạng tự động ảnh CT / MRI (bước tiếp nhận của website)

_Sinh bởi `scripts/87_modality_classifier.py` lúc 2026-10-27 11:00. Mô hình riêng, **không** thuộc pipeline AI đã khóa; không dùng tập Test._

## Thứ tự quyết định

1. **DICOM:** thẻ `Modality` (CT / MR).
2. **NIfTI:** hình học header. CT: 512×512, voxel 1 mm đẳng hướng. MRI: lát cắt dày ≥ 2 mm và dày ≥ 1,5 lần độ phân giải trong lát.
3. **Ảnh 2D (PNG/JPG, ảnh chụp màn hình, DICOM dạng chụp màn hình):** CNN nhỏ. Chỉ tự quyết khi độ tin cậy ≥ ngưỡng chọn trên Val-A; nếu thấp hơn, website hỏi người dùng. **Không dùng tên tệp.**

## Kiểm tra quy tắc header NIfTI (mọi tệp Train + Val)

| Ảnh thật | Nhận là CT | Nhận là MRI | Không xác định |
|---|---|---|---|
| CT | 1197 | 0 | 0 |
| MRI | 0 | 3152 | 0 |

## Mô hình ảnh 2D

- **Dữ liệu huấn luyện:** 4506 ảnh từ bệnh nhân Train. Mỗi lát cắt có 3 biến thể và đi qua cùng bước tiền xử lý của website.
- **Chọn epoch:** epoch 13, chọn trên Val-A (nửa số bệnh nhân Validation, 474 ảnh).
- **Ngưỡng tự quyết:** **0.95**, là giá trị nhỏ nhất trên lưới mà mọi loại ảnh của Val-A đều đúng ≥ 99,5% trong các ca tự quyết.
- **Báo cáo trên Val-B:** 72 bệnh nhân Validation khác, không dùng để chọn gì.

| Loại ảnh (Val-B) | Số ảnh | Đúng (không ngưỡng) | Tự quyết (≥ ngưỡng) | Đúng trong ca tự quyết | Nhầm CT↔MRI khi tự quyết |
|---|---|---|---|---|---|
| Lát cắt xuất gốc (PNG) | 216 | 99.5% | 98.1% | 100.0% | 0 (CT→MRI 0, MRI→CT 0) |
| Ảnh xuất bị nén / đổi kích thước / độ sáng | 216 | 99.1% | 96.3% | 99.5% | 1 (CT→MRI 1, MRI→CT 0) |
| Ảnh chụp màn hình / chụp lại màn hình (tổng hợp) | 141 | 97.9% | 92.9% | 99.2% | 1 (CT→MRI 1, MRI→CT 0) |
| **Tất cả** | 573 | 99.0% | 96.2% | 99.6% | 2 |

Bị bước tiền xử lý từ chối trước khi tới mô hình (Val, không tách được lát cắt): {'ct|screenshot': 1, 'mri|screenshot': 6, 'mri|clean': 1, 'mri|degraded': 1}.

## YOLO đã khóa trên ảnh chụp màn hình (mô tả độ lệch miền, không chỉnh gì)

Cùng lát cắt Val có tổn thương, đưa vào YOLO theo 3 dạng. Tính tỷ lệ tìm thấy tổn thương chuẩn (IoU ≥ 0,3, conf 0,12):

| | Số lát | Ảnh gốc | Ảnh chụp màn hình (JPEG) | Ảnh chụp lại màn hình | Bị từ chối (chụp màn hình / chụp lại) |
|---|---|---|---|---|---|
| CT | 200 | 133/200 | 130/200 | 121/200 | 2 / 15 |
| MRI | 200 | 147/200 | 141/200 | 112/200 | 14 / 38 |

Ca bị từ chối được tính là không tìm thấy.

## Giới hạn

- **Dữ liệu huấn luyện và đánh giá:** chỉ gồm hai bộ dữ liệu (CT: PLC-CECT, MRI: LLD-MMRI). Mô hình có thể học đặc trưng riêng của từng bộ thay vì đặc trưng CT/MRI; **chưa kiểm chứng trên ảnh bệnh viện thật**.
- **Ảnh chụp màn hình:** do chương trình tạo từ lát cắt Val, không phải ảnh chụp thật; độ chính xác trên ảnh thật có thể thấp hơn.
- **Ảnh không phải CT/MRI** (X-quang, siêu âm, ảnh thường): không có trong dữ liệu. Bước tiền xử lý chặn ảnh màu hoặc ảnh không có lát cắt trên nền tối, nhưng không đảm bảo chặn hết.
- Khi độ tin cậy dưới ngưỡng, website hỏi người dùng và không tự đưa ảnh vào YOLO CT hay MRI.
