# Lát cắt trùng hoàn toàn trong volume Validation (tiếp nhận 3D của website)

Tạo bởi `scripts/86_slice_duplicates.py`; số liệu: `outputs/slice_duplicates/` (summary.json, volumes.csv, comparison.csv).
Chỉ dùng Validation (không dùng Test). Lát cắt được tạo đúng như AI nhìn thấy (`load_volume_uint8` → `slice_image`).

**Định nghĩa:** hai lát là trùng khi giống nhau **từng byte** (hash BLAKE2b). Các lát liên tiếp chỉ *giống nhau* **không** bị gộp (tổn thương trải trên nhiều lát; chưa có ngưỡng tương đồng nào được kiểm chứng).

| | CT | MRI |
|---|---|---|
| Số volume | 211 | 624 |
| Số lát | 27560 | 30999 |
| Volume có lát trùng | 0 | 0 |
| Lát trùng (bản sao thừa) | 0 | 0 |
| … trong đó không phải lát đồng màu | 0 | 0 |
| Volume không đọc được | 0 | 0 |
| Volume đã so sánh có/không lọc trùng | 0 | 0 |
| … phát hiện YOLO từng lát trùng khớp | 0 | 0 |
| … vùng 3D (track, box Zoom) trùng khớp | 0 | 0 |
| Thời gian YOLO không lọc (s, các volume so sánh) | 0 | 0 |
| Thời gian YOLO có lọc (s) | 0 | 0 |

Cách dùng trên website: YOLO chạy một lần cho mỗi nội dung lát rồi gán lại kết quả cho mọi bản sao; volume không bị cắt bớt lát nào, chỉ số lát giữ nguyên. Volume không có lát trùng chạy y hệt pipeline đã khóa.

## Kiểm tra cơ chế gán lại kết quả

Vì Validation không có lát trùng, cơ chế được kiểm tra trên 3 volume CT và 3 volume MRI Validation (volume đầu tiên có vùng phát hiện), sau khi **chèn thêm bản sao** của 6 lát quanh lát có độ tin cậy cao nhất. Đây là dữ liệu kiểm thử dựng lại, không phải ca chụp thật và không dùng làm kết quả nghiên cứu.

| Loại | Volume | Số lát | Bản sao chèn | Lát trùng tìm thấy | YOLO chạy | Box từng lát trùng tuyệt đối | Lệch box tối đa (px) | Lệch độ tin cậy tối đa | Lát khác số box | Vùng 3D trùng khớp |
|---|---|---|---|---|---|---|---|---|---|---|
| CT | P0011_ct_C1.nii.gz | 95 | 6 | 6 | 89 | không | 0.125 | 0.001465 | 0 | có |
| CT | P0011_ct_C2.nii.gz | 93 | 6 | 6 | 87 | không | 0.375 | 0.001953 | 0 | có |
| CT | P0011_ct_C3.nii.gz | 95 | 6 | 6 | 89 | không | 0.125 | 0.000977 | 0 | có |
| MRI | MR-400851_6_C+A_0000.nii.gz | 78 | 6 | 6 | 72 | có | 0.0 | 0.0 | 0 | có |
| MRI | MR-400851_6_C+Delay_0000.nii.gz | 78 | 6 | 6 | 72 | có | 0.0 | 0.0 | 0 | có |
| MRI | MR-400851_6_C+V_0000.nii.gz | 78 | 6 | 6 | 72 | có | 0.0 | 0.0 | 0 | có |

**Diễn giải:** bỏ lát trùng làm thay đổi cách ghép batch khi chạy YOLO fp16 trên GPU, nên tọa độ box có thể lệch ở mức làm tròn fp16 (cột *Lệch box tối đa*); đây cũng là mức lệch xảy ra khi đổi kích thước batch, không phải lỗi. Số box và các vùng 3D (track, lát tốt nhất, box Tumor Zoom) phải trùng khớp. Với dữ liệu thật của Validation, không có lát trùng nên website chạy đúng y hệt pipeline đã khóa. Tắt tính năng: `LCAI_DEDUP_EXACT=0`.

