# So sánh pipeline trước và sau khi bật Evidence Gate (Validation)

_Tạo bởi `scripts/94_gate_comparison.py` lúc 2026-10-26 19:32. Chỉ dùng Validation; Test không được dùng. Checkpoint YOLO và Hybrid giữ nguyên, không huấn luyện lại. Ngưỡng cổng chọn trên Val ở bước phân tích hạn chế (`reports/pipeline_validation/han_che/bao_cao_han_che.md`), không chỉnh lại sau lần chạy này._

## 1. Cấu hình được kiểm tra

- **Nhiều pha / chuỗi:** chỉ phân loại khi có vùng với độ tin cậy ≥ 0.4 ở ít nhất 2 pha (CT) / chuỗi (MRI).
- **Một file:** chỉ phân loại khi độ tin cậy YOLO cao nhất ≥ 0.35.
- Không đạt → trạng thái *Chưa đủ bằng chứng để phân loại*: không có nhãn, **không** kết luận khỏe mạnh.
- Cổng chỉ dùng độ tin cậy và vị trí box YOLO (không dùng GT, không dùng nhãn hay xác suất của bộ phân loại).
- MRI: áp **nguyên quy tắc của CT** để đánh giá. MRI Val không có nhóm chứng nên không chọn được ngưỡng riêng cho MRI.

Cách đếm (mỗi BN thuộc đúng một nhóm): **Đúng** / **Sai** (có nhãn) · **Từ chối phân loại** (cổng không cho nhãn) · **Bỏ sót** (YOLO không có vùng nào). Ca bị từ chối **không** được tính là đúng: accuracy trên toàn bộ BN = Đúng / tổng số BN.

![so sánh](figures/so_sanh_evidence_gate.png)

## 2. Kết quả theo nhóm

### CT – 12 bệnh nhân nhóm chứng CN

CN không có lớp đúng trong bộ phân loại 3 lớp, nên không có ô *Đúng*. Hành vi mong muốn là không gán nhãn ung thư.

| | Bị gán nhãn ung thư (cảnh báo sai) | Từ chối phân loại | Không có vùng YOLO | Không bị gán nhãn ung thư |
|---|---|---|---|---|
| Tắt cổng | 11/12 (91.7%) (KTC 95% 65–99%) | 0 | 1 | 1/12 (8.3%) |
| Bật cổng | 2/12 (16.7%) (KTC 95% 5–45%) | 9 | 1 | 10/12 (83.3%) |

CN vẫn bị gán nhãn khi bật cổng: P0281 (HCC), P0288 (ICC).

### CT – 42 bệnh nhân ung thư (HCC / ICC / CHCC)

| | Đúng | Sai | Từ chối phân loại | Bỏ sót | Accuracy trên toàn bộ BN | Accuracy trên ca có nhãn | Tỷ lệ có nhãn |
|---|---|---|---|---|---|---|---|
| Tắt cổng | 38 | 4 | 0 | 0 | 38/42 (90.5%) | 38/42 (90.5%) | 42/42 (100.0%) |
| Bật cổng | 36 | 4 | 2 | 0 | 36/42 (85.7%) | 36/40 (90.0%) | 40/42 (95.2%) |

BN ung thư bị từ chối khi bật cổng: P0132 (ICC, trước đó đúng, conf max 0.42, 1 pha đạt ≥ 0.4), P0133 (ICC, trước đó đúng, conf max 0.34, 0 pha đạt ≥ 0.4).
BN ung thư mà không vùng YOLO nào trùng tổn thương thật (nhãn nếu có chỉ đến từ vùng giả): 0/42.

### MRI – 78 bệnh nhân (7 lớp, chưa đạt chất lượng)

| | Đúng | Sai | Từ chối phân loại | Bỏ sót | Accuracy trên toàn bộ BN | Accuracy trên ca có nhãn | Tỷ lệ có nhãn |
|---|---|---|---|---|---|---|---|
| Tắt cổng | 37 | 41 | 0 | 0 | 37/78 (47.4%) | 37/78 (47.4%) | 78/78 (100.0%) |
| Bật cổng | 37 | 41 | 0 | 0 | 37/78 (47.4%) | 37/78 (47.4%) | 78/78 (100.0%) |

MRI không có BN nhóm chứng trên Val, nên cổng **không thể** chứng minh giảm cảnh báo sai trên MRI; chỉ đo được cái giá (số BN mất nhãn).

MRI bị từ chối: 0 BN, trong đó trước đó đúng 0, sai 0. Mọi BN MRI đều có vùng độ tin cậy ≥ 0.4 ở ít nhất 3 chuỗi (độ tin cậy cao nhất thấp nhất 0.48), nên quy tắc CT không bao giờ kích hoạt trên MRI Val.
BN MRI mà không vùng YOLO nào trùng tổn thương thật (nhãn chỉ đến từ vùng giả, được tính là **Sai**): 2/78 (MR140232, MR142442).

### CT – toàn bộ 54 bệnh nhân Validation

| | Ung thư phân loại đúng + CN không bị gán nhãn | Nhãn sai (ung thư sai lớp + CN bị gán nhãn ung thư) | BN ung thư không có nhãn |
|---|---|---|---|
| Tắt cổng | 39/54 (72.2%) | 15/54 (27.8%) | 0/54 (0.0%) |
| Bật cổng | 46/54 (85.2%) | 6/54 (11.1%) | 2/54 (3.7%) |

Đây **không** phải accuracy phân loại 4 lớp (hệ thống không có lớp CN): CN được tính là "hành vi đúng" khi không bị gán nhãn ung thư.

## 3. Kiểm tra tính nhất quán

- CT (54 BN): kết quả chạy thật khớp với phát lại quy tắc trên vùng YOLO của lần chạy không cổng: có; nhãn của ca được giữ lại không đổi: có.
- MRI (78 BN): kết quả chạy thật khớp với phát lại quy tắc trên vùng YOLO của lần chạy không cổng: có; nhãn của ca được giữ lại không đổi: có.

## 4. Đánh giá

- **Cảnh báo ung thư sai ở CN:** 11/12 → 2/12 (KTC 95% 65–99% → 5–45%). Cổng giảm rõ, nhưng mẫu chỉ có 12 BN và ngưỡng được chọn trên chính 12 BN này, nên con số lạc quan; ước lượng leave-one-out trước đó là 9/12 bị từ chối.
- **BN ung thư mất nhãn:** 0/42 → 2/42 (4.8%; KTC 95% 1–16%). 2/2 ca bị từ chối trước đó được phân loại đúng; không có BN ung thư nào bị YOLO bỏ sót hoàn toàn. Tiêu chí đặt trước (≤ 5% BN ung thư mất nhãn): **đạt**.
- Số ca phân loại sai lớp ở BN ung thư không đổi (4 → 4): cổng chỉ xét bằng chứng phát hiện, nên không sửa được lỗi phân loại.
- **MRI:** quy tắc CT làm mất nhãn 0/78 BN (không kích hoạt lần nào), và không có nhóm chứng để chứng minh lợi ích. Accuracy trên toàn bộ BN 37/78 (47.4%) → 37/78 (47.4%).

## 5. Cấu hình đề xuất (dựa trên Validation)

- **CT: bật Evidence Gate** (≥ 0.4 ở ≥ 2 pha; một file: ≥ 0.35), `configs/cls.yaml → inference.ct.evidence_gate.enabled: true`.
- **MRI: không bật cổng** (`enabled: false`): không có nhóm chứng để chọn hoặc kiểm chứng ngưỡng. Nhãn MRI tiếp tục kèm cảnh báo chưa đáng tin cậy; đề xuất cho website: MRI chỉ hiển thị phát hiện + Tumor Zoom, phân loại MRI chỉ là thông tin tham khảo.
- YOLO (conf 0.12), Tumor Zoom, Hybrid CT và cách gộp mức BN giữ nguyên.
- Mọi con số trên là của Validation, dùng để chọn cấu hình. Kết quả cuối cùng phải đo **một lần** trên Test với cấu hình đã chốt này.

## 6. Hạn chế

- 12 BN CN là mẫu rất nhỏ; khoảng tin cậy rộng. Ngưỡng được chọn trên cùng tập Val nên kết quả trên Test có thể kém hơn.
- *Từ chối phân loại* không phải là kết luận âm tính: BN vẫn cần bác sĩ đọc ảnh gốc.
- CN là nhóm "không có bệnh gan mật" theo nguồn dữ liệu, không phải "khỏe mạnh hoàn toàn".

Tệp: `reports/pipeline_validation/so_sanh_evidence_gate.csv` (từng BN), `so_sanh_evidence_gate.json`, `reports/pipeline_validation_gate/` (báo cáo pipeline đầy đủ khi bật cổng).
