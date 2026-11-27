## 1. Mục tiêu và lý do chọn Faster R-CNN

**Mục tiêu.** Kiểm tra lựa chọn YOLO11n cho bước phát hiện tổn thương CT bằng cách so sánh với một bộ phát hiện **khác họ YOLO**, trong cùng điều kiện dữ liệu và cùng cách tính chỉ số. Đây là so sánh giữa hai **cách tiếp cận** phát hiện đối tượng khác nhau: một giai đoạn (YOLO11n) và hai giai đoạn (Faster R-CNN).

**Lý do chọn Faster R-CNN ResNet-50 FPN:**
- Đây là đại diện kinh điển của nhóm **hai giai đoạn** (đề xuất vùng rồi phân loại từng vùng). YOLO11n đại diện cho nhóm **một giai đoạn**. Hai nhóm này là hai hướng chính của bài toán phát hiện đối tượng.
- Faster R-CNN là mốc so sánh được dùng nhiều trong các nghiên cứu phát hiện tổn thương trên ảnh CT.
- Mô hình **cỡ trung bình** (khoảng 41 triệu tham số), lớn hơn YOLO11n khoảng 16 lần, nên thí nghiệm trả lời được câu hỏi: một mô hình lớn hơn nhiều, thuộc kiến trúc khác, có phát hiện tốt hơn trên dữ liệu của đề tài không.
- Có sẵn bản cài đặt chuẩn và trọng số COCO trong `torchvision`, nên không phải tự cài đặt kiến trúc.

**Nguyên tắc.** YOLO11n vẫn là mô hình chính, không bị thay thế dù kết quả thế nào; ngưỡng, Tumor Zoom, Evidence Gate và Test chính thức của pipeline đã khóa theo YOLO11n. Mọi lựa chọn của Faster R-CNN (checkpoint, ngưỡng) được làm trên Validation. Test được đánh giá **một lần** (khóa `outputs/runs/comparison/frcnn_ct/test_done.json`, 27/11/2026 11:35). Chỉ báo cáo số đo được.
