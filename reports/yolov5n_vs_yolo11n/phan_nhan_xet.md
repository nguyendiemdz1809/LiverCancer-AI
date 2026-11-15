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
