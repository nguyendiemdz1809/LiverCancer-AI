# So sánh Faster R-CNN ResNet-50 FPN và YOLO11n trên CT (thí nghiệm đối chứng)

_Sinh bởi `scripts/79_frcnn_comparison.py --stage report` lúc 2026-11-27 18:16. Chỉ số liệu đo được. YOLO11n vẫn là mô hình chính của đồ án, không bị thay thế._

## 1. Mục tiêu và lý do chọn Faster R-CNN

**Mục tiêu.** Kiểm tra lựa chọn YOLO11n cho bước phát hiện tổn thương CT bằng cách so sánh với một bộ phát hiện **khác họ YOLO**, trong cùng điều kiện dữ liệu và cùng cách tính chỉ số. Đây là so sánh giữa hai **cách tiếp cận** phát hiện đối tượng khác nhau: một giai đoạn (YOLO11n) và hai giai đoạn (Faster R-CNN).

**Lý do chọn Faster R-CNN ResNet-50 FPN:**
- Đây là đại diện kinh điển của nhóm **hai giai đoạn** (đề xuất vùng rồi phân loại từng vùng). YOLO11n đại diện cho nhóm **một giai đoạn**. Hai nhóm này là hai hướng chính của bài toán phát hiện đối tượng.
- Faster R-CNN là mốc so sánh được dùng nhiều trong các nghiên cứu phát hiện tổn thương trên ảnh CT.
- Mô hình **cỡ trung bình** (khoảng 41 triệu tham số), lớn hơn YOLO11n khoảng 16 lần, nên thí nghiệm trả lời được câu hỏi: một mô hình lớn hơn nhiều, thuộc kiến trúc khác, có phát hiện tốt hơn trên dữ liệu của đề tài không.
- Có sẵn bản cài đặt chuẩn và trọng số COCO trong `torchvision`, nên không phải tự cài đặt kiến trúc.

**Nguyên tắc.** YOLO11n vẫn là mô hình chính, không bị thay thế dù kết quả thế nào; ngưỡng, Tumor Zoom, Evidence Gate và Test chính thức của pipeline đã khóa theo YOLO11n. Mọi lựa chọn của Faster R-CNN (checkpoint, ngưỡng) được làm trên Validation. Test được đánh giá **một lần** (khóa `outputs/runs/comparison/frcnn_ct/test_done.json`, 27/11/2026 11:35). Chỉ báo cáo số đo được.

## 2. Hai mô hình được so sánh

| | YOLO11n | Faster R-CNN ResNet-50 FPN |
|---|---|---|
| Họ mô hình | YOLO (Ultralytics, 2024) | R-CNN (Ren và cộng sự, 2015; FPN: Lin và cộng sự, 2017) |
| Số giai đoạn | Một giai đoạn: dự đoán hộp trực tiếp trên bản đồ đặc trưng | Hai giai đoạn: mạng đề xuất vùng (RPN), rồi phân loại và tinh chỉnh từng vùng (RoI head) |
| Backbone / neck | C3k2, SPPF, C2PSA (chú ý theo vị trí) / PANet | ResNet-50 / Feature Pyramid Network |
| Đầu dự đoán | Anchor-free, tách nhánh, DFL | RPN dùng anchor; RoIAlign 7×7 + 2 lớp kết nối đầy đủ |
| Hậu xử lý | NMS IoU 0,7 | NMS IoU 0,5 (mặc định torchvision) |
| Khởi tạo | COCO (`yolo11n.pt`) | COCO (torchvision `COCO_V1`) |
| mAP50-95 công bố trên COCO | 39,5 | 37,0 |
| Tham số (đo trên checkpoint 1 lớp) | 2.604.883 | 41.299.161 |
| GFLOPs ở ảnh 512 (cùng bộ đếm `torch` FlopCounterMode) | 4,11 | 126,12 |
| Dung lượng `best.pt` (MB) | 5,48 | 165,71 |

Số trên COCO lấy từ trang công bố của Ultralytics và torchvision (ảnh tự nhiên, 80 lớp), không suy ra được cho CT một lớp tổn thương.

## 3. Dữ liệu và cách đánh giá

- **Cùng dữ liệu với YOLO11n-CT:** `yolo_ct_full.zip` (Train 34.873 / Val 14.627 lát 512×512, gồm cả lát không có tổn thương; chia theo bệnh nhân). Test = 7.163 lát có tổn thương của 42 BN ung thư Test. Test không có lát âm tính và nhóm chứng, nên precision trên Test không phản ánh báo nhầm trên lát bình thường.
- **Cùng thuật toán tính chỉ số:** mAP của Faster R-CNN tính bằng mã viết lại đúng thuật toán của Ultralytics (ghép cặp tham lam theo độ tin cậy giảm dần tại 10 ngưỡng IoU 0,50:0,95, rồi `ap_per_class`). P, R, F1 là điểm F1 lớn nhất, như Ultralytics báo cáo.
- **Kiểm chứng bộ tính:** chạy lại YOLO11n trên Val bằng mã này và so với `val` của Ultralytics: mAP50 lệch -0,0015, mAP50-95 lệch 0,0012 (do fp16/fp32); TP / FP / FN ở ngưỡng 0,12 trùng khớp (4.409 / 4.889 / 3.222); quy tắc chọn ngưỡng của dự án (F2 lớn nhất trên Val) áp lên YOLO11n cho lại đúng 0,12. Nguồn: `outputs/runs/comparison/frcnn_ct/eval/evaluator_check.json`.
- **Ngưỡng vận hành:** độ tin cậy của hai mô hình tính khác nhau (YOLO: sigmoid; Faster R-CNN: softmax 2 lớp), nên ngưỡng 0,12 của YOLO11n không tương đương với Faster R-CNN. Báo cáo cả hai: ở 0,12 và ở ngưỡng riêng của Faster R-CNN chọn **trên Val** bằng cùng quy tắc F2 (= 0,28). Với YOLO11n, ngưỡng riêng chính là 0,12.
- Ghép cặp ở ngưỡng vận hành: IoU ≥ 0,5, một-một, theo độ tin cậy giảm dần. Box nhỏ: diện tích < 32×32 px.

## 4. Cấu hình huấn luyện

| | YOLO11n | Faster R-CNN |
|---|---|---|
| Nơi huấn luyện | Colab T4 | Máy local, RTX 3050 Laptop 4 GB |
| Epoch tối đa / dừng sớm | 50 / patience 20 | 12 / patience 5 |
| Batch, optimizer | 32, SGD lr0 0,01, cosine | 4, SGD lr 0,005 (= 0,01 × 4/8), momentum 0,9, cosine, warm-up 2.000 vòng |
| Augmentation | mosaic, lật ngang, tịnh tiến, co giãn, độ sáng | lật ngang 0,5, độ sáng ±30 % |
| Chọn checkpoint | mAP50-95 lớn nhất trên Val | mAP50-95 lớn nhất trên Val (cùng tiêu chí) |
| Epoch đã chạy / epoch tốt nhất | 35 / 15 | 8 / 3 |
| Thời gian huấn luyện (không gồm đánh giá Val) | – | 7,46 giờ |
| Seed | 42 | 42 |

Faster R-CNN dùng batch 4 vì batch 8 vượt 4 GB VRAM (5,0 GB, tràn sang bộ nhớ chia sẻ, chậm một nửa); tốc độ học giảm theo quy tắc tỉ lệ tuyến tính. Mosaic là kỹ thuật riêng của YOLO; công thức chuẩn của torchvision cho Faster R-CNN chỉ dùng lật ngang.

## 5. Kết quả trên Validation (14.627 lát, 7.631 box)

| Mô hình | P | R | F1 | mAP50 | mAP50-95 | Ngưỡng | P | R | F1 | TP / FP / FN | Recall box nhỏ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11n | 0,688 | 0,480 | 0,566 | 0,551 | 0,282 | 0,12 | 0,474 | 0,578 | 0,521 | 4.409 / 4.889 / 3.222 | 0,263 |
| Faster R-CNN | 0,652 | 0,431 | 0,519 | 0,476 | 0,221 | 0,12 | 0,286 | 0,573 | 0,382 | 4.374 / 10.896 / 3.257 | 0,300 |
| Faster R-CNN | | | | | | 0,28 (riêng) | 0,425 | 0,523 | 0,469 | 3.988 / 5.387 / 3.643 | 0,256 |

## 6. Kết quả trên Test (7.163 lát có tổn thương)

| Mô hình | P | R | F1 | mAP50 | mAP50-95 | Ngưỡng | P | R | F1 | TP / FP / FN | Recall box nhỏ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11n | 0,598 | 0,515 | 0,553 | 0,523 | 0,261 | 0,12 | 0,498 | 0,569 | 0,531 | 4.427 / 4.466 / 3.349 | 0,195 |
| Faster R-CNN | 0,598 | 0,440 | 0,507 | 0,456 | 0,205 | 0,12 | 0,347 | 0,549 | 0,425 | 4.268 / 8.031 / 3.508 | 0,235 |
| Faster R-CNN | | | | | | 0,28 (riêng) | 0,489 | 0,487 | 0,488 | 3.786 / 3.963 / 3.990 | 0,182 |

Năm cột chỉ số đầu không phụ thuộc ngưỡng (P, R, F1 ở điểm F1 lớn nhất); các cột sau tính tại ngưỡng vận hành. Nguồn: `outputs/runs/comparison/frcnn_ct/eval/ct_{val,test}.json` (Faster R-CNN), `outputs/runs/comparison/frcnn_ct/yolo11n_ref/ct_{val,test}.json` (YOLO11n, Ultralytics `val`).

### Chênh lệch Faster R-CNN − YOLO11n (điểm phần trăm; dương = Faster R-CNN cao hơn)

| Tập | P | R | F1 | mAP50 | mAP50-95 | Recall @ngưỡng riêng | F1 @ngưỡng riêng | Recall box nhỏ @ngưỡng riêng |
|---|---|---|---|---|---|---|---|---|
| Validation | -3,6 | -5,0 | -4,7 | -7,5 | -6,2 | -5,5 | -5,2 | -0,7 |
| Test | +0,1 | -7,6 | -4,7 | -6,6 | -5,6 | -8,2 | -4,3 | -1,3 |

## 7. Tốc độ và kích thước (cùng máy, batch 1, ảnh 512)

GPU: NVIDIA GeForce RTX 3050 Laptop GPU; 200 ảnh Val, 20 ảnh khởi động, lấy trung vị; 3 lần đo. Thời gian = tiền xử lý + suy luận + hậu xử lý (NMS) cho một ảnh.

| Mô hình | fp16: ms/ảnh [min–max] | Ảnh/giây fp16 | fp32: ms/ảnh [min–max] | Ảnh/giây fp32 |
|---|---|---|---|---|
| YOLO11n | 10,12 [9,73–10,36] | 99 | 8,81 [8,73–9,03] | 114 |
| Faster R-CNN | 50,57 [49,71–53,28] | 20 | 61,34 [60,58–62,12] | 16 |

Nguồn: `outputs/runs/comparison/frcnn_ct/eval/bench_run*.json`.

## 8. Biểu đồ

**Hình 8.1.** Chỉ số chính (ngưỡng vận hành riêng của từng mô hình, chọn trên Val).

![so sánh](figures/frcnn_vs_yolo11n/so_sanh.png)

**Hình 8.2.** Đường huấn luyện (YOLO11n: `results.csv` của run đã khóa; Faster R-CNN: `results.csv` của run này). Một epoch = một lượt qua 34.873 ảnh Train cho cả hai mô hình.

![đường huấn luyện](figures/frcnn_vs_yolo11n/duong_huan_luyen.png)

## 9. Nhận xét

### 9a. Khả năng phát hiện tổn thương

**YOLO11n phát hiện tốt hơn Faster R-CNN trên CT, ở cả Validation và Test, theo mọi chỉ số không phụ thuộc ngưỡng trừ precision trên Test (ngang nhau).**
- Validation: mAP50 0,551 so với 0,476 (YOLO11n cao hơn 7,5 điểm), mAP50-95 0,282 so với 0,221 (+6,2 điểm), F1 0,566 so với 0,519, recall 0,480 so với 0,431.
- Test: mAP50 0,523 so với 0,456 (+6,6 điểm), mAP50-95 0,261 so với 0,205 (+5,6 điểm), F1 0,553 so với 0,507, recall 0,515 so với 0,440. Precision ngang nhau (0,598).
- Kết luận giống nhau trên hai tập độc lập (Val chọn checkpoint và ngưỡng, Test chỉ chạy một lần).

**Ở ngưỡng vận hành** (mỗi mô hình dùng ngưỡng riêng chọn trên Val bằng cùng quy tắc F2: YOLO11n 0,12, Faster R-CNN 0,28):
- Validation: YOLO11n tìm đúng 4.409 box, báo nhầm 4.889; Faster R-CNN tìm đúng 3.988, báo nhầm 5.387. Tức YOLO11n tìm đúng nhiều hơn 421 box và báo nhầm ít hơn 498 box.
- Test: YOLO11n tìm đúng 4.427 box (recall 0,569), Faster R-CNN 3.786 (recall 0,487).
- Nếu ép Faster R-CNN dùng ngưỡng 0,12 như YOLO11n, recall của nó tăng lên gần YOLO11n (Val 0,573 so với 0,578) nhưng số box báo nhầm tăng gấp đôi (Val 10.896 so với 4.889; Test 8.031 so với 4.466).

**Tổn thương nhỏ (< 32×32 px) vẫn là điểm yếu chung.** Ở ngưỡng riêng, recall box nhỏ của Faster R-CNN là 0,256 (Val) và 0,182 (Test), của YOLO11n là 0,263 và 0,195. Ở ngưỡng thấp 0,12, Faster R-CNN tìm được nhiều box nhỏ hơn (Val 0,300, Test 0,235) nhưng đổi lại báo nhầm gấp đôi. Mô hình hai giai đoạn lớn hơn **không** giải quyết được giới hạn tổn thương nhỏ; giới hạn này đến từ dữ liệu (tổn thương nhỏ trên ảnh 512 px, ảnh CT đã lượng tử hóa `uint8`).

**Đường huấn luyện (Hình 8.2):** Faster R-CNN đạt mAP50-95 trên Val tốt nhất ở epoch 3 (0,220), sau đó dao động trong khoảng 0,18–0,21 và dừng sớm ở epoch 8. YOLO11n đạt đỉnh 0,282 ở epoch 15. Mô hình lớn hơn hội tụ sớm hơn nhưng ở mức thấp hơn, có dấu hiệu quá khớp với tập Train (loss train giảm đều từ 0,085 xuống 0,033 trong khi mAP trên Val không tăng).

### 9b. Tốc độ và tài nguyên

- **Kích thước:** Faster R-CNN có 41,3 triệu tham số (gấp khoảng 16 lần YOLO11n: 2,6 triệu), checkpoint 166 MB (gấp khoảng 30 lần: 5,5 MB).
- **Khối lượng tính toán:** 126 GFLOPs so với 4,1 GFLOPs ở ảnh 512, gấp khoảng 31 lần (đo bằng cùng một bộ đếm cho cả hai mô hình).
- **Tốc độ trên RTX 3050 Laptop, batch 1:** YOLO11n khoảng 9 ms/ảnh (trên 100 ảnh/giây), Faster R-CNN khoảng 61 ms/ảnh fp32 và 51 ms/ảnh fp16 (khoảng 16–20 ảnh/giây), tức **chậm hơn khoảng 6–7 lần**. Với một volume CT khoảng 150 lát, phần phát hiện mất khoảng 1,4 giây với YOLO11n và khoảng 8–9 giây với Faster R-CNN.
- **Huấn luyện:** trên GPU 4 GB, Faster R-CNN chỉ chạy được batch 4 (batch 8 vượt VRAM) và mất 7,5 giờ cho 8 epoch, chưa tính thời gian đánh giá Val (khoảng 30 phút mỗi epoch).

## 10. Hạn chế của thí nghiệm

- **Mỗi mô hình chỉ huấn luyện một lần (seed 42).** Chưa đo dao động giữa các lần huấn luyện. Chênh lệch mAP50 6,6–7,5 điểm, cùng chiều trên cả Val và Test, lớn hơn nhiều so với mức dao động 1–2 điểm thường gặp, nhưng vẫn chỉ từ một lần chạy.
- **Cấu hình huấn luyện không giống hệt nhau** vì hai kiến trúc có công thức huấn luyện chuẩn khác nhau: Faster R-CNN dùng batch 4 (giới hạn GPU 4 GB), không dùng mosaic, tối đa 12 epoch (công thức chuẩn "1x" của Faster R-CNN), còn YOLO11n dùng batch 32, mosaic, 50 epoch trên Colab T4.
- **Dừng sớm cắt ngắn lịch học của Faster R-CNN.** Với patience 5, Faster R-CNN dừng ở epoch 8/12, trước 4 epoch cuối của lịch cosine (giai đoạn tốc độ học nhỏ nhất, thường cho thêm vài điểm mAP); YOLO11n được chờ 20 epoch. mAP50-95 trên Val của Faster R-CNN đang tăng lại ở epoch 6–7 (0,206; 0,214) trước khi giảm ở epoch 8 (0,201), nên chưa loại trừ được khả năng chạy đủ 12 epoch sẽ thu hẹp khoảng cách. Để vượt YOLO11n trên Val, mAP50-95 phải tăng thêm khoảng 6 điểm so với mức tốt nhất đã đạt (0,220 so với 0,282). Chưa thử tinh chỉnh siêu tham số riêng cho Faster R-CNN (ví dụ anchor nhỏ hơn, ảnh phóng to).
- Trong 8 epoch đầu, mAP50-95 trên Val của Faster R-CNN cao hơn YOLO11n ở cùng số epoch (Hình 8.2), vì YOLO11n khởi động chậm với warm-up và augmentation mạnh (mosaic). So sánh dùng checkpoint tốt nhất của mỗi mô hình, không so theo cùng số epoch.
- Chưa có khoảng tin cậy theo bệnh nhân; các lát của cùng một BN tương quan với nhau.
- Test chỉ gồm lát có tổn thương, nên precision trên Test không phản ánh báo nhầm trên lát bình thường hay BN nhóm chứng.
- Tốc độ chỉ đo trên một GPU laptop, batch 1, PyTorch.
- Thí nghiệm đo **phát hiện ở mức lát**, chưa đo ảnh hưởng tới Tumor Zoom, Evidence Gate và phân loại mức bệnh nhân.
- Chênh lệch rất nhỏ giữa số liệu trong lúc huấn luyện và lúc đánh giá lại (mAP50-95 Val 0,2195 so với 0,2205) do tính toán fp16 trên GPU không tất định; số trong bảng là của lần đánh giá chính thức. Số box nhỏ của Test là 1.209 theo bộ tính này và 1.210 theo bộ tính dùng cho YOLO11n, do làm tròn số thực ở đúng ngưỡng 32×32 px.

## 11. Kết luận

| Câu hỏi | Kết quả đo được |
|---|---|
| Phát hiện tổn thương CT | **YOLO11n tốt hơn** (Test: mAP50 +6,6 điểm, mAP50-95 +5,6, F1 +4,7; Val: mAP50 +7,5) |
| Ở ngưỡng vận hành | YOLO11n tìm đúng nhiều hơn và báo nhầm ít hơn (Val: +421 box đúng, −498 box nhầm) |
| Tổn thương nhỏ | Cả hai đều yếu (recall khoảng 0,18–0,30 tùy ngưỡng); mô hình lớn hơn không khắc phục được |
| Tốc độ | **YOLO11n nhanh hơn khoảng 6–7 lần** (9 so với 61 ms/ảnh) |
| Tài nguyên | YOLO11n nhỏ hơn khoảng 16 lần về tham số, 31 lần về GFLOPs, 30 lần về dung lượng |

Trên dữ liệu CT của đề tài, YOLO11n vừa chính xác hơn, vừa nhanh hơn, vừa nhẹ hơn nhiều so với Faster R-CNN ResNet-50 FPN. Kết quả này ủng hộ lựa chọn bộ phát hiện một giai đoạn, nhẹ cho hệ thống chạy trên GPU 4 GB. Mô hình lớn hơn nhiều không đem lại độ chính xác cao hơn trên bài toán một lớp này, trong khi chi phí tính toán tăng hàng chục lần.

### Gợi ý trả lời hội đồng (dựa trên số liệu trên)

- **Vì sao so với Faster R-CNN?** Để so YOLO11n (một giai đoạn, nhẹ) với đại diện kinh điển của hướng hai giai đoạn, khác họ YOLO, lớn hơn khoảng 16 lần, trong cùng dữ liệu và cùng cách tính chỉ số.
- **Kết quả?** YOLO11n tốt hơn trên cả Val và Test: mAP50 Test 0,523 so với 0,456; và nhanh hơn khoảng 6–7 lần (9 so với 61 ms/ảnh).
- **Mô hình lớn hơn sao lại kém hơn?** Dữ liệu chỉ có một lớp và tổn thương nhiều kích thước nhỏ; Faster R-CNN đạt tốt nhất ngay ở epoch 3 rồi không cải thiện thêm (dấu hiệu quá khớp), trong khi YOLO11n có augmentation mạnh hơn (mosaic) và kiến trúc thiết kế cho phát hiện nhanh. Em chưa tinh chỉnh riêng cho Faster R-CNN, và dừng sớm (patience 5) đã cắt 4 epoch cuối của lịch học, nên đây là so sánh với công thức huấn luyện chuẩn của mỗi mô hình, chưa phải giới hạn tốt nhất của Faster R-CNN.
- **Có công bằng không?** Cùng dữ liệu, cùng chia tập theo bệnh nhân, cùng tiêu chí chọn checkpoint (mAP50-95 trên Val), cùng quy tắc chọn ngưỡng (F2 trên Val), cùng thuật toán tính mAP (đã kiểm chứng ra đúng số của Ultralytics trên YOLO11n), Test chạy một lần. Khác biệt duy nhất là công thức huấn luyện chuẩn của từng kiến trúc (mục 4, mục 10).

