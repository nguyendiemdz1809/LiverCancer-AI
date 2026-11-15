# Kế hoạch: Automatic Tumor Zoom (GĐ3) và Website LiverCancer AI (GĐ6)

> Trạng thái (2026-10-25): **ĐANG TRIỂN KHAI.** Xong: C.1 (Tumor Zoom + đánh giá Val: `reports/roi_zoom_val.md`) và C.2 (khung website, có ROI zoom: `docs/WEBSITE_VSCODE.md`). Còn lại: C.3 (xuất crop cho GĐ4), C.4.
> Ghi chú ban đầu: Đầu vào đã chốt: hai detector E0 và ngưỡng conf 0.12
> (`configs/yolo_inference.yaml`, `reports/yolo_detection_summary.md`). Không train lại YOLO. Test chỉ dùng một lần ở đánh giá cuối.

## A. Automatic Tumor Zoom (GĐ3)

### A.1 Yêu cầu rút ra từ giai đoạn detection
| Quan sát trên Val | Hệ quả cho ROI zoom |
|---|---|
| Box trùng lặp bên trong cùng tổn thương (MR69046); box chứa box | Gộp box trong từng lát (NMS theo IoU và theo tỉ lệ chứa nhau) |
| Tổn thương lớn bị bao thiếu (P0035, P0267; CT mAP50-95/mAP50 0.51) | Margin tỉ lệ theo kích thước box; lấy box hợp qua các lát liền kề |
| Một box bao trùm nhiều ổ nhỏ (P0058) | Crop local vẫn giữ được các ổ, nên không tách box |
| Tổn thương xuất hiện trên nhiều lát (CT khoảng 45 lát mỗi volume, MRI khoảng 8) | Nối box thành **ứng viên 3D**, chọn lát đại diện |
| conf 0.12: 11/12 BN CN có ≥ 1 box; lát âm có FP 11–15% | Lọc ứng viên 3D theo độ dài và confidence. Ngưỡng lọc **chọn trên Val** |
| Tổn thương nhỏ bị bỏ sót nhiều | Không sửa được ở bước ROI; tính là lỗi pipeline và báo cáo |

### A.2 Thiết kế (module `src/livercancer/roi/`)
1. **Quét volume** (streaming, từng volume): `LesionDetector.predict` trên mọi lát ở conf 0.12. Lưu `outputs/predictions/<mod>/<split>/<BN>_<pha>.csv`.
2. **Gộp trong lát:** box có IoU > 0.5 hoặc chứa nhau > 80% thì hợp nhất; conf = max.
3. **Nối 3D:** ghép box ở lát k với lát k±1 (cho phép hở 1 lát) khi IoU ≥ 0.3, tạo track = ứng viên tổn thương.
   Thông số track: số lát, conf lớn nhất và trung bình, box hợp.
4. **Lọc track:** số lát tối thiểu và conf tối thiểu, chọn trên Val (CT dùng thêm FP trên BN CN làm ràng buộc).
5. **Chọn lát đại diện:** top-k lát theo diện tích × conf (k mặc định 3; 2.5D để dành cho ablation).
6. **Crop:**
   - *local* = box + margin max(15%, 8 px), làm vuông, resize 224;
   - *context* = cạnh 2.5× box (tối thiểu 128 px) quanh tâm, làm vuông, pad bằng 0 nếu vượt ảnh, resize 224;
   - giữ uint8 xám (3 kênh khi đưa vào CNN/ViT).
   CT: kiểm tra crop context có dính chữ thước đo in chồng không (confounder).
7. **Manifest:** `data/processed/roi/<mod>/manifest.csv` gồm BN, split, nhãn BN, pha/chuỗi, track, lát, box, conf, đường dẫn crop,
   `source = yolo|gt`. Riêng Val có thêm IoU với GT để phân tích.
8. **Bộ oracle** (crop từ GT, train/val/test) gắn nhãn **"oracle"**, chỉ để báo cáo riêng. Test end-to-end **chỉ** dùng crop từ YOLO.
   MRI: không dùng series trong `mri_bbox_quarantine.csv` để huấn luyện.

### A.3 Đánh giá trên Val (tiêu chí hoàn thành GĐ3)
- % tổn thương 3D (theo GT) có ít nhất 1 track khớp; IoU giữa box track và GT; số track giả trên mỗi volume; số track trên BN CN (CT).
- Thống kê theo kích thước, lớp và pha/chuỗi; ảnh minh họa local và context.
- Script dự kiến: `scripts/70_build_roi.py`, `scripts/71_eval_roi_val.py`; báo cáo `reports/roi_zoom_val.md`.
- Dữ liệu train classifier: crop GT (oracle) và crop YOLO trên train. YOLO đã thấy train nên crop YOLO trên train lạc quan hơn thực tế; ghi rõ điều này.

## B. Website LiverCancer AI (GĐ6, giao diện sáng, hoàn toàn tiếng Việt)

### B.1 Công nghệ (mặc định)
**Streamlit** (đã cài 1.66.0 trong `.venv`), chạy local trên RTX 3050 4 GB hoặc CPU. Lý do: GĐ6 trong `CLAUDE.md` đã chọn Streamlit;
toàn bộ bằng Python nên dùng lại trực tiếp `LesionDetector` và các module ROI; không cần server riêng.
Giao diện sáng bằng `.streamlit/config.toml` (`base = "light"`); toàn bộ chữ trên giao diện bằng tiếng Việt.
Không đưa dữ liệu lên mạng (MRI CC BY-NC 4.0).

### B.2 Trang và chức năng
| Trang | Nội dung |
|---|---|
| **Trang chủ** | Giới thiệu đề tài, pipeline (phát hiện → phóng to tổn thương → phân loại → giải thích), lời cảnh báo *"Chỉ phục vụ nghiên cứu, không dùng cho chẩn đoán"* |
| **Phân tích ảnh** | Chọn CT hoặc MRI; tải file `.nii.gz` hoặc chọn ca mẫu từ **Validation**; chọn pha/chuỗi; thanh trượt chọn lát; ảnh có box, nhãn "tổn thương" và độ tin cậy; danh sách lát có phát hiện; **ROI phóng to** (local và context) sau khi xong GĐ3; tải kết quả (CSV, PNG) |
| **Phân loại** | Để trống tới GĐ4, ghi rõ "chưa khả dụng". Sau GĐ4 sẽ hiển thị xác suất từng lớp và Grad-CAM/attention |
| **Về mô hình** | Số liệu Val đã chốt (`reports/yolo_detection_summary.md`), ngưỡng, hạn chế, giấy phép dữ liệu |

- Ca thuộc Test: chỉ hiển thị, kèm cảnh báo. Không dùng để chọn ngưỡng hay mô hình.
- Hiệu năng: nạp detector một lần (`st.cache_resource`), fp16 trên GPU, tự chuyển sang CPU; đo độ trễ mỗi volume.
- Mã nguồn: `app/streamlit_app.py` và `app/pages/`; chạy bằng `.venv\Scripts\streamlit run app\streamlit_app.py`, kèm cấu hình F5 trong `.vscode/launch.json`.

## C. Thứ tự đề xuất
1. GĐ3-a: quét volume, gộp box, nối 3D, crop (A.2 bước 1–7) trên Val; thống kê A.3; chọn tham số lọc trên Val.
2. GĐ6-a: khung website (Trang chủ, Phân tích ảnh có detection, Về mô hình), giao diện sáng, tiếng Việt.
3. GĐ3-b: xuất crop cho train/val (và Test để đánh giá cuối), đóng gói cho Colab. Website thêm phần ROI phóng to.
4. GĐ4: Hybrid CNN + Transformer, rồi tích hợp trang Phân loại.
