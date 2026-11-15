# Kiểm tra tính độc lập của tập Test và khóa cấu hình

_Sinh bởi `scripts/95_final_test_lock.py` lúc 2026-10-26 22:18. Bước này không chạy mô hình trên Test; ảnh Test chỉ được đọc để tính dấu vân tay (hash) khi tìm ảnh trùng giữa các split._

**Kết luận: ĐẠT — Test độc lập, cấu hình đã được khóa.**

Test: CT 54 BN (CHCC 13, CN 12, HCC 14, ICC 15); MRI 104 BN.

| Mã | Kết quả | Kiểm tra | Chi tiết |
|---|---|---|---|
| S1 | ✅ | CT: mỗi BN thuộc đúng một split (361 BN; Test 54) |  |
| S1 | ✅ | MRI: mỗi BN thuộc đúng một split (498 BN; Test 104) |  |
| S2 | ✅ | CT: nhãn trong split trùng `patient_data.csv` |  |
| S3 | ✅ | MRI: Val = 78 ID chính thức, không ID nào trong Test |  |
| S4 | ✅ | `tests/test_splits_no_leakage.py` pass | 6 passed in 0.12s |
| S5 | ✅ | split đóng băng (2026-10-19 12:25) trước mọi bộ dữ liệu huấn luyện (2026-10-19 13:30) |  |
| S6 | ✅ | 2 bản sao `split_manifest.csv` trong data/processed trùng bản gốc |  |
| Y1 | ✅ | YOLO pilot CT: 307 BN, split ['train', 'val'], không BN Test |  |
| Y1 | ✅ | YOLO full CT: 307 BN, split ['train', 'val'], không BN Test |  |
| Y1 | ✅ | YOLO pilot MRI: 394 BN, split ['train', 'val'], không BN Test |  |
| Y2 | ✅ | ZIP yolo_ct_pilot.zip: 307 BN Train/Val, 0 BN Test trong ảnh/nhãn/metadata, thư mục ['train', 'val'] (bảng split đầy đủ `yolo_ct/split_manifest.csv` trùng bản đóng băng) |  |
| Y2 | ✅ | ZIP yolo_ct_full.zip: 307 BN Train/Val, 0 BN Test trong ảnh/nhãn/metadata, thư mục ['train', 'val'] (bảng split đầy đủ `yolo_ct/split_manifest.csv` trùng bản đóng băng) |  |
| Y2 | ✅ | ZIP yolo_mri_pilot.zip: 394 BN Train/Val, 0 BN Test trong ảnh/nhãn/metadata, thư mục ['train', 'val'] (bảng split đầy đủ `yolo_mri/split_manifest.csv` trùng bản đóng băng) |  |
| Y3 | ✅ | YOLO CT run `yolo11n_e50_img512_baseline`: data `/content/yolo_ct/data_colab.yaml`, đánh giá trên split `val` |  |
| V1 | ✅ | YOLO CT conf 0.12 = argmax F2 trên Val (`threshold_choice.json`, 14627 lát Val) |  |
| Y3 | ✅ | YOLO MRI run `yolo11n_e50_img512_baseline`: data `/content/yolo_mri/data_colab.yaml`, đánh giá trên split `val` |  |
| V1 | ✅ | YOLO MRI conf 0.12 = argmax F2 trên Val (`threshold_choice.json`, 7491 lát Val) |  |
| C1 | ✅ | ROI CT manifest: 15055 ROI, split ['train', 'val'], không BN Test |  |
| C2 | ✅ | ZIP cls_ct.zip: 0 BN Test, thư mục ['train', 'val'] |  |
| C3 | ✅ | Hybrid CT `ct_hybrid_gt_s42`: huấn luyện 194 BN Train, chọn epoch trên 42 BN Val (khớp manifest) |  |
| C4 | ✅ | Hybrid CT: dự đoán dùng để chọn checkpoint chỉ trên 42 BN Val |  |
| V2 | ✅ | Cách gộp CT `mean_prob` chọn trên Val (`metrics_val.json`) |  |
| C1 | ✅ | ROI MRI manifest: 31254 ROI, split ['train', 'val'], không BN Test |  |
| C2 | ✅ | ZIP cls_mri.zip: 0 BN Test, thư mục ['train', 'val'] |  |
| C3 | ✅ | Hybrid MRI `mri_hybrid_gt_s42`: huấn luyện 316 BN Train, chọn epoch trên 78 BN Val (khớp manifest) |  |
| C4 | ✅ | Hybrid MRI: dự đoán dùng để chọn checkpoint chỉ trên 78 BN Val |  |
| V2 | ✅ | Cách gộp MRI `conf_weighted` chọn trên Val (`metrics_val.json`) |  |
| C5 | ✅ | ZIP code gửi lên Colab không chứa ID BN Test |  |
| V3 | ✅ | Cache `pipeline_validation/ct`: 54 BN = toàn bộ Val, không BN Test |  |
| V3 | ✅ | Cache `pipeline_validation_gate/ct`: 54 BN = toàn bộ Val, không BN Test |  |
| V4 | ✅ | Tumor Zoom: bộ lọc track chọn trên 54 BN Val (`roi_val/ct`) |  |
| V3 | ✅ | Cache `pipeline_validation/mri`: 78 BN = toàn bộ Val, không BN Test |  |
| V3 | ✅ | Cache `pipeline_validation_gate/mri`: 78 BN = toàn bộ Val, không BN Test |  |
| V4 | ✅ | Tumor Zoom: bộ lọc track chọn trên 78 BN Val (`roi_val/mri`) |  |
| V5 | ✅ | Evidence Gate: chọn từ phân tích Val (`limitations.json`) |  |
| O1 | ✅ | Không có ID BN Test trong outputs/, reports/, data/processed/ (ngoài bản sao bảng split và 4 file audit GĐ1 tạo trước khi có split) |  |
| D0 | ✅ | Đọc được dấu vân tay 5388/5388 volume |  |
| D1 | ✅ | Không volume Test nào trùng nội dung voxel với volume Train/Val của BN khác |  |
| G1 | ✅ | Code và cấu hình đã commit (src, scripts, configs, tests sạch) |  |

File có mã BN Test nhưng được chấp nhận (audit dữ liệu GĐ1, tạo **trước** khi split tồn tại, không có mô hình nào): `reports/dataset_audit.json`, `reports/figures/audit/mri_MR111774.png`, `reports/figures/audit/mri_MR170186.png`, `reports/tables/ct_class_distribution.md`. Bản sao `split_manifest.csv` (trùng SHA256 bản đóng băng) liệt kê mọi BN theo thiết kế.

Ảnh trùng: đã so 5388 volume (sha256 của 16 MB dữ liệu voxel, bỏ 8 MB đầu). Trùng giữa Test và Train/Val: 0. Trùng giữa các BN khác trong Train/Val: 0. Khối toàn 0 (không so được): 0. Volume đọc được một phần (file tải về bị cắt cụt) `ct|P0286|C2`: so 0 MB đọc được với 207 volume của nhóm split còn lại, trùng 0. Kiểm tra này chỉ phát hiện bản sao y hệt, không phát hiện cùng một người được chụp nhiều lần dưới hai mã BN.

## Cấu hình đã khóa

- Commit code: `2e42271f5b034d88e6973019f67275ba3d0cee55` (fix(GĐ5): correct the 'Test already started' guard of the lock script); 66 file code/cấu hình có SHA256 trong `locked_config.json`.
- **Khóa lại** (thay bản khóa `00eec07644` lúc 2026-10-26T22:08:55, lưu ở `locked_config_superseded_1.json`): Lỗi kỹ thuật ở bước xuất nhãn chuẩn Test (tiến trình con trên Windows không nạp được hàm xuất); đã sửa bằng hàm bọc, không đổi cách tính. Chưa có dự đoán nào trên Test trước khi khóa lại.

| | CT | MRI |
|---|---|---|
| YOLO weights | `outputs/runs/yolo_ct/yolo11n_e50_img512_baseline/weights/best.pt`<br>sha256 `65c126f310d8b775…` | `outputs/runs/yolo_mri/yolo11n_e50_img512_baseline/weights/best.pt`<br>sha256 `cc6fef44446514d5…` |
| Ngưỡng YOLO (conf) | 0.12 (argmax F2 trên Val) | 0.12 (argmax F2 trên Val) |
| Tumor Zoom | gộp IoU>0.5 / chứa>0.8, nối IoU≥0.3 gap 1, không lọc track (≥1 lát, conf≥0.12), crop 224 | gộp IoU>0.5 / chứa>0.8, nối IoU≥0.3 gap 1, không lọc track (≥1 lát, conf≥0.12), crop 224 |
| Bộ phân loại | `outputs/runs/cls/ct_hybrid_gt_s42/best.pt`<br>sha256 `291d3b7e50f838b1…` | `outputs/runs/cls/mri_hybrid_gt_s42/best.pt`<br>sha256 `77cc2d3cd19e9c6d…` |
| Lát mỗi vùng / khoảng cách | 5 / ≥2 | 5 / ≥1 |
| Gộp mức BN | `mean_prob` (chọn trên Val) | `conf_weighted` (chọn trên Val) |
| Evidence Gate | BẬT: conf ≥ 0.4 ở ≥ 2 pha; một file: conf cao nhất ≥ 0.35 | TẮT (không có nhóm chứng trên Val) |
| Độ tin cậy nhãn | đã chọn trên Val | **chưa đạt** — chỉ là thí nghiệm nghiên cứu |

`scripts/96_final_test.py` từ chối chạy nếu bất kỳ SHA256 nào ở trên hoặc code trong `src/`, `scripts/`, `configs/` khác với bản khóa, và chỉ chạy Test **một lần** (chỉ cho phép chạy tiếp các ca còn thiếu nếu bị gián đoạn).
