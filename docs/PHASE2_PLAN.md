# Kế hoạch Giai đoạn 2a — Chuẩn bị dữ liệu cho YOLO (split · 2D · bbox · pilot)

> Trạng thái: **ĐÃ DUYỆT, đang triển khai.** Quyết định chính thức của người dùng:
> (1) MRI dùng phương án A; (2) **CT YOLO chỉ có 1 lớp `lesion`**, liver mask giữ riêng cho segmentation/ROI;
> (3) pilot = Train + Val, Test giữ riêng; (4) ảnh grayscale 1 kênh; (5) người dùng tự upload ZIP;
> (6) series lỗi bị cách ly, không sửa nhãn; (7) dữ liệu gốc giữ nguyên.
> Đường dẫn output theo yêu cầu: `data/processed/pilot/yolo_ct/`, `data/processed/pilot/yolo_mri/`; notebook ở `notebooks/03_train_yolo_{ct,mri}_colab.ipynb`.
> Các mục bên dưới về lớp `liver` cho CT **không còn áp dụng**.
> Đầu vào: kết quả GĐ1 (`reports/audit_report.md`, commit `3030e68`). Training YOLO là GĐ2b, chỉ bắt đầu sau khi pilot được duyệt.

## 0. Nguyên tắc áp dụng (từ CLAUDE.md và quyết định của người dùng)
- Dữ liệu gốc `E:\LiverDataset` chỉ đọc. Mọi output nằm trong `data/` (manifest, split, processed).
- Split **theo bệnh nhân**, chia trước mọi bước khác; mọi pha/chuỗi/lát của một bệnh nhân thuộc cùng một split.
- Không dùng test để tuning. Test **không** nằm trong pilot.
- Xử lý streaming từng volume (volume lớn nhất ≈ 512×512×587 uint8 ≈ 154 MB, dạng 5D gấp 3 lần). `--workers` mặc định 2 (máy chỉ còn khoảng 1.6 GB RAM trống).
- Volume và series bị loại hoặc cách ly ở GĐ1 được loại khỏi nhãn huấn luyện, nhưng **bệnh nhân vẫn được giữ** trong split.

## 1. Chia Train / Validation / Test

### CT — PLC-CECT (361 BN)
- Phân tầng theo `cancer_type` (HCC/ICC/CHCC/CN), **70/15/15**, seed 42, chia trên danh sách bệnh nhân.
- Dự kiến khoảng 253 / 54 / 54 BN; mỗi lớp có khoảng 12–15 BN ở val và test.
- Tác giả không công bố split chính thức, nên đây là split của đồ án.
- Báo cáo kèm theo: phân bố lớp, giới và tuổi theo từng split (confounder E11), cùng số volume dùng được (1401) theo split.

### MRI — LLD-MMRI-MedSAM2 (498 BN)
Đối chiếu với split chính thức của LLD-MMRI2023 (316 / 78 / 104):

| Phần | Ánh xạ được? | Cách làm |
|---|---|---|
| **Validation (78)** | **Có**. 78/78 ID có trong MedSAM2, không trùng, mỗi ID là 1 BN × 8 series. Mức tin cậy: *suy luận* (file `LLDBaseline.json` có đúng 78 ID bằng kích thước val). | Dùng nguyên 78 ID làm Val |
| Train (316) / Test (104) | **Không**. Danh sách không công khai. | 420 BN còn lại chia phân tầng theo lớp, seed 42, **316 / 104** (cùng kích thước challenge) |

- **Ý nghĩa của phương án này:** Val trùng với val chính thức. Test là *test của đồ án*, **không phải** test chính thức, nên kết quả không so sánh trực tiếp được với leaderboard. Báo cáo sẽ ghi rõ điều này.
- Kiểm tra: phân bố lớp của 78 ID val so với toàn bộ dataset (xem bảng ở mục 3 của audit). FNH chỉ có 7 BN ở val, nên CI sẽ rộng.
- 31 series bị cách ly (18 BN) vẫn nằm trong split của bệnh nhân đó, nhưng không có nhãn YOLO và không dùng cho ROI.

### Output và kiểm tra
- `data/splits/ct_split.csv`, `data/splits/mri_split.csv`: cột `patient_id, label, split, source` (source = `official_val` / `stratified_seed42`). Được commit vì chỉ chứa ID.
- `tests/test_splits_no_leakage.py`:
  - không có ID nào xuất hiện ở hai split;
  - hợp các split = toàn bộ BN;
  - mọi ID val của MRI ∈ danh sách chính thức;
  - tỉ lệ từng lớp lệch ≤ 2 BN so với kỳ vọng.
- Sau khi tạo, split bị **đóng băng**. Muốn đổi phải có quyết định mới của người dùng.

## 2. Tiền xử lý 3D → 2D (`scripts/20_export_slices.py`)
- **Manifest trước** (`scripts/10_build_manifests.py`): mỗi dòng là một volume hoặc series, gồm đường dẫn, split, nhãn, shape, cờ `usable_lesion`, `liver_mask_ok`, `quarantined`. Lấy từ `reports/audit/*.csv`; không suy đoán nhãn.
- **Đọc:** `nifti_io.load_volume`, sau đó `as_3d` (CT 5D: kiểm tra 3 kênh giống nhau trên **toàn bộ** 676 volume, nếu khác thì dừng và báo lỗi). Mask đọc raw rồi so sánh `> 0` (xử lý 109 mask lưu 0/65535).
- **Hướng ảnh:** giữ thứ tự voxel gốc (LPS). Ảnh 2D = `slice[:, :, k].T` (hàng = j, cột = i), đúng quy ước đã kiểm chứng ở GĐ1. Script dừng nếu gặp orientation khác LPS.
- **Cường độ:**
  - CT: giữ nguyên uint8 0–255 (đã windowing [-200, 200] HU từ nguồn); **không** áp HU window lần nữa.
  - MRI: rescale theo percentile 0.5–99.5 **của từng volume** về 0–255. Trước khi áp dụng sẽ kiểm tra histogram trên train; nếu dữ liệu đã chuẩn hóa sẵn thì giữ nguyên và báo lại.
- **Kích thước:** giữ độ phân giải gốc (CT 512², MRI 176–512²); YOLO tự letterbox. Ghi lại shape gốc để ánh xạ bbox về sau.
- **Định dạng:** PNG grayscale 1 kênh. Tên file `{pid}_{phase|seq}_k{k:03d}.png`. Phương án 2.5D (k−1, k, k+1) để dành cho ablation sau.
- **Chọn lát cho train/val:**
  - *Dương tính:* lát có lesion mask.
  - *Âm tính:* lát không có tổn thương, lấy trong vùng gan (CT: liver ∪ lesion; với 13 volume liver rỗng thì lấy z-range của lesion ± 10 lát) cùng các lát của BN CN. MRI không có liver mask nên lấy ngẫu nhiên trong series.
  - Tỉ lệ âm:dương mặc định 1:1, chốt trên val.
  - Âm tính ở mức **lát cắt** chỉ có nghĩa là "lát này không có annotation tổn thương", **không** có nghĩa là bệnh nhân khỏe mạnh (quy tắc 19).
- **Test:** xuất **tất cả** lát, không chọn lọc theo GT, để mô phỏng suy luận thật (quy tắc 20). Việc xuất test chỉ thực hiện **sau** khi train/val xong và chỉ dùng ở máy local.

## 3. Bounding box YOLO từ mask hợp lệ (`scripts/30_build_yolo_labels.py`)
- **CT:** 2 lớp `0 = lesion`, `1 = liver` (theo kiến trúc đã chốt: liver giúp khoanh vùng khi suy luận).
  - lesion: mỗi thành phần liên thông 2D trên lát là 1 box. Lọc mảnh vụn theo diện tích tối thiểu (dự kiến < 16 px), ngưỡng chốt theo thống kê train (E17: P0268 có 22–27 thành phần).
  - liver: 1 box bao liver ∪ lesion trên lát. Không tạo box liver cho 13 volume có liver mask rỗng (E14).
  - Loại khỏi nhãn: P0051_P, P0165_C1 (mask rỗng) và P0231_C3 (mask lệch shape).
- **MRI:** 1 lớp `0 = lesion`. Box lấy từ mask MedSAM2 của **series không bị cách ly** (median IoU với JSON là 0.85). Kiểm tra thêm từng lát: IoU(box từ mask, JSON 2D_box); lát nào IoU < 0.3 thì ghi log để báo cáo, không tự sửa.
- Detection **không phân biệt loại** tổn thương. Phân loại thuộc nhánh classifier (GĐ4).
- **Định dạng YOLO:** `cls cx cy w h`, chuẩn hóa theo ảnh hiển thị (cx = i_center / n_i, cy = j_center / n_j). Có unit test `tests/test_bbox_conversion.py` cho chiều đi và chiều về.
- **QA:** vẽ lại khoảng 50 ảnh ngẫu nhiên từ chính file PNG và `.txt` đã xuất (round-trip), lưu ở `reports/figures/yolo_qa/`, để người dùng duyệt trước khi đóng gói.

## 4. Dataset pilot cho Colab (3–5 GB)
- **Nội dung:** **Train (lấy mẫu) + Val (đầy đủ)** của cả CT và MRI. **Không có Test.** Test chỉ ở local, upload sau nếu được phép.
- **Ngân sách dự kiến** (đo thực tế khoảng 60 KB/lát CT, 48 KB/lát MRI):

| Phần | Cách lấy | Ước lượng |
|---|---|---|
| CT dương tính | lát có tổn thương của train+val, bước 2 (các lát liền kề gần như trùng nhau) | ≈ 21k lát ≈ 1.2 GB |
| CT âm tính | 1:1 | ≈ 1.2 GB |
| MRI dương tính | lát trong mask của train+val, bỏ series cách ly | ≈ 27k lát ≈ 1.3 GB |
| MRI âm tính | 0.5:1 | ≈ 0.6 GB |
| **Tổng** | | **≈ 4.3 GB** (giới hạn cứng 5 GB) |

- Script `scripts/40_pack_pilot.py` chạy **dry-run trước** để in số lát và dung lượng. Chỉ đóng gói khi tổng ≤ 5 GB.
- **Đóng gói:** `pilot_ct.zip` và `pilot_mri.zip`, mỗi gói gồm `images/`, `labels/`, `dataset.yaml` (ultralytics), `manifest.csv`, `splits.csv`, `SHA256SUMS`, và `README_DATA.md` (nguồn, license: CT CC BY 4.0; MRI CC BY-NC 4.0, chỉ dùng phi thương mại, không chia sẻ lại). Lý do dùng zip: Colab đọc nhiều file nhỏ trực tiếp từ Drive rất chậm, nên sẽ giải nén ra `/content`.
- **Upload:** Claude **không** đăng nhập Google. Script chỉ tạo file zip trong `data/pilot/` (gitignore); **người dùng tự kéo thả** lên một thư mục Drive riêng tư, không chia sẻ.
- **Notebook** `notebooks/colab/00_check_pilot.ipynb`: mount Drive, kiểm tra SHA256, giải nén, đếm ảnh và nhãn, vẽ vài overlay. **Chưa train** ở bước này.

## 5. Thứ tự thực hiện và tiêu chí hoàn thành

| Bước | Lệnh | ✅ Xong khi |
|---|---|---|
| 2a-1 Split | `12_make_splits.py` + pytest | test no-leakage pass; bảng phân bố theo split được người dùng duyệt |
| 2a-2 Manifest | `10_build_manifests.py` | mọi volume trong CSV/JSON có đúng 1 dòng; số volume dùng được khớp với audit (CT 1401, MRI 3953 không cách ly) |
| 2a-3 Thử nhỏ | `20_export_slices.py --limit 5` | 5 BN mỗi modality xuất đúng; overlay round-trip đúng |
| 2a-4 Xuất train/val | `20_export_slices.py --splits train val` | không lỗi `as_3d`; log số lát; dung lượng đã biết |
| 2a-5 Nhãn YOLO | `30_build_yolo_labels.py` + pytest | QA overlay được duyệt; thống kê box/lát/lớp |
| 2a-6 Pilot | `40_pack_pilot.py --dry-run`, sau đó chạy thật | ≤ 5 GB; SHA256 OK; người dùng upload; notebook kiểm tra trên Colab pass |

Mỗi bước chạy lâu (ước lượng 2a-4: khoảng 1–2 giờ với 2 worker) sẽ báo trước và chạy nền. Dừng sau 2a-1 và 2a-5 để người dùng duyệt.

## 6. Điểm cần người dùng xác nhận
1. Split MRI như mục 1: val chính thức + 316/104 phân tầng cho phần còn lại.
2. CT detection 2 lớp (lesion + liver) hay chỉ lesion.
3. Pilot không chứa Test.
4. Ảnh 1 kênh làm baseline; 2.5D để dành cho ablation.
5. Người dùng tự upload file zip lên Drive.
