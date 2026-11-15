# Tổng kết Giai đoạn YOLO Detection (CT và MRI): đã chốt

*Chốt ngày 2026-10-25 theo quyết định của người dùng: dùng hai checkpoint E0 hiện tại và **không** chạy thêm thí nghiệm (không yolo11s,
không imgsz 640, không train lại). Mọi số liệu dưới đây lấy trên **Validation** và truy được về file trong `outputs/`. **Tập Test chưa được dùng**;
Test sẽ chỉ được đánh giá **một lần**, end-to-end, ở giai đoạn đánh giá cuối.*

## 1. Mô hình cuối

| | CT | MRI |
|---|---|---|
| Dataset | PLC-CECT (CT có thuốc cản quang, 4 pha) | LLD-MMRI-MedSAM2 (MRI, 8 chuỗi) |
| Checkpoint | `outputs/runs/yolo_ct/yolo11n_e50_img512_baseline/weights/best.pt` | `outputs/runs/yolo_mri/yolo11n_e50_img512_baseline/weights/best.pt` |
| SHA256 | `65c126f310d8b7751f67625d917252ff5b8984ff741795d2e92b496a831f5707` | `cc6fef44446514d598e6f4a862c78e0eb214d28f2f7a4dac82ce4b060b157377` |
| Kiến trúc / cấu hình | yolo11n COCO pretrained, 1 lớp `lesion`, imgsz 512, SGD, 50 epoch, patience 20, seed 42 | giống CT |
| Best epoch / early stop | 15 / 35 | 24 / 44 |
| Dữ liệu train/val | `yolo_ct_full.zip` (34.873 / 14.627 ảnh) | `yolo_mri_pilot.zip` (28.595 / 7.491 ảnh) |
| **Ngưỡng vận hành** | **conf 0.12** | **conf 0.12** |

Cấu hình suy luận: `configs/yolo_inference.yaml` (SHA256 được kiểm tra khi nạp; `conf` được test so khớp với `threshold_choice.json`).

## 2. Kết quả trên Validation

| | CT (54 BN, 7.631 box GT) | MRI (78 BN, 4.880 box GT) | Nguồn |
|---|---|---|---|
| Precision / Recall (Ultralytics, ngưỡng F1 lớn nhất) | 0.688 / 0.481 | 0.797 / 0.660 | `<run>/metrics_val.json` |
| mAP50 / mAP50-95 | 0.551 / 0.282 | 0.749 / 0.504 | `<run>/metrics_val.json` |
| Ở conf 0.12: P / R mức box (IoU ≥ 0.5) | 0.474 / 0.578 | 0.634 / 0.745 | `<run>/analysis/threshold_sweep.csv` |
| Ở conf 0.12: volume (CT) / series (MRI) có tổn thương được phát hiện | 91.4% (149/163) | 85.9% | như trên |
| Ở conf 0.12: BN có tổn thương được phát hiện | 42/42 | 77/78 | như trên |
| Ở conf 0.12: lát không annotation có ≥ 1 box | 11.1% | 14.7% | như trên |
| Ở conf 0.12: BN CN (nhóm chứng) có ≥ 1 box | 11/12 | (không có nhóm CN) | như trên |

- **Không xếp hạng CT với MRI bằng mAP:** hai bộ khác nhau về modality, BN, nguồn nhãn, cách lấy mẫu và kích thước tổn thương
  (`reports/ct_e0_analysis.md` §7.8, `reports/figures/yolo_e0_ct_vs_mri.png`).
- **Cách chọn ngưỡng:** F2 lớn nhất ở mức box, trên Val của **từng** modality, quét 0.01–0.80 (`scripts/53_threshold_sweep.py`).
  F2 ưu tiên hạn chế bỏ sót. Cả hai modality đều ra 0.12.
- **Điểm yếu đã biết:**
  - Tổn thương **nhỏ** (< 32² px): CT bỏ sót 72% ở conf 0.1, MRI bỏ sót 49% ở conf 0.25.
  - CT: ICC và pha P có recall thấp nhất; box kém khít (mAP50-95/mAP50 0.51, MRI là 0.67).
  - Cả hai: phần lớn FP là box lệch hoặc quá rộng quanh một tổn thương thật.
  - Ở conf 0.12, YOLO **không** dùng được để kết luận một BN có hay không có tổn thương, vì 11/12 BN CN có ít nhất 1 box.
- Chi tiết: `reports/ct_e0_analysis.md`, `reports/mri_e0_analysis.md`.

## 3. Kiểm tra demo và độ ổn định (2026-10-25)

| Kiểm tra | Kết quả | Nguồn |
|---|---|---|
| Ảnh dựng từ NIfTI gốc trùng từng pixel với ảnh train | 24/24 (demo) và 120/120 (mẫu ngẫu nhiên seed 7); có test tự động | `reports/inference_demo.md`, `tests/test_yolo_inference.py` |
| Box khi đưa ndarray so với đọc file PNG | lệch ≤ 0.25 px (fp16) | `outputs/predictions/inference_demo/summary_cuda.json` |
| GPU fp16 so với CPU fp32 | cùng TP/FP/FN | `summary_cuda.json`, `summary_cpu.json` |
| Tốc độ / VRAM (RTX 3050 4 GB) | GPU khoảng 5 ms/ảnh, khoảng 235 MiB cho cả hai mô hình; CPU khoảng 30 ms/ảnh | `reports/inference_demo.md` §1, §4.4 |
| Không có CUDA | tự chuyển sang CPU (thử với `CUDA_VISIBLE_DEVICES=-1`) | `tests/test_yolo_inference.py::test_device_fallback_to_cpu` |
| Mẫu ngẫu nhiên lớn hơn (40 lát dương + 20 lát âm mỗi modality, seed 7) | chạy không lỗi; CT TP 20 · FP 33 · FN 23; MRI TP 34 · FP 14 · FN 6 (**chỉ là mẫu, không phải đánh giá**) | `outputs/predictions/inference_demo_stability/summary_cuda.json` |
| CT 5D có chữ thước đo in chồng (P0132 C1, P) | đọc đúng; 0 box chạm vùng rìa 50 px có chữ thước đo | `outputs/predictions/adhoc/P0132_ct_{C1,P}/detections.csv` |
| MRI độ phân giải thấp (DWI 192²) và tên chuỗi có dấu cách (`In Phase`) | chạy đúng | `outputs/predictions/adhoc/MR179970_1_{DWI,InPhase}_0000/` |
| Bộ test | 31/31 PASS | `tests/` |

## 4. Thành phần dùng lại cho các giai đoạn sau
- `livercancer.data.slices`: tiền xử lý giống lúc train (LPS, CT 5D → 3D, uint8, `slice.T`, 3 kênh).
- `livercancer.data.sources`: tìm file NIfTI theo (BN, pha/chuỗi).
- `livercancer.detection.yolo_detector.LesionDetector`: nạp mô hình theo modality, kiểm SHA256, GPU fp16 hoặc CPU, ngưỡng từ config.
- `livercancer.evaluation.matching`: ghép box và đếm TP/FP/FN.
- `scripts/61_yolo_predict_nifti.py`: chạy trên một volume bất kỳ (nền cho website).

## 5. Hạn chế cần ghi trong luận văn
- Không có thí nghiệm so sánh n/s, imgsz hay nhiều seed (quyết định chốt sớm), nên **không** kết luận được về kích thước mô hình hay độ phân giải.
- CT Val là bộ đầy đủ; CT train chỉ dùng 1/2 số lát dương tính. CT overfit sớm (best epoch 15).
- Nhãn MRI là mask MedSAM2, chỉ giữ lát đồng thuận với 2D_box JSON gốc; 31 series bị cách ly.
- Metrics Test chưa có. Test sẽ được đánh giá end-to-end (YOLO → ROI → phân loại) một lần cho cấu hình cuối.
