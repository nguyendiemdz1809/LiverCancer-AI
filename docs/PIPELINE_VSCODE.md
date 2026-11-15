# Chạy pipeline AI (YOLO → Tumor Zoom → phân loại) trong VS Code

Pipeline dùng các mô hình đã huấn luyện (không huấn luyện lại):

| Bước | Mô hình / cấu hình |
|---|---|
| Phát hiện tổn thương | YOLO11n CT / MRI (`configs/yolo_inference.yaml`, ngưỡng 0.12 chọn trên Val) |
| Tumor Zoom | gộp box → nối vùng qua các lát → crop local + context 224×224 (`configs/roi.yaml`) |
| Phân loại | CT: `ct_hybrid_gt_s42` (HCC / ICC / CHCC). MRI: `mri_hybrid_gt_s42` (7 lớp, **chưa đạt chất lượng**). Cấu hình ở `configs/cls.yaml → inference` |

> Kết quả chỉ phục vụ nghiên cứu, **không** thay thế chẩn đoán của bác sĩ. Nhãn **MRI không đáng tin cậy**
> (Val acc 47.8%, macro-F1 42.8%) và luôn kèm cảnh báo. Khi YOLO không phát hiện vùng nào, pipeline trả `no_lesion_detected`
> và **không** kết luận bệnh nhân không mắc ung thư.

## 1. Chuẩn bị
1. Mở `E:\LiverCancer_AI_Graduation` trong VS Code; `Ctrl+Shift+P` → *Python: Select Interpreter* → chọn `.venv\Scripts\python.exe`.
2. Các checkpoint phải nằm ở `outputs\runs\cls\<run>\best.pt` (đã tải bundle) và `outputs\runs\yolo_*\...\best.pt`.
   SHA256 được kiểm tra khi nạp.
3. (Tùy chọn) `$env:YOLO_OFFLINE = "1"; .venv\Scripts\python -m pytest tests\test_pipeline.py -q`

## 2. Chạy bằng F5
`Ctrl+Shift+D` → chọn cấu hình → `F5`:

| Cấu hình | VS Code hỏi | Làm gì |
|---|---|---|
| **🧠 Pipeline AI: một bệnh nhân** | modality (ct / mri), mã BN (ví dụ `P0267`, `MR179970`) | chạy **mọi pha CT / chuỗi MRI có sẵn** của BN, gộp thành một dự đoán mức ca |
| **🧠 Pipeline AI: file NIfTI tự chọn** | modality, đường dẫn `.nii.gz` | chạy một volume bất kỳ (cùng định dạng dữ liệu huấn luyện: LPS, uint8) |
| **📊 Pipeline AI: đánh giá TOÀN BỘ Validation** | — | chạy pipeline trên mọi BN Val (CT 54, MRI 78; khoảng 30 phút, có cache theo ca), **cổng tắt** (mốc so sánh) → `reports/pipeline_validation/` |
| **🛡️ Pipeline AI: Validation khi BẬT Evidence Gate** | — | như trên nhưng bật cổng cho cả CT và MRI → `reports/pipeline_validation_gate/` |
| Pipeline AI: so sánh trước/sau Evidence Gate | — | đọc cache hai lần chạy → `reports/pipeline_validation/so_sanh_evidence_gate.md` |
| Pipeline AI: kiểm thử trên Validation + báo cáo | — | kiểm tra tính nhất quán + vài ca minh họa → `reports/pipeline_val_report.md` |

`result.png` và `result.json` **bằng tiếng Việt**:
- Ảnh nền trắng: ảnh gốc, YOLO khoanh vùng, Tumor Zoom cục bộ / ngữ cảnh, kết luận, xác suất từng lớp, cảnh báo.
- Với ca Validation có nhãn: hiển thị nhãn thực tế và huy hiệu **ĐÚNG / SAI**, kể cả khi mô hình sai.
- JSON có phần tóm tắt tiếng Việt và khối `Chi tiết kỹ thuật`.

## 2b. Xem báo cáo đánh giá trong VS Code
- Mở `reports/pipeline_validation/bao_cao_pipeline_validation.md` → `Ctrl+Shift+V` (preview, hiện cả ảnh ma trận nhầm lẫn và liên kết ảnh từng ca).
- Ảnh từng ca: `reports/pipeline_validation/figures/{ct,mri}/<mã BN>_{dung|sai|khong_phan_loai|CN_...}.png`; bấm vào file trong Explorer để xem.
- Bảng: `reports/pipeline_validation/{patients,volumes,errors,metrics_summary}.csv` (mở bằng Excel hoặc VS Code; mã hóa UTF-8 có BOM).

Kết quả: `outputs\predictions\pipeline\<mã ca>\result.png` (lát có khung, Tumor Zoom, biểu đồ điểm các lớp, cảnh báo)
và `result.json` (trạng thái, nhãn, điểm, từng vùng, cảnh báo, thời gian). Bản tóm tắt tiếng Việt được in ở terminal.

## 3. Chạy bằng terminal
```powershell
$env:YOLO_OFFLINE = "1"; $env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python scripts\90_run_pipeline.py --modality ct --patient P0267
.venv\Scripts\python scripts\90_run_pipeline.py --modality ct --patient P0267 --series C2 P
.venv\Scripts\python scripts\90_run_pipeline.py --modality mri --patient MR179970 --series T2WI "C+V"
.venv\Scripts\python scripts\90_run_pipeline.py --modality ct --nifti "E:\LiverDataset\CT\PLC-CECT\CECT\ct_files\P0267_ct_C3.nii.gz"
.venv\Scripts\python scripts\90_run_pipeline.py --modality ct --patient P0267 --cls-checkpoint outputs\runs\cls\ct_swin_t_gt_s42\best.pt
```
- `--series` của MRI dùng tên trong JSON (`In Phase`, `C+A`, …). CT dùng `C1 C2 C3 P`.
- Gợi ý ca **Validation**: CT `P0267` (CHCC; pipeline đoán sai, ví dụ thật của lỗi), các ca trong `reports/pipeline_val_report.md` §4.
- Khi chọn BN thuộc **Test**, script in cảnh báo và **không** hiển thị nhãn thật. Không dùng kết quả Test để chọn hay chỉnh bất cứ gì.
- `--cls-checkpoint` cho phép thử mô hình CT khác; kết quả khi đó bị đánh dấu "chưa được kiểm chứng".
- **Cổng bằng chứng (Evidence Gate) đã BẬT cho CT** (chọn trên Val, `configs/cls.yaml → inference.ct.evidence_gate`): CT chỉ phân loại khi có
  vùng YOLO độ tin cậy ≥ 0.4 ở ≥ 2 pha (một file: ≥ 0.35). Nếu không đạt, trạng thái là *Chưa đủ bằng chứng để phân loại*: **không** có nhãn
  và **không** kết luận khỏe mạnh. MRI: **không bật** (Val MRI không có nhóm chứng để chọn ngưỡng).
  `--evidence-gate off` tắt cổng (để so sánh), `on` ép bật cả cho MRI (chỉ để đánh giá).
  Ví dụ: `.venv\Scripts\python scripts\90_run_pipeline.py --modality ct --patient P0329` (CN, bị từ chối phân loại).
  Phân tích: `reports/pipeline_validation/han_che/bao_cao_han_che.md`; so sánh trước/sau: `reports/pipeline_validation/so_sanh_evidence_gate.md`.

## 4. Đọc kết quả
| Trường | Ý nghĩa |
|---|---|
| `status` | `ok` · `no_lesion_detected` (YOLO không thấy vùng nào: **không** phải "không ung thư") · `insufficient_evidence` (cổng bằng chứng từ chối: không nhãn, **không** phải "khỏe mạnh") · `classifier_unavailable` · `error` |
| `label`, `scores` | dự đoán gộp mức ca từ mọi vùng YOLO (cách gộp chọn trên Val: CT `mean_prob`, MRI `conf_weighted`) |
| `regions` | từng vùng nghi ngờ: volume, lát đầu–cuối, độ tin cậy YOLO, lát dùng để phân loại, nhãn của vùng |
| `reliable` | `false` với MRI và với checkpoint không phải bản đã chọn trên Val |
| `warnings` | cảnh báo bắt buộc hiển thị cùng kết quả |

## 5. Dùng trong Python (để nối website sau này)
```python
from livercancer.pipeline import LiverCancerPipeline
pipe = LiverCancerPipeline("ct")                       # nạp YOLO + Hybrid một lần
res = pipe.run_nifti([r"E:\...\P0267_ct_C2.nii.gz"])    # hoặc pipe.run([(tên, volume_uint8)]) / pipe.run_image(lát_2d)
res.status, res.label, res.scores, res.warnings
res.as_dict()                                           # JSON cho giao diện
```
