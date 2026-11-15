# Kế hoạch Giai đoạn 3: huấn luyện YOLO CT/MRI chính thức

> Trạng thái (2026-10-25): **HOÀN THÀNH.** CT-E0 và MRI-E0 là mô hình cuối, ngưỡng conf 0.12 (Val). Tổng kết: `reports/yolo_detection_summary.md`.
> Không cam kết độ chính xác; mọi con số chỉ đến từ thực nghiệm.
> Chỉ chọn mô hình và ngưỡng trên **Validation**. **Test** chỉ đánh giá **một lần** cho cấu hình cuối của mỗi modality.

## 0. Điều kiện để bắt đầu (gate)
1. ✅ Đã phân tích CT pilot từ file thật (`reports/ct_pilot_analysis.md`). Kết luận: mô hình **chưa huấn luyện đủ** (loss chưa bão hòa,
   recall 0.847 ở conf 0.001 nhưng 0.363 ở conf 0.25); FN tập trung ở **tổn thương nhỏ** (97% của box < 32² bị bỏ sót ở conf 0.25)
   và **ICC** (recall 0.197); 47% FP là lỗi định vị, 14% FP nằm ngoài gan. Không thấy lỗi nhãn có hệ thống.
2. Đã chạy MRI pilot (`notebooks/03_train_yolo_mri_colab.ipynb`) và phân tích (`reports/mri_pilot_analysis.md`).
3. Pipeline pilot không có lỗi dữ liệu: nhãn hợp lệ, không rò rỉ bệnh nhân, ảnh đối chiếu GT với dự đoán hợp lý.
   Nếu ảnh FP/FN cho thấy **lỗi nhãn có hệ thống**, phải sửa ở bước dữ liệu (GĐ2) trước, không bù bằng hyper-parameter.

## 1. Dữ liệu cho train chính thức

| | Pilot (đã có) | Đề xuất cho train chính thức | Ghi chú |
|---|---|---|---|
| CT train | 1/4 số lát dương tính, âm:dương = 1:1 | **toàn bộ** lát dương tính; âm:dương = 1:1 | Các lát CT liền kề rất giống nhau; lấy toàn bộ tăng dữ liệu nhưng cũng tăng thời gian/epoch. Phương án dự phòng: 1/2 |
| CT val | 1/2 số lát dương tính | **toàn bộ** lát dương tính (Val đầy đủ để chọn mô hình) | Val pilot (1/2) chỉ để kiểm tra pipeline |
| MRI train/val | toàn bộ lát dương tính đạt điều kiện đồng thuận với JSON; âm:dương = 0.5 | giữ nguyên, có thể dùng lại `yolo_mri_pilot.zip` | Đã là dữ liệu đầy đủ; thử tỉ lệ âm 1.0 nếu FP trên lát âm cao |
| Test | không có trong ZIP | giữ riêng, xuất **toàn bộ** lát khi đánh giá cuối | Không chọn mẫu theo GT |

- Lệnh xuất CT đầy đủ (chưa chạy; cần người dùng đồng ý vì dung lượng có thể vượt 5 GB):
  ```powershell
  .venv\Scripts\python scripts\20_build_yolo_pilot.py --modality ct --pos-stride-train 1 --pos-stride-val 1 --out data\processed\full\yolo_ct
  .venv\Scripts\python scripts\30_qa_pilot.py --modality ct --data data\processed\full\yolo_ct
  ```
  Script đóng gói `40_pack_pilot.py` hiện chỉ làm việc với `data/processed/pilot`. Cần thêm tham số `--set full` (xem mục 7).
- Dung lượng ước tính (đo từ pilot: khoảng 84 KB/lát CT): xem mục 2.

## 2. Thống kê dữ liệu làm căn cứ (đo trên pilot ở local)
Đo từ nhãn trong `data/processed/pilot/yolo_{ct,mri}` (diện tích box tính bằng pixel trên ảnh gốc):

| | Box | Diện tích p10 / p50 / p90 (px) | < 32² | 32²–96² | > 96² | Cạnh dài p10 / p50 (px) | Box/ảnh dương tính |
|---|---|---|---|---|---|---|---|
| CT train | 9.397 | 567 / 3.672 / 18.500 | 19.0% | 56.6% | 24.4% | 26 / 67 | 1.05 (max 7) |
| CT val | 3.850 | 728 / 3.466 / 21.457 | 15.1% | 63.2% | 21.7% | 30 / 66 | 1.04 (max 3) |
| MRI train | 18.661 | 448 / 2.304 / 9.416 | 27.9% | 61.6% | 10.5% | 23 / 52 | 1.00 (max 2) |
| MRI val | 4.880 | 441 / 2.346 / 15.298 | 26.1% | 53.6% | 20.3% | 22 / 52 | 1.00 (max 3) |

- Ảnh: CT đều 512×512. MRI có nhiều kích thước (512² 34%, 448² 29%, 480² 8%, 384² 7%, 352² 7%, 432² 4%, còn lại nhỏ hơn).
  Ở `imgsz = 512`, ảnh MRI nhỏ được phóng to, nên tổn thương không bị thu nhỏ thêm.
- **Tổn thương nhỏ (< 32²) chiếm 15–19% box CT và 26–28% box MRI**, với cạnh dài p10 khoảng 22–30 px. Đây là nhóm có nguy cơ
  bị bỏ sót cao nhất. E3 (`imgsz = 640`) được giữ trong ma trận thí nghiệm, và bảng FN theo kích thước sẽ quyết định có chạy hay không.
- **Dữ liệu chưa dùng:** CT train có **34.558** lát dương tính, nhưng pilot chỉ dùng 8.917 (26%). CT val có 7.314 lát, pilot dùng 3.691 (50%).
  MRI pilot đã dùng 93% số lát dương tính; 7% còn lại bị loại do mask và JSON box không đồng thuận, hoặc là mảnh vụn.
- **Ước tính dung lượng CT đầy đủ** (khoảng 84 KB/lát, đo từ pilot: 2.11 GB / 25.196 lát):
  toàn bộ lát dương tính + âm 1:1 ≈ 34.558×2 + 7.314×2 ≈ 83.700 lát ≈ **7.0 GB**.
  Nếu train lấy 1/2 lát dương tính (val đầy đủ) ≈ 49.200 lát ≈ **4.1 GB**.
  Cộng với MRI (2.7 GB), phương án 1/2 vẫn vừa 15 GB Drive miễn phí; phương án đầy đủ cần dọn bớt ZIP pilot CT.

## 3. Ma trận thí nghiệm (chạy theo thứ tự; mỗi dòng là một `TAG` trong notebook 04)

| # | Thí nghiệm | Thay đổi so với baseline | Mục đích | Chạy khi |
|---|---|---|---|---|
| E0 | `yolo11n` baseline | — | Mốc so sánh | Luôn chạy |
| E1 | `yolo11s` baseline | `MODEL='yolo11s.pt'` | So sánh n và s (yêu cầu đề tài) | Luôn chạy, nếu GPU/thời gian cho phép |
| E2 | `fliplr0` | `fliplr: 0.0` | Lật trái-phải đảo vị trí gan trong ảnh, có thể là augmentation không thực tế về giải phẫu. Kiểm tra bằng thực nghiệm | Sau E0/E1, trên mô hình tốt hơn |
| E3 | `img640` | `IMGSZ=640` | Tổn thương nhỏ: tăng độ phân giải đầu vào | Nếu FN tập trung ở tổn thương nhỏ (mục 5 của báo cáo phân tích) |
| E4 | `neg1` / `neg0.5` | Tỉ lệ lát âm (cần xuất lại dữ liệu) | Cân bằng FP trên lát âm và recall | Nếu FP trên lát âm (CN) cao ở ngưỡng vận hành |
| E5 | 2.5D (3 lát liền kề) | Dataset 3 kênh | Ablation đã thống nhất từ GĐ2 | Sau khi chốt baseline 2D |

- Mỗi thí nghiệm chạy **1 seed (42)** vì giới hạn GPU. Cấu hình cuối chạy thêm **2 seed** (43, 44) nếu thời gian cho phép,
  để báo cáo trung bình ± độ lệch.
- **Tiêu chí chọn mô hình (chỉ trên Val):** (1) mAP50-95. (2) Nếu chênh lệch < 0.01 thì xét Recall ở ngưỡng vận hành
  và tỉ lệ bệnh nhân có tổn thương được phát hiện. (3) Nếu vẫn hòa thì chọn mô hình nhỏ hơn (n).
- **Ngưỡng vận hành (confidence)** chọn trên Val bằng `scripts/50_analyze_yolo_run.py` (bảng theo ngưỡng): ưu tiên
  recall cao (bỏ sót tổn thương có hại hơn FP, vì FP còn được nhánh phân loại xử lý), với giới hạn FP trên lát âm sẽ chốt khi có số liệu.

## 4. Hyper-parameter baseline (đã ghi sẵn trong `notebooks/04_train_yolo_*_full_colab.ipynb`)

| Tham số | Giá trị | Lý do |
|---|---|---|
| model | `yolo11n.pt` / `yolo11s.pt` (COCO pretrained) | Transfer learning; dữ liệu y tế nhỏ |
| epochs / patience | **50** / 20 | Early stopping theo fitness = mAP50-95 trên Val. Ban đầu dự kiến 100; người dùng chốt 50 từ MRI-E0 (best epoch 24, dừng ở 44) |
| imgsz | 512 | Độ phân giải gốc của CT (512²); MRI ≤ 512² |
| batch | 32 (16 nếu OOM trên T4) | Ultralytics tích lũy gradient tới nominal batch 64 (`nbs`), nên 16 và 32 cho hiệu quả tương đương |
| optimizer | **SGD**, lr0 0.01, lrf 0.01, momentum 0.937, wd 5e-4, `cos_lr=True` | **Không dùng `auto`**: `auto` chọn AdamW (lr 0.002) khi ≤ 10.000 iteration và MuSGD (lr 0.01) khi > 10.000, nên các run khác độ dài sẽ có optimizer khác nhau và không so sánh được |
| warmup_epochs | 3 | 6% của 50 epoch (ở pilot là 60%) |
| mosaic / close_mosaic | 1.0 / 10 | Mosaic bật trong 90 epoch đầu (pilot không có mosaic) |
| hsv_h / hsv_s | 0 / 0 | Ảnh xám: hue và saturation không có tác dụng |
| hsv_v | 0.3 | Biến đổi độ sáng vừa phải; CT đã windowing cố định, MRI đã chuẩn hóa theo volume |
| fliplr / flipud | 0.5 / 0 | Giữ mặc định ở baseline; E2 kiểm tra `fliplr = 0` |
| degrees / translate / scale | 0 / 0.1 / 0.5 | Mặc định của Ultralytics |
| AMP | bật | Ultralytics tự kiểm tra AMP |

## 5. Tài nguyên và thời gian
- GPU: Colab T4 (16 GB). `yolo11n`/`yolo11s` ở 512 với batch 32 dự kiến vừa VRAM. Nếu OOM thì giảm batch xuống 16.
- **Thời gian/epoch đo từ CT pilot: khoảng 200 giây** trên T4 (25.196 ảnh train + val, yolo11n, 512). Ngoại suy tuyến tính (thô):
  CT train 1/2 + Val đầy đủ (khoảng 49k ảnh) ≈ 6–7 phút/epoch; CT đầy đủ (khoảng 84k ảnh) ≈ 11 phút/epoch; yolo11s chậm hơn n.
  **Đo thực tế MRI-E0** (T4, yolo11n, batch 32, mosaic): **316 s/epoch** cho 36.086 ảnh, tức khoảng 8,8 ms/ảnh. Từ đó CT-E0 (49.500 ảnh)
  ≈ **7,2 phút/epoch**, ≤ khoảng 6 giờ cho 50 epoch. Run có thể vượt một phiên Colab, nên **resume là bắt buộc** (MRI-E0 đã bị ngắt sau epoch 29 và resume thành công).
- Colab có thể ngắt phiên: notebook lưu `last.pt` trên Drive mỗi epoch và resume an toàn (đã kiểm thử: ngắt sau
  epoch 1 rồi resume từ epoch 2; run đã xong thì không resume nhầm).
- Drive (15 GB miễn phí): 2 ZIP pilot ≈ 4.8 GB. ZIP CT đầy đủ ước tính ở mục 2. Checkpoint n/s khoảng 5–20 MB mỗi run.

## 6. Lưu trữ cấu hình và kết quả (phục vụ viết báo cáo tốt nghiệp)

| Ở đâu | Nội dung |
|---|---|
| Drive `runs/yolo_<mod>/<RUN>/` | `args.yaml`, `results.csv`, `weights/{best,last}.pt`, biểu đồ, `metrics_val.json` (cấu hình, phiên bản, SHA256 dataset, GPU) |
| Drive `runs/yolo_<mod>/<RUN>_val_conf{0.001,0.25}/` | Confusion matrix, PR/F1 curve ở 2 ngưỡng |
| Drive `runs/yolo_<mod>/experiments.csv` | Mỗi run một dòng: model, tag, epochs, best epoch, metrics Val, hyper-parameter, SHA256 dataset |
| Local `outputs/runs/yolo_<mod>/<RUN>/` | Bundle tải về và `analysis/` (dự đoán Val, bảng theo ngưỡng) |
| Local `reports/<mod>_<tag>_analysis.md` + `reports/figures/<tag>/` | Báo cáo phân tích, có thể trích vào luận văn |
| Git | Code, notebook, cấu hình, báo cáo (không có dữ liệu, ảnh hay checkpoint) |

## 7. Việc còn lại của Giai đoạn 3
- [x] Tải kết quả CT pilot về `outputs/runs/yolo_ct/pilot_yolo11n_e5`, phân tích (`scripts/50_…`, `scripts/51_ct_fp_fn_breakdown.py`), viết `reports/ct_pilot_analysis.md`.
- [x] MRI pilot: train (Colab), phân tích (`reports/mri_pilot_analysis.md`). Tổng kết hai pilot: `reports/pilot_summary.md`.
- [x] `40_pack_pilot.py --set full --modalities ct` đã có và được kiểm thử trên 2 BN (xuất → QA PASS → ZIP + `SHA256SUMS_full.txt`).
- [x] CT full đã xuất + QA (C1–C7 PASS, 49.500 ảnh) + đóng gói + kiểm tra ZIP (PASS): `reports/full_dataset_report.md`, `yolo_ct_full.zip` 4.16 GB, SHA256 `0a7ed974…1cfdc`.
- [x] **MRI-E0** (yolo11n, 50 epoch, early stop ở 44, best ep 24) đã train trên Colab và phân tích: `reports/mri_e0_analysis.md`
  (Val: P 0.797 · R 0.660 · mAP50 0.749 · mAP50-95 0.504; mức series 501/622 ở conf 0.25). **Không train lại.**
- [x] Notebook 04 (CT và MRI): `EPOCHS = 50`, cảnh báo nếu GPU không phải T4, `last_backup.pt` + resume dự phòng (kiểm thử local PASS, §8.3).
- [x] **Người dùng:** upload `yolo_ct_full.zip` + `SHA256SUMS_full.txt` lên `MyDrive/LiverCancer_AI/data/` (đường dẫn chính xác: §9).
- [x] **Người dùng:** chạy **CT-E0** bằng notebook 04 CT (§8). E1–E4 **không chạy** (người dùng chốt giai đoạn ngày 2026-10-25).
- [x] **CT-E0** (yolo11n, 50 epoch, early stop 35, best ep 15) đã train và phân tích trên toàn Val: `reports/ct_e0_analysis.md`
  (P 0.688 · R 0.481 · mAP50 0.551 · mAP50-95 0.282; FN tập trung ở box nhỏ/vừa, ICC, pha P; 62–67% FP là lỗi định vị).
- [x] Quét ngưỡng trên Val (`scripts/53_threshold_sweep.py`), tiêu chí F2 lớn nhất: **đề xuất conf = 0.12 cho cả CT và MRI**
  (CT: volume phát hiện 91.4%, BN 42/42; MRI: series 85.9%, BN 77/78). So sánh CT/MRI: `reports/ct_e0_analysis.md` §7.8, `reports/figures/yolo_e0_ct_vs_mri.png`.
- [x] **Người dùng chốt (2026-10-25):** cấu hình cuối = E0 cho cả CT và MRI; ngưỡng vận hành **conf 0.12** cho từng modality (chọn trên Val riêng).
- [x] Demo suy luận CT/MRI hoàn thiện và đã kiểm tra độ ổn định: `reports/inference_demo.md`. Tổng kết: **`reports/yolo_detection_summary.md`**.
- [ ] (Chuyển sang giai đoạn đánh giá cuối) Xuất **Test** đầy đủ (mọi lát, không chọn theo GT), đánh giá **một lần**, end-to-end: metrics box, mức lát, mức bệnh nhân.
- [x] Tổng hợp cho báo cáo: `reports/yolo_detection_summary.md` (không có so sánh n/s hay ablation vì đã chốt sớm). **Giai đoạn YOLO Detection: HOÀN THÀNH.**

## 8. Cấu hình đề xuất cho E0 / E1 (sau khi phân tích hai pilot)

Căn cứ: `reports/pilot_summary.md`. Cả hai pilot đều **chưa hội tụ** ở epoch 5 (loss train chưa bão hòa, metrics MRI vẫn tăng đều).
FN tập trung ở **tổn thương nhỏ**; 42–47% FP là **lỗi định vị**.

### 8.1 Dữ liệu
| | E0/E1 dùng | Lệnh tạo (local) | Ước tính |
|---|---|---|---|
| CT | **`yolo_ct_full.zip`**: train lấy **1/2** số lát dương tính (khoảng 17,3k, gấp đôi pilot), **Val đầy đủ** (7.314 lát dương tính), âm:dương 1:1 | `20_build_yolo_pilot.py --modality ct --pos-stride-train 2 --pos-stride-val 1 --out data\processed\full\yolo_ct`, rồi `30_qa_pilot.py --modality ct --data data\processed\full\yolo_ct`, rồi `40_pack_pilot.py --set full --modalities ct` | Khoảng 4.1 GB, xuất khoảng 25–35 phút |
| MRI | **`yolo_mri_pilot.zip`** (dùng lại; đã chứa 93% số lát hợp lệ và Val đầy đủ) | không cần | 2.68 GB (đã có trên Drive) |

- Lý do chọn CT 1/2 thay vì đầy đủ (khoảng 7 GB): các lát CT liền kề rất giống nhau; 1/2 đã gấp đôi dữ liệu dương tính của pilot,
  vừa Drive 15 GB (cùng hai ZIP pilot khoảng 4.8 GB), và mỗi epoch nhanh hơn khoảng 40%. Nếu E0 CT cho thấy recall còn tăng theo dữ liệu, cân nhắc bộ đầy đủ ở E4.
- Val CT của bộ full **khác** Val pilot (đầy đủ thay vì 1/2), nên metrics E0/E1 CT **không so trực tiếp** với CT pilot.

### 8.2 Các run (notebook `04_train_yolo_<mod>_full_colab.ipynb`, mục 2, chỉ đổi `MODEL`)
| Run | Notebook | `ZIP_NAME` | `MODEL` | `TAG` | `RUN_NAME` sinh ra |
|---|---|---|---|---|---|
| **CT-E0** | 04 CT | `yolo_ct_full.zip` | `yolo11n.pt` | `baseline` | `yolo11n_e50_img512_baseline` |
| **CT-E1** | 04 CT | `yolo_ct_full.zip` | `yolo11s.pt` | `baseline` | `yolo11s_e50_img512_baseline` |
| **MRI-E0** | 04 MRI | `yolo_mri_pilot.zip` | `yolo11n.pt` | `baseline` | `yolo11n_e50_img512_baseline` |
| **MRI-E1** | 04 MRI | `yolo_mri_pilot.zip` | `yolo11s.pt` | `baseline` | `yolo11s_e50_img512_baseline` |

Hyper-parameter chung (đã ghi sẵn trong notebook, **không sửa** giữa E0 và E1 để so sánh công bằng):
`EPOCHS=50, PATIENCE=20, IMGSZ=512, BATCH=32 (16 nếu OOM, ghi lại vào TAG), SEED=42, amp=True`,
`optimizer=SGD, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, cos_lr=True, warmup_epochs=3`,
`mosaic=1.0, close_mosaic=10, mixup=0, fliplr=0.5, flipud=0, degrees=0, translate=0.1, scale=0.5, hsv_h=0, hsv_s=0, hsv_v=0.3`.

- Thứ tự gợi ý: MRI-E0 (dữ liệu đã có sẵn trên Drive) → CT-E0 (sau khi upload CT full) → MRI-E1 → CT-E1.
- Thời gian: MRI-E0 **đo được** 316 s/epoch, 44 epoch ≈ 3,9 giờ. CT-E0 ước tính khoảng 7,2 phút/epoch (≤ khoảng 6 giờ cho 50 epoch;
  ngoại suy theo số ảnh). yolo11s chậm hơn, **chưa đo**. Early stopping thường dừng sớm hơn.
  Một run có thể vượt một phiên Colab, nên khi bị ngắt hãy làm theo mục *Resume* của notebook.
- Sau mỗi run: tải `<RUN_NAME>_bundle.zip` về `outputs\runs\yolo_<mod>\` và chạy `50_analyze_yolo_run.py`
  (thêm `51_ct_fp_fn_breakdown.py --modality <mod>`). `--data` phải trỏ tới **đúng bộ dữ liệu của run**
  (CT: `data\processed\full\yolo_ct`).
- **Tiêu chí chọn E0 hay E1** (chỉ trên Val): mAP50-95; nếu chênh lệch < 0.01 thì xét recall ở ngưỡng vận hành và tỉ lệ BN được phát hiện,
  rồi đến kích thước mô hình (§3).

### 8.3 Kiểm tra mức sẵn sàng (đã xác minh)
| Yêu cầu | Trạng thái | Bằng chứng |
|---|---|---|
| Chuẩn bị dữ liệu đầy đủ (export → QA → ZIP → SHA256) | ✅ có script, đã test trên 2 BN | `40_pack_pilot.py --set full`; QA C1–C6 PASS; `SHA256SUMS_full.txt` |
| Notebook kiểm tra SHA256 của cả pilot và full | ✅ | Đọc mọi `SHA256SUMS*.txt` trên Drive |
| Checkpoint lưu trực tiếp lên Drive | ✅ | `project = RUN_PROJECT` trên Drive; `last.pt` và `best.pt` mỗi epoch (`save_period=-1`: không lưu thêm epochN.pt) |
| Resume khi mất kết nối | ✅ đã kiểm thử local; ✅ MRI-E0 trên Colab | Ngắt sau epoch 1 → resume từ epoch 2; run đã xong thì không resume nhầm (không tải coco8). MRI-E0 thực tế: ngắt sau epoch 29, resume từ epoch 30 đến early stop ở 44 |
| `last.pt` hỏng do ngắt giữa lúc ghi lên Drive | ✅ đã kiểm thử local (2026-10-24) | Callback `on_model_save` chép `last.pt` → `last_backup.pt` mỗi epoch; khi `RESUME=True` mà `last.pt` không đọc được thì resume từ `last_backup.pt`. Test: ngắt sau epoch 2, cắt đôi `last.pt` → resume từ backup, chạy tiếp epoch 3, kết thúc bình thường; chạy lại thì báo "already finished" |
| Không ghi đè run cũ | ✅ | `assert` khi `RUN_NAME` đã tồn tại mà `RESUME = False` |
| Metrics lên Drive | ✅ | `metrics_val.json` (conf 0.001 và 0.25, cấu hình, SHA256 dataset, phiên bản), `experiments.csv` (mỗi run 1 dòng), `<RUN>_bundle.zip` |
| Không dùng Test | ✅ | ZIP không có `images/test`; notebook `assert 'test' not in cfg`; test tự động cấm `split='test'` |

## 9. Upload dữ liệu CT-E0 lên Google Drive (người dùng tự upload)

| File local (đường dẫn chính xác) | Đích trên Drive | Kích thước | SHA256 |
|---|---|---|---|
| `E:\LiverCancer_AI_Graduation\data\processed\full\yolo_ct_full.zip` | `MyDrive/LiverCancer_AI/data/yolo_ct_full.zip` | 4.163.315.100 byte (4,16 GB) | `0a7ed974856f5634b2860b692b83fa717e820031d8974862607a78a5e281cfdc` |
| `E:\LiverCancer_AI_Graduation\data\processed\full\SHA256SUMS_full.txt` | `MyDrive/LiverCancer_AI/data/SHA256SUMS_full.txt` | 170 byte | (chứa SHA của ZIP và `split_manifest.csv`) |
| `E:\LiverCancer_AI_Graduation\notebooks\04_train_yolo_ct_full_colab.ipynb` | mở bằng Colab (*File → Upload notebook*) hoặc lưu ở `MyDrive/LiverCancer_AI/notebooks/` | | |

- SHA256 của ZIP đã được **tính lại ngày 2026-10-24** và khớp `SHA256SUMS_full.txt`; `42_verify_zip.py` PASS (`data/processed/full/yolo_ct_full.verify.json`).
- **Không** cần upload `split_manifest.csv` (đã nằm trong ZIP) hay thư mục `data/processed/full/yolo_ct/`.
- Nếu trên Drive đã có `SHA256SUMS_full.txt` cũ (ví dụ từ lần thử 2 BN), hãy **thay bằng bản này**. Notebook đọc mọi `SHA256SUMS*.txt` và sẽ dừng nếu SHA không khớp.
- Dung lượng Drive: CT full 4,16 GB + MRI pilot 2,68 GB + CT pilot 2,11 GB ≈ 9 GB (vừa 15 GB). Có thể xóa `yolo_ct_pilot.zip` trên Drive nếu thiếu chỗ (bản local vẫn còn).
- Trên Colab: mở notebook, *Runtime → Change runtime type → T4 GPU*, giữ mặc định ở mục 2 (`ZIP_NAME = 'yolo_ct_full.zip'`, `MODEL = 'yolo11n.pt'`, `TAG = 'baseline'`,
  `RESUME = False`), sẽ sinh `RUN_NAME = yolo11n_e50_img512_baseline`. Chạy từ trên xuống. Nếu bị ngắt: đặt `RESUME = True`, chạy lại cell 1→5 rồi cell 6.
- Sau khi xong: tải `MyDrive/LiverCancer_AI/runs/yolo_ct/yolo11n_e50_img512_baseline_bundle.zip`, giải nén vào
  `E:\LiverCancer_AI_Graduation\outputs\runs\yolo_ct\` (tạo thư mục `yolo11n_e50_img512_baseline`).

## 10. Danh sách việc sau khi CT-E0 train xong

Định nghĩa và tiêu chí hoàn thành của GĐ3/GĐ4: `docs/IMPLEMENTATION_PLAN.md`. Thứ tự dưới đây đi từ chốt detector đến ROI rồi đến classifier.

### 10.1 Chốt YOLO (kết thúc "Giai đoạn 3: YOLO")
1. Phân tích CT-E0 (local, chỉ Val, `--data` phải là bộ full):
   `50_analyze_yolo_run.py --modality ct --run outputs\runs\yolo_ct\yolo11n_e50_img512_baseline --data data\processed\full\yolo_ct --tag ct_e0`,
   `51_ct_fp_fn_breakdown.py --run … --data data\processed\full\yolo_ct` (FP trong/ngoài gan, FP trên BN CN),
   `52_series_level_detection.py --modality ct --run … --data data\processed\full\yolo_ct` (mức volume-pha và BN). Viết `reports/ct_e0_analysis.md`.
   Val CT full khác Val pilot, nên **không so trực tiếp** với CT pilot.
2. E1 (yolo11s) cho CT và MRI với cùng cấu hình; chọn n hay s **trên Val** theo §3. Nếu thiếu thời gian GPU, E0 là cấu hình cuối.
3. Chọn **ngưỡng vận hành trên Val** cho từng modality, ưu tiên recall mức series/volume (MRI-E0: conf 0.1 → 86% series, 0.25 → 80.5%),
   có giới hạn FP trên lát âm. Ghi vào `configs/yolo_final.yaml` (weights + SHA256, conf, iou NMS, imgsz).
4. **Viết script xuất Test đầy đủ** (`20_build_yolo_pilot.py` hiện chỉ cho phép train/val): mọi lát của mọi volume/series Test, không chọn theo GT,
   đánh giá **một lần** cho cấu hình cuối: box (P/R/mAP), mức lát, mức series/volume, mức BN. Với CT: FP trên BN CN.
5. Bảng tổng hợp YOLO cho luận văn: pilot → E0 → E1, ngưỡng, Test (một lần), kèm đường dẫn file nguồn.

### 10.2 GĐ3: Automatic Lesion ROI Zoom
1. **Inference streaming** từ NIfTI (cùng tiền xử lý GĐ2, từng volume, không đọc cả dataset vào RAM) bằng `best.pt` đã chốt,
   lưu mọi box ≥ ngưỡng thấp vào `outputs/predictions/<mod>/<split>.csv` (BN, pha/chuỗi, lát, box, conf).
2. **Hậu xử lý:** NMS, nối box qua các lát liền kề (IoU 2D) thành ứng viên tổn thương 3D, chọn top-k lát theo diện tích × conf.
   MRI-E0 cho thấy tổn thương lớn thường bị bao thiếu, nên **margin tỉ lệ theo kích thước box** và có thể lấy box hợp qua các lát.
3. **Crop:** local (box + margin 10–20%) và context (×2–3 quanh tâm), resize 224, xám → 3 kênh. Ghi manifest
   (`source = yolo|gt`, conf, IoU với GT nếu có).
   CT: kiểm tra crop context có dính **chữ thước đo in chồng** không (confounder, CLAUDE.md).
4. **Bộ ROI oracle** (từ GT) cho train/val và để báo cáo riêng, gắn nhãn **"oracle"**. Test end-to-end **chỉ** dùng crop từ YOLO.
   MRI: series trong `mri_bbox_quarantine.csv` không dùng để huấn luyện ROI.
5. **Thống kê ROI trên Val:** % series/volume có tổn thương được tạo ROI đúng, IoU ROI–GT, tỉ lệ ROI từ FP, ROI trên BN CN (CT).
   YOLO bỏ sót thì tính là lỗi của pipeline.
6. Đóng gói ROI cho Colab (ZIP + SHA256; MRI CC BY-NC: thư mục Drive riêng tư, người dùng tự upload).

### 10.3 GĐ4: Hybrid CNN + Transformer
1. Nhãn: CT 3 lớp HCC/ICC/CHCC (CN không vào classifier); MRI 7 lớp + nhóm benign/malignant (`Category_info`). Nhãn mức BN, lấy từ CSV/JSON.
2. Baseline trước, hybrid sau: ResNet18/34 trên crop local; Swin-T/ViT-S trên crop context; Hybrid concat + MLP. Ablation gated/cross-attention ở GĐ5.
3. Loss CE có class weight (hoặc focal), early stopping theo **macro-F1 trên Val**. Smoke test local (batch ≤ 8, AMP, accumulation), train chính trên Colab T4.
4. Gộp mức BN: trung bình xác suất qua lát, pha (CT) và chuỗi (MRI).
5. Báo cáo tách riêng **end-to-end** (YOLO → ROI → cls) và **oracle-ROI**. Test chạy một lần cho cấu hình cuối. XAI (Grad-CAM, attention rollout) ở GĐ5.

