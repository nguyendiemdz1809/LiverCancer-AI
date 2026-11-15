# Pipeline AI trên Validation: YOLO → Tumor Zoom → phân loại

*Sinh bởi `scripts/91_pipeline_val_demo.py` lúc 2026-10-26 10:57. Số liệu: `outputs/predictions/pipeline_val/summary.json`, các run ở `outputs/runs/cls/`. Hình: `reports/figures/pipeline/`. **Chỉ Validation; Test không được dùng. Không huấn luyện lại.***

## 1. Kiểm tra 4 run phân loại (bundle từ Colab)

| Run | Mô hình | Tham số | Best / dừng ở epoch | Val ROI acc | Val ROI macro-F1 | Val BN macro-F1 (ROI-YOLO) | Gộp BN | Train acc / Val acc tại best | Epoch val loss nhỏ nhất | Khớp experiments.csv | Đánh giá lại ở local trùng |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ct_hybrid_gt_s42 | hybrid | 39.2 M | 9 / 17 | 0.8623 | 0.8629 | 0.9020 | mean_prob | 0.982 / 0.862 | 17 | ✅ | ✅ |
| ct_resnet18_gt_s42 | resnet18 | 11.2 M | 23 / 31 | 0.8356 | 0.8334 | 0.9051 | top3_mean | 0.996 / 0.836 | 23 | ✅ | ✅ |
| ct_swin_t_gt_s42 | swin_t | 27.5 M | 4 / 12 | 0.8518 | 0.8519 | 0.9034 | mean_prob | 0.964 / 0.852 | 4 | ✅ | ✅ |
| mri_hybrid_gt_s42 | hybrid | 39.2 M | 10 / 18 | 0.4784 | 0.4282 | 0.3579 | conf_weighted | 0.973 / 0.478 | 1 | ✅ | ✅ |

Mọi `best.pt` nạp được (`livercancer.classification.evaluate.load_classifier`). Dữ liệu huấn luyện: CT 6.026 ROI-GT / 194 BN, MRI 11.230 / 316 BN; Val CT 1.046 / 42 BN, MRI 2.575 / 78 BN. SHA256 dữ liệu trong `experiments.csv` = SHA của `cls_ct.zip` / `cls_mri.zip` đã đóng gói.

## 2. Hạn chế của Hybrid MRI (Val acc 47.8%, macro-F1 42.8%): số liệu thật

| Lớp | Precision | Recall | Specificity | F1 | ROI Val | ROI Train (GT) | BN Val |
|---|---|---|---|---|---|---|---|
| HCC | 0.546 | 0.654 | 0.699 | 0.595 | 919 | 3833 | 25 |
| ICC | 0.373 | 0.301 | 0.925 | 0.333 | 332 | 1467 | 9 |
| Di căn | 0.243 | 0.145 | 0.953 | 0.182 | 241 | 1100 | 8 |
| U máu | 0.340 | 0.414 | 0.878 | 0.373 | 338 | 1380 | 13 |
| Nang | 0.489 | 0.463 | 0.961 | 0.476 | 190 | 842 | 8 |
| Áp xe | 0.559 | 0.335 | 0.964 | 0.419 | 310 | 1532 | 8 |
| FNH | 0.575 | 0.669 | 0.948 | 0.619 | 245 | 1076 | 7 |

Ma trận nhầm lẫn ROI-GT Val (hàng = nhãn thật, cột = dự đoán; thứ tự HCC, ICC, Di căn, U máu, Nang, Áp xe, FNH):

```
HCC      601   69   31  112    4   33   69
ICC      185  100    7   23    9    6    2
Di căn    57   36   35   48   20   31   14
U máu     84   12   24  140   48   11   19
Nang      25   10   30   31   88    0    6
Áp xe    120   37    0   30    8  104   11
FNH       28    4   17   28    3    1  164
```

- **Overfitting rất mạnh:** train acc 0.51 → 0.91 → 0.96 → 0.99 → 0.99; val acc 0.44 → 0.42 → 0.47 → 0.44 → 0.46. Val loss nhỏ nhất ở epoch 1 (1.642) rồi tăng, trong khi train loss giảm liên tục.
- **HCC là lớp 'hút' dự đoán:** nhiều ROI ICC và áp xe bị đoán là HCC (xem ma trận). HCC có nhiều ROI Train nhất (3.833).
- **Di căn** kém nhất (F1 thấp nhất); FNH và HCC tốt nhất.
- **Theo chuỗi MRI** (ROI acc): C+A 0.47, C+Delay 0.53, C+V 0.51, C-pre 0.49, DWI 0.46, In Phase 0.41, Out Phase 0.48, T2WI 0.45. Một mô hình duy nhất cho cả 8 chuỗi có đặc điểm hình ảnh rất khác nhau.
- **Gộp lành tính / ác tính** (suy ra từ dự đoán 7 lớp, chưa huấn luyện riêng): ROI 70.4%, bệnh nhân 74.4%.
- **ROI do YOLO** làm giảm thêm: ROI acc 35.8% với mọi ROI-YOLO (gồm vùng giả) so với 49.0% với ROI-YOLO khớp GT.
- **Val nhỏ:** 7–25 BN mỗi lớp, nên chỉ số theo lớp dao động lớn.
- *Giả thuyết, chưa kiểm chứng:* (1) mỗi ROI được phân loại độc lập theo từng chuỗi, mô hình không kết hợp thông tin đa pha / đa chuỗi của cùng tổn thương, trong khi chẩn đoán MRI gan thường dựa vào mẫu ngấm thuốc qua các pha; (2) mô hình 39 M tham số với khoảng 11 nghìn ROI rất tương quan (nhiều lát / chuỗi của cùng tổn thương) dễ học thuộc. Hướng thử trên Val: fusion đa chuỗi ở mức tổn thương, augmentation mạnh hơn, mô hình nhỏ hơn, phân loại lành / ác tính.

## 3. Pipeline và tính nhất quán tiền xử lý

`livercancer.pipeline.LiverCancerPipeline`: NIfTI → `load_volume_uint8` (giống lúc train YOLO) → YOLO (conf 0.12) → gộp box → track 3D → tối đa 5 lát đại diện mỗi vùng (diện tích × độ tin cậy, cách nhau ≥ 2 lát CT / ≥ 1 lát MRI, đúng như khi tạo ROI-YOLO của bộ dữ liệu) → crop local / context 224×224 → chuẩn hóa ImageNet → mô hình phân loại → gộp mức ca theo cách đã chọn trên Val.

| Kiểm tra (Val, chạy lại pipeline) | CT | MRI |
|---|---|---|
| Volume kiểm tra | 12 | 12 |
| ROI cùng vùng và cùng lát với bộ dữ liệu | 226 | 115 |
| Tổng ROI (hợp hai phía) | 226 | 115 |
| Tỉ lệ crop trùng từng pixel | 1.0 | 1.0 |
| Sai khác pixel trung bình lớn nhất | 0.0 | 0.0 |
| Sai khác xác suất lớn nhất (crop bộ dữ liệu so với crop pipeline) | 0.0 | 0.0 |

## 4. Ca minh họa trên Validation: pipeline (ROI do YOLO) so với ROI từ GT (oracle)

Mỗi ca chạy **mọi pha / chuỗi có sẵn** của bệnh nhân; nhãn là dự đoán gộp mức ca. ROI-GT là kết quả **oracle** (dùng nhãn chuẩn để cắt vùng), chỉ để so sánh, không phải kết quả của pipeline. Mẫu nhỏ chọn ngẫu nhiên (seed cố định), **không** phải đánh giá.

| Modality | BN | Nhãn thật | Số volume | Số vùng YOLO | Pipeline (ROI-YOLO) | Đúng? | ROI-GT (oracle) | Đúng? | Thời gian (s) | Hình |
|---|---|---|---|---|---|---|---|---|---|---|
| CT | P0058 | HCC | 4 | 49 | HCC (0.908) | ✅ | HCC (0.836) | ✅ | 6.6 | [hình](figures/pipeline/ct_P0058.png) |
| CT | P0014 | HCC | 4 | 20 | HCC (0.659) | ✅ | HCC (0.846) | ✅ | 1.33 | [hình](figures/pipeline/ct_P0014.png) |
| CT | P0096 | ICC | 3 | 45 | ICC (0.873) | ✅ | ICC (0.956) | ✅ | 6.66 | [hình](figures/pipeline/ct_P0096.png) |
| CT | P0168 | ICC | 4 | 34 | ICC (0.809) | ✅ | ICC (0.935) | ✅ | 3.53 | [hình](figures/pipeline/ct_P0168.png) |
| CT | P0224 | CHCC | 4 | 13 | CHCC (0.855) | ✅ | CHCC (0.776) | ✅ | 2.06 | [hình](figures/pipeline/ct_P0224.png) |
| CT | P0218 | CHCC | 4 | 16 | CHCC (0.88) | ✅ | CHCC (0.961) | ✅ | 4.09 | [hình](figures/pipeline/ct_P0218.png) |
| CT | P0301 | CN | 4 | 3 | ICC (0.506) | ❌ | None (None) | — | 6.91 | [hình](figures/pipeline/ct_P0301.png) |
| MRI | MR78507 | Hepatocellular_carcinoma | 8 | 25 | Hepatocellular_carcinoma (0.859) | ✅ | Hepatocellular_carcinoma (0.917) | ✅ | 1.57 | [hình](figures/pipeline/mri_MR78507.png) |
| MRI | MR174815 | Intrahepatic_cholangiocarcinoma | 8 | 50 | Hepatocellular_carcinoma (0.638) | ❌ | Hepatocellular_carcinoma (0.755) | ❌ | 2.16 | [hình](figures/pipeline/mri_MR174815.png) |
| MRI | MR127280 | Hepatic_metastasis | 8 | 72 | Hepatocellular_carcinoma (0.393) | ❌ | Hepatocellular_carcinoma (0.358) | ❌ | 2.22 | [hình](figures/pipeline/mri_MR127280.png) |
| MRI | MR201172 | Hepatic_hemangioma | 8 | 16 | Hepatic_hemangioma (0.422) | ✅ | Hepatic_metastasis (0.325) | ❌ | 1.38 | [hình](figures/pipeline/mri_MR201172.png) |
| MRI | MR140376 | Hepatic_cyst | 8 | 35 | Hepatic_hemangioma (0.474) | ❌ | Hepatic_hemangioma (0.898) | ❌ | 1.66 | [hình](figures/pipeline/mri_MR140376.png) |
| MRI | MR145114 | Hepatic_abscess | 8 | 47 | Intrahepatic_cholangiocarcinoma (0.251) | ❌ | Hepatic_abscess (0.627) | ✅ | 1.79 | [hình](figures/pipeline/mri_MR145114.png) |
| MRI | MR-470019 | FOCAL_NODULAR_HYPERPLASIA | 8 | 39 | FOCAL_NODULAR_HYPERPLASIA (0.658) | ✅ | FOCAL_NODULAR_HYPERPLASIA (0.431) | ✅ | 2.38 | [hình](figures/pipeline/mri_MR-470019.png) |

- CT: BN nhóm chứng CN **luôn** nhận một nhãn ung thư nếu YOLO có vùng (mô hình không có lớp 'không tổn thương'). Đây là hạn chế của thiết kế 3 lớp.
- MRI: mọi nhãn là **không đáng tin cậy** (cảnh báo đỏ trên hình và trong `warnings`).

## 5. Trường hợp YOLO không phát hiện vùng nào

| Modality | Loại volume | Số volume Val như vậy | Ví dụ | Trạng thái | Nhãn trả về | Hình |
|---|---|---|---|---|---|---|
| CT | có tổn thương (GT) | 1 | P0133_C3 | no_lesion_detected | None | [hình](figures/pipeline/ct_nodetect_P0133_C3.png) |
| CT | không có tổn thương annotation (CN) | 11 | P0301_C3 | no_lesion_detected | None | [hình](figures/pipeline/ct_nodetect_P0301_C3.png) |
| MRI | có tổn thương (GT) | 1 | MR201172_Out Phase | no_lesion_detected | None | [hình](figures/pipeline/mri_nodetect_MR201172_OutPhase.png) |
| MRI | không có ROI-GT | 0 | — | — | None | — |

Thông điệp trả về: *“YOLO không phát hiện vùng nghi ngờ nào ở ngưỡng tin cậy 0.12. Điều này KHÔNG có nghĩa là bệnh nhân không mắc ung thư: mô hình có thể bỏ sót tổn thương (nhất là tổn thương nhỏ). Cần bác sĩ đánh giá trên ảnh gốc.”*

## 6. Hạn chế và việc tiếp theo
- Mọi chỉ số trên là **Validation**; mô hình (CT hybrid) và cách gộp được chọn trên chính Val này, nên còn lạc quan. Test chỉ đánh giá một lần ở cuối.
- CT Val chỉ 42 BN: 3 mô hình CT có cùng độ chính xác mức BN (38/42); chênh lệch macro-F1 ROI 0.83–0.86 là nhỏ.
- MRI chưa đạt chất lượng; pipeline vẫn chạy được nhưng nhãn chỉ để minh họa.
- Chưa tích hợp vào website (theo yêu cầu). Interface đã sẵn: `LiverCancerPipeline.run_*()` → `PipelineResult.as_dict()`.
