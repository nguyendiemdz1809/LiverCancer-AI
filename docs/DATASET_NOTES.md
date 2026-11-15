# Ghi chú dataset (khảo sát ngày 2026-10-19, chỉ đọc)

## CT — PLC-CECT

- Gốc: `E:\LiverDataset\CT\PLC-CECT` (tải từ Hugging Face, có thư mục `.cache/huggingface`).
- License: CC BY 4.0 (`CECT/LICENSE`).
- `CECT/structure.md`: tên pha `{C1, C2, C3, P}`; "Some patients may not have all phases; masks follow availability of the corresponding CT."

### patient_data.csv
- Cột: `patient_id, age, gender, phase, cancer_type, ct_path, mask_path, liver_mask_path`.
- Đường dẫn là **tương đối so với `CECT/`**, ví dụ `ct_files/P0001_ct_P.nii.gz`.
- 1444 dòng, 361 bệnh nhân, mỗi bệnh nhân 4 dòng (C1, C2, C3, P). Mỗi bệnh nhân có đúng 1 `cancer_type`.
- `gender`: male 800, female 632, `NA` 12 (tính theo dòng). Tuổi từ 21 đến 86.

| cancer_type | Bệnh nhân | Dòng | Có mask tổn thương |
|---|---|---|---|
| ICC | 99 | 396 | có |
| HCC | 94 | 376 | có |
| CHCC | 85 | 340 | có |
| CN | 83 | 332 | **không** (`mask_path` rỗng) |

### Đối chiếu file
| Thư mục | Số file | Dung lượng | Ghi chú |
|---|---|---|---|
| `CECT/ct_files` | 1404 | 37.6 GB | **Thiếu 40** so với CSV |
| `CECT/liver_mask_files` | 1444 | 0.42 GB | đủ |
| `CECT/mask_files` | 1112 | 0.11 GB | = 1444 − 332 (CN) |

- 40 CT thiếu: CHCC 16, ICC 9, HCC 8, CN 7 (ví dụ `P0004_ct_C3`, `P0005_ct_C3`, `P0037_ct_C1`, ...). HF cache chỉ có 1404 file `.metadata` cho ct_files, nghĩa là các file này chưa từng được tải. Danh sách đầy đủ: `reports/dataset_audit.json`.
- Không có file nào trên đĩa nằm ngoài CSV.

### Header NIfTI (mẫu: P0202_ct_P, P0222_ct_C1, P0022_ct_C1)
- shape 512×512×{117, 118, 200}; zooms (1.0, 1.0, 1.0); orientation LPS; **dtype uint8**; scl_slope/inter = NaN; descrip "Anonymized".
- Kết luận: cường độ **không phải HU** (đã được lượng tử hóa sang 0–255). Spacing 1 mm đẳng hướng có thể do đã resample hoặc do header bị mất thông tin, cần xác minh.

### Cần xác minh (GĐ1)
- [x] Tiền xử lý CT: bài báo clip về [−200, 200] HU (WL 0 / WW 400); dữ liệu tải về là uint8 0–255, không còn HU. Không resample; spacing trong header (1×1×1) không đáng tin.
- [x] `CN` = nhóm chứng, *"83 individuals without hepatobiliary disease"*, *"no lesion data were collected"* (Luo et al., Sci Data 2025).
- [x] `CHCC` = Combined Hepatocellular-Cholangiocarcinoma (cHCC-CCA, M81800/3). Chẩn đoán qua mô học hoặc tế bào học.
- [~] Pha: bài báo dùng Plain/Arterial/Venous/Delayed nhưng **không** có mapping sang tên file. Repo tác giả có một ví dụ C1 = arterial; overlay P0320 mâu thuẫn. **Chưa xác nhận.**
- [x] Luo et al., *Sci Data* 12:768 (2025), doi 10.1038/s41597-025-05125-2; dữ liệu ở Science Data Bank, doi 10.57760/sciencedb.12207; code ở github.com/ljwa2323/PLC_CECT.

## MRI — LLD-MMRI-MedSAM2

- Gốc: `E:\LiverDataset\LLD-MMRI-MedSAM2` (HF `wanglab/LLD-MMRI-MedSAM2`).
- README: 498 bệnh nhân, 8 chuỗi MRI mỗi tổn thương, 3984 case; annotation bằng MedSAM2 theo kiểu human-in-the-loop. Cần trích dẫn cả LLD-MMRI (SDR-Former, Neural Networks 2025) và MedSAM2.

### LLD_MMRI_Annotation.json (19 MB)
```
{
  "Annotation_info": { "<patient_id>": [ 8 × {
      "studyUID", "seriesUID", "pixel_spacing":[x,y], "slice_spacing", "slice_thickness",
      "origin":[x,y,z], "phase": "T2WI|DWI|C-pre|C+A|C+V|C+Delay|In Phase|Out Phase",
      "annotation": { "num_targets": 1,
        "lesion": { "0": { "category": int,
          "bbox": { "2D_box": [ {slice_idx, x_min, y_min, x_max, y_max, area}, ... ],
                    "3D_box": {x_min, y_min, x_max, y_max, z_min, z_max, area} } } } } } ] },
  "Category_info": { "Hepatic_hemangioma":0, "Intrahepatic_cholangiocarcinoma":1, "Hepatic_abscess":2,
                     "Hepatic_metastasis":3, "Hepatic_cyst":4, "FOCAL_NODULAR_HYPERPLASIA":5,
                     "Hepatocellular_carcinoma":6, "Benign":[0,2,4,5], "Malignant":[1,3,6] }
}
```
- Mọi series có `num_targets = 1`. Mỗi bệnh nhân có 1 lớp duy nhất, nhất quán giữa 8 series.

| Lớp | id | Bệnh nhân | Nhóm |
|---|---|---|---|
| Hepatocellular_carcinoma | 6 | 157 | Malignant |
| Hepatic_hemangioma | 0 | 79 | Benign |
| Intrahepatic_cholangiocarcinoma | 1 | 58 | Malignant |
| Hepatic_abscess | 2 | 54 | Benign |
| Hepatic_cyst | 4 | 53 | Benign |
| Hepatic_metastasis | 3 | 51 | Malignant |
| FOCAL_NODULAR_HYPERPLASIA | 5 | 46 | Benign |

### File
- `images/`: 3984 file (17.3 GB), tên `MR-xxx_{cat}_{seq}_0000.nii.gz`. `{cat}` trùng với category id; `{seq}` ∈ {T2WI, DWI, C-pre, C+A, C+V, C+Delay, InPhase, OutPhase}.
- `labels/`: 3984 file mask, tên `MR-xxx_{cat}_{seq}.nii.gz`.
- 498 ID trong JSON khớp 100% với ID trong `images/` (mỗi ID có 8 file).
- Header: **dtype uint8**, LPS. `pixel_spacing` và `slice_spacing` trong JSON **khớp** với zooms trong header (đã kiểm tra MR55599, MR176861, MR-391135).
- Độ phân giải khác nhau giữa các chuỗi trong cùng một bệnh nhân (ví dụ MR55599: T2WI 384×384×31, DWI 192×192×31, Out Phase 352×352×31), nên bbox phải lấy theo đúng chuỗi.

### Cần xác minh (GĐ1)
- [~] Split: challenge 316/78/104. Suy ra 78 ID val từ `LLDBaseline.json`, khớp 78/78 với bản MedSAM2. Train/test **không công khai**; tác giả khuyến nghị 5-fold CV.
- [x] License: CC BY-NC 4.0; *"not to further copy, publish, or distribute any part"*. Upload lên Drive cần người dùng quyết định.
- [x] Hệ trục: x ↔ i, y ↔ j, slice_idx ↔ k (IoU 0.58–0.96 so với khoảng 0 khi đảo trục; đã xác nhận trên overlay 5 BN × 8 chuỗi).
- [x] Label nhị phân {0, 1} (trên mẫu; số liệu toàn bộ xem `reports/audit_report.md`).
