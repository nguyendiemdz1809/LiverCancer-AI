# Báo cáo chia Train / Validation / Test (GĐ2a)

Sinh bởi `scripts/10_make_splits.py --seed 42`. Split ở **mức bệnh nhân** và đã **đóng băng** (chạy lại cần `--force` và sự đồng ý của người dùng).

## CT — PLC-CECT (70/15/15 phân tầng theo `cancer_type`)

| label | train | val | test | Tổng | %Val | %Test |
|---|---|---|---|---|---|---|
| CHCC | 59 | 13 | 13 | 85 | 15.3 | 15.3 |
| CN | 59 | 12 | 12 | 83 | 14.5 | 14.5 |
| HCC | 66 | 14 | 14 | 94 | 14.9 | 14.9 |
| ICC | 69 | 15 | 15 | 99 | 15.2 | 15.2 |
| **Tổng** | 253 | 54 | 54 | 361 | 15.0 | 15.0 |

| Split | Nam/Nữ/NA | Tuổi TB |
|---|---|---|
| train | 138/112/3 | 55.8 |
| val | 32/22/0 | 56.9 |
| test | 30/24/0 | 55.6 |

## MRI — LLD-MMRI-MedSAM2 (phương án A)

Val = 78 ID chính thức của LLD-MMRI2023. 420 BN còn lại chia phân tầng theo lớp thành Train 316 / Test 104. **Test này không phải test chính thức của challenge.**

| label | train | val | test | Tổng | %Val | %Test |
|---|---|---|---|---|---|---|
| FOCAL_NODULAR_HYPERPLASIA | 29 | 7 | 10 | 46 | 15.2 | 21.7 |
| Hepatic_abscess | 35 | 8 | 11 | 54 | 14.8 | 20.4 |
| Hepatic_cyst | 34 | 8 | 11 | 53 | 15.1 | 20.8 |
| Hepatic_hemangioma | 50 | 13 | 16 | 79 | 16.5 | 20.3 |
| Hepatic_metastasis | 32 | 8 | 11 | 51 | 15.7 | 21.6 |
| Hepatocellular_carcinoma | 99 | 25 | 33 | 157 | 15.9 | 21.0 |
| Intrahepatic_cholangiocarcinoma | 37 | 9 | 12 | 58 | 15.5 | 20.7 |
| **Tổng** | 316 | 78 | 104 | 498 | 15.7 | 20.9 |

| group | train | val | test | Tổng | %Val | %Test |
|---|---|---|---|---|---|---|
| Benign | 148 | 36 | 48 | 232 | 15.5 | 20.7 |
| Malignant | 168 | 42 | 56 | 266 | 15.8 | 21.1 |
| **Tổng** | 316 | 78 | 104 | 498 | 15.7 | 20.9 |

