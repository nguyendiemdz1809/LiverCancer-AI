# Chỉ số bổ sung theo đề cương

Đầu ra đã lưu của lần chạy Test duy nhất (26/10/2026) và dự đoán Validation; không chạy lại mô hình.

## AUC mức vùng, CT Validation

- ResNet-18: macro 0.933 | HCC 0.943, ICC 0.926, CHCC 0.931
- Swin-T: macro 0.949 | HCC 0.951, ICC 0.943, CHCC 0.952
- Hybrid: macro 0.942 | HCC 0.941, ICC 0.942, CHCC 0.942

## AUC mức bệnh nhân, CT Test

- Pipeline (40 BN có nhãn): macro 0.991 | HCC 1.000, ICC 0.997, CHCC 0.974
- Oracle (42 BN): macro 0.989

## MRI Test lành/ác

{"n": 104, "n_malignant": 56, "n_benign": 48, "tp": 51, "tn": 24, "fp": 24, "fn": 5, "accuracy": 0.7211538461538461, "sensitivity_malignant": 0.9107142857142857, "specificity_benign": 0.5, "precision_malignant": 0.68, "f1_malignant": 0.7786259541984734, "auc_malignant_prob_sum": 0.8296130952380951}

## Thời gian (Test)

{"n_cases": 54, "device": ["cuda:0"], "total_s": {"median": 2.6799999999999997, "min": 0.75, "max": 7.81, "mean": 3.157037037037037}, "detection_zoom_s": {"median": 2.4000000000000004, "min": 0.66, "max": 7.5, "mean": 2.926111111111111}, "classification_s": {"median": 0.22, "min": 0.0, "max": 0.59, "mean": 0.23055555555555557}, "slices_per_case": {"median": 496.0, "min": 159, "max": 1268, "mean": 544.1666666666666}, "ms_per_slice_total": {"median": 4.781077281875978, "min": 3.9226519337016574, "max": 32.71551724137931, "mean": 6.791998632940684}, "classification_share": 0.0730290943219146}

{"n_cases": 104, "device": ["cuda:0"], "total_s": {"median": 2.09, "min": 1.28, "max": 4.29, "mean": 2.154326923076923}, "detection_zoom_s": {"median": 1.75, "min": 1.1, "max": 3.02, "mean": 1.7722115384615384}, "classification_s": {"median": 0.33, "min": 0.1, "max": 1.26, "mean": 0.3819230769230769}, "slices_per_case": {"median": 404.0, "min": 280, "max": 650, "mean": 393.0480769230769}, "ms_per_slice_total": {"median": 5.481006493506493, "min": 4.054054054054054, "max": 7.528735632183908, "mean": 5.4810789371251625}, "classification_share": 0.17728185672840882}

## YOLO đơn lẻ vs kết hợp (CT Test)

{"yolo_only": {"cancer_flagged": 42, "controls_flagged": 10, "n_cancer": 42, "n_controls": 12, "subtype": null}, "ungated": {"cancer_correct": 40, "macro_f1": 0.9529501915708812, "controls_labelled": 10}, "gated": {"cancer_correct": 38, "macro_f1": 0.9274074074074075, "controls_labelled": 2}, "oracle": {"cancer_correct": 41, "macro_f1": 0.9759139784946237}}
