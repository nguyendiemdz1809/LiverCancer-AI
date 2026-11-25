#!/usr/bin/env python
"""Chỉ số bổ sung theo đề cương (PI 1.4, 3.1, 3.2, 4.2) tính từ đầu ra ĐÃ LƯU, không chạy lại mô hình.

Nguồn (chỉ đọc):
  - outputs/runs/cls/ct_{resnet18,swin_t,hybrid}_gt_s42/val_predictions_best.csv  (xác suất từng vùng, Validation)
  - outputs/final_test/{ct,mri}/cases/*.json  (lần chạy Test DUY NHẤT 26/10/2026 với cấu hình khóa)
  - reports/final_test/metrics_test.json, outputs/web_e2e/e2e_report.json

Không chọn lại gì trên Test: đây chỉ là các chỉ số mô tả thêm (AUC-ROC, nhóm lành/ác MRI, thời gian xử lý,
so sánh YOLO đơn lẻ với YOLO + bộ phân loại) của đúng các dự đoán đã khóa.

  python scripts/100_bo_sung_de_cuong.py
Ra: reports/bo_sung_de_cuong/{chi_so_bo_sung.json, bao_cao_bo_sung.md}, reports/figures/bo_sung/*.png
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import auc, roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "bo_sung_de_cuong"
FIG = ROOT / "reports" / "figures" / "bo_sung"
CT_CLS = ["HCC", "ICC", "CHCC"]
MRI_MALIGNANT = {"Hepatocellular_carcinoma", "Intrahepatic_cholangiocarcinoma", "Hepatic_metastasis"}
MRI_BENIGN = {"Hepatic_hemangioma", "FOCAL_NODULAR_HYPERPLASIA", "Hepatic_cyst", "Hepatic_abscess"}
MODELS = [("ct_resnet18_gt_s42", "ResNet-18"), ("ct_swin_t_gt_s42", "Swin-T"), ("ct_hybrid_gt_s42", "Hybrid")]
COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def ovr_auc(y: np.ndarray, p: np.ndarray) -> dict:
    per = {c: float(roc_auc_score(y == i, p[:, i])) for i, c in enumerate(CT_CLS)}
    return {"macro_ovr": float(np.mean(list(per.values()))), "per_class": per}


def val_roi() -> dict:
    res, curves = {}, {}
    for run, name in MODELS:
        rows = list(csv.DictReader(open(ROOT / "outputs/runs/cls" / run / "val_predictions_best.csv", encoding="utf-8")))
        y = np.array([int(r["label_id"]) for r in rows])
        p = np.array([[float(r[f"p_{c}"]) for c in CT_CLS] for r in rows])
        res[name] = {"n_roi": len(rows), **ovr_auc(y, p)}
        curves[name] = (y, p)
    # đối chiếu với giá trị đã lưu khi huấn luyện
    saved = json.load(open(ROOT / "outputs/runs/cls/ct_hybrid_gt_s42/metrics_val_best.json", encoding="utf-8"))
    res["Hybrid"]["auc_saved_metrics"] = saved["roi"]["roc_auc_macro_ovr"]
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (ci, c) in zip(axes, enumerate(CT_CLS)):
        for (name, (y, p)), col in zip(curves.items(), COLORS):
            fpr, tpr, _ = roc_curve(y == ci, p[:, ci])
            ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{name} (AUC {auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=1)
        ax.set_title(f"{c} so với hai lớp còn lại")
        ax.set_xlabel("Tỷ lệ dương tính giả (1 − độ đặc hiệu)")
        ax.set_ylabel("Tỷ lệ dương tính thật (độ nhạy)")
        ax.legend(loc="lower right", fontsize=8.5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    fig.suptitle("Đường ROC mức vùng trên CT Validation (1.046 vùng theo nhãn chuẩn)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "roc_val_ba_bo_phan_loai.png", dpi=200)
    plt.close(fig)
    return res


def load_cases(mod: str) -> list[dict]:
    return [json.load(open(f, encoding="utf-8")) for f in sorted((ROOT / "outputs/final_test" / mod / "cases").glob("*.json"))]


def test_ct(cases: list[dict]) -> dict:
    cancer = [c for c in cases if c["true_label"] in CT_CLS]
    lab = [c for c in cancer if c.get("pipeline_scores")]
    y = np.array([CT_CLS.index(c["true_label"]) for c in lab])
    p = np.array([[c["pipeline_scores"][k] for k in CT_CLS] for c in lab])
    yo = np.array([CT_CLS.index(c["true_label"]) for c in cancer])
    po = np.array([[c["oracle_scores"][k] for k in CT_CLS] for c in cancer])
    out = {"pipeline": {"n": len(lab), "n_rejected_no_score": len(cancer) - len(lab), **ovr_auc(y, p)},
           "oracle": {"n": len(cancer), **ovr_auc(yo, po)}}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    for ax, (title, yy, pp) in zip(axes, [(f"Pipeline (YOLO + Hybrid + cổng), {len(lab)} BN có nhãn", y, p),
                                         (f"Oracle (vùng theo nhãn chuẩn), {len(cancer)} BN", yo, po)]):
        for ci, (c, col) in enumerate(zip(CT_CLS, COLORS)):
            fpr, tpr, _ = roc_curve(yy == ci, pp[:, ci])
            ax.plot(fpr, tpr, color=col, lw=1.8, drawstyle="steps-post", label=f"{c} (AUC {auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Tỷ lệ dương tính giả")
        ax.set_ylabel("Tỷ lệ dương tính thật")
        ax.legend(loc="lower right", fontsize=8.5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    fig.suptitle("Đường ROC mức bệnh nhân trên tập Test CT (một-so-với-còn-lại)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "roc_test_ct_benh_nhan.png", dpi=200)
    plt.close(fig)
    return out


def test_mri_binary(cases: list[dict]) -> dict:
    def grp(lbl: str) -> int:
        assert lbl in MRI_MALIGNANT | MRI_BENIGN, lbl
        return int(lbl in MRI_MALIGNANT)

    res = {}
    for key, lab_key, sc_key in [("pipeline", "pipeline_label", "pipeline_scores"), ("oracle", "oracle_label", "oracle_scores")]:
        y = np.array([grp(c["true_label"]) for c in cases])
        yhat = np.array([grp(c[lab_key]) for c in cases])
        s = np.array([sum(v for k, v in c[sc_key].items() if k in MRI_MALIGNANT) for c in cases])
        tp = int(((y == 1) & (yhat == 1)).sum()); tn = int(((y == 0) & (yhat == 0)).sum())
        fp = int(((y == 0) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
        prec = tp / (tp + fp); rec = tp / (tp + fn)
        res[key] = {"n": len(y), "n_malignant": int(y.sum()), "n_benign": int((y == 0).sum()),
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": (tp + tn) / len(y),
                    "sensitivity_malignant": rec, "specificity_benign": tn / (tn + fp),
                    "precision_malignant": prec, "f1_malignant": 2 * prec * rec / (prec + rec),
                    "auc_malignant_prob_sum": float(roc_auc_score(y, s))}
    return res


def timings(cases: list[dict]) -> dict:
    tot = [c["timings_s"]["total"] for c in cases]
    det = [c["timings_s"].get("detection_zoom", c["timings_s"]["total"]) for c in cases]
    cls = [c["timings_s"].get("classification", 0.0) for c in cases]
    sl = [sum(v["n_slices"] for v in c["volumes"]) for c in cases]
    per_slice_ms = [1000 * t / n for t, n in zip(tot, sl) if n]
    q = lambda xs: {"median": st.median(xs), "min": min(xs), "max": max(xs), "mean": st.mean(xs)}  # noqa: E731
    return {"n_cases": len(cases), "device": sorted({c["device"] for c in cases}), "total_s": q(tot),
            "detection_zoom_s": q(det), "classification_s": q(cls), "slices_per_case": q(sl),
            "ms_per_slice_total": q(per_slice_ms), "classification_share": sum(cls) / sum(tot)}


def yolo_only_vs_combined(cases: list[dict], m: dict) -> dict:
    cancer = [c for c in cases if c["true_label"] in CT_CLS]
    ctrl = [c for c in cases if c["true_label"] not in CT_CLS]
    has_region = lambda c: c["status"] != "no_lesion_detected"  # noqa: E731
    return {
        "yolo_only": {"cancer_flagged": sum(map(has_region, cancer)), "controls_flagged": sum(map(has_region, ctrl)),
                      "n_cancer": len(cancer), "n_controls": len(ctrl), "subtype": None},
        "ungated": {"cancer_correct": m["ungated_ablation"]["correct"], "macro_f1": m["ungated_ablation"]["macro_f1"],
                    "controls_labelled": sum(1 for c in ctrl if c.get("ungated_label"))},
        "gated": {"cancer_correct": m["pipeline"]["correct"], "macro_f1": m["pipeline"]["macro_f1"],
                  "controls_labelled": sum(1 for c in ctrl if c["status"] == "ok")},
        "oracle": {"cancer_correct": m["oracle"]["correct"], "macro_f1": m["oracle"]["macro_f1"]},
    }


def web_e2e() -> dict:
    d = json.load(open(ROOT / "outputs/web_e2e/e2e_report.json", encoding="utf-8"))
    rows = []
    for c in d["cases"]:
        t = c.get("time")
        rows.append({"case": c["case"], "viewport": c["viewport"], "dims": c.get("dims"), "state": c["state"],
                     "time_s": float(t.split()[0].replace(",", ".")) if t else None, "result": c.get("result"),
                     "slices": c.get("total_slices")})
    return {"n": len(rows), "errors": d["errors"], "cases": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = json.load(open(ROOT / "reports/final_test/metrics_test.json", encoding="utf-8"))
    ct, mri = load_cases("ct"), load_cases("mri")
    res = {"val_roi_auc": val_roi(), "test_ct_auc": test_ct(ct), "test_mri_benign_malignant": test_mri_binary(mri),
           "timing_test_ct": timings(ct), "timing_test_mri": timings(mri),
           "yolo_only_vs_combined_test_ct": yolo_only_vs_combined(ct, m["ct"]), "web_e2e": web_e2e(),
           "nguon": "Đầu ra đã lưu của lần chạy Test duy nhất (26/10/2026) và dự đoán Validation; không chạy lại mô hình."}
    (OUT / "chi_so_bo_sung.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    v, t, b = res["val_roi_auc"], res["test_ct_auc"], res["test_mri_benign_malignant"]["pipeline"]
    lines = ["# Chỉ số bổ sung theo đề cương", "", res["nguon"], "", "## AUC mức vùng, CT Validation", ""]
    lines += [f"- {k}: macro {x['macro_ovr']:.3f} | " + ", ".join(f"{c} {a:.3f}" for c, a in x["per_class"].items()) for k, x in v.items()]
    lines += ["", "## AUC mức bệnh nhân, CT Test", "",
              f"- Pipeline ({t['pipeline']['n']} BN có nhãn): macro {t['pipeline']['macro_ovr']:.3f} | " + ", ".join(f"{c} {a:.3f}" for c, a in t['pipeline']['per_class'].items()),
              f"- Oracle ({t['oracle']['n']} BN): macro {t['oracle']['macro_ovr']:.3f}", "",
              "## MRI Test lành/ác", "", json.dumps(b, ensure_ascii=False), "",
              "## Thời gian (Test)", "", json.dumps(res["timing_test_ct"], ensure_ascii=False), "",
              json.dumps(res["timing_test_mri"], ensure_ascii=False), "",
              "## YOLO đơn lẻ vs kết hợp (CT Test)", "", json.dumps(res["yolo_only_vs_combined_test_ct"], ensure_ascii=False)]
    (OUT / "bao_cao_bo_sung.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
