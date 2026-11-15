"""GĐ5: check the 4 trained classifiers and the end-to-end pipeline on Validation (never Test).

A  runs: load every best.pt, compare config / summary / metrics with experiments.csv and the Colab eval (eval_val/)
   against the local re-evaluation (eval_val_local/, scripts/82_eval_classifier.py).
B  MRI hybrid limitations from its real Val predictions (per class, confusion, per sequence, benign/malignant, curves).
C  preprocessing consistency: the pipeline re-run on Val volumes must select the same tracks / key slices and produce
   the same crops and probabilities as the Val YOLO ROIs of the classification dataset (used to choose the aggregation).
D  demo cases (Val patients, all available phases / sequences): pipeline (YOLO ROIs) vs GT ROIs (oracle), figures.
E  no-detection handling: a Val volume where YOLO finds nothing -> status no_lesion_detected, no label.
Outputs: outputs/predictions/pipeline_val/summary.json, reports/figures/pipeline/*.png, reports/pipeline_val_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.classification.inference import RoiInput  # noqa: E402
from livercancer.classification.metrics import aggregate, compute_metrics  # noqa: E402
from livercancer.config import CONFIG_DIR, load_paths, load_yaml  # noqa: E402
from livercancer.data.slices import load_volume_uint8  # noqa: E402
from livercancer.data.sources import image_path  # noqa: E402
from livercancer.pipeline import LiverCancerPipeline  # noqa: E402
from livercancer.pipeline_viz import render_case  # noqa: E402

RUNS = ROOT / "outputs" / "runs" / "cls"
OUT = ROOT / "outputs" / "predictions" / "pipeline_val"
FIG = ROOT / "reports" / "figures" / "pipeline"


def rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------- A
def check_runs() -> list[dict]:
    from livercancer.classification.evaluate import load_classifier
    exp = {r["run"]: r for r in rows(RUNS / "experiments.csv")}
    out = []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        c = json.loads((d / "config.json").read_text(encoding="utf-8"))
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        mb = json.loads((d / "metrics_val_best.json").read_text(encoding="utf-8"))
        ev = json.loads((d / "eval_val" / "metrics_val.json").read_text(encoding="utf-8"))
        loc = d / "eval_val_local" / "metrics_val.json"
        evl = json.loads(loc.read_text(encoding="utf-8")) if loc.exists() else None
        h = rows(d / "history.csv")
        model, ck, _ = load_classifier(d / "best.pt", "cpu")
        e = exp.get(d.name, {})
        bi = s["best_epoch"] - 1
        vl = [float(x["val_loss"]) for x in h]
        r = {"run": d.name, "modality": c["modality"], "model": c["model"], "loads": True, "ckpt_epoch": ck["epoch"],
             "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "best_epoch": s["best_epoch"],
             "epochs_run": s["epochs_run"], "n_train": f"{c['n_train_rois']} ROI / {c['n_train_patients']} BN",
             "n_val": f"{c['n_val_rois']} ROI / {c['n_val_patients']} BN",
             "val_roi_acc": mb["roi"]["accuracy"], "val_roi_macro_f1": mb["roi"]["macro_f1"],
             "val_patient_yolo_macro_f1": ev["yolo_roi_all"]["patient"]["macro_f1"],
             "val_patient_yolo_acc": ev["yolo_roi_all"]["patient"]["accuracy"], "aggregation": ev["aggregation"]["chosen"],
             "train_acc_at_best": float(h[bi]["train_acc"]), "val_acc_at_best": float(h[bi]["val_acc"]),
             "val_loss_min_epoch": int(np.argmin(vl)) + 1, "val_loss_at_best": vl[bi], "val_loss_min": min(vl)}
        r["matches_experiments_csv"] = bool(e) and all(abs(float(e[k]) - r[v]) < 6e-5 for k, v in (
            ("val_roi_acc", "val_roi_acc"), ("val_roi_macro_f1", "val_roi_macro_f1"), ("val_patient_macro_f1_yolo", "val_patient_yolo_macro_f1"))) \
            and int(e["best_epoch"]) == s["best_epoch"] and e["aggregation"] == r["aggregation"]
        r["local_reeval_identical"] = evl is not None and all(
            abs(evl[k][lvl]["macro_f1"] - ev[k][lvl]["macro_f1"]) < 1e-6 for k in ("gt_roi", "yolo_roi_all") for lvl in ("roi", "patient"))
        out.append(r)
    return out


# ----------------------------------------------------------------------------- B
def mri_limits() -> dict:
    d = RUNS / "mri_hybrid_gt_s42"
    ev = json.loads((d / "eval_val" / "metrics_val.json").read_text(encoding="utf-8"))
    classes = ev["classes"]
    man = {r["roi_id"]: r for r in rows(ROOT / "data" / "processed" / "cls" / "mri" / "manifest.csv")}
    pr = rows(d / "val_predictions_best.csv")
    mal = {i for i, c in enumerate(classes) if c in load_yaml(CONFIG_DIR / "cls.yaml")["mri"]["groups"]["Malignant"]}
    seq, grp = defaultdict(lambda: [0, 0]), [0, 0]
    for r in pr:
        y, p = int(r["label_id"]), int(r["pred_id"])
        s = man[r["roi_id"]]["series"]
        seq[s][0] += y == p
        seq[s][1] += 1
        grp[0] += (y in mal) == (p in mal)
        grp[1] += 1
    y = np.array([int(r["label_id"]) for r in pr])
    P = np.array([[float(r[f"p_{c}"]) for c in classes] for r in pr])
    by = defaultdict(list)
    for i, r in enumerate(pr):
        by[r["patient_id"]].append(i)
    pg = np.mean([(y[ix[0]] in mal) == (int(aggregate(P[ix], "mean_prob").argmax()) in mal) for ix in by.values()])
    h = rows(d / "history.csv")
    train_counts = Counter(r["label"] for r in man.values() if r["split"] == "train" and r["source"] == "gt")
    val_pat = Counter(r["label"] for r in {r["patient_id"]: r for r in man.values() if r["split"] == "val" and r["source"] == "gt"}.values())
    m = ev["gt_roi"]["roi"]
    return {"classes": classes, "per_class": m["per_class"], "confusion_matrix": m["confusion_matrix"],
            "roi_acc_by_sequence": {k: round(v[0] / v[1], 3) for k, v in sorted(seq.items())},
            "benign_malignant_roi_acc": round(grp[0] / grp[1], 3), "benign_malignant_patient_acc": round(float(pg), 3),
            "train_rois_per_class": dict(train_counts), "val_patients_per_class": dict(val_pat),
            "curve": [{k: x[k] for k in ("epoch", "train_loss", "val_loss", "train_acc", "val_acc", "val_macro_f1")} for x in h],
            "yolo_all_roi_acc": ev["yolo_roi_all"]["roi"]["accuracy"], "yolo_matched_roi_acc": ev["yolo_roi_matched"]["roi"]["accuracy"]}


# ----------------------------------------------------------------------------- C
def consistency(mod: str, pipe: LiverCancerPipeline, paths, n: int, seed: int) -> dict:
    root = ROOT / "data" / "processed" / "cls" / mod
    man = [r for r in rows(root / "manifest.csv") if r["split"] == "val" and r["source"] == "yolo"]
    byvol = defaultdict(list)
    for r in man:
        byvol[(r["patient_id"], r["series"])].append(r)
    keys = random.Random(seed).sample(sorted(byvol), min(n, len(byvol)))
    same_sel = tot_sel = 0
    px, prob_diff, n_cmp = [], [], 0
    for pid, series in keys:
        vol = load_volume_uint8(image_path(paths, mod, pid, series))
        res = pipe.run([(series, vol)], pid)
        want = {(int(re.search(r"_T(\d+)_k(\d+)$", r["roi_id"]).group(1)), int(r["slice_k"])): r for r in byvol[(pid, series)]}
        got = {(ro.track_id, ro.slice_k): ro for ro in res.rois}
        tot_sel += len(set(want) | set(got))
        same = set(want) & set(got)
        same_sel += len(same)
        if same:
            mrois = []
            for key in sorted(same):
                r, ro = want[key], got[key]
                loc = cv2.imread(str(root / r["local_path"]), cv2.IMREAD_GRAYSCALE)
                ctx = cv2.imread(str(root / r["context_path"]), cv2.IMREAD_GRAYSCALE)
                px.append(float(np.abs(loc.astype(int) - ro.local.astype(int)).mean()))
                px.append(float(np.abs(ctx.astype(int) - ro.context.astype(int)).mean()))
                mrois.append((RoiInput(loc, ctx, r["roi_id"], ro.track_id, ro.slice_k), ro))
            a = pipe.classifier.classify([m for m, _ in mrois])
            b = pipe.classifier.classify([g for _, g in mrois])
            for ra, rb in zip(a.per_roi, b.per_roi):
                prob_diff.append(max(abs(ra["scores"][c] - rb["scores"][c]) for c in ra["scores"]))
                n_cmp += 1
    return {"volumes": len(keys), "selected_rois_union": tot_sel, "identical_track_and_slice": same_sel,
            "crop_mean_abs_pixel_diff_max": round(max(px), 4) if px else None, "crop_identical_fraction": round(float(np.mean(np.array(px) == 0)), 4) if px else None,
            "max_prob_diff_dataset_vs_pipeline": round(max(prob_diff), 5) if prob_diff else None, "rois_compared": n_cmp}


# ----------------------------------------------------------------------------- D / E
def gt_scores(pipe, mod, pid) -> dict | None:
    root = ROOT / "data" / "processed" / "cls" / mod
    rs = [r for r in rows(root / "manifest.csv") if r["split"] == "val" and r["source"] == "gt" and r["patient_id"] == pid]
    if not rs:
        return None
    rois = [RoiInput(cv2.imread(str(root / r["local_path"]), cv2.IMREAD_GRAYSCALE), cv2.imread(str(root / r["context_path"]), cv2.IMREAD_GRAYSCALE),
                     r["roi_id"], int(r["lesion_id"]), int(r["slice_k"])) for r in rs]
    c = pipe.classifier.classify(rois)
    return {"label": c.label, "scores": c.scores, "n_rois": len(rois)}


def demo_cases(mod, pipe, paths, per_class: int, seed: int, cfg) -> list[dict]:
    root = ROOT / "data" / "processed" / "cls" / mod
    pats = [r for r in rows(root / "patients.csv") if r["split"] == "val"]
    rng = random.Random(seed)
    chosen = []
    for c in cfg["classes"] + (["CN"] if mod == "ct" else []):
        ps = sorted(p["patient_id"] for p in pats if p["label"] == c)
        chosen += rng.sample(ps, min(per_class if c != "CN" else 1, len(ps)))
    vols = [r for r in rows(ROOT / load_yaml(CONFIG_DIR / "cls.yaml")[mod]["yolo_dataset"] / "volumes.csv") if r["split"] == "val"]
    names_vi = dict(zip(cfg["classes"], cfg.get("class_names_vi", cfg["classes"])))
    out = []
    FIG.mkdir(parents=True, exist_ok=True)
    for pid in chosen:
        series = [v["series"] for v in vols if v["patient_id"] == pid and v["status"] == "ok"]
        files = [image_path(paths, mod, pid, s) for s in series]
        res = pipe.run_nifti(files, pid, [s.replace(" ", "") for s in series])
        true = next(p["label"] for p in pats if p["patient_id"] == pid)
        g = gt_scores(pipe, mod, pid)
        fig = FIG / f"{mod}_{pid}.png"
        render_case(res, fig, f"{pid} ({mod.upper()}, Validation)", true, g["scores"] if g else None, names_vi=names_vi)
        out.append({"patient_id": pid, "true_label": true, "n_volumes": len(files), "status": res.status,
                    "n_regions": len(res.regions), "pipeline_label": res.label, "pipeline_score": round(max(res.scores.values()), 3) if res.scores else None,
                    "pipeline_correct": res.label == true if res.label else None,
                    "gt_roi_label": g["label"] if g else None, "gt_roi_score": round(max(g["scores"].values()), 3) if g else None,
                    "gt_roi_correct": (g["label"] == true) if g else None, "seconds": res.timings_s.get("total"),
                    "figure": str(fig.relative_to(ROOT)).replace("\\", "/")})
        print(f"  {pid} {true:34s} pipeline={res.label} gt_roi={g['label'] if g else None} regions={len(res.regions)}", flush=True)
    return out


def no_detection(mod, pipe, paths) -> list[dict]:
    root = ROOT / "data" / "processed" / "cls" / mod
    with_roi = {(r["patient_id"], r["series"]) for r in rows(root / "manifest.csv") if r["split"] == "val" and r["source"] == "yolo"}
    vols = [r for r in rows(ROOT / load_yaml(CONFIG_DIR / "cls.yaml")[mod]["yolo_dataset"] / "volumes.csv") if r["split"] == "val" and r["status"] == "ok"]
    gt_vol = {(r["patient_id"], r["series"]) for r in rows(root / "manifest.csv") if r["split"] == "val" and r["source"] == "gt"}
    cand = [v for v in vols if (v["patient_id"], v["series"]) not in with_roi]
    out = []
    for kind, pick in (("có tổn thương (GT)", [v for v in cand if (v["patient_id"], v["series"]) in gt_vol]),
                       ("không có tổn thương annotation (CN)" if mod == "ct" else "không có ROI-GT", [v for v in cand if (v["patient_id"], v["series"]) not in gt_vol])):
        if not pick:
            out.append({"kind": kind, "n_candidates": 0})
            continue
        v = pick[0]
        res = pipe.run_nifti([image_path(paths, mod, v["patient_id"], v["series"])], v["patient_id"], [v["series"].replace(" ", "")])
        fig = FIG / f"{mod}_nodetect_{v['patient_id']}_{v['series'].replace(' ', '')}.png"
        render_case(res, fig, f"{v['patient_id']} · {v['series']} ({mod.upper()}, Validation)", v["label"])
        out.append({"kind": kind, "n_candidates": len(pick), "example": f"{v['patient_id']}_{v['series']}", "true_label": v["label"],
                    "status": res.status, "label_returned": res.label, "message": res.message,
                    "figure": str(fig.relative_to(ROOT)).replace("\\", "/")})
    return out


def report(s: dict) -> None:
    L = ["# Pipeline AI trên Validation: YOLO → Tumor Zoom → phân loại", "",
         f"*Sinh bởi `scripts/91_pipeline_val_demo.py` lúc {datetime.now():%Y-%m-%d %H:%M}. Số liệu: `outputs/predictions/pipeline_val/summary.json`, "
         "các run ở `outputs/runs/cls/`. Hình: `reports/figures/pipeline/`. **Chỉ Validation; Test không được dùng. Không huấn luyện lại.***", "",
         "## 1. Kiểm tra 4 run phân loại (bundle từ Colab)", "",
         "| Run | Mô hình | Tham số | Best / dừng ở epoch | Val ROI acc | Val ROI macro-F1 | Val BN macro-F1 (ROI-YOLO) | Gộp BN | Train acc / Val acc tại best | Epoch val loss nhỏ nhất | Khớp experiments.csv | Đánh giá lại ở local trùng |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["runs"]:
        L.append(f"| {r['run']} | {r['model']} | {r['params_M']} M | {r['best_epoch']} / {r['epochs_run']} | {r['val_roi_acc']:.4f} | {r['val_roi_macro_f1']:.4f} | "
                 f"{r['val_patient_yolo_macro_f1']:.4f} | {r['aggregation']} | {r['train_acc_at_best']:.3f} / {r['val_acc_at_best']:.3f} | {r['val_loss_min_epoch']} | "
                 f"{'✅' if r['matches_experiments_csv'] else '❌'} | {'✅' if r['local_reeval_identical'] else '❌'} |")
    L += ["", "Mọi `best.pt` nạp được (`livercancer.classification.evaluate.load_classifier`). Dữ liệu huấn luyện: CT 6.026 ROI-GT / 194 BN, MRI 11.230 / 316 BN; "
          "Val CT 1.046 / 42 BN, MRI 2.575 / 78 BN. SHA256 dữ liệu trong `experiments.csv` = SHA của `cls_ct.zip` / `cls_mri.zip` đã đóng gói.", ""]
    m = s["mri_limits"]
    short = {"Hepatocellular_carcinoma": "HCC", "Intrahepatic_cholangiocarcinoma": "ICC", "Hepatic_metastasis": "Di căn", "Hepatic_hemangioma": "U máu",
             "Hepatic_cyst": "Nang", "Hepatic_abscess": "Áp xe", "FOCAL_NODULAR_HYPERPLASIA": "FNH"}
    L += ["## 2. Hạn chế của Hybrid MRI (Val acc 47.8%, macro-F1 42.8%): số liệu thật", "",
          "| Lớp | Precision | Recall | Specificity | F1 | ROI Val | ROI Train (GT) | BN Val |", "|---|---|---|---|---|---|---|---|"]
    for c in m["classes"]:
        x = m["per_class"][c]
        L.append(f"| {short[c]} | {x['precision']:.3f} | {x['recall_sensitivity']:.3f} | {x['specificity']:.3f} | {x['f1']:.3f} | {x['support']} | "
                 f"{m['train_rois_per_class'].get(c, 0)} | {m['val_patients_per_class'].get(c, 0)} |")
    L += ["", "Ma trận nhầm lẫn ROI-GT Val (hàng = nhãn thật, cột = dự đoán; thứ tự " + ", ".join(short[c] for c in m["classes"]) + "):", "", "```"]
    L += [f"{short[c]:7s} " + " ".join(f"{v:4d}" for v in row) for c, row in zip(m["classes"], m["confusion_matrix"])]
    L += ["```", ""]
    cv = m["curve"]
    L += ["- **Overfitting rất mạnh:** train acc " + " → ".join(f"{float(x['train_acc']):.2f}" for x in cv[::4]) +
          "; val acc " + " → ".join(f"{float(x['val_acc']):.2f}" for x in cv[::4]) +
          f". Val loss nhỏ nhất ở epoch {int(np.argmin([float(x['val_loss']) for x in cv])) + 1} ({min(float(x['val_loss']) for x in cv):.3f}) rồi tăng, trong khi train loss giảm liên tục.",
          "- **HCC là lớp 'hút' dự đoán:** nhiều ROI ICC và áp xe bị đoán là HCC (xem ma trận). HCC có nhiều ROI Train nhất (3.833).",
          "- **Di căn** kém nhất (F1 thấp nhất); FNH và HCC tốt nhất.",
          "- **Theo chuỗi MRI** (ROI acc): " + ", ".join(f"{k} {v:.2f}" for k, v in m["roi_acc_by_sequence"].items()) +
          ". Một mô hình duy nhất cho cả 8 chuỗi có đặc điểm hình ảnh rất khác nhau.",
          f"- **Gộp lành tính / ác tính** (suy ra từ dự đoán 7 lớp, chưa huấn luyện riêng): ROI {100 * m['benign_malignant_roi_acc']:.1f}%, "
          f"bệnh nhân {100 * m['benign_malignant_patient_acc']:.1f}%.",
          f"- **ROI do YOLO** làm giảm thêm: ROI acc {100 * m['yolo_all_roi_acc']:.1f}% với mọi ROI-YOLO (gồm vùng giả) so với "
          f"{100 * m['yolo_matched_roi_acc']:.1f}% với ROI-YOLO khớp GT.",
          "- **Val nhỏ:** 7–25 BN mỗi lớp, nên chỉ số theo lớp dao động lớn.",
          "- *Giả thuyết, chưa kiểm chứng:* (1) mỗi ROI được phân loại độc lập theo từng chuỗi, mô hình không kết hợp thông tin đa pha / đa chuỗi "
          "của cùng tổn thương, trong khi chẩn đoán MRI gan thường dựa vào mẫu ngấm thuốc qua các pha; (2) mô hình 39 M tham số với khoảng 11 nghìn ROI "
          "rất tương quan (nhiều lát / chuỗi của cùng tổn thương) dễ học thuộc. Hướng thử trên Val: fusion đa chuỗi ở mức tổn thương, augmentation mạnh hơn, "
          "mô hình nhỏ hơn, phân loại lành / ác tính.", ""]
    L += ["## 3. Pipeline và tính nhất quán tiền xử lý", "",
          "`livercancer.pipeline.LiverCancerPipeline`: NIfTI → `load_volume_uint8` (giống lúc train YOLO) → YOLO (conf 0.12) → gộp box → track 3D "
          "→ tối đa 5 lát đại diện mỗi vùng (diện tích × độ tin cậy, cách nhau ≥ 2 lát CT / ≥ 1 lát MRI, đúng như khi tạo ROI-YOLO của bộ dữ liệu) "
          "→ crop local / context 224×224 → chuẩn hóa ImageNet → mô hình phân loại → gộp mức ca theo cách đã chọn trên Val.", "",
          "| Kiểm tra (Val, chạy lại pipeline) | CT | MRI |", "|---|---|---|"]
    cc = s["consistency"]
    for k, lab in (("volumes", "Volume kiểm tra"), ("identical_track_and_slice", "ROI cùng vùng và cùng lát với bộ dữ liệu"),
                   ("selected_rois_union", "Tổng ROI (hợp hai phía)"), ("crop_identical_fraction", "Tỉ lệ crop trùng từng pixel"),
                   ("crop_mean_abs_pixel_diff_max", "Sai khác pixel trung bình lớn nhất"),
                   ("max_prob_diff_dataset_vs_pipeline", "Sai khác xác suất lớn nhất (crop bộ dữ liệu so với crop pipeline)")):
        L.append(f"| {lab} | {cc['ct'].get(k)} | {cc['mri'].get(k)} |")
    L += ["", "## 4. Ca minh họa trên Validation: pipeline (ROI do YOLO) so với ROI từ GT (oracle)", "",
          "Mỗi ca chạy **mọi pha / chuỗi có sẵn** của bệnh nhân; nhãn là dự đoán gộp mức ca. ROI-GT là kết quả **oracle** (dùng nhãn chuẩn để cắt vùng), "
          "chỉ để so sánh, không phải kết quả của pipeline. Mẫu nhỏ chọn ngẫu nhiên (seed cố định), **không** phải đánh giá.", "",
          "| Modality | BN | Nhãn thật | Số volume | Số vùng YOLO | Pipeline (ROI-YOLO) | Đúng? | ROI-GT (oracle) | Đúng? | Thời gian (s) | Hình |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for mod in ("ct", "mri"):
        for c in s["cases"][mod]:
            ok = lambda v: "✅" if v else ("❌" if v is False else "—")  # noqa: E731
            L.append(f"| {mod.upper()} | {c['patient_id']} | {c['true_label']} | {c['n_volumes']} | {c['n_regions']} | {c['pipeline_label']} ({c['pipeline_score']}) | "
                     f"{ok(c['pipeline_correct'])} | {c['gt_roi_label']} ({c['gt_roi_score']}) | {ok(c['gt_roi_correct'])} | {c['seconds']} | "
                     f"[hình]({c['figure'].replace('reports/', '')}) |")
    L += ["", "- CT: BN nhóm chứng CN **luôn** nhận một nhãn ung thư nếu YOLO có vùng (mô hình không có lớp 'không tổn thương'). Đây là hạn chế của thiết kế 3 lớp.",
          "- MRI: mọi nhãn là **không đáng tin cậy** (cảnh báo đỏ trên hình và trong `warnings`).", "",
          "## 5. Trường hợp YOLO không phát hiện vùng nào", "", "| Modality | Loại volume | Số volume Val như vậy | Ví dụ | Trạng thái | Nhãn trả về | Hình |", "|---|---|---|---|---|---|---|"]
    for mod in ("ct", "mri"):
        for c in s["no_detection"][mod]:
            L.append(f"| {mod.upper()} | {c['kind']} | {c['n_candidates']} | {c.get('example', '—')} | {c.get('status', '—')} | {c.get('label_returned')} | "
                     f"{('[hình](' + c['figure'].replace('reports/', '') + ')') if c.get('figure') else '—'} |")
    L += ["", "Thông điệp trả về: *“" + next((c["message"] for mod in ("ct", "mri") for c in s["no_detection"][mod] if c.get("message")), "") + "”*", "",
          "## 6. Hạn chế và việc tiếp theo",
          "- Mọi chỉ số trên là **Validation**; mô hình (CT hybrid) và cách gộp được chọn trên chính Val này, nên còn lạc quan. Test chỉ đánh giá một lần ở cuối.",
          "- CT Val chỉ 42 BN: 3 mô hình CT có cùng độ chính xác mức BN (38/42); chênh lệch macro-F1 ROI 0.83–0.86 là nhỏ.",
          "- MRI chưa đạt chất lượng; pipeline vẫn chạy được nhưng nhãn chỉ để minh họa.",
          "- Chưa tích hợp vào website (theo yêu cầu). Interface đã sẵn: `LiverCancerPipeline.run_*()` → `PipelineResult.as_dict()`."]
    (ROOT / "reports" / "pipeline_val_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-consistency", type=int, default=12)
    ap.add_argument("--per-class", type=int, default=2, help="CT demo patients per class (MRI: 1 per class)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    paths = load_paths()
    OUT.mkdir(parents=True, exist_ok=True)
    allc = load_yaml(CONFIG_DIR / "cls.yaml")
    s = {"time": datetime.now().isoformat(timespec="seconds"), "split": "val", "runs": check_runs(), "mri_limits": mri_limits(),
         "consistency": {}, "cases": {}, "no_detection": {}}
    print("A runs checked:", [(r["run"], r["matches_experiments_csv"], r["local_reeval_identical"]) for r in s["runs"]], flush=True)
    for mod in ("ct", "mri"):
        pipe = LiverCancerPipeline(mod, args.device)
        s["consistency"][mod] = consistency(mod, pipe, paths, args.n_consistency, args.seed)
        print(f"C {mod}:", s["consistency"][mod], flush=True)
        s["cases"][mod] = demo_cases(mod, pipe, paths, args.per_class if mod == "ct" else 1, args.seed, allc[mod])
        s["no_detection"][mod] = no_detection(mod, pipe, paths)
        print(f"E {mod}:", [(c["kind"], c.get("status")) for c in s["no_detection"][mod]], flush=True)
    (OUT / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    report(s)
    print("written reports/pipeline_val_report.md,", OUT / "summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
