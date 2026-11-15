"""GĐ5: evaluate the complete pipeline (YOLO -> Tumor Zoom -> Hybrid) on EVERY Validation patient, CT and MRI.
Never Test; no retraining; no threshold / model change (everything comes from configs chosen earlier on Val).

Stage `run` (cached per patient in outputs/predictions/pipeline_validation/<mod>/cases/<pid>.json):
  pipeline on all usable volumes of the patient (status ok in the exported dataset), then
  - detection: every YOLO track is matched to the GT 3D lesions of its volume (exported Val labels, IoU >= 0.3 on a shared
    slice) -> detected / missed lesions, small lesions (< 32x32 px), multi-lesion volumes, false tracks;
  - classification: pipeline label (all YOLO ROIs, the real system), "matched-only" label (only ROIs of tracks matching GT;
    diagnostic, uses GT) and the GT-ROI "oracle" label (classifier on the GT crops of the classification dataset);
    the same at volume level (per CT phase / MRI sequence);
  - a Vietnamese figure per patient.
Stage `report`: metrics, CSV tables, confusion matrices, error analysis, P0267 case study, Vietnamese report in
  reports/pipeline_validation/.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.classification.inference import RoiInput  # noqa: E402
from livercancer.classification.metrics import aggregate, compute_metrics, plot_confusion  # noqa: E402
from livercancer.config import CONFIG_DIR, load_paths, load_yaml  # noqa: E402
from livercancer.data.sources import image_path  # noqa: E402
from livercancer.pipeline import AGG_VI, LiverCancerPipeline  # noqa: E402
from livercancer.pipeline_viz import render_case, short  # noqa: E402
from livercancer.roi.boxes import iou  # noqa: E402

_spec = importlib.util.spec_from_file_location("b80", ROOT / "scripts" / "80_build_cls_dataset.py")
b80 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b80)

OUT = ROOT / "outputs" / "predictions" / "pipeline_validation"
REP = ROOT / "reports" / "pipeline_validation"
SMALL = 32 * 32


def rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def agg_label(prob_list, method, classes):
    if not prob_list:
        return None, {}
    v = aggregate(np.array(prob_list), method)
    return classes[int(v.argmax())], {c: round(float(x), 4) for c, x in zip(classes, v)}


# ----------------------------------------------------------------------------- run
def run_modality(mod: str, args, paths) -> None:
    cfg = load_yaml(CONFIG_DIR / "cls.yaml")
    data = ROOT / cfg[mod]["yolo_dataset"]
    classes = cfg[mod]["classes"]
    names_vi = dict(zip(classes, cfg[mod]["class_names_vi"]))
    vols = [r for r in rows(data / "volumes.csv") if r["split"] == "val" and r["status"] == "ok"]
    split = {r["patient_id"]: r for r in rows(paths["project"]["splits"] / f"{mod}_split.csv")}
    pats = sorted({v["patient_id"] for v in vols})
    if args.limit:
        pats = pats[:args.limit]
    assert all(split[p]["split"] == "val" for p in pats), "only Validation patients may be evaluated here"
    gt_les, _ = b80.gt_lesions(data, "val")
    croot = ROOT / cfg["roi"]["out_dir"] / mod
    gt_rows = defaultdict(list)
    for r in rows(croot / "manifest.csv"):
        if r["split"] == "val" and r["source"] == "gt":
            gt_rows[r["patient_id"]].append(r)
    pipe = LiverCancerPipeline(mod, args.device, evidence_gate=args.evidence_gate)
    method = pipe.classifier.aggregation
    cdir, fdir = OUT / mod / "cases", REP / "figures" / mod
    cdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    for i, pid in enumerate(pats):
        cf = cdir / f"{pid}.json"
        if cf.exists() and not args.redo:
            continue
        series = [v["series"] for v in vols if v["patient_id"] == pid]
        name_of = {s: s.replace(" ", "") for s in series}
        res = pipe.run_nifti([image_path(paths, mod, pid, s) for s in series], pid, [name_of[s] for s in series])
        true = split[pid]["label"]
        # --- detection vs GT lesions, per volume
        tracks_info, vol_info = [], []
        for s in series:
            nm = name_of[s]
            rr = res.roi_results.get(nm)
            les = gt_les.get((pid, s), [])
            first = min((r["track_id"] for r in res.regions if r["volume"] == nm), default=0)
            hit = set()
            for ti, t in enumerate(rr.tracks if rr else []):
                best = max((iou(t.boxes[k].xyxy, g.boxes[k].xyxy) for g in les for k in t.boxes if k in g.boxes), default=0.0)
                m = [li for li, g in enumerate(les) if any(k in g.boxes and iou(t.boxes[k].xyxy, g.boxes[k].xyxy) >= 0.3 for k in t.boxes)]
                hit.update(m)
                tracks_info.append({"track_id": first + ti, "volume": nm, "matched": bool(m), "best_iou": round(best, 3),
                                    "n_slices": t.n_slices, "max_conf": round(t.max_conf, 4),
                                    "max_area": round(max(b.area for b in t.boxes.values()), 1)})
            areas = [max(b.area for b in g.boxes.values()) for g in les]
            vol_info.append({"series": s, "volume": nm, "n_slices": rr.n_slices if rr else 0, "n_tracks": len(rr.tracks) if rr else 0,
                             "n_gt_lesions": len(les), "n_gt_detected": len(hit), "gt_lesion_max_areas": [round(a) for a in areas],
                             "n_small_gt": sum(a < SMALL for a in areas), "n_small_detected": sum(1 for li in hit if areas[li] < SMALL)})
        matched_ids = {t["track_id"] for t in tracks_info if t["matched"]}
        per_roi = (res.classification or {}).get("per_roi", [])
        probs = lambda sel: [[r["scores"][c] for c in classes] for r in per_roi if sel(r)]  # noqa: E731
        vol_of = {t["track_id"]: t["volume"] for t in tracks_info}
        for v in vol_info:
            v["pipeline_label"], v["pipeline_scores"] = agg_label(probs(lambda r, nm=v["volume"]: vol_of.get(r["track_id"]) == nm), method, classes)
        m_label, m_scores = agg_label(probs(lambda r: r["track_id"] in matched_ids), method, classes)
        n_fp_rois = sum(1 for r in per_roi if r["track_id"] not in matched_ids)
        # --- oracle GT ROIs
        g = gt_rows.get(pid, [])
        o_label, o_scores, o_vol = None, {}, {}
        if g:
            rois = [RoiInput(cv2.imread(str(croot / r["local_path"]), cv2.IMREAD_GRAYSCALE), cv2.imread(str(croot / r["context_path"]), cv2.IMREAD_GRAYSCALE),
                     r["roi_id"], int(r["lesion_id"]), int(r["slice_k"])) for r in g]
            c = pipe.classifier.classify(rois)
            pr = [[x["scores"][k] for k in classes] for x in c.per_roi]
            o_label, o_scores = agg_label(pr, method, classes)
            for s in series:
                idx = [j for j, r in enumerate(g) if r["series"] == s]
                o_vol[name_of[s]] = agg_label([pr[j] for j in idx], method, classes)[0] if idx else None
        for v in vol_info:
            v["oracle_label"] = o_vol.get(v["volume"])
        case = {"patient_id": pid, "modality": mod, "true_label": true, "status": res.status, "pipeline_label": res.label,
                "pipeline_scores": res.scores, "matched_only_label": m_label, "matched_only_scores": m_scores,
                "oracle_label": o_label, "oracle_scores": o_scores, "n_oracle_rois": len(g), "n_rois": len(per_roi), "n_fp_rois": n_fp_rois,
                "aggregation": method, "volumes": vol_info, "tracks": tracks_info, "timings_s": res.timings_s, "device": res.device,
                "evidence": res.evidence, "evidence_gate": args.evidence_gate}
        if true not in classes:   # CT controls (CN): not a class of the model -> never "correct/wrong"
            verdict = "CN_khong_co_vung" if res.status != "ok" else "CN_bi_gan_nhan_ung_thu"
        else:
            verdict = "khong_phan_loai" if res.status != "ok" else ("dung" if res.label == true else "sai")
        case["figure"] = f"figures/{mod}/{pid}_{verdict}.png"
        note = (f"Đối chiếu (oracle, vùng cắt theo nhãn chuẩn): {names_vi.get(o_label, o_label)}" if o_label else
                "Không có vùng tổn thương chuẩn (GT) để đối chiếu." if true in classes else
                "Bệnh nhân nhóm chứng (CN): không có lớp này trong mô hình 3 lớp.")
        render_case(res, REP / case["figure"], f"Validation · {mod.upper()} · ca {pid}", true if true in classes else None,
                    o_scores or None, names_vi, note)
        cf.write_text(json.dumps(case, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  [{i + 1}/{len(pats)}] {pid} {true:32s} {res.status:18s} pipeline={res.label} oracle={o_label} "
              f"tracks={len(tracks_info)} matched={len(matched_ids)} {res.timings_s.get('total')}s", flush=True)


# ----------------------------------------------------------------------------- report
def write_csv(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in recs:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)


def metric_block(cases, key, classes):
    """Patient-level metrics over cases with a label in `classes`; cases without a prediction count as wrong."""
    lab = [c for c in cases if c["true_label"] in classes]
    have = [c for c in lab if c[key]]
    y = [classes.index(c["true_label"]) for c in have]
    p = np.eye(len(classes))[[classes.index(c[key]) for c in have]] if have else np.zeros((0, len(classes)))
    m = compute_metrics(y, p, classes) if have else None
    correct = sum(c[key] == c["true_label"] for c in have)
    return {"n_patients": len(lab), "n_classified": len(have), "n_unclassified": len(lab) - len(have),
            "correct": correct, "accuracy_all": correct / max(1, len(lab)), "metrics_classified": m}


def report(args) -> None:
    cfg = load_yaml(CONFIG_DIR / "cls.yaml")
    REP.mkdir(parents=True, exist_ok=True)
    summary, pat_rows, vol_rows, err_rows = {"time": datetime.now().isoformat(timespec="seconds"), "split": "val"}, [], [], []
    L = ["# Đánh giá pipeline AI trên toàn bộ Validation (CT và MRI)", "",
         f"*Sinh bởi `scripts/92_pipeline_validation.py --stage report` lúc {datetime.now():%Y-%m-%d %H:%M}. "
         "Dữ liệu từng ca: `outputs/predictions/pipeline_validation/<ct|mri>/cases/*.json`; tổng hợp: `summary.json`; bảng: `reports/pipeline_validation/*.csv`. "
         "**Chỉ Validation, không dùng Test, không huấn luyện lại, không đổi ngưỡng hay mô hình** (YOLO conf 0.12, Tumor Zoom không lọc, "
         "CT Hybrid / MRI Hybrid và cách gộp đã chọn trên Val trước đó).*", "",
         "**Cách đọc:**",
         "- *Pipeline* = hệ thống thật: YOLO tự tìm vùng → Tumor Zoom → Hybrid, gộp mọi vùng của bệnh nhân.",
         "- *Oracle (ROI-GT)* = cùng mô hình phân loại nhưng trên vùng cắt theo nhãn chuẩn. Chỉ để đối chiếu, **không** phải kết quả của hệ thống.",
         "- *Chỉ vùng khớp GT* = pipeline nhưng bỏ các vùng YOLO không trùng tổn thương thật. Đây là phân tích chẩn đoán (cần GT), không dùng được khi chạy thật.",
         "- Bệnh nhân không được phân loại (YOLO không thấy vùng nào) bị tính là **sai** khi tính độ chính xác.", ""]
    for mod in ("ct", "mri"):
        classes, nvi = cfg[mod]["classes"], dict(zip(cfg[mod]["classes"], cfg[mod]["class_names_vi"]))
        sv = lambda c: short(nvi.get(c, c)) if c else "—"  # noqa: E731
        cases = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / mod / "cases").glob("*.json"))]
        if not cases:
            continue
        lab = [c for c in cases if c["true_label"] in classes]
        ctrl = [c for c in cases if c["true_label"] not in classes]
        blocks = {k: metric_block(cases, k, classes) for k in ("pipeline_label", "matched_only_label", "oracle_label")}
        summary[mod] = {"n_patients": len(cases), "n_lesion_patients": len(lab), "n_controls": len(ctrl), "blocks": blocks}
        # --- per-patient table + error analysis
        for c in cases:
            vs = c["volumes"]
            n_gt = sum(v["n_gt_lesions"] for v in vs)
            n_det = sum(v["n_gt_detected"] for v in vs)
            vol_det = sum(v["n_gt_detected"] > 0 for v in vs if v["n_gt_lesions"])
            vol_gt = sum(v["n_gt_lesions"] > 0 for v in vs)
            all_small = n_gt > 0 and all(a < SMALL for v in vs for a in v["gt_lesion_max_areas"])
            multi = any(v["n_gt_lesions"] > 1 for v in vs)
            fp_frac = c["n_fp_rois"] / c["n_rois"] if c["n_rois"] else 0.0
            r = {"modality": mod.upper(), "patient_id": c["patient_id"], "true_label": c["true_label"], "status": c["status"],
                 "pipeline_label": c["pipeline_label"], "pipeline_prob": round(max(c["pipeline_scores"].values()), 4) if c["pipeline_scores"] else "",
                 "pipeline_correct": (c["pipeline_label"] == c["true_label"]) if c["true_label"] in classes else "",
                 "oracle_label": c["oracle_label"], "oracle_correct": (c["oracle_label"] == c["true_label"]) if c["oracle_label"] else "",
                 "matched_only_label": c["matched_only_label"], "n_volumes": len(vs), "n_tracks": len(c["tracks"]),
                 "n_tracks_matched": sum(t["matched"] for t in c["tracks"]), "n_rois": c["n_rois"], "fp_roi_fraction": round(fp_frac, 3),
                 "gt_lesions": n_gt, "gt_lesions_detected": n_det, "volumes_with_lesion": vol_gt, "volumes_lesion_detected": vol_det,
                 "all_lesions_small": all_small, "multi_lesion_volume": multi, "seconds": c["timings_s"].get("total"), "figure": c["figure"]}
            pat_rows.append(r)
            for v in vs:
                vol_rows.append({"modality": mod.upper(), "patient_id": c["patient_id"], "series": v["series"], "true_label": c["true_label"],
                                 "n_slices": v["n_slices"], "n_tracks": v["n_tracks"], "n_gt_lesions": v["n_gt_lesions"],
                                 "n_gt_detected": v["n_gt_detected"], "n_small_gt": v["n_small_gt"], "n_small_detected": v["n_small_detected"],
                                 "pipeline_label": v["pipeline_label"], "oracle_label": v["oracle_label"]})
            if c["true_label"] in classes and c["pipeline_label"] != c["true_label"]:
                if c["status"] == "insufficient_evidence":
                    cause = "Bị cổng bằng chứng từ chối (không đưa nhãn)"
                elif c["status"] != "ok" or r["n_tracks_matched"] == 0:
                    cause = "YOLO bỏ sót toàn bộ tổn thương (không có vùng nào trùng tổn thương thật)"
                elif c["oracle_label"] == c["true_label"] and c["matched_only_label"] == c["true_label"]:
                    cause = "Vùng giả của YOLO làm lệch kết quả gộp (chỉ vùng thật thì đúng; oracle đúng)"
                elif c["oracle_label"] == c["true_label"]:
                    cause = "Vùng YOLO cắt khác vùng chuẩn (oracle đúng, vùng YOLO thật vẫn sai)"
                else:
                    cause = "Mô hình phân loại sai ngay cả với vùng chuẩn (oracle cũng sai)"
                err_rows.append({**{k: r[k] for k in ("modality", "patient_id", "true_label", "pipeline_label", "pipeline_prob", "oracle_label",
                                                      "matched_only_label", "n_tracks", "n_tracks_matched", "fp_roi_fraction", "gt_lesions",
                                                      "gt_lesions_detected", "all_lesions_small", "multi_lesion_volume", "figure")}, "cause": cause})
        # --- confusion matrices (+ "not classified" column)
        for key, ttl in (("pipeline_label", "Pipeline (vùng YOLO)"), ("oracle_label", "Oracle (vùng GT)")):
            k = len(classes)
            cm = np.zeros((k, k + 1), int)
            for c in lab:
                j = classes.index(c[key]) if c[key] else k
                cm[classes.index(c["true_label"]), j] += 1
            summary[mod][f"cm_{key}"] = cm.tolist()
            _plot_cm(cm, [sv(c) for c in classes], ttl, mod, REP / f"cm_{mod}_{key.replace('_label', '')}.png")
        # --- detection
        vs_all = [v for c in cases for v in c["volumes"]]
        det = {"volumes_with_lesion": sum(v["n_gt_lesions"] > 0 for v in vs_all),
               "volumes_lesion_detected": sum(v["n_gt_detected"] > 0 for v in vs_all if v["n_gt_lesions"]),
               "gt_lesions": sum(v["n_gt_lesions"] for v in vs_all), "gt_lesions_detected": sum(v["n_gt_detected"] for v in vs_all),
               "small_lesions": sum(v["n_small_gt"] for v in vs_all), "small_detected": sum(v["n_small_detected"] for v in vs_all),
               "patients_with_lesion": len(lab), "patients_no_detected_lesion": sum(1 for c in lab if not any(t["matched"] for t in c["tracks"])),
               "patients_no_region": sum(1 for c in cases if c["status"] == "no_lesion_detected"),
               "tracks": sum(len(c["tracks"]) for c in cases), "tracks_false": sum(1 for c in cases for t in c["tracks"] if not t["matched"]),
               "volumes": len(vs_all)}
        summary[mod]["detection"] = det
        # --- per phase / sequence (volume level)
        per_series = defaultdict(lambda: Counter())
        for c in lab:
            for v in c["volumes"]:
                s = per_series[v["series"]]
                s["n"] += 1
                s["pipeline_correct"] += v["pipeline_label"] == c["true_label"]
                s["pipeline_none"] += v["pipeline_label"] is None
                s["oracle_n"] += v["oracle_label"] is not None
                s["oracle_correct"] += v["oracle_label"] == c["true_label"]
                s["lesion_detected"] += v["n_gt_detected"] > 0
                s["with_lesion"] += v["n_gt_lesions"] > 0
        summary[mod]["per_series"] = {k: dict(v) for k, v in sorted(per_series.items())}
        # --- per class (patient level)
        per_class = {}
        for cl in classes:
            cs = [c for c in lab if c["true_label"] == cl]
            per_class[cl] = {"n": len(cs), "pipeline_correct": sum(c["pipeline_label"] == cl for c in cs),
                             "oracle_correct": sum(c["oracle_label"] == cl for c in cs),
                             "predicted_as": dict(Counter(c["pipeline_label"] or "không phân loại" for c in cs))}
        summary[mod]["per_class"] = per_class
        if mod == "ct":
            summary[mod]["controls"] = {"n": len(ctrl), "no_region": sum(c["status"] == "no_lesion_detected" for c in ctrl),
                                        "rejected": sum(c["status"] == "insufficient_evidence" for c in ctrl),
                                        "labelled": sum(c["pipeline_label"] is not None for c in ctrl),
                                        "given_cancer_label": dict(Counter(c["pipeline_label"] for c in ctrl if c["pipeline_label"]))}
        if mod == "mri":
            grp = cfg["mri"]["groups"]
            g_of = {x: g for g, xs in grp.items() for x in xs}
            for key in ("pipeline_label", "oracle_label"):
                ok = [c for c in lab if c[key]]
                summary[mod][f"benign_malignant_{key}"] = {"n": len(lab), "correct": sum(g_of[c[key]] == g_of[c["true_label"]] for c in ok)}
        L += _mod_section(mod, summary[mod], classes, nvi, [e for e in err_rows if e["modality"] == mod.upper()], cases)
    # P0267
    p = OUT / "ct" / "cases" / "P0267.json"
    if p.exists():
        L += _p0267(json.loads(p.read_text(encoding="utf-8")), cfg)
    L += _recommendations(summary)
    at = next(i for i, x in enumerate(L) if x.startswith("## 1."))
    L[at:at] = _summary(summary, err_rows)
    write_csv(REP / "patients.csv", pat_rows)
    write_csv(REP / "volumes.csv", vol_rows)
    write_csv(REP / "errors.csv", err_rows)
    write_csv(REP / "metrics_summary.csv", _metric_rows(summary))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    (REP / "bao_cao_pipeline_validation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("written", REP / "bao_cao_pipeline_validation.md")


def _plot_cm(cm, names, title, mod, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams["font.family"] = "DejaVu Sans"
    cols = names + ["Không\nphân loại"]
    fig, ax = plt.subplots(figsize=(1.35 * len(cols) + 3, 1.1 * len(names) + 2.6), facecolor="white")
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(len(names)), names, fontsize=11)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({100 * norm[i, j]:.0f}%)", ha="center", va="center", fontsize=10,
                    color="white" if norm[i, j] > 0.5 else "#111827")
    acc = np.trace(cm[:, :len(names)]) / max(1, cm.sum())
    ax.set_title(f"{mod.upper()} · {title}\nmức bệnh nhân · Validation · đúng {np.trace(cm[:, :len(names)])}/{cm.sum()} ({100 * acc:.1f}%)",
                 fontsize=12, color="#0B3D91")
    ax.set_xlabel("Dự đoán", fontsize=12)
    ax.set_ylabel("Nhãn thực tế", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def _fmt(m):
    return f"{100 * m:.1f}%" if m is not None else "—"


def _mod_section(mod, s, classes, nvi, errs, cases) -> list[str]:
    sv = lambda c: short(nvi.get(c, c)) if c else "không phân loại"  # noqa: E731
    b, d = s["blocks"], s["detection"]
    L = [f"## {'1' if mod == 'ct' else '2'}. {mod.upper()}: {s['n_patients']} bệnh nhân Validation"
         + (f" ({s['n_lesion_patients']} có ung thư + {s['n_controls']} nhóm chứng CN)" if mod == "ct" else ""), ""]
    L += ["### Phân loại mức bệnh nhân", "",
          "| Cách lấy vùng | BN có nhãn | Được phân loại | Đúng | Accuracy (tính BN không phân loại là sai) | Macro-Precision | Macro-Recall | Macro-F1 | Balanced acc |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, nm in (("pipeline_label", "**Pipeline (vùng YOLO, hệ thống thật)**"), ("matched_only_label", "Chỉ vùng YOLO khớp GT (chẩn đoán)"),
                    ("oracle_label", "Oracle: vùng cắt theo nhãn chuẩn")):
        x = b[key]
        m = x["metrics_classified"] or {}
        L.append(f"| {nm} | {x['n_patients']} | {x['n_classified']} | {x['correct']} | **{_fmt(x['accuracy_all'])}** | "
                 f"{_fmt(m.get('macro_precision'))} | {_fmt(m.get('macro_recall'))} | {_fmt(m.get('macro_f1'))} | {_fmt(m.get('balanced_accuracy'))} |")
    L += ["", f"Precision / Recall / F1 tính trên các BN được phân loại; accuracy tính trên mọi BN có nhãn. Ma trận nhầm lẫn: "
          f"[pipeline](cm_{mod}_pipeline.png), [oracle](cm_{mod}_oracle.png).", "",
          "| Lớp | Số BN | Pipeline đúng | Oracle đúng | Pipeline dự đoán thành |", "|---|---|---|---|---|"]
    for cl in classes:
        x = s["per_class"][cl]
        L.append(f"| {nvi.get(cl, cl)} | {x['n']} | {x['pipeline_correct']} | {x['oracle_correct']} | "
                 + ", ".join(f"{sv(k) if k in classes else k}: {v}" for k, v in sorted(x["predicted_as"].items(), key=lambda z: -z[1])) + " |")
    pm = (b["pipeline_label"]["metrics_classified"] or {}).get("per_class", {})
    if pm:
        L += ["", "| Lớp (pipeline, mức BN) | Precision | Recall (Sensitivity) | Specificity | F1 |", "|---|---|---|---|---|"]
        for cl in classes:
            x = pm[cl]
            L.append(f"| {sv(cl)} | {_fmt(x['precision'])} | {_fmt(x['recall_sensitivity'])} | {_fmt(x['specificity'])} | {_fmt(x['f1'])} |")
    if mod == "ct":
        c = s["controls"]
        L += ["", f"**Nhóm chứng CN ({c['n']} BN, không có tổn thương được thu thập):** {c['no_region']} BN không có vùng YOLO nào; "
              f"{c.get('rejected', 0)} BN bị cổng bằng chứng từ chối (không đưa nhãn); {c.get('labelled', c['n'] - c['no_region'])} BN bị gán nhãn ung thư từ vùng giả: "
              + ", ".join(f"{sv(k)} {v}" for k, v in c["given_cancer_label"].items()) + ". Đây là dương tính giả ở mức bệnh nhân, "
              "do mô hình 3 lớp không có lớp 'không tổn thương'."]
    if mod == "mri":
        for key, nm in (("pipeline_label", "Pipeline"), ("oracle_label", "Oracle")):
            x = s[f"benign_malignant_{key}"]
            L.append(f"- Gộp **lành tính / ác tính** từ dự đoán 7 lớp ({nm}): đúng {x['correct']}/{x['n']} ({_fmt(x['correct'] / max(1, x['n']))}).")
    L += ["", "### Phát hiện (YOLO) và Tumor Zoom", "",
          "| Chỉ số | Giá trị |", "|---|---|",
          f"| Volume có tổn thương được YOLO tìm thấy | {d['volumes_lesion_detected']}/{d['volumes_with_lesion']} ({_fmt(d['volumes_lesion_detected'] / max(1, d['volumes_with_lesion']))}) |",
          f"| Tổn thương 3D được tìm thấy | {d['gt_lesions_detected']}/{d['gt_lesions']} ({_fmt(d['gt_lesions_detected'] / max(1, d['gt_lesions']))}) |",
          f"| Tổn thương nhỏ (< 32×32 px) được tìm thấy | {d['small_detected']}/{d['small_lesions']} ({_fmt(d['small_detected'] / max(1, d['small_lesions']))}) |",
          f"| BN có tổn thương mà YOLO bỏ sót **mọi** tổn thương | {d['patients_no_detected_lesion']}/{d['patients_with_lesion']} |",
          f"| BN không có vùng YOLO nào (trạng thái `no_lesion_detected`) | {d['patients_no_region']}/{s['n_patients']} |",
          f"| Vùng YOLO giả (không trùng tổn thương thật) | {d['tracks_false']}/{d['tracks']} ({_fmt(d['tracks_false'] / max(1, d['tracks']))}), "
          f"trung bình {d['tracks_false'] / max(1, d['volumes']):.1f} mỗi volume |", "",
          f"### Theo {'pha CT' if mod == 'ct' else 'chuỗi MRI'} (mức volume, BN có tổn thương)", "",
          "| " + ("Pha" if mod == "ct" else "Chuỗi") + " | Volume | Tổn thương được YOLO tìm thấy | Pipeline đúng | Oracle đúng |", "|---|---|---|---|---|"]
    for k, x in s["per_series"].items():
        L.append(f"| {k} | {x['n']} | {x['lesion_detected']}/{x['with_lesion']} | {x['pipeline_correct']}/{x['n']} ({_fmt(x['pipeline_correct'] / max(1, x['n']))}) | "
                 f"{x['oracle_correct']}/{x['oracle_n']} ({_fmt(x['oracle_correct'] / max(1, x['oracle_n']))}) |")
    L += ["", "### Phân tích lỗi (BN có tổn thương bị pipeline phân loại sai hoặc không phân loại)", ""]
    if not errs:
        L.append("Không có.")
    else:
        L += ["| BN | Nhãn thực tế | Pipeline | Oracle | Chỉ vùng khớp GT | Vùng (khớp GT / tổng) | Tỉ lệ ROI từ vùng giả | Tổn thương tìm thấy | Toàn tổn thương nhỏ | Nhiều tổn thương | Nguyên nhân | Hình |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in errs:
            L.append(f"| {e['patient_id']} | {sv(e['true_label'])} | {sv(e['pipeline_label'])} | {sv(e['oracle_label'])} | {sv(e['matched_only_label'])} | "
                     f"{e['n_tracks_matched']}/{e['n_tracks']} | {100 * e['fp_roi_fraction']:.0f}% | {e['gt_lesions_detected']}/{e['gt_lesions']} | "
                     f"{'có' if e['all_lesions_small'] else 'không'} | {'có' if e['multi_lesion_volume'] else 'không'} | {e['cause']} | [hình]({e['figure']}) |")
        cnt = Counter(e["cause"] for e in errs)
        L += ["", "Tổng hợp nguyên nhân: " + "; ".join(f"{k}: **{v}**" for k, v in cnt.most_common()) + "."]
    ok = [c for c in cases if c["true_label"] in classes and c["pipeline_label"] == c["true_label"]][:4]
    if ok:
        L += ["", "Ví dụ ca đúng: " + ", ".join(f"[{c['patient_id']}]({c['figure']})" for c in ok) + ". Mọi ca có hình trong `figures/" + mod + "/`."]
    L.append("")
    return L


def _p0267(c, cfg) -> list[str]:
    nvi = dict(zip(cfg["ct"]["classes"], cfg["ct"]["class_names_vi"]))
    sv = lambda x: short(nvi.get(x, x)) if x else "—"  # noqa: E731
    L = ["## 3. Ca P0267 (nhãn thực tế CHCC, pipeline dự đoán " + sv(c["pipeline_label"]) + ")", "",
         f"![P0267]({c['figure']})", "",
         f"Xác suất pipeline: " + ", ".join(f"{sv(k)} {100 * v:.1f}%" for k, v in c["pipeline_scores"].items()) +
         f". Oracle (vùng GT): **{sv(c['oracle_label'])}** (" + ", ".join(f"{sv(k)} {100 * v:.1f}%" for k, v in c["oracle_scores"].items()) +
         f"). Chỉ vùng YOLO khớp GT: **{sv(c['matched_only_label'])}**.", "",
         "| Pha | Lát | Vùng YOLO | Tổn thương GT | GT được tìm thấy | Diện tích lớn nhất của tổn thương GT (px) | Pipeline (pha) | Oracle (pha) |",
         "|---|---|---|---|---|---|---|---|"]
    for v in c["volumes"]:
        L.append(f"| {v['series']} | {v['n_slices']} | {v['n_tracks']} | {v['n_gt_lesions']} | {v['n_gt_detected']} | {v['gt_lesion_max_areas']} | "
                 f"{sv(v['pipeline_label'])} | {sv(v['oracle_label'])} |")
    m = sum(t["matched"] for t in c["tracks"])
    L += ["", f"- {len(c['tracks'])} vùng YOLO trên 4 pha, trong đó {m} vùng trùng tổn thương thật; {100 * c['n_fp_rois'] / max(1, c['n_rois']):.0f}% "
          "số ROI đưa vào phân loại đến từ vùng giả.",
          ("- **Bộ phân loại sai ngay cả trên vùng chuẩn (oracle → " + sv(c["oracle_label"]) + ")** và cả khi chỉ dùng vùng YOLO khớp GT "
           "(→ " + sv(c["matched_only_label"]) + "): lỗi nằm ở **bộ phân loại**, không phải do YOLO bỏ sót hay Tumor Zoom cắt sai."
           if c["oracle_label"] != c["true_label"] else
           "- Oracle (vùng chuẩn) đoán đúng CHCC: lỗi đến từ cách lấy vùng của pipeline (vùng giả / vùng cắt lệch)."),
          ("- YOLO tìm thấy " + str(sum(v["n_gt_detected"] for v in c["volumes"])) + "/" + str(sum(v["n_gt_lesions"] for v in c["volumes"]))
           + " tổn thương GT 3D trên 4 pha; P0267 có nhiều ổ tổn thương trong cùng pha. Các ổ không được tìm thấy không làm đổi kết luận, "
           "vì oracle (đủ mọi ổ) cũng đoán HCC."),
          "- CHCC (ung thư hỗn hợp tế bào gan – đường mật) mang đặc điểm của cả HCC và ICC, nên dễ nhầm sang một trong hai lớp này. "
          "Đây là giải thích hợp lý về mặt bệnh học, nhưng *chưa được kiểm chứng* bằng thực nghiệm riêng.",
          "- Đã kiểm tra tiền xử lý: crop của pipeline trùng từng pixel với dữ liệu đánh giá (`reports/pipeline_val_report.md` §3), nên không phải lỗi tiền xử lý.", ""]
    return L


def _summary(s, errs) -> list[str]:
    L = ["## Tóm tắt", ""]
    for mod in ("ct", "mri"):
        if mod not in s:
            continue
        x, d = s[mod], s[mod]["detection"]
        b = x["blocks"]
        p, o = b["pipeline_label"], b["oracle_label"]
        cause = Counter(e["cause"] for e in errs if e["modality"] == mod.upper())
        line = (f"- **{mod.upper()}** ({x['n_patients']} BN Val"
                + (f", {x['n_lesion_patients']} có ung thư + {x['n_controls']} CN" if mod == "ct" else "") + "): "
                + ("trên **BN ung thư** (không phải toàn bộ CT) " if mod == "ct" else "")
                + f"pipeline đúng **{p['correct']}/{p['n_patients']} ({100 * p['accuracy_all']:.1f}%)**, macro-F1 "
                f"{100 * (p['metrics_classified'] or {}).get('macro_f1', 0):.1f}%; oracle (vùng GT) {o['correct']}/{o['n_patients']} "
                f"({100 * o['accuracy_all']:.1f}%). YOLO tìm thấy tổn thương ở {d['volumes_lesion_detected']}/{d['volumes_with_lesion']} volume; "
                f"bỏ sót mọi tổn thương ở {d['patients_no_detected_lesion']} BN. Lỗi: "
                + "; ".join(f"{k.split(' (')[0]}: {v}" for k, v in cause.most_common()) + ".")
        if mod == "ct":
            fp = x["controls"].get("labelled", x["controls"]["n"] - x["controls"]["no_region"])
            ok_all = p["correct"] + x["controls"]["n"] - fp
            line += (f" **Nhóm chứng CN: {fp}/{x['controls']['n']} BN bị gán nhãn ung thư** (dương tính giả mức BN). "
                     f"**Toàn bộ {x['n_patients']} BN CT:** kết quả chấp nhận được (ung thư đúng + CN không bị gán nhãn) {ok_all}/{x['n_patients']} "
                     f"({100 * ok_all / x['n_patients']:.1f}%). Cổng bằng chứng đề xuất: `han_che/bao_cao_han_che.md`.")
        else:
            line += " **Mô hình MRI chưa đạt chất lượng**; nhãn MRI không đáng tin cậy."
        L.append(line)
    L += ["- Ca **P0267** (CHCC → HCC): bộ phân loại đoán HCC ngay cả trên vùng GT ở cả 4 pha; YOLO tìm đúng khối chính. Xem §3.", ""]
    return L


def _recommendations(s) -> list[str]:
    return ["## 4. Đề xuất cải thiện (có căn cứ từ số liệu trên; mọi thử nghiệm chỉ chọn trên Validation)", "",
            "1. **Vùng giả chi phối kết quả gộp.** Tỉ lệ ROI đến từ vùng giả rất cao; với các BN mà oracle đúng nhưng pipeline sai, "
            "nên thử *gộp có trọng số theo độ tin cậy YOLO*, hoặc chỉ dùng k vùng tin cậy nhất (chọn trên Val). Không dùng GT lúc suy luận.",
            "2. **Thêm lớp 'không phải tổn thương' cho bộ phân loại** (huấn luyện bằng các vùng YOLO giả trên Train) để loại vùng giả và tránh "
            "gán nhãn ung thư cho BN nhóm chứng CN.",
            "3. **CT: CHCC** là lớp khó nhất về bản chất (hỗn hợp HCC và ICC). Cần xem ma trận nhầm lẫn theo lớp, có thể dùng nhiều pha cùng lúc "
            "(fusion C1/C2/C3/P ở mức tổn thương).",
            "4. **MRI:** mô hình overfit nặng (train acc ≈ 0.97, Val ≈ 0.48). Hướng thử: fusion đa chuỗi ở mức tổn thương (8 chuỗi của cùng tổn thương "
            "cùng vào một mẫu), mô hình nhỏ hơn hoặc đóng băng một phần backbone, augmentation mạnh hơn, và bài toán lành / ác tính 2 lớp.",
            "5. **YOLO bỏ sót tổn thương nhỏ:** đã ghi nhận ở giai đoạn detection. Nếu được phép huấn luyện thêm: imgsz 640 hoặc bộ CT đầy đủ.",
            "6. Mọi số trên là Validation (đã dùng để chọn mô hình và cách gộp), nên còn lạc quan. Kết luận cuối phải chờ đánh giá Test **một lần**.", ""]


def _metric_rows(s) -> list[dict]:
    out = []
    for mod in ("ct", "mri"):
        if mod not in s:
            continue
        for key, b in s[mod]["blocks"].items():
            m = b["metrics_classified"] or {}
            out.append({"modality": mod.upper(), "roi_source": {"pipeline_label": "pipeline_yolo", "matched_only_label": "yolo_matched_gt_diagnostic",
                                                                "oracle_label": "oracle_gt_roi"}[key],
                        "patients": b["n_patients"], "classified": b["n_classified"], "correct": b["correct"],
                        "accuracy_all": round(b["accuracy_all"], 4), "macro_precision": round(m.get("macro_precision", 0), 4),
                        "macro_recall": round(m.get("macro_recall", 0), 4), "macro_f1": round(m.get("macro_f1", 0), 4),
                        "balanced_accuracy": round(m.get("balanced_accuracy", 0), 4)})
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["run", "report", "all"], default="all")
    ap.add_argument("--modality", choices=["ct", "mri", "both"], default="both")
    ap.add_argument("--redo", action="store_true", help="recompute cached cases")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="first N patients only (smoke run)")
    ap.add_argument("--out-root", type=Path, default=None, help="alternative output root (smoke run)")
    ap.add_argument("--evidence-gate", choices=["config", "on", "off"], default="off",
                    help="reject option; default off = the baseline evaluation (results in pipeline_validation/)")
    ap.add_argument("--tag", default="", help="suffix of the output folders, e.g. gate -> pipeline_validation_gate/")
    args = ap.parse_args()
    global OUT, REP
    if args.tag:
        OUT = OUT.with_name(OUT.name + '_' + args.tag)
        REP = REP.parent / f'{REP.name}_{args.tag}'
    if args.out_root:
        OUT, REP = args.out_root / "pred", args.out_root / "report"
    paths = load_paths()
    if args.stage in ("run", "all"):
        for mod in (["ct", "mri"] if args.modality == "both" else [args.modality]):
            print(f"== run {mod.upper()}", flush=True)
            run_modality(mod, args, paths)
    if args.stage in ("report", "all"):
        report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
