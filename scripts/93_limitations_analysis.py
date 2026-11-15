"""GĐ5: analyse the two main limitations on Validation (never Test; no retraining; checkpoints / data untouched).

A  CT controls (CN) labelled as cancer: evidence gate (reject option) chosen on Val with a rule fixed beforehand
   (max CN rejected s.t. >= 95 % of cancer patients still classified), leave-one-out estimate of its optimism,
   zero-loss alternative, single-volume variant (website upload of one file). Outcomes reported separately for cancer
   patients, controls and all CT patients.
B  CN false regions: YOLO re-run on the CN volumes (detector only) -> where the false regions are (inside / outside the
   liver, GT liver mask used for analysis only), size, length, confidence; figures.
C  MRI: confusion, prediction prior vs true distribution, over-confidence, per sequence, lesion size and intensity of
   Train vs Val ROIs per class, selective classification (coverage vs accuracy).
D  YOLO misses (CT + MRI): every GT lesion matched against the tracks from the cached Val scan (same tracks as the
   pipeline): missed lesions by size / class / phase, fully missed volumes and patients; figures.
Inputs: outputs/predictions/pipeline_validation/<mod>/cases/*.json (scripts/92_), outputs/predictions/roi_val/<mod>/raw_dets.csv,
        data/processed/cls/<mod>/manifest.csv, outputs/runs/cls/mri_hybrid_gt_s42/.
Outputs: reports/pipeline_validation/han_che/ (report, figures; CSV gitignored), outputs/predictions/pipeline_validation/limitations.json
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import CONFIG_DIR, load_paths, load_yaml  # noqa: E402
from livercancer.data.slices import load_volume_uint8, slice_image  # noqa: E402
from livercancer.data.sources import image_path  # noqa: E402
from livercancer.evidence import GateRule, features, passes  # noqa: E402
from livercancer.roi.boxes import iou, merge_boxes  # noqa: E402
from livercancer.roi.pipeline import load_roi_config  # noqa: E402
from livercancer.roi.render import draw  # noqa: E402
from livercancer.roi.tracking import filter_tracks, link_tracks  # noqa: E402

PV = ROOT / "outputs" / "predictions" / "pipeline_validation"
OUT = ROOT / "reports" / "pipeline_validation" / "han_che"
SMALL = 32 * 32


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


b80, r71 = _load("b80", "80_build_cls_dataset.py"), _load("r71", "71_eval_roi_val.py")


def rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def cases(mod):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((PV / mod / "cases").glob("*.json"))]


def write_csv(path, recs):
    keys = []
    for r in recs:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)


# ----------------------------------------------------------------------------- A. evidence gate
GRID = ([GateRule("max_conf", round(t, 2)) for t in np.arange(0.15, 0.81, 0.05)]
        + [GateRule("n_volumes", round(t, 2), m) for t in np.arange(0.2, 0.61, 0.05) for m in (1, 2, 3, 4)]
        + [GateRule("longest", round(t, 2), L) for t in (0.3, 0.4, 0.5) for L in (3, 5, 10, 15, 20)])


def evaluate_rule(rule, cs):
    """Counts for one rule over CT cases (cancer = label in classes; control = CN)."""
    out = Counter()
    for c in cs:
        cancer = c["true_label"] != "CN"
        has_label = c["status"] == "ok"
        keep = has_label and passes(c["tracks"], rule)
        if cancer:
            out["cancer"] += 1
            out["cancer_kept"] += keep
            out["cancer_kept_correct"] += keep and c["pipeline_label"] == c["true_label"]
            out["cancer_rejected_correct"] += (not keep) and c["pipeline_label"] == c["true_label"]
        else:
            out["cn"] += 1
            out["cn_rejected"] += not keep
    return out


def choose(rules, cs, min_keep=0.95):
    """Rule fixed beforehand: maximise CN rejected s.t. cancer kept >= min_keep; ties -> more cancer kept, simpler kind."""
    order = {"max_conf": 0, "n_volumes": 1, "longest": 2}
    best = None
    for r in rules:
        e = evaluate_rule(r, cs)
        if e["cancer_kept"] < min_keep * e["cancer"] - 1e-9:
            continue
        key = (e["cn_rejected"], e["cancer_kept"], -order[r.kind], -r.conf_min)
        if best is None or key > best[0]:
            best = (key, r, e)
    return best[1], best[2]


def gate_analysis(ct):
    chosen, ev = choose(GRID, ct)
    zero, ez = choose(GRID, ct, min_keep=1.0)
    base = evaluate_rule(GateRule("max_conf", 0.0), ct)
    # leave-one-out: select on the other patients, apply to the held-out one
    loo = Counter()
    for i, c in enumerate(ct):
        r, _ = choose(GRID, ct[:i] + ct[i + 1:])
        e = evaluate_rule(r, [c])
        for k, v in e.items():
            loo[k] += v
        loo[f"rule:{r.kind}/{r.conf_min}/{r.k}"] += 1
    # single volume (one uploaded file): per volume, max_conf or longest only
    vols = []
    for c in ct:
        for v in c["volumes"]:
            tr = [t for t in c["tracks"] if t["volume"] == v["volume"]]
            vols.append({"true_label": "CN" if c["true_label"] == "CN" else ("lesion" if v["n_gt_lesions"] else "no_gt"),
                         "status": "ok" if tr else "no_lesion_detected", "tracks": tr, "pipeline_label": c["pipeline_label"]})
    vols = [v for v in vols if v["true_label"] != "no_gt"]
    vgrid = [r for r in GRID if r.kind != "n_volumes"]
    vchosen, vev = choose(vgrid, [{**v, "true_label": "CN" if v["true_label"] == "CN" else "X", "pipeline_label": None} for v in vols])
    per_patient = []
    for c in ct:
        f = features(c["tracks"], chosen.conf_min)
        per_patient.append({"patient_id": c["patient_id"], "true_label": c["true_label"], "pipeline_label": c["pipeline_label"],
                            "max_conf": round(f["max_conf"], 3), f"n_pha_vung_>={chosen.conf_min:g}": f["n_volumes_strong"],
                            "n_regions": len(c["tracks"]), "classifier_max_prob": round(max(c["pipeline_scores"].values()), 3) if c["pipeline_scores"] else "",
                            "gate_pass": c["status"] == "ok" and passes(c["tracks"], chosen)})
    return {"chosen": chosen.__dict__, "chosen_counts": dict(ev), "zero_loss": zero.__dict__, "zero_loss_counts": dict(ez),
            "no_gate": dict(base), "loo": dict(loo), "volume_rule": vchosen.__dict__, "volume_rule_counts": dict(vev),
            "per_patient": per_patient, "grid_size": len(GRID)}


def outcome_table(ct, rule):
    """Separate outcomes for cancer patients, controls, all CT patients."""
    o = Counter()
    for c in ct:
        keep = c["status"] == "ok" and (rule is None or passes(c["tracks"], rule))
        if c["true_label"] != "CN":
            o["cancer"] += 1
            if not keep:
                o["cancer_no_label"] += 1
            elif c["pipeline_label"] == c["true_label"]:
                o["cancer_correct"] += 1
            else:
                o["cancer_wrong"] += 1
        else:
            o["cn"] += 1
            o["cn_no_label" if not keep else "cn_cancer_label"] += 1
    o["all"] = o["cancer"] + o["cn"]
    o["all_acceptable"] = o["cancer_correct"] + o["cn_no_label"]
    o["all_wrong_label"] = o["cancer_wrong"] + o["cn_cancer_label"]
    return dict(o)


# ----------------------------------------------------------------------------- B. CN false regions
def cn_regions(ct, paths, figdir):
    from livercancer.detection.yolo_detector import LesionDetector
    from livercancer.roi.pipeline import analyze_volume
    det = LesionDetector.from_config("ct")
    csv_rows = {(r["patient_id"], r["phase"]): r for r in rows(Path(paths["ct"]["csv"]))}
    recs, figs = [], []
    for c in [c for c in ct if c["true_label"] == "CN"]:
        for v in c["volumes"]:
            vol = load_volume_uint8(image_path(paths, "ct", c["patient_id"], v["series"]))
            res = analyze_volume(vol, det)
            lm = Path(paths["ct"]["data_dir"]) / csv_rows[(c["patient_id"], v["series"])]["liver_mask_path"]
            liver = None
            if lm.exists():
                m = np.asanyarray(nib.load(str(lm)).dataobj.get_unscaled()) > 0
                liver = m.reshape(m.shape[:3] + (-1,)).any(-1) if m.ndim > 3 else m
            for t in res.tracks:
                k = t.best_slice
                b = t.boxes[k]
                cx, cy = int((b.x0 + b.x1) / 2), int((b.y0 + b.y1) / 2)
                inside = None
                if liver is not None and liver[:, :, k].any():
                    inside = bool(liver[min(cx, liver.shape[0] - 1), min(cy, liver.shape[1] - 1), k])   # display (x=i, y=j)
                elif liver is not None:
                    inside = False
                recs.append({"patient_id": c["patient_id"], "phase": v["series"], "slice": k, "n_slices": t.n_slices,
                             "max_conf": round(t.max_conf, 3), "box_w": round(b.x1 - b.x0), "box_h": round(b.y1 - b.y0),
                             "centre_in_liver": inside, "liver_mask_empty": liver is not None and not liver.any()})
                if t.max_conf >= 0.4:
                    z = res.zoom(t, k)
                    img = draw(res.images[k], [b.xyxy], [], [(*z["local_window"], "local")], [f"vùng giả {t.max_conf:.2f}"])
                    figs.append((t.max_conf, c["patient_id"], v["series"], k, img, z["local"], inside))
            del vol
    figs.sort(key=lambda x: -x[0])
    seen, diverse = set(), []
    for f in figs:   # strongest false region of each CN patient (variety of false-positive types)
        if f[1] not in seen:
            seen.add(f[1])
            diverse.append(f)
    paths_out = []
    for conf, pid, ph, k, img, loc, inside in diverse[:6]:
        p = figdir / f"cn_vung_gia_{pid}_{ph}_k{k:03d}.png"
        _panel([img, loc], [f"{pid} · pha {ph} · lát {k} · độ tin cậy {conf:.2f} · {'trong gan' if inside else 'ngoài gan' if inside is False else 'không rõ'}",
                            "Tumor Zoom của vùng giả"], p)
        paths_out.append(p.name)
    return recs, paths_out


def _panel(imgs, titles, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(1, len(imgs), figsize=(6.2 * len(imgs), 6.6), facecolor="white")
    for a, im, t in zip(np.ravel(ax), imgs, titles):
        a.imshow(im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=255)
        a.set_title(t, fontsize=12, color="#0B3D91")
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- C. MRI
def mri_analysis(mri):
    cfg = load_yaml(CONFIG_DIR / "cls.yaml")["mri"]
    classes = cfg["classes"]
    man = rows(ROOT / "data" / "processed" / "cls" / "mri" / "manifest.csv")
    run = ROOT / "outputs" / "runs" / "cls" / "mri_hybrid_gt_s42"
    pr = rows(run / "val_predictions_best.csv")
    ev = json.loads((run / "eval_val" / "metrics_val.json").read_text(encoding="utf-8"))
    y = np.array([int(r["label_id"]) for r in pr])
    P = np.array([[float(r[f"p_{c}"]) for c in classes] for r in pr])
    pred = P.argmax(1)
    conf = P.max(1)
    out = {"roi_true_dist": {c: int((y == i).sum()) for i, c in enumerate(classes)},
           "roi_pred_dist": {c: int((pred == i).sum()) for i, c in enumerate(classes)},
           "mean_conf_correct": round(float(conf[pred == y].mean()), 3), "mean_conf_wrong": round(float(conf[pred != y].mean()), 3),
           "wrong_with_conf_ge_0.8": round(float((conf[pred != y] >= 0.8).mean()), 3)}
    tr = [r for r in man if r["source"] == "gt" and r["split"] == "train"]
    va = [r for r in man if r["source"] == "gt" and r["split"] == "val"]
    out["train_roi_dist"] = dict(Counter(r["label"] for r in tr))
    size = {}
    for c in classes:
        a_tr = [float(r["box_w"]) * float(r["box_h"]) for r in tr if r["label"] == c]
        a_va = [float(r["box_w"]) * float(r["box_h"]) for r in va if r["label"] == c]
        size[c] = {"train_median_area": round(float(np.median(a_tr))), "val_median_area": round(float(np.median(a_va))),
                   "train_small_frac": round(float(np.mean(np.array(a_tr) < SMALL)), 3), "val_small_frac": round(float(np.mean(np.array(a_va) < SMALL)), 3)}
    out["lesion_size"] = size
    rng = np.random.default_rng(0)
    root = ROOT / "data" / "processed" / "cls" / "mri"
    inten = {}
    for split, rs in (("train", tr), ("val", va)):
        smp = [rs[i] for i in rng.choice(len(rs), min(600, len(rs)), replace=False)]
        vals = [cv2.imread(str(root / r["local_path"]), cv2.IMREAD_GRAYSCALE).mean() for r in smp]
        inten[split] = {"mean": round(float(np.mean(vals)), 1), "std": round(float(np.std(vals)), 1)}
    out["local_crop_intensity"] = inten
    # selective classification on patients (pipeline scores)
    lab = [c for c in mri if c["pipeline_scores"]]
    sel = []
    for t in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8):
        kept = [c for c in lab if max(c["pipeline_scores"].values()) >= t]
        sel.append({"min_prob": t, "coverage": round(len(kept) / len(lab), 3), "n": len(kept),
                    "accuracy_on_kept": round(float(np.mean([c["pipeline_label"] == c["true_label"] for c in kept])), 3) if kept else None})
    out["selective_patient"] = sel
    out["roi_metrics"] = {"gt": ev["gt_roi"]["roi"]["accuracy"], "yolo_matched": ev["yolo_roi_matched"]["roi"]["accuracy"],
                          "yolo_all": ev["yolo_roi_all"]["roi"]["accuracy"]}
    return out


# ----------------------------------------------------------------------------- D. misses
def misses(mod, paths, figdir, max_figs=6):
    cfg = load_yaml(CONFIG_DIR / "cls.yaml")[mod]
    data = ROOT / cfg["yolo_dataset"]
    params = load_roi_config(mod)
    dets, _ = r71.load_scan(ROOT / "outputs" / "predictions" / "roi_val" / mod / "raw_dets.csv")
    lesions, files = b80.gt_lesions(data, "val")
    split = {r["patient_id"]: r for r in rows(paths["project"]["splits"] / f"{mod}_split.csv")}
    recs = []
    for (pid, series), les in lesions.items():
        merged = {k: merge_boxes(d, params["merge_iou"], params["merge_containment"]) for k, d in dets.get((pid, series), {}).items()}
        tracks = filter_tracks(link_tracks(merged, params["link_iou"], params["link_max_gap"]), params["min_track_slices"], params["min_track_conf"])
        for li, g in enumerate(les):
            best = max((iou(t.boxes[k].xyxy, g.boxes[k].xyxy) for t in tracks for k in t.boxes if k in g.boxes), default=0.0)
            kmax = max(g.boxes, key=lambda k: g.boxes[k].area)
            recs.append({"modality": mod.upper(), "patient_id": pid, "series": series, "label": split[pid]["label"], "lesion": li,
                         "n_slices": g.n_slices, "max_area": round(g.boxes[kmax].area), "slice_max_area": kmax,
                         "detected": best >= 0.3, "best_iou": round(best, 3), "n_tracks_volume": len(tracks)})
    by_vol = defaultdict(list)
    for r in recs:
        by_vol[(r["patient_id"], r["series"])].append(r)
    missed_vols = [k for k, v in by_vol.items() if not any(x["detected"] for x in v)]
    by_pat = defaultdict(list)
    for r in recs:
        by_pat[r["patient_id"]].append(r)
    missed_pats = [p for p, v in by_pat.items() if not any(x["detected"] for x in v)]
    figs = []
    cand = sorted(missed_vols, key=lambda k: -max(x["max_area"] for x in by_vol[k]))[:max_figs]
    for pid, series in cand:
        r = max(by_vol[(pid, series)], key=lambda x: x["max_area"])
        k = r["slice_max_area"]
        img = cv2.imread(str(files[(pid, series)][k]), cv2.IMREAD_GRAYSCALE)
        g = lesions[(pid, series)][r["lesion"]].boxes[k].xyxy
        pred = [b.xyxy for b in merge_boxes(dets.get((pid, series), {}).get(k, []), params["merge_iou"], params["merge_containment"])]
        im = draw(img, pred, [g], [], [f"YOLO {d.conf:.2f}" for d in sorted(dets.get((pid, series), {}).get(k, []), key=lambda d: -d.conf)][:len(pred)])
        p = figdir / f"bo_sot_{mod}_{pid}_{series.replace(' ', '')}_k{k:03d}.png"
        _panel([im], [f"{mod.upper()} · {pid} ({split[pid]['label']}) · {series} · lát {k} · tổn thương {r['max_area']} px²\n"
                      f"xanh lá = nhãn chuẩn, đỏ = YOLO (nếu có) · không vùng nào trùng tổn thương"], p)
        figs.append(p.name)
    size_bins = {}
    for name, lo, hi in (("nhỏ < 32²", 0, SMALL), ("vừa 32²–96²", SMALL, 96 ** 2), ("lớn > 96²", 96 ** 2, 1e12)):
        s = [r for r in recs if lo <= r["max_area"] < hi]
        size_bins[name] = {"n": len(s), "detected": sum(r["detected"] for r in s)}
    by_class = {c: {"n": len(v), "detected": sum(r["detected"] for r in v)}
                for c, v in sorted(defaultdict(list, {c: [r for r in recs if r["label"] == c] for c in {r["label"] for r in recs}}).items())}
    by_series = {s: {"n": len(v), "detected": sum(r["detected"] for r in v)}
                 for s, v in sorted({s: [r for r in recs if r["series"] == s] for s in {r["series"] for r in recs}}.items())}
    single_slice = [r for r in recs if r["n_slices"] == 1]
    return {"lesions": len(recs), "detected": sum(r["detected"] for r in recs), "size_bins": size_bins, "by_class": by_class,
            "by_series": by_series, "volumes_with_lesion": len(by_vol), "volumes_missed": [f"{p}_{s}" for p, s in missed_vols],
            "patients_missed": missed_pats, "missed_patient_details": {p: by_pat[p] for p in missed_pats},
            "single_slice_lesions": {"n": len(single_slice), "detected": sum(r["detected"] for r in single_slice)},
            "figures": figs, "_recs": recs}


# ----------------------------------------------------------------------------- report
def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "—"


def report(R) -> None:
    A, o0, o1, oz = R["gate"], R["outcome_no_gate"], R["outcome_gate"], R["outcome_zero_loss"]
    rule, zrule, vrule = GateRule(**A["chosen"]), GateRule(**A["zero_loss"]), GateRule(**A["volume_rule"])
    L = ["# Phân tích hạn chế và cơ chế từ chối phân loại (Validation)", "",
         f"*Sinh bởi `scripts/93_limitations_analysis.py` lúc {datetime.now():%Y-%m-%d %H:%M}. Nguồn: cache pipeline `outputs/predictions/pipeline_validation/` "
         "(scripts/92_), quét YOLO Val `outputs/predictions/roi_val/`, bộ ROI phân loại, run `mri_hybrid_gt_s42`. Số liệu tổng hợp: "
         "`outputs/predictions/pipeline_validation/limitations.json`. **Chỉ Validation; không dùng Test; không huấn luyện lại; checkpoint và dữ liệu gốc không đổi.***", "",
         "## 1. Kết quả CT phải đọc theo từng nhóm bệnh nhân", "",
         "Con số 90,5% (38/42) là **chỉ trên bệnh nhân ung thư**, **không** phải accuracy trên toàn bộ CT Validation (54 BN).", "",
         "| CT Validation | Không có cổng bằng chứng (hiện tại) | **Có cổng bằng chứng (đề xuất)** | Cổng 'không mất ca ung thư' |", "|---|---|---|---|"]
    def r_(o, k):
        return o.get(k, 0)
    for lab, k, base in (("BN ung thư: phân loại đúng", "cancer_correct", "cancer"), ("BN ung thư: phân loại sai", "cancer_wrong", "cancer"),
                         ("BN ung thư: không đưa nhãn (từ chối / không có vùng)", "cancer_no_label", "cancer"),
                         ("Nhóm chứng CN: không đưa nhãn (đúng mong muốn)", "cn_no_label", "cn"),
                         ("Nhóm chứng CN: bị gán nhãn ung thư (dương tính giả)", "cn_cancer_label", "cn"),
                         ("Toàn bộ CT: kết quả chấp nhận được (ung thư đúng + CN không nhãn)", "all_acceptable", "all"),
                         ("Toàn bộ CT: đưa ra nhãn sai (ung thư sai + CN bị gán ung thư)", "all_wrong_label", "all")):
        L.append(f"| {lab} | {r_(o0, k)}/{o0[base]} ({pct(r_(o0, k), o0[base])}) | **{r_(o1, k)}/{o1[base]} ({pct(r_(o1, k), o1[base])})** | "
                 f"{r_(oz, k)}/{oz[base]} ({pct(r_(oz, k), oz[base])}) |")
    L += ["", f"- Cổng đề xuất: **chỉ phân loại khi {rule.describe_vi('pha CT')}**; nếu không đạt, trạng thái `insufficient_evidence`, "
          "**không** đưa nhãn và **không** kết luận bệnh nhân khỏe mạnh.",
          f"- Phương án 'không mất ca ung thư nào' trên Val: {zrule.describe_vi('pha CT')}.",
          f"- Cổng khi chỉ có **một file** (một pha) tải lên: {vrule.describe_vi()} "
          f"(mức volume Val: giữ {A['volume_rule_counts'].get('cancer_kept', 0)}/{A['volume_rule_counts'].get('cancer', 0)} volume có tổn thương, "
          f"từ chối {A['volume_rule_counts'].get('cn_rejected', 0)}/{A['volume_rule_counts'].get('cn', 0)} volume CN).", ""]
    loo = A["loo"]
    L += ["### Cách chọn cổng và độ lạc quan", "",
          f"- Quy tắc chọn (khai báo trước, chỉ trên Val): trong {A['grid_size']} luật (độ tin cậy cao nhất; số pha có vùng mạnh; độ dài vùng mạnh), "
          "chọn luật **từ chối được nhiều BN CN nhất** mà vẫn **phân loại ≥ 95% BN ung thư** (≥ 40/42); hòa thì giữ nhiều BN ung thư hơn, luật đơn giản hơn.",
          f"- Ngưỡng được chọn trên chính 54 BN này (chỉ 12 CN) nên **lạc quan**. Ước lượng bằng leave-one-out (chọn luật trên 53 BN, áp cho BN còn lại): "
          f"giữ {loo.get('cancer_kept', 0)}/{loo.get('cancer', 0)} BN ung thư, từ chối {loo.get('cn_rejected', 0)}/{loo.get('cn', 0)} BN CN.",
          "- Xác suất của bộ phân loại **không** tách được CN (BN CN vẫn có xác suất lớn nhất tới 0,96), nên cổng dựa trên **bằng chứng phát hiện**, không dựa trên xác suất lớp.", "",
          "| BN | Nhãn thực tế | Pipeline | YOLO cao nhất | Số pha có vùng mạnh | Số vùng | Xác suất lớp lớn nhất | Qua cổng |", "|---|---|---|---|---|---|---|---|"]
    key_n = next(k for k in A["per_patient"][0] if k.startswith("n_pha_vung"))
    for p in sorted(A["per_patient"], key=lambda p: (p["true_label"] != "CN", p["max_conf"])):
        if p["true_label"] == "CN" or not p["gate_pass"]:
            L.append(f"| {p['patient_id']} | {p['true_label']} | {p['pipeline_label']} | {p['max_conf']} | {p[key_n]} | {p['n_regions']} | "
                     f"{p['classifier_max_prob']} | {'có' if p['gate_pass'] else '**không**'} |")
    L += ["", "(Bảng trên liệt kê mọi BN CN và các BN ung thư bị cổng từ chối; đầy đủ ở `gate_per_patient.csv`.)", ""]
    B = R["cn_regions"]
    n_in = sum(1 for r in B["recs"] if r["centre_in_liver"] is True)
    n_out = sum(1 for r in B["recs"] if r["centre_in_liver"] is False)
    strong = [r for r in B["recs"] if r["max_conf"] >= 0.4]
    L += ["## 2. Vì sao nhóm chứng CT bị gán nhãn ung thư", "",
          f"- **Mô hình phân loại chỉ có 3 lớp ung thư.** Không có lớp 'không phải tổn thương', nên mọi vùng YOLO (kể cả vùng giả) đều nhận HCC / ICC / CHCC.",
          f"- **YOLO ở ngưỡng 0.12 (chọn để hạn chế bỏ sót) tạo vùng giả ở {sum(1 for c in R['_ct'] if c['true_label'] == 'CN' and c['tracks'])}/12 BN CN**: "
          f"tổng {len(B['recs'])} vùng trên {len({(r['patient_id'], r['phase']) for r in B['recs']})} volume CN; "
          f"tâm vùng nằm **trong gan** {n_in}, **ngoài gan** {n_out} (liver mask chỉ dùng để phân tích). "
          f"{len(strong)} vùng có độ tin cậy ≥ 0.4.",
          f"- Độ tin cậy vùng giả ở CN: trung vị {np.median([r['max_conf'] for r in B['recs']]):.2f}, lớn nhất {max(r['max_conf'] for r in B['recs']):.2f}; "
          f"độ dài trung vị {np.median([r['n_slices'] for r in B['recs']]):.0f} lát. Vùng thật ở BN ung thư thường lặp lại ở 3–4 pha với độ tin cậy cao.",
          "- Một vài BN CN có vùng giả mạnh và lặp lại qua các pha (ví dụ P0281, P0288): cổng không loại được; cần lớp 'không tổn thương' hoặc YOLO tốt hơn.",
          "- ROI và tiền xử lý **không** phải nguyên nhân: crop của pipeline trùng từng pixel với dữ liệu đánh giá (`reports/pipeline_val_report.md` §3).", ""]
    L += [f"![{f}]({f})" for f in B["figures"][:4]] + [""]
    C = R["mri"]
    cl = load_yaml(CONFIG_DIR / "cls.yaml")["mri"]
    nv = dict(zip(cl["classes"], cl["class_names_vi"]))
    L += ["## 3. MRI: vì sao chỉ đạt 47,4% (37/78 BN)", "",
          "| Lớp | ROI Train (GT) | ROI Val thật | ROI Val bị dự đoán là lớp này | Diện tích trung vị Train / Val (px²) | Tỉ lệ tổn thương nhỏ Train / Val |", "|---|---|---|---|---|---|"]
    for c in cl["classes"]:
        s = C["lesion_size"][c]
        L.append(f"| {nv[c]} | {C['train_roi_dist'].get(c, 0)} | {C['roi_true_dist'][c]} | {C['roi_pred_dist'][c]} | {s['train_median_area']} / {s['val_median_area']} | "
                 f"{100 * s['train_small_frac']:.0f}% / {100 * s['val_small_frac']:.0f}% |")
    L += ["", f"- **Dự đoán bị hút về HCC:** {C['roi_pred_dist'][cl['classes'][0]]} ROI Val bị đoán là HCC so với {C['roi_true_dist'][cl['classes'][0]]} ROI HCC thật; "
          "HCC chiếm nhiều ROI Train nhất.",
          f"- **Quá tự tin khi sai:** độ tin cậy trung bình khi đúng {C['mean_conf_correct']}, khi **sai** vẫn {C['mean_conf_wrong']}; "
          f"{100 * C['wrong_with_conf_ge_0.8']:.0f}% số dự đoán sai có xác suất ≥ 0.8. Đây là dấu hiệu overfit (train acc 0.97 so với Val 0.48; val loss nhỏ nhất ở epoch 1).",
          f"- **Không phải lỗi nhãn:** nhãn MRI khớp 100% giữa JSON, 8 chuỗi và split (`reports/cls_data_report.md`). **Không phải lỗi tiền xử lý:** crop trùng từng pixel; "
          f"cường độ trung bình crop local Train {C['local_crop_intensity']['train']['mean']} ± {C['local_crop_intensity']['train']['std']}, "
          f"Val {C['local_crop_intensity']['val']['mean']} ± {C['local_crop_intensity']['val']['std']}.",
          f"- **ROI do YOLO làm giảm thêm:** ROI acc GT {100 * C['roi_metrics']['gt']:.1f}% → YOLO khớp GT {100 * C['roi_metrics']['yolo_matched']:.1f}% → "
          f"mọi vùng YOLO {100 * C['roi_metrics']['yolo_all']:.1f}%; 83% vùng YOLO MRI là vùng giả.",
          "- **Không có chuỗi MRI nào tốt:** theo chuỗi, oracle chỉ 40–56% (`bao_cao_pipeline_validation.md` §2), nên không phải do một chuỗi kém.",
          "- **Giả thuyết chưa kiểm chứng:** mô hình xử lý từng chuỗi riêng lẻ, trong khi phân biệt u gan trên MRI dựa vào mẫu ngấm thuốc qua nhiều pha; "
          "Val nhỏ (7–25 BN/lớp) và là tập val chính thức LLD-MMRI2023 (có thể khác phân bố so với Train).", "",
          "Phân loại có chọn lọc (chỉ trả nhãn khi xác suất lớn nhất ≥ ngưỡng), mức BN, pipeline:", "",
          "| Xác suất tối thiểu | Tỉ lệ BN được trả nhãn | Accuracy trên BN được trả nhãn |", "|---|---|---|"]
    for s in C["selective_patient"]:
        L.append(f"| {s['min_prob']} | {100 * s['coverage']:.0f}% ({s['n']}) | {_p(s['accuracy_on_kept'])} |")
    hi = max((s for s in C["selective_patient"] if s["n"] >= 15), key=lambda s: s["accuracy_on_kept"] or 0, default=None)
    if hi:
        L += ["", f"- Chỉ giữ ca tự tin làm tăng accuracy trên phần được giữ (ví dụ ngưỡng {hi['min_prob']}: {_p(hi['accuracy_on_kept'])} trên "
              f"{hi['n']}/{len(mri_c := [c for c in R['_mri'] if c['pipeline_scores']])} BN, tức chỉ {100 * hi['coverage']:.0f}% số BN), "
              "nhưng ngưỡng này được nhìn trên chính Val nhỏ (khoảng 20 BN) nên **chưa đủ tin cậy**, và phần lớn BN sẽ không có nhãn."]
    shift = [c for c in cl["classes"] if abs(C["lesion_size"][c]["train_small_frac"] - C["lesion_size"][c]["val_small_frac"]) >= 0.15]
    if shift:
        L.append("- **Phân bố kích thước khác nhau giữa Train và Val** ở: " + ", ".join(
            f"{nv[c]} (tổn thương nhỏ {100 * C['lesion_size'][c]['train_small_frac']:.0f}% Train / {100 * C['lesion_size'][c]['val_small_frac']:.0f}% Val)"
            for c in shift) + ". Đây là một nguồn chênh lệch giữa Train và Val có thể đo được.")
    L += ["- Kết luận: **không đưa nhãn MRI vào website như kết quả chẩn đoán.** Nếu cần minh họa, chỉ hiện ở chế độ nghiên cứu, kèm cảnh báo đỏ.", ""]
    L += ["## 4. Các trường hợp YOLO bỏ sót", ""]
    for mod in ("ct", "mri"):
        D = R["misses"][mod]
        L += [f"### {mod.upper()}: {D['detected']}/{D['lesions']} tổn thương 3D được tìm thấy ({pct(D['detected'], D['lesions'])})", "",
              "| Kích thước tổn thương | Số tổn thương | Được tìm thấy |", "|---|---|---|"]
        for k, v in D["size_bins"].items():
            L.append(f"| {k} | {v['n']} | {v['detected']} ({pct(v['detected'], v['n'])}) |")
        L += ["", f"- Tổn thương chỉ xuất hiện trên **1 lát**: tìm thấy {D['single_slice_lesions']['detected']}/{D['single_slice_lesions']['n']}.",
              "- Theo lớp: " + ", ".join(f"{k} {v['detected']}/{v['n']}" for k, v in D["by_class"].items()) + ".",
              f"- Theo {'pha' if mod == 'ct' else 'chuỗi'}: " + ", ".join(f"{k} {v['detected']}/{v['n']}" for k, v in D["by_series"].items()) + ".",
              f"- Volume có tổn thương mà YOLO không tìm thấy tổn thương nào: **{len(D['volumes_missed'])}/{D['volumes_with_lesion']}**; "
              f"BN bị bỏ sót **mọi** tổn thương: **{len(D['patients_missed'])}**"
              + (" (" + ", ".join(D["patients_missed"]) + ")" if D["patients_missed"] else "") + ".", ""]
        for p, ls in D["missed_patient_details"].items():
            L.append(f"  - {p}: " + "; ".join(f"{x['series']} {x['max_area']} px² ({x['n_slices']} lát)" for x in ls[:8]))
        L += [""] + [f"![{f}]({f})" for f in D["figures"][:3]] + [""]
    L += ["- YOLO bỏ sót chủ yếu **tổn thương nhỏ và chỉ trên 1–2 lát**. Ở mức BN, CT không bỏ sót BN ung thư nào; MRI có 2 BN bị bỏ sót hoàn toàn. "
          "Theo quy tắc của dự án, các ca này là **lỗi của pipeline** và vẫn được tính khi đánh giá.", ""]
    L += ["## 5. Cấu hình cuối đề xuất cho website (chưa triển khai)", "",
          "| Thành phần | CT | MRI |", "|---|---|---|",
          "| Detection | YOLO11n CT, conf 0.12 | YOLO11n MRI, conf 0.12 |",
          "| Tumor Zoom | local + context, không lọc vùng | như CT |",
          f"| Phân loại | Hybrid CT (`ct_hybrid_gt_s42`), gộp trung bình xác suất | **Tắt mặc định.** Chỉ hiển thị phát hiện và Tumor Zoom; nhãn MRI chỉ ở chế độ 'nghiên cứu' kèm cảnh báo đỏ |",
          f"| Cổng bằng chứng | **Bật:** {rule.describe_vi('pha CT')}; một file: {vrule.describe_vi()} | không áp dụng (không phân loại) |",
          "| Khi không có vùng / không qua cổng | 'Không phát hiện vùng nghi ngờ' / 'Chưa đủ bằng chứng', **không** kết luận khỏe mạnh | 'Không phát hiện vùng nghi ngờ' |",
          "| Cách trình bày kết quả CT | Ghi rõ: kết quả Val **trên BN ung thư**; mô hình không nhận biết người không có bệnh gan | — |", "",
          "Các giá trị đã ghi vào `configs/cls.yaml → inference.ct.evidence_gate` với `enabled: false` (**chờ người dùng duyệt**). "
          "Bật bằng `enabled: true`; pipeline sẽ trả `insufficient_evidence` khi không đạt. Không đánh giá Test trước khi chốt cấu hình này.", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bao_cao_han_che.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def _p(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    paths = load_paths()
    OUT.mkdir(parents=True, exist_ok=True)
    ct, mri = cases("ct"), cases("mri")
    R = {"time": datetime.now().isoformat(timespec="seconds"), "split": "val", "_ct": ct, "_mri": mri}
    R["gate"] = gate_analysis(ct)
    rule, zrule = GateRule(**R["gate"]["chosen"]), GateRule(**R["gate"]["zero_loss"])
    R["outcome_no_gate"], R["outcome_gate"], R["outcome_zero_loss"] = outcome_table(ct, None), outcome_table(ct, rule), outcome_table(ct, zrule)
    print("A gate:", R["gate"]["chosen"], R["outcome_gate"], "| zero-loss:", R["gate"]["zero_loss"], "| LOO:", {k: v for k, v in R["gate"]["loo"].items() if not k.startswith("rule")}, flush=True)
    recs, figs = cn_regions(ct, paths, OUT)
    R["cn_regions"] = {"recs": recs, "figures": figs}
    print("B CN regions:", len(recs), Counter(r["centre_in_liver"] for r in recs), flush=True)
    R["mri"] = mri_analysis(mri)
    print("C MRI:", {k: R["mri"][k] for k in ("mean_conf_correct", "mean_conf_wrong", "roi_pred_dist")}, flush=True)
    R["misses"] = {m: misses(m, paths, OUT) for m in ("ct", "mri")}
    for m in ("ct", "mri"):
        D = R["misses"][m]
        print(f"D {m}:", D["detected"], "/", D["lesions"], D["size_bins"], "patients missed", D["patients_missed"], flush=True)
    report(R)
    write_csv(OUT / "gate_per_patient.csv", R["gate"]["per_patient"])
    write_csv(OUT / "cn_vung_gia.csv", recs)
    write_csv(OUT / "ton_thuong_bo_sot.csv", [r for m in ("ct", "mri") for r in R["misses"][m]["_recs"] if not r["detected"]])
    for m in ("ct", "mri"):
        R["misses"][m].pop("_recs")
    R.pop("_ct")
    R.pop("_mri")
    (ROOT / "outputs" / "predictions" / "pipeline_validation" / "limitations.json").write_text(
        json.dumps(R, indent=1, ensure_ascii=False, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else float(o)), encoding="utf-8")
    print("written", OUT / "bao_cao_han_che.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
