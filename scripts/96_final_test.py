"""Final Test, step 2: the OFFICIAL, one-time evaluation of the locked pipeline on the Test split (CT and MRI).

Stages (in this order):
  selfcheck  (before the lock) re-runs the evaluation code below on a few Validation patients and compares with the
             Validation caches -> proves that Test is evaluated with exactly the code that produced the Val numbers.
  export     Test GT only: every lesion slice of every usable Test volume, with the SAME exporter and options as the Val
             data the models were selected on (scripts/20_build_yolo_pilot.py export_volume; options read from the Val
             datasets' build_args.json). Negative slices are not needed. -> data/processed/final_test/yolo_<mod>/
  run        verifies the lock (scripts/95_final_test_lock.py) -> refuses on ANY difference in code, configs, splits or
             checkpoints; runs once (a started run can only be resumed for missing patients, never redone).
             Pipeline = YOLO boxes only (never GT) with the locked config (CT evidence gate on, MRI off).
             GT is used afterwards for scoring only (detection matching, oracle reference).
  report     tables, confusion matrices, metrics and the Vietnamese report in reports/final_test/.
Nothing is tuned here: no threshold, model, aggregation or gate value is read from Test results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.classification.inference import RoiInput  # noqa: E402
from livercancer.classification.metrics import aggregate  # noqa: E402
from livercancer.config import CONFIG_DIR, load_paths, load_yaml  # noqa: E402
from livercancer.data.sources import image_path  # noqa: E402
from livercancer.roi.boxes import iou  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b20 = _load("b20", "20_build_yolo_pilot.py")
b80 = _load("b80", "80_build_cls_dataset.py")

OUT = ROOT / "outputs" / "final_test"
REP = ROOT / "reports" / "final_test"
DATA = ROOT / "data" / "processed" / "final_test"
VAL_CACHE = ROOT / "outputs" / "predictions" / "pipeline_validation_gate"   # Val run with the gate (= locked config for CT)
SMALL = 32 * 32
MODS = ("ct", "mri")


def rows(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def write_csv(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in recs:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(2 ** 22), b""):
            h.update(chunk)
    return h.hexdigest()


def agg_label(prob_list, method, classes):
    if not prob_list:
        return None, {}
    v = aggregate(np.array(prob_list), method)
    return classes[int(v.argmax())], {c: round(float(x), 4) for c, x in zip(classes, v)}


# ----------------------------------------------------------------------------- shared per-patient evaluation
def oracle_rois(mod: str, pid: str, series: list[str], gt_les: dict, files: dict) -> list[tuple[str, RoiInput]]:
    """GT-ROI crops built exactly like the classification dataset (scripts/80 stage_gt, Val setting), in memory."""
    cfg = load_yaml(CONFIG_DIR / "cls.yaml")
    n_per, gap = cfg["roi"]["slices_per_lesion"]["val"], cfg[mod]["slice_min_gap"]
    params = b80.load_roi_config(mod)
    out = []
    for s in series:
        for li, t in enumerate(gt_les.get((pid, s), [])):
            for k in b80.choose_slices(t, n_per, gap):
                img = cv2.imread(str(files[(pid, s)][k]), cv2.IMREAD_GRAYSCALE)
                z = b80.zoom(img, t.boxes[k].xyxy, params["crop_size"], params["margin_frac"], params["min_margin_px"],
                             params["context_scale"], params["context_min_px"])
                out.append((s, RoiInput(z["local"], z["context"], f"{pid}_{s}_L{li}_k{k:03d}", li, k)))
    return out


def evaluate_case(pipe, paths, mod, pid, series, true, gt_les, orois, classes) -> tuple[dict, object]:
    """Same computation as scripts/92_pipeline_validation.py (run stage); GT is used only after the pipeline ran."""
    method = pipe.classifier.aggregation
    name_of = {s: s.replace(" ", "") for s in series}
    res = pipe.run_nifti([image_path(paths, mod, pid, s) for s in series], pid, [name_of[s] for s in series])
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
    o_label, o_scores, o_vol = None, {}, {}
    if orois:
        c = pipe.classifier.classify([r for _, r in orois])
        pr = [[x["scores"][k] for k in classes] for x in c.per_roi]
        o_label, o_scores = agg_label(pr, method, classes)
        for s in series:
            idx = [j for j, (ss, _) in enumerate(orois) if ss == s]
            o_vol[name_of[s]] = agg_label([pr[j] for j in idx], method, classes)[0] if idx else None
    for v in vol_info:
        v["oracle_label"] = o_vol.get(v["volume"])
    cl = res.classification or {}
    case = {"patient_id": pid, "modality": mod, "true_label": true, "status": res.status, "pipeline_label": res.label,
            "pipeline_scores": res.scores, "ungated_label": cl.get("label") if cl.get("status") == "ok" else None,
            "matched_only_label": m_label, "matched_only_scores": m_scores,
            "oracle_label": o_label, "oracle_scores": o_scores, "n_oracle_rois": len(orois), "n_rois": len(per_roi), "n_fp_rois": n_fp_rois,
            "aggregation": method, "volumes": vol_info, "tracks": tracks_info, "timings_s": res.timings_s, "device": res.device,
            "evidence": res.evidence, "message": res.message}
    return case, res


# ----------------------------------------------------------------------------- selfcheck on Validation
def selfcheck(args) -> int:
    from livercancer.pipeline import LiverCancerPipeline
    paths, cfg = load_paths(), load_yaml(CONFIG_DIR / "cls.yaml")
    bad = 0
    for mod in MODS:
        data = ROOT / cfg[mod]["yolo_dataset"]
        vols = [r for r in rows(data / "volumes.csv") if r["split"] == "val" and r["status"] == "ok"]
        split = {r["patient_id"]: r for r in rows(paths["project"]["splits"] / f"{mod}_split.csv")}
        gt_les, files = b80.gt_lesions(data, "val")
        pipe = LiverCancerPipeline(mod, args.device, evidence_gate="on")   # the Val cache was produced with --evidence-gate on
        cache = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (VAL_CACHE / mod / "cases").glob("*.json")}
        pats = sorted({v["patient_id"] for v in vols})
        pick = [p for p in pats if cache[p]["true_label"] == "CN"][:1] + [p for p in pats if cache[p]["status"] == "insufficient_evidence"][:1]
        pick += [p for p in pats if p not in pick][: args.selfcheck_n - len(pick)]
        for pid in pick:
            series = [v["series"] for v in vols if v["patient_id"] == pid]
            case, _ = evaluate_case(pipe, paths, mod, pid, series, split[pid]["label"], gt_les,
                                    oracle_rois(mod, pid, series, gt_les, files), cfg[mod]["classes"])
            ref = cache[pid]
            diffs = [k for k in ("status", "pipeline_label", "oracle_label", "matched_only_label", "n_rois", "n_fp_rois", "n_oracle_rois")
                     if case[k] != ref[k]]
            for a, b in ((case["pipeline_scores"], ref["pipeline_scores"]), (case["oracle_scores"], ref["oracle_scores"])):
                if set(a) != set(b) or any(abs(a[c] - b[c]) > 2e-3 for c in a):
                    diffs.append("scores")
            key = lambda t: (t["volume"], t["track_id"], t["matched"], t["n_slices"], round(t["max_conf"], 3))  # noqa: E731
            if [key(t) for t in case["tracks"]] != [key(t) for t in ref["tracks"]]:
                diffs.append("tracks")
            if [(v["n_gt_lesions"], v["n_gt_detected"]) for v in case["volumes"]] != [(v["n_gt_lesions"], v["n_gt_detected"]) for v in ref["volumes"]]:
                diffs.append("gt_matching")
            bad += bool(diffs)
            print(f"  selfcheck {mod} {pid}: {'IDENTICAL' if not diffs else 'DIFF ' + ', '.join(diffs)}", flush=True)
    print("SELFCHECK", "PASS" if not bad else f"FAIL ({bad})")
    return 1 if bad else 0


# ----------------------------------------------------------------------------- lock verification
def verify_lock() -> dict:
    lp = OUT / "locked_config.json"
    if not lp.exists():
        sys.exit("STOP: no lock (run scripts/95_final_test_lock.py first)")
    lock = json.loads(lp.read_text(encoding="utf-8"))
    problems = [f"check {c['code']} failed" for c in lock.get("checks", []) if not c["ok"]]
    if not lock.get("checks"):
        problems.append("lock has no independence checks")
    for f, h in lock["code_sha256"].items():
        if not (ROOT / f).exists() or sha256(ROOT / f) != h:
            problems.append(f"code changed: {f}")
    tracked = subprocess.run(["git", "ls-files", "src", "scripts", "configs"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.split()
    new = [f for f in tracked if f.endswith((".py", ".yaml", ".yml")) and f not in lock["code_sha256"]]
    problems += [f"code added after lock: {f}" for f in new]
    st = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    problems += [f"uncommitted: {l}" for l in st.splitlines() if l.strip() and "__pycache__" not in l]
    sp = load_paths()["project"]["splits"]
    problems += [f"split changed: {f}" for f, h in lock["splits"].items() if sha256(sp / f) != h]
    for m, x in lock["modalities"].items():
        if sha256(ROOT / x["yolo"]["weights"]) != x["yolo"]["sha256"]:
            problems.append(f"{m} YOLO weights changed")
        if sha256(ROOT / x["classifier"]["checkpoint"]) != x["classifier"]["sha256"]:
            problems.append(f"{m} classifier checkpoint changed")
    if problems:
        sys.exit("STOP: lock verification failed:\n  " + "\n  ".join(problems))
    print(f"lock verified: commit {lock['git']['commit'][:10]}, {len(lock['code_sha256'])} code files, checkpoints and splits unchanged")
    return lock


# ----------------------------------------------------------------------------- export Test GT
def _export_job(job: dict) -> dict:
    """Module-level wrapper: worker processes (spawn on Windows) re-import this script, not the importlib alias b20."""
    return b20.export_volume(job)


def export(args) -> int:
    lock = verify_lock()
    paths, cfg = load_paths(), load_yaml(CONFIG_DIR / "cls.yaml")
    for mod in args.modality:
        val_args = json.loads((ROOT / cfg[mod]["yolo_dataset"] / "build_args.json").read_text(encoding="utf-8"))
        opt = {"seed": val_args["seed"], "pos_stride": {"test": 1}, "neg_gap": val_args["neg_gap"], "min_area": val_args["min_area"],
               "max_channel_diff": val_args["max_channel_diff"], "mri_min_json_iou": val_args["mri_min_json_iou"]}
        assert val_args["pos_stride"]["val"] == 1
        out = DATA / f"yolo_{mod}"
        if (out / "slices.csv").exists():
            print(f"{mod}: Test GT already exported ({out})")
            continue
        for d in ("images", "labels"):
            (out / d / "test").mkdir(parents=True, exist_ok=True)
        vols = (b20.ct_volumes if mod == "ct" else b20.mri_volumes)(paths, {"test"})
        assert {v["split"] for v in vols} == {"test"} and len({v["patient_id"] for v in vols}) == lock["n_test"][mod]
        todo = [v for v in vols if v["status"] == "ok"]
        print(f"{mod.upper()}: {len(vols)} Test volumes, exporting GT of {len(todo)}; excluded "
              f"{dict(Counter(v['status'] for v in vols if v['status'] != 'ok'))}", flush=True)
        notes, slices = {}, []
        with ProcessPoolExecutor(args.workers) as ex:
            futs = [ex.submit(_export_job, {"vol": v, "out": str(out), "opt": opt}) for v in todo]
            for n, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                notes[r["key"]] = r
                slices += r["slices"]
                if n % 100 == 0 or n == len(futs):
                    print(f"  {n}/{len(futs)} volumes, {len(slices)} lesion slices", flush=True)
        for v in vols:
            r = notes.get((v["patient_id"], v["series"]))
            if r:
                if r["error"]:
                    v["status"] = f"error:{r['error']}"
                elif r.get("corrupt"):
                    v["status"], v["detail"] = "excluded:source_file_corrupt", r["corrupt"]
                v["n_pos"] = sum(s["kind"] == "pos" for s in r["slices"])
                v.update({k: r["note"].get(k, "") for k in ("converted", "channel_diff_frac", "n_pos_avail", "skipped_all_fragments",
                                                             "skipped_no_json_box", "skipped_low_json_iou")})
            for k in ("json_2d", "image", "mask", "liver"):
                v.pop(k, None)
        b20.write_csv(vols, out / "volumes.csv")
        b20.write_csv(sorted(slices, key=lambda s: s["file"]), out / "slices.csv")
        (out / "build_args.json").write_text(json.dumps({"split": "test", "opt": opt, "source": "scripts/96_final_test.py export",
                                                         "val_build_args": val_args}, indent=2), encoding="utf-8")
        err = [v for v in vols if v["status"].startswith("error")]
        print(f"  done: {len(slices)} lesion slices; errors {len(err)}")
        for v in err[:10]:
            print("   ", v["patient_id"], v["series"], v["status"])
        if err:
            return 1
    return 0


# ----------------------------------------------------------------------------- run (once)
def run(args) -> int:
    from livercancer.pipeline import LiverCancerPipeline
    from livercancer.pipeline_viz import render_case
    lock = verify_lock()
    paths, cfg = load_paths(), load_yaml(CONFIG_DIR / "cls.yaml")
    marker = OUT / "test_run.json"
    state = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
    if state and not args.resume:
        sys.exit(f"STOP: the official Test run was already started at {state['started']} — it is run ONCE. "
                 "Use --resume only to finish patients missing after an interruption.")
    if not state:
        state = {"started": datetime.now().isoformat(timespec="seconds"), "lock_commit": lock["git"]["commit"], "resumed": []}
    else:
        state["resumed"].append(datetime.now().isoformat(timespec="seconds"))
    OUT.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(state, indent=1), encoding="utf-8")
    for mod in args.modality:
        data = DATA / f"yolo_{mod}"
        classes = cfg[mod]["classes"]
        names_vi = dict(zip(classes, cfg[mod]["class_names_vi"]))
        allv = rows(data / "volumes.csv")
        vols = [r for r in allv if r["status"] == "ok"]
        split = {r["patient_id"]: r for r in rows(paths["project"]["splits"] / f"{mod}_split.csv")}
        pats = sorted({v["patient_id"] for v in allv})
        assert all(split[p]["split"] == "test" for p in pats) and len(pats) == lock["n_test"][mod]
        gt_les, files = b80.gt_lesions(data, "test")
        pipe = LiverCancerPipeline(mod, args.device, evidence_gate="config")
        g = lock["modalities"][mod]["evidence_gate"]
        assert (pipe.gate is not None) == g["enabled"], "evidence gate differs from the lock"
        assert pipe.detector.conf == lock["modalities"][mod]["yolo"]["conf"] and pipe.classifier.aggregation == lock["modalities"][mod]["classifier"]["aggregation"]
        cdir, fdir = OUT / mod / "cases", REP / "figures" / mod
        cdir.mkdir(parents=True, exist_ok=True)
        fdir.mkdir(parents=True, exist_ok=True)
        for i, pid in enumerate(pats):
            cf = cdir / f"{pid}.json"
            if cf.exists():
                continue
            series = [v["series"] for v in vols if v["patient_id"] == pid]
            true = split[pid]["label"]
            if not series:   # no usable volume: reported, never silently dropped
                case = {"patient_id": pid, "modality": mod, "true_label": true, "status": "no_usable_volume", "pipeline_label": None,
                        "excluded_volumes": {v["series"]: v["status"] for v in allv if v["patient_id"] == pid}}
                cf.write_text(json.dumps(case, indent=1, ensure_ascii=False), encoding="utf-8")
                print(f"  [{i + 1}/{len(pats)}] {pid} NO USABLE VOLUME", flush=True)
                continue
            case, res = evaluate_case(pipe, paths, mod, pid, series, true, gt_les, oracle_rois(mod, pid, series, gt_les, files), classes)
            case["excluded_volumes"] = {v["series"]: v["status"] for v in allv if v["patient_id"] == pid and v["status"] != "ok"}
            if true not in classes:
                verdict = "CN_khong_co_vung" if res.status == "no_lesion_detected" else "CN_tu_choi" if res.status != "ok" else "CN_bi_gan_nhan_ung_thu"
            else:
                verdict = "khong_co_vung" if res.status == "no_lesion_detected" else "tu_choi" if res.status != "ok" else (
                    "dung" if res.label == true else "sai")
            case["figure"] = f"figures/{mod}/{pid}_{verdict}.png"
            note = (f"Đối chiếu (oracle, vùng cắt theo nhãn chuẩn): {names_vi.get(case['oracle_label'], case['oracle_label'])}" if case["oracle_label"] else
                    "Không có vùng tổn thương chuẩn (GT) để đối chiếu." if true in classes else
                    "Bệnh nhân nhóm chứng (CN): không có lớp này trong mô hình 3 lớp.")
            render_case(res, REP / case["figure"], f"TEST · {mod.upper()} · ca {pid}", true if true in classes else None,
                        case["oracle_scores"] or None, names_vi, note)
            cf.write_text(json.dumps(case, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"  [{i + 1}/{len(pats)}] {pid} {true:32s} {res.status:22s} pipeline={res.label} oracle={case['oracle_label']} "
                  f"tracks={len(case['tracks'])} matched={sum(t['matched'] for t in case['tracks'])} {res.timings_s.get('total')}s", flush=True)
    state["finished"] = datetime.now().isoformat(timespec="seconds")
    marker.write_text(json.dumps(state, indent=1), encoding="utf-8")
    return 0


# ----------------------------------------------------------------------------- metrics
def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    m, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, m - h), min(1.0, m + h))


def outcome(c):
    if c["status"] == "insufficient_evidence":
        return "rejected"
    if c["status"] == "no_lesion_detected":
        return "no_region"
    if c["status"] != "ok":
        return "failed"
    return "correct" if c["pipeline_label"] == c["true_label"] else "wrong"


def cls_metrics(cases, classes, key="pipeline_label"):
    """Patient level over labelled patients; a missing label counts as wrong (never as correct)."""
    k = len(classes)
    cm = np.zeros((k, k + 1), int)            # last column = no label
    for c in cases:
        p = c.get(key)
        cm[classes.index(c["true_label"]), classes.index(p) if p in classes else k] += 1
    per = {}
    for i, n in enumerate(classes):
        tp, sup, pred = cm[i, i], cm[i].sum(), cm[:k, i].sum()
        pr, rc = (tp / pred if pred else 0.0), (tp / sup if sup else 0.0)
        per[n] = {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0, "support": int(sup)}
    pres = [n for n in classes if per[n]["support"]]
    n = int(cm.sum())
    correct = int(np.trace(cm[:, :k]))
    labelled = int(cm[:, :k].sum())
    return {"n": n, "correct": correct, "labelled": labelled, "accuracy_all": correct / n if n else 0.0,
            "accuracy_all_ci95": wilson(correct, n), "accuracy_labelled": correct / labelled if labelled else 0.0,
            "coverage": labelled / n if n else 0.0, "macro_f1": float(np.mean([per[x]["f1"] for x in pres])) if pres else 0.0,
            "balanced_accuracy": float(np.mean([per[x]["recall"] for x in pres])) if pres else 0.0,
            "per_class": per, "confusion": cm.tolist()}


def bootstrap(cases, classes, key, reps=2000, seed=42):
    rng = np.random.default_rng(seed)
    acc, f1 = [], []
    for _ in range(reps):
        s = [cases[i] for i in rng.integers(0, len(cases), len(cases))]
        m = cls_metrics(s, classes, key)
        acc.append(m["accuracy_all"])
        f1.append(m["macro_f1"])
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))  # noqa: E731
    return {"accuracy_all": q(acc), "macro_f1": q(f1)}


def detection(cases):
    vs = [v for c in cases if c.get("volumes") for v in c["volumes"]]
    les = [v for v in vs if v["n_gt_lesions"]]
    lab = [c for c in cases if any(v["n_gt_lesions"] for v in c.get("volumes", []))]
    return {"volumes": len(vs), "volumes_with_lesion": len(les), "volumes_lesion_detected": sum(v["n_gt_detected"] > 0 for v in les),
            "lesions": sum(v["n_gt_lesions"] for v in vs), "lesions_detected": sum(v["n_gt_detected"] for v in vs),
            "small": sum(v["n_small_gt"] for v in vs), "small_detected": sum(v["n_small_detected"] for v in vs),
            "patients_with_gt": len(lab), "patients_no_true_lesion_found": sum(1 for c in lab if not any(t["matched"] for t in c["tracks"])),
            "patients_no_region": sum(1 for c in cases if c["status"] == "no_lesion_detected"),
            "tracks": sum(len(c.get("tracks", [])) for c in cases),
            "false_tracks": sum(1 for c in cases for t in c.get("tracks", []) if not t["matched"]),
            "false_tracks_per_volume": (sum(1 for c in cases for t in c.get("tracks", []) if not t["matched"]) / len(vs)) if vs else 0.0}


def counts(cases):
    return {o: sum(outcome(c) == o for c in cases) for o in ("correct", "wrong", "rejected", "no_region", "failed")}


# ----------------------------------------------------------------------------- report
def pct(k, n, d=1):
    return f"{k}/{n} ({100 * k / n:.{d}f}%)" if n else "–"


def ci_txt(t):
    return f"{100 * t[0]:.1f}–{100 * t[1]:.1f}%"


def plot_cm(cm, rows_lab, cols_lab, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams["font.family"] = "DejaVu Sans"
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(1.25 * len(cols_lab) + 2.6, 0.75 * len(rows_lab) + 1.9), facecolor="white")
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11, color="white" if norm[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks(range(len(cols_lab)), cols_lab, rotation=30, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(rows_lab)), rows_lab, fontsize=9.5)
    ax.set_xlabel("Hệ thống trả về", fontsize=10)
    ax.set_ylabel("Nhãn thật", fontsize=10)
    ax.set_title(title, fontsize=11, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def load_cases(base: Path, mod: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((base / mod / "cases").glob("*.json"))]


def report(args) -> int:
    from livercancer.pipeline_viz import short
    cfg = load_yaml(CONFIG_DIR / "cls.yaml")
    lock = json.loads((OUT / "locked_config.json").read_text(encoding="utf-8"))
    state = json.loads((OUT / "test_run.json").read_text(encoding="utf-8"))
    REP.mkdir(parents=True, exist_ok=True)
    M, pat_rows, vol_rows = {"lock_commit": lock["git"]["commit"], "run": state}, [], []
    L = ["# Đánh giá chính thức trên tập Test (chạy một lần)", "",
         f"_Sinh bởi `scripts/96_final_test.py --stage report` lúc {datetime.now():%Y-%m-%d %H:%M}. Lần chạy Test: bắt đầu {state['started']}, "
         f"kết thúc {state.get('finished', '?')}{', có chạy tiếp sau gián đoạn' if state.get('resumed') else ''}. "
         f"Cấu hình khóa trước khi chạy: commit `{lock['git']['commit'][:10]}`, `reports/final_test/locked_config.json`; "
         "kiểm tra độc lập: `reports/final_test/kiem_tra_doc_lap.md`. Không huấn luyện lại, không đổi checkpoint, ngưỡng, cách gộp hay cổng._", "",
         "**Cách đếm (mỗi BN thuộc đúng một nhóm):** *Đúng* / *Sai* (hệ thống trả nhãn) · *Từ chối* (Evidence Gate không cho nhãn) · "
         "*YOLO không có vùng* (không phát hiện vùng nào). Ca bị từ chối hoặc không có vùng **không** được tính là đúng. "
         "Pipeline chỉ dùng box của YOLO; nhãn chuẩn (GT) chỉ dùng **sau** khi chạy để chấm điểm. "
         "Kết quả *oracle* (phân loại trên vùng cắt theo GT) chỉ để đối chiếu, **không** phải kết quả của hệ thống.", ""]
    for mod in MODS:
        classes = cfg[mod]["classes"]
        nvi = dict(zip(classes, cfg[mod]["class_names_vi"]))
        sv = lambda c: short(nvi.get(c, c)) if c in nvi else (c or "—")  # noqa: E731
        cases = load_cases(OUT, mod)
        val_cases = load_cases(VAL_CACHE, mod)
        lab, ctrl = [c for c in cases if c["true_label"] in classes], [c for c in cases if c["true_label"] not in classes]
        vlab, vctrl = [c for c in val_cases if c["true_label"] in classes], [c for c in val_cases if c["true_label"] not in classes]
        failed = [c for c in cases if outcome(c) == "failed"]
        m = {"n_patients": len(cases), "n_lesion_patients": len(lab), "n_controls": len(ctrl), "failed": [c["patient_id"] for c in failed],
             "counts": counts(lab), "pipeline": cls_metrics(lab, classes), "bootstrap": bootstrap(lab, classes, "pipeline_label"),
             "oracle": cls_metrics([c for c in lab if c.get("volumes")], classes, "oracle_label"),
             "ungated_ablation": cls_metrics([c for c in lab if c.get("volumes")], classes, "ungated_label"),
             "detection": detection([c for c in cases if c.get("volumes")]),
             "val_reference": {"counts": counts(vlab), "pipeline": cls_metrics(vlab, classes), "detection": detection(val_cases)}}
        excl = Counter(s for c in cases for s in (c.get("excluded_volumes") or {}).values())
        m["excluded_volumes"] = dict(excl)
        for c in cases:
            pat_rows.append({"modality": mod, "patient_id": c["patient_id"], "true_label": c["true_label"], "status": c["status"],
                             "outcome": outcome(c) if c["true_label"] in classes else ("cancer_label" if c["status"] == "ok" else c["status"]),
                             "pipeline_label": c.get("pipeline_label") or "", "ungated_label": c.get("ungated_label") or "",
                             "oracle_label": c.get("oracle_label") or "",
                             "max_conf": (c.get("evidence") or {}).get("max_conf", max((t["max_conf"] for t in c.get("tracks", [])), default="")),
                             "n_volumes_strong": (c.get("evidence") or {}).get("n_volumes_strong", ""),
                             "n_volumes": len(c.get("volumes", [])), "n_tracks": len(c.get("tracks", [])),
                             "true_lesion_found": int(any(t["matched"] for t in c.get("tracks", []))) if c["true_label"] in classes else "",
                             "excluded_volumes": json.dumps(c.get("excluded_volumes") or {}, ensure_ascii=False)})
            for v in c.get("volumes", []):
                vol_rows.append({"modality": mod, "patient_id": c["patient_id"], "true_label": c["true_label"],
                                 **{k: v[k] for k in ("series", "n_slices", "n_tracks", "n_gt_lesions", "n_gt_detected", "n_small_gt",
                                                      "n_small_detected", "pipeline_label", "oracle_label")}})
        p, cnt, b, d = m["pipeline"], m["counts"], m["bootstrap"], m["detection"]
        lucky = [c for c in lab if outcome(c) == "correct" and not any(t["matched"] for t in c.get("tracks", []))]
        m["correct_without_true_lesion"] = [c["patient_id"] for c in lab if outcome(c) == "correct" and not any(t["matched"] for t in c.get("tracks", []))]
        m["strict_correct"] = cnt["correct"] - len(lucky)
        vlucky = sum(1 for c in vlab if outcome(c) == "correct" and not any(t["matched"] for t in c.get("tracks", [])))
        m["val_reference"]["strict_correct"] = m["val_reference"]["counts"]["correct"] - vlucky
        strict_txt = (f"- **Đúng nhưng không tìm thấy tổn thương thật** (nhãn đến từ vùng giả, đúng do may): "
                      f"{', '.join(c['patient_id'] for c in lucky) or 'không có'}. Nếu chỉ tính đúng khi YOLO tìm thấy tổn thương thật: "
                      f"Test {pct(m['strict_correct'], len(lab))}, Val {pct(m['val_reference']['strict_correct'], len(vlab))}.")
        cn_lab = [c for c in ctrl if c["status"] == "ok"]
        if mod == "ct":
            m["controls"] = {"n": len(ctrl), "cancer_label": len(cn_lab), "by_class": dict(Counter(c["pipeline_label"] for c in cn_lab)),
                             "rejected": sum(c["status"] == "insufficient_evidence" for c in ctrl),
                             "no_region": sum(c["status"] == "no_lesion_detected" for c in ctrl),
                             "cancer_label_ci95": wilson(len(cn_lab), len(ctrl)),
                             "ungated_cancer_label": sum(1 for c in ctrl if c.get("ungated_label")),
                             "val_reference": {"n": len(vctrl), "cancer_label": sum(c["status"] == "ok" for c in vctrl)}}
        else:
            grp = {c: g for g, cl in cfg["mri"]["groups"].items() for c in cl}
            labelled = [c for c in lab if c["status"] == "ok"]
            m["groups"] = {"benign_as_malignant": sum(1 for c in labelled if grp[c["true_label"]] == "Benign" and grp[c["pipeline_label"]] == "Malignant"),
                           "malignant_as_benign": sum(1 for c in labelled if grp[c["true_label"]] == "Malignant" and grp[c["pipeline_label"]] == "Benign"),
                           "group_correct": sum(1 for c in labelled if grp[c["true_label"]] == grp[c["pipeline_label"]]),
                           "n_benign": sum(grp[c["true_label"]] == "Benign" for c in lab), "n_malignant": sum(grp[c["true_label"]] == "Malignant" for c in lab),
                           "n": len(lab)}
        M[mod] = m
        # ---- confusion matrices
        cols = [sv(c) for c in classes] + ["Từ chối / không có vùng"]
        cm = [list(r) for r in p["confusion"]]
        rws = [sv(c) for c in classes]
        if mod == "ct":
            cm.append([sum(1 for c in cn_lab if c["pipeline_label"] == k) for k in classes] + [len(ctrl) - len(cn_lab)])
            rws.append("CN (nhóm chứng)")
        plot_cm(cm, rws, cols, f"TEST {mod.upper()} – pipeline (mức bệnh nhân, cấu hình đã khóa)", REP / "figures" / f"cm_test_{mod}_pipeline.png")
        plot_cm(m["oracle"]["confusion"], [sv(c) for c in classes], cols, f"TEST {mod.upper()} – ORACLE (vùng cắt theo GT, chỉ để đối chiếu)",
                REP / "figures" / f"cm_test_{mod}_oracle.png")

        # ---- text
        if mod == "ct":
            cc = m["controls"]
            ok_all = cnt["correct"] + cc["n"] - cc["cancer_label"]
            L += ["## 1. CT (PLC-CECT) — HCC / ICC / CHCC, Evidence Gate bật", "",
                  f"Test CT: {len(cases)} BN = {len(lab)} BN ung thư ({', '.join(f'{k} {v}' for k, v in sorted(Counter(c['true_label'] for c in lab).items()))}) "
                  f"+ {len(ctrl)} BN nhóm chứng CN. Volume bị loại theo đúng quy tắc của Val: "
                  f"{', '.join(f'{k} {v}' for k, v in excl.items()) or 'không có'}. Lỗi khi chạy: {len(failed)}.", "",
                  "### 1a. Nhóm ung thư", "",
                  "| | Đúng | Sai | Từ chối phân loại | YOLO không có vùng | Accuracy trên toàn bộ BN (KTC 95%) | Accuracy trên ca có nhãn | Macro-F1 (bỏ sót = sai) |",
                  "|---|---|---|---|---|---|---|---|",
                  f"| **Test** | {cnt['correct']} | {cnt['wrong']} | {cnt['rejected']} | {cnt['no_region']} | {pct(p['correct'], p['n'])} ({ci_txt(p['accuracy_all_ci95'])}) | "
                  f"{pct(p['correct'], p['labelled'])} | {p['macro_f1']:.3f} (bootstrap {b['macro_f1'][0]:.3f}–{b['macro_f1'][1]:.3f}) |"]
            vr = m["val_reference"]
            L += [f"| Val (tham chiếu) | {vr['counts']['correct']} | {vr['counts']['wrong']} | {vr['counts']['rejected']} | {vr['counts']['no_region']} | "
                  f"{pct(vr['pipeline']['correct'], vr['pipeline']['n'])} | {pct(vr['pipeline']['correct'], vr['pipeline']['labelled'])} | {vr['pipeline']['macro_f1']:.3f} |", ""]
            L += ["| Lớp | Precision | Recall | F1 | Số BN |", "|---|---|---|---|---|"]
            L += [f"| {sv(k)} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |" for k, v in p["per_class"].items()]
            rej = [c for c in lab if c["status"] == "insufficient_evidence"]
            wrong = [c for c in lab if outcome(c) == "wrong"]
            L += ["", f"- Sai: {', '.join(f'{c['patient_id']} ({sv(c['true_label'])}→{sv(c['pipeline_label'])}; oracle {sv(c['oracle_label'])})' for c in wrong) or 'không có'}.",
                  f"- Bị từ chối: {', '.join(f'{c['patient_id']} ({sv(c['true_label'])}; conf cao nhất {c['evidence'].get('max_conf', 0):.2f}, {c['evidence'].get('n_volumes_strong')} pha đạt)' for c in rej) or 'không có'}.",
                  f"- **YOLO bỏ sót:** BN ung thư không có vùng nào {cnt['no_region']}; BN ung thư mà không vùng nào trùng tổn thương thật "
                  f"{d['patients_no_true_lesion_found']}/{d['patients_with_gt']} ({', '.join(c['patient_id'] for c in lab if c.get('volumes') and not any(t['matched'] for t in c['tracks'])) or 'không có'}; nhãn của các BN này, nếu có, chỉ đến từ vùng giả).", strict_txt, "",
                  "### 1b. Nhóm chứng CN (không có lớp đúng trong mô hình 3 lớp)", "",
                  "| | Bị gán nhãn ung thư (cảnh báo sai, KTC 95%) | Từ chối phân loại | YOLO không có vùng | Không bị gán nhãn ung thư |", "|---|---|---|---|---|",
                  f"| **Test** | {pct(cc['cancer_label'], cc['n'])} ({ci_txt(cc['cancer_label_ci95'])}) | {cc['rejected']} | {cc['no_region']} | {pct(cc['n'] - cc['cancer_label'], cc['n'])} |",
                  f"| Val (tham chiếu) | {pct(cc['val_reference']['cancer_label'], cc['val_reference']['n'])} | | | |", "",
                  f"CN bị gán nhãn: {', '.join(f'{c['patient_id']} ({sv(c['pipeline_label'])})' for c in cn_lab) or 'không có'}.", "",
                  "### 1c. Toàn bộ Test CT", "",
                  f"- Ung thư phân loại đúng + CN không bị gán nhãn ung thư: {pct(ok_all, len(cases))}. Nhãn sai (ung thư sai lớp + CN bị gán nhãn): "
                  f"{pct(cnt['wrong'] + cc['cancer_label'], len(cases))}. BN ung thư không có nhãn: {pct(cnt['rejected'] + cnt['no_region'], len(cases))}.",
                  "  Đây không phải accuracy 4 lớp (hệ thống không có lớp CN).", ""]
        else:
            g = m["groups"]
            L += ["## 2. MRI (LLD-MMRI) — THÍ NGHIỆM NGHIÊN CỨU, CHƯA ĐẠT YÊU CẦU", "",
                  "> Phân loại MRI 7 lớp **không** đạt yêu cầu trên Validation (ROI accuracy 0.478) và **không** được tích hợp làm chức năng chẩn đoán "
                  "của website. Kết quả dưới đây chỉ công bố như một thí nghiệm nghiên cứu. Evidence Gate không bật cho MRI.", "",
                  f"Test MRI: {len(cases)} BN ({', '.join(f'{sv(k)} {v}' for k, v in sorted(Counter(c['true_label'] for c in lab).items()))}). "
                  f"Volume bị loại (cách ly bbox, như Val): {', '.join(f'{k} {v}' for k, v in excl.items()) or 'không có'}. Lỗi khi chạy: {len(failed)}.", "",
                  "| | Đúng | Sai | Từ chối | YOLO không có vùng | Accuracy trên toàn bộ BN (KTC 95%) | Macro-F1 | Balanced accuracy |", "|---|---|---|---|---|---|---|---|",
                  f"| **Test** | {cnt['correct']} | {cnt['wrong']} | {cnt['rejected']} | {cnt['no_region']} | {pct(p['correct'], p['n'])} ({ci_txt(p['accuracy_all_ci95'])}) | "
                  f"{p['macro_f1']:.3f} (bootstrap {b['macro_f1'][0]:.3f}–{b['macro_f1'][1]:.3f}) | {p['balanced_accuracy']:.3f} |"]
            vr = m["val_reference"]
            L += [f"| Val (tham chiếu) | {vr['counts']['correct']} | {vr['counts']['wrong']} | {vr['counts']['rejected']} | {vr['counts']['no_region']} | "
                  f"{pct(vr['pipeline']['correct'], vr['pipeline']['n'])} | {vr['pipeline']['macro_f1']:.3f} | {vr['pipeline']['balanced_accuracy']:.3f} |", "",
                  "| Lớp | Precision | Recall | F1 | Số BN |", "|---|---|---|---|---|"]
            L += [f"| {sv(k)} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |" for k, v in p["per_class"].items()]
            L += ["", f"- Nhóm lành / ác tính (tham khảo): đúng nhóm {pct(g['group_correct'], g['n'])}; tổn thương lành bị gán lớp ác tính "
                  f"(cảnh báo ung thư sai) {g['benign_as_malignant']}/{g['n_benign']}; tổn thương ác tính bị gán lớp lành {g['malignant_as_benign']}/{g['n_malignant']}.",
                  f"- **YOLO bỏ sót:** BN không có vùng nào {cnt['no_region']}; BN mà không vùng nào trùng tổn thương thật "
                  f"{d['patients_no_true_lesion_found']}/{d['patients_with_gt']}.", strict_txt, ""]
        o, ab = m["oracle"], m["ungated_ablation"]
        dv = m["val_reference"]["detection"]
        L += [f"**Phát hiện (YOLO, conf {lock['modalities'][mod]['yolo']['conf']}):** volume có tổn thương được tìm thấy "
              f"{pct(d['volumes_lesion_detected'], d['volumes_with_lesion'])}; tổn thương 3D {pct(d['lesions_detected'], d['lesions'])}; "
              f"tổn thương nhỏ (< 32×32 px) {pct(d['small_detected'], d['small'])}; vùng giả trung bình {d['false_tracks_per_volume']:.1f}/volume. "
              f"Val (tham chiếu): volume {pct(dv['volumes_lesion_detected'], dv['volumes_with_lesion'])}; tổn thương 3D {pct(dv['lesions_detected'], dv['lesions'])}; "
              f"tổn thương nhỏ {pct(dv['small_detected'], dv['small'])}; vùng giả {dv['false_tracks_per_volume']:.1f}/volume.", "",
              f"**Oracle (vùng cắt theo GT — chỉ để đối chiếu, không phải kết quả hệ thống):** {pct(o['correct'], o['n'])}, macro-F1 {o['macro_f1']:.3f}.", ""]
        if mod == "ct":
            L += [f"**Ablation tắt cổng (chỉ để mô tả; cấu hình đã khóa trước, không dùng để chọn):** BN ung thư đúng {pct(ab['correct'], ab['n'])}; "
                  f"CN bị gán nhãn ung thư {pct(m['controls']['ungated_cancer_label'], m['controls']['n'])}.", ""]
        L += [f"![CM pipeline](figures/cm_test_{mod}_pipeline.png) ![CM oracle](figures/cm_test_{mod}_oracle.png)", ""]

    c, r = M["ct"], M["mri"]
    no_err = not c["failed"] and not r["failed"]
    L += ["## 3. Mức sẵn sàng cho giai đoạn website", "",
          "Tiêu chí kỹ thuật (đặt trước khi chạy Test): (1) pipeline chạy hết mọi BN Test không lỗi; (2) chạy đúng cấu hình đã khóa; "
          "(3) mọi trạng thái không nhãn đều có thông điệp không kết luận khỏe mạnh.", "",
          f"- (1) {'Đạt' if no_err else 'KHÔNG đạt'}: lỗi CT {len(c['failed'])}, MRI {len(r['failed'])}.",
          f"- (2) Đạt: khóa được kiểm tra lại trước khi chạy (commit `{lock['git']['commit'][:10]}`).",
          "- (3) Đạt: trạng thái `insufficient_evidence` / `no_lesion_detected` luôn kèm câu \"không kết luận khỏe mạnh\" (`tests/test_pipeline.py`).",
          "- CT: tích hợp **phát hiện + Tumor Zoom + phân loại 3 lớp có Evidence Gate**, luôn kèm cảnh báo \"không phải chẩn đoán\" và \"không có lớp không tổn thương\".",
          "- MRI: chỉ tích hợp **phát hiện + Tumor Zoom**; phân loại MRI không đưa vào chức năng chẩn đoán.",
          "- Đây là mức sẵn sàng **kỹ thuật cho một hệ thống nghiên cứu**, không phải đánh giá lâm sàng.", "",
          "## 4. Hạn chế", "",
          "- Tập Test nhỏ (khoảng tin cậy rộng, đặc biệt nhóm CN). Một lần chạy, một seed huấn luyện.",
          "- CN là nhóm \"không có bệnh gan mật\" theo nguồn dữ liệu, không phải \"khỏe mạnh hoàn toàn\". Từ chối phân loại không phải kết luận âm tính.",
          "- Kiểm tra ảnh trùng chỉ phát hiện bản sao y hệt giữa các split.",
          f"- YOLO CT chỉ tìm thấy {pct(c['detection']['small_detected'], c['detection']['small'])} tổn thương nhỏ (< 32×32 px); "
          "nhãn mức BN được gộp từ mọi vùng YOLO của BN, nên có thể đúng hoặc sai ngay cả khi tổn thương thật bị bỏ sót.",
          f"- CT Test (accuracy {100 * c['pipeline']['accuracy_all']:.1f}%) cao hơn Val ({100 * c['val_reference']['pipeline']['accuracy_all']:.1f}%) "
          "nằm trong dao động do mẫu nhỏ (khoảng tin cậy chồng lấn); không nên hiểu là mô hình tốt hơn trên Test. "
          f"Nếu chỉ tính đúng khi tổn thương thật được tìm thấy: Test {c['strict_correct']}/{c['n_lesion_patients']}, "
          f"Val {c['val_reference']['strict_correct']}/{c['n_lesion_patients']}.", "",
          "Tệp: `reports/final_test/metrics_test.json` (chỉ số tổng hợp), `patients.csv`, `volumes.csv` (từng BN / volume; không commit), "
          "`figures/` (confusion matrix, hình từng ca), dữ liệu từng ca `outputs/final_test/<ct|mri>/cases/`."]
    write_csv(REP / "patients.csv", pat_rows)
    write_csv(REP / "volumes.csv", vol_rows)
    (REP / "metrics_test.json").write_text(json.dumps(M, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    (REP / "bao_cao_test.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("written", REP / "bao_cao_test.md")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["selfcheck", "export", "run", "report"], required=True)
    ap.add_argument("--modality", nargs="+", choices=list(MODS), default=list(MODS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--selfcheck-n", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="finish patients missing after an interruption (never re-runs a patient)")
    args = ap.parse_args()
    return {"selfcheck": selfcheck, "export": export, "run": run, "report": report}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
