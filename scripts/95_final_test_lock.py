"""Final Test, step 1: verify that Test is independent, then LOCK the configuration chosen on Validation.

Nothing here reads Test predictions; Test images are only fingerprinted (hash of voxel bytes) to look for duplicates
across splits. Checks (any FAIL -> no lock, exit code 1):
  S  splits      one split per patient, labels = source, frozen before any dataset/training artefact, copies identical
  Y  YOLO data   exported datasets + uploaded ZIPs contain no Test patient and no 'test' folder; runs validated on 'val'
  C  classifier  ROI manifests + uploaded ZIPs without Test patients; run configs = Train/Val counts; Val predictions only Val
  V  selection   threshold sweep, Tumor Zoom filter, aggregation, evidence gate: every cache is Val-only
  O  outputs     no Test patient id anywhere in outputs/ or in reports (except split / audit tables)
  D  duplicates  no Test volume has the same voxel content as a Train/Val volume (exact, first 24 MB of voxel data)
  G  git         code / configs committed (clean tree for src, scripts, configs, tests)
Lock (only when every check passes): reports/final_test/locked_config.json + outputs/final_test/locked_config.json
  git commit, sha256 of checkpoints / configs / every code file, YOLO conf, Tumor Zoom params, classifier, aggregation, gate.
Report: reports/final_test/kiem_tra_doc_lap.md
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import CONFIG_DIR, load_paths, load_yaml  # noqa: E402

REP = ROOT / "reports" / "final_test"
OUT = ROOT / "outputs" / "final_test"
ID_RE = re.compile(r"(?<![A-Za-z0-9])(P\d{4}|MR-?\d{3,})(?![0-9])")
TEXT_EXT = {".csv", ".json", ".md", ".txt", ".log", ".yaml", ".yml", ".err"}
MAX_TEXT = 60 * 2 ** 20
FP_SKIP, FP_BYTES = 8 * 2 ** 20, 16 * 2 ** 20      # fingerprint = sha256 of voxel bytes [8 MB, 24 MB) (whole data if smaller)

_spec = importlib.util.spec_from_file_location("b20", ROOT / "scripts" / "20_build_yolo_pilot.py")
b20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b20)


def rows(p: Path) -> list[dict]:
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(2 ** 22), b""):
            h.update(chunk)
    return h.hexdigest()


class Checks:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, code: str, ok: bool, text: str, detail: str = ""):
        self.items.append({"code": code, "ok": bool(ok), "text": text, "detail": detail})
        print(f"  [{'OK ' if ok else 'FAIL'}] {code} {text}" + (f" — {detail}" if detail and not ok else ""), flush=True)

    @property
    def passed(self) -> bool:
        return all(i["ok"] for i in self.items)


def ids_in_text(s: str) -> set[str]:
    return set(ID_RE.findall(s))


def ids_in_zip(zp: Path, split_manifest_sha: str = "") -> tuple[set[str], set[str], list[str]]:
    """(patient ids in entry names + small text members, split folder names, split-manifest copies accepted).
    A copy of the frozen split table (identical sha256) lists every patient by design and is not training data."""
    ids, folders, accepted = set(), set(), []
    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            ids |= ids_in_text(info.filename)
            parts = info.filename.replace("\\", "/").split("/")
            folders |= {p for p in parts[:-1] if p in ("train", "val", "test")}
            if Path(info.filename).suffix.lower() in TEXT_EXT and info.file_size <= MAX_TEXT:
                data = z.read(info)
                if Path(info.filename).name == "split_manifest.csv" and hashlib.sha256(data).hexdigest() == split_manifest_sha:
                    accepted.append(info.filename)
                    continue
                ids |= ids_in_text(data.decode("utf-8", "ignore"))
    return ids, folders, accepted


# ----------------------------------------------------------------------------- fingerprints
def fingerprint(job: tuple[str, str]) -> tuple[str, str, str]:
    key, path = job
    try:
        import nibabel as nib
        hdr = nib.load(path).header
        off = int(hdr["vox_offset"])
        head, partial = b"", False
        with gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb") as f:
            try:
                f.read(off)
                while len(head) < FP_SKIP + FP_BYTES:
                    chunk = f.read(min(2 ** 16, FP_SKIP + FP_BYTES - len(head)))
                    if not chunk:
                        break
                    head += chunk
            except (EOFError, zlib.error):   # truncated download (e.g. P0286_C2): fingerprint the readable part
                partial = True
        block = head[FP_SKIP:] if len(head) > FP_SKIP + 2 ** 20 else head
        if not block.strip(b"\x00"):
            return key, "constant", str(tuple(hdr.get_data_shape()))
        h = hashlib.sha256(block).hexdigest()
        return key, (f"{h}" if not partial or len(head) >= FP_SKIP + FP_BYTES else f"partial{len(head)}:{h}"), str(tuple(hdr.get_data_shape()))
    except Exception as e:  # noqa: BLE001
        return key, f"error:{type(e).__name__}", ""


def all_volumes(paths) -> list[dict]:
    out = []
    for v in b20.ct_volumes(paths, {"train", "val", "test"}):
        if v["status"] != "excluded:ct_file_missing":
            out.append({"mod": "ct", "patient_id": v["patient_id"], "series": v["series"], "split": v["split"], "image": v["image"]})
    for v in b20.mri_volumes(paths, {"train", "val", "test"}):
        out.append({"mod": "mri", "patient_id": v["patient_id"], "series": v["series"], "split": v["split"], "image": v["image"]})
    return out


def duplicates(paths, workers: int) -> dict:
    cache = OUT / "fingerprints.csv"
    vols = all_volumes(paths)
    done = {}
    if cache.exists():
        done = {r["key"]: r for r in rows(cache)}
    todo = [(f"{v['mod']}|{v['patient_id']}|{v['series']}", v["image"]) for v in vols]
    todo = [t for t in todo if t[0] not in done or done[t[0]]["hash"].startswith("error")]
    if todo:
        print(f"  fingerprinting {len(todo)} volumes ({workers} workers) ...", flush=True)
        with ProcessPoolExecutor(workers) as ex:
            for n, (key, h, shape) in enumerate(ex.map(fingerprint, todo, chunksize=8), 1):
                done[key] = {"key": key, "hash": h, "shape": shape}
                if n % 500 == 0:
                    print(f"    {n}/{len(todo)}", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["key", "hash", "shape"])
            w.writeheader()
            w.writerows(done.values())
    split_of = {f"{v['mod']}|{v['patient_id']}|{v['series']}": v["split"] for v in vols}
    by_hash = defaultdict(list)
    errors, const = [], []
    for k, r in done.items():
        if k not in split_of:
            continue
        if r["hash"].startswith("error"):
            errors.append(k)
        elif r["hash"] == "constant":
            const.append(k)
        else:
            by_hash[(k.split("|")[0], r["hash"])].append(k)
    cross, same_patient_other = [], []
    for keys in by_hash.values():
        pats = {k.split("|")[1] for k in keys}
        if len(pats) < 2:
            continue
        sp = {split_of[k] for k in keys}
        (cross if "test" in sp and len(sp) > 1 else same_patient_other).append(sorted(keys))
    # partially readable (truncated) volumes: compare the readable byte range with the SAME range of every volume of the
    # same modality in the other split group (Test vs Train/Val), so a truncated file cannot hide a duplicate
    img = {f"{v['mod']}|{v['patient_id']}|{v['series']}": v["image"] for v in vols}
    partial = {}
    for k, r in done.items():
        if k in split_of and r["hash"].startswith("partial"):
            n_read = int(r["hash"][7:].split(":")[0])
            lo = FP_SKIP if n_read > FP_SKIP + 2 ** 20 else 0
            ref = _range_hash(img[k], lo, n_read)
            other = [o for o in split_of if o.split("|")[0] == k.split("|")[0] and o.split("|")[1] != k.split("|")[1]
                     and (split_of[o] == "test") != (split_of[k] == "test")]
            print(f"  partial volume {k}: {n_read} readable bytes, comparing with {len(other)} volumes of the other split group", flush=True)
            with ProcessPoolExecutor(workers) as ex:
                same = [o for o, h in zip(other, ex.map(_range_hash_job, [(img[o], lo, n_read) for o in other], chunksize=8)) if h == ref]
            partial[k] = {"readable_bytes": n_read, "compared": len(other), "identical": same}
            cross += [[k] + same] if same else []
    return {"n_volumes": len(split_of), "errors": errors, "constant": const, "test_vs_trainval": cross,
            "other_cross_patient": same_patient_other, "partial": partial}


def _range_hash(path: str, lo: int, hi: int) -> str:
    import nibabel as nib
    off = int(nib.load(path).header["vox_offset"])
    data = b""
    with gzip.open(path, "rb") as f:
        try:
            f.read(off)
            while len(data) < hi:
                chunk = f.read(min(2 ** 16, hi - len(data)))
                if not chunk:
                    break
                data += chunk
        except (EOFError, zlib.error):
            pass
    return hashlib.sha256(data[lo:hi]).hexdigest() if len(data) >= hi else f"short{len(data)}"


def _range_hash_job(job):
    return _range_hash(*job)


# ----------------------------------------------------------------------------- checks
def run_checks(args) -> tuple[Checks, dict]:
    paths = load_paths()
    ck, info = Checks(), {}
    sp_dir = paths["project"]["splits"]
    split = {m: {r["patient_id"]: r for r in rows(sp_dir / f"{m}_split.csv")} for m in ("ct", "mri")}
    test_ids = {m: {p for p, r in split[m].items() if r["split"] == "test"} for m in ("ct", "mri")}
    trainval_ids = {m: {p for p, r in split[m].items() if r["split"] != "test"} for m in ("ct", "mri")}
    all_test = test_ids["ct"] | test_ids["mri"]
    info["n_test"] = {m: len(v) for m, v in test_ids.items()}
    info["test_classes"] = {m: dict(Counter(split[m][p]["label"] for p in test_ids[m])) for m in ("ct", "mri")}
    print("S splits", flush=True)

    # S ---------------------------------------------------------------- splits
    for m in ("ct", "mri"):
        ids = [r["patient_id"] for r in rows(sp_dir / f"{m}_split.csv")]
        ck.add("S1", len(ids) == len(set(ids)) and {r["split"] for r in split[m].values()} == {"train", "val", "test"},
               f"{m.upper()}: mỗi BN thuộc đúng một split ({len(ids)} BN; Test {len(test_ids[m])})")
    src = {r["patient_id"]: r["cancer_type"] for r in rows(Path(paths["ct"]["csv"]))}
    ck.add("S2", {p: r["label"] for p, r in split["ct"].items()} == src, "CT: nhãn trong split trùng `patient_data.csv`")
    off_val = {r["patient_id"] if "patient_id" in r else list(r.values())[0] for r in rows(ROOT / "data" / "external" / "lld_mmri2023_official_val_ids.csv")}
    mri_val = {p for p, r in split["mri"].items() if r["split"] == "val"}
    ck.add("S3", mri_val == off_val and not (off_val & test_ids["mri"]), f"MRI: Val = {len(off_val)} ID chính thức, không ID nào trong Test")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests" / "test_splits_no_leakage.py")],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    ck.add("S4", r.returncode == 0, "`tests/test_splits_no_leakage.py` pass", r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-300:])
    t_split = max((sp_dir / f).stat().st_mtime for f in ("ct_split.csv", "mri_split.csv"))
    artefacts = [ROOT / "data/processed/pilot/yolo_ct/volumes.csv", ROOT / "data/processed/pilot/yolo_mri/volumes.csv",
                 ROOT / "data/processed/full/yolo_ct/volumes.csv", ROOT / "data/processed/cls/ct/manifest.csv",
                 ROOT / "data/processed/cls/mri/manifest.csv"]
    first = min(p.stat().st_mtime for p in artefacts if p.exists())
    ck.add("S5", t_split < first, f"split đóng băng ({datetime.fromtimestamp(t_split):%Y-%m-%d %H:%M}) trước mọi bộ dữ liệu huấn luyện "
           f"({datetime.fromtimestamp(first):%Y-%m-%d %H:%M})")
    man = sha256(sp_dir / "split_manifest.csv")
    copies = [p for p in (ROOT / "data/processed").glob("*/split_manifest.csv")]
    ck.add("S6", all(sha256(p) == man for p in copies), f"{len(copies)} bản sao `split_manifest.csv` trong data/processed trùng bản gốc")
    info["split_sha256"] = {f: sha256(sp_dir / f) for f in ("ct_split.csv", "mri_split.csv", "split_manifest.csv")}

    # Y ---------------------------------------------------------------- YOLO data and runs
    print("Y YOLO", flush=True)
    yolo_cfg = load_yaml(CONFIG_DIR / "yolo_inference.yaml")
    for name, d, m in (("pilot CT", "data/processed/pilot/yolo_ct", "ct"), ("full CT", "data/processed/full/yolo_ct", "ct"),
                       ("pilot MRI", "data/processed/pilot/yolo_mri", "mri")):
        d = ROOT / d
        sl, vo = rows(d / "slices.csv"), rows(d / "volumes.csv")
        bad = {s["patient_id"] for s in sl} & test_ids[m] | {v["patient_id"] for v in vo} & test_ids[m]
        splits = {s["split"] for s in sl} | {v["split"] for v in vo}
        folders = {p.name for p in (d / "images").iterdir() if p.is_dir()} | {p.name for p in (d / "labels").iterdir() if p.is_dir()}
        ck.add("Y1", not bad and splits <= {"train", "val"} and folders <= {"train", "val"},
               f"YOLO {name}: {len({s['patient_id'] for s in sl})} BN, split {sorted(splits)}, không BN Test", ", ".join(sorted(bad)[:10]))
    for zp, m in ((ROOT / "data/processed/pilot/yolo_ct_pilot.zip", "ct"), (ROOT / "data/processed/full/yolo_ct_full.zip", "ct"),
                  (ROOT / "data/processed/pilot/yolo_mri_pilot.zip", "mri")):
        ids, folders, acc = ids_in_zip(zp, man)
        ck.add("Y2", not (ids & all_test) and "test" not in folders,
               f"ZIP {zp.name}: {len(ids & trainval_ids[m])} BN Train/Val, 0 BN Test trong ảnh/nhãn/metadata, thư mục {sorted(folders)}"
               + (f" (bảng split đầy đủ `{', '.join(acc)}` trùng bản đóng băng)" if acc else ""), ", ".join(sorted(ids & all_test)[:10]))
    for m in ("ct", "mri"):
        run = ROOT / yolo_cfg[m]["run"]
        a = load_yaml(run / "args.yaml")
        ck.add("Y3", a.get("split") == "val" and "test" not in str(a.get("data", "")),
               f"YOLO {m.upper()} run `{run.name}`: data `{a.get('data')}`, đánh giá trên split `{a.get('split')}`")
        tc = json.loads((ROOT / yolo_cfg[m]["conf_source"]).read_text(encoding="utf-8"))
        ck.add("V1", "Val" in tc.get("rule", "") and abs(tc["F2_optimal"]["conf"] - yolo_cfg[m]["conf"]) < 1e-9,
               f"YOLO {m.upper()} conf {yolo_cfg[m]['conf']} = argmax F2 trên Val (`threshold_choice.json`, {tc.get('n_val_slices')} lát Val)")

    # C ---------------------------------------------------------------- classifier data and runs
    print("C classifier", flush=True)
    cls = load_yaml(CONFIG_DIR / "cls.yaml")
    for m in ("ct", "mri"):
        mf = rows(ROOT / cls["roi"]["out_dir"] / m / "manifest.csv")
        bad = {r["patient_id"] for r in mf} & test_ids[m]
        ck.add("C1", not bad and {r["split"] for r in mf} <= {"train", "val"},
               f"ROI {m.upper()} manifest: {len(mf)} ROI, split {sorted({r['split'] for r in mf})}, không BN Test", ", ".join(sorted(bad)[:10]))
        ids, folders, _ = ids_in_zip(ROOT / cls["roi"]["out_dir"] / f"cls_{m}.zip", man)
        ck.add("C2", not (ids & all_test) and "test" not in folders, f"ZIP cls_{m}.zip: 0 BN Test, thư mục {sorted(folders)}",
               ", ".join(sorted(ids & all_test)[:10]))
        run = ROOT / cls["inference"][m]["checkpoint"]
        cfg = json.loads((run.parent / "config.json").read_text(encoding="utf-8"))
        tr = {r["patient_id"] for r in mf if r["split"] == "train" and r["source"] == "gt"}
        va = {r["patient_id"] for r in mf if r["split"] == "val" and r["source"] == "gt"}
        ck.add("C3", cfg["n_train_patients"] == len(tr) and cfg["n_val_patients"] == len(va),
               f"Hybrid {m.upper()} `{run.parent.name}`: huấn luyện {cfg['n_train_patients']} BN Train, chọn epoch trên {cfg['n_val_patients']} BN Val (khớp manifest)")
        pr = rows(run.parent / "val_predictions_best.csv")
        pids = {r["patient_id"] for r in pr}
        ck.add("C4", pids <= {p for p, r in split[m].items() if r["split"] == "val"},
               f"Hybrid {m.upper()}: dự đoán dùng để chọn checkpoint chỉ trên {len(pids)} BN Val")
        mv = json.loads((ROOT / cls["inference"][m]["metrics"]).read_text(encoding="utf-8"))
        ck.add("V2", str(mv.get("split", "val")) == "val", f"Cách gộp {m.upper()} `{cls['inference'][m]['aggregation']}` chọn trên Val (`{Path(cls['inference'][m]['metrics']).name}`)")
    ids, _, _ = ids_in_zip(ROOT / cls["roi"]["out_dir"] / "livercancer_code.zip", man)
    ck.add("C5", not (ids & all_test), "ZIP code gửi lên Colab không chứa ID BN Test")

    # V ---------------------------------------------------------------- selection caches (Val only)
    print("V selection", flush=True)
    for m in ("ct", "mri"):
        val = {p for p, r in split[m].items() if r["split"] == "val"}
        for d in ("pipeline_validation", "pipeline_validation_gate"):
            got = {p.stem for p in (ROOT / "outputs/predictions" / d / m / "cases").glob("*.json")}
            ck.add("V3", got == val, f"Cache `{d}/{m}`: {len(got)} BN = toàn bộ Val, không BN Test")
        roi = ROOT / "outputs/predictions/roi_val" / m / "raw_dets.csv"
        if roi.exists():
            got = {r["patient_id"] for r in rows(roi)}
            ck.add("V4", got <= val, f"Tumor Zoom: bộ lọc track chọn trên {len(got)} BN Val (`roi_val/{m}`)")
    lim = json.loads((ROOT / "outputs/predictions/pipeline_validation/limitations.json").read_text(encoding="utf-8"))
    ck.add("V5", "test" not in json.dumps(lim).lower().replace("contest", ""), "Evidence Gate: chọn từ phân tích Val (`limitations.json`)")

    # O ---------------------------------------------------------------- any Test id in outputs / reports / processed data
    print("O scan", flush=True)
    allowed = ("reports/audit", "reports/split_report", "data/processed/pilot/split_manifest", "data/processed/full/split_manifest",
               "reports/final_test", "outputs/final_test")
    hits, pre_split = defaultdict(set), {}
    for base in ("outputs", "reports", "data/processed"):
        for p in (ROOT / base).rglob("*"):
            rel = p.relative_to(ROOT).as_posix()
            if not p.is_file() or rel.startswith(allowed):
                continue
            found = ids_in_text(p.name)
            if p.suffix.lower() in TEXT_EXT and p.stat().st_size <= MAX_TEXT:
                found |= ids_in_text(p.read_text(encoding="utf-8", errors="ignore"))
            found &= all_test
            if not found:
                continue
            if p.name == "split_manifest.csv" and sha256(p) == man:
                continue                                  # copy of the frozen split table
            if p.stat().st_mtime < t_split:              # GĐ1 data audit, written before any split / model existed
                pre_split[rel] = len(found)
                continue
            hits[rel] |= found
    info["test_id_hits"] = {k: sorted(v) for k, v in hits.items()}
    info["pre_split_audit_files"] = pre_split
    ck.add("O1", not hits, f"Không có ID BN Test trong outputs/, reports/, data/processed/ (ngoài bản sao bảng split và "
           f"{len(pre_split)} file audit GĐ1 tạo trước khi có split)",
           "; ".join(f"{k}: {sorted(v)[:5]}" for k, v in list(hits.items())[:8]))

    # D ---------------------------------------------------------------- duplicates
    if not args.skip_duplicates:
        print("D duplicates", flush=True)
        d = duplicates(paths, args.workers)
        info["duplicates"] = d
        ck.add("D0", not d["errors"], f"Đọc được dấu vân tay {d['n_volumes'] - len(d['errors'])}/{d['n_volumes']} volume", ", ".join(d["errors"][:10]))
        ck.add("D1", not d["test_vs_trainval"], f"Không volume Test nào trùng nội dung voxel với volume Train/Val của BN khác",
               "; ".join(" = ".join(g) for g in d["test_vs_trainval"][:5]))

    # G ---------------------------------------------------------------- git
    st = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs", "tests"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    dirty = [l for l in st.splitlines() if l.strip() and "__pycache__" not in l]
    ck.add("G1", not dirty, "Code và cấu hình đã commit (src, scripts, configs, tests sạch)", "; ".join(dirty[:10]))
    return ck, info


# ----------------------------------------------------------------------------- lock
def code_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "src", "scripts", "configs"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.split()
    return sorted(f for f in out if f.endswith((".py", ".yaml", ".yml")))


def build_lock(info: dict) -> dict:
    yc, rc, cc = (load_yaml(CONFIG_DIR / f) for f in ("yolo_inference.yaml", "roi.yaml", "cls.yaml"))
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.strip()  # noqa: E731
    files = code_files()
    lock = {"created": datetime.now().isoformat(timespec="seconds"), "purpose": "official Test evaluation, run once",
            "git": {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                    "subject": git("log", "-1", "--format=%s")},
            "code_sha256": {f: sha256(ROOT / f) for f in files},
            "splits": info["split_sha256"], "n_test": info["n_test"], "test_classes": info["test_classes"], "modalities": {}}
    for m in ("ct", "mri"):
        inf = cc["inference"][m]
        yw, cw = ROOT / yc[m]["weights"], ROOT / inf["checkpoint"]
        g = inf.get("evidence_gate") or {}
        lock["modalities"][m] = {
            "yolo": {"weights": yc[m]["weights"], "sha256": sha256(yw), "sha256_config": yc[m]["sha256"], "conf": yc[m]["conf"],
                     "imgsz": yc[m]["imgsz"], "iou": yc[m]["iou"], "max_det": yc[m]["max_det"], "chosen_on": yc[m]["conf_source"]},
            "tumor_zoom": {**rc["common"], **rc[m]},
            "classifier": {"checkpoint": inf["checkpoint"], "sha256": sha256(cw), "sha256_config": inf["sha256"],
                           "classes": cc[m]["classes"], "aggregation": inf["aggregation"], "aggregation_chosen_on": inf["metrics"],
                           "slices_per_track": cc["inference"]["slices_per_track"][m], "slice_min_gap": cc[m]["slice_min_gap"],
                           "reliable": inf["reliable"]},
            "evidence_gate": {"enabled": bool(g.get("enabled")), "kind": g.get("kind"), "conf_min": g.get("conf_min"), "k": g.get("k"),
                              "single_volume": g.get("single_volume"),
                              "chosen_on": "Validation (scripts/93_limitations_analysis.py, scripts/94_gate_comparison.py)"},
        }
        assert lock["modalities"][m]["yolo"]["sha256"] == yc[m]["sha256"], f"{m}: YOLO weights changed"
        assert lock["modalities"][m]["classifier"]["sha256"] == inf["sha256"], f"{m}: classifier checkpoint changed"
    return lock


def write_report(ck: Checks, info: dict, lock: dict | None) -> None:
    REP.mkdir(parents=True, exist_ok=True)
    L = ["# Kiểm tra tính độc lập của tập Test và khóa cấu hình", "",
         f"_Sinh bởi `scripts/95_final_test_lock.py` lúc {datetime.now():%Y-%m-%d %H:%M}. Bước này không chạy mô hình trên Test; "
         "ảnh Test chỉ được đọc để tính dấu vân tay (hash) khi tìm ảnh trùng giữa các split._", "",
         f"**Kết luận: {'ĐẠT — Test độc lập, cấu hình đã được khóa' if ck.passed and lock else 'KHÔNG ĐẠT — dừng, không chạy Test'}.**", "",
         f"Test: CT {info['n_test']['ct']} BN ({', '.join(f'{k} {v}' for k, v in sorted(info['test_classes']['ct'].items()))}); "
         f"MRI {info['n_test']['mri']} BN.", "",
         "| Mã | Kết quả | Kiểm tra | Chi tiết |", "|---|---|---|---|"]
    for i in ck.items:
        L.append(f"| {i['code']} | {'✅' if i['ok'] else '❌'} | {i['text']} | {i['detail'].replace('|', '/')} |")
    pre = info.get("pre_split_audit_files") or {}
    if pre:
        L += ["", f"File có mã BN Test nhưng được chấp nhận (audit dữ liệu GĐ1, tạo **trước** khi split tồn tại, không có mô hình nào): "
              + ", ".join(f"`{k}`" for k in sorted(pre)) + ". Bản sao `split_manifest.csv` (trùng SHA256 bản đóng băng) liệt kê mọi BN theo thiết kế."]
    d = info.get("duplicates")
    if d:
        L += ["", f"Ảnh trùng: đã so {d['n_volumes']} volume (sha256 của 16 MB dữ liệu voxel, bỏ 8 MB đầu). "
              f"Trùng giữa Test và Train/Val: {len(d['test_vs_trainval'])}. Trùng giữa các BN khác trong Train/Val: {len(d['other_cross_patient'])}"
              + (f" ({'; '.join(' = '.join(g) for g in d['other_cross_patient'][:5])})" if d["other_cross_patient"] else "") +
              f". Khối toàn 0 (không so được): {len(d['constant'])}. "
              + "".join(f"Volume đọc được một phần (file tải về bị cắt cụt) `{k}`: so {x['readable_bytes'] // 2 ** 20} MB đọc được với "
                        f"{x['compared']} volume của nhóm split còn lại, trùng {len(x['identical'])}. " for k, x in (d.get('partial') or {}).items())
              + f"Kiểm tra này chỉ phát hiện bản sao y hệt, "
              "không phát hiện cùng một người được chụp nhiều lần dưới hai mã BN."]
    if lock:
        L += ["", "## Cấu hình đã khóa", "",
              f"- Commit code: `{lock['git']['commit']}` ({lock['git']['subject']}); {len(lock['code_sha256'])} file code/cấu hình có SHA256 trong `locked_config.json`.",
              *([f"- **Khóa lại** (thay bản khóa `{lock['supersedes']['commit'][:10]}` lúc {lock['supersedes']['created']}, lưu ở "
                 f"`{lock['supersedes']['file']}`): {lock['supersedes']['reason']} Chưa có dự đoán nào trên Test trước khi khóa lại."]
                if lock.get("supersedes") else []),
              "", "| | CT | MRI |", "|---|---|---|"]
        c, m = lock["modalities"]["ct"], lock["modalities"]["mri"]
        rowsL = [("YOLO weights", lambda x: f"`{x['yolo']['weights']}`<br>sha256 `{x['yolo']['sha256'][:16]}…`"),
                 ("Ngưỡng YOLO (conf)", lambda x: f"{x['yolo']['conf']} (argmax F2 trên Val)"),
                 ("Tumor Zoom", lambda x: f"gộp IoU>{x['tumor_zoom']['merge_iou']} / chứa>{x['tumor_zoom']['merge_containment']}, nối IoU≥{x['tumor_zoom']['link_iou']} gap {x['tumor_zoom']['link_max_gap']}, "
                                         f"không lọc track (≥{x['tumor_zoom']['min_track_slices']} lát, conf≥{x['tumor_zoom']['min_track_conf']}), crop {x['tumor_zoom']['crop_size']}"),
                 ("Bộ phân loại", lambda x: f"`{x['classifier']['checkpoint']}`<br>sha256 `{x['classifier']['sha256'][:16]}…`"),
                 ("Lát mỗi vùng / khoảng cách", lambda x: f"{x['classifier']['slices_per_track']} / ≥{x['classifier']['slice_min_gap']}"),
                 ("Gộp mức BN", lambda x: f"`{x['classifier']['aggregation']}` (chọn trên Val)"),
                 ("Evidence Gate", lambda x: (f"BẬT: conf ≥ {x['evidence_gate']['conf_min']} ở ≥ {x['evidence_gate']['k']} pha; một file: conf cao nhất ≥ {x['evidence_gate']['single_volume']['conf_min']}"
                                              if x["evidence_gate"]["enabled"] else "TẮT (không có nhóm chứng trên Val)")),
                 ("Độ tin cậy nhãn", lambda x: "đã chọn trên Val" if x["classifier"]["reliable"] else "**chưa đạt** — chỉ là thí nghiệm nghiên cứu")]
        for name, f in rowsL:
            L.append(f"| {name} | {f(c)} | {f(m)} |")
        L += ["", "`scripts/96_final_test.py` từ chối chạy nếu bất kỳ SHA256 nào ở trên hoặc code trong `src/`, `scripts/`, `configs/` "
              "khác với bản khóa, và chỉ chạy Test **một lần** (chỉ cho phép chạy tiếp các ca còn thiếu nếu bị gián đoạn)."]
    (REP / "kiem_tra_doc_lap.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--skip-duplicates", action="store_true", help="development only: a lock is never written without D checks")
    ap.add_argument("--no-lock", action="store_true", help="run the checks only")
    ap.add_argument("--supersede", default="", help="replace the lock (reason required); refused once the Test run has started")
    args = ap.parse_args()
    ck, info = run_checks(args)
    lock = None
    if ck.passed and not args.no_lock and not args.skip_duplicates:
        old = OUT / "locked_config.json"
        if old.exists() and not args.supersede:
            print("lock already exists: outputs/final_test/locked_config.json (never overwritten)")
            lock = json.loads(old.read_text(encoding="utf-8"))
        else:
            sup = None
            if old.exists():   # allowed only before ANY Test prediction; the old lock is kept and referenced
                if (OUT / "test_run.json").exists() or any(any((OUT / m / "cases").glob("*.json")) for m in ("ct", "mri")):
                    sys.exit("STOP: the Test run has started; the lock can no longer be replaced")
                prev = json.loads(old.read_text(encoding="utf-8"))
                n = 1 + len(list(OUT.glob("locked_config_superseded_*.json")))
                keep = OUT / f"locked_config_superseded_{n}.json"
                keep.write_text(old.read_text(encoding="utf-8"), encoding="utf-8")
                (REP / keep.name).write_text(old.read_text(encoding="utf-8"), encoding="utf-8")
                sup = {"file": keep.name, "commit": prev["git"]["commit"], "created": prev["created"], "reason": args.supersede,
                       "test_predictions_before": 0}
            lock = build_lock(info)
            if sup:
                lock["supersedes"] = sup
            lock["checks"] = ck.items
            OUT.mkdir(parents=True, exist_ok=True)
            REP.mkdir(parents=True, exist_ok=True)
            for p in (OUT / "locked_config.json", REP / "locked_config.json"):
                p.write_text(json.dumps(lock, indent=1, ensure_ascii=False), encoding="utf-8")
    write_report(ck, info, lock)
    print("PASS" if ck.passed else "FAIL", "->", REP / "kiem_tra_doc_lap.md")
    return 0 if ck.passed else 1


if __name__ == "__main__":
    sys.exit(main())
