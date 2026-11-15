"""GĐ4: build the lesion-classification ROI dataset (Train + Val only; Test is never exported here).

Stages (--stages, default: gt yolo_val merge):
  gt         GT lesions: exported YOLO labels (QA'd, pixel-identical to NIfTI) linked into 3D lesions per volume
             (same linker as Tumor Zoom); up to N key slices per lesion (largest area first, slices >= min_gap apart);
             local + context crops (livercancer.roi.crop.zoom, configs/roi.yaml) saved as 224x224 uint8 PNG.
  yolo_val   Pipeline simulation on Val: cached YOLO scan (outputs/predictions/roi_val/<mod>/raw_dets.csv, conf 0.12)
             -> merge -> 3D tracks -> ALL kept tracks (also false ones and CN volumes) with matches_gt / gt_iou flags.
  yolo_train Same on Train (scans Train volumes once, cached in outputs/predictions/roi_train/<mod>/raw_dets.csv);
             only tracks matching a GT lesion are kept (YOLO saw these images during training: optimistic).
  merge      manifest.csv (+ patients.csv), patient-level split check against the frozen splits.
  pack       cls_<mod>.zip (+ livercancer_code.zip, SHA256SUMS_cls.txt) for Colab.
Labels: CT = patient diagnosis (patient_data.csv), MRI = lesion category (JSON); never derived from boxes.
CN (CT controls) has no lesion -> only appears as YOLO false-ROI rows (label CN, label_id -1), never trained on.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import CONFIG_DIR, load_paths, load_yaml  # noqa: E402
from livercancer.data.slices import load_volume_uint8, slice_image  # noqa: E402
from livercancer.data.sources import image_path  # noqa: E402
from livercancer.roi.boxes import Box, iou, merge_boxes  # noqa: E402
from livercancer.roi.crop import zoom  # noqa: E402
from livercancer.roi.pipeline import load_roi_config  # noqa: E402
from livercancer.roi.tracking import Track, filter_tracks, link_tracks  # noqa: E402

_spec = importlib.util.spec_from_file_location("roi_eval", Path(__file__).with_name("71_eval_roi_val.py"))
roi_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roi_eval)

SPLITS = ("train", "val")
FIELDS = ["roi_id", "modality", "split", "source", "patient_id", "series", "volume_id", "lesion_id", "slice_k",
          "label", "label_id", "group", "x0", "y0", "x1", "y1", "box_w", "box_h", "conf", "matches_gt", "gt_iou",
          "lesion_n_slices", "local_path", "context_path"]


def read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def write_csv(rows: list[dict], p: Path, fields: list[str]) -> None:
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9+\-]", "", s)


def labels_for(mod: str, paths: dict, cfg: dict) -> tuple[dict, dict]:
    """patient -> (split, label) from the frozen split file; class -> id; never from boxes."""
    rows = read_csv(paths["project"]["splits"] / f"{mod}_split.csv")
    pats = {r["patient_id"]: (r["split"], r["label"]) for r in rows}
    return pats, {c: i for i, c in enumerate(cfg["classes"])}


def group_of(mod: str, label: str, cfg: dict) -> str:
    if mod == "ct":
        return "Malignant" if label in cfg["classes"] else "Control"
    return next((g for g, cs in cfg["groups"].items() if label in cs), "")


def choose_slices(track: Track, n: int, min_gap: int, score=lambda b: b.area) -> list[int]:
    out: list[int] = []
    for k in sorted(track.boxes, key=lambda s: -score(track.boxes[s])):
        if all(abs(k - c) >= min_gap for c in out):
            out.append(k)
        if len(out) == n:
            break
    return sorted(out)


def gt_lesions(data: Path, split: str) -> tuple[dict, dict]:
    """(patient, series) -> GT 3D lesions (tracks of GT boxes); and -> {k: exported PNG path}."""
    per_vol, files = defaultdict(dict), defaultdict(dict)
    for r in read_csv(data / "slices.csv"):
        if r["split"] != split or r["kind"] != "pos":
            continue
        w, h = int(r["width"]), int(r["height"])
        bs = []
        for line in (data / "labels" / split / f"{r['file']}.txt").read_text().splitlines():
            if line.strip():
                _, cx, cy, bw, bh = map(float, line.split())
                bs.append(Box((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, 1.0))
        key = (r["patient_id"], r["series"])
        per_vol[key][int(r["slice_k"])] = bs
        files[key][int(r["slice_k"])] = data / "images" / split / f"{r['file']}.png"
    params = load_roi_config("ct")  # link params are common
    return {k: link_tracks(v, params["link_iou"], params["link_max_gap"]) for k, v in per_vol.items()}, files


def save_crops(img, box, params, out_dir: Path, roi_id: str) -> tuple[str, str]:
    z = zoom(img, box, params["crop_size"], params["margin_frac"], params["min_margin_px"], params["context_scale"], params["context_min_px"])
    lp, cp = out_dir / f"{roi_id}_local.png", out_dir / f"{roi_id}_context.png"
    if not (cv2.imwrite(str(lp), z["local"]) and cv2.imwrite(str(cp), z["context"])):
        raise IOError(f"cv2.imwrite failed for {roi_id}")
    return lp.name, cp.name


def stage_gt(mod, cfg, paths, out: Path, pats, cls_id) -> list[dict]:
    data = ROOT / cfg["yolo_dataset"]
    params = load_roi_config(mod)
    n_per = load_yaml(CONFIG_DIR / "cls.yaml")["roi"]["slices_per_lesion"]
    rows = []
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        lesions, files = gt_lesions(data, split)
        for (pid, series), tracks in sorted(lesions.items()):
            sp, label = pats[pid]
            assert sp == split, f"{pid}: exported in {split} but frozen split is {sp}"
            for li, t in enumerate(tracks):
                for k in choose_slices(t, n_per[split], cfg["slice_min_gap"]):
                    img = cv2.imread(str(files[(pid, series)][k]), cv2.IMREAD_GRAYSCALE)
                    b = t.boxes[k]
                    roi_id = f"{mod}_{split}_gt_{pid}_{safe(series)}_L{li}_k{k:03d}"
                    lp, cp = save_crops(img, b.xyxy, params, out / "images" / split, roi_id)
                    rows.append(row(mod, roi_id, split, "gt", pid, series, li, k, label, cls_id, cfg, b, 1.0, True, 1.0, t.n_slices,
                                    f"images/{split}/{lp}", f"images/{split}/{cp}"))
        print(f"  gt {split}: {sum(r['split'] == split for r in rows)} ROIs", flush=True)
    return rows


def row(mod, roi_id, split, source, pid, series, li, k, label, cls_id, cfg, b, conf, matched, gt_iou, n_sl, lp, cp) -> dict:
    return {"roi_id": roi_id, "modality": mod.upper(), "split": split, "source": source, "patient_id": pid, "series": series,
            "volume_id": f"{pid}_{safe(series)}", "lesion_id": li, "slice_k": k, "label": label, "label_id": cls_id.get(label, -1),
            "group": group_of(mod, label, cfg), "x0": round(b.x0, 1), "y0": round(b.y0, 1), "x1": round(b.x1, 1), "y1": round(b.y1, 1),
            "box_w": round(b.x1 - b.x0, 1), "box_h": round(b.y1 - b.y0, 1), "conf": round(conf, 4), "matches_gt": int(matched),
            "gt_iou": round(gt_iou, 3), "lesion_n_slices": n_sl, "local_path": lp, "context_path": cp}


def stage_yolo(mod, cfg, paths, out: Path, pats, cls_id, split: str, device: str) -> list[dict]:
    data = ROOT / cfg["yolo_dataset"]
    params = load_roi_config(mod)
    n_per = load_yaml(CONFIG_DIR / "cls.yaml")["roi"]["slices_per_lesion"][split]
    vols = [r for r in read_csv(data / "volumes.csv") if r["split"] == split and r["status"] == "ok"]
    cache = paths["project"]["outputs"] / "predictions" / f"roi_{split}" / mod / "raw_dets.csv"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        roi_eval.scan(mod, vols, paths, cache, device)
    dets, _ = roi_eval.load_scan(cache)
    lesions, _ = gt_lesions(data, split)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    rows = []
    stats = Counter()
    for vi, v in enumerate(vols):
        pid, series = v["patient_id"], v["series"]
        sp, label = pats[pid]
        assert sp == split
        merged = {k: merge_boxes(d, params["merge_iou"], params["merge_containment"]) for k, d in dets.get((pid, series), {}).items()}
        tracks = filter_tracks(link_tracks(merged, params["link_iou"], params["link_max_gap"]), params["min_track_slices"], params["min_track_conf"])
        gts = lesions.get((pid, series), [])
        todo = []
        for ti, t in enumerate(sorted(tracks, key=lambda t: -t.max_conf)):
            best = max((iou(t.boxes[k].xyxy, b.xyxy) for g in gts for k in t.boxes if k in g.boxes for b in [g.boxes[k]]), default=0.0)
            matched = best >= 0.3
            stats["tracks"] += 1
            stats["matched"] += matched
            if split == "train" and not matched:
                continue
            todo.append((ti, t, matched, best))
        if not todo:
            continue
        vol = load_volume_uint8(image_path(paths, mod, pid, series))
        for ti, t, matched, best in todo:
            for k in choose_slices(t, n_per, cfg["slice_min_gap"], score=lambda b: b.area * b.conf):
                b = t.boxes[k]
                roi_id = f"{mod}_{split}_yolo_{pid}_{safe(series)}_T{ti}_k{k:03d}"
                lp, cp = save_crops(slice_image(vol, k), b.xyxy, params, out / "images" / split, roi_id)
                rows.append(row(mod, roi_id, split, "yolo", pid, series, ti, k, label, cls_id, cfg, b, b.conf, matched, best,
                                t.n_slices, f"images/{split}/{lp}", f"images/{split}/{cp}"))
        del vol
        if (vi + 1) % 100 == 0:
            print(f"  yolo {split}: {vi + 1}/{len(vols)} volumes, {len(rows)} ROIs", flush=True)
    print(f"  yolo {split}: {stats['tracks']} tracks ({stats['matched']} match GT) -> {len(rows)} ROIs", flush=True)
    return rows


def merge(mod, cfg, out: Path, pats) -> dict:
    rows = []
    for name in ("manifest_gt.csv", "manifest_yolo_val.csv", "manifest_yolo_train.csv"):
        if (out / name).exists():
            rows += read_csv(out / name)
    # patient-level independence: every patient in exactly one split, equal to the frozen split; no Test at all
    seen = defaultdict(set)
    for r in rows:
        seen[r["patient_id"]].add(r["split"])
        assert r["split"] in SPLITS, f"unexpected split {r['split']}"
        assert pats[r["patient_id"]][0] == r["split"], f"{r['patient_id']} not in its frozen split"
        assert (out / r["local_path"]).exists() and (out / r["context_path"]).exists(), r["roi_id"]
    assert all(len(s) == 1 for s in seen.values()), "patient in several splits"
    write_csv(rows, out / "manifest.csv", FIELDS)
    pat_rows = []
    for pid in sorted(seen):
        rs = [r for r in rows if r["patient_id"] == pid]
        pat_rows.append({"patient_id": pid, "split": pats[pid][0], "label": pats[pid][1],
                         "label_id": next(iter({r["label_id"] for r in rs})), "n_gt_rois": sum(r["source"] == "gt" for r in rs),
                         "n_yolo_rois": sum(r["source"] == "yolo" for r in rs),
                         "n_yolo_rois_matched": sum(r["source"] == "yolo" and r["matches_gt"] == "1" for r in rs)})
    write_csv(pat_rows, out / "patients.csv", list(pat_rows[0]))
    summ = {"modality": mod, "classes": cfg["classes"], "n_rois": len(rows), "time": datetime.now().isoformat(timespec="seconds")}
    tab = defaultdict(Counter)
    for r in rows:
        tab[(r["split"], r["source"], r["label"])]["rois"] += 1
    for p in pat_rows:
        for src, n in (("gt", p["n_gt_rois"]), ("yolo", p["n_yolo_rois"])):
            if n:
                tab[(p["split"], src, p["label"])]["patients"] += 1
    summ["table"] = {"|".join(k): dict(v) for k, v in sorted(tab.items())}
    (out / "class_map.json").write_text(json.dumps({"classes": cfg["classes"], "class_names_vi": cfg.get("class_names_vi", cfg["classes"]),
                                                    "groups": cfg.get("groups", {})}, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  merged {len(rows)} ROIs, {len(pat_rows)} patients", flush=True)
    return summ


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pack(mods: list[str], root_out: Path) -> None:
    sums = []
    for mod in mods:
        src = root_out / mod
        zp = root_out / f"cls_{mod}.zip"
        files = sorted(p for p in src.rglob("*") if p.is_file() and not p.name.startswith("manifest_"))
        assert not any("test" in p.relative_to(src).parts for p in files), "Test must not be packed"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:  # PNG is already compressed
            for p in files:
                z.write(p, f"cls_{mod}/{p.relative_to(src).as_posix()}")
        sums.append((sha256(zp), zp.name))
        print(f"  {zp.name}: {len(files)} files, {zp.stat().st_size / 1e9:.2f} GB", flush=True)
    code = root_out / "livercancer_code.zip"
    with zipfile.ZipFile(code, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / "src" / "livercancer").rglob("*.py")):
            z.write(p, p.relative_to(ROOT).as_posix())
        for name in ("cls.yaml", "roi.yaml", "paths.yaml"):
            z.write(CONFIG_DIR / name, f"configs/{name}")
    sums.append((sha256(code), code.name))
    (root_out / "SHA256SUMS_cls.txt").write_text("".join(f"{h}  {n}\n" for h, n in sums), encoding="utf-8")
    print("  written", root_out / "SHA256SUMS_cls.txt", flush=True)


def report(mods: list[str], root_out: Path, paths: dict) -> None:
    """reports/cls_data_report.md: verified labels, ROI / patient counts per split, source and class."""
    allc = load_yaml(CONFIG_DIR / "cls.yaml")
    L = ["# Dữ liệu phân loại tổn thương (ROI) — CT và MRI", "",
         f"*Sinh bởi `scripts/80_build_cls_dataset.py --stages report` lúc {datetime.now():%Y-%m-%d %H:%M}. Dữ liệu: `{allc['roi']['out_dir']}/<mod>/` "
         "(`manifest.csv`, `patients.csv`, `summary.json`, `class_map.json`). **Chỉ Train và Val; Test không được xuất.***", "",
         "## 1. Nhãn đã xác minh", "",
         "| | CT — PLC-CECT | MRI — LLD-MMRI |", "|---|---|---|",
         "| Nguồn nhãn | `patient_data.csv` → `cancer_type` (chẩn đoán mô học/tế bào học, **mức bệnh nhân**) | `LLD_MMRI_Annotation.json` → `lesion.category` (**mức tổn thương**, 1 tổn thương/series) |",
         "| Kiểm tra | 361 BN, mỗi BN đúng 1 nhãn ở cả 4 pha; khớp 100% `data/splits/ct_split.csv` | 498 BN × 8 series; nhãn giống nhau ở cả 8 series của mỗi BN; khớp 100% `mri_split.csv`; `num_targets = 1` ở cả 3.984 series |",
         f"| Lớp phân loại | {', '.join(allc['ct']['classes'])} (CN = nhóm chứng không có tổn thương → **không** phải lớp) | {', '.join(allc['mri']['classes'])} (+ nhóm Lành tính / Ác tính) |",
         "| Gán nhãn cho ROI | mọi tổn thương của BN nhận nhãn chẩn đoán của BN (nhãn yếu ở mức tổn thương) | nhãn của chính tổn thương |",
         "| Không làm | không suy nhãn từ bounding box; không tạo nhãn cho BN CN; không dùng Test | như CT; series bị cách ly (`mri_bbox_quarantine.csv`) không tạo ROI-GT |", ""]
    for i, mod in enumerate(mods, 2):
        out = root_out / mod
        if not (out / "summary.json").exists():
            continue
        s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        pats = read_csv(out / "patients.csv")
        L += [f"## {i}. {mod.upper()} — {s['n_rois']} ROI", "",
              "| Lớp | Train: BN / ROI-GT | Train: ROI-YOLO khớp GT | Val: BN / ROI-GT | Val: ROI-YOLO (tất cả / khớp GT) |", "|---|---|---|---|---|"]
        man = read_csv(out / "manifest.csv")
        for c in allc[mod]["classes"] + (["CN"] if mod == "ct" else []):
            def n(split, src, matched=None):
                return sum(r["split"] == split and r["source"] == src and r["label"] == c and (matched is None or r["matches_gt"] == matched) for r in man)
            bn = lambda split: sum(p["split"] == split and p["label"] == c and int(p["n_gt_rois"]) > 0 for p in pats)  # noqa: E731
            L.append(f"| {c} | {bn('train')} / {n('train', 'gt')} | {n('train', 'yolo')} | {bn('val')} / {n('val', 'gt')} | "
                     f"{n('val', 'yolo')} / {n('val', 'yolo', '1')} |")
        L.append("")
    L += ["## Cách tạo ROI", "",
          "- **ROI-GT:** box GT đã xuất ở GĐ2 (đã QA, ảnh trùng từng pixel với NIfTI) được nối thành tổn thương 3D bằng cùng thuật toán của Tumor Zoom; "
          f"mỗi tổn thương lấy tối đa {allc['roi']['slices_per_lesion']['train']} (Train) / {allc['roi']['slices_per_lesion']['val']} (Val) lát có diện tích lớn nhất "
          "(CT: các lát cách nhau ≥ 2). Hai ảnh 224×224 uint8: **local** (box + lề 15%, ≥ 8 px) và **context** (2.5× cạnh box, ≥ 128 px).",
          "- **ROI-YOLO (Val):** mô phỏng pipeline thật: YOLO (conf 0.12) → gộp box → track 3D (configs/roi.yaml) → mọi track được giữ, kể cả track giả và track trên BN CN; "
          "cột `matches_gt` (IoU ≥ 0.3 với một tổn thương GT) chỉ dùng để phân tích.",
          "- **ROI-YOLO (Train, tùy chọn):** chỉ giữ track khớp GT, nên nhãn của BN/tổn thương áp đúng vào tổn thương thật. YOLO đã thấy các ảnh này khi train, nên đây là dữ liệu lạc quan.",
          "- **Không có ROI:** BN/volume mà YOLO không tạo track nào sẽ không có dòng ROI-YOLO. Khi đánh giá pipeline, những BN này bị tính là **sai** (`pipeline_patient` trong đánh giá).",
          "- **Chuẩn hóa cho backbone pretrained:** ảnh xám → 3 kênh giống nhau → [0, 1] → chuẩn hóa mean/std ImageNet (`livercancer.classification.data.to_tensor`).", ""]
    p = paths["project"]["reports"] / "cls_data_report.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("  written", p, flush=True)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["ct", "mri", "both"], default="both")
    ap.add_argument("--stages", nargs="+", default=["gt", "yolo_val", "merge"],
                    choices=["gt", "yolo_val", "yolo_train", "merge", "pack", "report"])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    paths = load_paths()
    allc = load_yaml(CONFIG_DIR / "cls.yaml")
    root_out = ROOT / allc["roi"]["out_dir"]
    mods = ["ct", "mri"] if args.modality == "both" else [args.modality]
    for mod in mods:
        cfg = allc[mod]
        out = root_out / mod
        out.mkdir(parents=True, exist_ok=True)
        pats, cls_id = labels_for(mod, paths, cfg)
        print(f"== {mod.upper()} classes {cfg['classes']}", flush=True)
        if "gt" in args.stages:
            write_csv(stage_gt(mod, cfg, paths, out, pats, cls_id), out / "manifest_gt.csv", FIELDS)
        if "yolo_val" in args.stages:
            write_csv(stage_yolo(mod, cfg, paths, out, pats, cls_id, "val", args.device), out / "manifest_yolo_val.csv", FIELDS)
        if "yolo_train" in args.stages:
            write_csv(stage_yolo(mod, cfg, paths, out, pats, cls_id, "train", args.device), out / "manifest_yolo_train.csv", FIELDS)
        if "merge" in args.stages:
            merge(mod, cfg, out, pats)
    if "report" in args.stages:
        report(["ct", "mri"], root_out, paths)
    if "pack" in args.stages:
        pack(mods, root_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
