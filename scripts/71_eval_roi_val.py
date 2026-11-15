"""GĐ3: evaluate Automatic Tumor Zoom on Validation volumes and choose the track filter. Never touches Test.

A  scan: every slice of every usable Val volume (CT patient x phase / MRI series; status ok in the exported dataset) is
   predicted with the final detector (configs/yolo_inference.yaml, conf 0.12). Cached in
   outputs/predictions/roi_val/<mod>/raw_dets.csv (re-used unless --rescan).
B  GT (analysis only): exported Val labels per slice, linked into 3D GT lesions with the same linker as the pipeline.
   Slices whose source mask has lesion voxels but which were not exported as positive (MRI: mask/JSON disagreement;
   CT: fragment-only slices) are 'unknown': a predicted track lying only on such slices is neither TP nor FP.
C  pipeline (merge -> link -> filter) for a grid of filters (min_track_slices x min_track_conf). A GT lesion is hit
   when a kept track has IoU >= --hit-iou (default 0.3: the zoom crop adds a margin) with it on a shared slice.
   Selection rule (fixed before running, Val only): among filters whose lesion recall is >= unfiltered - 0.02 and
   volume recall >= unfiltered - 0.01, take the one with the fewest false tracks per volume; ties -> smaller filter.
D  crop coverage of the chosen setting (fraction of the GT box inside the local / context window on the track's
   best slice) and before/after figures (original | boxes + zoom windows | local zoom | context zoom).
Outputs: outputs/predictions/roi_val/<mod>/{raw_dets.csv, grid.csv, choice.json, tracks.csv},
         reports/figures/roi_<mod>/*.png, reports/roi_zoom_val.md (both modalities), configs/roi.yaml (--write-config)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import CONFIG_DIR, PROJECT_ROOT, load_paths, load_yaml  # noqa: E402
from livercancer.data.slices import load_volume_uint8, slice_image  # noqa: E402
from livercancer.data.sources import image_path, lesion_mask_path  # noqa: E402
from livercancer.detection.yolo_detector import Detection, LesionDetector  # noqa: E402
from livercancer.roi.boxes import Box, iou, merge_boxes  # noqa: E402
from livercancer.roi.crop import context_window, local_window  # noqa: E402
from livercancer.roi.pipeline import load_roi_config  # noqa: E402
from livercancer.roi.crop import zoom  # noqa: E402
from livercancer.roi.render import draw, font, panel  # noqa: E402
from livercancer.roi.tracking import filter_tracks, link_tracks  # noqa: E402

GRID_SLICES = (1, 2, 3, 4, 5)
GRID_CONF = (0.12, 0.15, 0.2, 0.25, 0.3, 0.4)


def read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def val_volumes(data: Path) -> list[dict]:
    return [r for r in read_csv(data / "volumes.csv") if r["split"] == "val" and r["status"] == "ok"]


def gt_by_volume(data: Path) -> tuple[dict, dict]:
    """(patient, series) -> {k: [GT xyxy]} from exported Val labels; and -> set of exported positive slices."""
    gt, pos = defaultdict(dict), defaultdict(set)
    for r in read_csv(data / "slices.csv"):
        if r["split"] != "val" or r["kind"] != "pos":
            continue
        w, h = int(r["width"]), int(r["height"])
        boxes = []
        for line in (data / "labels" / "val" / f"{r['file']}.txt").read_text().splitlines():
            if line.strip():
                _, cx, cy, bw, bh = map(float, line.split())
                boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        key = (r["patient_id"], r["series"])
        gt[key][int(r["slice_k"])] = boxes
        pos[key].add(int(r["slice_k"]))
    return gt, pos


def mask_slices(path: Path | None) -> set[int]:
    if path is None or not path.exists():
        return set()
    m = np.asanyarray(nib.load(str(path)).dataobj.get_unscaled()) > 0
    if m.ndim > 3:
        m = m.reshape(m.shape[:3] + (-1,)).any(-1)
    return set(np.nonzero(m.any(axis=(0, 1)))[0].tolist())


def scan(mod: str, vols: list[dict], paths: dict, out: Path, device: str) -> None:
    det = LesionDetector.from_config(mod, device=device)
    split = vols[0]["split"] if vols else "?"
    print(f"scan {mod.upper()}: {len(vols)} {split} volumes on {det.device}, conf {det.conf}", flush=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "series", "n_slices", "slice_k", "x0", "y0", "x1", "y1", "conf"])
        for i, v in enumerate(vols):
            vol = load_volume_uint8(image_path(paths, mod, v["patient_id"], v["series"]))
            imgs = [slice_image(vol, k) for k in range(vol.shape[2])]
            dets = det.predict(imgs, batch_size=16)
            w.writerow([v["patient_id"], v["series"], len(imgs), -1, "", "", "", "", ""])  # volume marker (no box)
            for k, d in enumerate(dets):
                for x in d:
                    w.writerow([v["patient_id"], v["series"], len(imgs), k, round(x.x0, 2), round(x.y0, 2), round(x.x1, 2), round(x.y1, 2), round(x.conf, 5)])
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(vols)}", flush=True)
    print("written", out, flush=True)


def load_scan(path: Path) -> tuple[dict, dict]:
    dets, n = defaultdict(lambda: defaultdict(list)), {}
    for r in read_csv(path):
        key = (r["patient_id"], r["series"])
        n[key] = int(r["n_slices"])
        if int(r["slice_k"]) >= 0:
            dets[key][int(r["slice_k"])].append(Detection(*(float(r[c]) for c in ("x0", "y0", "x1", "y1")), conf=float(r["conf"])))
    return dets, n


def build_tracks(keys, dets, gt, params) -> tuple[dict, dict]:
    """Pipeline tracks (merge + link, unfiltered) and GT lesions per volume; computed once for the whole grid."""
    pred = {}
    for key in keys:
        merged = {k: merge_boxes(d, params["merge_iou"], params["merge_containment"]) for k, d in dets.get(key, {}).items()}
        pred[key] = link_tracks(merged, params["link_iou"], params["link_max_gap"])
    gt_tracks = {key: link_tracks({k: [Box(*b, 1.0) for b in bs] for k, bs in gt.get(key, {}).items()},
                                  params["link_iou"], params["link_max_gap"]) for key in keys}
    return pred, gt_tracks


def evaluate(mod, keys, pred, gt_tracks, unknown, labels, min_slices, min_conf, hit_iou):
    """Metrics of one filter setting; also returns per-track records."""
    n_les = hit_les = small_les = small_hit = 0
    vol_with_les = vol_hit = fp = 0
    neg_vol_with_track = 0
    pats_les, pats_hit, neg_pats, neg_pats_track = set(), set(), set(), set()
    recs = []
    for key in keys:
        tracks = filter_tracks(pred[key], min_slices, min_conf)
        les = gt_tracks[key]
        hit = set()
        for t in tracks:
            m = [li for li, g in enumerate(les) if any(k in g.boxes and iou(t.boxes[k].xyxy, g.boxes[k].xyxy) >= hit_iou for k in t.boxes)]
            hit.update(m)
            status = "TP" if m else ("unknown" if set(t.ks) <= unknown.get(key, set()) else "FP")
            fp += status == "FP"
            recs.append({"key": key, "track": t, "status": status, "lesions": m})
        n_les += len(les)
        hit_les += len(hit)
        for li, g in enumerate(les):
            if max(b.area for b in g.boxes.values()) < 32 ** 2:
                small_les += 1
                small_hit += li in hit
        pid = key[0]
        if les:
            vol_with_les += 1
            vol_hit += bool(hit)
            pats_les.add(pid)
            if hit:
                pats_hit.add(pid)
        if labels.get(pid) == "CN":
            neg_pats.add(pid)
            neg_vol_with_track += bool(tracks)
            if tracks:
                neg_pats_track.add(pid)
    n_neg_vol = sum(labels.get(k[0]) == "CN" for k in keys)
    m = {"min_track_slices": min_slices, "min_track_conf": min_conf, "gt_lesions": n_les,
         "lesion_recall": round(hit_les / max(n_les, 1), 4), "small_lesion_recall": round(small_hit / max(small_les, 1), 4),
         "n_small_lesions": small_les, "volume_recall": round(vol_hit / max(vol_with_les, 1), 4),
         "volumes_with_lesion": vol_with_les, "patients_detected": f"{len(pats_hit)}/{len(pats_les)}",
         "fp_tracks": fp, "fp_tracks_per_volume": round(fp / max(len(keys), 1), 3),
         "kept_tracks": len(recs), "unknown_tracks": sum(r["status"] == "unknown" for r in recs)}
    if mod == "ct":
        m.update({"cn_volumes_with_track": f"{neg_vol_with_track}/{n_neg_vol}", "cn_patients_with_track": f"{len(neg_pats_track)}/{len(neg_pats)}"})
    return m, recs


def coverage(win, b) -> float:
    ix = max(0.0, min(win[2], b[2]) - max(win[0], b[0]))
    iy = max(0.0, min(win[3], b[3]) - max(win[1], b[1]))
    a = (b[2] - b[0]) * (b[3] - b[1])
    return ix * iy / a if a > 0 else 0.0


def figures(mod, recs, gt, paths, params, fig_dir: Path, n: int, seed: int) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for old in fig_dir.glob("*.png"):
        old.unlink()
    rng = random.Random(seed)
    tp = [r for r in recs if r["status"] == "TP"]
    fp = [r for r in recs if r["status"] == "FP"]
    chosen = [("TP", r) for r in rng.sample(tp, min(n, len(tp)))] + [("FP", r) for r in rng.sample(fp, min(max(2, n // 3), len(fp)))]
    chosen.sort(key=lambda x: (x[1]["key"], x[0]))
    out, cache = [], {}
    for i, (kind, r) in enumerate(chosen, 1):
        key, t = r["key"], r["track"]
        if key not in cache:
            cache.clear()
            cache[key] = load_volume_uint8(image_path(paths, mod, *key))
        k = t.best_slice
        img = slice_image(cache[key], k)
        z = zoom(img, t.boxes[k].xyxy, params["crop_size"], params["margin_frac"], params["min_margin_px"],
                 params["context_scale"], params["context_min_px"])
        g = gt.get(key, {}).get(k, [])
        boxed = draw(img, [t.boxes[k].xyxy], g, [(*z["local_window"], "local"), (*z["context_window"], "context")],
                     [f"tổn thương {t.boxes[k].conf:.2f}"])
        title = (f"{key[0]} · {key[1]} · lát {k} · vùng {t.n_slices} lát ({t.ks[0]}–{t.ks[-1]}) · "
                 f"{'khớp GT' if kind == 'TP' else 'không khớp GT'} · khung xanh dương = vùng zoom local / context")
        strip = panel([img, boxed, z["local"], z["context"]], ["Ảnh gốc", "YOLO (đỏ) · GT (xanh lá)", "Zoom local", "Zoom context"],
                      height=300, title_size=15)
        im = Image.new("RGB", (strip.shape[1], strip.shape[0] + 26), "white")
        ImageDraw.Draw(im).text((4, 4), title, fill=(17, 24, 39), font=font(15))
        im.paste(Image.fromarray(strip), (0, 26))
        p = fig_dir / f"{i:02d}_{kind}_{key[0]}_{key[1].replace(' ', '')}_k{k:03d}.png"
        im.save(p)
        out.append(str(p.relative_to(PROJECT_ROOT)))
    return out


def run_modality(mod, args, paths) -> dict:
    cfg = load_yaml(CONFIG_DIR / "yolo_inference.yaml")[mod]
    data = PROJECT_ROOT / cfg["dataset"]
    out = paths["project"]["outputs"] / "predictions" / "roi_val" / mod
    out.mkdir(parents=True, exist_ok=True)
    vols = val_volumes(data)
    raw = out / "raw_dets.csv"
    if args.rescan or not raw.exists():
        scan(mod, vols, paths, raw, args.device)
    dets, n_slices = load_scan(raw)
    keys = [(v["patient_id"], v["series"]) for v in vols]
    missing = [k for k in keys if k not in n_slices]
    if missing:
        raise RuntimeError(f"{len(missing)} Val volumes missing from {raw}: rerun with --rescan")
    labels = {v["patient_id"]: v["label"] for v in vols}
    gt, pos = gt_by_volume(data)
    unknown = {}
    for v in vols:
        key = (v["patient_id"], v["series"])
        unknown[key] = mask_slices(lesion_mask_path(paths, mod, *key)) - pos.get(key, set())
    params = load_roi_config(mod)
    pred, gt_tracks = build_tracks(keys, dets, gt, params)
    grid = []
    for ms in GRID_SLICES:
        for mc in GRID_CONF:
            m, _ = evaluate(mod, keys, pred, gt_tracks, unknown, labels, ms, mc, args.hit_iou)
            grid.append(m)
    with open(out / "grid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0]))
        w.writeheader()
        w.writerows(grid)
    base = next(g for g in grid if g["min_track_slices"] == 1 and g["min_track_conf"] == GRID_CONF[0])
    ok = [g for g in grid if g["lesion_recall"] >= base["lesion_recall"] - 0.02 and g["volume_recall"] >= base["volume_recall"] - 0.01]
    best = min(ok, key=lambda g: (g["fp_tracks_per_volume"], g["min_track_slices"], g["min_track_conf"]))
    m, recs = evaluate(mod, keys, pred, gt_tracks, unknown, labels, best["min_track_slices"], best["min_track_conf"], args.hit_iou)
    m50, _ = evaluate(mod, keys, pred, gt_tracks, unknown, labels, best["min_track_slices"], best["min_track_conf"], 0.5)
    cov_l, cov_c, off = [], [], 0
    rows = []
    for r in recs:
        t, key = r["track"], r["key"]
        k = t.best_slice
        rows.append({"patient_id": key[0], "series": key[1], "track": t.track_id, "status": r["status"], "n_slices": t.n_slices,
                     "k_first": t.ks[0], "k_last": t.ks[-1], "best_slice": k, "max_conf": round(t.max_conf, 4),
                     "box": ";".join(f"{v:.1f}" for v in t.boxes[k].xyxy)})
        if r["status"] != "TP":
            continue
        g = [gt_tracks[key][li].boxes[k].xyxy for li in r["lesions"] if k in gt_tracks[key][li].boxes]
        if not g:  # the matched GT lesion is not annotated on the track's best slice
            off += 1
            continue
        b = t.boxes[k].xyxy
        gb = max(g, key=lambda x: iou(x, b))
        cov_l.append(coverage(local_window(b, (0, 0), params["margin_frac"], params["min_margin_px"]), gb))
        cov_c.append(coverage(context_window(b, (0, 0), params["context_scale"], params["context_min_px"]), gb))
    with open(out / "tracks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cov = {"tp_tracks": len(cov_l) + off, "best_slice_without_gt": off,
           "local_median": round(float(np.median(cov_l)), 3), "local_ge_0.95": round(float(np.mean(np.array(cov_l) >= 0.95)), 3),
           "context_median": round(float(np.median(cov_c)), 3), "context_ge_0.95": round(float(np.mean(np.array(cov_c) >= 0.95)), 3)}
    figs = figures(mod, recs, gt, paths, params, paths["project"]["reports"] / "figures" / f"roi_{mod}", args.n_figures, args.seed)
    choice = {"rule": "min FP tracks/volume s.t. lesion recall >= unfiltered-0.02 and volume recall >= unfiltered-0.01 (Val)",
              "hit_iou": args.hit_iou, "n_volumes": len(keys), "unfiltered": base, "chosen": m, "chosen_at_hit_iou_0.5": m50,
              "crop_coverage": cov, "figures": figs, "detector": {"weights": cfg["weights"], "sha256": cfg["sha256"], "conf": cfg["conf"]},
              "params": params, "time": datetime.now().isoformat(timespec="seconds")}
    (out / "choice.json").write_text(json.dumps(choice, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{mod.upper()} unfiltered: {base}\n{mod.upper()} chosen: {m}\n{mod.upper()} coverage: {cov}", flush=True)
    return choice


def write_config(choices: dict) -> None:
    """Replace the per-modality blocks of configs/roi.yaml with the filters chosen on Val (common block untouched)."""
    p = CONFIG_DIR / "roi.yaml"
    out, skip = [], False
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ct:", "mri:")):
            skip = True
            continue
        if skip and line.startswith("  "):
            continue
        skip = False
        out.append(line)
    for mod, c in choices.items():
        ch = c["chosen"]
        out += [f"{mod}:   # chosen on Val by scripts/71_eval_roi_val.py ({c['time']}): {c['rule']}",
                f"  min_track_slices: {ch['min_track_slices']}",
                f"  min_track_conf: {ch['min_track_conf']}"]
    p.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_report(choices: dict, path: Path) -> None:
    L = ["# Automatic Tumor Zoom trên Validation (CT và MRI)", "",
         f"*Sinh bởi `scripts/71_eval_roi_val.py` lúc {datetime.now():%Y-%m-%d %H:%M}. Detector và ngưỡng đã chốt (`configs/yolo_inference.yaml`, conf 0.12). "
         "Số liệu: `outputs/predictions/roi_val/<mod>/` (`raw_dets.csv`, `grid.csv`, `choice.json`, `tracks.csv`). "
         "Hình: `reports/figures/roi_<mod>/`. **Chỉ dùng Validation; Test không được dùng.***", "",
         "## 1. Cách làm", "",
         "1. Quét **mọi lát** của mỗi volume Val dùng được (CT: BN × pha; MRI: series) bằng YOLO ở conf 0.12.",
         "2. Gộp box trong cùng lát (IoU > 0.5 hoặc chứa nhau > 80%) → nối box qua các lát liên tiếp thành **track** (IoU ≥ 0.3, cho phép hở 1 lát).",
         "3. Lọc track theo số lát tối thiểu và confidence lớn nhất tối thiểu. **Quy tắc chọn (khai báo trước, chỉ trên Val):** trong các tổ hợp giữ "
         "recall tổn thương ≥ (không lọc − 0.02) và recall volume ≥ (không lọc − 0.01), chọn tổ hợp có ít track giả nhất trên mỗi volume.",
         "4. Mỗi track được zoom trên lát đại diện (diện tích × conf lớn nhất): **local** = box + margin max(15%, 8 px), vuông, 224×224; "
         "**context** = 2.5 × cạnh box (≥ 128 px), vuông, 224×224.",
         "5. GT (chỉ để đánh giá): nhãn Val đã xuất, nối thành tổn thương 3D bằng cùng thuật toán. Một tổn thương GT được tính là **có ROI** khi có track "
         "trùng với nó (IoU ≥ 0.3) trên ít nhất một lát chung. Lát có mask tổn thương nhưng không được xuất nhãn (MRI: mask và JSON không đồng thuận) "
         "là vùng *không xác định*: track chỉ nằm trên những lát đó không bị tính là track giả.", ""]
    for mod, c in choices.items():
        u, ch, c50, cv = c["unfiltered"], c["chosen"], c["chosen_at_hit_iou_0.5"], c["crop_coverage"]
        L += [f"## {'2' if mod == 'ct' else '3'}. {mod.upper()} ({c['n_volumes']} volume Val, {u['gt_lesions']} tổn thương GT 3D)", "",
              "| | Không lọc (1 lát, conf 0.12) | **Đã chọn** ({} lát, conf {}) | Đã chọn, tiêu chí IoU 0.5 |".format(ch["min_track_slices"], ch["min_track_conf"]),
              "|---|---|---|---|"]
        rows = [("Tổn thương GT có ROI (recall)", "lesion_recall"), ("Tổn thương nhỏ < 32² px có ROI", "small_lesion_recall"),
                ("Volume có tổn thương có ≥ 1 ROI đúng", "volume_recall"), ("BN có tổn thương có ROI", "patients_detected"),
                ("Track giả / volume", "fp_tracks_per_volume"), ("Tổng track giữ lại", "kept_tracks"), ("Track vùng không xác định", "unknown_tracks")]
        if mod == "ct":
            rows += [("Volume BN CN có ≥ 1 track", "cn_volumes_with_track"), ("BN CN có ≥ 1 track", "cn_patients_with_track")]
        for lab, k in rows:
            L.append(f"| {lab} | {u[k]} | **{ch[k]}** | {c50[k]} |")
        L += ["", f"**Độ phủ của crop** (track khớp GT, trên lát đại diện; {cv['tp_tracks']} track, trong đó {cv['best_slice_without_gt']} có lát đại diện "
              f"không có box GT): phần box GT nằm trong khung local, trung vị **{cv['local_median']}** ({100 * cv['local_ge_0.95']:.1f}% track ≥ 0.95); "
              f"trong khung context, trung vị **{cv['context_median']}** ({100 * cv['context_ge_0.95']:.1f}% ≥ 0.95).", "",
              "Hình trước/sau (ảnh gốc | box + khung zoom | zoom local | zoom context):", ""]
        L += [f"- [{Path(f).name}]({Path(f).relative_to('reports').as_posix()})" for f in c["figures"]]
        L.append("")
    L += ["<!-- ANALYST-NOTES -->", "", "## 4. Nhận xét", "", "(người phân tích bổ sung)"]
    keep = ""
    if path.exists() and "<!-- ANALYST-NOTES -->" in path.read_text(encoding="utf-8"):
        keep = path.read_text(encoding="utf-8").split("<!-- ANALYST-NOTES -->", 1)[1]
    txt = "\n".join(L) + "\n"
    if keep.strip() and "(người phân tích bổ sung)" not in keep:
        txt = txt.split("<!-- ANALYST-NOTES -->", 1)[0] + "<!-- ANALYST-NOTES -->" + keep
    path.write_text(txt, encoding="utf-8")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["ct", "mri", "both"], default="both")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rescan", action="store_true", help="re-run YOLO on all Val volumes")
    ap.add_argument("--hit-iou", type=float, default=0.3)
    ap.add_argument("--n-figures", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-config", action="store_true", help="store the chosen filter in configs/roi.yaml")
    args = ap.parse_args()
    paths = load_paths()
    mods = ["ct", "mri"] if args.modality == "both" else [args.modality]
    choices = {m: run_modality(m, args, paths) for m in mods}
    if args.write_config:
        write_config(choices)
        print("updated configs/roi.yaml")
    if args.modality == "both":
        write_report(choices, paths["project"]["reports"] / "roi_zoom_val.md")
        print("written reports/roi_zoom_val.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
