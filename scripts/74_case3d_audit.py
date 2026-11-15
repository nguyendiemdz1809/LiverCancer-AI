"""Audit of the website's 3D CT output (suspicious slices, merged regions) against the source masks.

Runs exactly the website / locked-pipeline detection path (load_volume_uint8 -> analyze_volume with the locked CT
YOLO, conf 0.12, configs/roi.yaml merge + 3D linking, no track filter) and compares with GT masks (read-only):
  * every kept box: is its centre inside the liver (liver mask U lesion mask, as the liver mask excludes tumours)?
    is the slice a slice without any liver voxel?
  * every region (3D track): does it hit a GT lesion (IoU >= 0.3 with a GT lesion box on a common slice, the
    definition of scripts/71_eval_roi_val.py), only touch it, or not at all?
  * every GT lesion (3D connected component of the lesion mask): how many regions hit it (split), and does one region
    hit several lesions (merged)?
Volumes: the 92-slice case shown on the website (P0002 C1, a TRAIN patient) and every Validation CT volume.
Never Test. GT masks are used for this analysis only, never by the website.

    .venv\\Scripts\\python scripts\\74_case3d_audit.py       -> reports/case3d_audit.md, outputs/case3d_audit/
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLO_OFFLINE", "1")

from livercancer.config import load_paths  # noqa: E402
from livercancer.data.nifti_io import to_display  # noqa: E402
from livercancer.data.slices import load_volume_uint8  # noqa: E402
from livercancer.roi.boxes import iou  # noqa: E402

OUT = ROOT / "outputs" / "case3d_audit"
REPORT = ROOT / "reports" / "case3d_audit.md"
HIT_IOU = 0.3
SKIP_LESION = {("P0051", "P"), ("P0165", "C1"), ("P0231", "C3")}       # CLAUDE.md: empty / misaligned lesion masks
SKIP_LIVER = {("P0112", s) for s in ("C1", "C2", "C3", "P")} | {("P0175", s) for s in ("C1", "C2", "C3", "P")} \
    | {("P0193", s) for s in ("C1", "C2", "C3", "P")} | {("P0161", "P")}


def rows(paths):
    with open(paths["ct"]["csv"], encoding="utf-8") as f:
        return [r for r in csv.DictReader(l for l in f if not l.startswith("#"))]


def mask3d(path: Path, shape) -> np.ndarray | None:
    if not path.exists():
        return None
    m = np.asanyarray(nib.load(str(path)).dataobj.get_unscaled()) > 0
    if m.ndim > 3:
        m = m.reshape(m.shape[:3] + (-1,)).any(-1)
    return m if m.shape == tuple(shape) else None


def lesion_boxes(les: np.ndarray, k: int) -> list[tuple]:
    sl = to_display(les[:, :, k]).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(sl, connectivity=8)
    return [(float(x), float(y), float(x + w), float(y + h)) for x, y, w, h, _ in st[1:]]


def audit(vol_path, les_path, liver_path, det, key) -> dict:
    from scipy import ndimage
    from livercancer.roi.pipeline import analyze_volume
    vol = load_volume_uint8(vol_path)
    rr = analyze_volume(vol, det)
    n = vol.shape[2]
    raw = sum(len(v) for v in rr.raw.values())
    raw_slices = sum(1 for v in rr.raw.values() if v)
    kept = [(t, k, b) for t in rr.tracks for k, b in t.boxes.items()]
    sus = sorted({k for _, k, _ in kept})
    les = None if (key in SKIP_LESION or not les_path) else mask3d(les_path, vol.shape)
    liv = None if key in SKIP_LIVER else mask3d(liver_path, vol.shape)
    rec = {"volume": f"{key[0]}_{key[1]}", "n_slices": n, "raw_boxes": raw, "raw_slices": raw_slices,
           "kept_boxes": len(kept), "suspicious_slices": len(sus), "regions": len(rr.tracks),
           "has_lesion_gt": les is not None and bool(les.any()) if les_path else False, "control": les_path is None}
    if liv is not None and liv.any():
        organ = liv | (les if les is not None else False)
        organ_disp = {k: to_display(organ[:, :, k]) for k in sus}
        slice_has = organ.any(axis=(0, 1))
        rec["liver_slices"] = int(slice_has.sum())
        out_c = 0
        no_liver = set()
        for _, k, b in kept:
            x0, y0, x1, y1 = b.xyxy
            cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
            m = organ_disp[k]
            inside = 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1] and m[cy, cx]
            out_c += not inside
            if not slice_has[k]:
                no_liver.add(k)
        rec["boxes_centre_outside_liver"] = out_c
        rec["suspicious_slices_without_liver"] = len(no_liver)
        rec["regions_all_outside_liver"] = sum(
            all(not organ_disp[k][min(organ_disp[k].shape[0] - 1, int((b.xyxy[1] + b.xyxy[3]) / 2)),
                                  min(organ_disp[k].shape[1] - 1, int((b.xyxy[0] + b.xyxy[2]) / 2))] for k, b in t.boxes.items())
            for t in rr.tracks)
    if les is not None and les.any():
        lab, n_les = ndimage.label(les, structure=np.ones((3, 3, 3)))
        gtb = {}                                    # (lesion id, k) -> GT box on that slice
        for li in range(1, n_les + 1):
            ks = np.nonzero((lab == li).any(axis=(0, 1)))[0]
            for k in ks:
                m = to_display((lab[:, :, k] == li)).astype(np.uint8)
                ys, xs = np.nonzero(m)
                gtb[(li, int(k))] = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        # size of the YOLO boxes vs the real lesion: area(box) / area(GT lesion box) for every kept box overlapping a GT lesion
        ratios = []
        for t in rr.tracks:
            for k, bx in t.boxes.items():
                ov = [g for (li, kk), g in gtb.items() if kk == k and iou(bx.xyxy, g) > 0]
                if ov:
                    g = max(ov, key=lambda g: iou(bx.xyxy, g))
                    ratios.append(round(bx.area / max(1.0, (g[2] - g[0]) * (g[3] - g[1])), 3))
        rec["box_gt_area_ratios"] = ratios
        hits_per_track = []
        hit_by = Counter()
        for t in rr.tracks:
            hit = {li for (li, k), g in gtb.items() if k in t.boxes and iou(t.boxes[k].xyxy, g) >= HIT_IOU}
            touch = {li for (li, k), g in gtb.items() if k in t.boxes and iou(t.boxes[k].xyxy, g) > 0}
            hits_per_track.append((t, hit, touch))
            for li in hit:
                hit_by[li] += 1
        rec.update({"gt_lesions": n_les, "gt_lesions_detected": len(hit_by),
                    "regions_hitting_lesion": sum(1 for _, h, _ in hits_per_track if h),
                    "regions_touching_only": sum(1 for _, h, tc in hits_per_track if not h and tc),
                    "regions_without_lesion": sum(1 for _, h, tc in hits_per_track if not tc),
                    "lesions_split_in_several_regions": sum(1 for c in hit_by.values() if c > 1),
                    "regions_merging_several_lesions": sum(1 for _, h, _ in hits_per_track if len(h) > 1),
                    "lesion_slices": int(les.any(axis=(0, 1)).sum()),
                    "suspicious_slices_with_lesion": sum(1 for k in sus if les[:, :, k].any())})
        rec["regions_detail"] = [{"slices": f"{t.ks[0] + 1}-{t.ks[-1] + 1}", "n_slices": t.n_slices, "max_conf": round(t.max_conf, 3),
                                  "best_slice": t.best_slice + 1, "hits_lesion": bool(h), "touches_lesion": bool(tc)}
                                 for t, h, tc in hits_per_track]
    return rec, rr, les, liv


def figure(rr, les, liv, path: Path, n_cols=8):
    """Contact sheet: every suspicious slice with the website boxes (red), GT lesion (green) and liver (blue) outlines."""
    sus = sorted({k for t in rr.tracks for k in t.boxes})
    tiles = []
    for k in sus:
        img = cv2.cvtColor(rr.images[k], cv2.COLOR_GRAY2BGR)
        for m, col in ((liv, (235, 140, 40)), (les, (60, 200, 60))):
            if m is not None:
                cs, _ = cv2.findContours(to_display(m[:, :, k]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, col, 2)
        for _, b in rr.boxes_on(k):
            x0, y0, x1, y1 = (int(round(v)) for v in b.xyxy)
            cv2.rectangle(img, (x0, y0), (x1, y1), (40, 40, 235), 2)
            cv2.putText(img, f"{b.conf:.2f}", (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 235), 2)
        cv2.putText(img, f"lat {k + 1}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA))
    while len(tiles) % n_cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + n_cols]) for i in range(0, len(tiles), n_cols)])
    cv2.imwrite(str(path), grid)


def main() -> int:
    from livercancer.detection.yolo_detector import LesionDetector
    paths = load_paths()
    OUT.mkdir(parents=True, exist_ok=True)
    data = Path(paths["ct"]["data_dir"])
    split = {r["patient_id"]: r["split"] for r in csv.DictReader(open(ROOT / "data/splits/ct_split.csv", encoding="utf-8"))}
    det = LesionDetector.from_config("ct")
    todo = [r for r in rows(paths) if (r["patient_id"], r["phase"]) == ("P0002", "C1")]
    todo += [r for r in rows(paths) if split.get(r["patient_id"]) == "val"]
    recs = []
    for r in todo:
        key = (r["patient_id"], r["phase"])
        img = data / r["ct_path"]
        if not img.exists():
            continue
        les_path = None if r["mask_path"] in ("", "NA") else data / r["mask_path"]
        try:
            rec, rr, les, liv = audit(img, les_path, data / r["liver_mask_path"], det, key)
        except Exception as e:  # noqa: BLE001 - e.g. the truncated source file: reported
            recs.append({"volume": f"{key[0]}_{key[1]}", "error": type(e).__name__})
            continue
        rec["split"] = split.get(key[0])
        rec["label"] = r["cancer_type"]
        recs.append(rec)
        if key == ("P0002", "C1"):
            figure(rr, les, liv, OUT / "P0002_C1_lat_nghi_ngo.jpg")
            (OUT / "P0002_C1.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        print(rec["volume"], rec["split"], {k: rec.get(k) for k in ("n_slices", "suspicious_slices", "regions", "regions_hitting_lesion",
                                                                    "regions_without_lesion", "boxes_centre_outside_liver")}, flush=True)
    allr = [x for r in recs if r.get("split") == "val" for x in r.get("box_gt_area_ratios", [])]
    if allr:
        (OUT / "box_gt_area_ratio.json").write_text(json.dumps({"n_boxes": len(allr), "median": float(np.median(allr)),
            "p75": float(np.percentile(allr, 75)), "p90": float(np.percentile(allr, 90)),
            "share_over_4x": float(np.mean(np.array(allr) > 4))}, indent=1), encoding="utf-8")
    keys = sorted({k for r in recs for k in r if k not in ("regions_detail", "box_gt_area_ratios")})
    with open(OUT / "volumes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows([{k: r.get(k) for k in keys} for r in recs])
    report(recs)
    return 0


def report(recs: list[dict]) -> None:
    case = next(r for r in recs if r["volume"] == "P0002_C1")
    val = [r for r in recs if r.get("split") == "val" and "error" not in r]
    canc = [r for r in val if r.get("gt_lesions")]
    ctrl = [r for r in val if r.get("control")]
    liv = [r for r in val if "boxes_centre_outside_liver" in r]
    S = lambda rs, k: sum(r.get(k, 0) or 0 for r in rs)  # noqa: E731
    lines = ["# Kiểm toán kết quả 3D CT của website (lát nghi ngờ, vùng sau gộp)", "",
             "Tạo bởi `scripts/74_case3d_audit.py`; số liệu: `outputs/case3d_audit/` (volumes.csv, P0002_C1.json, ảnh P0002_C1_lat_nghi_ngo.jpg).",
             "Chạy đúng đường xử lý của website / pipeline đã khóa (YOLO CT ngưỡng 0,12, gộp box trong lát, nối vùng 3D theo configs/roi.yaml, "
             "không lọc vùng). Mask gốc chỉ dùng để kiểm toán, website không dùng. Không dùng Test.", "",
             "Định nghĩa: *lát nghi ngờ* = lát có ít nhất một box của các vùng được giữ; *vùng* = chuỗi box nối qua các lát liên tiếp; "
             f"vùng *trúng tổn thương* = IoU ≥ {HIT_IOU} với khung tổn thương thật trên một lát chung (cùng định nghĩa `scripts/71_eval_roi_val.py`); "
             "*trong gan* = tâm box nằm trong mask gan ∪ mask tổn thương.", "",
             "## Ca 92 lát trên website: P0002, pha C1 (bệnh nhân thuộc tập **Train**)", "",
             "Ca này thuộc tập huấn luyện: YOLO và Hybrid đã học trên chính ca này, nên kết quả (HCC 94,6%) **không** phản ánh "
             "khả năng của mô hình trên ca mới. Để trình diễn trung thực nên dùng ca Validation.", "",
             "| Chỉ số | Giá trị |", "|---|---|"]
    for k, lab in (("n_slices", "Số lát"), ("raw_boxes", "Box YOLO thô (≥ 0,12)"), ("raw_slices", "Lát có box YOLO thô"),
                   ("kept_boxes", "Box sau gộp trong lát"), ("suspicious_slices", "Lát nghi ngờ"), ("regions", "Vùng nghi ngờ (sau nối 3D)"),
                   ("liver_slices", "Lát có gan (mask)"), ("suspicious_slices_without_liver", "Lát nghi ngờ không có gan"),
                   ("boxes_centre_outside_liver", "Box có tâm ngoài gan"), ("regions_all_outside_liver", "Vùng nằm hoàn toàn ngoài gan"),
                   ("gt_lesions", "Tổn thương thật (khối 3D trong mask)"), ("gt_lesions_detected", "… được ít nhất một vùng trúng"),
                   ("lesion_slices", "Lát có tổn thương thật"), ("suspicious_slices_with_lesion", "Lát nghi ngờ có tổn thương thật"),
                   ("regions_hitting_lesion", "Vùng trúng tổn thương (IoU ≥ 0,3)"), ("regions_touching_only", "Vùng chỉ chạm tổn thương (IoU < 0,3)"),
                   ("regions_without_lesion", "Vùng không chạm tổn thương"),
                   ("lesions_split_in_several_regions", "Tổn thương bị tách thành nhiều vùng"),
                   ("regions_merging_several_lesions", "Vùng gộp nhầm nhiều tổn thương")):
        lines.append(f"| {lab} | {case.get(k, '—')} |")
    lines += ["", "Từng vùng của ca này:", "", "| Vùng (theo độ tin cậy) | Lát | Số lát | Độ tin cậy cao nhất | Lát đại diện | Trúng tổn thương | Chạm tổn thương |",
              "|---|---|---|---|---|---|---|"]
    for i, d in enumerate(case.get("regions_detail", []), 1):
        lines.append(f"| {i} | {d['slices']} | {d['n_slices']} | {d['max_conf']} | {d['best_slice']} | {'có' if d['hits_lesion'] else 'không'} | {'có' if d['touches_lesion'] else 'không'} |")
    lines += ["", f"## Toàn bộ CT Validation ({len(val)} volume; {len(canc)} có tổn thương, {len(ctrl)} nhóm chứng CN)", "",
              "| Chỉ số | Tổng |", "|---|---|",
              f"| Lát | {S(val, 'n_slices')} |", f"| Lát nghi ngờ | {S(val, 'suspicious_slices')} |",
              f"| Vùng nghi ngờ | {S(val, 'regions')} (trung bình {S(val, 'regions') / max(1, len(val)):.1f}/volume) |",
              f"| Volume CN (không có tổn thương) có vùng nghi ngờ | {sum(1 for r in ctrl if r['regions'])}/{len(ctrl)} |",
              f"| Box có tâm ngoài gan (volume có mask gan) | {S(liv, 'boxes_centre_outside_liver')}/{S(liv, 'kept_boxes')} |",
              f"| Lát nghi ngờ không có gan | {S(liv, 'suspicious_slices_without_liver')}/{S(liv, 'suspicious_slices')} |",
              f"| Vùng nằm hoàn toàn ngoài gan | {S(liv, 'regions_all_outside_liver')}/{S(liv, 'regions')} |",
              f"| Tổn thương thật được trúng | {S(canc, 'gt_lesions_detected')}/{S(canc, 'gt_lesions')} |",
              f"| Vùng trúng / chỉ chạm / không chạm tổn thương (volume có tổn thương) | {S(canc, 'regions_hitting_lesion')} / {S(canc, 'regions_touching_only')} / {S(canc, 'regions_without_lesion')} |",
              f"| Tổn thương bị tách thành nhiều vùng | {S(canc, 'lesions_split_in_several_regions')} |",
              f"| Vùng gộp nhầm nhiều tổn thương | {S(canc, 'regions_merging_several_lesions')} |", ""]
    ratio = OUT / "box_gt_area_ratio.json"
    if ratio.exists():
        q = json.loads(ratio.read_text(encoding="utf-8"))
        lines += [f"**Kích thước khung YOLO so với tổn thương thật** ({q['n_boxes']} khung chạm tổn thương, Validation): diện tích khung / diện tích "
                  f"khung tổn thương thật: trung vị {q['median']:.2f}, P75 {q['p75']:.2f}, P90 {q['p90']:.2f}; {100 * q['share_over_4x']:.0f}% khung lớn hơn 4 lần. "
                  "Đa số khung gần kích thước tổn thương, nhưng một phần nhỏ lớn hơn nhiều lần; khối bao 3D (bao các khung qua các lát) "
                  "vì vậy không đo kích thước khối u.", ""]
    lines += [
              "Lỗi đọc: " + (", ".join(f"{r['volume']} ({r['error']})" for r in recs if "error" in r) or "không có") + ".", ""]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
