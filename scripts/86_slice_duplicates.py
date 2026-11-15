"""Exact-duplicate slices in Validation volumes, and equivalence of the duplicate-skipping YOLO run (website 3D intake).

For every Validation volume (CT: all phases on disk; MRI: all 8 sequences), slices are produced exactly as the AI sees
them (load_volume_uint8 -> slice_image). Byte-identical slices are counted (livercancer.intake.dedup). For every volume
that has duplicates, YOLO is run twice through analyze_volume - plain and through DedupDetector - and the detections,
3D tracks and Tumor Zoom boxes are compared. Validation only (never Test). Read-only on the dataset.

    .venv\\Scripts\\python scripts\\86_slice_duplicates.py            -> reports/slice_duplicates.md, outputs/slice_duplicates/
    ... --mechanism-only   reuse outputs/slice_duplicates/summary.json, redo only the mechanism check

Mechanism check: Validation has no duplicate slice at all, so the copy-back path is also checked on 3 CT + 3 MRI
Validation volumes into which copies of 6 slices around the most confident detection were INSERTED (a constructed
test input, not a real scan, never used as a result): plain YOLO vs DedupDetector must give the same detections per
slice and the same 3D tracks.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLO_OFFLINE", "1")

from livercancer.config import load_paths  # noqa: E402
from livercancer.data.slices import load_volume_uint8, slice_image  # noqa: E402
from livercancer.intake.dedup import DedupDetector, find_duplicates  # noqa: E402

OUT = ROOT / "outputs" / "slice_duplicates"
REPORT = ROOT / "reports" / "slice_duplicates.md"


def val_ids(mod: str) -> list[str]:
    with open(ROOT / "data" / "splits" / f"{mod}_split.csv", encoding="utf-8") as f:
        return [r["patient_id"] for r in csv.DictReader(f) if r["split"] == "val"]


def volumes(paths: dict, mod: str):
    for pid in val_ids(mod):
        if mod == "ct":
            files = sorted(glob.glob(os.path.join(paths["ct"]["data_dir"], "ct_files", f"{pid}_ct_*.nii.gz")))
        else:
            files = sorted(glob.glob(os.path.join(paths["mri"]["images"], f"{pid}_*_0000.nii.gz")))
        for f in files:
            yield pid, Path(f)


def tracks_sig(rr) -> list:
    return [(t.ks[0], t.ks[-1], round(t.max_conf, 6), t.best_slice, tuple(round(v, 3) for v in t.boxes[t.best_slice].xyxy))
            for t in rr.tracks]


def det_sig(dets) -> list:
    return [(round(d.x0, 3), round(d.y0, 3), round(d.x1, 3), round(d.y1, 3), round(d.conf, 5)) for d in dets]


def mechanism(paths) -> list[dict]:
    import numpy as np
    from livercancer.detection.yolo_detector import LesionDetector
    from livercancer.roi.pipeline import analyze_volume
    out = []
    for mod in ("ct", "mri"):
        det = LesionDetector.from_config(mod)
        picked = 0
        for pid, f in volumes(paths, mod):
            if picked == 3:
                break
            vol = load_volume_uint8(f)
            a = analyze_volume(vol, det)
            if not a.tracks:
                continue
            picked += 1
            kc = a.tracks[0].best_slice
            ks = [k for k in range(max(0, kc - 3), min(vol.shape[2], kc + 3))]
            order = []
            for k in range(vol.shape[2]):
                order.append(k)
                if k in ks:
                    order.append(k)                        # an exact copy right after the slice
            vol2 = np.ascontiguousarray(vol[:, :, order])
            plain = analyze_volume(vol2, det)
            dd = DedupDetector(det)
            dedup = analyze_volume(vol2, dd)
            st = dd.stats[0]
            px = cf = 0.0
            nb = 0
            for k in range(vol2.shape[2]):
                x, y = plain.raw[k], dedup.raw[k]
                if len(x) != len(y):
                    nb += 1
                    continue
                for d, e in zip(x, y):
                    px = max(px, abs(d.x0 - e.x0), abs(d.y0 - e.y0), abs(d.x1 - e.x1), abs(d.y1 - e.y1))
                    cf = max(cf, abs(d.conf - e.conf))
            out.append({"modality": mod, "patient": pid, "file": f.name, "n_slices": vol2.shape[2], "inserted_copies": len(ks),
                        "duplicates_found": st.n_duplicates, "yolo_slices": st.n_unique,
                        "raw_identical": all(det_sig(plain.raw[k]) == det_sig(dedup.raw[k]) for k in range(vol2.shape[2])),
                        "slices_box_count_differs": nb, "max_box_diff_px": round(px, 4), "max_conf_diff": round(cf, 6),
                        "tracks_identical": tracks_sig(plain) == tracks_sig(dedup), "n_tracks": len(plain.tracks)})
            print("mechanism", out[-1], flush=True)
    return out


def main() -> int:
    from livercancer.detection.yolo_detector import LesionDetector
    from livercancer.roi.pipeline import analyze_volume
    paths = load_paths()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, comps = [], []
    if "--mechanism-only" in sys.argv:
        summ = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
        return report(summ, mechanism(paths))
    for mod in ("ct", "mri"):
        det = LesionDetector.from_config(mod)
        for pid, f in volumes(paths, mod):
            try:
                vol = load_volume_uint8(f)
            except Exception as e:  # noqa: BLE001 - e.g. a truncated archive: reported, not hidden
                rows.append({"modality": mod, "patient": pid, "file": f.name, "error": type(e).__name__})
                continue
            images = [slice_image(vol, k) for k in range(vol.shape[2])]
            _, _, st = find_duplicates(images)
            rows.append({"modality": mod, "patient": pid, "file": f.name, "n_slices": st.n_slices, "n_unique": st.n_unique,
                         "n_duplicates": st.n_duplicates, "groups": len(st.groups), "blank_groups": st.blank_groups,
                         "nonblank_dup_slices": sum(len(g) - 1 for g in st.groups if int(images[g[0]].max()) != int(images[g[0]].min())),
                         "error": ""})
            if st.n_duplicates:
                t0 = time.perf_counter()
                a = analyze_volume(vol, det)
                t1 = time.perf_counter()
                dd = DedupDetector(det)
                b = analyze_volume(vol, dd)
                t2 = time.perf_counter()
                raw_same = all([(round(d.x0, 3), round(d.y0, 3), round(d.x1, 3), round(d.y1, 3), round(d.conf, 5)) for d in a.raw[k]]
                               == [(round(d.x0, 3), round(d.y0, 3), round(d.x1, 3), round(d.y1, 3), round(d.conf, 5)) for d in b.raw[k]]
                               for k in range(st.n_slices))
                dup_det = sum(1 for g in st.groups for k in g if a.raw[k])
                comps.append({"modality": mod, "patient": pid, "file": f.name, "n_duplicates": st.n_duplicates,
                              "raw_identical": raw_same, "tracks_identical": tracks_sig(a) == tracks_sig(b),
                              "n_tracks": len(a.tracks), "dup_slices_with_boxes": dup_det,
                              "seconds_plain": round(t1 - t0, 2), "seconds_dedup": round(t2 - t1, 2)})
                print(f"{mod} {f.name}: {st.n_duplicates} dup, raw same={raw_same}, tracks same={comps[-1]['tracks_identical']}", flush=True)
    for name, data in (("volumes.csv", rows), ("comparison.csv", comps)):
        if data:
            keys = list(dict.fromkeys(k for r in data for k in r))
            with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, keys)
                w.writeheader()
                w.writerows(data)
    summ = {}
    for mod in ("ct", "mri"):
        r = [x for x in rows if x["modality"] == mod and not x["error"]]
        c = [x for x in comps if x["modality"] == mod]
        summ[mod] = {"volumes": len(r), "slices": sum(x["n_slices"] for x in r),
                     "volumes_with_duplicates": sum(1 for x in r if x["n_duplicates"]),
                     "duplicate_slices": sum(x["n_duplicates"] for x in r),
                     "nonblank_duplicate_slices": sum(x["nonblank_dup_slices"] for x in r),
                     "read_errors": sum(1 for x in rows if x["modality"] == mod and x["error"]),
                     "compared": len(c), "raw_identical": sum(x["raw_identical"] for x in c),
                     "tracks_identical": sum(x["tracks_identical"] for x in c),
                     "seconds_plain": round(sum(x["seconds_plain"] for x in c), 1),
                     "seconds_dedup": round(sum(x["seconds_dedup"] for x in c), 1)}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    return report(summ, mechanism(paths))


def report(summ: dict, mech: list[dict]) -> int:
    if mech:
        keys = list(mech[0])
        with open(OUT / "mechanism.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            w.writerows(mech)
    lines = ["# Lát cắt trùng hoàn toàn trong volume Validation (tiếp nhận 3D của website)", "",
             "Tạo bởi `scripts/86_slice_duplicates.py`; số liệu: `outputs/slice_duplicates/` (summary.json, volumes.csv, comparison.csv).",
             "Chỉ dùng Validation (không dùng Test). Lát cắt được tạo đúng như AI nhìn thấy (`load_volume_uint8` → `slice_image`).", "",
             "**Định nghĩa:** hai lát là trùng khi giống nhau **từng byte** (hash BLAKE2b). Các lát liên tiếp chỉ *giống nhau* "
             "**không** bị gộp (tổn thương trải trên nhiều lát; chưa có ngưỡng tương đồng nào được kiểm chứng).", "",
             "| | CT | MRI |", "|---|---|---|"]
    lab = [("volumes", "Số volume"), ("slices", "Số lát"), ("volumes_with_duplicates", "Volume có lát trùng"),
           ("duplicate_slices", "Lát trùng (bản sao thừa)"), ("nonblank_duplicate_slices", "… trong đó không phải lát đồng màu"),
           ("read_errors", "Volume không đọc được"), ("compared", "Volume đã so sánh có/không lọc trùng"),
           ("raw_identical", "… phát hiện YOLO từng lát trùng khớp"), ("tracks_identical", "… vùng 3D (track, box Zoom) trùng khớp"),
           ("seconds_plain", "Thời gian YOLO không lọc (s, các volume so sánh)"), ("seconds_dedup", "Thời gian YOLO có lọc (s)")]
    for k, v in lab:
        lines.append(f"| {v} | {summ['ct'][k]} | {summ['mri'][k]} |")
    lines += ["", "Cách dùng trên website: YOLO chạy một lần cho mỗi nội dung lát rồi gán lại kết quả cho mọi bản sao; "
              "volume không bị cắt bớt lát nào, chỉ số lát giữ nguyên. Volume không có lát trùng chạy y hệt pipeline đã khóa.", "",
              "## Kiểm tra cơ chế gán lại kết quả", "",
              "Vì Validation không có lát trùng, cơ chế được kiểm tra trên 3 volume CT và 3 volume MRI Validation (volume đầu tiên có "
              "vùng phát hiện), sau khi **chèn thêm bản sao** của 6 lát quanh lát có độ tin cậy cao nhất. Đây là dữ liệu kiểm thử dựng "
              "lại, không phải ca chụp thật và không dùng làm kết quả nghiên cứu.", "",
              "| Loại | Volume | Số lát | Bản sao chèn | Lát trùng tìm thấy | YOLO chạy | Box từng lát trùng tuyệt đối | Lệch box tối đa (px) | Lệch độ tin cậy tối đa | Lát khác số box | Vùng 3D trùng khớp |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in mech:
        lines.append(f"| {m['modality'].upper()} | {m['file']} | {m['n_slices']} | {m['inserted_copies']} | {m['duplicates_found']} | "
                     f"{m['yolo_slices']} | {'có' if m['raw_identical'] else 'không'} | {m['max_box_diff_px']} | {m['max_conf_diff']} | "
                     f"{m['slices_box_count_differs']} | {'có' if m['tracks_identical'] else 'KHÔNG'} |")
    lines += ["", "**Diễn giải:** bỏ lát trùng làm thay đổi cách ghép batch khi chạy YOLO fp16 trên GPU, nên tọa độ box có thể lệch "
              "ở mức làm tròn fp16 (cột *Lệch box tối đa*); đây cũng là mức lệch xảy ra khi đổi kích thước batch, không phải lỗi. "
              "Số box và các vùng 3D (track, lát tốt nhất, box Tumor Zoom) phải trùng khớp. Với dữ liệu thật của Validation, "
              "không có lát trùng nên website chạy đúng y hệt pipeline đã khóa. Tắt tính năng: `LCAI_DEDUP_EXACT=0`.", ""]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
