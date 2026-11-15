"""GĐ2a: package the QA-passed YOLO pilot datasets for Google Colab (no upload — the user uploads manually).

Requires reports from 30_qa_pilot.py with all checks passing. Produces in data/processed/pilot/:
  yolo_ct_pilot.zip, yolo_mri_pilot.zip   (root folder yolo_ct/ | yolo_mri/, PNGs stored without recompression)
  split_manifest.csv                       (copy of the frozen data/splits/split_manifest.csv)
  SHA256SUMS.txt
  pilot_report.md                          (also copied to reports/pilot_report.md)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import load_paths  # noqa: E402

LICENSES = {
    "ct": ("PLC-CECT (Luo et al., Sci Data 12:768, 2025; doi 10.1038/s41597-025-05125-2; "
           "Science Data Bank doi 10.57760/sciencedb.12207)", "CC BY 4.0"),
    "mri": ("LLD-MMRI (Lou et al., Neural Networks 2025) with MedSAM2 masks (wanglab/LLD-MMRI-MedSAM2)",
            "CC BY-NC 4.0 — non-commercial research only; do not further copy, publish or distribute"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def readme(mod: str, qa: dict, build: dict, set_name: str = "pilot") -> str:
    src, lic = LICENSES[mod]
    return f"""# YOLO {set_name} dataset — {mod.upper()} (Train + Val only)

Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/20_build_yolo_pilot.py` (QA: `scripts/30_qa_pilot.py`, all checks passed).

- Source: {src}
- License: **{lic}**. Keep this folder in a private Drive location; do not share.
- Task: single-class detection, class `0 = lesion`. Patient-level diagnosis is NOT a YOLO class.
- Images: 2D slices, grayscale uint8 PNG, native resolution, displayed as `volume[:, :, k].T` (rows = j, cols = i).
  Intensities as stored at the source (CT pre-windowed [-200, 200] HU; MRI per-volume percentile-normalized).
- Labels: one box per 8-connected lesion-mask component (fragments < {build['min_area']} px dropped).
  Negative slices have an empty label file and mean only "no lesion annotated on this slice" — never a healthy patient.
- Sampling: lesion slices every {build['pos_stride']['train']}th in train, every {build['pos_stride']['val']}th in val; negatives ratio {build['neg_ratio']} per split,
  at least {build['neg_gap']} slices away from any annotated slice.
- Splits are patient-level (frozen `split_manifest.csv`). **The Test split is not in this archive** and must not be used
  for model selection.
- Traceability: `slices.csv` (patient_id, series/phase, slice_k, kind, boxes), `volumes.csv` (per-volume status,
  exclusions), `qa_report.json`, `build_args.json`.
- Counts: {qa['n_images']} images, {qa['n_box_lines']} boxes; slices {qa['slices']}; patients {qa['patients']}.
"""


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Vietnamese text on Windows consoles / redirected logs
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=["pilot", "full"], default="pilot",
                    help="dataset set under data/processed/<set>/yolo_<mod>; zips are named yolo_<mod>_<set>.zip")
    ap.add_argument("--modalities", nargs="+", choices=["ct", "mri"], default=["ct", "mri"])
    ap.add_argument("--max-gb", type=float, default=None, help="size limit (default 5 GB for pilot, none for full)")
    args = ap.parse_args()

    paths = load_paths()
    SET, MODS = args.set, args.modalities
    max_gb = args.max_gb if args.max_gb is not None else (5.0 if SET == "pilot" else float("inf"))
    pilot = paths["project"]["processed"] / SET
    qas, builds, vols = {}, {}, {}
    for mod in MODS:
        d = pilot / f"yolo_{mod}"
        qa = json.loads((d / "qa_report.json").read_text(encoding="utf-8"))
        failed = {c: v["failures"] for c, v in qa["checks"].items() if v["failures"]}
        if failed:
            print(f"{mod}: QA failures {failed}; refusing to package.")
            return 1
        qas[mod], builds[mod] = qa, json.loads((d / "build_args.json").read_text(encoding="utf-8"))
        with open(d / "volumes.csv", encoding="utf-8") as f:
            vols[mod] = list(csv.DictReader(f))
    total = sum(q["size_bytes"] for q in qas.values())
    if total > max_gb * 1e9:
        print(f"{SET} size {total / 1e9:.2f} GB exceeds {max_gb} GB; reduce sampling.")
        return 1

    shutil.copy2(paths["project"]["splits"] / "split_manifest.csv", pilot / "split_manifest.csv")
    zips = {}
    for mod in MODS:
        d = pilot / f"yolo_{mod}"
        (d / "README_DATA.md").write_text(readme(mod, qas[mod], builds[mod], SET), encoding="utf-8")
        shutil.copy2(pilot / "split_manifest.csv", d / "split_manifest.csv")
        zp = pilot / f"yolo_{mod}_{SET}.zip"
        files = sorted(p for p in d.rglob("*") if p.is_file())
        with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for p in files:
                comp = zipfile.ZIP_STORED if p.suffix == ".png" else zipfile.ZIP_DEFLATED
                z.write(p, arcname=f"yolo_{mod}/{p.relative_to(d).as_posix()}", compress_type=comp)
        with zipfile.ZipFile(zp) as z:
            bad = z.testzip()
            names = z.namelist()
        if bad or len(names) != len(files):
            print(f"{zp.name}: integrity check failed (bad={bad}, entries={len(names)} vs files={len(files)})")
            return 1
        zips[mod] = {"path": zp, "entries": len(names), "bytes": zp.stat().st_size}
        print(f"{zp.name}: {len(names)} entries, {zp.stat().st_size / 1e9:.2f} GB, testzip OK", flush=True)

    sums = {p.name: sha256(p) for p in [z["path"] for z in zips.values()] + [pilot / "split_manifest.csv"]}
    sha_name = "SHA256SUMS.txt" if SET == "pilot" else f"SHA256SUMS_{SET}.txt"  # notebooks read SHA256SUMS*.txt
    (pilot / sha_name).write_text("".join(f"{h}  {n}\n" for n, h in sums.items()), encoding="utf-8")

    # ---------------------------------------------------------------- pilot_report.md
    with open(paths["project"]["splits"] / "split_manifest.csv", encoding="utf-8") as f:
        man = list(csv.DictReader(f))
    L = [f"# {'Pilot' if SET == 'pilot' else 'Full'} dataset report — {'GĐ2a' if SET == 'pilot' else 'GĐ3 (train chính thức)'}\n",
         f"*Sinh tự động bởi `scripts/40_pack_pilot.py --set {SET}` lúc {datetime.now():%Y-%m-%d %H:%M}.* "
         "Bộ dữ liệu chỉ gồm **Train + Val**. **Test được giữ riêng**, không nằm trong ZIP.\n",
         "## 1. Bệnh nhân theo split (toàn bộ dataset, split đóng băng)\n",
         "| Modality | Train | Val | Test | Tổng |", "|---|---|---|---|---|"]
    for mod in ("CT", "MRI"):
        c = Counter(r["split"] for r in man if r["modality"] == mod)
        L.append(f"| {mod} | {c['train']} | {c['val']} | {c['test']} | {sum(c.values())} |")
    L.append("\nChi tiết theo lớp: `reports/split_report.md`. Kiểm tra rò rỉ: `tests/test_splits_no_leakage.py` "
             "và QA check C4 (không BN nào thuộc hai split, không có BN test trong pilot).\n")
    for mod in MODS:
        q, b, v = qas[mod], builds[mod], vols[mod]
        st = Counter(x["status"] for x in v)
        L += [f"## 2{'a' if mod == 'ct' else 'b'}. {mod.upper()} — `yolo_{mod}_{SET}.zip`\n",
              f"- Ảnh: **{q['n_images']}**, file nhãn: **{q['n_label_files']}**, box: **{q['n_box_lines']}**; "
              f"lát cắt theo split và loại: {q['slices']}",
              f"- BN có trong dataset: {q['patients']}",
              f"- Lấy mẫu: lấy 1/{b['pos_stride']['train']} số lát dương tính ở train, 1/{b['pos_stride']['val']} ở val; "
              f"tỉ lệ âm:dương = {b['neg_ratio']}; lát âm cách lát có annotation ≥ {b['neg_gap']} lát; "
              f"mảnh mask < {b['min_area']} px bị bỏ ({q['fragments_dropped']} mảnh)",
              f"- Trạng thái volume (train+val): {dict(st)}",
              "- Lát bị bỏ (không là dương tính, cũng không là âm tính): "
              + ", ".join(f"{k} = {sum(int(x.get(k) or 0) for x in v)}" for k in
                          ("skipped_all_fragments", "skipped_no_json_box", "skipped_low_json_iou") if k in v[0]),
              f"- Volume RGB 5D có kênh khác nhau chỉ ở chữ thước đo in chồng (ngoài gan/tổn thương, dùng kênh 0): "
              f"{sum(1 for x in v if x.get('channel_diff_frac') not in (None, '', '0', '0.0'))}"
              if mod == "ct" else "",
              f"- Dung lượng ảnh+nhãn: {q['size_bytes'] / 1e9:.2f} GB; ZIP: **{zips[mod]['bytes'] / 1e9:.2f} GB** "
              f"({zips[mod]['entries']} entries)",
              f"- QA ({'–'.join(sorted(q['checks'])[::len(q['checks']) - 1])}): "
              + ", ".join(f"{c} {x['failures']} lỗi" for c, x in q["checks"].items())
              + f"; đối chiếu NIfTI gốc trên {q['roundtrip_volumes']} volume / {q['roundtrip_slices']} lát"
              + (f"; C3 đọc {q['images_checked_C3']} ảnh" if "images_checked_C3" in q else ""), ""]
        if q.get("patient_coverage"):
            L += ["Độ phủ bệnh nhân (QA C7: không BN Train/Val nào có volume dùng được bị mất do lấy mẫu lát):\n",
                  "| Split | Lớp | BN trong split | BN có volume dùng được | BN có trong dataset |", "|---|---|---|---|---|",
                  *[f"| {c['split']} | {c['label']} | {c['patients_in_split']} | {c['patients_with_usable_volumes']} | "
                    f"{c['patients_exported']} |" for c in q["patient_coverage"]], ""]
        if mod == "mri":
            m = q["mri_json_iou_on_positive_slices"]
            L.append(f"- Mỗi lát MRI dương tính đều phải có JSON 2D_box gốc đồng thuận với box từ mask "
                     f"(IoU ≥ {b.get('mri_min_json_iou')}). Lát không đồng thuận bị bỏ (xem dòng trên), không sửa nhãn. "
                     f"Trên các lát được giữ: IoU p5/p25/p50 = {m['quantiles_p5_p25_p50']} ({m['n']} lát).\n")
        L += ["Phân bố (lớp là chẩn đoán mức bệnh nhân, chỉ dùng để truy vết; YOLO chỉ có 1 lớp `lesion`):\n",
              *q["distribution_table"], ""]
    L += ["## 3. Dữ liệu lỗi / bị cách ly (không vào training)\n",
          "| Modality | Loại | Số volume/series (train+val) |", "|---|---|---|"]
    for mod in MODS:
        for s, n in sorted(Counter(x["status"] for x in vols[mod] if x["status"] != "ok").items()):
            L.append(f"| {mod.upper()} | {s} | {n} |")
    integ = paths["project"]["reports"] / "audit" / "source_integrity_test.csv"
    if integ.exists():
        with open(integ, encoding="utf-8") as f:
            ir = list(csv.DictReader(f))
        bad = [r for r in ir if r["status"] != "ok"]
        L += ["", f"**Tập Test (không đóng gói), kiểm tra gzip toàn vẹn** (`scripts/21_check_source_integrity.py`): "
              f"{len(ir)} file; " + ", ".join(f"{m} {s}: {n}" for (m, s), n in sorted(Counter(
                  (r['modality'], r['status'].split(':')[0]) for r in ir).items()))]
        L += [f"- {r['modality']} {r['patient_id']} {r['series']} {r['kind']}: {r['status'][:60]}" for r in bad
              if not r["status"].startswith("missing")]
    L += ["", "Danh sách đầy đủ: `volumes.csv` trong mỗi ZIP; MRI cách ly: `data/quarantine/mri_bbox_quarantine.csv`.\n",
          "## 4. File đầu ra\n", "| File | Dung lượng | SHA256 |", "|---|---|---|"]
    for n, h in sums.items():
        L.append(f"| `data/processed/{SET}/{n}` | {(pilot / n).stat().st_size / 1e9:.3f} GB | `{h}` |")
    nb = "03_train_yolo_{m}_colab.ipynb" if SET == "pilot" else "04_train_yolo_{m}_full_colab.ipynb"
    L += ["", "## 5. Cách dùng\n",
          "1. Upload " + ", ".join(f"`yolo_{m}_{SET}.zip`" for m in MODS) + f", `{sha_name}` vào `MyDrive/LiverCancer_AI/data/` "
          "(thư mục riêng tư).",
          "2. Mở " + " / ".join(f"`notebooks/{nb.format(m=m)}`" for m in MODS) + " trên Colab (Runtime → GPU)"
          + ("" if SET == "pilot" else f" (kiểm tra `ZIP_NAME` ở mục 2: CT mặc định `yolo_ct_full.zip`)") + ", rồi chạy từ trên xuống.",
          "3. Kết quả (best.pt, last.pt, biểu đồ, metrics JSON) lưu ở `MyDrive/LiverCancer_AI/runs/`.", ""]
    text = "\n".join(L) + "\n"
    report = paths["project"]["reports"] / ("pilot_report.md" if SET == "pilot" else f"{SET}_dataset_report.md")
    (pilot / report.name).write_text(text, encoding="utf-8")
    report.write_text(text, encoding="utf-8")
    print(f"total ZIP {sum(z['bytes'] for z in zips.values()) / 1e9:.2f} GB; report -> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
