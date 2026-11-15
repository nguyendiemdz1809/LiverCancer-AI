"""Roboflow illustration set with lesion slices only: up to 10,000 CT + 10,000 MRI from the TRAIN split.

Not a new training set. Images and YOLO labels are copied byte-for-byte from the GĐ2 YOLO exports
(CT: data/processed/full/yolo_ct, MRI: data/processed/pilot/yolo_mri); every selected image has >= 1 `lesion` box.
Nothing is relabelled, augmented or duplicated. If a modality has fewer valid lesion slices than the quota,
the real count is exported and reported.

Selection (per modality), in priority order; each image records the pass that picked it:
  pass 1  strict diversity  : >= gap1 slices apart in a volume, no near-duplicate of the same patient (dHash), <= cap1 / patient
  pass 2  relaxed           : smaller gap, stricter near-duplicate threshold only, <= cap2 / patient
  pass 3  fill              : any remaining distinct lesion slice (still no exact duplicate)
Inside every pass: round robin over patients, then volumes; slices with a clear lesion (largest box short side
>= --clear-min-side px) before smaller ones; larger lesions first inside a volume.

QA: Q1-Q6, Q9 (from scripts/88_), Q11 every label has >= 1 box, Q12 round trip on a sample of volumes:
PNG pixels and boxes recomputed from the source NIfTI image + lesion mask must match exactly.

Run: .venv\\Scripts\\python scripts\\89_build_roboflow_positive.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import load_paths  # noqa: E402
from livercancer.data.nifti_io import as_3d, to_display  # noqa: E402
from livercancer.data.yolo_boxes import component_boxes, to_yolo  # noqa: E402

_spec = importlib.util.spec_from_file_location("rf88", ROOT / "scripts/88_build_roboflow_export.py")
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)

REJ = {"near_duplicate_pass1": "gần trùng ảnh khác của cùng BN (mức 1)", "near_duplicate_pass2": "gần như y hệt (mức 2)",
       "near_blank": "gần như trống", "unreadable": "không đọc được"}
ZIP_FOLDER = {"CT": "CT", "MRI": "MRI_offline"}
PASS_VI = {1: "Mức 1 – đa dạng chặt", 2: "Mức 2 – nới vừa", 3: "Mức 3 – bổ sung (lát khác nhau)"}


class CachedPicker(rf.Picker):
    """Same rules as scripts/88_ Picker, but caches dHash/std so later passes do not reread images."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.cache: dict[str, tuple[float, np.ndarray] | None] = {}

    def _ok(self, row) -> bool:
        if any(abs(row.slice_k - k) < self.gap for k in self.taken_k[row.volume]):
            return False
        if row.file not in self.cache:
            img = cv2.imread(str(self.src / "images/train" / f"{row.file}.png"), cv2.IMREAD_GRAYSCALE)
            self.cache[row.file] = None if img is None else (float(img.std()), rf.dhash(img))
        c = self.cache[row.file]
        if c is None:
            self.rejected["unreadable"] += 1
            return False
        std, h = c
        if std < self.min_std:
            self.rejected["near_blank"] += 1
            return False
        if self.dup >= 0 and rf.min_hamming(h, self.hashes[row.patient_id]) <= self.dup:
            self.rejected[f"near_duplicate_pass{self.pass_no}"] += 1
            return False
        self.taken_k[row.volume].append(int(row.slice_k))
        self.hashes[row.patient_id].append(h)
        self.per_patient[row.patient_id] += 1
        return True


def build(mod: str, args, out: Path) -> dict:
    t0 = time.time()
    pool = rf.load_pool(mod, args.clear_min_side)
    pos = pool[pool.kind == "pos"]
    # a "pos" slice in the source always has >= 1 box; checked again here and in QA
    assert (pos.n_boxes >= 1).all()
    gaps = {"CT": (args.gap1_ct, 2), "MRI": (args.gap1_mri, 1)}[mod]
    passes = [(1, gaps[0], args.dup1, args.cap1), (2, gaps[1], args.dup2, args.cap2), (3, 1, -1, 10**9)]
    pk = CachedPicker(mod, passes[0][1], args.dup1, args.min_std, args.seed, args.cap1)
    pass_of: dict[int, int] = {}
    for no, gap, dup, cap in passes:
        pk.gap, pk.dup, pk.cap, pk.pass_no = gap, dup, cap, no
        for subset in (pos[pos.clear], pos[~pos.clear]):
            need = args.per_modality - len(pk.chosen)
            if need <= 0:
                break
            for i in pk.pick(subset, need, "clear"):
                pass_of[i] = no
    sel = pool.loc[pk.chosen].copy()
    sel["pass"] = [pass_of[i] for i in pk.chosen]
    sel["tier"] = np.where(sel.clear, "pos_clear", "pos_small")
    assert sel.file.is_unique and (sel.kind == "pos").all()

    pmap = {p: f"BN{i + 1:03d}" for i, p in enumerate(sorted(sel.patient_id.unique()))}
    sel["patient_code"] = sel.patient_id.map(pmap)
    sel["series_code"] = sel.series.map(lambda x: rf.SERIES_CODE.get(x, x))
    sel["out_name"] = mod + "_" + sel.patient_code + "_" + sel.series_code + "_k" + sel.slice_k.map("{:03d}".format)
    assert sel.out_name.is_unique and sel.out_name.str.fullmatch(r"[A-Za-z0-9_]+").all()
    sel = sel.sort_values("out_name").reset_index(drop=True)
    sel["part"] = sel.index // args.part_size + 1
    n_parts = int(sel.part.max())

    ds = out / f"Roboflow_{mod}_Positive_{args.per_modality}"
    if ds.exists():
        shutil.rmtree(ds)
    (ds / "train/images").mkdir(parents=True)
    (ds / "train/labels").mkdir(parents=True)
    src = rf.SOURCES[mod]
    ish, lsh = [], []
    for f, o in zip(sel.file, sel.out_name):
        di, dl = ds / "train/images" / f"{o}.png", ds / "train/labels" / f"{o}.txt"
        shutil.copyfile(src / "images/train" / f"{f}.png", di)
        shutil.copyfile(src / "labels/train" / f"{f}.txt", dl)
        ish.append(rf.sha256(di))
        lsh.append(rf.sha256(dl))
    sel["image_sha256"], sel["label_sha256"] = ish, lsh
    sel["source_image"] = [str((src / "images/train" / f"{f}.png").relative_to(ROOT)) for f in sel.file]
    (ds / "data.yaml").write_text(
        f"# LiverCancer AI - Roboflow_{mod}_Positive_{args.per_modality} (YOLO object detection, illustration only)\n"
        f"# {len(sel)} 2D slices from the project's TRAIN split, each with >= 1 lesion box. Not a new training set.\n"
        "# Labels were converted automatically from the source dataset's lesion masks (not drawn by hand).\n"
        "# Class 0 = lesion (any focal liver lesion).\n"
        f"# Source / license: {rf.LICENSE[mod]}\n"
        "path: .\ntrain: train/images\nnc: 1\nnames: ['lesion']\n", encoding="ascii")
    return {"mod": mod, "sel": sel, "ds": ds, "pool": pool, "n_parts": n_parts, "rejected": dict(pk.rejected),
            "available": len(pos), "seconds": round(time.time() - t0, 1)}


def roundtrip(r: dict, n_vol: int, seed: int) -> dict:
    """Recompute PNG pixels and YOLO boxes from the source NIfTI for every selected slice of a sample of volumes."""
    paths = load_paths()
    build_args = json.loads((rf.SOURCES[r["mod"]] / "build_args.json").read_text(encoding="utf-8"))
    sel, ds = r["sel"], r["ds"]
    vols = sorted(sel.volume.unique())
    sample = random.Random(seed).sample(vols, min(n_vol, len(vols)))
    info = json.loads(Path(paths["mri"]["annotation"]).read_text(encoding="utf-8"))["Annotation_info"] \
        if r["mod"] == "MRI" else None
    fails, n = [], 0
    for vol in sample:
        pid, series = vol.split("|")
        if r["mod"] == "CT":
            d = Path(paths["ct"]["data_dir"])
            img_p, mask_p = d / "ct_files" / f"{pid}_ct_{series}.nii.gz", d / "mask_files" / f"{pid}_mask_{series}.nii.gz"
        else:
            s = next(x for x in info[pid] if x["phase"] == series)
            c = s["annotation"]["lesion"]["0"]["category"]
            f = rf.SERIES_CODE.get(series, series) if series in ("In Phase", "Out Phase") else series
            img_p = Path(paths["mri"]["images"]) / f"{pid}_{c}_{f}_0000.nii.gz"
            mask_p = Path(paths["mri"]["labels"]) / f"{pid}_{c}_{f}.nii.gz"
        img, _ = as_3d(np.asanyarray(nib.load(str(img_p)).dataobj), max_diff_frac=build_args["max_channel_diff"])
        mask = np.asanyarray(nib.load(str(mask_p)).dataobj.get_unscaled()) > 0
        for row in sel[sel.volume == vol].itertuples():
            k = int(row.slice_k)
            png = cv2.imread(str(ds / "train/images" / f"{row.out_name}.png"), cv2.IMREAD_UNCHANGED)
            if not np.array_equal(png, to_display(img[:, :, k])):
                fails.append(f"{row.out_name}: pixels differ from NIfTI")
            exp = sorted(tuple(round(x, 6) for x in to_yolo(b, img.shape[0], img.shape[1]))
                         for b in component_boxes(mask[:, :, k], build_args["min_area"])[0])
            got = sorted(tuple(round(x, 6) for x in b[1:]) for b in rf.read_boxes(ds / "train/labels" / f"{row.out_name}.txt"))
            if not exp or len(exp) != len(got) or not np.allclose(exp, got, atol=2e-6):
                fails.append(f"{row.out_name}: boxes differ from mask")
            n += 1
    return {"volumes": len(sample), "slices": n, "failures": len(fails), "examples": fails[:5]}


POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def near_dup_pairs(ds: Path, names: list[str], thr: int) -> int:
    """Pairs of exported images whose 256-bit dHash differ by <= thr bits (information only)."""
    hs = np.stack([rf.dhash(cv2.imread(str(ds / "train/images" / f"{x}.png"), cv2.IMREAD_GRAYSCALE)) for x in names])
    cnt = 0
    for i in range(len(hs) - 1):
        cnt += int((POPCOUNT[np.bitwise_xor(hs[i + 1:], hs[i])].sum(1) <= thr).sum())
    return cnt


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-modality", type=int, default=10000)
    ap.add_argument("--clear-min-side", type=float, default=16.0)
    ap.add_argument("--gap1-ct", type=int, default=4)
    ap.add_argument("--gap1-mri", type=int, default=3)
    ap.add_argument("--dup1", type=int, default=16, help="pass 1 near-duplicate threshold (of 256 bits)")
    ap.add_argument("--dup2", type=int, default=6, help="pass 2: only almost identical images rejected")
    ap.add_argument("--cap1", type=int, default=40)
    ap.add_argument("--cap2", type=int, default=80)
    ap.add_argument("--min-std", type=float, default=8.0)
    ap.add_argument("--part-size", type=int, default=1000)
    ap.add_argument("--roundtrip-volumes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data/processed/roboflow_positive_20k"))
    ap.add_argument("--report", default=str(ROOT / "reports/roboflow_positive_20k_report.md"))
    args = ap.parse_args()
    out = Path(args.out)
    zdir, pdir = out / "zips", out / "previews"
    for d in (zdir, pdir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    summary = {"args": vars(args), "modalities": {}}
    manifests, uploads = [], []
    for mod in ("CT", "MRI"):
        print(f"[{mod}] selecting + copying ...", flush=True)
        r = build(mod, args, out)
        print(f"[{mod}] QA ...", flush=True)
        q = rf.qa(r)
        sel, ds = r["sel"], r["ds"]
        empty = sum(1 for o in sel.out_name if not rf.read_boxes(ds / "train/labels" / f"{o}.txt"))
        rt = roundtrip(r, args.roundtrip_volumes, args.seed)
        print(f"[{mod}] round trip {rt}", flush=True)
        r["near_dup_global"] = near_dup_pairs(ds, list(sel.out_name), 6)
        print(f"[{mod}] zips + previews ...", flush=True)
        zsub = zdir / ZIP_FOLDER[mod]  # CT and MRI zips kept apart so MRI is not uploaded by mistake
        zsub.mkdir(parents=True, exist_ok=True)
        rows = rf.make_zips(r, zsub)
        vz = rf.verify_zips(rows, zsub, r)
        for x in rows:
            x["folder"] = f"zips/{ZIP_FOLDER[mod]}"
        rf.previews(r, pdir, 50, 0, args.seed)
        uploads += rows
        m = sel.assign(modality=mod, zip=[f"{ds.name}_part{p:02d}of{r['n_parts']:02d}.zip" for p in sel.part])
        manifests.append(m)
        st = rf.stats(r)
        summary["modalities"][mod] = {
            "images": len(sel), "quota": args.per_modality, "available_lesion_slices": r["available"],
            "shortage": max(0, args.per_modality - len(sel)), "boxes": int(sel.n_boxes.sum()),
            "patients": sel.patient_id.nunique(), "volumes": sel.volume.nunique(),
            "train_pool": st["train_pool"], "empty_label_files": empty,
            "by_pass": {int(k): int(v) for k, v in sel["pass"].value_counts().sort_index().items()},
            "clear_vs_small": st["tiers"], "boxes_per_image": st["boxes_per_image"],
            "box_short_side_px": st["box_short_side"], "per_patient_images": st["per_patient_images"],
            "per_volume_images": st["per_volume_images"], "rejected_during_selection": r["rejected"],
            "qa": q, "roundtrip": rt, "near_dup_pairs_global_le6": r["near_dup_global"], "zip_check": vz,
            "dataset_bytes": dir_size(ds), "zip_bytes": int(sum(x["size_bytes"] for x in rows)),
            "by_label": st["by_label"].to_dict("records"), "by_series": st["by_series"].to_dict("records"),
            "license": rf.LICENSE[mod], "seconds": r["seconds"]}
        st["per_patient"].to_csv(out / f"patient_stats_{mod}.csv", index=False)
        print(f"[{mod}] {len(sel)} images, QA {q['failures'] or 'OK'}, empty labels {empty}, zips {vz['failures'] or 'OK'}",
              flush=True)

    allm = pd.concat(manifests, ignore_index=True)
    allm.to_csv(out / "manifest.csv", index=False, columns=[
        "modality", "out_name", "zip", "part", "patient_code", "patient_id", "patient_label", "series", "slice_k",
        "n_slices", "n_boxes", "box_short", "pass", "tier", "width", "height", "source_image", "image_sha256", "label_sha256"])
    pd.DataFrame(uploads).to_csv(out / "upload_manifest.csv", index=False)
    ct, mr = manifests
    summary["cross"] = {
        "shared_image_sha256": len(set(ct.image_sha256) & set(mr.image_sha256)),
        "shared_patient_ids": len(set(ct.patient_id) & set(mr.patient_id)),
        "non_train_patients": int(sum((pd.read_csv(rf.SPLITS[m]).set_index("patient_id").loc[
            g.patient_id.unique(), "split"] != "train").sum() for m, g in (("CT", ct), ("MRI", mr)))),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_report(summary, pd.DataFrame(uploads), out, Path(args.report))
    print(json.dumps(summary["cross"]), flush=True)
    print("done", flush=True)


def write_report(s: dict, up: pd.DataFrame, out: Path, path: Path) -> None:
    a, n, kv = s["args"], rf._n, rf._kv
    c, m = s["modalities"]["CT"], s["modalities"]["MRI"]
    def rel(p) -> str:
        p = Path(p).resolve()
        return str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p).replace("\\", "/")
    ok = lambda d, k: "PASS" if d["qa"]["failures"].get(k, 0) == 0 else f"FAIL ({d['qa']['failures'][k]})"
    yn = lambda b: "PASS" if b else "FAIL"
    L = ["# Bộ minh họa Roboflow chỉ gồm ảnh có tổn thương: CT + MRI", "",
         f"*Sinh tự động bởi `scripts/89_build_roboflow_positive.py` lúc {time.strftime('%Y-%m-%d %H:%M')} (seed {a['seed']}). "
         f"Số liệu: `{rel(out)}/summary.json`, `manifest.csv`.*", "",
         "> **Đây chỉ là bộ minh họa để xem nhãn trên Roboflow, không phải tập Train mới.** Không train lại, không đổi YOLO, Hybrid, "
         "checkpoint, Evidence Gate hay split Train/Val/Test. Nhãn **chuyển đổi tự động** từ mask của dataset gốc, không gán tay.", "",
         "## 1. Tổng quan", "", "| | CT | MRI | Tổng |", "|---|---|---|---|"]
    for key, lab in (("images", "Ảnh (đều có ≥ 1 box)"), ("boxes", "Bounding box `lesion`"), ("patients", "Bệnh nhân"),
                     ("volumes", "Volume (pha CT / chuỗi MRI)")):
        L.append(f"| {lab} | {n(c[key])} | {n(m[key])} | {n(c[key] + m[key])} |")
    L += [f"| Chỉ tiêu | {n(c['quota'])} | {n(m['quota'])} | {n(c['quota'] + m['quota'])} |",
          f"| Thiếu so với chỉ tiêu | {n(c['shortage'])} | {n(m['shortage'])} | {n(c['shortage'] + m['shortage'])} |",
          f"| Lát có tổn thương sẵn có trong Train (nguồn) | {n(c['available_lesion_slices'])} | {n(m['available_lesion_slices'])} | |",
          f"| Dung lượng dataset (ảnh + nhãn) | {rf._gb(c['dataset_bytes'])} | {rf._gb(m['dataset_bytes'])} | {rf._gb(c['dataset_bytes'] + m['dataset_bytes'])} |",
          f"| Dung lượng ZIP | {rf._gb(c['zip_bytes'])} | {rf._gb(m['zip_bytes'])} | {rf._gb(c['zip_bytes'] + m['zip_bytes'])} |", ""]
    for mod, d in (("CT", c), ("MRI", m)):
        if d["shortage"]:
            L.append(f"- **{mod} thiếu {n(d['shortage'])} ảnh**: chỉ có {n(d['images'])} lát có tổn thương hợp lệ trong Train. "
                     "Không nhân bản ảnh, không lấy Val/Test để bù.")
    L += ["", "## 2. Nhãn và nguồn dữ liệu", "",
          "- Danh sách nhãn YOLO: **một lớp duy nhất `0 = lesion`** (tổn thương gan khu trú). File `data.yaml`: `nc: 1`, `names: ['lesion']`.",
          "- Chẩn đoán mức bệnh nhân (HCC, ICC, ...) **không** phải lớp YOLO; chỉ ghi trong `manifest.csv` (local) để thống kê.",
          f"- CT: {c['license']}. Ảnh lấy từ `data/processed/full/yolo_ct` (Train).",
          f"- MRI: {m['license']}. Ảnh lấy từ `data/processed/pilot/yolo_mri` (Train; đã loại series bị cách ly do bbox JSON lệch mask).",
          "- Quy trình gốc: NIfTI 3D → lát 2D PNG xám độ phân giải gốc (`slice.T`, cường độ uint8 như nguồn) → mỗi thành phần liên thông 8 hướng "
          "của mask tổn thương (≥ 10 px) thành 1 box YOLO chuẩn hóa. MRI chỉ giữ lát mà mask và `2D_box` JSON gốc đồng thuận (IoU ≥ 0.3). "
          "Chi tiết: `reports/roboflow_dataset_report.md` mục 3–4.", "",
          "| Chẩn đoán (mức BN) | CT: BN | CT: ảnh | CT: box | | Chẩn đoán (mức BN) | MRI: BN | MRI: ảnh | MRI: box |",
          "|---|---|---|---|---|---|---|---|---|"]
    lc, lm = c["by_label"], m["by_label"]
    for i in range(max(len(lc), len(lm))):
        x = lc[i] if i < len(lc) else None
        y = lm[i] if i < len(lm) else None
        L.append((f"| {x['patient_label']} | {x['patients']} | {n(x['images'])} | {n(x['boxes'])} |" if x else "| | | | |")
                 + (f" | {y['patient_label']} | {y['patients']} | {n(y['images'])} | {n(y['boxes'])} |" if y else " | | | | |"))
    L += ["", "## 3. Cách chọn ảnh", "",
          "Chỉ lấy lát có ≥ 1 box (`kind = pos`) của BN thuộc Train đã đóng băng. Ba mức ưu tiên; mức sau chỉ dùng khi mức trước đã cạn:", "",
          "| Mức | Điều kiện | CT | MRI |", "|---|---|---|---|",
          f"| 1 – đa dạng chặt | cách ≥ {a['gap1_ct']} lát (CT) / ≥ {a['gap1_mri']} lát (MRI) trong một volume; không gần trùng ảnh khác của cùng BN "
          f"(dHash ≤ {a['dup1']}/256); ≤ {a['cap1']} ảnh/BN | {n(c['by_pass'].get('1', c['by_pass'].get(1, 0)))} | {n(m['by_pass'].get('1', m['by_pass'].get(1, 0)))} |",
          f"| 2 – nới vừa | cách ≥ 2 lát (CT) / ≥ 1 lát (MRI); chỉ loại ảnh gần như y hệt (dHash ≤ {a['dup2']}/256); ≤ {a['cap2']} ảnh/BN "
          f"| {n(c['by_pass'].get('2', c['by_pass'].get(2, 0)))} | {n(m['by_pass'].get('2', m['by_pass'].get(2, 0)))} |",
          f"| 3 – bổ sung | mọi lát có tổn thương còn lại (khác lát, không trùng tuyệt đối) "
          f"| {n(c['by_pass'].get('3', c['by_pass'].get(3, 0)))} | {n(m['by_pass'].get('3', m['by_pass'].get(3, 0)))} |", "",
          "Trong mỗi mức: lần lượt mỗi BN một ảnh mỗi vòng, xoay vòng qua các volume; lát có tổn thương rõ "
          f"(cạnh ngắn box lớn nhất ≥ {a['clear_min_side']:g} px) trước, tổn thương lớn trước trong cùng volume; bỏ ảnh gần như trống "
          f"(độ lệch chuẩn < {a['min_std']:g}). Cột `pass` trong `manifest.csv` ghi mức của từng ảnh.", "",
          "Lưu ý: CT gốc đã lấy mẫu 1/2 lát (2 mm) ở Train, nên ảnh mức 2–3 của CT là các lát cách nhau 2–4 mm của cùng tổn thương; "
          "chúng hợp lệ nhưng giống nhau hơn ảnh mức 1.", "",
          "| Phân bố | CT | MRI |", "|---|---|---|",
          f"| Tổn thương rõ / nhỏ | {n(c['clear_vs_small'].get('pos_clear', 0))} / {n(c['clear_vs_small'].get('pos_small', 0))} "
          f"| {n(m['clear_vs_small'].get('pos_clear', 0))} / {n(m['clear_vs_small'].get('pos_small', 0))} |",
          f"| Ảnh theo cạnh ngắn box lớn nhất | {kv(c['box_short_side_px'], ' px')} | {kv(m['box_short_side_px'], ' px')} |",
          f"| Ảnh theo số box trong ảnh | {kv(c['boxes_per_image'], ' box')} | {kv(m['boxes_per_image'], ' box')} |",
          f"| Ảnh mỗi BN (min / trung vị / max) | {c['per_patient_images']['min']:.0f} / {c['per_patient_images']['50%']:.0f} / {c['per_patient_images']['max']:.0f} "
          f"| {m['per_patient_images']['min']:.0f} / {m['per_patient_images']['50%']:.0f} / {m['per_patient_images']['max']:.0f} |",
          f"| Ảnh mỗi volume (min / trung vị / max) | {c['per_volume_images']['min']:.0f} / {c['per_volume_images']['50%']:.0f} / {c['per_volume_images']['max']:.0f} "
          f"| {m['per_volume_images']['min']:.0f} / {m['per_volume_images']['50%']:.0f} / {m['per_volume_images']['max']:.0f} |",
          f"| Lượt bị loại khi chọn | {kv(c['rejected_during_selection'], '', REJ) or '0'} | {kv(m['rejected_during_selection'], '', REJ) or '0'} |", "",
          f"Theo pha CT: " + ", ".join(f"{r['series']} {n(r['images'])}" for r in c["by_series"]) + ". "
          f"Theo chuỗi MRI: " + ", ".join(f"{r['series']} {n(r['images'])}" for r in m["by_series"]) + ".", "",
          "## 4. Kiểm tra sau khi xuất", "", "| Kiểm tra | CT | MRI |", "|---|---|---|",
          f"| Q11. **Mọi ảnh có ≥ 1 box `lesion`** (file nhãn rỗng: CT {c['empty_label_files']}, MRI {m['empty_label_files']}) "
          f"| {yn(c['empty_label_files'] == 0)} | {yn(m['empty_label_files'] == 0)} |"]
    for k, lab in (("Q1_open_size", "Q1. Ảnh mở được, đúng kích thước"),
                   ("Q2_pairing", "Q2. Mỗi ảnh có đúng một `.txt` cùng tên, không thừa/thiếu"),
                   ("Q3_label_format_bounds", "Q3. Đúng 5 cột, class ID = 0, box trong [0, 1], w, h > 0"),
                   ("Q4_box_count_vs_source", "Q4. Số box khớp manifest nguồn"),
                   ("Q5_image_differs_from_source", "Q5a. Ảnh trùng SHA256 với file nguồn"),
                   ("Q5_label_differs_from_source", "Q5b. Nhãn trùng SHA256 với file nguồn"),
                   ("Q6_duplicate_image_in_modality", "Q6. Không có ảnh trùng tuyệt đối"),
                   ("Q9_png_metadata", "Q9. PNG không chứa metadata văn bản")):
        L.append(f"| {lab} | {ok(c, k)} | {ok(m, k)} |")
    x = s["cross"]
    L += [f"| Q12. Đối chiếu NIfTI gốc: pixel PNG và box tính lại từ mask trùng khớp ({c['roundtrip']['volumes']} / {m['roundtrip']['volumes']} volume, "
          f"{n(c['roundtrip']['slices'])} / {n(m['roundtrip']['slices'])} ảnh) | {yn(c['roundtrip']['failures'] == 0)} | {yn(m['roundtrip']['failures'] == 0)} |",
          f"| Q7. Mọi BN thuộc Train (BN ngoài Train: {x['non_train_patients']}) | {yn(x['non_train_patients'] == 0)} | (chung) |",
          f"| Q8. Không trùng CT–MRI (SHA256 ảnh {x['shared_image_sha256']}, ID BN {x['shared_patient_ids']}) "
          f"| {yn(not x['shared_image_sha256'] and not x['shared_patient_ids'])} | (chung) |",
          f"| Q10. ZIP: CRC, đủ cặp ảnh–nhãn, các phần không trùng, gộp lại đủ bộ | {yn(not c['zip_check']['failures'])} | {yn(not m['zip_check']['failures'])} |", "",
          f"Thông tin thêm: cặp ảnh gần như y hệt (dHash ≤ 6/256) trong toàn bộ: CT {n(c['near_dup_pairs_global_le6'])}, "
          f"MRI {n(m['near_dup_pairs_global_le6'])} (lát liền nhau của cùng một tổn thương; không phải ảnh trùng).", "",
          "## 5. File đầu ra", "", f"Thư mục: `{rel(out)}/` (không đưa vào git).", "",
          "| ZIP | Ảnh | Box | BN | Dung lượng | SHA256 |", "|---|---|---|---|---|---|"]
    for r in up.itertuples():
        L.append(f"| `{r.folder}/{r.zip}` | {n(r.images)} | {n(r.boxes)} | {r.patients} | {rf._gb(r.size_bytes)} | `{r.sha256[:16]}…` |")
    L += ["", f"- Dataset chưa nén: `{rel(out)}/Roboflow_CT_Positive_{a['per_modality']}/`, `Roboflow_MRI_Positive_{a['per_modality']}/` "
          "(`data.yaml`, `train/images/`, `train/labels/`).",
          f"- `{rel(out)}/manifest.csv`: từng ảnh (tên mới ↔ ID BN, pha/chuỗi, lát, số box, mức chọn, ZIP, file nguồn, SHA256). Chỉ lưu local.",
          f"- `{rel(out)}/upload_manifest.csv`: theo dõi từng ZIP đã upload (cột `uploaded_to_roboflow`, `upload_date`, `roboflow_project`).",
          f"- `{rel(out)}/previews/`: 100 ảnh vẽ box thật (50 CT + 50 MRI, mỗi ảnh một BN khác nhau) + `contact_sheet_CT.jpg`, `contact_sheet_MRI.jpg`.",
          "", "## 6. Upload", "",
          "- **CT** (CC BY 4.0): có thể upload lên Roboflow, ghi nguồn PLC-CECT trong mô tả project. Upload từng ZIP 1.000 ảnh trong `zips/` "
          "(`{rel(out)}/zips/CT/`).",
          "- **MRI** (CC BY-NC 4.0, *\"not to further copy, publish, or distribute\"*): **giữ offline** (`{rel(out)}/zips/MRI_offline/`); không upload khi chưa có quyền phân phối lại.",
          "- Chưa có file nào được upload. Chỉnh sửa box trên Roboflow (nếu có) là hiệu chỉnh bổ sung, phải ghi lại riêng.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
