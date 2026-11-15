"""GĐ2a QA of an exported YOLO pilot dataset (read-only on the source data).

Checks (any failure -> exit code 1):
  C1 every image has a label file and vice versa; slices.csv == files on disk
  C2 100 % of label lines valid: 5 fields, class 0, finite, w,h > 0, box inside [0, 1]
  C3 every image readable, single channel uint8, size == slices.csv
  C4 no patient in two splits; split of every patient == frozen split file; no test patient; no excluded/quarantined volume
  C5 positives have >= 1 box, negatives have 0 boxes
  C7 patient coverage: every Train/Val patient with usable volumes is exported; lesion patients have positives
  C6 source round trip on a random sample: PNG == NIfTI slice.T pixel-exact, YOLO boxes == boxes recomputed from mask,
     negatives have no mask voxels (and, MRI, no JSON 2D_box) on that slice
Also writes distribution tables and N overlay figures (from the exported PNG + txt only).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import load_paths  # noqa: E402
from livercancer.data.nifti_io import as_3d, to_display  # noqa: E402
from livercancer.data.yolo_boxes import component_boxes, from_yolo, to_yolo  # noqa: E402

MRI_SEQ_TO_FILE = {"In Phase": "InPhase", "Out Phase": "OutPhase"}


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def parse_label(path: Path) -> tuple[list[tuple[float, ...]], list[str]]:
    boxes, errs = [], []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        p = line.split()
        if len(p) != 5:
            errs.append(f"{path.name}:{n} fields={len(p)}")
            continue
        try:
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
        except ValueError:
            errs.append(f"{path.name}:{n} not numeric")
            continue
        vals = (cx, cy, w, h)
        if c != 0 or not all(math.isfinite(x) for x in vals) or w <= 0 or h <= 0 or \
                cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
            errs.append(f"{path.name}:{n} invalid {line}")
            continue
        boxes.append(vals)
    return boxes, errs


def source_paths(paths: dict, modality: str, pid: str, series: str, label: str) -> tuple[str, str, dict | None]:
    if modality == "CT":
        d = Path(paths["ct"]["data_dir"])
        return (str(d / "ct_files" / f"{pid}_ct_{series}.nii.gz"),
                str(d / "mask_files" / f"{pid}_mask_{series}.nii.gz") if label != "CN" else "", None)
    return "", "", None  # MRI handled with JSON in caller


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Vietnamese text on Windows consoles / redirected logs
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", choices=["ct", "mri"], required=True)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--n-roundtrip-volumes", type=int, default=40)
    ap.add_argument("--n-figures", type=int, default=20)
    ap.add_argument("--check-all-images", action="store_true", help="C3 on every image instead of a 3000-image sample")
    ap.add_argument("--fig-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = load_paths()
    data = args.data or paths["project"]["processed"] / "pilot" / f"yolo_{args.modality}"
    fig_dir = args.fig_dir or paths["project"]["reports"] / "figures" / "pilot_qa"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(data / "slices.csv")
    vols = read_csv(data / "volumes.csv")
    fails: dict[str, list[str]] = defaultdict(list)
    MOD = args.modality.upper()

    # C1 file correspondence
    for split in sorted({r["split"] for r in rows}):
        imgs = {p.stem for p in (data / "images" / split).glob("*.png")}
        labs = {p.stem for p in (data / "labels" / split).glob("*.txt")}
        csv_ = {r["file"] for r in rows if r["split"] == split}
        for name, a, b in (("image_without_label", imgs, labs), ("label_without_image", labs, imgs),
                           ("csv_without_file", csv_, imgs), ("file_not_in_csv", imgs, csv_)):
            fails["C1"] += [f"{split}/{x} {name}" for x in sorted(a - b)][:20]

    build = json.loads((data / "build_args.json").read_text(encoding="utf-8"))
    # C2, C3, C5
    n_lines = 0
    for r in rows:
        boxes, errs = parse_label(data / "labels" / r["split"] / f"{r['file']}.txt")
        n_lines += len(boxes) + len(errs)
        fails["C2"] += errs
        if (r["kind"] == "pos") != (len(boxes) > 0) or len(boxes) != int(r["n_boxes"]):
            fails["C5"].append(f"{r['file']} kind={r['kind']} boxes={len(boxes)} csv={r['n_boxes']}")
        if MOD == "MRI" and r["kind"] == "pos" and "mri_min_json_iou" in build and \
                (not r.get("json_iou") or float(r["json_iou"]) < build["mri_min_json_iou"]):
            fails["C5"].append(f"{r['file']} MRI positive without agreeing JSON box (iou={r.get('json_iou')})")
        r["_boxes"] = boxes
    for r in rows if (args.check_all_images or len(rows) <= 3000) else random.Random(args.seed).sample(rows, 3000):
        im = cv2.imread(str(data / "images" / r["split"] / f"{r['file']}.png"), cv2.IMREAD_UNCHANGED)
        if im is None or im.dtype != np.uint8 or im.ndim != 2 or im.shape != (int(r["height"]), int(r["width"])):
            fails["C3"].append(f"{r['file']} {None if im is None else (im.dtype, im.shape)}")

    # C4 leakage / exclusions
    frozen = {r["patient_id"]: r["split"] for r in read_csv(paths["project"]["splits"] / f"{args.modality}_split.csv")}
    psplit = defaultdict(set)
    for r in rows:
        psplit[r["patient_id"]].add(r["split"])
    fails["C4"] += [f"{p} in {s}" for p, s in psplit.items() if len(s) > 1]
    fails["C4"] += [f"{p} split {s} != frozen {frozen.get(p)}" for p, s in psplit.items() if s != {frozen.get(p)}]
    fails["C4"] += [f"{p} is a test patient" for p in psplit if frozen.get(p) == "test"]
    bad_vols = {(v["patient_id"], v["series"]) for v in vols if v["status"] != "ok"}
    fails["C4"] += [f"{r['patient_id']} {r['series']} excluded volume exported" for r in rows
                    if (r["patient_id"], r["series"]) in bad_vols][:20]
    if MOD == "MRI":
        q = {(r["patient_id"], r["sequence"]) for r in read_csv(paths["project"]["splits"].parent / "quarantine" / "mri_bbox_quarantine.csv")}
        fails["C4"] += [f"{r['patient_id']} {r['series']} quarantined series exported" for r in rows
                        if (r["patient_id"], r["series"]) in q][:20]

    # C7 patient coverage: no eligible Train/Val patient (hence no rare class) lost by slice sampling
    split_rows = read_csv(paths["project"]["splits"] / f"{args.modality}_split.csv")
    exported_splits = {r["split"] for r in rows}
    ok_vol_pat = {v["patient_id"] for v in vols if v["status"] == "ok"}
    expected = {r["patient_id"]: r for r in split_rows if r["split"] in exported_splits and r["patient_id"] in ok_vol_pat}
    got = {r["patient_id"] for r in rows}
    fails["C7"] += [f"{p} ({r['label']}, {r['split']}) has usable volumes but no exported slice"
                    for p, r in expected.items() if p not in got]
    pos_pat = {r["patient_id"] for r in rows if r["kind"] == "pos"}
    lesion_vol_pat = {v["patient_id"] for v in vols if v["status"] == "ok" and int(v.get("n_pos_avail") or 0) > 0}
    fails["C7"] += [f"{p} ({expected[p]['label']}) has lesion slices in source but no positive slice exported"
                    for p in lesion_vol_pat & set(expected) if p not in pos_pat]
    coverage = []
    for sp in sorted(exported_splits):
        for lab in sorted({r["label"] for r in split_rows}):
            n_split = sum(1 for r in split_rows if r["split"] == sp and r["label"] == lab)
            n_exp = sum(1 for p, r in expected.items() if r["split"] == sp and r["label"] == lab)
            n_got = len({r["patient_id"] for r in rows if r["split"] == sp and r["patient_label"] == lab})
            coverage.append({"split": sp, "label": lab, "patients_in_split": n_split,
                             "patients_with_usable_volumes": n_exp, "patients_exported": n_got})
            if n_got != n_exp:
                fails["C7"].append(f"{sp}/{lab}: exported {n_got} patients, expected {n_exp}")

    # C6 round trip against the source NIfTI
    by_vol = defaultdict(list)
    for r in rows:
        by_vol[(r["patient_id"], r["series"])].append(r)
    sample = random.Random(args.seed).sample(sorted(by_vol), min(args.n_roundtrip_volumes, len(by_vol)))
    info = json.loads(Path(paths["mri"]["annotation"]).read_text(encoding="utf-8"))["Annotation_info"] if MOD == "MRI" else None
    n_checked = 0
    for pid, series in sample:
        rs = by_vol[(pid, series)]
        if MOD == "CT":
            img_p, mask_p, _ = source_paths(paths, "CT", pid, series, rs[0]["patient_label"])
            json_k: set[int] = set()
        else:
            s = next(x for x in info[pid] if x["phase"] == series)
            c = s["annotation"]["lesion"]["0"]["category"]
            f = MRI_SEQ_TO_FILE.get(series, series)
            img_p = str(Path(paths["mri"]["images"]) / f"{pid}_{c}_{f}_0000.nii.gz")
            mask_p = str(Path(paths["mri"]["labels"]) / f"{pid}_{c}_{f}.nii.gz")
            json_k = {int(b["slice_idx"]) for b in s["annotation"]["lesion"]["0"]["bbox"]["2D_box"]}
        vol, _ = as_3d(np.asanyarray(nib.load(img_p).dataobj), max_diff_frac=build.get("max_channel_diff", 0.0))
        mask = (np.asanyarray(nib.load(mask_p).dataobj.get_unscaled()) > 0) if mask_p else np.zeros(vol.shape, bool)
        for r in rs:
            k = int(r["slice_k"])
            png = cv2.imread(str(data / "images" / r["split"] / f"{r['file']}.png"), cv2.IMREAD_UNCHANGED)
            if not np.array_equal(png, to_display(vol[:, :, k])):
                fails["C6"].append(f"{r['file']} pixels differ from source")
            exp, _ = component_boxes(mask[:, :, k], build["min_area"])
            exp_y = sorted(tuple(round(x, 6) for x in to_yolo(b, vol.shape[0], vol.shape[1])) for b in exp)
            got = sorted(tuple(round(x, 6) for x in b) for b in r["_boxes"])
            if r["kind"] == "pos" and not np.allclose(exp_y, got, atol=2e-6):
                fails["C6"].append(f"{r['file']} boxes differ from mask")
            if r["kind"] == "neg" and (mask[:, :, k].any() or k in json_k):
                fails["C6"].append(f"{r['file']} negative slice has an annotation")
            n_checked += 1

    # Distribution
    dist = Counter((r["split"], r["patient_label"], r["kind"]) for r in rows)
    pats = Counter((r["split"], r["patient_label"]) for r in {(r["patient_id"], r["split"], r["patient_label"]): r
                                                              for r in rows}.values())
    labels = sorted({r["patient_label"] for r in rows})
    table = ["| Split | Lớp (BN) | BN | Lát dương | Lát âm | Box |", "|---|---|---|---|---|---|"]
    for sp in ("train", "val"):
        for lab in labels:
            nb = sum(int(r["n_boxes"]) for r in rows if r["split"] == sp and r["patient_label"] == lab)
            table.append(f"| {sp} | {lab} | {pats[(sp, lab)]} | {dist[(sp, lab, 'pos')]} | {dist[(sp, lab, 'neg')]} | {nb} |")
    size = sum(p.stat().st_size for p in (data / "images").rglob("*.png")) + sum(p.stat().st_size for p in (data / "labels").rglob("*.txt"))

    # Figures (from exported files only)
    rng = random.Random(args.seed)
    pos = [r for r in rows if r["kind"] == "pos"]
    neg = [r for r in rows if r["kind"] == "neg"]
    n_neg = max(1, args.n_figures // 5)
    picks = rng.sample(pos, min(len(pos), args.n_figures - n_neg)) + rng.sample(neg, min(len(neg), n_neg))
    for old in fig_dir.glob(f"{args.modality}_*.png"):
        old.unlink()
    for i, r in enumerate(picks, 1):
        im = cv2.imread(str(data / "images" / r["split"] / f"{r['file']}.png"), cv2.IMREAD_UNCHANGED)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        for b in r["_boxes"]:
            x0, x1, y0, y1 = from_yolo(*b, width=im.shape[1], height=im.shape[0])
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=1.2))
        ax.set_title(f"{MOD} {r['split']} {r['patient_id']} {r['series']} k={r['slice_k']}\n"
                     f"patient label: {r['patient_label']} | {r['kind']} ({r['n_boxes']} box)", fontsize=8)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{args.modality}_{i:02d}_{r['kind']}_{r['file']}.png", dpi=90)
        plt.close(fig)

    summary = {
        "modality": MOD, "data": str(data), "n_images": len(rows), "n_label_files": len(rows), "n_box_lines": n_lines,
        "slices": {f"{k[0]}_{k[1]}": v for k, v in sorted(Counter((r["split"], r["kind"]) for r in rows).items())},
        "patients": {sp: len({r["patient_id"] for r in rows if r["split"] == sp}) for sp in ("train", "val")},
        "volumes_status": dict(Counter(v["status"] for v in vols)),
        "fragments_dropped": sum(int(r["fragments_dropped"]) for r in rows),
        "size_bytes": size, "roundtrip_volumes": len(sample), "roundtrip_slices": n_checked,
        "checks": {c: {"failures": len(fails[c]), "examples": fails[c][:5]} for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")},
        "patient_coverage": coverage, "images_checked_C3": len(rows) if (args.check_all_images or len(rows) <= 3000) else 3000,
        "distribution_table": table, "figures": sorted(p.name for p in fig_dir.glob(f"{args.modality}_*.png")),
    }
    if MOD == "MRI":
        ious = [float(r["json_iou"]) for r in rows if r.get("json_iou")]
        summary["mri_json_iou_on_positive_slices"] = {
            "n": len(ious), "no_json_box_on_slice": sum(1 for r in rows if r["kind"] == "pos" and not r.get("json_iou")),
            "quantiles_p5_p25_p50": np.percentile(ious, [5, 25, 50]).round(3).tolist() if ious else [],
            "lt_0.3": sum(i < 0.3 for i in ious)}
    (data / "qa_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("distribution_table", "figures")}, indent=1, ensure_ascii=False))
    print("\n".join(table))
    ok = all(not fails[c] for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7"))
    print("QA:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
