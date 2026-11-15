"""GĐ2a: build the YOLO pilot dataset (Train + Val only; Test is never exported here).

Per volume (streamed, one at a time per worker):
  read NIfTI -> check orientation LPS + image/mask shape & affine -> CT 5D -> 3D (identical channels required)
  -> 2D slices displayed as slice.T -> positive slices (lesion mask present) get one YOLO box per 8-connected
  component (fragments < --min-area dropped; a slice whose components are all fragments is skipped, never negative)
  -> negative slices = no lesion annotation and >= --neg-gap slices away from any annotated slice
     (CT: liver ∪ lesion present on the slice; MRI: also outside the JSON 2D_box slices).
Intensities are kept as stored: CT is already windowed [-200, 200] HU -> uint8 at the source; MRI is already
per-volume clipped at the 99.5th percentile -> uint8 (≈0.5 % voxels = 255 in every series checked).
A negative slice means "no lesion annotated on this slice"; it is never a healthy-patient label.

Excluded (logged in volumes.csv): missing CT files, empty/mismatched lesion masks, quarantined MRI series.

Outputs in --out: images/{train,val}/*.png, labels/{train,val}/*.txt, data.yaml, slices.csv, volumes.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import load_paths  # noqa: E402
from livercancer.data.nifti_io import as_3d, to_display  # noqa: E402
from livercancer.data.yolo_boxes import box_iou, component_boxes, to_yolo  # noqa: E402

MRI_SEQ_TO_FILE = {"In Phase": "InPhase", "Out Phase": "OutPhase"}
LPS = ("L", "P", "S")
NA = ("", "NA")


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def zrange(s: str) -> tuple[int, int] | None:
    return tuple(map(int, s.split("-"))) if s else None


# ----------------------------------------------------------------------------- volume lists
def ct_volumes(paths: dict, splits: set[str]) -> list[dict]:
    split = {r["patient_id"]: r["split"] for r in read_csv(paths["project"]["splits"] / "ct_split.csv")}
    audit = {(r["patient_id"], r["phase"]): r for r in read_csv(paths["project"]["reports"] / "audit" / "ct_volumes.csv")}
    data_dir = Path(paths["ct"]["data_dir"])
    vols = []
    for r in read_csv(Path(paths["ct"]["csv"])):
        a = audit[(r["patient_id"], r["phase"])]
        v = {"modality": "CT", "patient_id": r["patient_id"], "series": r["phase"], "label": r["cancer_type"],
             "split": split[r["patient_id"]], "image": str(data_dir / r["ct_path"]),
             "mask": "" if r["mask_path"] in NA else str(data_dir / r["mask_path"]),
             "liver": str(data_dir / r["liver_mask_path"]), "status": "ok"}
        if a["ct_exists"] != "True":
            v["status"] = "excluded:ct_file_missing"
        elif v["mask"] and a.get("lesion_empty") == "True":
            v["status"] = "excluded:lesion_mask_empty"
        elif v["mask"] and a.get("lesion_shape_match_ct") == "False":
            v["status"] = "excluded:lesion_mask_shape_mismatch"
        # estimates for negative quotas (from GD1 audit, refined at export time)
        lz, mz = zrange(a.get("liver_z", "")), zrange(a.get("lesion_z", ""))
        v["est_pos"] = int(a.get("lesion_slices") or 0)
        lo_hi = lz or (mz and (max(0, mz[0] - 10), mz[1] + 10))
        v["est_neg_cand"] = max(0, (lo_hi[1] - lo_hi[0] + 1 if lo_hi else 0) - (v["est_pos"] + 4 if mz else 0))
        vols.append(v)
    return [v for v in vols if v["split"] in splits]


def mri_volumes(paths: dict, splits: set[str]) -> list[dict]:
    split = {r["patient_id"]: r for r in read_csv(paths["project"]["splits"] / "mri_split.csv")}
    audit = {(r["patient_id"], r["sequence"]): r for r in read_csv(paths["project"]["reports"] / "audit" / "mri_series.csv")}
    quarantine = {(r["patient_id"], r["sequence"]) for r in
                  read_csv(paths["project"]["splits"].parent / "quarantine" / "mri_bbox_quarantine.csv")}
    info = json.loads(Path(paths["mri"]["annotation"]).read_text(encoding="utf-8"))["Annotation_info"]
    vols = []
    for pid, series in info.items():
        for s in series:
            les = s["annotation"]["lesion"]["0"]
            f = MRI_SEQ_TO_FILE.get(s["phase"], s["phase"])
            a = audit[(pid, s["phase"])]
            v = {"modality": "MRI", "patient_id": pid, "series": s["phase"], "file_seq": f,
                 "label": split[pid]["label"], "split": split[pid]["split"],
                 "image": str(Path(paths["mri"]["images"]) / f"{pid}_{les['category']}_{f}_0000.nii.gz"),
                 "mask": str(Path(paths["mri"]["labels"]) / f"{pid}_{les['category']}_{f}.nii.gz"),
                 "json_2d": json.dumps({int(b["slice_idx"]): [b["x_min"], b["x_max"], b["y_min"], b["y_max"]]
                                        for b in les["bbox"]["2D_box"]}),
                 "status": "quarantined:json_box_vs_mask_mismatch" if (pid, s["phase"]) in quarantine else "ok"}
            mz = zrange(a.get("mask_z", ""))
            n = int(a["image_shape"].split("x")[2])
            v["est_pos"] = (mz[1] - mz[0] + 1) if mz else 0
            v["est_neg_cand"] = max(0, n - v["est_pos"] - 4)
            vols.append(v)
    return [v for v in vols if v["split"] in splits]


def assign_quotas(vols: list[dict], pos_stride: dict[str, int], neg_ratio: float) -> None:
    """Negative quota per volume, proportional to its candidate negatives, budget = neg_ratio x positives per split."""
    for split in {v["split"] for v in vols}:
        ok = [v for v in vols if v["split"] == split and v["status"] == "ok"]
        budget = round(neg_ratio * sum(math.ceil(v["est_pos"] / pos_stride[split]) for v in ok))
        total = sum(v["est_neg_cand"] for v in ok) or 1
        for v in ok:
            v["neg_quota"] = round(budget * v["est_neg_cand"] / total)


# ----------------------------------------------------------------------------- per-volume export
def load_checked(path: str, ref: nib.Nifti1Image | None = None) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    if nib.aff2axcodes(img.affine) != LPS:
        raise ValueError(f"orientation {nib.aff2axcodes(img.affine)} != LPS: {path}")
    if ref is not None and (tuple(img.shape[:3]) != tuple(ref.shape[:3]) or not np.allclose(img.affine, ref.affine, atol=1e-3)):
        raise ValueError(f"shape/affine mismatch vs image: {path}")
    return np.asanyarray(img.dataobj.get_unscaled()) if ref is not None else np.asanyarray(img.dataobj), img


def export_volume(job: dict) -> dict:
    v, out, opt = job["vol"], Path(job["out"]), job["opt"]
    res = {"key": (v["patient_id"], v["series"]), "slices": [], "error": "", "note": {}}
    try:
        img, ref = load_checked(v["image"])
        img, note = as_3d(img, max_diff_frac=opt["max_channel_diff"])
        diff_mask = note.pop("diff_mask", None)
        res["note"] = note
        if img.dtype != np.uint8:
            raise ValueError(f"unexpected dtype {img.dtype}")
        nk = img.shape[2]
        lesion = load_checked(v["mask"], ref)[0] > 0 if v["mask"] else np.zeros(img.shape, bool)
        les_k = np.nonzero(lesion.any(axis=(0, 1)))[0]
        if v["modality"] == "CT":
            region = (load_checked(v["liver"], ref)[0] > 0) | lesion
            if diff_mask is not None and (diff_mask & region).any():
                raise ValueError(f"differing RGB channels overlap liver/lesion ({int((diff_mask & region).sum())} voxels)")
            region_k = set(np.nonzero(region.any(axis=(0, 1)))[0].tolist())
            if not region_k and len(les_k):  # empty liver mask (GD1 E14): lesion z-range +- 10
                region_k = set(range(max(0, les_k.min() - 10), min(nk, les_k.max() + 11)))
            json_k: set[int] = set()
        else:
            region_k = set(range(nk))
            json_2d = {int(k): b for k, b in json.loads(v["json_2d"]).items()}
            json_k = set(json_2d)
        annotated = set(les_k.tolist()) | json_k
        rng = random.Random(f"{opt['seed']}-{v['patient_id']}-{v['series']}")

        stride = opt["pos_stride"][v["split"]]
        pos_k = sorted(les_k.tolist())[::stride]
        neg_cand = sorted(k for k in region_k if all(abs(k - a) >= opt["neg_gap"] for a in annotated))
        neg_k = sorted(rng.sample(neg_cand, min(v.get("neg_quota", 0), len(neg_cand))))

        stem_seq = v.get("file_seq", v["series"])
        for kind, ks in (("pos", pos_k), ("neg", neg_k)):
            for k in ks:
                boxes, filtered = component_boxes(lesion[:, :, k], opt["min_area"]) if kind == "pos" else ([], 0)
                if kind == "pos" and not boxes:
                    res["note"]["skipped_all_fragments"] = res["note"].get("skipped_all_fragments", 0) + 1
                    continue
                json_iou = None
                if v["modality"] == "MRI" and kind == "pos":
                    jb = json_2d.get(k)
                    union = (min(b[0] for b in boxes), max(b[1] for b in boxes), min(b[2] for b in boxes), max(b[3] for b in boxes))
                    json_iou = round(box_iou(union, tuple(jb)), 4) if jb else None
                    if json_iou is None or json_iou < opt["mri_min_json_iou"]:
                        # mask and original JSON box disagree on this slice -> unverified, neither positive nor negative
                        key = "skipped_no_json_box" if json_iou is None else "skipped_low_json_iou"
                        res["note"][key] = res["note"].get(key, 0) + 1
                        continue
                name = f"{v['patient_id']}_{stem_seq}_k{k:03d}"
                ni, nj = img.shape[0], img.shape[1]
                if not cv2.imwrite(str(out / "images" / v["split"] / f"{name}.png"), np.ascontiguousarray(to_display(img[:, :, k]))):
                    raise IOError(f"cv2.imwrite failed for {name}")
                lines = ["0 {:.6f} {:.6f} {:.6f} {:.6f}".format(*to_yolo(b, ni, nj)) for b in boxes]
                (out / "labels" / v["split"] / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                rec = {"modality": v["modality"], "patient_id": v["patient_id"], "patient_label": v["label"],
                       "split": v["split"], "series": v["series"], "slice_k": k, "n_slices": nk,
                       "kind": kind, "n_boxes": len(boxes), "fragments_dropped": filtered,
                       "width": ni, "height": nj, "file": name}
                if json_iou is not None:
                    rec["json_iou"] = json_iou
                res["slices"].append(rec)
        res["note"].update(n_pos_avail=len(les_k), n_neg_cand=len(neg_cand), quota=v.get("neg_quota", 0))
    except (EOFError, zlib.error) as e:  # truncated / corrupt .nii.gz (incomplete download)
        res["corrupt"] = f"{type(e).__name__}: {e}"
        res["slices"] = []
    except Exception as e:  # noqa: BLE001 - logged per volume, run continues
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def write_csv(rows: list[dict], path: Path) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", choices=["ct", "mri"], required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pos-stride-train", type=int, default=None, help="keep every n-th lesion slice in train")
    ap.add_argument("--pos-stride-val", type=int, default=None, help="keep every n-th lesion slice in val")
    ap.add_argument("--neg-ratio", type=float, default=None, help="negatives : positives per split")
    ap.add_argument("--neg-gap", type=int, default=2, help="min slice distance of a negative to any annotated slice")
    ap.add_argument("--min-area", type=int, default=10, help="px; smaller lesion components are fragments")
    ap.add_argument("--max-channel-diff", type=float, default=0.01,
                    help="CT 5D: max fraction of voxels whose RGB channels differ (burned-in text); must not touch liver/lesion")
    ap.add_argument("--mri-min-json-iou", type=float, default=0.3,
                    help="MRI positive slice kept only if the mask box agrees with the original JSON 2D_box on that slice")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit-patients", type=int, default=0, help="dry run on first N patients per split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    defaults = {"ct": (4, 1.0, 2), "mri": (1, 0.5, 1)}[args.modality]  # (train stride, neg ratio, val stride): <= 5 GB total
    stride_tr = args.pos_stride_train or defaults[0]
    neg_ratio = args.neg_ratio if args.neg_ratio is not None else defaults[1]
    paths = load_paths()
    out = args.out or paths["project"]["processed"] / "pilot" / f"yolo_{args.modality}"
    if out.exists() and any(out.rglob("*.png")):
        print(f"{out} already contains images; remove it first (generated data only).")
        return 1
    for d in ("images", "labels"):
        for s in args.splits:
            (out / d / s).mkdir(parents=True, exist_ok=True)

    vols = (ct_volumes if args.modality == "ct" else mri_volumes)(paths, set(args.splits))
    if args.limit_patients:
        keep = set()
        for s in args.splits:
            keep |= set(sorted({v["patient_id"] for v in vols if v["split"] == s})[: args.limit_patients])
        vols = [v for v in vols if v["patient_id"] in keep]
    pos_stride = {"train": stride_tr, "val": args.pos_stride_val or defaults[2]}
    assign_quotas(vols, pos_stride, neg_ratio)
    opt = {"seed": args.seed, "pos_stride": pos_stride, "neg_gap": args.neg_gap, "min_area": args.min_area,
           "max_channel_diff": args.max_channel_diff, "mri_min_json_iou": args.mri_min_json_iou}

    todo = [v for v in vols if v["status"] == "ok"]
    print(f"{args.modality.upper()}: {len(vols)} volumes in {args.splits}, exporting {len(todo)}; "
          f"excluded {dict(Counter(v['status'] for v in vols if v['status'] != 'ok'))}", flush=True)
    t0, slices, notes = time.time(), [], {}
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(export_volume, {"vol": v, "out": str(out), "opt": opt}) for v in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            notes[r["key"]] = r
            slices += r["slices"]
            if n % 50 == 0 or n == len(futs):
                print(f"  {n}/{len(futs)} volumes, {len(slices)} slices, {time.time() - t0:.0f}s", flush=True)

    for v in vols:
        r = notes.get((v["patient_id"], v["series"]))
        if r:
            if r["error"]:
                v["status"] = f"error:{r['error']}"
            elif r.get("corrupt"):
                v["status"] = "excluded:source_file_corrupt"
                v["detail"] = r["corrupt"]
            v.update({f"n_{k}": sum(s['kind'] == k for s in r["slices"]) for k in ("pos", "neg")})
            v.update({k: r["note"].get(k, "") for k in ("converted", "channel_diff_frac", "n_pos_avail", "n_neg_cand",
                                                         "skipped_all_fragments", "skipped_no_json_box",
                                                         "skipped_low_json_iou")})
        for k in ("json_2d", "image", "mask", "liver"):
            v.pop(k, None)
    write_csv(vols, out / "volumes.csv")
    write_csv(sorted(slices, key=lambda s: (s["split"], s["file"])), out / "slices.csv")
    (out / "data.yaml").write_text(
        f"# YOLO pilot ({args.modality.upper()}), generated by scripts/20_build_yolo_pilot.py. Test split is NOT included.\n"
        f"# 'path' is rewritten by the Colab notebook / QA script to the extracted location.\n"
        f"path: /content/yolo_{args.modality}\ntrain: images/train\nval: images/val\nnames:\n  0: lesion\n",
        encoding="utf-8")
    (out / "build_args.json").write_text(json.dumps({**vars(args), "out": str(out), "pos_stride": pos_stride,
                                                     "neg_ratio": neg_ratio}, indent=2), encoding="utf-8")
    errors = [v for v in vols if v["status"].startswith("error")]
    print(f"done: {len(slices)} slices ({dict(Counter((s['split'], s['kind']) for s in slices))}), "
          f"errors {len(errors)}, {time.time() - t0:.0f}s")
    for v in errors[:10]:
        print("  ", v["patient_id"], v["series"], v["status"])
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
