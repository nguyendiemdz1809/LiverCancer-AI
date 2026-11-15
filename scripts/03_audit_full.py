"""GĐ1 full correspondence audit (read-only).

Image volumes: header only (shape, ndim, affine) — voxel data is NOT read.
Masks/labels: fully read one at a time as raw integers (small), to check emptiness, values, extent.

Outputs:
  reports/audit/ct_volumes.csv    one row per CSV row (patient x phase)
  reports/audit/mri_series.csv    one row per MRI series (patient x sequence)
  reports/audit/full_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import load_paths  # noqa: E402

MRI_SEQ_TO_FILE = {"In Phase": "InPhase", "Out Phase": "OutPhase"}
NA = ("", "NA")


def read_mask(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj.get_unscaled()), img


def scaled(img: nib.Nifti1Image) -> bool:
    # nibabel moves scl_slope/inter into the array proxy and resets the header fields to NaN
    slope, inter = getattr(img.dataobj, "slope", 1.0), getattr(img.dataobj, "inter", 0.0)
    return bool(slope != 1.0 or inter != 0.0)


# ----------------------------------------------------------------------------- CT
def ct_row(args: tuple[dict, str]) -> dict:
    r, data_dir = args
    data_dir = Path(data_dir)
    out = {k: r[k] for k in ("patient_id", "phase", "cancer_type")}
    ct_p = data_dir / r["ct_path"]
    out["ct_exists"] = ct_p.exists()
    liver, limg = read_mask(data_dir / r["liver_mask_path"])
    liver = liver > 0
    out["liver_shape"] = "x".join(map(str, liver.shape))
    out["liver_voxels"] = int(liver.sum())
    out["liver_empty"] = out["liver_voxels"] == 0
    zl = np.nonzero(liver.any(axis=(0, 1)))[0]
    out["liver_z"] = f"{zl.min()}-{zl.max()}" if len(zl) else ""

    if out["ct_exists"]:
        cimg = nib.load(str(ct_p))
        out["ct_shape"] = "x".join(map(str, cimg.shape))
        out["ct_ndim"] = len(cimg.shape)
        out["ct_dtype"] = str(cimg.get_data_dtype())
        out["ct_zooms"] = "x".join(f"{z:g}" for z in cimg.header.get_zooms()[:3])
        out["liver_shape_match"] = tuple(cimg.shape[:3]) == liver.shape
        out["liver_affine_match"] = bool(np.allclose(cimg.affine, limg.affine, atol=1e-3))

    out["has_lesion_mask"] = r["mask_path"] not in NA
    if out["has_lesion_mask"]:
        raw, mimg = read_mask(data_dir / r["mask_path"])
        les = raw > 0
        out["lesion_dtype"] = str(mimg.get_data_dtype())
        out["lesion_scaled_header"] = scaled(mimg)
        out["lesion_raw_values"] = ";".join(str(int(v)) for v in np.unique(raw)[:6])
        out["lesion_shape_match_liver"] = les.shape == liver.shape
        out["lesion_affine_match_liver"] = bool(np.allclose(mimg.affine, limg.affine, atol=1e-3))
        if out["ct_exists"]:
            out["lesion_shape_match_ct"] = tuple(cimg.shape[:3]) == les.shape
        out["lesion_voxels"] = int(les.sum())
        out["lesion_empty"] = out["lesion_voxels"] == 0
        if not out["lesion_empty"]:
            out["lesion_components"] = int(ndimage.label(les)[1])
            zs = np.nonzero(les.any(axis=(0, 1)))[0]
            out["lesion_z"] = f"{zs.min()}-{zs.max()}"
            out["lesion_slices"] = int(len(zs))
            if les.shape == liver.shape:
                out["lesion_in_liver_frac"] = round(float((les & liver).sum() / les.sum()), 4)
                # fraction of lesion within liver mask dilated/closed bbox (lesion may be excluded from liver mask)
                lb = np.nonzero(liver)
                box = tuple(slice(a.min(), a.max() + 1) for a in lb) if len(lb[0]) else None
                out["lesion_in_liver_bbox_frac"] = (round(float(les[box].sum() / les.sum()), 4) if box else 0.0)
    return out


# ----------------------------------------------------------------------------- MRI
def mri_row(args: tuple[str, dict, str, str]) -> dict:
    pid, s, images, labels = args
    les = s["annotation"]["lesion"]["0"]
    cat = les["category"]
    seq = MRI_SEQ_TO_FILE.get(s["phase"], s["phase"])
    out = {"patient_id": pid, "sequence": s["phase"], "category_id": cat,
           "num_targets": s["annotation"]["num_targets"]}
    ip = Path(images) / f"{pid}_{cat}_{seq}_0000.nii.gz"
    lp = Path(labels) / f"{pid}_{cat}_{seq}.nii.gz"
    out["image_exists"], out["label_exists"] = ip.exists(), lp.exists()
    if not (out["image_exists"] and out["label_exists"]):
        return out
    img = nib.load(str(ip))
    lab_raw, limg = read_mask(lp)
    lab = lab_raw > 0
    shp = tuple(img.shape[:3])
    out["image_shape"] = "x".join(map(str, img.shape))
    out["image_ndim"] = len(img.shape)
    out["label_shape_match"] = shp == lab.shape
    out["label_affine_match"] = bool(np.allclose(img.affine, limg.affine, atol=1e-3))
    zooms = img.header.get_zooms()[:3]
    json_sp = [*s["pixel_spacing"], s["slice_spacing"]]
    out["spacing_match_json"] = bool(np.allclose(zooms, json_sp, rtol=1e-3, atol=1e-3))
    out["label_raw_values"] = ";".join(str(int(v)) for v in np.unique(lab_raw)[:6])
    out["label_voxels"] = int(lab.sum())
    out["label_empty"] = out["label_voxels"] == 0

    b3 = les["bbox"]["3D_box"]
    slices = les["bbox"]["2D_box"]
    out["n_2d_boxes"] = len(slices)
    ks = [int(d["slice_idx"]) for d in slices]
    out["json_slice_in_range"] = all(0 <= k < shp[2] for k in ks)
    out["json_box_in_bounds"] = bool(b3["x_min"] >= -1 and b3["y_min"] >= -1
                                     and b3["x_max"] <= shp[0] + 1 and b3["y_max"] <= shp[1] + 1)
    out["json_z"] = f"{b3['z_min']}-{b3['z_max']}"
    if not out["label_empty"]:
        idx = np.nonzero(lab)
        i0, i1, j0, j1 = idx[0].min(), idx[0].max() + 1, idx[1].min(), idx[1].max() + 1
        out["mask_z"] = f"{idx[2].min()}-{idx[2].max()}"
        ix = max(0.0, min(b3["x_max"], i1) - max(b3["x_min"], i0))
        iy = max(0.0, min(b3["y_max"], j1) - max(b3["y_min"], j0))
        inter = ix * iy
        union = (b3["x_max"] - b3["x_min"]) * (b3["y_max"] - b3["y_min"]) + (i1 - i0) * (j1 - j0) - inter
        out["iou_json3d_vs_mask_xy"] = round(inter / union, 4) if union > 0 else 0.0
        out["z_overlap"] = bool(idx[2].min() <= b3["z_max"] and idx[2].max() >= b3["z_min"])
        out["mask_components"] = int(ndimage.label(lab)[1])
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def count(rows: list[dict], key: str, val=True) -> int:
    return sum(1 for r in rows if r.get(key) == val)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only first N rows per modality (dry run)")
    args = ap.parse_args()

    paths = load_paths()
    out_dir = paths["project"]["reports"] / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(paths["ct"]["csv"], encoding="utf-8") as f:
        ct_rows = list(csv.DictReader(f))
    with open(paths["mri"]["annotation"], encoding="utf-8") as f:
        info = json.load(f)["Annotation_info"]
    mri_jobs = [(pid, s, paths["mri"]["images"], paths["mri"]["labels"]) for pid, ss in info.items() for s in ss]
    ct_jobs = [(r, paths["ct"]["data_dir"]) for r in ct_rows]
    if args.limit:
        ct_jobs, mri_jobs = ct_jobs[: args.limit], mri_jobs[: args.limit]

    with ProcessPoolExecutor(args.workers) as ex:
        ct = []
        for n, row in enumerate(ex.map(ct_row, ct_jobs, chunksize=4), 1):
            ct.append(row)
            if n % 100 == 0:
                print(f"CT  {n}/{len(ct_jobs)}", flush=True)
        mri = []
        for n, row in enumerate(ex.map(mri_row, mri_jobs, chunksize=8), 1):
            mri.append(row)
            if n % 500 == 0:
                print(f"MRI {n}/{len(mri_jobs)}", flush=True)

    write_csv(ct, out_dir / "ct_volumes.csv")
    write_csv(mri, out_dir / "mri_series.csv")

    ct_ok = [r for r in ct if r["ct_exists"]]
    les = [r for r in ct if r.get("has_lesion_mask")]
    summary = {
        "ct": {
            "rows": len(ct), "ct_exists": len(ct_ok), "ct_missing": len(ct) - len(ct_ok),
            "ct_ndim": dict(Counter(r["ct_ndim"] for r in ct_ok)),
            "ct_5d_patients": sorted({r["patient_id"] for r in ct_ok if r["ct_ndim"] != 3}),
            "ct_dtype": dict(Counter(r["ct_dtype"] for r in ct_ok)),
            "ct_zooms": dict(Counter(r["ct_zooms"] for r in ct_ok)),
            "liver_empty": count(ct, "liver_empty"),
            "liver_shape_mismatch": count(ct_ok, "liver_shape_match", False),
            "liver_affine_mismatch": count(ct_ok, "liver_affine_match", False),
            "lesion_masks": len(les),
            "lesion_empty": count(les, "lesion_empty"),
            "lesion_empty_rows": [f"{r['patient_id']}_{r['phase']}" for r in les if r.get("lesion_empty")],
            "lesion_shape_mismatch_ct": count(les, "lesion_shape_match_ct", False),
            "lesion_shape_mismatch_liver": count(les, "lesion_shape_match_liver", False),
            "lesion_affine_mismatch_liver": count(les, "lesion_affine_match_liver", False),
            "lesion_scaled_header": count(les, "lesion_scaled_header"),
            "lesion_dtype": dict(Counter(r.get("lesion_dtype") for r in les)),
            "lesion_components": dict(Counter(r.get("lesion_components") for r in les if not r.get("lesion_empty"))),
            "lesion_in_liver_frac_lt_0.5": sum(1 for r in les if r.get("lesion_in_liver_frac") is not None
                                               and r["lesion_in_liver_frac"] < 0.5),
            "lesion_in_liver_bbox_frac_lt_0.9": sum(1 for r in les if r.get("lesion_in_liver_bbox_frac") is not None
                                                    and r["lesion_in_liver_bbox_frac"] < 0.9),
        },
        "mri": {
            "series": len(mri),
            "missing_files": sum(1 for r in mri if not (r["image_exists"] and r["label_exists"])),
            "image_ndim": dict(Counter(r.get("image_ndim") for r in mri)),
            "label_shape_mismatch": count(mri, "label_shape_match", False),
            "label_affine_mismatch": count(mri, "label_affine_match", False),
            "spacing_mismatch_json": count(mri, "spacing_match_json", False),
            "label_empty": count(mri, "label_empty"),
            "label_empty_series": [f"{r['patient_id']}_{r['sequence']}" for r in mri if r.get("label_empty")],
            "label_raw_values": dict(Counter(r.get("label_raw_values") for r in mri)),
            "json_slice_out_of_range": count(mri, "json_slice_in_range", False),
            "json_box_out_of_bounds": count(mri, "json_box_in_bounds", False),
            "z_no_overlap": count(mri, "z_overlap", False),
            "iou_json_vs_mask_quantiles": (np.percentile([r["iou_json3d_vs_mask_xy"] for r in mri
                                                         if "iou_json3d_vs_mask_xy" in r], [0, 5, 25, 50, 75]).round(3).tolist()),
            "iou_lt_0.3": sum(1 for r in mri if r.get("iou_json3d_vs_mask_xy", 1) < 0.3),
            "mask_components": dict(Counter(r.get("mask_components") for r in mri if not r.get("label_empty"))),
        },
    }
    (out_dir / "full_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
