"""GĐ1 trial audit: fully read a few CT and MRI cases, check image/mask correspondence, export overlays.

Read-only on the dataset. Outputs:
  reports/audit/samples.json
  reports/figures/audit/ct_<patient>.png, mri_<patient>.png
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import load_paths  # noqa: E402
from livercancer.data.nifti_io import as_3d, load_volume, mask_bbox, to_display  # noqa: E402

CT_PHASES = ["C1", "C2", "C3", "P"]
MRI_SEQS = ["C-pre", "C+A", "C+V", "C+Delay", "T2WI", "DWI", "In Phase", "Out Phase"]
MRI_SEQ_TO_FILE = {"In Phase": "InPhase", "Out Phase": "OutPhase"}


def intensity_stats(vol: np.ndarray, region: np.ndarray | None = None) -> dict:
    v = vol[region > 0] if region is not None and region.any() else vol
    p = np.percentile(v, [1, 50, 99])
    return {"min": float(vol.min()), "max": float(vol.max()), "p1": float(p[0]),
            "p50": float(p[1]), "p99": float(p[2]), "mean": round(float(v.mean()), 2)}


def best_slice(mask: np.ndarray | None, fallback: np.ndarray | None = None) -> int:
    """Axial slice (axis k) with the largest mask area."""
    m = mask if mask is not None and mask.any() else fallback
    if m is None or not m.any():
        return 0
    return int(np.argmax((m > 0).sum(axis=(0, 1))))


# ----------------------------------------------------------------------------- CT
def audit_ct_case(pid: str, rows: list[dict], data_dir: Path, fig_dir: Path) -> dict:
    by_phase = {r["phase"]: r for r in rows}
    res = {"patient_id": pid, "cancer_type": rows[0]["cancer_type"], "phases": {}}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
    for c, ph in enumerate(CT_PHASES):
        r = by_phase.get(ph)
        info: dict = {}
        ax_raw, ax_ov = axes[0, c], axes[1, c]
        ct_p = data_dir / r["ct_path"] if r else None
        if r is None or not ct_p.exists():
            info["status"] = "missing_ct_file"
            for ax in (ax_raw, ax_ov):
                ax.text(0.5, 0.5, f"{ph}: CT file missing", ha="center", va="center")
                ax.axis("off")
            res["phases"][ph] = info
            continue

        ct, aff = load_volume(ct_p)
        ct, info["ct_channel_note"] = as_3d(ct)
        liver, aff_l = load_volume(data_dir / r["liver_mask_path"])
        has_mask = r["mask_path"] not in ("", "NA")
        lesion, aff_m = (load_volume(data_dir / r["mask_path"]) if has_mask else (None, None))

        info["shape"] = list(ct.shape)
        info["dtype"] = str(ct.dtype)
        info["liver_shape_match"] = liver.shape == ct.shape
        info["liver_affine_match"] = bool(np.allclose(aff, aff_l, atol=1e-3))
        info["liver_values"] = [int(x) for x in np.unique(liver)]
        info["liver_voxels"] = int((liver > 0).sum())
        info["intensity_all"] = intensity_stats(ct)
        info["intensity_liver"] = intensity_stats(ct, liver)
        if lesion is not None:
            info["lesion_shape_match"] = lesion.shape == ct.shape
            info["lesion_affine_match"] = bool(np.allclose(aff, aff_m, atol=1e-3))
            info["lesion_values"] = [int(x) for x in np.unique(lesion)]
            les = lesion > 0
            info["lesion_voxels"] = int(les.sum())
            info["lesion_slices"] = int(les.any(axis=(0, 1)).sum())
            info["lesion_components_3d"] = int(ndimage.label(les)[1])
            info["lesion_in_liver_frac"] = (round(float((les & (liver > 0)).sum() / les.sum()), 4)
                                            if les.any() else None)
            info["intensity_lesion"] = intensity_stats(ct, lesion)
        else:
            info["lesion_mask"] = f"none (mask_path={r['mask_path']!r} in CSV)"
        k = best_slice(lesion, liver)
        info["shown_slice_k"] = k

        img = to_display(ct[:, :, k])
        ax_raw.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax_raw.set_title(f"{ph}  k={k}/{ct.shape[2]}")
        ax_ov.imshow(img, cmap="gray", vmin=0, vmax=255)
        lv = to_display(liver[:, :, k] > 0)
        if lv.any():
            ax_ov.contour(lv, levels=[0.5], colors="lime", linewidths=0.8)
        if lesion is not None:
            ls = to_display(lesion[:, :, k] > 0)
            if ls.any():
                ax_ov.imshow(np.ma.masked_where(~ls, ls), cmap="autumn", alpha=0.45)
                ax_ov.contour(ls, levels=[0.5], colors="red", linewidths=0.8)
        ax_ov.set_title(f"{ph} overlay (green=liver, red=lesion)")
        for ax in (ax_raw, ax_ov):
            ax.axis("off")
        res["phases"][ph] = info

    fig.suptitle(f"CT {pid} — label (CSV): {res['cancer_type']}", fontsize=14)
    fig.tight_layout()
    out = fig_dir / f"ct_{pid}.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    res["figure"] = str(out)
    return res


# ----------------------------------------------------------------------------- MRI
def box_iou(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = ix * iy
    ua = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / ua if ua > 0 else 0.0


def audit_mri_case(pid: str, series: list[dict], id_to_name: dict, cfg: dict, fig_dir: Path) -> dict:
    by_seq = {s["phase"]: s for s in series}
    lesion0 = series[0]["annotation"]["lesion"]["0"]
    cat = lesion0["category"]
    res = {"patient_id": pid, "category_id": cat, "category": id_to_name[cat], "series": {}}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.6))
    for n, seq in enumerate(MRI_SEQS):
        s = by_seq[seq]
        les = s["annotation"]["lesion"]["0"]
        fname = MRI_SEQ_TO_FILE.get(seq, seq)
        img, aff = load_volume(Path(cfg["images"]) / f"{pid}_{cat}_{fname}_0000.nii.gz")
        img, ch_note = as_3d(img)
        lab, aff_l = load_volume(Path(cfg["labels"]) / f"{pid}_{cat}_{fname}.nii.gz")
        zooms = [float(np.linalg.norm(aff[:3, i])) for i in range(3)]

        info = {
            "image_channel_note": ch_note,
            "shape": list(img.shape), "dtype": str(img.dtype),
            "label_shape_match": lab.shape == img.shape,
            "label_affine_match": bool(np.allclose(aff, aff_l, atol=1e-3)),
            "label_values": [int(x) for x in np.unique(lab)],
            "label_voxels": int((lab > 0).sum()),
            "spacing_json": [*s["pixel_spacing"], s["slice_spacing"]],
            "spacing_header": [round(z, 4) for z in zooms],
            "intensity": intensity_stats(img),
        }
        b3 = les["bbox"]["3D_box"]
        mb = mask_bbox(lab > 0)
        info["json_3d_box"] = b3
        info["mask_bbox_ijk"] = mb
        if mb is not None:
            # Convention A: x->i, y->j ; Convention B: x->j, y->i  (mask bbox inclusive -> +1)
            mask_xy = (mb[0], mb[1] + 1, mb[2], mb[3] + 1)
            info["iou_conventionA_x=i_y=j"] = round(box_iou((b3["x_min"], b3["x_max"], b3["y_min"], b3["y_max"]), mask_xy), 4)
            info["iou_conventionB_x=j_y=i"] = round(box_iou((b3["y_min"], b3["y_max"], b3["x_min"], b3["x_max"]), mask_xy), 4)
            info["z_json"] = [b3["z_min"], b3["z_max"]]
            info["z_mask"] = [mb[4], mb[5]]
        slices = les["bbox"]["2D_box"]
        best = max(slices, key=lambda d: d["area"])
        k = int(best["slice_idx"])
        info["shown_slice_k"] = k
        info["n_2d_boxes"] = len(slices)

        ax = axes[n // 4, n % 4]
        disp = to_display(img[:, :, k]).astype(np.float32)
        lo, hi = np.percentile(disp, [0.5, 99.5])
        ax.imshow(disp, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
        lm = to_display(lab[:, :, k] > 0)
        if lm.any():
            ax.contour(lm, levels=[0.5], colors="red", linewidths=0.9)
        # JSON box drawn with x on columns (i) and y on rows (j)
        ax.add_patch(Rectangle((best["x_min"], best["y_min"]), best["x_max"] - best["x_min"],
                               best["y_max"] - best["y_min"], fill=False, ec="yellow", lw=1.0))
        ax.set_title(f"{seq}  k={k}  {img.shape[0]}x{img.shape[1]}x{img.shape[2]}")
        ax.axis("off")
        res["series"][seq] = info

    fig.suptitle(f"MRI {pid} — label (JSON): {res['category']}  (red=label mask, yellow=JSON 2D_box)", fontsize=13)
    fig.tight_layout()
    out = fig_dir / f"mri_{pid}.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    res["figure"] = str(out)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ct", nargs="*", help="explicit CT patient ids (default: 1 per class + 1 with missing phase)")
    ap.add_argument("--mri", nargs="*", help="explicit MRI patient ids (default: 5 different classes)")
    args = ap.parse_args()

    paths = load_paths()
    rng = random.Random(args.seed)
    reports = paths["project"]["reports"]
    fig_dir = reports / "figures" / "audit"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (reports / "audit").mkdir(parents=True, exist_ok=True)

    # --- CT selection
    data_dir = Path(paths["ct"]["data_dir"])
    with open(paths["ct"]["csv"], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_pid = defaultdict(list)
    for r in rows:
        by_pid[r["patient_id"]].append(r)
    complete = {p for p, rs in by_pid.items() if all((data_dir / r["ct_path"]).exists() for r in rs)}
    incomplete = sorted(set(by_pid) - complete)
    if args.ct:
        ct_ids = args.ct
    else:
        ct_ids = []
        for lab in ("HCC", "ICC", "CHCC", "CN"):
            ct_ids.append(rng.choice(sorted(p for p in complete if by_pid[p][0]["cancer_type"] == lab)))
        ct_ids.append(rng.choice(incomplete))

    # --- MRI selection
    with open(paths["mri"]["annotation"], encoding="utf-8") as f:
        ann = json.load(f)
    info, cats = ann["Annotation_info"], ann["Category_info"]
    id_to_name = {v: k for k, v in cats.items() if isinstance(v, int)}
    if args.mri:
        mri_ids = args.mri
    else:
        classes = rng.sample(sorted(id_to_name), 5)
        mri_ids = [rng.choice(sorted(p for p, s in info.items()
                                     if s[0]["annotation"]["lesion"]["0"]["category"] == c)) for c in classes]

    out = {"seed": args.seed, "ct": [], "mri": []}
    for pid in ct_ids:
        print(f"[CT ] {pid} ...", flush=True)
        out["ct"].append(audit_ct_case(pid, by_pid[pid], data_dir, fig_dir))
    for pid in mri_ids:
        print(f"[MRI] {pid} ...", flush=True)
        out["mri"].append(audit_mri_case(pid, info[pid], id_to_name, paths["mri"], fig_dir))

    js = reports / "audit" / "samples.json"
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nWritten: {js}\nFigures: {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
