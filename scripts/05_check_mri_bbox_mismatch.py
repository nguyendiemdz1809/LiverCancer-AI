"""GĐ1 follow-up: investigate MRI series whose JSON 3D_box disagrees with the MedSAM2 label mask.

Read-only on the dataset. Hypotheses tested for every series of each affected patient:
  H1 geometry   : image vs label affine/shape, JSON spacing vs header, JSON origin (LPS) vs affine (RAS), orientation
  H2 axis/order : 3D IoU(JSON box, mask bbox) under identity, flip i, flip j, flip k (slice order), swap i/j
  H3 2D boxes   : per-slice 2D_box z-range == 3D_box z-range, slice_idx strictly ordered/contiguous
  H4 location   : world-space (mm) centre of JSON box / mask vs the patient's other sequences
No bbox is modified. Series that no hypothesis explains are written to a quarantine list.

Outputs:
  reports/audit/mri_bbox_investigation.csv   one row per series of affected patients
  data/quarantine/mri_bbox_quarantine.csv    series excluded from YOLO / ROI training (IDs only)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import load_paths  # noqa: E402

MRI_SEQ_TO_FILE = {"In Phase": "InPhase", "Out Phase": "OutPhase"}
LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])


def iou3d(a: np.ndarray, b: np.ndarray) -> float:
    """Boxes as [i0, i1, j0, j1, k0, k1] with exclusive upper bounds."""
    inter = np.prod([max(0.0, min(a[2 * d + 1], b[2 * d + 1]) - max(a[2 * d], b[2 * d])) for d in range(3)])
    vol = lambda x: np.prod([x[2 * d + 1] - x[2 * d] for d in range(3)])  # noqa: E731
    u = vol(a) + vol(b) - inter
    return float(inter / u) if u > 0 else 0.0


def transforms(box: np.ndarray, shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    i0, i1, j0, j1, k0, k1 = box
    ni, nj, nk = shape
    return {
        "identity": box,
        "flip_i": np.array([ni - i1, ni - i0, j0, j1, k0, k1]),
        "flip_j": np.array([i0, i1, nj - j1, nj - j0, k0, k1]),
        "flip_k": np.array([i0, i1, j0, j1, nk - k1, nk - k0]),
        "swap_ij": np.array([j0, j1, i0, i1, k0, k1]),
    }


def world(aff: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    return aff[:3, :3] @ ijk + aff[:3, 3]


def analyse_series(pid: str, s: dict, cfg: dict) -> dict:
    les = s["annotation"]["lesion"]["0"]
    cat, seq = les["category"], s["phase"]
    f = MRI_SEQ_TO_FILE.get(seq, seq)
    img = nib.load(str(Path(cfg["images"]) / f"{pid}_{cat}_{f}_0000.nii.gz"))
    lab_img = nib.load(str(Path(cfg["labels"]) / f"{pid}_{cat}_{f}.nii.gz"))
    lab = np.asanyarray(lab_img.dataobj.get_unscaled()) > 0
    aff, shape = img.affine, tuple(img.shape[:3])
    out = {"patient_id": pid, "category_id": cat, "sequence": seq, "shape": "x".join(map(str, shape)),
           "orientation": "".join(nib.aff2axcodes(aff))}

    # H1 geometry
    out["label_affine_match"] = bool(np.allclose(aff, lab_img.affine, atol=1e-3)) and lab.shape == shape
    out["spacing_match_json"] = bool(np.allclose(img.header.get_zooms()[:3],
                                                 [*s["pixel_spacing"], s["slice_spacing"]], rtol=1e-3, atol=1e-3))
    out["origin_match_json_lps"] = bool(np.allclose(aff[:3, 3], np.array(s["origin"]) * LPS_TO_RAS, atol=0.05))

    # H2 axis conventions / slice order (3D)
    b3 = les["bbox"]["3D_box"]
    jbox = np.array([b3["x_min"], b3["x_max"], b3["y_min"], b3["y_max"], b3["z_min"], b3["z_max"] + 1], float)
    idx = np.nonzero(lab)
    mbox = np.array([idx[0].min(), idx[0].max() + 1, idx[1].min(), idx[1].max() + 1,
                     idx[2].min(), idx[2].max() + 1], float)
    ious = {k: round(iou3d(v, mbox), 4) for k, v in transforms(jbox, shape).items()}
    out.update({f"iou3d_{k}": v for k, v in ious.items()})
    out["best_transform"] = max(ious, key=ious.get)
    out["mask_voxels"] = int(lab.sum())
    out["json_box_voxels"] = round(float(np.prod(jbox[1::2] - jbox[::2])), 1)

    # H3 2D boxes
    ks = [int(d["slice_idx"]) for d in les["bbox"]["2D_box"]]
    out["2d_slices_sorted_contiguous"] = ks == list(range(min(ks), max(ks) + 1))
    out["2d_z_equals_3d_z"] = (min(ks), max(ks)) == (b3["z_min"], b3["z_max"])
    mask_ks = sorted(set(idx[2].tolist()))
    out["mask_slices_in_json_z"] = round(sum(b3["z_min"] <= k <= b3["z_max"] for k in mask_ks) / len(mask_ks), 3)

    # H4 world-space centres (mm)
    jc = np.array([(jbox[0] + jbox[1]) / 2, (jbox[2] + jbox[3]) / 2, (jbox[4] + jbox[5] - 1) / 2])
    mc = np.array([i.mean() for i in idx])
    out["_json_world"], out["_mask_world"] = world(aff, jc), world(aff, mc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iou-thr", type=float, default=0.3, help="JSON-vs-mask xy IoU below which a series is flagged")
    ap.add_argument("--dist-thr", type=float, default=20.0, help="mm; world-centre deviation counted as 'off'")
    args = ap.parse_args()

    paths = load_paths()
    root = paths["project"]["reports"].parent
    with open(paths["project"]["reports"] / "audit" / "mri_series.csv", encoding="utf-8") as f:
        audit = list(csv.DictReader(f))
    flagged = {(r["patient_id"], r["sequence"]) for r in audit
               if r.get("iou_json3d_vs_mask_xy") and float(r["iou_json3d_vs_mask_xy"]) < args.iou_thr}
    patients = sorted({p for p, _ in flagged})
    info = json.loads(Path(paths["mri"]["annotation"]).read_text(encoding="utf-8"))["Annotation_info"]

    rows = []
    for pid in patients:
        series = [analyse_series(pid, s, paths["mri"]) for s in info[pid]]
        good = [r for r in series if (pid, r["sequence"]) not in flagged]
        ref = np.median([r["_json_world"] for r in good] + [r["_mask_world"] for r in good], axis=0) if good else None
        for r in series:
            r["flagged_low_iou"] = (pid, r["sequence"]) in flagged
            if ref is not None:
                r["json_dist_to_ref_mm"] = round(float(np.linalg.norm(r["_json_world"] - ref)), 1)
                r["mask_dist_to_ref_mm"] = round(float(np.linalg.norm(r["_mask_world"] - ref)), 1)
            r["n_reference_series"] = len(good)
            del r["_json_world"], r["_mask_world"]
            rows.append(r)
        print(f"{pid}: {sum(r['flagged_low_iou'] for r in series)} flagged / {len(series)}", flush=True)

    out_csv = paths["project"]["reports"] / "audit" / "mri_bbox_investigation.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fl = [r for r in rows if r["flagged_low_iou"]]
    explained = [r for r in fl if r["best_transform"] != "identity"
                 and r[f"iou3d_{r['best_transform']}"] >= 0.5]
    geom_bad = [r for r in fl if not (r["label_affine_match"] and r["spacing_match_json"] and r["origin_match_json_lps"])]
    print(f"\nflagged series: {len(fl)} / patients {len({r['patient_id'] for r in fl})}")
    print(f"H1 geometry mismatch among flagged: {len(geom_bad)}")
    print(f"H2 best transform among flagged: {dict(Counter(r['best_transform'] for r in fl))}; "
          f"explained by a transform (IoU>=0.5): {len(explained)}")
    print(f"H3 2D slices contiguous: {dict(Counter(r['2d_slices_sorted_contiguous'] for r in fl))}, "
          f"2D z == 3D z: {dict(Counter(r['2d_z_equals_3d_z'] for r in fl))}")
    h4 = [r for r in fl if "json_dist_to_ref_mm" in r]
    print("H4 json off-ref (> thr mm):", sum(r["json_dist_to_ref_mm"] > args.dist_thr for r in h4),
          "| mask off-ref:", sum(r["mask_dist_to_ref_mm"] > args.dist_thr for r in h4), f"of {len(h4)}")
    print("    mask/json volume ratio (median):",
          round(float(np.median([r["mask_voxels"] / max(r["json_box_voxels"], 1) for r in fl])), 3))

    # Quarantine: every flagged series not explained by a verifiable geometric convention.
    q_dir = root / "data" / "quarantine"
    q_dir.mkdir(parents=True, exist_ok=True)
    quarantine = [r for r in fl if r not in explained]
    with open(q_dir / "mri_bbox_quarantine.csv", "w", newline="", encoding="utf-8") as f:
        f.write(f"# Generated by scripts/05_check_mri_bbox_mismatch.py (JSON-vs-mask xy IoU < {args.iou_thr}, "
                "not explained by orientation/affine/spacing/slice-order checks). Excluded from YOLO + ROI training.\n")
        w = csv.writer(f)
        w.writerow(["patient_id", "sequence", "category_id", "reason"])
        for r in quarantine:
            w.writerow([r["patient_id"], r["sequence"], r["category_id"], "json_box_vs_mask_mismatch_unverified"])
    print(f"\nquarantined: {len(quarantine)} series -> {q_dir / 'mri_bbox_quarantine.csv'}")
    print(f"investigation table -> {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
