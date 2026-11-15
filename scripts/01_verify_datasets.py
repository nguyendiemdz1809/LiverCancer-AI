"""Read-only audit of the CT (PLC-CECT) and MRI (LLD-MMRI-MedSAM2) datasets.

Checks that paths exist, cross-checks CSV/JSON against files on disk and writes
`reports/dataset_audit.json`. Never writes under the dataset roots and never loads
image volumes into memory (optional `--headers N` reads only NIfTI headers).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import load_paths  # noqa: E402

MRI_SEQ_TO_FILE = {"In Phase": "InPhase", "Out Phase": "OutPhase"}


def audit_ct(cfg: dict) -> dict:
    data_dir = Path(cfg["data_dir"])
    with open(cfg["csv"], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    labels_per_patient = defaultdict(set)
    for r in rows:
        labels_per_patient[r["patient_id"]].add(r["cancer_type"])
    inconsistent = sorted(p for p, s in labels_per_patient.items() if len(s) > 1)
    patient_labels = Counter(next(iter(s)) for s in labels_per_patient.values())

    missing, empty_mask = defaultdict(list), Counter()
    for r in rows:
        for key in ("ct_path", "mask_path", "liver_mask_path"):
            if r[key] in ("", "NA"):
                empty_mask[f"{key}:{r['cancer_type']}"] += 1
            elif not (data_dir / r[key]).exists():
                missing[key].append({"patient_id": r["patient_id"], "phase": r["phase"],
                                     "cancer_type": r["cancer_type"], "path": r[key]})

    extra = {}
    for folder, key in (("ct_files", "ct_path"), ("mask_files", "mask_path"),
                        ("liver_mask_files", "liver_mask_path")):
        listed = {Path(r[key]).name for r in rows if r[key] not in ("", "NA")}
        on_disk = {p.name for p in (data_dir / folder).iterdir() if p.is_file()}
        extra[folder] = {"n_files": len(on_disk), "not_in_csv": sorted(on_disk - listed)}

    return {
        "paths_exist": {k: Path(v).exists() for k, v in cfg.items()},
        "n_rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "n_patients": len(labels_per_patient),
        "patients_with_inconsistent_label": inconsistent,
        "patient_label_counts": dict(patient_labels),
        "row_label_counts": dict(Counter(r["cancer_type"] for r in rows)),
        "phase_counts": dict(Counter(r["phase"] for r in rows)),
        "empty_path_counts": dict(empty_mask),
        "missing_files": {k: {"count": len(v), "by_label": dict(Counter(x["cancer_type"] for x in v)),
                              "items": v} for k, v in missing.items()},
        "folders": extra,
    }


def audit_mri(cfg: dict) -> dict:
    with open(cfg["annotation"], encoding="utf-8") as f:
        ann = json.load(f)
    info, cats = ann["Annotation_info"], ann["Category_info"]
    id_to_name = {v: k for k, v in cats.items() if isinstance(v, int)}

    images = {p.name for p in Path(cfg["images"]).iterdir() if p.is_file()}
    labels = {p.name for p in Path(cfg["labels"]).iterdir() if p.is_file()}

    patient_cat, seq_counts, num_targets = {}, Counter(), Counter()
    inconsistent, missing_img, missing_lbl = [], [], []
    for pid, series in info.items():
        pcats = set()
        for s in series:
            seq_counts[s["phase"]] += 1
            num_targets[s["annotation"]["num_targets"]] += 1
            scats = {l["category"] for l in s["annotation"]["lesion"].values()}
            pcats |= scats
            seq = MRI_SEQ_TO_FILE.get(s["phase"], s["phase"])
            for c in scats:
                if f"{pid}_{c}_{seq}_0000.nii.gz" not in images:
                    missing_img.append(f"{pid}_{c}_{seq}")
                if f"{pid}_{c}_{seq}.nii.gz" not in labels:
                    missing_lbl.append(f"{pid}_{c}_{seq}")
        if len(pcats) != 1:
            inconsistent.append(pid)
        patient_cat[pid] = sorted(pcats)

    img_ids = Counter(n.split("_")[0] for n in images)
    return {
        "paths_exist": {k: Path(v).exists() for k, v in cfg.items()},
        "category_info": cats,
        "n_patients_json": len(info),
        "n_images": len(images),
        "n_labels": len(labels),
        "series_per_patient": dict(Counter(len(s) for s in info.values())),
        "sequence_counts": dict(seq_counts),
        "num_targets_counts": {str(k): v for k, v in num_targets.items()},
        "patients_with_multiple_categories": inconsistent,
        "patient_label_counts": dict(Counter(id_to_name[c[0]] for c in patient_cat.values() if c)),
        "json_ids_without_images": sorted(set(info) - set(img_ids)),
        "image_ids_without_json": sorted(set(img_ids) - set(info)),
        "missing_image_files": missing_img,
        "missing_label_files": missing_lbl,
    }


def sample_headers(paths: list[Path], n: int, seed: int = 0) -> list[dict]:
    """Read only NIfTI headers (nibabel loads lazily) for a random sample of files."""
    import nibabel as nib

    out = []
    for p in random.Random(seed).sample(paths, min(n, len(paths))):
        img = nib.load(str(p))
        out.append({"file": p.name, "shape": list(img.shape),
                    "zooms": [round(float(z), 4) for z in img.header.get_zooms()],
                    "orientation": "".join(nib.aff2axcodes(img.affine)),
                    "dtype": str(img.get_data_dtype())})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headers", type=int, default=0, help="sample N NIfTI headers per modality")
    ap.add_argument("--out", default=None, help="output JSON (default reports/dataset_audit.json)")
    args = ap.parse_args()

    paths = load_paths()
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "ct": audit_ct(paths["ct"]), "mri": audit_mri(paths["mri"])}

    if args.headers:
        ct_dir = Path(paths["ct"]["data_dir"])
        report["ct"]["header_sample"] = sample_headers(sorted((ct_dir / "ct_files").glob("*.nii.gz")), args.headers)
        report["mri"]["header_sample"] = sample_headers(sorted(Path(paths["mri"]["images"]).glob("*.nii.gz")), args.headers)

    out = Path(args.out) if args.out else paths["project"]["reports"] / "dataset_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    ct, mri = report["ct"], report["mri"]
    print("== CT (PLC-CECT) ==")
    print(f"  paths exist      : {ct['paths_exist']}")
    print(f"  rows / patients  : {ct['n_rows']} / {ct['n_patients']}")
    print(f"  patient labels   : {ct['patient_label_counts']}")
    print(f"  empty paths      : {ct['empty_path_counts']}")
    for k, v in ct["missing_files"].items():
        print(f"  missing {k:15s}: {v['count']} {v['by_label']}")
    for k, v in ct["folders"].items():
        print(f"  {k:17s}: {v['n_files']} files, {len(v['not_in_csv'])} not in CSV")
    print("== MRI (LLD-MMRI-MedSAM2) ==")
    print(f"  paths exist      : {mri['paths_exist']}")
    print(f"  patients / images / labels: {mri['n_patients_json']} / {mri['n_images']} / {mri['n_labels']}")
    print(f"  sequences        : {mri['sequence_counts']}")
    print(f"  patient labels   : {mri['patient_label_counts']}")
    print(f"  missing img/lbl  : {len(mri['missing_image_files'])} / {len(mri['missing_label_files'])}")
    print(f"  id mismatch json<->images: {len(mri['json_ids_without_images'])} / {len(mri['image_ids_without_json'])}")
    for mod in ("ct", "mri"):
        for h in report[mod].get("header_sample", []):
            print(f"  [{mod} header] {h}")
    print(f"\nReport written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
