"""Resolve source NIfTI image paths (read-only dataset) from (patient, phase/sequence), as in the GĐ2a export."""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

MRI_SEQ_TO_FILE = {"In Phase": "InPhase", "Out Phase": "OutPhase"}  # JSON name -> file name


@lru_cache(maxsize=2)
def _ct_index(csv_path: str) -> dict[tuple[str, str], str]:
    with open(csv_path, encoding="utf-8") as f:
        return {(r["patient_id"], r["phase"]): r["ct_path"] for r in csv.DictReader(l for l in f if not l.startswith("#"))}


@lru_cache(maxsize=2)
def _mri_categories(annotation: str) -> dict[tuple[str, str], str]:
    info = json.loads(Path(annotation).read_text(encoding="utf-8"))["Annotation_info"]
    return {(pid, s["phase"]): s["annotation"]["lesion"]["0"]["category"] for pid, series in info.items() for s in series}


def ct_image_path(paths: dict, patient_id: str, phase: str) -> Path:
    """CT phase volume, from patient_data.csv (paths relative to CECT/)."""
    return Path(paths["ct"]["data_dir"]) / _ct_index(str(paths["ct"]["csv"]))[(patient_id, phase)]


def mri_image_path(paths: dict, patient_id: str, sequence: str) -> Path:
    """MRI series volume; `sequence` uses the JSON names ('In Phase'), files use 'InPhase'."""
    cat = _mri_categories(str(paths["mri"]["annotation"]))[(patient_id, sequence)]
    f = MRI_SEQ_TO_FILE.get(sequence, sequence)
    return Path(paths["mri"]["images"]) / f"{patient_id}_{cat}_{f}_0000.nii.gz"


def image_path(paths: dict, modality: str, patient_id: str, series: str) -> Path:
    return (ct_image_path if modality.lower() == "ct" else mri_image_path)(paths, patient_id, series)


@lru_cache(maxsize=2)
def _ct_masks(csv_path: str) -> dict[tuple[str, str], str]:
    with open(csv_path, encoding="utf-8") as f:
        return {(r["patient_id"], r["phase"]): r["mask_path"] for r in csv.DictReader(l for l in f if not l.startswith("#"))}


def lesion_mask_path(paths: dict, modality: str, patient_id: str, series: str) -> Path | None:
    """GT lesion mask (analysis on Train/Val only; never an inference input). None for CT controls (mask_path NA)."""
    if modality.lower() == "ct":
        m = _ct_masks(str(paths["ct"]["csv"]))[(patient_id, series)]
        return None if m in ("", "NA") else Path(paths["ct"]["data_dir"]) / m
    cat = _mri_categories(str(paths["mri"]["annotation"]))[(patient_id, series)]
    return Path(paths["mri"]["labels"]) / f"{patient_id}_{cat}_{MRI_SEQ_TO_FILE.get(series, series)}.nii.gz"
