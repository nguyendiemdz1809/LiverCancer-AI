"""Frozen-split integrity: no patient in two splits, full coverage, official MRI val respected, stratification."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import load_paths  # noqa: E402

SPLITS = ROOT / "data" / "splits"
pytestmark = pytest.mark.skipif(not (SPLITS / "ct_split.csv").exists(), reason="splits not generated yet")


def read(name: str) -> list[dict]:
    with open(SPLITS / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("name", ["ct_split.csv", "mri_split.csv"])
def test_one_split_per_patient(name):
    rows = read(name)
    ids = [r["patient_id"] for r in rows]
    assert len(ids) == len(set(ids)), "patient listed more than once"
    assert {r["split"] for r in rows} == {"train", "val", "test"}


def test_ct_covers_all_patients_with_source_labels():
    paths = load_paths()
    with open(paths["ct"]["csv"], encoding="utf-8") as f:
        src = {r["patient_id"]: r["cancer_type"] for r in csv.DictReader(f)}
    rows = read("ct_split.csv")
    assert {r["patient_id"]: r["label"] for r in rows} == src


def test_mri_covers_all_patients_and_official_val():
    paths = load_paths()
    ann = json.loads(Path(paths["mri"]["annotation"]).read_text(encoding="utf-8"))
    rows = read("mri_split.csv")
    assert {r["patient_id"] for r in rows} == set(ann["Annotation_info"])
    with open(ROOT / "data" / "external" / "lld_mmri2023_official_val_ids.csv", encoding="utf-8") as f:
        official = {r["patient_id"] for r in csv.DictReader(l for l in f if not l.startswith("#"))}
    assert {r["patient_id"] for r in rows if r["split"] == "val"} == official
    assert Counter(r["split"] for r in rows) == {"train": 316, "val": 78, "test": 104}


def test_ct_stratification_within_two_patients():
    rows = read("ct_split.csv")
    for lab in {r["label"] for r in rows}:
        n = sum(r["label"] == lab for r in rows)
        for split, frac in (("val", 0.15), ("test", 0.15)):
            got = sum(r["label"] == lab and r["split"] == split for r in rows)
            assert abs(got - n * frac) <= 2, (lab, split, got)


def test_manifest_consistent_with_split_files():
    man = read("split_manifest.csv")
    for mod, name in (("CT", "ct_split.csv"), ("MRI", "mri_split.csv")):
        a = {(r["patient_id"], r["split"]) for r in man if r["modality"] == mod}
        b = {(r["patient_id"], r["split"]) for r in read(name)}
        assert a == b
