"""Stream-decompress source .nii.gz files to verify gzip integrity (CRC + length). Read-only, low memory.

The GĐ1 audit read headers only, so truncated archives (e.g. P0286_C2, found during the pilot export) went unnoticed.
The pilot export already reads every Train/Val file; this script covers the remaining split(s) (default: test).

Output: reports/audit/source_integrity_<splits>.csv and a printed summary.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import load_paths  # noqa: E402

MRI_SEQ_TO_FILE = {"In Phase": "InPhase", "Out Phase": "OutPhase"}


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def check(path: str) -> str:
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 22):
                pass
        return "ok"
    except FileNotFoundError:
        return "missing"
    except (EOFError, OSError, zlib.error) as e:
        return f"corrupt: {type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    paths = load_paths()

    jobs = []
    ct_split = {r["patient_id"]: r["split"] for r in read_csv(paths["project"]["splits"] / "ct_split.csv")}
    data_dir = Path(paths["ct"]["data_dir"])
    for r in read_csv(Path(paths["ct"]["csv"])):
        if ct_split[r["patient_id"]] in args.splits:
            for kind in ("ct_path", "mask_path", "liver_mask_path"):
                if r[kind] not in ("", "NA"):
                    jobs.append({"modality": "CT", "patient_id": r["patient_id"], "series": r["phase"],
                                 "split": ct_split[r["patient_id"]], "kind": kind, "path": str(data_dir / r[kind])})
    mri_split = {r["patient_id"]: r["split"] for r in read_csv(paths["project"]["splits"] / "mri_split.csv")}
    info = json.loads(Path(paths["mri"]["annotation"]).read_text(encoding="utf-8"))["Annotation_info"]
    for pid, series in info.items():
        if mri_split[pid] not in args.splits:
            continue
        for s in series:
            c = s["annotation"]["lesion"]["0"]["category"]
            f = MRI_SEQ_TO_FILE.get(s["phase"], s["phase"])
            for kind, p in (("image", Path(paths["mri"]["images"]) / f"{pid}_{c}_{f}_0000.nii.gz"),
                            ("label", Path(paths["mri"]["labels"]) / f"{pid}_{c}_{f}.nii.gz")):
                jobs.append({"modality": "MRI", "patient_id": pid, "series": s["phase"], "split": mri_split[pid],
                             "kind": kind, "path": str(p)})

    print(f"checking {len(jobs)} files in splits {args.splits} ...", flush=True)
    with ThreadPoolExecutor(args.threads) as ex:  # zlib releases the GIL
        for j, status in zip(jobs, ex.map(lambda j: check(j["path"]), jobs)):
            j["status"] = status
    out = paths["project"]["reports"] / "audit" / f"source_integrity_{'_'.join(args.splits)}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["modality", "patient_id", "series", "split", "kind", "status", "path"])
        w.writeheader()
        w.writerows(jobs)
    summary = Counter((j["modality"], j["status"].split(":")[0]) for j in jobs)
    print(dict(summary))
    for j in jobs:
        if j["status"] != "ok":
            print(" ", j["modality"], j["patient_id"], j["series"], j["kind"], j["status"])
    print("written", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
