"""GĐ4: verify the classification ZIPs before upload (read-only on the source folders).

Checks per ZIP: SHA256 == SHA256SUMS_cls.txt; CRC of every entry (testzip); every manifest image is in the ZIP;
no Test rows / files; each patient in exactly one split and equal to the frozen split; labels equal to the split file;
entry count == source file count; byte-identical on a random sample. Code ZIP: imports + model build from the ZIP alone.
Output: data/processed/cls/verify_cls.json
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "cls"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_data_zip(mod: str, listed: dict) -> dict:
    zp, src = OUT / f"cls_{mod}.zip", OUT / mod
    r = {"zip": str(zp.relative_to(ROOT)), "bytes": zp.stat().st_size, "sha256": sha256(zp)}
    r["sha256_ok"] = r["sha256"] == listed.get(zp.name)
    with zipfile.ZipFile(zp) as z:
        r["testzip_first_bad"] = z.testzip()
        names = set(z.namelist())
        prefix = f"cls_{mod}/"
        man = list(csv.DictReader(io.TextIOWrapper(z.open(prefix + "manifest.csv"), encoding="utf-8")))
        missing = [x["roi_id"] for x in man for p in (x["local_path"], x["context_path"]) if prefix + p not in names]
        src_files = [p for p in src.rglob("*") if p.is_file() and not p.name.startswith("manifest_")]
        sample = random.Random(0).sample(src_files, min(300, len(src_files)))
        diff = [p.name for p in sample if z.read(prefix + p.relative_to(src).as_posix()) != p.read_bytes()]
    split = {x["patient_id"]: x for x in csv.DictReader(open(ROOT / "data" / "splits" / f"{mod}_split.csv", encoding="utf-8"))}
    per = {}
    for x in man:
        per.setdefault(x["patient_id"], set()).add(x["split"])
    r.update(entries=len(names), source_files=len(src_files), rows=len(man), missing_images=len(missing), bytes_mismatch=diff,
             has_test=any(x["split"] == "test" for x in man) or any("/test/" in n for n in names),
             patients_in_two_splits=[p for p, s in per.items() if len(s) > 1],
             split_mismatch=[x["roi_id"] for x in man if split[x["patient_id"]]["split"] != x["split"]][:5],
             label_mismatch=[x["roi_id"] for x in man if split[x["patient_id"]]["label"] != x["label"]][:5])
    r["ok"] = (r["sha256_ok"] and r["testzip_first_bad"] is None and not missing and not diff and not r["has_test"]
               and not r["patients_in_two_splits"] and not r["split_mismatch"] and not r["label_mismatch"]
               and r["entries"] == r["source_files"])
    return r


def check_code_zip(listed: dict) -> dict:
    zp = OUT / "livercancer_code.zip"
    r = {"zip": str(zp.relative_to(ROOT)), "bytes": zp.stat().st_size, "sha256": sha256(zp)}
    r["sha256_ok"] = r["sha256"] == listed.get(zp.name)
    with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(zp) as z:
        z.extractall(td)
        prog = ("import sys; sys.path.insert(0, 'src'); import yaml; "
                "from livercancer.classification.models import build_model; from livercancer.classification.train import train; "
                "from livercancer.classification.evaluate import evaluate_val; from livercancer.classification.inference import LesionClassifier; "
                "c = yaml.safe_load(open('configs/cls.yaml', encoding='utf-8')); "
                "[build_model(m, len(c['ct']['classes']), pretrained=False) for m in ('resnet18', 'swin_t', 'hybrid')]; print('IMPORT-OK')")
        p = subprocess.run([sys.executable, "-I", "-c", prog], cwd=td, capture_output=True, text=True)
        r["import_and_build_from_zip"] = "IMPORT-OK" in p.stdout
        r["stderr_tail"] = p.stderr[-300:] if p.returncode else ""
    r["ok"] = r["sha256_ok"] and r["import_and_build_from_zip"]
    return r


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    listed = {n: h for h, n in (l.split() for l in (OUT / "SHA256SUMS_cls.txt").read_text().splitlines() if l.strip())}
    res = {m: check_data_zip(m, listed) for m in ("ct", "mri")}
    res["code"] = check_code_zip(listed)
    res["ok"] = all(v["ok"] for v in res.values() if isinstance(v, dict))
    (OUT / "verify_cls.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in res.items():
        if isinstance(v, dict):
            print(k, {kk: vv for kk, vv in v.items() if kk not in ("stderr_tail",)})
    print("ALL OK" if res["ok"] else "FAILED")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
