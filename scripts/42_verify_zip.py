"""Verify a packaged dataset ZIP before upload: checksum, full extraction, file list/size and byte equality.

  1. SHA256 of the ZIP == entry in the SHA256SUMS*.txt next to it
  2. zipfile.testzip() (CRC of every member)
  3. full extraction into a temporary directory (same drive), then:
     - extracted file list == source dataset dir (relative paths), sizes equal for every file
     - byte-for-byte equality on a random sample of files
  4. temporary extraction removed
Read-only on the source dataset dir and the ZIP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True, help="dataset dir that was zipped (e.g. data/processed/full/yolo_ct)")
    ap.add_argument("--sums", type=Path, required=True, help="SHA256SUMS*.txt listing the zip")
    ap.add_argument("--n-bytes-check", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = {"zip": str(args.zip), "bytes": args.zip.stat().st_size}
    listed = dict(reversed(l.split()) for l in args.sums.read_text(encoding="utf-8").splitlines() if l.strip())
    res["sha256"] = sha256(args.zip)
    res["sha256_listed"] = listed.get(args.zip.name)
    res["sha256_ok"] = res["sha256"] == res["sha256_listed"]
    print("sha256", res["sha256"], "OK" if res["sha256_ok"] else f"MISMATCH (listed {res['sha256_listed']})", flush=True)

    with zipfile.ZipFile(args.zip) as z:
        res["testzip_first_bad"] = z.testzip()
        names = z.namelist()
    root = names[0].split("/")[0]
    res["entries"], res["root_folder"] = len(names), root
    print("testzip", "OK" if res["testzip_first_bad"] is None else res["testzip_first_bad"], "|", len(names), "entries", flush=True)

    tmp = args.zip.parent / f"_verify_{args.zip.stem}"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(tmp)
        ext = tmp / root
        src_files = {p.relative_to(args.source).as_posix(): p for p in args.source.rglob("*") if p.is_file()}
        ext_files = {p.relative_to(ext).as_posix(): p for p in ext.rglob("*") if p.is_file()}
        res["extracted_files"], res["source_files"] = len(ext_files), len(src_files)
        res["missing_in_zip"] = sorted(set(src_files) - set(ext_files))[:10]
        res["extra_in_zip"] = sorted(set(ext_files) - set(src_files))[:10]
        res["size_mismatch"] = [k for k in set(src_files) & set(ext_files)
                                if src_files[k].stat().st_size != ext_files[k].stat().st_size][:10]
        sample = random.Random(args.seed).sample(sorted(set(src_files) & set(ext_files)),
                                                 min(args.n_bytes_check, len(ext_files)))
        res["bytes_checked"] = len(sample)
        res["bytes_mismatch"] = [k for k in sample if src_files[k].read_bytes() != ext_files[k].read_bytes()][:10]
        res["n_images"] = sum(1 for k in ext_files if k.startswith("images/") and k.endswith(".png"))
        res["n_labels"] = sum(1 for k in ext_files if k.startswith("labels/") and k.endswith(".txt"))
        res["has_test_split"] = any(k.startswith(("images/test", "labels/test")) for k in ext_files)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res["temp_removed"] = not tmp.exists()

    ok = (res["sha256_ok"] and res["testzip_first_bad"] is None and not res["missing_in_zip"] and not res["extra_in_zip"]
          and not res["size_mismatch"] and not res["bytes_mismatch"] and not res["has_test_split"]
          and res["extracted_files"] == res["source_files"] == res["entries"])
    res["ok"] = ok
    out = args.zip.with_suffix(".verify.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("zip",)}, indent=1))
    print("VERIFY:", "PASS" if ok else "FAIL", "->", out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
