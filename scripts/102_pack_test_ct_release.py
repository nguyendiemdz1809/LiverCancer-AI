"""Đóng gói dữ liệu Test CT cho notebook Colab 06 (phần D: đánh giá lại tập Test bằng trọng số đã khóa).

Người dùng quyết định (2026-11-25, theo yêu cầu GVHD) chia sẻ tập Test CT để người khác tự kiểm chứng số liệu Test của báo cáo.
Không huấn luyện, không chọn gì trên Test: notebook chỉ chạy lại đúng pipeline đã khóa.

Nội dung test_ct.zip (thư mục gốc `test_ct/`):
  CECT/ct_files/*.nii.gz      ảnh CT 3D GỐC của PLC-CECT (giữ nguyên từng byte) cho 206 volume dùng được của 54 BN Test
  CECT/test_volumes.csv       patient_id, phase, ct_path (đường dẫn tương đối so với CECT/, giống patient_data.csv)
  yolo_ct/                    lát có tổn thương của Test (data/processed/final_test/yolo_ct, do 96_final_test.py xuất) + data.yaml
  bao_cao_test_ct_patients.csv kết quả từng BN của lần chạy Test chính thức (reports/final_test/patients.csv, chỉ CT)

PLC-CECT: CC BY 4.0 (Luo et al., Sci Data 2025) -> được phân phối lại khi ghi nguồn. KHÔNG có MRI (CC BY-NC, cấm phân phối lại).
ZIP không nén (ảnh đã nén sẵn), thời điểm cố định để SHA256 tái lập được; cắt thành các phần < 2 GB (giới hạn GitHub Releases).

  python scripts/102_pack_test_ct_release.py [--out E:/LiverCancer_AI_GitHub_release/test_ct]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import load_paths  # noqa: E402
from livercancer.data.sources import ct_image_path  # noqa: E402

TEST = ROOT / "data" / "processed" / "final_test" / "yolo_ct"
PATIENTS = ROOT / "reports" / "final_test" / "patients.csv"
PART = 1_900_000_000
DT = (2026, 11, 25, 0, 0, 0)


def rows(p: Path) -> list[dict]:
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def csv_bytes(recs: list[dict], keys: list[str]) -> bytes:
    s = io.StringIO()
    w = csv.DictWriter(s, fieldnames=keys, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    w.writerows(recs)
    return s.getvalue().encode("utf-8")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="E:/LiverCancer_AI_GitHub_release/test_ct")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = load_paths()
    vols = [v for v in rows(TEST / "volumes.csv") if v["status"] == "ok"]
    assert {v["split"] for v in vols} == {"test"} and len(vols) == 206, len(vols)
    pats = {v["patient_id"] for v in rows(TEST / "volumes.csv")}
    assert len(pats) == 54
    data_dir = Path(paths["ct"]["data_dir"])
    entries = []   # (arcname, source path | bytes)
    vol_rows = []
    for v in sorted(vols, key=lambda r: (r["patient_id"], r["series"])):
        src = ct_image_path(paths, v["patient_id"], v["series"])
        rel = src.relative_to(data_dir).as_posix()
        assert rel.startswith("ct_files/") and src.exists(), src
        vol_rows.append({"patient_id": v["patient_id"], "phase": v["series"], "ct_path": rel})
        entries.append((f"test_ct/CECT/{rel}", src))
    entries.append(("test_ct/CECT/test_volumes.csv", csv_bytes(vol_rows, ["patient_id", "phase", "ct_path"])))
    for p in sorted(TEST.rglob("*")):
        if p.is_file():
            entries.append((f"test_ct/yolo_ct/{p.relative_to(TEST).as_posix()}", p))
    entries.append(("test_ct/yolo_ct/data.yaml", b"path: .\ntrain: images/test\nval: images/test\nnames:\n  0: lesion\n"))
    pr = [r for r in rows(PATIENTS) if r["modality"] == "ct"]
    assert len(pr) == 54 and {r["patient_id"] for r in pr} == pats
    entries.append(("test_ct/bao_cao_test_ct_patients.csv",
                    csv_bytes(pr, ["patient_id", "true_label", "status", "outcome", "pipeline_label", "oracle_label"])))

    z = out / "test_ct.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as f:
        for i, (arc, src) in enumerate(entries):
            info = zipfile.ZipInfo(arc, DT)
            info.external_attr = 0o644 << 16
            data = src if isinstance(src, bytes) else src.read_bytes()
            f.writestr(info, data)
            if i % 2000 == 0:
                print(f"  {i}/{len(entries)}", flush=True)
    size = z.stat().st_size
    full = sha256(z)
    lines, parts = [], []
    with open(z, "rb") as f:
        n = 1
        while chunk := f.read(PART):
            p = out / f"test_ct.zip.part{n}"
            p.write_bytes(chunk)
            parts.append(p)
            lines.append(f"{sha256(p)}  {p.name}")
            n += 1
    lines.append(f"{full}  test_ct.zip")
    (out / "SHA256SUMS_test_ct.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    z.unlink()
    print(f"test_ct.zip {size / 1e9:.2f} GB, {len(entries)} tệp, {len(parts)} phần, SHA256 {full}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
