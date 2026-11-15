"""Copy of the locked TEST set (3D NIfTI volumes) organised for manual website testing -> TEST_3D_WEBSITE/.

Read-only on the dataset: files are COPIED (never moved / modified). Test patients come from the frozen splits
(data/splits/{ct,mri}_split.csv), file paths from the source metadata (CT patient_data.csv, MRI LLD_MMRI_Annotation.json)
via livercancer.data.sources - never guessed from file names. Every copy is checked: SHA256 equal to the source,
NIfTI header readable, gzip stream fully decompressible.

Layout:  TEST_3D_WEBSITE/CT/<patient>/<patient>_ct_<phase>.nii.gz          (all phases on disk)
         TEST_3D_WEBSITE/MRI/<patient>/<patient>_<sequence>.nii.gz        (all sequences; the source name holds the
                                                                           lesion category id, removed here)
         TEST_3D_WEBSITE/TEST_NHANH/{CT,MRI}/<patient>/...                (hard links to the copies: no extra space)
         DANH_SACH_TEST_3D.csv, BAO_CAO_TEST_3D.md
The quick set is chosen WITHOUT looking at any model output: first patients in ID order per class with every
phase / sequence on disk (CT: with a lesion mask; MRI: not in the bbox quarantine).
These files are the locked Test set: using them on the website is a demonstration, never a tuning or a new result.
Folder is git-ignored. MRI is CC BY-NC 4.0 ("not to further copy, publish, or distribute").

    .venv\\Scripts\\python scripts\\99_build_test3d_website.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import load_paths  # noqa: E402
from livercancer.data.sources import MRI_SEQ_TO_FILE, image_path  # noqa: E402

DEST = ROOT / "TEST_3D_WEBSITE"
CT_PHASES = ("C1", "C2", "C3", "P")
MRI_QUICK = ("Hepatocellular_carcinoma", "Intrahepatic_cholangiocarcinoma", "Hepatic_metastasis", "Hepatic_hemangioma", "Hepatic_cyst")
CT_QUICK = (("HCC", 2), ("ICC", 2), ("CHCC", 1))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def gzip_ok(p: Path) -> bool:
    try:
        with gzip.open(p, "rb") as f:
            while f.read(1 << 24):
                pass
        return True
    except Exception:  # noqa: BLE001
        return False


def header_info(p: Path) -> tuple[bool, str, str]:
    try:
        h = nib.load(str(p)).header
        shape = "x".join(str(int(s)) for s in h.get_data_shape())
        texts = []
        for k in ("descrip", "aux_file", "db_name", "intent_name"):
            if k in h:
                v = h[k]
                v = v.item() if hasattr(v, "item") else v
                v = v.decode("latin-1", "ignore") if isinstance(v, bytes) else str(v)
                if v.strip("\x00 ").strip():
                    texts.append(f"{k}={v.strip(chr(0)).strip()}")
        return True, shape, "; ".join(texts)
    except Exception:  # noqa: BLE001
        return False, "", ""


def split_rows(mod: str) -> dict[str, dict]:
    with open(ROOT / "data" / "splits" / f"{mod}_split.csv", encoding="utf-8") as f:
        return {r["patient_id"]: r for r in csv.DictReader(f)}


def train_val_image_ids() -> set[str]:
    """Patients present in the 85,586 Train + Validation images (exported slice tables of the two trained archives)."""
    ids = set()
    for ds in ("full/yolo_ct", "pilot/yolo_mri"):
        p = ROOT / "data" / "processed" / ds / "slices.csv"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                ids |= {r["patient_id"] for r in csv.DictReader(f)}
    return ids


def plan(paths) -> list[dict]:
    rows = []
    ct = split_rows("ct")
    with open(paths["ct"]["csv"], encoding="utf-8") as f:
        src = [r for r in csv.DictReader(l for l in f if not l.startswith("#"))]
    by_pat = defaultdict(dict)
    for r in src:
        by_pat[r["patient_id"]][r["phase"]] = r
    for pid in sorted(p for p, r in ct.items() if r["split"] == "test"):
        for ph in CT_PHASES:
            r = by_pat[pid].get(ph)
            if r is None:
                rows.append({"modality": "CT", "patient_id": pid, "label": ct[pid]["label"], "series": ph, "status": "khong_co_trong_csv"})
                continue
            s = image_path(paths, "ct", pid, ph)
            m = None if r["mask_path"] in ("", "NA") else Path(paths["ct"]["data_dir"]) / r["mask_path"]
            rows.append({"modality": "CT", "patient_id": pid, "label": ct[pid]["label"], "series": ph, "source": str(s),
                         "dest": str(DEST / "CT" / pid / s.name), "lesion_mask": "co" if (m and m.exists()) else "khong",
                         "status": "ok" if s.exists() else "thieu_tren_dia"})
    mri = split_rows("mri")
    ann = json.load(open(paths["mri"]["annotation"], encoding="utf-8"))["Annotation_info"]
    quarantine = set()
    qp = ROOT / "data" / "quarantine" / "mri_bbox_quarantine.csv"
    if qp.exists():
        with open(qp, encoding="utf-8") as f:
            quarantine = {(r["patient_id"], r["sequence"]) for r in csv.DictReader(l for l in f if not l.startswith("#"))}
    for pid in sorted(p for p, r in mri.items() if r["split"] == "test"):
        for ser in ann.get(pid, []):
            seq = ser["phase"]
            s = image_path(paths, "mri", pid, seq)
            name = f"{pid}_{MRI_SEQ_TO_FILE.get(seq, seq)}.nii.gz"
            rows.append({"modality": "MRI", "patient_id": pid, "label": mri[pid]["label"], "series": seq, "source": str(s),
                         "dest": str(DEST / "MRI" / pid / name), "quarantine": "co" if (pid, seq) in quarantine else "khong",
                         "status": "ok" if s.exists() else "thieu_tren_dia"})
    return rows


def quick(rows) -> set[str]:
    pats = defaultdict(list)
    for r in rows:
        pats[(r["modality"], r["patient_id"])].append(r)
    chosen = []
    for lab, n in CT_QUICK:
        ok = [pid for (m, pid), rs in sorted(pats.items()) if m == "CT" and rs[0]["label"] == lab
              and len(rs) == 4 and all(r["status"] == "ok" and r.get("lesion_mask") == "co" for r in rs)]
        chosen += ok[:n]
    for lab in MRI_QUICK:
        ok = [pid for (m, pid), rs in sorted(pats.items()) if m == "MRI" and rs[0]["label"] == lab
              and len(rs) == 8 and all(r["status"] == "ok" and r.get("quarantine") == "khong" for r in rs)]
        chosen += ok[:1]
    return set(chosen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    paths = load_paths()
    t0 = time.time()
    rows = plan(paths)
    # no Train / Validation patient may enter: split membership and the 85,586 training images
    tv = train_val_image_ids()
    leak = sorted({r["patient_id"] for r in rows} & tv)
    other = {m: {p for p, r in split_rows(m).items() if r["split"] != "test"} for m in ("ct", "mri")}
    leak += sorted({r["patient_id"] for r in rows if r["patient_id"] in other[r["modality"].lower()]})
    if leak:
        raise SystemExit(f"STOP: Train/Validation patient(s) in the Test list: {leak}")
    for r in rows:
        if r["status"] == "ok":
            r["size_bytes"] = os.path.getsize(r["source"])
    total = sum(r.get("size_bytes", 0) for r in rows)
    free = shutil.disk_usage(ROOT).free
    print(f"files {len(rows)} ok {sum(r['status'] == 'ok' for r in rows)} size {total / 2**30:.2f} GB, free {free / 2**30:.1f} GB", flush=True)
    full_copy = total < free - 10 * 2**30                     # keep 10 GB margin
    q = quick(rows)
    if not full_copy:
        print("Not enough space: only the quick set is copied", flush=True)
    if args.dry_run:
        print(Counter((r["modality"], r["status"]) for r in rows), sorted(q))
        return 0
    copied = skipped = 0
    for i, r in enumerate(rows):
        r["quick_test"] = "co" if r["patient_id"] in q else "khong"
        if r["status"] != "ok":
            continue
        src, dst = Path(r["source"]), Path(r["dest"])
        if not full_copy and r["quick_test"] != "co":
            r["copied"] = "khong (thieu dung luong; dung duong dan goc)"
            continue
        r["sha256_source"] = sha256(src)
        if dst.exists() and sha256(dst) == r["sha256_source"]:
            skipped += 1                                       # already copied identically: no duplicate copy
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        r["sha256_copy"] = sha256(dst)
        r["sha_match"] = "co" if r["sha256_copy"] == r["sha256_source"] else "KHONG"
        ok, shape, texts = header_info(dst)
        r["header_ok"], r["shape"], r["header_text"] = ("co" if ok else "KHONG"), shape, texts
        r["gzip_ok"] = "co" if gzip_ok(dst) else "KHONG"
        r["copied"] = "co"
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    # quick set: hard links to the copies (same files, no extra space)
    for r in rows:
        if r.get("quick_test") == "co" and r.get("copied") == "co":
            d = Path(r["dest"])
            link = DEST / "TEST_NHANH" / r["modality"] / r["patient_id"] / d.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                try:
                    os.link(d, link)
                except OSError:
                    shutil.copy2(d, link)
            r["quick_path"] = str(link)
    keys = ["modality", "patient_id", "label", "series", "status", "quick_test", "copied", "dest", "quick_path", "source", "size_bytes",
            "shape", "sha_match", "header_ok", "gzip_ok", "header_text", "lesion_mask", "quarantine", "sha256_source"]
    DEST.mkdir(parents=True, exist_ok=True)
    with open(DEST / "DANH_SACH_TEST_3D.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    report(rows, copied, skipped, total, time.time() - t0)
    return 0


def size_of(p: Path) -> int:
    seen, n = set(), 0
    for f in p.rglob("*"):
        if f.is_file():
            st = f.stat()
            key = (st.st_dev, st.st_ino)
            if key not in seen:                                  # hard links counted once
                seen.add(key)
                n += st.st_size
    return n


def report(rows, copied, skipped, total, secs) -> None:
    def pats(m, cond=lambda r: True):
        return sorted({r["patient_id"] for r in rows if r["modality"] == m and cond(r)})
    miss = [r for r in rows if r["status"] != "ok"]
    bad = [r for r in rows if r.get("copied") == "co" and not (r["sha_match"] == r["header_ok"] == r["gzip_ok"] == "co")]
    hdr = Counter(r.get("header_text") or "(trống)" for r in rows if r.get("copied") == "co")
    ct_full = [p for p in pats("CT") if sum(1 for r in rows if r["patient_id"] == p and r["status"] == "ok") == 4]
    lines = ["# Bộ Test 3D để kiểm thử website", "",
             f"Tạo bởi `scripts/99_build_test3d_website.py` lúc {time.strftime('%Y-%m-%d %H:%M')} ({secs / 60:.1f} phút). "
             "Danh sách chi tiết từng file: `DANH_SACH_TEST_3D.csv`.", "",
             "**Đây là tập Test đã khóa** (đánh giá chính thức một lần ngày 2026-10-26). Dùng để trình diễn website, "
             "**không** dùng để chỉnh mô hình, ngưỡng hay báo cáo thành kết quả mới.", "",
             "## Nguồn", "",
             "- Danh sách bệnh nhân Test: `data/splits/ct_split.csv`, `data/splits/mri_split.csv` (split đóng băng theo bệnh nhân).",
             "- File CT: `E:\\LiverDataset\\CT\\PLC-CECT\\CECT\\ct_files` (đường dẫn lấy từ `patient_data.csv`).",
             "- File MRI: `E:\\LiverDataset\\LLD-MMRI-MedSAM2\\images` (chuỗi lấy từ `LLD_MMRI_Annotation.json`).",
             "- Đã đối chiếu: không bệnh nhân Test nào có trong split Train/Validation hay trong bảng lát của bộ 85.586 ảnh huấn luyện.", "",
             "## Số lượng", "", "| | CT | MRI |", "|---|---|---|",
             f"| Bệnh nhân Test | {len(pats('CT'))} | {len(pats('MRI'))} |",
             f"| File volume 3D theo metadata | {sum(r['modality'] == 'CT' for r in rows)} | {sum(r['modality'] == 'MRI' for r in rows)} |",
             f"| File có trên đĩa | {sum(r['modality'] == 'CT' and r['status'] == 'ok' for r in rows)} | {sum(r['modality'] == 'MRI' and r['status'] == 'ok' for r in rows)} |",
             f"| File đã sao chép và kiểm tra | {sum(r['modality'] == 'CT' and r.get('copied') == 'co' for r in rows)} | {sum(r['modality'] == 'MRI' and r.get('copied') == 'co' for r in rows)} |",
             f"| Bệnh nhân đủ pha/chuỗi | {len(ct_full)} (đủ 4 pha) | {sum(1 for p in pats('MRI') if sum(1 for r in rows if r['patient_id'] == p and r['status'] == 'ok') == 8)} (đủ 8 chuỗi) |",
             "", f"Dung lượng nguồn: {total / 2**30:.2f} GB. Lần chạy này sao chép mới {copied} file, bỏ qua {skipped} file đã có bản giống hệt. "
             f"Dung lượng thư mục `TEST_3D_WEBSITE`: {size_of(DEST) / 2**30:.2f} GB (TEST_NHANH là hard link, không tốn thêm).", "",
             "## File thiếu hoặc lỗi", ""]
    if miss:
        for r in miss:
            lines.append(f"- {r['modality']} {r['patient_id']} {r['series']}: {'không có trên đĩa' if r['status'] == 'thieu_tren_dia' else r['status']}"
                         f" (nhãn {r['label']})")
    else:
        lines.append("- Không có file thiếu.")
    lines.append(f"- File sao chép lỗi (SHA256 / header / gzip): {len(bad)}" + (": " + ", ".join(Path(r['dest']).name for r in bad) if bad else "."))
    lines += ["", "## TEST NHANH (5 CT + 5 MRI)", "",
              "Chọn theo thứ tự mã bệnh nhân trong từng nhóm, **không** dựa vào kết quả của mô hình: CT 2 HCC, 2 ICC, 1 CHCC "
              "(đủ 4 pha, có mask tổn thương); MRI 1 ca mỗi nhóm HCC, ICC, di căn, u máu, nang (đủ 8 chuỗi, không bị cách ly bbox). "
              "Mọi chẩn đoán ung thư CT có xác nhận mô học/tế bào học (nguồn dữ liệu).", "",
              "| Loại | Bệnh nhân | Nhãn tham chiếu | Thư mục |", "|---|---|---|---|"]
    for r in sorted({(r["modality"], r["patient_id"], r["label"]) for r in rows if r.get("quick_test") == "co"}):
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | `TEST_3D_WEBSITE\\TEST_NHANH\\{r[0]}\\{r[1]}` |")
    lines += ["", "## Cách tải lên website", "",
              "- **CT:** vào `CT\\<mã BN>\\`, chọn **Phân tích ca 3D** → **CT Scan**, rồi chọn 1 file (1 pha) hoặc cả 4 file `_ct_C1/C2/C3/P` "
              "(4 pha của cùng một bệnh nhân; website phân tích riêng từng pha và gộp kết quả theo ca).",
              "- **MRI:** vào `MRI\\<mã BN>\\`, chọn **Phân tích ca 3D** → **MRI**, rồi chọn 1 hoặc nhiều chuỗi (tối đa 8) của cùng một bệnh nhân.",
              "- Không chọn file của hai bệnh nhân khác nhau trong một lần tải.",
              "- Dữ liệu gốc là NIfTI; bộ dữ liệu không có DICOM nên không tạo DICOM (website vẫn nhận NIfTI trực tiếp).", "",
              "## Thông tin cá nhân và quyền sử dụng", "",
              "- Hai bộ dữ liệu là dữ liệu nghiên cứu công khai đã ẩn danh; mã bệnh nhân (P####, MR-######) là mã của bộ dữ liệu.",
              "- Trường văn bản trong header NIfTI (descrip, aux_file, db_name, intent_name) của các bản sao: "
              + "; ".join(f"`{k}` ({v} file)" for k, v in hdr.most_common(6)) + ".",
              "- **CT (PLC-CECT, CC BY 4.0):** được phép dùng và chia sẻ, ghi nguồn.",
              "- **MRI (LLD-MMRI, CC BY-NC 4.0, \"not to further copy, publish, or distribute\"):** chỉ dùng phi thương mại; "
              "**không được phân phối lại / đăng công khai**. Tải lên website để máy chủ của nhóm phân tích là xử lý tạm thời "
              "(không lưu), nhưng dữ liệu đi qua Cloudflare; để chắc chắn, kiểm thử MRI trên `http://127.0.0.1:8000` khi có thể. "
              "Không chia sẻ thư mục `MRI` cho người khác.",
              "- Thư mục `TEST_3D_WEBSITE/` được loại khỏi Git (`.gitignore`). Không upload thư mục này lên Internet.", ""]
    (DEST / "BAO_CAO_TEST_3D.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    sys.exit(main())
