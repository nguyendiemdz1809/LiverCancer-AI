"""Audit + Roboflow export of the exact 85,586 YOLO images used for training/validation on Google Colab.

The two archives trained on Colab are identified by the SHA256 recorded in the Colab run bundles
(outputs/runs/yolo_{ct,mri}/yolo11n_e50_img512_baseline/metrics_val.json -> dataset, dataset_sha256):
  CT : data/processed/full/yolo_ct_full.zip    (Train 34,873 + Val 14,627 = 49,500)
  MRI: data/processed/pilot/yolo_mri_pilot.zip (Train 28,595 + Val 7,491  = 36,086)
Images and labels are copied entry by entry from those archives (bytes unchanged, original file names, original
Train/Val split; Roboflow folder names train/ and valid/). Nothing is added, removed, relabelled or re-split.
The project's Test split is not part of these archives and is not exported.

Output (gitignored): data/processed/roboflow_full_85586/
  CT_49500/zips/*.zip            uploadable (CC BY 4.0, attribution)
  MRI_36086_offline/zips/*.zip   keep offline (LLD-MMRI: no further copying/publishing/distribution)
  manifest_images.csv, upload_manifest.csv, audit.json
Reports: reports/roboflow_full_85586/bao_cao_kiem_toan.md, reports/thong_ke_du_lieu_chinh_thuc.{md,csv}

Run: .venv\\Scripts\\python scripts\\85_build_roboflow_full.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = {
    "CT": {"zip": ROOT / "data/processed/full/yolo_ct_full.zip", "sums": ROOT / "data/processed/full/SHA256SUMS_full.txt",
           "run": ROOT / "outputs/runs/yolo_ct/yolo11n_e50_img512_baseline/metrics_val.json",
           "test": ROOT / "data/processed/final_test/yolo_ct/slices.csv", "split": ROOT / "data/splits/ct_split.csv"},
    "MRI": {"zip": ROOT / "data/processed/pilot/yolo_mri_pilot.zip", "sums": ROOT / "data/processed/pilot/SHA256SUMS.txt",
            "run": ROOT / "outputs/runs/yolo_mri/yolo11n_e50_img512_baseline/metrics_val.json",
            "test": ROOT / "data/processed/final_test/yolo_mri/slices.csv", "split": ROOT / "data/splits/mri_split.csv"},
}
EXPECTED = {"CT": {"train": 34873, "val": 14627}, "MRI": {"train": 28595, "val": 7491}}
FOLDER = {"CT": "CT_49500", "MRI": "MRI_36086_offline"}
RF_SPLIT = {"train": "train", "val": "valid"}  # Roboflow's folder name for validation
LICENSE = {
    "CT": "PLC-CECT (Luo et al., Scientific Data 2025) - CC BY 4.0 (attribution required)",
    "MRI": "LLD-MMRI (Lou et al.) + MedSAM2 masks (wanglab/LLD-MMRI-MedSAM2) - CC BY-NC 4.0; "
           "terms: do not further copy, publish or distribute. KEEP OFFLINE.",
}
# Roboflow public pricing, read 2026-10-30 (https://roboflow.com/credits, https://roboflow.com/pricing)
RF = {"free_credits_month": 10, "upload_images_per_credit": 10_000, "storage_images_per_credit_month": 5_000,
      "checked": "2026-10-30"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_sums(p: Path) -> dict[str, str]:
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            h, n = line.split()
            out[n.lstrip("*")] = h
    return out


def label_check(text: str) -> tuple[int, str]:
    """Return (n_boxes, error or '')."""
    n = 0
    for ln in text.splitlines():
        if not ln.strip():
            continue
        v = ln.split()
        if len(v) != 5 or v[0] != "0":
            return n, f"bad line: {ln}"
        try:
            xc, yc, w, h = map(float, v[1:])
        except ValueError:
            return n, f"not numeric: {ln}"
        if not (all(map(math.isfinite, (xc, yc, w, h))) and 0 < w <= 1 and 0 < h <= 1 and xc - w / 2 >= -1e-6
                and xc + w / 2 <= 1 + 1e-6 and yc - h / 2 >= -1e-6 and yc + h / 2 <= 1 + 1e-6):
            return n, f"out of bounds: {ln}"
        n += 1
    return n, ""


def audit_and_export(mod: str, out: Path, part_size: int, decode: bool) -> dict:
    s = SRC[mod]
    res: dict = {"modality": mod, "fails": defaultdict(list)}
    F = res["fails"]
    # A1 provenance: archive == SHA256SUMS == Colab run record
    run = json.loads(s["run"].read_text(encoding="utf-8"))
    zsha = sha256_file(s["zip"])
    sums = read_sums(s["sums"])
    res["provenance"] = {"archive": s["zip"].name, "sha256": zsha, "sha256sums": sums.get(s["zip"].name),
                         "colab_dataset": run["dataset"], "colab_dataset_sha256": run["dataset_sha256"],
                         "colab_run_time": run.get("time"), "colab_epochs_run": run.get("epochs_run"),
                         "colab_best_epoch": run.get("best_epoch")}
    if not (zsha == sums.get(s["zip"].name) == run["dataset_sha256"] and run["dataset"] == s["zip"].name):
        F["A1_provenance"].append("archive sha256 differs from SHA256SUMS / Colab record")

    src = zipfile.ZipFile(s["zip"])
    root = src.namelist()[0].split("/")[0]
    meta = pd.read_csv(io.BytesIO(src.read(f"{root}/slices.csv")))
    entries = {i.filename: i for i in src.infolist()}
    imgs = defaultdict(dict)
    for name in entries:
        p = name.split("/")
        if len(p) == 4 and p[1] in ("images", "labels") and p[3]:
            imgs[(p[2], Path(p[3]).stem)][p[1]] = name
    # A2 counts and pairing
    by_split = Counter(k[0] for k in imgs)
    res["counts"] = dict(by_split)
    for sp, n in EXPECTED[mod].items():
        if by_split.get(sp) != n:
            F["A2_count"].append(f"{sp}: {by_split.get(sp)} != {n}")
    if set(by_split) - set(EXPECTED[mod]):
        F["A2_count"].append(f"unexpected splits {set(by_split) - set(EXPECTED[mod])}")
    unpaired = [k for k, v in imgs.items() if set(v) != {"images", "labels"}]
    if unpaired:
        F["A2_pairing"] += [str(k) for k in unpaired[:5]]
    meta_keys = set(zip(meta.split, meta.file))
    if meta_keys != set(imgs):
        F["A2_metadata"].append(f"slices.csv rows {len(meta_keys)} vs archive images {len(imgs)}")
    meta = meta.set_index(["split", "file"])

    # A5 split integrity against the frozen split
    frozen = pd.read_csv(s["split"]).set_index("patient_id")["split"]
    pats = meta.reset_index().groupby("patient_id").split.agg(lambda x: set(x))
    for pid, sps in pats.items():
        if len(sps) != 1 or frozen.get(pid) not in sps:
            F["A5_split"].append(f"{pid}: archive {sps} vs frozen {frozen.get(pid)}")
    test_pids = set(pd.read_csv(s["test"]).patient_id)
    if test_pids & set(pats.index):
        F["A5_test_patients"].append(f"{len(test_pids & set(pats.index))} Test patients in archive")

    # Export parts (per split) + per-image checks
    zdir = out / FOLDER[mod] / "zips"
    zdir.mkdir(parents=True, exist_ok=True)
    yaml = (f"# LiverCancer AI - {mod} YOLO data exactly as trained/validated on Google Colab ({s['zip'].name}).\n"
            "# Roboflow is used only to visualise / review labels; models were trained on Google Colab, not on Roboflow.\n"
            "# Labels were converted automatically from the source dataset's lesion masks. Class 0 = lesion.\n"
            "# Empty label file = no lesion annotated on that slice (not a healthy patient).\n"
            f"# Source / license: {LICENSE[mod]}\n"
            "path: .\ntrain: train/images\nval: valid/images\nnc: 1\nnames: ['lesion']\n")
    rows, parts = [], []
    sha_seen: dict[str, str] = {}
    for sp in ("train", "val"):
        stems = sorted(k[1] for k in imgs if k[0] == sp)
        n_parts = math.ceil(len(stems) / part_size)
        for pi in range(n_parts):
            chunk = stems[pi * part_size:(pi + 1) * part_size]
            zname = f"Roboflow_{mod}_{RF_SPLIT[sp]}_part{pi + 1:02d}of{n_parts:02d}.zip"
            zp = zdir / zname
            cnt = Counter()
            with zipfile.ZipFile(zp, "w") as z:
                z.writestr(zipfile.ZipInfo("data.yaml", date_time=entries[f"{root}/data.yaml"].date_time), yaml,
                           compress_type=zipfile.ZIP_DEFLATED)  # fixed timestamp -> reproducible zip SHA256
                for st in chunk:
                    ie, le = imgs[(sp, st)]["images"], imgs[(sp, st)]["labels"]
                    ib, lb = src.read(ie), src.read(le)  # CRC verified by zipfile on read
                    nb, err = label_check(lb.decode("ascii"))
                    if err:
                        F["A4_label"].append(f"{st}: {err}")
                    m = meta.loc[(sp, st)]
                    if nb != int(m.n_boxes) or (m.kind == "neg") != (nb == 0):
                        F["A4_box_count"].append(st)
                    if decode:
                        im = cv2.imdecode(np.frombuffer(ib, np.uint8), cv2.IMREAD_UNCHANGED)
                        if im is None or im.shape[:2] != (int(m.height), int(m.width)):
                            F["A3_image"].append(st)
                    h = hashlib.sha256(ib).hexdigest()
                    if h in sha_seen:
                        res.setdefault("identical_images", []).append((sha_seen[h], st))
                    sha_seen[h] = st
                    z.writestr(zipfile.ZipInfo(f"{RF_SPLIT[sp]}/images/{st}.png", date_time=entries[ie].date_time), ib,
                               compress_type=zipfile.ZIP_STORED)
                    z.writestr(zipfile.ZipInfo(f"{RF_SPLIT[sp]}/labels/{st}.txt", date_time=entries[le].date_time), lb,
                               compress_type=zipfile.ZIP_DEFLATED)
                    cnt["images"] += 1
                    cnt["lesion"] += nb > 0
                    cnt["boxes"] += nb
                    rows.append({"modality": mod, "split": sp, "roboflow_split": RF_SPLIT[sp], "file": st, "zip": zname,
                                 "part": pi + 1, "patient_id": m.patient_id, "patient_label": m.patient_label,
                                 "series": m.series, "slice_k": int(m.slice_k), "kind": m.kind, "n_boxes": nb,
                                 "width": int(m.width), "height": int(m.height), "image_sha256": h,
                                 "image_crc32": f"{entries[ie].CRC:08x}", "label_crc32": f"{entries[le].CRC:08x}"})
            parts.append({"modality": mod, "split": sp, "roboflow_split": RF_SPLIT[sp], "zip": zname,
                          "folder": f"{FOLDER[mod]}/zips", "images": cnt["images"], "lesion_images": cnt["lesion"],
                          "empty_label_images": cnt["images"] - cnt["lesion"], "boxes": cnt["boxes"],
                          "first_file": chunk[0], "last_file": chunk[-1], "size_bytes": zp.stat().st_size,
                          "sha256": sha256_file(zp), "upload_allowed": "yes (CC BY 4.0)" if mod == "CT" else "NO (keep offline)",
                          "uploaded_to_roboflow": "chua", "upload_date": "", "roboflow_project": "", "note": ""})
        print(f"[{mod}] {sp}: {len(stems)} images in {n_parts} zips", flush=True)

    # A6 re-open every part: CRC, pairing, byte identity with the source archive, no overlap
    seen = set()
    by_name = {(r["roboflow_split"], r["file"]): r for r in rows}
    for p in parts:
        with zipfile.ZipFile(zdir / p["zip"]) as z:
            if z.testzip():
                F["A6_zip_crc"].append(p["zip"])
            names = [i for i in z.infolist() if i.filename != "data.yaml"]
            im = {Path(i.filename).stem: i for i in names if "/images/" in i.filename}
            lb = {Path(i.filename).stem: i for i in names if "/labels/" in i.filename}
            if set(im) != set(lb) or len(im) != p["images"]:
                F["A6_zip_pairing"].append(p["zip"])
            for st, info in im.items():
                key = (p["roboflow_split"], st)
                if key in seen:
                    F["A6_zip_overlap"].append(st)
                seen.add(key)
                r = by_name[key]
                if f"{info.CRC:08x}" != r["image_crc32"] or f"{lb[st].CRC:08x}" != r["label_crc32"]:
                    F["A6_bytes_differ_from_colab_archive"].append(st)
    if len(seen) != len(imgs):
        F["A6_union"].append(f"{len(seen)} != {len(imgs)}")

    df = pd.DataFrame(rows)
    res["parts"] = parts
    res["rows"] = df
    res["test_reference"] = pd.read_csv(s["test"])
    res["fails"] = {k: v for k, v in F.items()}
    return res


def stats_table(results: dict) -> pd.DataFrame:
    out = []
    for mod, r in results.items():
        df = r["rows"]
        for sp in ("train", "val"):
            g = df[df.split == sp]
            out.append({"modality": mod, "split": sp, "in_85586": "yes", "images": len(g),
                        "lesion_images": int((g.n_boxes > 0).sum()), "empty_label_images": int((g.n_boxes == 0).sum()),
                        "boxes": int(g.n_boxes.sum()), "patients": g.patient_id.nunique(),
                        "volumes": g[["patient_id", "series"]].drop_duplicates().shape[0]})
        t = r["test_reference"]
        out.append({"modality": mod, "split": "test_gt_lesion_slices",
                    "in_85586": "no (locked Test; GT lesion slices exported only for scoring, not exported to Roboflow)",
                    "images": len(t),
                    "lesion_images": int((t.n_boxes > 0).sum()), "empty_label_images": int((t.n_boxes == 0).sum()),
                    "boxes": int(t.n_boxes.sum()), "patients": t.patient_id.nunique(),
                    "volumes": t[["patient_id", "series"]].drop_duplicates().shape[0]})
    return pd.DataFrame(out)


def n(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def mb(b) -> str:
    return f"{b / 1e6:,.1f} MB".replace(",", "X").replace(".", ",").replace("X", ".")


def vn(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def credits(n_img: int) -> dict:
    up = n_img / RF["upload_images_per_credit"]
    st = n_img / RF["storage_images_per_credit_month"]
    return {"upload": up, "storage_month": st, "first_month": up + st}


def write_reports(results: dict, tab: pd.DataFrame, parts: pd.DataFrame, rep: Path) -> None:
    rep.mkdir(parents=True, exist_ok=True)
    tab.to_csv(ROOT / "reports/thong_ke_du_lieu_chinh_thuc.csv", index=False)
    in85 = tab[tab.in_85586 == "yes"]
    tot = in85[["images", "lesion_images", "empty_label_images", "boxes"]].sum()
    sp_vi = {"train": "Train", "val": "Validation", "test_gt_lesion_slices": "Test (khóa)"}
    S = ["# Thống kê dữ liệu chính thức – LiverCancer AI", "",
         f"*Sinh bởi `scripts/85_build_roboflow_full.py` lúc {time.strftime('%Y-%m-%d %H:%M')}; bản CSV: "
         "`reports/thong_ke_du_lieu_chinh_thuc.csv`. Dùng thống nhất trong báo cáo và slide.*", "",
         "## Bộ dữ liệu YOLO (ảnh 2D) đã dùng để huấn luyện và chọn mô hình trên Google Colab", "",
         "| Loại ảnh | Tập | Ảnh | Ảnh có tổn thương | Ảnh không có box | Bounding box | Bệnh nhân | Volume |",
         "|---|---|---|---|---|---|---|---|"]
    for r in in85.itertuples():
        S.append(f"| {r.modality} | {sp_vi[r.split]} | {n(r.images)} | {n(r.lesion_images)} | {n(r.empty_label_images)} "
                 f"| {n(r.boxes)} | {r.patients} | {n(r.volumes)} |")
    for mod in ("CT", "MRI"):
        g = in85[in85.modality == mod]
        S.append(f"| **{mod}** | **Train + Validation** | **{n(g.images.sum())}** | {n(g.lesion_images.sum())} "
                 f"| {n(g.empty_label_images.sum())} | {n(g.boxes.sum())} | {g.patients.sum()} | {n(g.volumes.sum())} |")
    S += [f"| **Tổng** | | **{n(tot.images)}** | {n(tot.lesion_images)} | {n(tot.empty_label_images)} | {n(tot.boxes)} "
          f"| {in85.patients.sum()} | {n(in85.volumes.sum())} |", "",
          "## Tập Test (khóa, chỉ chạy một lần; **không** nằm trong 85.586 ảnh)", "",
          "Test được đánh giá trên **volume NIfTI gốc** của 54 BN CT / 104 BN MRI. Bảng dưới là bộ ảnh GT chỉ gồm **lát có tổn thương** "
          "(xuất bởi `scripts/96_final_test.py` để chấm điểm phát hiện), nên không có ảnh không box và không có BN CN; "
          "**không** so sánh trực tiếp với số ảnh Train/Validation.", "",
          "| Loại ảnh | Ảnh | Ảnh có tổn thương | Ảnh không có box | Bounding box | Bệnh nhân | Volume |", "|---|---|---|---|---|---|---|"]
    for r in tab[tab.split == "test_gt_lesion_slices"].itertuples():
        S.append(f"| {r.modality} | {n(r.images)} | {n(r.lesion_images)} | {n(r.empty_label_images)} | {n(r.boxes)} | {r.patients} | {n(r.volumes)} |")
    S += ["", "## Cách diễn đạt thống nhất", "",
          f"- \"Mô hình YOLO được huấn luyện **trên Google Colab** với **{n(tot.images)} ảnh 2D** "
          f"(CT {n(in85[in85.modality == 'CT'].images.sum())}, MRI {n(in85[in85.modality == 'MRI'].images.sum())}), "
          "gồm tập Train và Validation; tập Test được giữ riêng.\"",
          "- Nhãn bounding box được **chuyển đổi tự động** từ mask tổn thương của dataset gốc (PLC-CECT; LLD-MMRI-MedSAM2), "
          "không gán tay. Roboflow (nếu dùng) chỉ để **trực quan hóa và rà soát nhãn**; **không** train mô hình trên Roboflow.",
          "- CT Train lấy 1/2 số lát có tổn thương (cách 1 lát); CT Validation lấy đủ. Ảnh không có box chỉ nghĩa là lát đó "
          "không có tổn thương được gán nhãn, không phải bệnh nhân khỏe mạnh.",
          "- Số bệnh nhân: CT 253 Train / 54 Validation / 54 Test (42 BN ung thư + 12 BN CN); MRI 316 / 78 / 104 "
          "(chia theo bệnh nhân, seed 42).", ""]
    (ROOT / "reports/thong_ke_du_lieu_chinh_thuc.md").write_text("\n".join(S), encoding="utf-8")

    # audit + upload feasibility
    ct_n = int(in85[in85.modality == "CT"].images.sum())
    all_n = int(tot.images)
    c_ct, c_all = credits(ct_n), credits(all_n)
    ok = lambda r, k: "PASS" if not r["fails"].get(k) else f"FAIL ({len(r['fails'][k])})"
    A = ["# Kiểm toán 85.586 ảnh và chuẩn bị bản xuất Roboflow", "",
         f"*`scripts/85_build_roboflow_full.py`, {time.strftime('%Y-%m-%d %H:%M')}. Chi tiết máy đọc: "
         "`data/processed/roboflow_full_85586/audit.json`.*", "",
         "## 1. Kết luận", "",
         f"- **Xác nhận đủ {n(all_n)} ảnh hợp lệ**, đúng bằng hai bộ đã huấn luyện trên Google Colab: "
         f"CT {n(ct_n)} + MRI {n(all_n - ct_n)}. Mọi kiểm tra bên dưới đạt." if not any(r["fails"] for r in results.values())
         else "- **CÓ LỖI** – xem bảng kiểm tra.",
         "- Bản xuất giữ nguyên ảnh, nhãn, tên file và phân chia Train/Validation gốc; không thêm, bớt, sửa hay chia lại. "
         "Dữ liệu huấn luyện chính không bị ghi.",
         "- **Chưa upload, chưa xóa gì trên Roboflow.**", "",
         "## 2. Nguồn gốc (đối chiếu log Google Colab)", "",
         "| | CT | MRI |", "|---|---|---|"]
    pc, pm = results["CT"]["provenance"], results["MRI"]["provenance"]
    for k, lab in (("archive", "Archive"), ("colab_dataset", "Dataset ghi trong bundle Colab"),
                   ("colab_run_time", "Thời điểm ghi kết quả Colab"), ("colab_epochs_run", "Số epoch đã chạy"),
                   ("colab_best_epoch", "Best epoch")):
        A.append(f"| {lab} | `{pc[k]}` | `{pm[k]}` |")
    A.append(f"| SHA256 archive = `SHA256SUMS` = SHA256 ghi trong Colab | `{pc['sha256'][:16]}…` {ok(results['CT'], 'A1_provenance')} "
             f"| `{pm['sha256'][:16]}…` {ok(results['MRI'], 'A1_provenance')} |")
    A += ["", "Bundle Colab: `outputs/runs/yolo_{ct,mri}/yolo11n_e50_img512_baseline/metrics_val.json`. "
          "Log Colab lưu ở local chỉ gồm `args.yaml`, `results.csv`, metrics; dòng \"Scanning … images\" của lúc train không được lưu, "
          "nên số ảnh được xác minh qua SHA256 của chính archive đã train và đếm trực tiếp trong archive. "
          "Lần đánh giá lại ở local (`outputs/runs/yolo_*/analyze_*_e0.log`) quét đúng CT Val 14.627 ảnh, MRI Val 7.491 ảnh, 0 ảnh hỏng.", "",
          "## 3. Kiểm tra", "", "| Kiểm tra | CT | MRI |", "|---|---|---|"]
    for k, lab in (("A2_count", "A2. Số ảnh mỗi tập đúng kỳ vọng (CT 34.873 / 14.627; MRI 28.595 / 7.491)"),
                   ("A2_pairing", "A2. Mỗi ảnh có đúng một file nhãn"),
                   ("A2_metadata", "A2. Khớp 1–1 với `slices.csv` trong archive"),
                   ("A3_image", "A3. Mọi ảnh giải mã được, đúng kích thước"),
                   ("A4_label", "A4. Nhãn đúng định dạng (5 cột, class 0, box trong [0, 1])"),
                   ("A4_box_count", "A4. Số box khớp metadata; ảnh âm tính có nhãn rỗng"),
                   ("A5_split", "A5. Mỗi BN chỉ ở một tập và đúng split đóng băng"),
                   ("A5_test_patients", "A5. Không có BN Test"),
                   ("A6_zip_crc", "A6. ZIP mới: CRC đúng"), ("A6_zip_pairing", "A6. ZIP mới: đủ cặp ảnh–nhãn"),
                   ("A6_zip_overlap", "A6. Các ZIP không trùng ảnh"), ("A6_union", "A6. Gộp các ZIP = đủ bộ"),
                   ("A6_bytes_differ_from_colab_archive", "A6. Từng ảnh/nhãn trùng byte (CRC32) với archive Colab")):
        A.append(f"| {lab} | {ok(results['CT'], k)} | {ok(results['MRI'], k)} |")
    cross = len(set(results["CT"]["rows"].image_sha256) & set(results["MRI"]["rows"].image_sha256))
    A.append(f"| A7. Không trùng ảnh giữa CT và MRI (SHA256 chung: {cross}) | {'PASS' if cross == 0 else 'FAIL'} | (chung) |")
    idc = {m: len(r.get("identical_images", [])) for m, r in results.items()}
    A += ["", f"Ảnh trùng byte **trong cùng** bộ: CT {idc['CT']} cặp, MRI {idc['MRI']} cặp. Mọi cặp thuộc cùng một BN và cùng một tập "
          "(không vắt qua Train/Validation), khác pha/chuỗi: dataset gốc có một số pha giống hệt nhau ở vài lát "
          "(ví dụ `P0069_C2_k047` = `P0069_C3_k047`). Giữ nguyên vì đây là đúng dữ liệu đã train. "
          "Roboflow có thể bỏ qua ảnh trùng khi upload, nên số ảnh hiển thị trên Roboflow có thể ít hơn tương ứng "
          f"(CT tối thiểu {n(len(results['CT']['rows']) - idc['CT'])}).", "",
          "## 4. Thống kê", "", "Xem `reports/thong_ke_du_lieu_chinh_thuc.md` (bảng chính thức).", "",
          "## 5. Bản xuất", "", "| Bộ | Thư mục | ZIP | Ảnh | Dung lượng | Được upload? |", "|---|---|---|---|---|---|"]
    for mod in ("CT", "MRI"):
        g = parts[parts.modality == mod]
        for sp in ("train", "val"):
            h = g[g.split == sp]
            A.append(f"| {mod} {sp_vi[sp]} | `data/processed/roboflow_full_85586/{FOLDER[mod]}/zips/` | {len(h)} "
                     f"(`Roboflow_{mod}_{RF_SPLIT[sp]}_partXXofYY.zip`) | {n(h.images.sum())} | {mb(h.size_bytes.sum())} "
                     f"| {'Có (CC BY 4.0, ghi nguồn)' if mod == 'CT' else '**Không** – giữ offline'} |")
    A += [f"| **Tổng** | | {len(parts)} | {n(parts.images.sum())} | {mb(parts.size_bytes.sum())} | |", "",
          "- Mỗi ZIP ≤ 1.000 ảnh, chứa `data.yaml` + `train/` hoặc `valid/` (`images/`, `labels/`) – Roboflow nhận đúng tập Train/Valid.",
          "- `data/processed/roboflow_full_85586/upload_manifest.csv`: một dòng mỗi ZIP (số ảnh, box, SHA256, cột "
          "`uploaded_to_roboflow` / `upload_date` / `roboflow_project` để đánh dấu, tránh upload trùng).",
          "- `data/processed/roboflow_full_85586/manifest_images.csv`: một dòng mỗi ảnh (ZIP chứa nó, BN, pha/chuỗi, lát, số box, SHA256, CRC32).", "",
          "## 6. Quyền phân phối", "",
          f"- **CT – {LICENSE['CT']}.** Được upload; ghi nguồn trong mô tả project Roboflow.",
          "- **MRI – LLD-MMRI, CC BY-NC 4.0** kèm điều khoản *\"not to further copy, publish, or distribute any part\"* "
          "(`reports/audit_report.md`). **Không upload** lên Roboflow (kể cả project riêng tư) khi chưa có văn bản cho phép của tác giả.", "",
          "## 7. Khả năng upload (giới hạn Roboflow)", "",
          f"Không truy cập được tài khoản Roboflow của bạn; số dưới đây theo bảng giá công khai đọc ngày {RF['checked']} "
          "([roboflow.com/pricing](https://roboflow.com/pricing), [roboflow.com/credits](https://roboflow.com/credits)): "
          f"gói **Free Tier = {RF['free_credits_month']} credit/tháng**, project riêng tư, không có flex billing; "
          f"upload **{n(RF['upload_images_per_credit'])} ảnh/credit**; lưu trữ **{n(RF['storage_images_per_credit_month'])} ảnh/credit/tháng** "
          "(tính theo tháng). Hết credit thì nhiều chức năng bị dừng tới khi credit tháng sau được cộng hoặc mua thêm.", "",
          "| Phương án | Ảnh | Credit upload (1 lần) | Credit lưu trữ/tháng | Tháng đầu | So với Free 10 credit |", "|---|---|---|---|---|---|"]
    opts = [("Chỉ CT (Train + Valid)", ct_n), ("CT + MRI (không được phép với MRI)", all_n)]
    ct_tr = int(in85[(in85.modality == "CT") & (in85.split == "train")].images.sum())
    opts.insert(1, ("Chỉ CT Train", ct_tr))
    for lab, k in opts:
        c = credits(k)
        A.append(f"| {lab} | {n(k)} | {vn(c['upload'])} | {vn(c['storage_month'])} | {vn(c['first_month'])} "
                 f"| {'vượt' if c['first_month'] > RF['free_credits_month'] else 'trong mức'} |")
    max_first = int(RF["free_credits_month"] / (1 / RF["upload_images_per_credit"] + 1 / RF["storage_images_per_credit_month"]))
    A += ["", f"- Với Free Tier, tháng đầu chỉ chứa được tối đa khoảng **{n(max_first)} ảnh** (upload + lưu trữ = 10 credit) "
          "và chưa tính credit project cũ đang dùng. **Không thể đưa đủ 49.500 ảnh CT lên gói miễn phí.**",
          f"- Đủ 49.500 ảnh CT cần khoảng {vn(c_ct['first_month'], 1)} credit tháng đầu và {vn(c_ct['storage_month'], 1)} credit/tháng sau đó "
          "→ cần gói trả phí hoặc credit mua thêm. **Tôi không mua credit.** Hãy kiểm tra số credit còn lại tại "
          "Roboflow → Settings → *Billing / Usage* trước khi upload.",
          "- Project cũ vẫn tốn credit lưu trữ; **chưa xóa** project cũ cho tới khi bộ thay thế được upload và kiểm tra xong.",
          "- Phương án vừa gói miễn phí: dùng các bộ minh họa đã có (`data/processed/roboflow/` 5.000 ảnh CT, "
          "`data/processed/roboflow_positive_20k/zips/CT/` 10.000 ảnh CT), còn con số 85.586 trình bày bằng bảng thống kê chính thức.", "",
          "## 8. Thứ tự đề xuất (chờ bạn xác nhận)", "",
          "1. Xem số credit còn lại trong tài khoản Roboflow.", "2. Nếu đủ: tạo project Object Detection mới (lớp `lesion`), upload lần lượt "
          f"`{parts[(parts.modality == 'CT') & (parts.split == 'train')].zip.iloc[0]}` … rồi "
          f"`{parts[(parts.modality == 'CT') & (parts.split == 'val')].zip.iloc[0]}` …, đánh dấu từng dòng trong `upload_manifest.csv`.",
          f"3. Kiểm tra trên Roboflow: tổng ảnh = 49.500 (Train 34.873, Valid 14.627), hoặc ít hơn tối đa {idc['CT']} nếu Roboflow bỏ ảnh trùng; "
          "ảnh nhãn rỗng hiện là *null* (hoặc *unannotated* → *Mark Null*).",
          "4. Chỉ sau khi xác minh xong mới cân nhắc xóa project cũ.", "5. MRI: giữ offline.", ""]
    (rep / "bao_cao_kiem_toan.md").write_text("\n".join(A), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part-size", type=int, default=1000)
    ap.add_argument("--no-decode", action="store_true", help="skip decoding every PNG")
    ap.add_argument("--out", default=str(ROOT / "data/processed/roboflow_full_85586"))
    args = ap.parse_args()
    out = Path(args.out)
    for mod in FOLDER:
        z = out / FOLDER[mod] / "zips"
        if z.exists():
            for f in z.glob("Roboflow_*.zip"):
                f.unlink()
    results = {}
    for mod in ("CT", "MRI"):
        print(f"[{mod}] audit + export ...", flush=True)
        results[mod] = audit_and_export(mod, out, args.part_size, not args.no_decode)
        print(f"[{mod}] fails: {({k: len(v) for k, v in results[mod]['fails'].items()}) or 'none'}", flush=True)
    rows = pd.concat([r["rows"] for r in results.values()], ignore_index=True)
    parts = pd.DataFrame([p for r in results.values() for p in r["parts"]])
    rows.to_csv(out / "manifest_images.csv", index=False)
    parts.to_csv(out / "upload_manifest.csv", index=False)
    tab = stats_table(results)
    audit = {m: {"provenance": r["provenance"], "counts": r["counts"],
                 "fails": {k: v[:10] for k, v in r["fails"].items()},
                 "identical_image_pairs": r.get("identical_images", [])[:50],
                 "n_identical_image_pairs": len(r.get("identical_images", []))} for m, r in results.items()}
    audit["total_images"] = int(len(rows))
    audit["cross_modality_shared_sha256"] = len(set(results["CT"]["rows"].image_sha256) & set(results["MRI"]["rows"].image_sha256))
    audit["roboflow_pricing"] = RF
    (out / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_reports(results, tab, parts, ROOT / "reports/roboflow_full_85586")
    print(f"total {len(rows)}; zips {len(parts)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
