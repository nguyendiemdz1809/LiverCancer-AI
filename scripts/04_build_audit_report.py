"""Build reports/tables/{ct,mri}_class_distribution.md and reports/audit_report.md.

Reads only metadata (patient_data.csv, LLD_MMRI_Annotation.json, file existence) — no image voxels.
Inputs: reports/dataset_audit.json, reports/audit/{samples.json, full_summary.json, ct_volumes.csv, mri_series.csv},
        data/external/lld_mmri2023_official_val_ids.csv
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import load_paths  # noqa: E402

CT_CLASSES = ["HCC", "ICC", "CHCC", "CN"]


def load_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ct_table(cfg: dict) -> str:
    data_dir = Path(cfg["data_dir"])
    by_pid = defaultdict(list)
    for r in load_csv(Path(cfg["csv"])):
        by_pid[r["patient_id"]].append(r)
    exists = {(r["patient_id"], r["phase"]): (data_dir / r["ct_path"]).exists()
              for rs in by_pid.values() for r in rs}

    def stats(pids: list[str]) -> str:
        n_vol = sum(len(by_pid[p]) for p in pids)
        have = sum(exists[(p, r["phase"])] for p in pids for r in by_pid[p])
        full = sum(all(exists[(p, r["phase"])] for r in by_pid[p]) for p in pids)
        ages = [float(by_pid[p][0]["age"]) for p in pids if by_pid[p][0]["age"] != "NA"]
        g = Counter(by_pid[p][0]["gender"] for p in pids)
        return (f"{len(pids)} | {n_vol} | {have} | {n_vol - have} | {full} | {len(pids) - full} | "
                f"{mean(ages):.1f} ({min(ages):.0f}–{max(ages):.0f}) | {g['male']}/{g['female']}/{g['NA']}")

    L = ["| Lớp | Bệnh nhân | Volume trong CSV | Volume có file | Volume thiếu | BN đủ 4 pha | BN thiếu ≥1 pha | "
         "Tuổi TB (min–max) | Nam/Nữ/NA |", "|---|---|---|---|---|---|---|---|---|"]
    for lab in CT_CLASSES:
        L.append(f"| {lab} | " + stats([p for p, rs in by_pid.items() if rs[0]["cancer_type"] == lab]) + " |")
    L.append("| Tổng | " + stats(list(by_pid)) + " |")
    missing_phase = Counter(ph for (p, ph), ok in exists.items() if not ok)
    incons = sorted(p for p, rs in by_pid.items() if len({(r["age"], r["gender"]) for r in rs}) > 1)
    incomplete = sorted(p for p, rs in by_pid.items() if not all(exists[(p, r["phase"])] for r in rs))
    L += ["", f"Volume CT thiếu theo pha: {dict(missing_phase)}",
          f"Bệnh nhân có age/gender không nhất quán giữa các dòng: {len(incons)} {incons}",
          f"Bệnh nhân thiếu ≥1 pha: {incomplete}"]
    return "\n".join(L) + "\n"


def mri_table(cfg: dict, val_ids: set[str]) -> str:
    ann = json.loads(Path(cfg["annotation"]).read_text(encoding="utf-8"))
    info, cats = ann["Annotation_info"], ann["Category_info"]
    benign = set(cats["Benign"])
    pcat = {p: s[0]["annotation"]["lesion"]["0"]["category"] for p, s in info.items()}
    L = ["| id | Lớp | Nhóm | Bệnh nhân | Series | Trong val chính thức | Còn lại (train+test) |",
         "|---|---|---|---|---|---|---|"]
    for name, cid in sorted(((k, v) for k, v in cats.items() if isinstance(v, int)), key=lambda x: x[1]):
        ps = [p for p, c in pcat.items() if c == cid]
        n_val = sum(p in val_ids for p in ps)
        L.append(f"| {cid} | {name} | {'Benign' if cid in benign else 'Malignant'} | {len(ps)} | "
                 f"{sum(len(info[p]) for p in ps)} | {n_val} | {len(ps) - n_val} |")
    n_val = sum(p in val_ids for p in pcat)
    L.append(f"| | **Tổng** | | {len(pcat)} | {sum(len(s) for s in info.values())} | {n_val} | {len(pcat) - n_val} |")
    return "\n".join(L) + "\n"


def load_csv_commented(p: Path) -> list[dict]:
    """CSV whose leading lines starting with '#' are provenance comments."""
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def yes(v: str | None) -> bool:
    return v == "True"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Vietnamese text on Windows consoles / redirected logs
    audit = json.loads((REP / "dataset_audit.json").read_text(encoding="utf-8"))
    samples = json.loads((REP / "audit" / "samples.json").read_text(encoding="utf-8"))
    full = json.loads((REP / "audit" / "full_summary.json").read_text(encoding="utf-8"))
    ct_rows = load_csv(REP / "audit" / "ct_volumes.csv")
    mri_rows = load_csv(REP / "audit" / "mri_series.csv")
    paths = load_paths()
    val_ids = {r["patient_id"] for r in load_csv_commented(ROOT / "data" / "external" / "lld_mmri2023_official_val_ids.csv")}
    ct_tab, mri_tab = ct_table(paths["ct"]), mri_table(paths["mri"], val_ids)
    (REP / "tables").mkdir(parents=True, exist_ok=True)
    (REP / "tables" / "ct_class_distribution.md").write_text(ct_tab, encoding="utf-8")
    (REP / "tables" / "mri_class_distribution.md").write_text(mri_tab, encoding="utf-8")
    fc, fm = full["ct"], full["mri"]

    assert fc["rows"] == 1444 and fm["series"] == 3984, "full audit incomplete (run 03_audit_full.py without --limit)"

    missing_ct = audit["ct"]["missing_files"]["ct_path"]["items"]
    low_liver = [r for r in ct_rows if r.get("lesion_in_liver_frac") and float(r["lesion_in_liver_frac"]) < 0.5]
    out_liver_bbox = [r for r in ct_rows if r.get("lesion_in_liver_bbox_frac")
                      and float(r["lesion_in_liver_bbox_frac"]) < 0.9]
    low_iou = sorted((r for r in mri_rows if r.get("iou_json3d_vs_mask_xy")
                      and float(r["iou_json3d_vs_mask_xy"]) < 0.3),
                     key=lambda r: float(r["iou_json3d_vs_mask_xy"]))
    multi_comp_mri = [r for r in mri_rows if r.get("mask_components") and int(r["mask_components"]) > 1]
    z_range = [int(r["ct_shape"].split("x")[2]) for r in ct_rows if r.get("ct_shape")]
    raw_vals = Counter(r.get("lesion_raw_values") for r in ct_rows if yes(r.get("has_lesion_mask")))
    n_65535 = sum(v for k, v in raw_vals.items() if k and "65535" in k)
    liver_empty = [f"{r['patient_id']}_{r['phase']}" for r in ct_rows if yes(r.get("liver_empty"))]
    les_shape_bad = [f"{r['patient_id']}_{r['phase']}" for r in ct_rows
                     if r.get("lesion_shape_match_ct") == "False" or r.get("lesion_shape_match_liver") == "False"]
    z_bad = [f"{r['patient_id']} {r['sequence']} (JSON z {r['json_z']}, mask z {r.get('mask_z')})"
             for r in mri_rows if r.get("z_overlap") == "False"]

    L: list[str] = []
    a = L.append
    a("# Báo cáo Audit Dataset — Giai đoạn 1\n")
    a(f"*Sinh tự động bởi `scripts/04_build_audit_report.py` lúc {datetime.now():%Y-%m-%d %H:%M}. "
      "Mọi số liệu lấy từ các file trong `reports/` (không chứa ảnh gốc).*\n")
    a("Các lệnh đã chạy (đều chỉ đọc dataset gốc):\n")
    a("```\n.venv\\Scripts\\python scripts\\00_check_env.py\n"
      ".venv\\Scripts\\python scripts\\01_verify_datasets.py --headers 3   # đối chiếu CSV/JSON <-> file\n"
      ".venv\\Scripts\\python scripts\\02_audit_samples.py                 # đọc thử 5 CT + 5 MRI, xuất overlay\n"
      ".venv\\Scripts\\python scripts\\03_audit_full.py --workers 2        # toàn bộ mask + header ảnh\n"
      ".venv\\Scripts\\python scripts\\04_build_audit_report.py\n```\n")

    a("## 1. Môi trường\n")
    a("Python 3.12.10 (.venv) · torch 2.11.0+cu128 · CUDA available = True · "
      "RTX 3050 Laptop 4.00 GB (compute capability 8.6) · đủ thư viện (`00_check_env.py` → READY).\n")

    a("## 2. Xác minh từ tài liệu gốc\n")
    a("| Nội dung | Kết quả | Nguồn |\n|---|---|---|")
    a("| **CN** (CT) | Nhóm chứng: *\"83 individuals without hepatobiliary disease as a control group\"*; "
      "*\"confirmed to have no hepatobiliary disease, and no lesion data were collected\"*. CN = **không có bệnh gan mật**. "
      "Bài báo **không** nói đây là \"người khỏe mạnh hoàn toàn\" và không nêu tiêu chí xác nhận. Có liver mask, không có lesion mask. | [P1] |")
    a("| **HCC / ICC / CHCC** | Hepatocellular Carcinoma / Intrahepatic Cholangiocarcinoma / Combined "
      "Hepatocellular-Cholangiocarcinoma (bài báo viết *cHCC-CCA*; CSV viết `CHCC`). Mã hình thái ICD-O: HCC M81700/3, "
      "ICC M81600/3, cHCC-CCA M81800/3. Chẩn đoán: *\"confirmed as liver cancer through histological and/or cytological "
      "examination\"* trên mẫu phẫu thuật hoặc sinh thiết. | [P1] |")
    a("| **Pha CT** | Bài báo: *Plain (Non-Contrast), Arterial, Venous, Delayed*. **Bài báo không nhắc đến tên C1/C2/C3/P.** "
      "Bằng chứng duy nhất là README của repo tác giả: demo dùng *stage \"C1\"* để hiển thị *\"arterial-phase slices\"* (ca P0059). "
      "Trên overlay: P0089, P0110, P0172, P0198 cho thấy P không tăng quang, C1 có động mạch chủ sáng, C2/C3 ở pha muộn hơn. "
      "Nhưng **P0320 (CN) có C1 trông không tăng quang**. Kết luận: giả thuyết P = Plain, C1 = Arterial, C2 = Venous, C3 = Delayed "
      "**chưa được xác nhận chính thức và có thể không nhất quán**. Pipeline xử lý từng pha theo tên file, không gán tên lâm sàng. | [P1], [G1], overlay |")
    a("| **Tiền xử lý CT** | Bài báo: *\"clipping the pixel intensity to a specified range (−200 to 200)\"*, WL 0 / WW 400. "
      "Repo có `window_adjust.py` clip về [−200, 200]. Bài báo không nói về việc chuyển sang uint8, nhưng dữ liệu tải về là `uint8` 0–255. "
      "**Không còn HU gốc**; giá trị 0 tương ứng ≤ −200 HU (nền và khí). | [P1], [G1] |")
    a("| **Số pha / BN** | Bài báo: *\"All individuals ... had complete 3D CECT imaging data across the Plain, Arterial, Venous, "
      "and Delayed phases\"*. Vậy 40 file thiếu là do **bản tải về không đầy đủ**, không phải do thiết kế dataset. | [P1] |")
    a("| License CT | CC BY 4.0 (Science Data Bank, doi 10.57760/sciencedb.12207) | [P1], `CECT/LICENSE` |")
    a("| **Split CT** | Tác giả **không** công bố split. | [P1] |")
    a("| **Split MRI** | Challenge LLD-MMRI2023: train 316 / val 78 / test 104. Danh sách chính thức (`labels.txt`, "
      "`labels_val_inaccessible.txt`) chỉ có trong **gói dữ liệu gốc**, không có trong bản HF MedSAM2. File "
      "`main/output/LLDBaseline.json` (repo challenge) là *\"prediction result file format\"* của baseline, gồm đúng **78 image_id**, "
      "bằng kích thước tập val. → **Suy luận** đây là tập val (độ tin cậy cao nhưng không phải tuyên bố trực tiếp). "
      "Đã lưu ở `data/external/lld_mmri2023_official_val_ids.csv`; khớp 78/78 với web, và cả 78 ID có trong dữ liệu. "
      "**Danh sách train và test không công khai.** Tác giả khuyến nghị *5-fold cross-validation* trên phần dữ liệu có nhãn. | [G2] |")
    a("| Nhãn MRI | `Category_info` trong JSON: 0 Hemangioma, 1 ICC, 2 Abscess, 3 Metastasis, 4 Cyst, 5 FNH, 6 HCC; "
      "Benign [0,2,4,5], Malignant [1,3,6]. Khớp với danh sách 7 loại tổn thương trong README HF. 8 chuỗi khớp với README. | README HF, [G2] |")
    a("| License MRI | **CC BY-NC 4.0**, *\"solely for noncommercial research purposes\"*; *\"You agree not to further copy, publish, "
      "or distribute any part of the LLD-MMRI dataset\"* (ngoại trừ dùng nội bộ). **Upload lên Google Drive/Colab cần người dùng "
      "tự cân nhắc theo điều khoản này và cho phép rõ ràng.** | [G3] |\n")
    a("Nguồn: [P1] Luo et al., *Comprehensive multi-phase 3D contrast-enhanced CT imaging for primary liver cancer*, "
      "Sci Data 12:768 (2025), doi 10.1038/s41597-025-05125-2 (PMC12065907). [G1] github.com/ljwa2323/PLC_CECT. "
      "[G2] github.com/LMMMEng/LLD-MMRI2023 (README, main/README.md, main/output/LLDBaseline.json). "
      "[G3] github.com/LMMMEng/LLD-MMRI-Dataset. Đọc ngày 2026-10-19.\n")

    a("## 3. Thống kê\n")
    a(f"- **CT:** {audit['ct']['n_patients']} bệnh nhân · {audit['ct']['n_rows']} volume trong CSV · "
      f"{fc['ct_exists']} volume có file.")
    a(f"- **MRI:** {audit['mri']['n_patients_json']} bệnh nhân · {audit['mri']['n_images']} series ảnh · "
      f"{audit['mri']['n_labels']} label. Mỗi bệnh nhân có 8 chuỗi và 1 tổn thương.\n")
    a("### CT — PLC-CECT\n")
    a(ct_tab)
    a("### MRI — LLD-MMRI-MedSAM2\n")
    a(mri_tab)

    a("## 4. Tương ứng ảnh – mask – nhãn (toàn bộ dataset)\n")
    a("### CT\n")
    a("| Kiểm tra | Kết quả |\n|---|---|")
    a(f"| Volume CT có file | {fc['ct_exists']}/{fc['rows']} (thiếu {fc['ct_missing']}) |")
    a(f"| Số chiều volume CT | {fc['ct_ndim']} |")
    a(f"| dtype CT / spacing header | {fc['ct_dtype']} / {fc['ct_zooms']} |")
    a(f"| Số lát cắt (trục k) | {min(z_range)}–{max(z_range)} |")
    a(f"| Liver mask rỗng | {fc['liver_empty']} |")
    a(f"| Liver mask ≠ CT (shape / affine) | {fc['liver_shape_mismatch']} / {fc['liver_affine_mismatch']} |")
    a(f"| Lesion mask (dòng có mask) | {fc['lesion_masks']} (CN: `mask_path = \"NA\"`) |")
    a(f"| Lesion mask rỗng | {fc['lesion_empty']} {fc['lesion_empty_rows'][:20]} |")
    a(f"| Lesion ≠ CT shape / ≠ liver shape / ≠ liver affine | {fc['lesion_shape_mismatch_ct']} / "
      f"{fc['lesion_shape_mismatch_liver']} / {fc['lesion_affine_mismatch_liver']} |")
    a(f"| Giá trị raw của lesion mask | {dict(raw_vals)} |")
    a(f"| Số thành phần liên thông tổn thương / volume | {fc['lesion_components']} |")
    a(f"| Tổn thương nằm trong liver mask < 50% | {fc['lesion_in_liver_frac_lt_0.5']} |")
    a(f"| Tổn thương nằm trong bbox gan < 90% | {fc['lesion_in_liver_bbox_frac_lt_0.9']} |\n")
    a("### MRI\n")
    a("| Kiểm tra | Kết quả |\n|---|---|")
    a(f"| Series thiếu file ảnh/label | {fm['missing_files']} |")
    a(f"| Số chiều ảnh | {fm['image_ndim']} |")
    a(f"| Label ≠ ảnh (shape / affine) | {fm['label_shape_mismatch']} / {fm['label_affine_mismatch']} |")
    a(f"| Spacing header ≠ JSON | {fm['spacing_mismatch_json']} |")
    a(f"| Label rỗng | {fm['label_empty']} {fm['label_empty_series'][:20]} |")
    a(f"| Giá trị label (raw) | {fm['label_raw_values']} |")
    a(f"| JSON slice_idx ngoài phạm vi / bbox ngoài ảnh | {fm['json_slice_out_of_range']} / {fm['json_box_out_of_bounds']} |")
    a(f"| Khoảng z của JSON và mask không giao nhau | {fm['z_no_overlap']} |")
    a(f"| IoU(JSON 3D_box, bbox của mask) — min/p5/p25/median/p75 | {fm['iou_json_vs_mask_quantiles']} |")
    a(f"| IoU < 0.3 | {fm['iou_lt_0.3']} |")
    a(f"| Số thành phần liên thông của mask | {fm['mask_components']} |\n")
    a("**Hệ trục đã chốt:** JSON `x` ↔ trục i, `y` ↔ trục j, `slice_idx` ↔ trục k của mảng nibabel. "
      "Trên 40 series mẫu, IoU theo cách hiểu này là 0.58–0.96, còn cách hiểu đảo x/y cho IoU ≈ 0 (xác nhận bằng overlay). "
      f"Trên toàn bộ 3984 series: median IoU {fm['iou_json_vs_mask_quantiles'][3]}, p25 {fm['iou_json_vs_mask_quantiles'][2]}; "
      "các ca IoU thấp xem E9. Hiển thị: `slice.T` (hàng = j, cột = i).\n")
    a("### 4b. Điều tra 31 series MRI có JSON box lệch với mask (`scripts/05_check_mri_bbox_mismatch.py`)\n")
    inv = load_csv(REP / "audit" / "mri_bbox_investigation.csv")
    fl = [r for r in inv if yes(r["flagged_low_iou"])]
    gd = [r for r in inv if not yes(r["flagged_low_iou"])]
    quar = load_csv_commented(ROOT / "data" / "quarantine" / "mri_bbox_quarantine.csv")
    id_to_name = {v: k for k, v in json.loads(Path(paths["mri"]["annotation"]).read_text(encoding="utf-8"))
                  ["Category_info"].items() if isinstance(v, int)}

    def med(rows: list[dict], key: str) -> float:
        return sorted(float(r[key]) for r in rows if r.get(key))[len(rows) // 2]

    def ratio(rows: list[dict]) -> float:
        return sorted(int(r["mask_voxels"]) / max(float(r["json_box_voxels"]), 1) for r in rows)[len(rows) // 2]

    geom_ok = sum(yes(r["label_affine_match"]) and yes(r["spacing_match_json"]) and yes(r["origin_match_json_lps"]) for r in inv)
    a(f"Phạm vi: toàn bộ 8 chuỗi của {len({r['patient_id'] for r in inv})} BN bị ảnh hưởng ({len(inv)} series), trong đó "
      f"{len(fl)} series bị gắn cờ; {len(gd)} series còn lại dùng làm mốc đối chiếu.\n")
    a("| Giả thuyết | Kiểm tra | Kết quả | Kết luận |\n|---|---|---|---|")
    a(f"| Orientation / affine / spacing | affine ảnh = affine label; spacing JSON = header; origin JSON (LPS) = affine (RAS, đảo dấu x, y) | "
      f"{geom_ok}/{len(inv)} series nhất quán | Loại trừ |")
    a(f"| Quy ước trục / thứ tự lát cắt | IoU 3D khi giữ nguyên, lật i, lật j, lật k, hoán đổi i/j | biến đổi tốt nhất: "
      f"{dict(Counter(r['best_transform'] for r in fl))}; không biến đổi nào cho IoU ≥ 0.5 | Loại trừ |")
    a(f"| 2D_box ↔ 3D_box | z của các 2D_box = z của 3D_box; slice_idx liên tục | z khớp {sum(yes(r['2d_z_equals_3d_z']) for r in fl)}/{len(fl)}; "
      f"không liên tục {sum(not yes(r['2d_slices_sorted_contiguous']) for r in fl)} "
      f"({', '.join(r['patient_id'] + ' ' + r['sequence'] for r in fl if not yes(r['2d_slices_sorted_contiguous']))}) | Không giải thích được |")
    a(f"| Vị trí trong tọa độ thế giới (mm) | khoảng cách từ tâm box/mask đến median của các chuỗi tốt cùng BN | "
      f"series lệch: JSON {med(fl, 'json_dist_to_ref_mm'):.1f} mm, mask {med(fl, 'mask_dist_to_ref_mm'):.1f} mm; "
      f"series tốt: JSON {med(gd, 'json_dist_to_ref_mm'):.1f} mm, mask {med(gd, 'mask_dist_to_ref_mm'):.1f} mm | "
      "**Cả hai nguồn đều lệch**, không xác định được nguồn nào đúng |")
    a(f"| Kích thước | thể tích mask / thể tích JSON box | series lệch {ratio(fl):.3f}, series tốt {ratio(gd):.3f} | "
      "Mask thường nhỏ hơn nhiều so với box |\n")
    q_pat = {r["patient_id"]: r["category_id"] for r in quar}
    a(f"**Kết luận:** không xác minh được nguyên nhân từ orientation, affine, spacing, thứ tự lát cắt hay hệ tọa độ. "
      f"**Cách ly {len(quar)} series / {len(q_pat)} BN** khỏi huấn luyện YOLO và ROI classification. "
      "Bệnh nhân vẫn được giữ với các chuỗi còn lại. Không sửa bbox, không xóa file gốc.\n")
    a(f"- Series bị cách ly theo lớp: {dict(Counter(id_to_name[int(r['category_id'])] for r in quar))}")
    a(f"- BN bị ảnh hưởng theo lớp: {dict(Counter(id_to_name[int(c)] for c in q_pat.values()))}")
    a(f"- Số series bị cách ly / BN: {dict(sorted(Counter(Counter(r['patient_id'] for r in quar).values()).items()))} "
      "(không BN nào mất cả 8 chuỗi)")
    a("- Overlay minh họa: `figures/audit/mri_lowiou_check.png`.\n")
    a("### Volume CT dùng được\n")
    usable = [r for r in ct_rows if yes(r.get("ct_exists")) and (r["cancer_type"] == "CN" or (
        r.get("lesion_empty") == "False" and r.get("lesion_shape_match_ct") == "True"))]
    per_pid = Counter(r["patient_id"] for r in usable)
    a(f"Một volume được tính là dùng được nếu có file CT và (với lớp ung thư) lesion mask không rỗng, đúng shape. "
      f"Kết quả: **{len(usable)}/{len(ct_rows)}** ({dict(Counter(r['cancer_type'] for r in usable))}). "
      f"Số BN theo số pha dùng được: {dict(sorted(Counter(per_pid.values()).items(), reverse=True))}; "
      f"BN có 0 pha dùng được: {len({r['patient_id'] for r in ct_rows}) - len(per_pid)}. "
      "Liver mask rỗng (E14) không làm volume mất tác dụng cho lesion detection.\n")

    a("## 5. Đọc thử 5 CT + 5 MRI và ảnh overlay\n")
    a("| Modality | Bệnh nhân | Nhãn (từ CSV/JSON) | Ghi chú | Ảnh |\n|---|---|---|---|---|")
    for c in samples["ct"]:
        notes = []
        for ph, i in c["phases"].items():
            if i.get("status"):
                notes.append(f"{ph}: thiếu file")
            elif i.get("ct_channel_note", {}).get("converted"):
                notes.append(f"{ph}: 5D→3D")
        a(f"| CT | {c['patient_id']} | {c['cancer_type']} | {', '.join(notes) or 'OK'} | "
          f"[png](figures/audit/ct_{c['patient_id']}.png) |")
    for m in samples["mri"]:
        ious = [s.get("iou_conventionA_x=i_y=j") for s in m["series"].values()]
        a(f"| MRI | {m['patient_id']} | {m['category']} | IoU JSON–mask {min(ious):.2f}–{max(ious):.2f} | "
          f"[png](figures/audit/mri_{m['patient_id']}.png) |")
    a("\nCT: xanh lá = liver mask, đỏ = lesion mask. MRI: đỏ = label mask, vàng = JSON 2D_box. "
      "Mask khớp giải phẫu trên toàn bộ mẫu đã xem.\n")

    a("## 6. Lỗi và dữ liệu không hợp lệ\n")
    a("| # | Vấn đề | Quy mô | Mức độ | Cách xử lý (không sửa dữ liệu gốc) |\n|---|---|---|---|---|")
    a(f"| E1 | File CT có trong CSV nhưng không tồn tại | {len(missing_ct)} volume / {len({x['patient_id'] for x in missing_ct})} BN | "
      "Trung bình | Bài báo nói mọi BN đủ 4 pha, nên đây là **bản tải về không đầy đủ** (HF cache chỉ có 1404 metadata). "
      "Loại các volume này khỏi manifest; bệnh nhân vẫn được giữ với các pha còn lại. Không tải lại (quy tắc 4). |")
    a(f"| E2 | CT lưu dạng 5D (X,Y,Z,1,3) | {fc['ct_ndim']} · {len(fc['ct_5d_patients'])} BN | Thấp | "
      "Khi đọc: kiểm tra 3 kênh giống nhau rồi lấy kênh 0 (`nifti_io.as_3d`); nếu các kênh khác nhau thì báo lỗi. "
      "Audit toàn bộ chỉ đọc header; việc 3 kênh giống nhau mới được xác nhận trên các ca mẫu (P0110, P0172, P0320), "
      "phần còn lại sẽ được kiểm tra khi xuất lát cắt. |")
    a(f"| E3 | Lesion mask lưu raw 0/65535 kèm slope 1/65535 (đọc ra ≈ 0.9999999998) | {n_65535} | Thấp | "
      "Đọc giá trị raw (`get_unscaled`) rồi nhị phân hóa `> 0`. |")
    a(f"| E4 | Liver mask không bao trọn tổn thương (tổn thương nằm trong liver mask < 50%) | {len(low_liver)} volume | "
      "Trung bình | Liver mask là nhu mô gan và có thể loại trừ khối u. Khi cần vùng gan: dùng **hợp liver ∪ lesion** (chỉ trên train/val). |")
    a(f"| E5 | Tổn thương nằm ngoài bbox gan (< 90%) | {len(out_liver_bbox)} volume | Thấp/TB | Như E4. Kiểm tra trực quan khi sinh nhãn YOLO. |")
    a(f"| E6 | Lesion mask rỗng (BN ung thư nhưng mask trống) | {fc['lesion_empty']}: {', '.join(fc['lesion_empty_rows'])} | "
      f"{'TB' if fc['lesion_empty'] else '—'} | **Không coi là âm tính** (BN có ung thư); loại volume khỏi detection và classification. |")
    a(f"| E7 | Spacing header CT luôn là {fc['ct_zooms']} trong khi số lát cắt dao động {min(z_range)}–{max(z_range)} | toàn bộ | "
      "Trung bình | **Không tin được spacing thật**, nên không dùng phép đo thể tích theo mm. Pipeline 2D không bị ảnh hưởng. |")
    a("| E8 | CT đã windowing sẵn [-200, 200] HU về uint8 | toàn bộ | Thông tin | Không áp HU window; giữ 0–255 và chuẩn hóa theo thống kê của train. |")
    low_iou_seq = Counter(r["sequence"] for r in low_iou)
    a(f"| E9 | MRI: JSON 3D_box lệch với mask MedSAM2 (IoU < 0.3) | {fm['iou_lt_0.3']} series / "
      f"{len({r['patient_id'] for r in low_iou})} BN; theo chuỗi {dict(low_iou_seq)} | TB | "
      "Điều tra ở mục 4b: **không xác minh được nguyên nhân**. → **Cách ly toàn bộ** khỏi huấn luyện YOLO và ROI classification "
      "(`data/quarantine/mri_bbox_quarantine.csv`). Không sửa bbox, không xóa file. "
      f"Ví dụ: {', '.join(r['patient_id'] + ' ' + r['sequence'] + ' ' + r['iou_json3d_vs_mask_xy'] for r in low_iou[:5])} |")
    a(f"| E10 | MRI: mask có nhiều thành phần liên thông | {len(multi_comp_mri)} series | Thấp | "
      "num_targets = 1 nên có thể là mảnh vụn của mask. Khi tạo bbox, lấy thành phần lớn nhất hoặc bbox hợp, quyết định ở GĐ2. |")
    a("| E11 | **Confounder nhân khẩu học**: CN 15 nam/68 nữ, tuổi TB 45.1; nhóm ung thư đa số nam, tuổi 56–64 | 83 BN | "
      "**Cao** (bias) | Không dùng tuổi/giới tính làm đặc trưng; nêu trong phần hạn chế; split phân tầng theo nhãn. |")
    a("| E12 | Không có mapping chính thức C1/C2/C3/P → pha lâm sàng; overlay P0320 mâu thuẫn với giả thuyết C1 = arterial | 4 pha | "
      "Thấp/TB | Xử lý từng pha độc lập theo tên file; không gán tên pha lâm sàng khi chưa có nguồn. Nếu cần, kiểm tra thêm bằng "
      "cường độ động mạch chủ (GĐ1b). |")
    a("| E13 | MRI: không có danh sách tách train/test chính thức | 420 BN | Trung bình | Xem mục 7. |")
    a(f"| E14 | Liver mask rỗng | {len(liver_empty)} volume / {len({x.split('_')[0] for x in liver_empty})} BN: "
      f"{', '.join(liver_empty)} (đều là ICC và dạng 5D; P0112/P0193 có tổn thương rất lớn, khoảng 0.4–0.55 triệu voxel) | "
      "Trung bình | Không sinh nhãn YOLO lớp `liver` cho các volume này. Vẫn dùng được lesion mask "
      "(và lát cắt trong z-range của tổn thương). |")
    a(f"| E15 | Lesion mask khác shape với CT (lệch số lát cắt) | {len(les_shape_bad)}: {', '.join(les_shape_bad)} | "
      "Cao (cho volume đó) | P0231: mask C3 có 132 lát, bằng CT C2, nhưng không trùng voxel với mask C2; mức chồng lên liver "
      "C3 và C2 là 0.91 và 0.93 → **không xác định được pha thật**. **Loại lesion mask của volume này**; các pha khác vẫn dùng được. |")
    a(f"| E16 | MRI: khoảng z của JSON và mask không giao nhau | {len(z_bad)}: {'; '.join(z_bad)} | TB | "
      "Thuộc nhóm E9 (overlay `mri_lowiou_check.png`). Xử lý theo phương án E9 mà người dùng chốt. |")
    many_comp = [f"{r['patient_id']}_{r['phase']} ({r['lesion_components']})" for r in ct_rows
                 if r.get("lesion_components") and int(r["lesion_components"]) >= 5]
    a(f"| E17 | CT: lesion mask có nhiều thành phần liên thông (≥ 5) | {len(many_comp)}: {', '.join(many_comp)} | Thấp | "
      "Có thể là đa ổ hoặc mảnh vụn của annotation. Khi sinh bbox YOLO: mỗi thành phần đủ lớn là một box, mảnh vụn thì lọc theo diện tích "
      "(ngưỡng chốt ở GĐ2 trên train/val). |\n")

    a("## 7. Đánh giá điều kiện chia Train/Validation/Test\n")
    a("**CT (361 BN):** Lớp nhỏ nhất là CN với 83 BN. Split 70/15/15 phân tầng cho khoảng 58/12–13/12–13 BN cho lớp nhỏ nhất. "
      "→ **Đủ điều kiện.** Split ở mức bệnh nhân; volume thiếu được loại trước khi split nhưng bệnh nhân vẫn được giữ.\n")
    a("**MRI (498 BN):** Hướng người dùng chọn là ưu tiên split chính thức nếu ánh xạ được chính xác. Mức ánh xạ được:\n")
    a("- **Validation chính thức (78 BN): ánh xạ được.** 78/78 ID có trong MedSAM2; mỗi ID là đúng một BN với 8 series; "
      "không trùng lặp. Mức tin cậy: suy luận (78 ID trong file dự đoán baseline = kích thước tập val).")
    a("- **Train (316) / Test (104) chính thức: không ánh xạ được**, vì danh sách không được công bố công khai "
      "(chỉ có trong gói dữ liệu gốc mà project không có). 316 + 104 = 420 = đúng số ID còn lại.")
    a("- Vì vậy chỉ có thể giữ phần val chính thức. 420 BN còn lại phải tự chia Train/Test (phân tầng theo lớp, **mức bệnh nhân**, "
      "seed cố định, kích thước 316/104 như challenge). Tập Test này **không phải** test chính thức của challenge, "
      "và kết quả không so sánh trực tiếp được với leaderboard. Chi tiết trong `docs/PHASE2_PLAN.md`.\n")

    (REP / "audit_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Written: {REP / 'audit_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
