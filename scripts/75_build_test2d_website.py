"""2D test images (PNG slices) for manual testing of the website's "Phân tích ảnh 2D" mode -> TEST_2D_WEBSITE/.

Companion of scripts/99_build_test3d_website.py (TEST_3D_WEBSITE/): same locked Test set, same 10 quick patients
(TEST_NHANH: CT 2 HCC, 2 ICC, 1 CHCC; MRI 1 HCC, ICC, metastasis, hemangioma, cyst), plus CT control (CN) slices.
Nothing is chosen from a model output:
  * lesion slices: for every phase / sequence of a quick patient, the Test slice (scripts/96_final_test.py export,
    data/processed/final_test/yolo_<mod>/) with the largest GT box (ties -> lowest slice). The PNG is copied byte for
    byte and checked pixel for pixel against the slice of the 3D NIfTI in TEST_3D_WEBSITE (same preprocessing as
    training, livercancer.data.slices).
  * CT control slices: first 3 CN Test patients (ID order) with a non-empty liver mask; per phase on disk, the slice
    with the largest liver-mask area. CN = "without hepatobiliary disease", no lesion recorded (mask_path NA); this is
    NOT "healthy", and a slice without a lesion never means a healthy patient.
  * DAP_AN/: the same slice with the GT box drawn (green) for comparison by eye. Never upload these to the website.
These are locked Test images: website demonstration only, never tuning or a new result. Folder is git-ignored.
MRI is CC BY-NC 4.0 ("not to further copy, publish, or distribute").

    .venv\\Scripts\\python scripts\\75_build_test2d_website.py
"""
from __future__ import annotations

import csv
import hashlib
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import load_paths  # noqa: E402
from livercancer.data.nifti_io import as_3d  # noqa: E402
from livercancer.data.slices import load_volume_uint8, slice_image  # noqa: E402

DEST = ROOT / "TEST_2D_WEBSITE"
T3D = ROOT / "TEST_3D_WEBSITE"
FT = ROOT / "data" / "processed" / "final_test"
CT_PHASES = ("C1", "C2", "C3", "P")
N_CN = 3
SEQ_FILE = {"In Phase": "InPhase", "Out Phase": "OutPhase"}
LABEL_VI = {"HCC": "HCC", "ICC": "ICC", "CHCC": "CHCC (cHCC-CCA)", "CN": "CN – nhóm chứng (không ghi nhận tổn thương)",
            "Hepatocellular_carcinoma": "HCC (ung thư biểu mô tế bào gan)", "Intrahepatic_cholangiocarcinoma": "ICC (ung thư đường mật trong gan)",
            "Hepatic_metastasis": "Di căn gan", "Hepatic_hemangioma": "U máu gan", "Hepatic_cyst": "Nang gan"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def gt_boxes(label_txt: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
    out = []
    for line in label_txt.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            _, cx, cy, bw, bh = map(float, line.split())
            out.append((round((cx - bw / 2) * w), round((cy - bh / 2) * h), round((cx + bw / 2) * w), round((cy + bh / 2) * h)))
    return out


def draw_answer(gray: np.ndarray, boxes, text: str) -> np.ndarray:
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)
    return img


def quick_volumes() -> dict[tuple[str, str], list[dict]]:
    """(modality, patient) -> rows of DANH_SACH_TEST_3D.csv of the quick set, in file order."""
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in read_csv(T3D / "DANH_SACH_TEST_3D.csv"):
        if r["quick_test"] == "co" and r["copied"] == "co":
            out[(r["modality"], r["patient_id"])].append(r)
    if not out:
        raise SystemExit("TEST_3D_WEBSITE/DANH_SACH_TEST_3D.csv has no quick set: run scripts/99_build_test3d_website.py first")
    return out


def lesion_rows(quick) -> list[dict]:
    rows = []
    for mod in ("CT", "MRI"):
        base = FT / f"yolo_{mod.lower()}"
        best: dict[tuple[str, str], tuple] = {}
        for s in read_csv(base / "slices.csv"):
            if s["kind"] != "pos" or (mod, s["patient_id"]) not in quick:
                continue
            w, h = int(s["width"]), int(s["height"])
            boxes = gt_boxes(base / "labels" / "test" / f"{s['file']}.txt", w, h)
            area = max((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes)
            key = (s["patient_id"], s["series"])
            cand = (area, -int(s["slice_k"]), s, boxes)
            if key not in best or cand[:2] > best[key][:2]:
                best[key] = cand
        for (mod_, pid), vols in quick.items():
            if mod_ != mod:
                continue
            for v in vols:
                if (pid, v["series"]) not in best:
                    rows.append({"modality": mod, "patient_id": pid, "label": v["label"], "series": v["series"], "skip": "không có lát tổn thương trong bộ Test đã xuất"})
                    continue
                area, _, s, boxes = best[(pid, v["series"])]
                rows.append({"modality": mod, "patient_id": pid, "label": v["label"], "series": v["series"], "slice_k": int(s["slice_k"]),
                             "n_slices": int(s["n_slices"]), "src_png": base / "images" / "test" / f"{s['file']}.png", "boxes": boxes,
                             "box_area_px": area, "nifti": Path(v["dest"]), "kind": "co_ton_thuong"})
    return rows


def cn_rows(paths) -> list[dict]:
    split = [r for r in read_csv(ROOT / "data" / "splits" / "ct_split.csv") if r["label"] == "CN" and r["split"] == "test"]
    meta = {(r["patient_id"], r["phase"]): r for r in read_csv(Path(paths["ct"]["csv"]))}
    data_dir = Path(paths["ct"]["data_dir"])
    rows, n_pat = [], 0
    for pid in sorted(r["patient_id"] for r in split):
        got = []
        for ph in CT_PHASES:
            m = meta.get((pid, ph))
            if not m:
                continue
            img_p, liv_p = data_dir / m["ct_path"], data_dir / m["liver_mask_path"]
            if not img_p.exists() or not liv_p.exists() or m["mask_path"] != "NA":
                continue
            liv = nib.load(str(liv_p))
            lm, _ = as_3d(np.asanyarray(liv.dataobj.get_unscaled()))
            lm = lm > 0
            area = lm.sum(axis=(0, 1))
            if area.max() == 0:
                continue
            k = int(np.argmax(area))
            got.append({"modality": "CT", "patient_id": pid, "label": "CN", "series": ph, "slice_k": k, "n_slices": int(lm.shape[2]),
                        "boxes": [], "box_area_px": 0, "liver_area_px": int(area[k]), "nifti": img_p, "liver_shape": lm.shape,
                        "kind": "khong_ton_thuong_nhom_CN"})
        if got:
            rows += got
            n_pat += 1
        if n_pat == N_CN:
            break
    return rows


def main() -> None:
    t0 = time.time()
    paths = load_paths()
    quick = quick_volumes()
    rows = lesion_rows(quick) + cn_rows(paths)
    if DEST.exists():
        for sub in ("CT", "MRI", "DAP_AN"):
            shutil.rmtree(DEST / sub, ignore_errors=True)
    out, errors = [], []
    for r in rows:
        if "skip" in r:
            out.append({**{k: r[k] for k in ("modality", "patient_id", "label", "series")}, "trang_thai": r["skip"]})
            continue
        vol = load_volume_uint8(r["nifti"])
        gray = slice_image(vol, r["slice_k"])
        if r["kind"] == "khong_ton_thuong_nhom_CN" and vol.shape != tuple(r["liver_shape"]):
            errors.append(f"{r['patient_id']} {r['series']}: liver mask shape {r['liver_shape']} != image {vol.shape}")
            continue
        series_f = SEQ_FILE.get(r["series"], r["series"])
        name = f"{r['modality']}_{r['patient_id']}_{series_f}_lat{r['slice_k'] + 1:03d}.png"
        sub = DEST / r["modality"] / r["kind"]
        sub.mkdir(parents=True, exist_ok=True)
        dst = sub / name
        if "src_png" in r:
            shutil.copyfile(r["src_png"], dst)
            same_bytes = sha256(dst) == sha256(r["src_png"])
            png = cv2.imread(str(dst), cv2.IMREAD_GRAYSCALE)
            pix = png is not None and png.shape == gray.shape and np.array_equal(png, gray)
            if not (same_bytes and pix):
                errors.append(f"{name}: byte copy {same_bytes}, pixel match with 3D NIfTI {pix}")
        else:
            cv2.imwrite(str(dst), gray)
            pix = np.array_equal(cv2.imread(str(dst), cv2.IMREAD_GRAYSCALE), gray)
        ans_dir = DEST / "DAP_AN" / r["modality"]
        ans_dir.mkdir(parents=True, exist_ok=True)
        cap = f"{r['label']} | {r['series']} | lat {r['slice_k'] + 1}/{r['n_slices']}" + ("" if r["boxes"] else " | khong co box GT")
        cv2.imwrite(str(ans_dir / name.replace(".png", "_dap_an.png")), draw_answer(gray, r["boxes"], cap))
        rel3d = r["nifti"].relative_to(ROOT) if r["nifti"].is_relative_to(ROOT) else f"(nguồn) {r['nifti'].name}"
        out.append({"modality": r["modality"], "patient_id": r["patient_id"], "label": r["label"], "series": r["series"],
                    "trang_thai": "ok", "nhom": r["kind"], "file": str(dst.relative_to(DEST)).replace("\\", "/"),
                    "lat_hien_thi": r["slice_k"] + 1, "slice_k": r["slice_k"], "n_slices": r["n_slices"],
                    "width": gray.shape[1], "height": gray.shape[0],
                    "gt_boxes_x1y1x2y2": ";".join(",".join(map(str, b)) for b in r["boxes"]), "box_area_px": r["box_area_px"],
                    "liver_area_px": r.get("liver_area_px", ""), "file_3d": str(rel3d).replace("\\", "/"),
                    "trung_pixel_voi_3d": "co" if pix else "KHONG", "sha256": sha256(dst)})
        del vol
    cols = ["modality", "patient_id", "label", "series", "trang_thai", "nhom", "file", "lat_hien_thi", "slice_k", "n_slices", "width", "height",
            "gt_boxes_x1y1x2y2", "box_area_px", "liver_area_px", "file_3d", "trung_pixel_voi_3d", "sha256"]
    with open(DEST / "DANH_SACH_TEST_2D.csv", "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(out)
    write_guide(out, errors, time.time() - t0)
    print(f"{sum(o['trang_thai'] == 'ok' for o in out)} images, {len(errors)} errors -> {DEST}")
    for e in errors:
        print("ERROR", e)
    if errors:
        raise SystemExit(1)


def write_guide(out: list[dict], errors: list[str], secs: float) -> None:
    ok = [o for o in out if o["trang_thai"] == "ok"]
    cnt = defaultdict(int)
    for o in ok:
        cnt[(o["modality"], o["nhom"])] += 1
    L = ["# Bộ ảnh 2D để test website", "",
         f"Tạo bởi `scripts/75_build_test2d_website.py` lúc {time.strftime('%Y-%m-%d %H:%M')} ({secs:.0f} s). Danh sách đầy đủ: `DANH_SACH_TEST_2D.csv`.", "",
         "**Đây là ảnh của tập Test đã khóa** (đánh giá chính thức một lần ngày 2026-10-26), cùng 10 bệnh nhân với `TEST_3D_WEBSITE/TEST_NHANH`. "
         "Chỉ dùng để trình diễn / kiểm thử website, **không** dùng để chỉnh mô hình, ngưỡng hay báo cáo thành kết quả mới.", "",
         "## Có gì trong thư mục", "",
         "| Thư mục | Nội dung | Số ảnh |", "|---|---|---|",
         f"| `CT/co_ton_thuong/` | Mỗi pha của 5 BN CT (2 HCC, 2 ICC, 1 CHCC): lát có tổn thương lớn nhất theo nhãn gốc | {cnt[('CT', 'co_ton_thuong')]} |",
         f"| `CT/khong_ton_thuong_nhom_CN/` | {N_CN} BN nhóm chứng CN đầu tiên (theo mã): lát có vùng gan lớn nhất | {cnt[('CT', 'khong_ton_thuong_nhom_CN')]} |",
         f"| `MRI/co_ton_thuong/` | Mỗi chuỗi của 5 BN MRI (HCC, ICC, di căn, u máu, nang): lát có tổn thương lớn nhất | {cnt[('MRI', 'co_ton_thuong')]} |",
         f"| `DAP_AN/` | Cùng lát đó, vẽ **khung xanh = vị trí tổn thương theo nhãn gốc** để so với khung của AI | {len(ok)} |", "",
         "Tên tệp: `<CT|MRI>_<mã BN>_<pha/chuỗi>_lat<số lát>.png`. Số lát đếm từ 1, **trùng với \"lát\" trên website** khi mở file 3D tương ứng "
         "(cột `file_3d` trong CSV), nên có thể đối chiếu 2D với 3D.", "",
         "Ảnh là lát cắt xám 512 × 512 (MRI theo kích thước gốc của chuỗi), **cùng cách tiền xử lý như lúc huấn luyện**. "
         f"Đã kiểm tra: mọi ảnh trùng từng điểm ảnh với lát tương ứng của file NIfTI 3D ({sum(o['trung_pixel_voi_3d'] == 'co' for o in ok)}/{len(ok)}); "
         f"lỗi: {len(errors)}.", "",
         "## Cách test trên website", "",
         "1. Mở website, chọn **Phân tích ảnh 2D**.",
         "2. Chọn **CT Scan** cho ảnh trong `CT/`, **MRI** cho ảnh trong `MRI/` (website không tự đoán theo tên tệp).",
         "3. Tải **một** ảnh PNG; hệ thống tự phân tích.",
         "4. Mở ảnh cùng tên trong `DAP_AN/` để so khung của AI với khung xanh của nhãn gốc.", "",
         "Kết quả mong đợi khi test (để kiểm tra giao diện, không phải để chấm mô hình):", "",
         "- **CT có tổn thương:** khung YOLO, Tumor Zoom; có nhãn HCC/ICC/CHCC **chỉ khi** Evidence Gate đủ bằng chứng. "
         "Với 1 ảnh 2D, cổng cần độ tin cậy cao nhất ≥ 0,35, nếu không sẽ báo \"chưa đủ bằng chứng\". Đó là hành vi đúng.",
         "- **CT nhóm chứng CN:** có thể không có khung, hoặc có khung yếu (YOLO ở ngưỡng 0,12 vẫn có thể báo vùng giả). "
         "\"Không phát hiện\" **không** có nghĩa là gan bình thường; website cũng ghi như vậy.",
         "- **MRI:** chỉ phát hiện + Tumor Zoom, **không** phân loại (mô hình phân loại MRI chưa đạt).",
         "- Ảnh trong `DAP_AN/` có vẽ khung và chữ: **không tải lên website** (khung xanh sẽ làm sai ảnh đầu vào).", "",
         "Muốn thử ảnh chụp màn hình: mở một ảnh PNG ở đây, chụp màn hình rồi tải ảnh chụp lên. Website sẽ tự tách lát cắt khỏi nền, "
         "nhưng chất lượng có thể giảm (ảnh chụp lại màn hình chưa được kiểm chứng trên dữ liệu bệnh viện thật).", "",
         "## Danh sách", "", "| Ảnh | Nhãn tham chiếu | Pha/chuỗi | Lát |", "|---|---|---|---|"]
    for o in ok:
        L.append(f"| `{o['file']}` | {LABEL_VI.get(o['label'], o['label'])} | {o['series']} | {o['lat_hien_thi']}/{o['n_slices']} |")
    skipped = [o for o in out if o["trang_thai"] != "ok"]
    if skipped:
        L += ["", "Không tạo được:", ""] + [f"- {o['modality']} {o['patient_id']} {o['series']}: {o['trang_thai']}" for o in skipped]
    L += ["", "## Quyền sử dụng", "",
          "- **CT (PLC-CECT, CC BY 4.0):** được dùng và chia sẻ, ghi nguồn.",
          "- **MRI (LLD-MMRI, CC BY-NC 4.0, \"not to further copy, publish, or distribute\"):** không phân phối lại / đăng công khai. "
          "Nên test MRI trên `http://127.0.0.1:8000`; không gửi thư mục `MRI` cho người khác.",
          "- Thư mục `TEST_2D_WEBSITE/` được loại khỏi Git. Không upload lên Internet."]
    (DEST / "HUONG_DAN_TEST_2D.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
