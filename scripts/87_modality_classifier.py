"""CT vs MRI recognition for 2D images (website intake). A SEPARATE small model; the locked AI pipeline is not touched.

Stages
  build     slices of Train and Validation patients (frozen patient-level splits; Test is never read) from the exported
            datasets; each slice in 3 variants, all passed through the website's screenshot pre-processing
            (livercancer.intake.image2d.prepare_2d): clean slice, degraded export (resize / gamma / blur / JPEG),
            synthetic screenshot or photo of a screen (livercancer.intake.synthetic). -> data/processed/modality/*.npz
  train     small CNN (livercancer.intake.modality.build_model), Train patients only; Validation patients split in two
            fixed halves: Val-A selects the epoch and the confidence threshold, Val-B is only used for reporting.
  eval      Val-B accuracy / coverage per variant at the chosen threshold, confusion; header rule check on NIfTI
            (Train + Val files); writes configs/modality.yaml and reports/modality_recognition.md
  yolo_shot how much the LOCKED YOLO detectors lose when the same Val lesion slices arrive as synthetic screenshots
            (detection of the GT lesion box, IoU >= 0.3, conf 0.12) - a description of the domain gap, not a tuning step.
Threshold rule (Val-A): smallest confidence in a grid whose accepted accuracy is >= 99.5 % on every variant.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import CONFIG_DIR, load_yaml  # noqa: E402
from livercancer.intake.image2d import find_body, prepare_2d  # noqa: E402
from livercancer.intake.modality import INPUT, build_model, to_input  # noqa: E402
from livercancer.intake.synthetic import degrade, jpeg, photo_of_screen, render_screenshot  # noqa: E402

DATA = ROOT / "data" / "processed" / "modality"
RUN = ROOT / "outputs" / "runs" / "modality" / "modality_cnn_s42"
REP = ROOT / "reports" / "modality_recognition.md"
SRC = {"ct": ROOT / "data/processed/full/yolo_ct", "mri": ROOT / "data/processed/pilot/yolo_mri"}
PER_PATIENT = {"train": 8, "val": 8}
VARIANTS = ("clean", "degraded", "screenshot")
SEED = 42


def rows(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def val_half(pid: str) -> str:
    return "A" if int(hashlib.sha256(pid.encode()).hexdigest(), 16) % 2 == 0 else "B"


# ----------------------------------------------------------------------------- build
def build(args) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        rng = random.Random(f"{SEED}-{split}")
        X, y, pid_l, var_l, refused = [], [], [], [], defaultdict(int)
        for mi, mod in enumerate(("ct", "mri")):
            sl = [r for r in rows(SRC[mod] / "slices.csv") if r["split"] == split]
            assert {r["split"] for r in sl} == {split}
            by_p = defaultdict(list)
            for r in sl:
                by_p[r["patient_id"]].append(r)
            for n, (pid, rs) in enumerate(sorted(by_p.items())):
                pick = rng.sample(rs, min(PER_PATIENT[split], len(rs)))
                imgs = [cv2.imread(str(SRC[mod] / "images" / split / f"{r['file']}.png"), cv2.IMREAD_GRAYSCALE) for r in pick]
                for j, g in enumerate(imgs):
                    seed = rng.randrange(1 << 30)
                    v = VARIANTS[j % 3]
                    if v == "clean":
                        src = g
                    elif v == "degraded":
                        src = degrade(g, seed)
                    else:
                        shot, _ = render_screenshot([g] + imgs[:3], seed=seed)
                        src = photo_of_screen(shot, seed) if seed % 3 == 0 else jpeg(shot, rng.randint(40, 95))
                    p = prepare_2d(src)
                    if not p.ok:
                        refused[(mod, v)] += 1
                        continue
                    X.append(to_input(p.gray))
                    y.append(mi)
                    pid_l.append(pid)
                    var_l.append(v)
                if n % 50 == 0:
                    print(f"  {split} {mod} {n}/{len(by_p)} patients, {len(X)} images", flush=True)
        np.savez_compressed(DATA / f"{split}.npz", X=np.stack(X), y=np.array(y), pid=np.array(pid_l), variant=np.array(var_l))
        (DATA / f"{split}_refused.json").write_text(json.dumps({f"{m}|{v}": c for (m, v), c in refused.items()}), encoding="utf-8")
        print(f"{split}: {len(X)} images, refused by pre-processing {dict(refused)}", flush=True)


# ----------------------------------------------------------------------------- train
def _augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(x)
    for i, img in enumerate(x):
        g = img.astype(np.float32)
        if rng.random() < 0.5:
            g = g[:, ::-1]
        s = rng.uniform(0.85, 1.0)
        c = int(INPUT * s)
        y0, x0 = rng.integers(0, INPUT - c + 1, 2)
        g = cv2.resize(np.ascontiguousarray(g[y0:y0 + c, x0:x0 + c]), (INPUT, INPUT))
        g = np.clip(255 * (g / 255.0) ** rng.uniform(0.8, 1.25) * rng.uniform(0.85, 1.15) + rng.normal(0, rng.uniform(0, 4), g.shape), 0, 255)
        out[i] = g.astype(np.uint8)
    return out


def train(args) -> None:
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    tr, va = np.load(DATA / "train.npz"), np.load(DATA / "val.npz")
    a = np.array([val_half(p) == "A" for p in va["pid"]])
    Xtr, ytr = tr["X"], tr["y"]
    Xa, ya = va["X"][a], va["y"][a]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    epochs = args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * ((len(Xtr) + 127) // 128))
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(len(ytr) / (2 * np.bincount(ytr)), dtype=torch.float32, device=dev))
    rng = np.random.default_rng(SEED)
    best, hist = (1e9, -1), []
    RUN.mkdir(parents=True, exist_ok=True)

    def run_eval(X):
        model.eval()
        ps = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                xb = torch.from_numpy(X[i:i + 512].astype(np.float32) / 255)[:, None].to(dev)
                ps.append(torch.softmax(model(xb).float(), 1).cpu().numpy())
        return np.concatenate(ps)

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        idx = rng.permutation(len(Xtr))
        tot = 0.0
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            xb = torch.from_numpy(_augment(Xtr[b], rng).astype(np.float32) / 255)[:, None].to(dev)
            yb = torch.from_numpy(ytr[b]).long().to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = lossf(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(b)
        pa = run_eval(Xa)
        vloss = float(-np.log(np.clip(pa[np.arange(len(ya)), ya], 1e-7, 1)).mean())
        vacc = float((pa.argmax(1) == ya).mean())
        hist.append({"epoch": ep + 1, "train_loss": tot / len(idx), "valA_loss": vloss, "valA_acc": vacc, "s": round(time.time() - t0, 1)})
        print(f"  epoch {ep + 1}/{epochs} train {tot / len(idx):.4f} | Val-A loss {vloss:.4f} acc {vacc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vloss < best[0]:
            best = (vloss, ep + 1)
            torch.save(model.state_dict(), RUN / "best.pt")
    (RUN / "history.json").write_text(json.dumps({"history": hist, "best_epoch": best[1], "selected_on": "Val-A loss",
                                                  "n_train": int(len(Xtr)), "n_valA": int(len(Xa)), "seed": SEED, "input": INPUT}, indent=1),
                                      encoding="utf-8")
    print("best epoch", best[1])


# ----------------------------------------------------------------------------- eval
def _probs(X):
    import torch
    model = build_model()
    model.load_state_dict(torch.load(RUN / "best.pt", map_location="cpu", weights_only=True))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            out.append(torch.softmax(model(torch.from_numpy(X[i:i + 512].astype(np.float32) / 255)[:, None].to(dev)).float(), 1).cpu().numpy())
    return np.concatenate(out)


def evaluate(args) -> None:
    va = np.load(DATA / "val.npz")
    half = np.array([val_half(p) for p in va["pid"]])
    p = _probs(va["X"])
    conf, pred, y, var = p.max(1), p.argmax(1), va["y"], va["variant"]
    grid = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
    choice = None
    for t in grid:      # Val-A only
        ok = True
        for v in VARIANTS:
            m = (half == "A") & (var == v) & (conf >= t)
            if m.sum() == 0 or (pred[m] == y[m]).mean() < 0.995:
                ok = False
        if ok:
            choice = t
            break
    choice = choice or grid[-1]

    def block(mask):
        acc_all = float((pred[mask] == y[mask]).mean())
        acc_m = mask & (conf >= choice)
        return {"n": int(mask.sum()), "accuracy_without_threshold": acc_all, "accepted": int(acc_m.sum()),
                "coverage": float(acc_m.sum() / max(1, mask.sum())),
                "accepted_accuracy": float((pred[acc_m] == y[acc_m]).mean()) if acc_m.any() else None,
                "accepted_errors": int((pred[acc_m] != y[acc_m]).sum()),
                "errors_ct_as_mri": int(((pred == 1) & (y == 0) & acc_m).sum()), "errors_mri_as_ct": int(((pred == 0) & (y == 1) & acc_m).sum())}
    res = {"threshold": choice, "threshold_rule": "smallest grid value with accepted accuracy >= 99.5% on every variant of Val-A",
           "valB": {v: block((half == "B") & (var == v)) for v in VARIANTS}, "valB_all": block(half == "B"),
           "valA_all": block(half == "A"), "n_patients_valB": int(len({q for q, h in zip(va["pid"], half) if h == "B"})),
           "refused_by_preprocessing_val": json.loads((DATA / "val_refused.json").read_text(encoding="utf-8")),
           "history": json.loads((RUN / "history.json").read_text(encoding="utf-8"))}
    # NIfTI header rule on every Train + Val image file (header only)
    res["nifti_header_rule"] = header_rule_check()
    (RUN / "metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    sha = hashlib.sha256((RUN / "best.pt").read_bytes()).hexdigest()
    (CONFIG_DIR / "modality.yaml").write_text(
        "# CT vs MRI recognition of 2D images (website intake). Separate model, NOT part of the locked AI pipeline.\n"
        "# Trained on Train-patient slices + synthetic screenshot variants; epoch and threshold chosen on Val-A,\n"
        "# reported on Val-B (scripts/87_modality_classifier.py, reports/modality_recognition.md). Test never used.\n"
        f"checkpoint: {RUN.relative_to(ROOT).as_posix()}/best.pt\nsha256: {sha}\ninput: {INPUT}\n"
        f"threshold: {choice}          # API \"auto\" only (the website uses the user's CT / MRI choice); below it the request is refused\n"
        f"metrics: {RUN.relative_to(ROOT).as_posix()}/metrics.json\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("threshold", "valB", "valB_all", "nifti_header_rule")}, indent=1))


def header_rule_check() -> dict:
    sys.path.insert(0, str(ROOT))
    from livercancer.config import load_paths
    from livercancer.data.sources import image_path
    from server.inspect_file import inspect_head
    paths = load_paths()
    out = {}
    for mod in ("ct", "mri"):
        split = {r["patient_id"]: r["split"] for r in rows(paths["project"]["splits"] / f"{mod}_split.csv")}
        vols = [r for r in rows(SRC[mod] / "volumes.csv") if split[r["patient_id"]] in ("train", "val")]
        got = defaultdict(int)
        for r in vols:
            f = image_path(paths, mod, r["patient_id"], r["series"])
            if not f.exists():
                continue
            with open(f, "rb") as fh:
                head = fh.read(1 << 18)
            got[inspect_head(0, "anh.nii.gz", head, f.stat().st_size, 1 << 40).modality] += 1
        out[mod] = dict(got)
    return out


# ----------------------------------------------------------------------------- YOLO on screenshots
def yolo_shot(args) -> None:
    from livercancer.detection.yolo_detector import LesionDetector
    from livercancer.roi.boxes import iou
    rng = random.Random(SEED)
    out = {}
    for mod in ("ct", "mri"):
        det = LesionDetector.from_config(mod)
        sl = [r for r in rows(SRC[mod] / "slices.csv") if r["split"] == "val" and r["kind"] == "pos"]
        rng.shuffle(sl)
        sl = sl[:args.n]
        hit = {"original": 0, "screenshot": 0, "photo": 0}
        refused = {"screenshot": 0, "photo": 0}
        n = 0
        for i, r in enumerate(sl):
            g = cv2.imread(str(SRC[mod] / "images" / "val" / f"{r['file']}.png"), cv2.IMREAD_GRAYSCALE)
            h, w = g.shape
            lines = (SRC[mod] / "labels" / "val" / f"{r['file']}.txt").read_text().split()
            gts = []
            for k in range(0, len(lines), 5):
                cx, cy, bw, bh = map(float, lines[k + 1:k + 5])
                gts.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
            n += 1
            d0 = det.predict([g])[0]
            hit["original"] += any(iou((d.x0, d.y0, d.x1, d.y1), b) >= 0.3 for d in d0 for b in gts)
            for kind in ("screenshot", "photo"):
                shot, info = render_screenshot([g], seed=i, panels=1)
                shot = jpeg(shot, 80) if kind == "screenshot" else photo_of_screen(shot, i)
                p = prepare_2d(shot, mod)
                if not p.ok:
                    refused[kind] += 1
                    continue
                vx0, vy0, vx1, vy1 = info["viewport"]
                s_v = (vx1 - vx0) / w
                cx0, cy0, cx1, cy1 = p.crop
                s_c = 512 / (cx1 - cx0)
                mapped = [((b[0] * s_v + vx0 - cx0) * s_c, (b[1] * s_v + vy0 - cy0) * s_c, (b[2] * s_v + vx0 - cx0) * s_c, (b[3] * s_v + vy0 - cy0) * s_c) for b in gts]
                d1 = det.predict([p.gray])[0]
                hit[kind] += any(iou((d.x0, d.y0, d.x1, d.y1), b) >= 0.3 for d in d1 for b in mapped)
        out[mod] = {"n_lesion_slices": n, "detected": hit, "refused": refused}
        print(mod, out[mod], flush=True)
    m = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    m["yolo_on_synthetic_screenshots"] = out
    (RUN / "metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- report
def report(args) -> None:
    m = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    vn = {"clean": "Lát cắt xuất gốc (PNG)", "degraded": "Ảnh xuất bị nén / đổi kích thước / độ sáng", "screenshot": "Ảnh chụp màn hình / chụp lại màn hình (tổng hợp)"}
    pct = lambda x: "–" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    L = ["# Nhận dạng tự động ảnh CT / MRI (bước tiếp nhận của website)", "",
         f"_Sinh bởi `scripts/87_modality_classifier.py` lúc {datetime.now():%Y-%m-%d %H:%M}. Mô hình riêng, **không** thuộc pipeline AI đã khóa; "
         "không dùng tập Test._", "",
         "## Thứ tự quyết định", "",
         "1. **DICOM:** thẻ `Modality` (CT / MR).",
         "2. **NIfTI:** hình học header. CT: 512×512, voxel 1 mm đẳng hướng. MRI: lát cắt dày ≥ 2 mm và dày ≥ 1,5 lần độ phân giải trong lát.",
         "3. **Ảnh 2D (PNG/JPG, ảnh chụp màn hình, DICOM dạng chụp màn hình):** CNN nhỏ. Chỉ tự quyết khi độ tin cậy ≥ ngưỡng chọn trên Val-A; "
         "nếu thấp hơn, website hỏi người dùng. **Không dùng tên tệp.**", "",
         "## Kiểm tra quy tắc header NIfTI (mọi tệp Train + Val)", "",
         "| Ảnh thật | Nhận là CT | Nhận là MRI | Không xác định |", "|---|---|---|---|"]
    for mod in ("ct", "mri"):
        g = m["nifti_header_rule"][mod]
        L.append(f"| {mod.upper()} | {g.get('ct', 0)} | {g.get('mri', 0)} | {g.get('unknown', 0)} |")
    h = m["history"]
    L += ["", "## Mô hình ảnh 2D", "",
          f"- **Dữ liệu huấn luyện:** {h['n_train']} ảnh từ bệnh nhân Train. Mỗi lát cắt có 3 biến thể và đi qua cùng bước tiền xử lý của website.",
          f"- **Chọn epoch:** epoch {h['best_epoch']}, chọn trên Val-A (nửa số bệnh nhân Validation, {h['n_valA']} ảnh).",
          f"- **Ngưỡng tự quyết:** **{m['threshold']}**, là giá trị nhỏ nhất trên lưới mà mọi loại ảnh của Val-A đều đúng ≥ 99,5% trong các ca tự quyết.",
          f"- **Báo cáo trên Val-B:** {m['n_patients_valB']} bệnh nhân Validation khác, không dùng để chọn gì.", "",
          "| Loại ảnh (Val-B) | Số ảnh | Đúng (không ngưỡng) | Tự quyết (≥ ngưỡng) | Đúng trong ca tự quyết | Nhầm CT↔MRI khi tự quyết |", "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        b = m["valB"][v]
        L.append(f"| {vn[v]} | {b['n']} | {pct(b['accuracy_without_threshold'])} | {pct(b['coverage'])} | {pct(b['accepted_accuracy'])} | "
                 f"{b['accepted_errors']} (CT→MRI {b['errors_ct_as_mri']}, MRI→CT {b['errors_mri_as_ct']}) |")
    b = m["valB_all"]
    L.append(f"| **Tất cả** | {b['n']} | {pct(b['accuracy_without_threshold'])} | {pct(b['coverage'])} | {pct(b['accepted_accuracy'])} | {b['accepted_errors']} |")
    r = m.get("refused_by_preprocessing_val", {})
    L += ["", f"Bị bước tiền xử lý từ chối trước khi tới mô hình (Val, không tách được lát cắt): {r or 'không có'}.", ""]
    ys = m.get("yolo_on_synthetic_screenshots")
    if ys:
        L += ["## YOLO đã khóa trên ảnh chụp màn hình (mô tả độ lệch miền, không chỉnh gì)", "",
              "Cùng lát cắt Val có tổn thương, đưa vào YOLO theo 3 dạng. Tính tỷ lệ tìm thấy tổn thương chuẩn (IoU ≥ 0,3, conf 0,12):", "",
              "| | Số lát | Ảnh gốc | Ảnh chụp màn hình (JPEG) | Ảnh chụp lại màn hình | Bị từ chối (chụp màn hình / chụp lại) |", "|---|---|---|---|---|---|"]
        for mod in ("ct", "mri"):
            x = ys[mod]
            n = x["n_lesion_slices"]
            L.append(f"| {mod.upper()} | {n} | {x['detected']['original']}/{n} | {x['detected']['screenshot']}/{n} | {x['detected']['photo']}/{n} | "
                     f"{x['refused']['screenshot']} / {x['refused']['photo']} |")
        L += ["", "Ca bị từ chối được tính là không tìm thấy."]
    L += ["", "## Giới hạn", "",
          "- **Dữ liệu huấn luyện và đánh giá:** chỉ gồm hai bộ dữ liệu (CT: PLC-CECT, MRI: LLD-MMRI). Mô hình có thể học đặc trưng riêng của từng bộ "
          "thay vì đặc trưng CT/MRI; **chưa kiểm chứng trên ảnh bệnh viện thật**.",
          "- **Ảnh chụp màn hình:** do chương trình tạo từ lát cắt Val, không phải ảnh chụp thật; độ chính xác trên ảnh thật có thể thấp hơn.",
          "- **Ảnh không phải CT/MRI** (X-quang, siêu âm, ảnh thường): không có trong dữ liệu. Bước tiền xử lý chặn ảnh màu hoặc ảnh không có lát cắt trên "
          "nền tối, nhưng không đảm bảo chặn hết.",
          "- Khi độ tin cậy dưới ngưỡng, website hỏi người dùng và không tự đưa ảnh vào YOLO CT hay MRI."]
    REP.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("written", REP)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["build", "train", "eval", "yolo_shot", "report", "all"], required=True)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--n", type=int, default=200, help="yolo_shot: lesion slices per modality")
    args = ap.parse_args()
    stages = ["build", "train", "eval", "yolo_shot", "report"] if args.stage == "all" else [args.stage]
    for s in stages:
        print("==", s, flush=True)
        {"build": build, "train": train, "eval": evaluate, "yolo_shot": yolo_shot, "report": report}[s](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
