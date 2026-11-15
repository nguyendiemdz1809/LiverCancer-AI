"""Build two Roboflow-ready YOLO datasets (5000 CT + 5000 MRI slices) from the project's TRAIN split.

Purpose: visualise / inspect / present the existing labels on Roboflow. Nothing here trains a model or
creates labels: images and label files are copied byte-for-byte from the YOLO exports already built by
scripts/20_build_yolo_pilot.py (CT: data/processed/full/yolo_ct, MRI: data/processed/pilot/yolo_mri),
whose boxes were converted automatically from the source lesion masks.

Rules
- Only patients whose frozen split (data/splits/{ct,mri}_split.csv) is "train". No Val, no Test.
- Sources are read only. Output goes to data/processed/roboflow/ (gitignored) + reports/.
- Selection spreads images over patients (round robin) and volumes, keeps a minimum slice gap inside a
  volume, rejects near-identical images of the same patient (dHash) and near-blank images.
  Lesion slices whose largest box has a short side >= --clear-min-side px are taken first ("clear").
- File names in the ZIPs use neutral codes (CT_BN001_C1_k045); the mapping to dataset IDs stays local.
- If a modality cannot reach its quota the shortage is reported; nothing is duplicated.

Run: .venv\\Scripts\\python scripts\\88_build_roboflow_export.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import struct
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "CT": ROOT / "data/processed/full/yolo_ct",
    "MRI": ROOT / "data/processed/pilot/yolo_mri",
}
SPLITS = {"CT": ROOT / "data/splits/ct_split.csv", "MRI": ROOT / "data/splits/mri_split.csv"}
SERIES_CODE = {"C+A": "CA", "C+V": "CV", "C+Delay": "CDelay", "C-pre": "Cpre", "In Phase": "InPhase", "Out Phase": "OutPhase"}
LICENSE = {
    "CT": "PLC-CECT (Luo et al., Scientific Data 2025) - CC BY 4.0 (attribution required)",
    "MRI": "LLD-MMRI (Lou et al.) + MedSAM2 masks (wanglab/LLD-MMRI-MedSAM2) - CC BY-NC 4.0, "
           "terms: do not further copy, publish or distribute. Do NOT upload without the owner's decision.",
}
PNG_TEXT_CHUNKS = {b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME"}


# ---------------------------------------------------------------- helpers
def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_boxes(p: Path) -> list[list[float]]:
    out = []
    for line in p.read_text(encoding="ascii").splitlines():
        if line.strip():
            out.append([float(v) for v in line.split()])
    return out


def dhash(img: np.ndarray) -> np.ndarray:
    """256-bit difference hash (16x16 gradients), packed into 32 bytes."""
    small = cv2.resize(img, (17, 16), interpolation=cv2.INTER_AREA).astype(np.int16)
    return np.packbits((small[:, 1:] > small[:, :-1]).flatten())


def min_hamming(h: np.ndarray, others: list[np.ndarray]) -> int:
    if not others:
        return 10**6
    return int(np.unpackbits(np.bitwise_xor(np.stack(others), h), axis=1).sum(1).min())


def png_chunks(p: Path) -> set[bytes]:
    data = p.read_bytes()
    pos, kinds = 8, set()
    while pos + 8 <= len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        kinds.add(kind)
        pos += 12 + n
        if kind == b"IEND":
            break
    return kinds


# ---------------------------------------------------------------- pools
def load_pool(mod: str, clear_min: float) -> pd.DataFrame:
    src = SOURCES[mod]
    s = pd.read_csv(src / "slices.csv")
    split = pd.read_csv(SPLITS[mod])[["patient_id", "split"]].rename(columns={"split": "frozen_split"})
    s = s.merge(split, on="patient_id", how="left")
    assert s.frozen_split.notna().all(), "patient missing from frozen split"
    assert (s.split == s.frozen_split).all(), "export split differs from frozen split"
    t = s[s.frozen_split == "train"].copy()
    cache = ROOT / f"data/processed/roboflow/meta/box_sizes_{mod}.csv"  # speeds up reruns; rebuilt if stale
    if cache.exists():
        c = pd.read_csv(cache)
        if list(c.file) == list(t.file):
            t["box_short"], t["box_area"] = c.box_short.values, c.box_area.values
            t["clear"] = (t.kind == "pos") & (t.box_short >= clear_min)
            t["volume"] = t.patient_id + "|" + t.series
            return t.reset_index(drop=True)
    short, area = [], []
    for f, w, h in zip(t.file, t.width, t.height):
        b = read_boxes(src / "labels/train" / f"{f}.txt")
        if not b:
            short.append(0.0)
            area.append(0.0)
            continue
        dims = [(bw * w, bh * h) for _, _, _, bw, bh in b]
        i = int(np.argmax([a * c for a, c in dims]))
        short.append(min(dims[i]))
        area.append(dims[i][0] * dims[i][1])
    t["box_short"] = short
    t["box_area"] = area
    cache.parent.mkdir(parents=True, exist_ok=True)
    t[["file", "box_short", "box_area"]].to_csv(cache, index=False)
    t["clear"] = (t.kind == "pos") & (t.box_short >= clear_min)
    t["volume"] = t.patient_id + "|" + t.series
    return t.reset_index(drop=True)


class Picker:
    """Round robin over patients, then volumes; slice gap + near-duplicate + blank checks."""

    def __init__(self, mod: str, gap: int, dup_hamming: int, min_std: float, seed: int, cap: int):
        self.src = SOURCES[mod]
        self.cap = cap  # max images per patient (lesion + empty together)
        self.per_patient: dict[str, int] = defaultdict(int)
        self.gap, self.dup, self.min_std = gap, dup_hamming, min_std
        self.rng = random.Random(seed)
        self.taken_k: dict[str, list[int]] = defaultdict(list)  # volume -> slice indices
        self.hashes: dict[str, list[np.ndarray]] = defaultdict(list)  # patient -> dHash
        self.rejected = defaultdict(int)
        self.chosen: list[int] = []

    def _ok(self, row) -> bool:
        if any(abs(row.slice_k - k) < self.gap for k in self.taken_k[row.volume]):
            return False
        img = cv2.imread(str(self.src / "images/train" / f"{row.file}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.rejected["unreadable"] += 1
            return False
        if float(img.std()) < self.min_std:
            self.rejected["near_blank"] += 1
            return False
        h = dhash(img)
        if min_hamming(h, self.hashes[row.patient_id]) <= self.dup:
            self.rejected["near_duplicate_same_patient"] += 1
            return False
        self.taken_k[row.volume].append(int(row.slice_k))
        self.hashes[row.patient_id].append(h)
        self.per_patient[row.patient_id] += 1
        return True

    def pick(self, pool: pd.DataFrame, quota: int, order: str) -> list[int]:
        """order='clear' -> largest lesion first inside a volume; 'random' -> seeded shuffle."""
        vols: dict[str, dict[str, list[int]]] = defaultdict(dict)
        for vol, g in pool.groupby("volume", sort=True):
            idx = list(g.sort_values(["box_area", "slice_k"], ascending=[False, True]).index) if order == "clear" \
                else self.rng.sample(list(g.index), len(g))
            vols[g.patient_id.iloc[0]][vol] = idx
        patients = sorted(vols)
        self.rng.shuffle(patients)
        cursor = {p: 0 for p in patients}
        vol_order = {p: self.rng.sample(sorted(vols[p]), len(vols[p])) for p in patients}
        got: list[int] = []
        active = list(patients)
        while len(got) < quota and active:
            still = []
            for p in active:
                if len(got) >= quota:
                    still.append(p)
                    continue
                if self.per_patient[p] >= self.cap:
                    continue
                found = False
                vl = vol_order[p]
                for step in range(len(vl)):  # next volume of this patient that still has an acceptable slice
                    vol = vl[(cursor[p] + step) % len(vl)]
                    cand = vols[p][vol]
                    while cand:
                        i = cand.pop(0)
                        if self._ok(pool.loc[i]):
                            got.append(i)
                            found = True
                            break
                    if found:
                        cursor[p] = (cursor[p] + step + 1) % len(vl)
                        break
                if found:
                    still.append(p)
            active = still
        self.chosen += got
        return got


# ---------------------------------------------------------------- build
def build(mod: str, args, out: Path) -> dict:
    t0 = time.time()
    pool = load_pool(mod, args.clear_min_side)
    q_neg = round(args.per_modality * args.neg_fraction)
    q_pos = args.per_modality - q_neg
    pk = Picker(mod, args.gap[mod], args.dup_hamming, args.min_std, args.seed, args.max_per_patient)
    pos_clear = pk.pick(pool[pool.clear], q_pos, "clear")
    pos_other = pk.pick(pool[(pool.kind == "pos") & ~pool.clear], q_pos - len(pos_clear), "clear") \
        if len(pos_clear) < q_pos else []
    neg = pk.pick(pool[pool.kind == "neg"], q_neg, "random")
    # A shortage of negatives is filled with further lesion slices (never duplicated), and vice versa.
    extra_pos = []
    if len(neg) < q_neg:
        extra_pos = pk.pick(pool[(pool.kind == "pos")], q_neg - len(neg), "clear")
    extra_neg = []
    n_pos = len(pos_clear) + len(pos_other) + len(extra_pos)
    if n_pos < q_pos:
        extra_neg = pk.pick(pool[pool.kind == "neg"], q_pos - n_pos, "random")
    sel = pool.loc[pk.chosen].copy()
    sel["tier"] = ""
    sel.loc[pos_clear, "tier"] = "pos_clear"
    sel.loc[pos_other + extra_pos, "tier"] = "pos_small"
    sel.loc[extra_pos, "tier"] = sel.loc[extra_pos].clear.map({True: "pos_clear", False: "pos_small"})
    sel.loc[neg + extra_neg, "tier"] = "neg"
    assert sel.file.is_unique

    # neutral codes
    pmap = {p: f"BN{i + 1:03d}" for i, p in enumerate(sorted(sel.patient_id.unique()))}
    sel["patient_code"] = sel.patient_id.map(pmap)
    sel["series_code"] = sel.series.map(lambda x: SERIES_CODE.get(x, x))
    sel["out_name"] = mod + "_" + sel.patient_code + "_" + sel.series_code + "_k" + sel.slice_k.map("{:03d}".format)
    assert sel.out_name.is_unique and sel.out_name.str.fullmatch(r"[A-Za-z0-9_]+").all(), "unsafe file name"
    sel = sel.sort_values("out_name").reset_index(drop=True)
    sel["part"] = sel.index // args.part_size + 1
    n_parts = int(sel.part.max())

    # copy (byte for byte)
    ds = out / f"Roboflow_{mod}_{args.per_modality}"
    if ds.exists():
        shutil.rmtree(ds)
    (ds / "train/images").mkdir(parents=True)
    (ds / "train/labels").mkdir(parents=True)
    src = SOURCES[mod]
    img_sha, lab_sha = [], []
    for f, o in zip(sel.file, sel.out_name):
        si, sl = src / "images/train" / f"{f}.png", src / "labels/train" / f"{f}.txt"
        di, dl = ds / "train/images" / f"{o}.png", ds / "train/labels" / f"{o}.txt"
        shutil.copyfile(si, di)
        shutil.copyfile(sl, dl)
        img_sha.append(sha256(di))
        lab_sha.append(sha256(dl))
    sel["image_sha256"] = img_sha
    sel["label_sha256"] = lab_sha
    sel["source_image"] = [str((src / "images/train" / f"{f}.png").relative_to(ROOT)) for f in sel.file]
    yaml = (f"# LiverCancer AI - Roboflow_{mod}_{args.per_modality} (YOLO object detection)\n"
            f"# {len(sel)} 2D slices from the project's TRAIN split only (no Val / Test).\n"
            "# Labels were converted automatically from the source dataset's lesion masks (not drawn by hand).\n"
            "# Class 0 = lesion (any focal liver lesion). Empty label file = no lesion annotated on that slice,\n"
            "# which does NOT mean the patient is healthy.\n"
            f"# Source / license: {LICENSE[mod]}\n"
            "path: .\ntrain: train/images\nnc: 1\nnames: ['lesion']\n")
    (ds / "data.yaml").write_text(yaml, encoding="ascii")
    return {"mod": mod, "sel": sel, "ds": ds, "pool": pool, "q_pos": q_pos, "q_neg": q_neg,
            "rejected": dict(pk.rejected), "n_parts": n_parts, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- QA
def qa(r: dict) -> dict:
    sel, ds, src = r["sel"], r["ds"], SOURCES[r["mod"]]
    fails = defaultdict(list)
    imgs = {p.stem for p in (ds / "train/images").iterdir()}
    labs = {p.stem for p in (ds / "train/labels").iterdir()}
    if imgs != labs or imgs != set(sel.out_name):
        fails["Q2_pairing"].append(f"images {len(imgs)} labels {len(labs)} manifest {len(sel)}")
    n_box = 0
    for row in sel.itertuples():
        di, dl = ds / "train/images" / f"{row.out_name}.png", ds / "train/labels" / f"{row.out_name}.txt"
        img = cv2.imread(str(di), cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[:2] != (row.height, row.width):
            fails["Q1_open_size"].append(row.out_name)
        if png_chunks(di) & PNG_TEXT_CHUNKS:
            fails["Q9_png_metadata"].append(row.out_name)
        lines = [ln for ln in dl.read_text(encoding="ascii").splitlines() if ln.strip()]
        for ln in lines:
            v = ln.split()
            ok = len(v) == 5 and v[0] == "0"
            if ok:
                xc, yc, w, h = map(float, v[1:])
                ok = (0 < w <= 1 and 0 < h <= 1 and xc - w / 2 >= -1e-6 and xc + w / 2 <= 1 + 1e-6
                      and yc - h / 2 >= -1e-6 and yc + h / 2 <= 1 + 1e-6)
            if not ok:
                fails["Q3_label_format_bounds"].append(f"{row.out_name}: {ln}")
        n_box += len(lines)
        if len(lines) != row.n_boxes or (row.kind == "neg") != (len(lines) == 0):
            fails["Q4_box_count_vs_source"].append(row.out_name)
        if row.image_sha256 != sha256(src / "images/train" / f"{row.file}.png"):
            fails["Q5_image_differs_from_source"].append(row.out_name)
        if row.label_sha256 != sha256(src / "labels/train" / f"{row.file}.txt"):
            fails["Q5_label_differs_from_source"].append(row.out_name)
    if not sel.image_sha256.is_unique:
        fails["Q6_duplicate_image_in_modality"].append(int((~sel.image_sha256.duplicated(keep=False)).sum()))
    return {"failures": {k: len(v) for k, v in fails.items()}, "examples": {k: v[:5] for k, v in fails.items()},
            "n_boxes": n_box}


def near_dup_global(ds: Path, names: list[str], thr: int) -> int:
    hs = np.stack([dhash(cv2.imread(str(ds / "train/images" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)) for n in names])
    cnt = 0
    for i in range(len(hs) - 1):
        d = np.unpackbits(np.bitwise_xor(hs[i + 1:], hs[i]), axis=1).sum(1)
        cnt += int((d <= thr).sum())
    return cnt


# ---------------------------------------------------------------- zips, previews, stats
def make_zips(r: dict, zdir: Path) -> list[dict]:
    sel, ds, mod = r["sel"], r["ds"], r["mod"]
    rows = []
    for part, g in sel.groupby("part"):
        name = f"{ds.name}_part{part:02d}of{r['n_parts']:02d}.zip"
        zp = zdir / name
        with zipfile.ZipFile(zp, "w") as z:
            z.write(ds / "data.yaml", "data.yaml", compress_type=zipfile.ZIP_DEFLATED)
            for o in g.out_name:
                z.write(ds / "train/images" / f"{o}.png", f"train/images/{o}.png", compress_type=zipfile.ZIP_STORED)
                z.write(ds / "train/labels" / f"{o}.txt", f"train/labels/{o}.txt", compress_type=zipfile.ZIP_DEFLATED)
        rows.append({"modality": mod, "part": int(part), "zip": name, "images": len(g),
                     "lesion_images": int((g.kind == "pos").sum()), "empty_label_images": int((g.kind == "neg").sum()),
                     "boxes": int(g.n_boxes.sum()), "patients": g.patient_id.nunique(),
                     "first_file": g.out_name.iloc[0], "last_file": g.out_name.iloc[-1],
                     "size_bytes": zp.stat().st_size, "sha256": sha256(zp),
                     "uploaded_to_roboflow": "chua", "upload_date": "", "roboflow_project": "", "note": ""})
    return rows


def verify_zips(rows: list[dict], zdir: Path, r: dict) -> dict:
    seen, fails = set(), []
    for row in rows:
        with zipfile.ZipFile(zdir / row["zip"]) as z:
            bad = z.testzip()
            names = z.namelist()
        if bad:
            fails.append(f"{row['zip']}: CRC {bad}")
        im = {Path(n).stem for n in names if n.startswith("train/images/")}
        lb = {Path(n).stem for n in names if n.startswith("train/labels/")}
        if im != lb or len(im) != row["images"] or "data.yaml" not in names:
            fails.append(f"{row['zip']}: pairing")
        if seen & im:
            fails.append(f"{row['zip']}: overlap with earlier part")
        seen |= im
    if seen != set(r["sel"].out_name):
        fails.append("union of parts != dataset")
    return {"failures": fails, "images_in_parts": len(seen)}


def previews(r: dict, pdir: Path, n_pos: int, n_neg: int, seed: int) -> list[str]:
    sel, ds, mod = r["sel"], r["ds"], r["mod"]
    rng = random.Random(seed)
    out = []
    for kind, n in (("pos", n_pos), ("neg", n_neg)):
        g = sel[sel.kind == kind]
        pats = sorted(g.patient_code.unique())
        rng.shuffle(pats)
        picks = [g[g.patient_code == p].sample(1, random_state=seed).index[0] for p in pats[:n]]
        if len(picks) < n:
            rest = g.drop(picks)
            picks += list(rest.sample(n - len(picks), random_state=seed).index)
        out += picks
    d = pdir / mod
    d.mkdir(parents=True, exist_ok=True)
    thumbs, names = [], []
    for i in out:
        row = sel.loc[i]
        img = cv2.imread(str(ds / "train/images" / f"{row.out_name}.png"), cv2.IMREAD_GRAYSCALE)
        if img.shape[1] != 512:  # display only: same width for every preview so captions stay readable
            img = cv2.resize(img, (512, round(img.shape[0] * 512 / img.shape[1])), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        boxes = read_boxes(ds / "train/labels" / f"{row.out_name}.txt")
        for _, xc, yc, bw, bh in boxes:
            x0, y0 = int(round((xc - bw / 2) * w)), int(round((yc - bh / 2) * h))
            x1, y1 = int(round((xc + bw / 2) * w)), int(round((yc + bh / 2) * h))
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 0), 2)
        bar = np.full((34, w, 3), 255, np.uint8)
        txt = f"{row.out_name}  |  {len(boxes)} box" if boxes else f"{row.out_name}  |  no box (empty label)"
        cv2.putText(bar, txt, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        vis = np.vstack([bar, vis])
        name = f"{row.out_name}.jpg"
        cv2.imwrite(str(d / name), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        names.append(f"{mod}/{name}")
        thumbs.append(cv2.resize(vis, (256, int(256 * vis.shape[0] / vis.shape[1])), interpolation=cv2.INTER_AREA))
    th = max(t.shape[0] for t in thumbs)
    thumbs = [np.vstack([t, np.full((th - t.shape[0], 256, 3), 255, np.uint8)]) for t in thumbs]
    cols = 10
    rows_ = [np.hstack(thumbs[i:i + cols] + [np.full((th, 256, 3), 255, np.uint8)] * (cols - len(thumbs[i:i + cols])))
             for i in range(0, len(thumbs), cols)]
    cv2.imwrite(str(pdir / f"contact_sheet_{mod}.jpg"), np.vstack(rows_), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return names


def stats(r: dict) -> dict:
    sel, pool = r["sel"], r["pool"]
    lab = sel.groupby("patient_label").agg(patients=("patient_id", "nunique"), volumes=("volume", "nunique"),
                                           images=("file", "size"), lesion_images=("kind", lambda k: int((k == "pos").sum())),
                                           empty_label_images=("kind", lambda k: int((k == "neg").sum())),
                                           boxes=("n_boxes", "sum")).reset_index()
    ser = sel.groupby("series").agg(images=("file", "size"), lesion_images=("kind", lambda k: int((k == "pos").sum())),
                                    boxes=("n_boxes", "sum")).reset_index()
    per_pat = sel.groupby(["patient_code", "patient_id", "patient_label"]).agg(
        volumes=("volume", "nunique"), images=("file", "size"),
        lesion_images=("kind", lambda k: int((k == "pos").sum())),
        empty_label_images=("kind", lambda k: int((k == "neg").sum())), boxes=("n_boxes", "sum")).reset_index()
    per_vol = sel.groupby("volume").size()
    pos = sel[sel.kind == "pos"]
    bins = [0, 8, 16, 32, 64, 10_000]
    size_tab = pd.cut(pos.box_short, bins, right=False, labels=["<8", "8-15", "16-31", "32-63", ">=64"]).value_counts().sort_index()
    train_pool = {"patients": pool.patient_id.nunique(), "volumes": pool.volume.nunique(), "images": len(pool),
                  "lesion_images": int((pool.kind == "pos").sum()), "empty_label_images": int((pool.kind == "neg").sum()),
                  "clear_lesion_images": int(pool.clear.sum())}
    return {"by_label": lab, "by_series": ser, "per_patient": per_pat, "train_pool": train_pool,
            "per_patient_images": per_pat.images.describe()[["min", "50%", "max"]].to_dict(),
            "per_volume_images": per_vol.describe()[["min", "50%", "max"]].to_dict(),
            "boxes_per_image": sel.n_boxes.value_counts().sort_index().to_dict(),
            "box_short_side": {str(k): int(v) for k, v in size_tab.items()},
            "tiers": sel.tier.value_counts().to_dict()}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-modality", type=int, default=5000)
    ap.add_argument("--neg-fraction", type=float, default=0.2)
    ap.add_argument("--clear-min-side", type=float, default=16.0)
    ap.add_argument("--gap-ct", type=int, default=4)
    ap.add_argument("--max-per-patient", type=int, default=40)
    ap.add_argument("--gap-mri", type=int, default=3)
    ap.add_argument("--dup-hamming", type=int, default=16, help="of 256 bits")
    ap.add_argument("--min-std", type=float, default=8.0)
    ap.add_argument("--part-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data/processed/roboflow"))
    ap.add_argument("--report", default=str(ROOT / "reports/roboflow_dataset_report.md"), help="'' to skip")
    args = ap.parse_args()
    args.gap = {"CT": args.gap_ct, "MRI": args.gap_mri}
    out = Path(args.out)
    zdir, pdir, mdir = out / "zips", out / "previews", out / "meta"
    for d in (zdir, pdir):
        if d.exists():
            shutil.rmtree(d)
    for d in (zdir, pdir, mdir):
        d.mkdir(parents=True, exist_ok=True)

    results, summary, upload_rows = {}, {"args": {k: v for k, v in vars(args).items()}, "modalities": {}}, []
    for mod in ("CT", "MRI"):
        print(f"[{mod}] selecting + copying ...", flush=True)
        r = build(mod, args, out)
        print(f"[{mod}] QA ...", flush=True)
        q = qa(r)
        r["near_dup_global"] = near_dup_global(r["ds"], list(r["sel"].out_name), args.dup_hamming)
        print(f"[{mod}] zips + previews ...", flush=True)
        rows = make_zips(r, zdir)
        vz = verify_zips(rows, zdir, r)
        prev = previews(r, pdir, 40, 10, args.seed)
        st = stats(r)
        results[mod] = r
        upload_rows += rows
        sel = r["sel"]
        sel.to_csv(mdir / f"images_manifest_{mod}.csv", index=False, columns=[
            "out_name", "part", "patient_code", "patient_id", "patient_label", "series", "slice_k", "n_slices",
            "kind", "tier", "n_boxes", "box_short", "width", "height", "file", "source_image", "image_sha256",
            "label_sha256"])
        st["per_patient"].to_csv(mdir / f"patient_stats_{mod}.csv", index=False)
        st["by_label"].to_csv(mdir / f"label_stats_{mod}.csv", index=False)
        summary["modalities"][mod] = {
            "images": len(sel), "quota": args.per_modality, "shortage": max(0, args.per_modality - len(sel)),
            "quota_lesion": r["q_pos"], "quota_empty": r["q_neg"],
            "lesion_images": int((sel.kind == "pos").sum()), "empty_label_images": int((sel.kind == "neg").sum()),
            "boxes": int(sel.n_boxes.sum()), "patients": sel.patient_id.nunique(), "volumes": sel.volume.nunique(),
            "patients_in_train_pool": st["train_pool"]["patients"], "train_pool": st["train_pool"],
            "rejected_during_selection": r["rejected"], "qa": q, "near_dup_pairs_global": r["near_dup_global"],
            "zip_check": vz, "previews": prev, "per_patient_images": st["per_patient_images"],
            "per_volume_images": st["per_volume_images"], "boxes_per_image": st["boxes_per_image"],
            "box_short_side_px": st["box_short_side"], "tiers": st["tiers"],
            "by_label": st["by_label"].to_dict("records"), "by_series": st["by_series"].to_dict("records"),
            "seconds": r["seconds"], "license": LICENSE[mod]}
        print(f"[{mod}] {len(sel)} images, QA {q['failures'] or 'OK'}, zips {vz['failures'] or 'OK'}", flush=True)

    # cross-modality checks
    ct, mr = results["CT"]["sel"], results["MRI"]["sel"]
    summary["cross"] = {
        "shared_image_sha256": len(set(ct.image_sha256) & set(mr.image_sha256)),
        "shared_patient_ids": len(set(ct.patient_id) & set(mr.patient_id)),
        "shared_file_names": len(set(ct.out_name) & set(mr.out_name)),
        "non_train_patients": int(sum((pd.read_csv(SPLITS[m]).set_index("patient_id").loc[
            results[m]["sel"].patient_id.unique(), "split"] != "train").sum() for m in ("CT", "MRI"))),
    }
    pd.DataFrame(upload_rows).to_csv(out / "upload_manifest.csv", index=False)
    (mdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary["cross"]), flush=True)
    if args.report:
        write_report(summary, pd.DataFrame(upload_rows), out, Path(args.report))
    print("done", flush=True)


# ---------------------------------------------------------------- report (Vietnamese)
def _gb(n: float) -> str:
    return f"{n / 1e6:,.1f} MB".replace(",", ".")


REJ_VI = {"near_duplicate_same_patient": "gần trùng ảnh khác của cùng BN", "near_blank": "gần như trống",
          "unreadable": "không đọc được"}


def _kv(d: dict, unit: str, names: dict | None = None) -> str:
    return "; ".join(f"{(names or {}).get(str(k), str(k) + unit)}: {_n(v)}" for k, v in d.items())


def _n(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def write_report(s: dict, up: pd.DataFrame, out: Path, path: Path) -> None:
    a = s["args"]
    rel = lambda p: str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/") if str(Path(p).resolve()).startswith(str(ROOT)) else str(p)
    L = [f"# Bộ dữ liệu YOLO cho Roboflow: CT 5.000 + MRI 5.000 ảnh",
         "",
         f"*Sinh tự động bởi `scripts/88_build_roboflow_export.py` lúc {time.strftime('%Y-%m-%d %H:%M')} "
         f"(seed {a['seed']}). Mọi con số lấy từ `{rel(out)}/meta/summary.json`.*",
         "",
         "## 1. Mục đích và nguồn gốc nhãn",
         "",
         "Hai bộ dữ liệu này dùng để **nhập vào Roboflow nhằm trực quan hóa, kiểm tra và trình bày nhãn** trước hội đồng. "
         "Chúng không dùng để huấn luyện lại mô hình.",
         "",
         "- **Nhãn không được gán thủ công trên Roboflow.** Mọi bounding box được **chuyển đổi tự động** từ mask tổn thương "
         "của dataset gốc (CT: mask tổn thương của PLC-CECT; MRI: mask MedSAM2 của LLD-MMRI-MedSAM2, có đối chiếu với bbox "
         "JSON gốc của LLD-MMRI). Roboflow chỉ dùng để xem, rà soát và nếu cần thì hiệu chỉnh bổ sung; mọi hiệu chỉnh như vậy "
         "phải được ghi lại riêng.",
         "- Ảnh và file nhãn được **sao chép nguyên byte** từ bộ YOLO đã dùng ở GĐ2b (`data/processed/full/yolo_ct`, "
         "`data/processed/pilot/yolo_mri`); SHA256 của từng ảnh và từng nhãn được so khớp với file nguồn (kiểm tra Q5).",
         "- Chỉ có **1 lớp: `0 = lesion`** (tổn thương gan khu trú). Chẩn đoán mức bệnh nhân (HCC, ICC, ...) **không** phải lớp YOLO; "
         "chỉ ghi trong manifest local để thống kê.",
         "- File nhãn rỗng nghĩa là **lát đó không có tổn thương được gán nhãn**, không có nghĩa là bệnh nhân khỏe mạnh.",
         "",
         "## 2. Giấy phép và quyền riêng tư (kiểm tra trước khi upload)",
         "",
         "| Bộ | Nguồn | Giấy phép | Upload lên Roboflow |",
         "|---|---|---|---|",
         "| CT | PLC-CECT (Luo et al., *Scientific Data* 2025) | CC BY 4.0 (`CECT/LICENSE`) | Được phép, **phải ghi nguồn**. |",
         "| MRI | LLD-MMRI (Lou et al.) + mask MedSAM2 (`wanglab/LLD-MMRI-MedSAM2`) | CC BY-NC 4.0; điều khoản LLD-MMRI: "
         "*\"solely for noncommercial research purposes\"*, *\"not to further copy, publish, or distribute any part\"* "
         "(`reports/audit_report.md`) | **Chưa nên upload.** Upload lên dịch vụ bên ngoài là sao chép dữ liệu ra ngoài; "
         "workspace công khai (gói miễn phí của Roboflow) là công bố dữ liệu. Chỉ làm khi bạn đã cân nhắc điều khoản và quyết định rõ; "
         "an toàn nhất là chỉ trình bày MRI bằng ảnh minh họa local. |",
         "",
         "Quyền riêng tư:",
         "",
         "- Ảnh là PNG xám cắt ra từ NIfTI (không có header DICOM). Kiểm tra Q9: không PNG nào chứa chunk văn bản/metadata "
         "(`tEXt`, `zTXt`, `iTXt`, `eXIf`, `tIME`).",
         "- Tên file trong ZIP dùng **mã trung tính** (`CT_BN001_C1_k045`, `MRI_BN001_T2WI_k012`); bảng nối mã BN với ID của dataset "
         f"chỉ nằm ở máy local (`{rel(out)}/meta/images_manifest_*.csv`), không có trong ZIP.",
         "- Một số volume CT có **chữ thước đo in chồng** (\"15cm\", \"20cm\") ở rìa ảnh (GĐ1); đây không phải thông tin định danh. "
         "100 ảnh minh họa đã được xem để phát hiện chữ in chồng khác.",
         "",
         "## 3. Từ NIfTI 3D sang ảnh 2D (`scripts/20_build_yolo_pilot.py`, GĐ2a)",
         "",
         "1. Đọc từng volume (streaming, không nạp cả dataset), kiểm tra hướng LPS, shape và affine của ảnh so với mask.",
         "2. CT lưu dạng 5D `(X,Y,Z,1,3)` được đưa về 3D bằng `livercancer.data.nifti_io.as_3d` (chỉ chấp nhận khi các kênh khác nhau "
         "< 1% voxel và phần khác không chồng lên gan/tổn thương). Mask đọc giá trị raw (`get_unscaled()`) rồi so sánh `> 0`.",
         "3. Cường độ giữ nguyên như nguồn (`uint8`): CT đã được cửa sổ hóa [−200, 200] HU tại nguồn; MRI đã chuẩn hóa theo từng volume. "
         "Không áp thêm cửa sổ hay tăng cường ảnh.",
         "4. Mỗi lát `k` được lưu thành PNG xám ở **độ phân giải gốc** với `slice.T` (hàng = trục j, cột = trục i).",
         "5. Lát dương tính = lát có mask tổn thương; lát âm tính = lát không có annotation và cách mọi lát có annotation ≥ 2 lát "
         "(CT: chỉ lấy lát có gan; MRI: cũng nằm ngoài các lát có `2D_box` trong JSON).",
         "6. Volume lỗi bị loại và ghi log (`volumes.csv`): CT thiếu file, mask rỗng/lệch shape, file nén hỏng; MRI: 26 series Train bị "
         "cách ly do bbox JSON lệch mask (`data/quarantine/mri_bbox_quarantine.csv`).",
         "",
         "## 4. Từ mask sang nhãn YOLO (`src/livercancer/data/yolo_boxes.py`)",
         "",
         "- Mỗi **thành phần liên thông 8 hướng** của mask trên lát → 1 box (hộp bao nhỏ nhất của thành phần).",
         "- Mảnh < 10 pixel bị bỏ; lát mà mọi thành phần đều là mảnh nhỏ bị bỏ hẳn (không coi là âm tính).",
         "- Tọa độ chuẩn hóa theo kích thước ảnh: `0 x_center y_center width height` (∈ [0, 1]).",
         "- MRI: lát dương tính chỉ được giữ khi hộp bao của mask và `2D_box` gốc trong JSON đồng thuận (IoU ≥ 0.3).",
         "",
         "## 5. Cách chọn 10.000 ảnh",
         "",
         f"- Nguồn: **chỉ tập Train** đã đóng băng (`data/splits/*_split.csv`); không có ảnh Val hay Test. Bộ YOLO gốc không bị sửa.",
         f"- Mỗi loại {_n(a['per_modality'])} ảnh: mục tiêu {_n(round(a['per_modality'] * (1 - a['neg_fraction'])))} ảnh có tổn thương "
         f"+ {_n(round(a['per_modality'] * a['neg_fraction']))} ảnh không có box ({a['neg_fraction']:.0%}), để Roboflow có cả ảnh nền.",
         "- Chia đều theo bệnh nhân: lần lượt mỗi bệnh nhân một ảnh mỗi vòng, xoay vòng qua các volume (pha CT / chuỗi MRI) của bệnh nhân đó.",
         f"- Mỗi bệnh nhân tối đa {a['max_per_patient']} ảnh. Trong cùng một volume, hai lát được chọn cách nhau ≥ {a['gap_ct']} lát (CT) / "
         f"≥ {a['gap_mri']} lát (MRI).",
         "- Nếu không đủ ảnh có tổn thương thỏa các điều kiện trên, phần thiếu được bù bằng ảnh không box (và ngược lại); "
         "**không nhân bản ảnh** và không nới điều kiện.",
         f"- Loại ảnh gần giống hệt ảnh đã chọn của cùng bệnh nhân (dHash 256 bit, khoảng cách Hamming ≤ {a['dup_hamming']}/256) "
         f"và ảnh gần như trống (độ lệch chuẩn cường độ < {a['min_std']:g}).",
         f"- Ưu tiên **tổn thương rõ**: lát có box lớn nhất với cạnh ngắn ≥ {a['clear_min_side']:g} px được chọn trước "
         "(trong mỗi volume, lát có box lớn hơn được xét trước). Chỉ khi thiếu mới lấy lát có tổn thương nhỏ hơn.",
         "",
         "## 6. Kết quả",
         "",
         "| | CT | MRI | Tổng |",
         "|---|---|---|---|"]
    c, m = s["modalities"]["CT"], s["modalities"]["MRI"]
    notes: list[str] = []
    for key, label in (("images", "Ảnh"), ("lesion_images", "Ảnh có box tổn thương"),
                       ("empty_label_images", "Ảnh không có box (nhãn rỗng)"), ("boxes", "Bounding box"),
                       ("patients", "Bệnh nhân"), ("volumes", "Volume (pha CT / chuỗi MRI)")):
        L.append(f"| {label} | {_n(c[key])} | {_n(m[key])} | {_n(c[key] + m[key])} |")
    for mod, d in (("CT", c), ("MRI", m)):
        if d["lesion_images"] < d["quota_lesion"]:
            notes.append(f"- **{mod}:** chỉ có {_n(d['lesion_images'])} ảnh có tổn thương thỏa các điều kiện đa dạng (mục tiêu {_n(d['quota_lesion'])}); "
                         f"{_n(d['quota_lesion'] - d['lesion_images'])} suất còn lại được bù bằng ảnh không box, nên tỷ lệ là "
                         f"{d['lesion_images'] / d['images']:.0%} có box / {d['empty_label_images'] / d['images']:.0%} không box. "
                         "Nguyên nhân: nhiều BN có ít lát chứa tổn thương, và lát cùng mức cắt ở các pha khác nhau của một BN gần như giống hệt nhau nên bị loại.")
    L.append(f"| Thiếu so với chỉ tiêu | {_n(c['shortage'])} | {_n(m['shortage'])} | {_n(c['shortage'] + m['shortage'])} |")
    L.append(f"| BN Train có trong nguồn | {_n(c['patients_in_train_pool'])} | {_n(m['patients_in_train_pool'])} | |")
    L += [""] + notes + ["", "Chi tiết theo mức độ rõ của tổn thương và phân bố:", "",
          "| | CT | MRI |", "|---|---|---|",
          f"| Ảnh tổn thương rõ / nhỏ | {_n(c['tiers'].get('pos_clear', 0))} / {_n(c['tiers'].get('pos_small', 0))} "
          f"| {_n(m['tiers'].get('pos_clear', 0))} / {_n(m['tiers'].get('pos_small', 0))} |",
          f"| Số ảnh theo số box trong ảnh | {_kv(c['boxes_per_image'], ' box')} | {_kv(m['boxes_per_image'], ' box')} |",
          f"| Ảnh có box theo cạnh ngắn của box lớn nhất | {_kv(c['box_short_side_px'], ' px')} | {_kv(m['box_short_side_px'], ' px')} |",
          f"| Ảnh mỗi BN (min / trung vị / max) | {c['per_patient_images']['min']:.0f} / {c['per_patient_images']['50%']:.0f} / "
          f"{c['per_patient_images']['max']:.0f} | {m['per_patient_images']['min']:.0f} / {m['per_patient_images']['50%']:.0f} / "
          f"{m['per_patient_images']['max']:.0f} |",
          f"| Ảnh mỗi volume (min / trung vị / max) | {c['per_volume_images']['min']:.0f} / {c['per_volume_images']['50%']:.0f} / "
          f"{c['per_volume_images']['max']:.0f} | {m['per_volume_images']['min']:.0f} / {m['per_volume_images']['50%']:.0f} / "
          f"{m['per_volume_images']['max']:.0f} |",
          f"| Ảnh bị loại khi chọn | {_kv(c['rejected_during_selection'], '', REJ_VI) or '0'} "
          f"| {_kv(m['rejected_during_selection'], '', REJ_VI) or '0'} |", ""]
    for mod, d in (("CT", c), ("MRI", m)):
        L += [f"### {mod}: theo chẩn đoán mức bệnh nhân (chỉ để thống kê, không phải lớp YOLO)", "",
              "| Chẩn đoán | BN | Volume | Ảnh | Có box | Không box | Box |", "|---|---|---|---|---|---|---|"]
        for r in d["by_label"]:
            L.append(f"| {r['patient_label']} | {r['patients']} | {r['volumes']} | {_n(r['images'])} | {_n(r['lesion_images'])} "
                     f"| {_n(r['empty_label_images'])} | {_n(r['boxes'])} |")
        L += ["", f"Theo {'pha' if mod == 'CT' else 'chuỗi'}: " + ", ".join(
            f"{r['series']} {_n(r['images'])} ảnh ({_n(r['lesion_images'])} có box)" for r in d["by_series"]) + ".", ""]
    L += ["Theo nhãn YOLO: chỉ có lớp `0 = lesion`, gồm "
          f"{_n(c['boxes'])} box ở CT và {_n(m['boxes'])} box ở MRI.",
          "", f"Bảng theo từng bệnh nhân: `{rel(out)}/meta/patient_stats_CT.csv`, `patient_stats_MRI.csv` (local, có ID dataset).", "",
          "## 7. Kiểm tra chất lượng", "", "| Kiểm tra | CT | MRI |", "|---|---|---|"]
    checks = [("Q1_open_size", "Q1. Ảnh mở được, đúng kích thước"),
              ("Q2_pairing", "Q2. Mỗi ảnh có đúng một file nhãn cùng tên, không thừa/thiếu"),
              ("Q3_label_format_bounds", "Q3. Nhãn đúng định dạng (5 cột, class 0), box trong [0, 1], w, h > 0"),
              ("Q4_box_count_vs_source", "Q4. Số box khớp manifest nguồn; ảnh âm tính có nhãn rỗng"),
              ("Q5_image_differs_from_source", "Q5a. Ảnh trùng SHA256 với file nguồn"),
              ("Q5_label_differs_from_source", "Q5b. Nhãn trùng SHA256 với file nguồn (giữ nguyên class và tọa độ)"),
              ("Q6_duplicate_image_in_modality", "Q6. Không có ảnh trùng SHA256 trong cùng loại"),
              ("Q9_png_metadata", "Q9. PNG không chứa metadata văn bản")]
    res = lambda d, k: "PASS" if d["qa"]["failures"].get(k, 0) == 0 else f"FAIL ({d['qa']['failures'][k]})"
    for k, label in checks:
        L.append(f"| {label} | {res(c, k)} | {res(m, k)} |")
    x = s["cross"]
    L += [f"| Q7. Mọi BN thuộc Train đã đóng băng (BN ngoài Train: {x['non_train_patients']}) | "
          f"{'PASS' if x['non_train_patients'] == 0 else 'FAIL'} | (chung) |",
          f"| Q8. Không trùng giữa CT và MRI: SHA256 ảnh {x['shared_image_sha256']}, ID BN {x['shared_patient_ids']}, "
          f"tên file {x['shared_file_names']} | {'PASS' if not any((x['shared_image_sha256'], x['shared_patient_ids'], x['shared_file_names'])) else 'FAIL'} | (chung) |",
          f"| Q10. ZIP: CRC đúng, ảnh-nhãn đủ cặp, các phần không trùng ảnh, hợp lại đủ bộ | "
          f"{'PASS' if not c['zip_check']['failures'] else 'FAIL'} | {'PASS' if not m['zip_check']['failures'] else 'FAIL'} |",
          "",
          f"Thông tin thêm (không phải lỗi): cặp ảnh gần giống nhau giữa **các BN khác nhau** (dHash ≤ {a['dup_hamming']}/256): "
          f"CT {c['near_dup_pairs_global']}, MRI {m['near_dup_pairs_global']}. Lát cắt bụng của hai người có thể có bố cục rất giống nhau; "
          "đây không phải ảnh trùng (Q6 đã kiểm tra trùng tuyệt đối).",
          "",
          "## 8. File đầu ra", "",
          f"Thư mục: `{rel(out)}/` (không đưa vào git).", "",
          "| File | Ảnh | Có box | Không box | Box | BN | Dung lượng | SHA256 |", "|---|---|---|---|---|---|---|---|"]
    for r in up.itertuples():
        L.append(f"| `zips/{r.zip}` | {_n(r.images)} | {_n(r.lesion_images)} | {_n(r.empty_label_images)} | {_n(r.boxes)} "
                 f"| {r.patients} | {_gb(r.size_bytes)} | `{r.sha256[:16]}…` |")
    for mod in ("CT", "MRI"):
        g = up[up.modality == mod]
        L.append(f"| **Tổng {mod}** | {_n(g.images.sum())} | {_n(g.lesion_images.sum())} | {_n(g.empty_label_images.sum())} "
                 f"| {_n(g.boxes.sum())} | | {_gb(g.size_bytes.sum())} | |")
    L += ["",
          f"- Thư mục dataset đầy đủ (không nén): `{rel(out)}/Roboflow_CT_{a['per_modality']}/`, `Roboflow_MRI_{a['per_modality']}/` "
          "(mỗi bộ: `data.yaml`, `train/images/`, `train/labels/`).",
          f"- Theo dõi upload: `{rel(out)}/upload_manifest.csv` (cột `uploaded_to_roboflow`, `upload_date`, `roboflow_project` để bạn điền).",
          f"- Manifest từng ảnh (local): `{rel(out)}/meta/images_manifest_{{CT,MRI}}.csv` (tên file mới ↔ ID BN, pha/chuỗi, lát, file nguồn, SHA256).",
          f"- 100 ảnh minh họa đã vẽ box (50 CT + 50 MRI; mỗi loại 40 có box + 10 không box, mỗi ảnh một BN khác nhau): "
          f"`{rel(out)}/previews/{{CT,MRI}}/`, tổng quan: `previews/contact_sheet_CT.jpg`, `contact_sheet_MRI.jpg`.",
          "",
          "## 9. Nhập vào Roboflow", "",
          "1. Tạo project loại **Object Detection**, đặt tên lớp `lesion`. Với CT, ghi nguồn PLC-CECT (CC BY 4.0) trong mô tả project. "
          "Gói miễn phí của Roboflow thường để workspace **công khai**: không upload MRI vào đó.",
          "2. Chọn **Upload Data**, kéo thả một file ZIP (ví dụ `Roboflow_CT_5000_part01of05.zip`). Roboflow nhận dạng định dạng "
          "YOLO (`data.yaml` + `train/images` + `train/labels`) và hiện số ảnh, số box trước khi xác nhận.",
          "3. Khi được hỏi chia tập, giữ nguyên **Train** (mọi ảnh đều từ tập Train của đồ án). Không dùng Roboflow để tạo Val/Test "
          "thay cho split đã đóng băng.",
          "4. Bấm **Save and Continue**. Ảnh có file nhãn rỗng có thể hiện là *null* (ảnh nền) hoặc *unannotated* tùy cách Roboflow đọc; "
          "nếu là *unannotated*, chọn các ảnh đó rồi **Mark Null**. Danh sách ảnh không box: cột `kind = neg` trong manifest.",
          "5. Sau mỗi phần, ghi `da upload`, ngày và tên project vào `upload_manifest.csv`. Các phần không trùng ảnh nên có thể upload theo thứ tự bất kỳ.",
          "6. Xem và chỉnh box: mở ảnh trong **Annotate** → kéo cạnh box để sửa, `Delete` để xóa, vẽ box mới bằng công cụ Bounding Box. "
          "Mọi chỉnh sửa trên Roboflow là **hiệu chỉnh bổ sung** so với nhãn chuyển đổi từ dataset gốc và phải được ghi lại (ai sửa, ảnh nào, lý do) "
          "nếu muốn đưa vào báo cáo; mô hình và kết quả đã báo cáo **không** dùng các chỉnh sửa này.",
          "",
          "## 10. Hạn chế", "",
          "- Nhãn là hộp bao của mask tự động/bán tự động của dataset gốc (MRI: MedSAM2 có người kiểm tra trong vòng lặp), "
          "không phải bbox do bác sĩ vẽ trực tiếp trên Roboflow.",
          "- Một thành phần liên thông = một box: tổn thương bị mask tách làm nhiều mảnh sẽ có nhiều box; hai tổn thương sát nhau có thể chung một box.",
          "- Ảnh không có box chỉ có nghĩa là không có tổn thương được gán nhãn trên lát đó.",
          "- Không train lại mô hình, không đổi checkpoint, ngưỡng, Evidence Gate, split hay dữ liệu Test.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
