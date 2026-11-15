"""GĐ2a step 1: patient-level Train/Val/Test splits (frozen afterwards).

CT  : stratified per cancer_type, 70/15/15 (per-class allocation, seed fixed).
MRI : option A — Val = 78 official LLD-MMRI2023 validation IDs; remaining 420 patients split per class into
      Train 316 / Test 104 (largest-remainder allocation, seed fixed).
Labels are read from the source metadata only (patient_data.csv / LLD_MMRI_Annotation.json).

Outputs (IDs + labels only, tracked in git):
  data/splits/ct_split.csv, data/splits/mri_split.csv, data/splits/split_manifest.csv
  reports/split_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livercancer.config import load_paths  # noqa: E402

SPLITS = ("train", "val", "test")


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Integer allocation of `total` proportional to `counts`."""
    n = sum(counts.values())
    raw = {k: total * v / n for k, v in counts.items()}
    out = {k: int(r) for k, r in raw.items()}
    for k in sorted(raw, key=lambda k: (raw[k] - out[k], k), reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


def read_commented_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def split_ct(cfg: dict, seed: int, val_frac: float, test_frac: float) -> list[dict]:
    with open(cfg["csv"], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    label = {}
    for r in rows:
        if label.setdefault(r["patient_id"], r["cancer_type"]) != r["cancer_type"]:
            raise ValueError(f"inconsistent cancer_type for {r['patient_id']}")
    by_cls = defaultdict(list)
    for pid, lab in label.items():
        by_cls[lab].append(pid)
    out = []
    for lab in sorted(by_cls):
        pids = sorted(by_cls[lab])
        random.Random(f"{seed}-ct-{lab}").shuffle(pids)
        n_val, n_test = round(len(pids) * val_frac), round(len(pids) * test_frac)
        for i, pid in enumerate(pids):
            split = "val" if i < n_val else "test" if i < n_val + n_test else "train"
            out.append({"patient_id": pid, "label": lab, "split": split, "source": f"stratified_seed{seed}"})
    return sorted(out, key=lambda r: r["patient_id"])


def split_mri(cfg: dict, val_ids_path: Path, seed: int, n_test: int) -> list[dict]:
    ann = json.loads(Path(cfg["annotation"]).read_text(encoding="utf-8"))
    info, cats = ann["Annotation_info"], ann["Category_info"]
    id_to_name = {v: k for k, v in cats.items() if isinstance(v, int)}
    benign = set(cats["Benign"])
    label = {}
    for pid, series in info.items():
        c = {l["category"] for s in series for l in s["annotation"]["lesion"].values()}
        if len(c) != 1:
            raise ValueError(f"{pid}: categories {c}")
        label[pid] = c.pop()
    val_ids = {r["patient_id"] for r in read_commented_csv(val_ids_path)}
    missing = val_ids - set(label)
    if missing:
        raise ValueError(f"official val IDs not in dataset: {sorted(missing)}")

    rest = defaultdict(list)
    for pid, c in label.items():
        if pid not in val_ids:
            rest[c].append(pid)
    alloc = largest_remainder({c: len(p) for c, p in rest.items()}, n_test)
    test = set()
    for c in sorted(rest):
        pids = sorted(rest[c])
        random.Random(f"{seed}-mri-{c}").shuffle(pids)
        test |= set(pids[: alloc[c]])

    out = []
    for pid, c in sorted(label.items()):
        split, src = (("val", "official_lld_mmri2023_val") if pid in val_ids else
                      ("test", f"stratified_seed{seed}") if pid in test else ("train", f"stratified_seed{seed}"))
        out.append({"patient_id": pid, "label": id_to_name[c], "category_id": c,
                    "group": "Benign" if c in benign else "Malignant", "split": split, "source": src})
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def table(rows: list[dict], key: str = "label") -> list[str]:
    labs = sorted({r[key] for r in rows})
    c = Counter((r[key], r["split"]) for r in rows)
    L = ["| " + key + " | " + " | ".join(SPLITS) + " | Tổng | %Val | %Test |", "|---|---|---|---|---|---|---|"]
    for lab in labs + ["**Tổng**"]:
        n = [sum(v for (l, s), v in c.items() if s == sp and (lab == "**Tổng**" or l == lab)) for sp in SPLITS]
        t = sum(n)
        L.append(f"| {lab} | {n[0]} | {n[1]} | {n[2]} | {t} | {100 * n[1] / t:.1f} | {100 * n[2] / t:.1f} |")
    return L


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Vietnamese text on Windows consoles / redirected logs
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ct-val", type=float, default=0.15)
    ap.add_argument("--ct-test", type=float, default=0.15)
    ap.add_argument("--mri-test", type=int, default=104, help="challenge test size")
    ap.add_argument("--force", action="store_true", help="overwrite existing frozen splits")
    args = ap.parse_args()

    paths = load_paths()
    root = paths["project"]["splits"]
    if (root / "ct_split.csv").exists() and not args.force:
        print(f"Splits already frozen in {root}; use --force only with explicit user approval.")
        return 1

    ct = split_ct(paths["ct"], args.seed, args.ct_val, args.ct_test)
    mri = split_mri(paths["mri"], root.parent / "external" / "lld_mmri2023_official_val_ids.csv", args.seed, args.mri_test)
    write_csv(ct, root / "ct_split.csv")
    write_csv(mri, root / "mri_split.csv")
    write_csv([{"modality": "CT", "patient_id": r["patient_id"], "label": r["label"], "split": r["split"],
                "source": r["source"]} for r in ct]
              + [{"modality": "MRI", "patient_id": r["patient_id"], "label": r["label"], "split": r["split"],
                  "source": r["source"]} for r in mri], root / "split_manifest.csv")

    # demographics per CT split (confounder check, report only)
    with open(paths["ct"]["csv"], encoding="utf-8") as f:
        demo = {r["patient_id"]: r for r in csv.DictReader(f)}
    sp = {r["patient_id"]: r["split"] for r in ct}
    L = ["# Báo cáo chia Train / Validation / Test (GĐ2a)\n",
         f"Sinh bởi `scripts/10_make_splits.py --seed {args.seed}`. Split ở **mức bệnh nhân** và đã **đóng băng** "
         "(chạy lại cần `--force` và sự đồng ý của người dùng).\n",
         "## CT — PLC-CECT (70/15/15 phân tầng theo `cancer_type`)\n", *table(ct), ""]
    L += ["| Split | Nam/Nữ/NA | Tuổi TB |", "|---|---|---|"]
    for s in SPLITS:
        ps = [p for p, x in sp.items() if x == s]
        g = Counter(demo[p]["gender"] for p in ps)
        ages = [float(demo[p]["age"]) for p in ps if demo[p]["age"] != "NA"]
        L.append(f"| {s} | {g['male']}/{g['female']}/{g['NA']} | {sum(ages) / len(ages):.1f} |")
    L += ["", "## MRI — LLD-MMRI-MedSAM2 (phương án A)\n",
          "Val = 78 ID chính thức của LLD-MMRI2023. 420 BN còn lại chia phân tầng theo lớp thành Train 316 / Test 104. "
          "**Test này không phải test chính thức của challenge.**\n", *table(mri), "", *table(mri, "group"), ""]
    rep = paths["project"]["reports"] / "split_report.md"
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
