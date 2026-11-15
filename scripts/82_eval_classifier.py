"""GĐ5: evaluate a trained classifier on Validation (ROI + patient level, GT ROIs and YOLO pipeline ROIs) and choose
the patient-level aggregation on Validation. Test is locked (see livercancer.classification.evaluate.evaluate_test).

  .venv\\Scripts\\python scripts\\82_eval_classifier.py --run outputs\\runs\\cls\\ct_hybrid_gt_s42 --modality ct
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.classification.evaluate import evaluate_val  # noqa: E402
from livercancer.config import CONFIG_DIR, load_yaml  # noqa: E402


def fmt(m: dict | None, lvl: str) -> str:
    if not m or lvl not in m:
        return "—"
    x = m[lvl]
    auc = f"{x['roc_auc_macro_ovr']:.3f}" if x["roc_auc_macro_ovr"] is not None else "—"
    return (f"n={x['n']} · Acc {x['accuracy']:.3f} · BalAcc {x['balanced_accuracy']:.3f} · macro-P {x['macro_precision']:.3f} · "
            f"macro-R {x['macro_recall']:.3f} · macro-Spec {x['macro_specificity']:.3f} · macro-F1 {x['macro_f1']:.3f} · AUC {auc}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--modality", choices=["ct", "mri"], required=True)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-subdir", default="eval_val_local", help="output folder inside the run (Colab results stay in eval_val/)")
    args = ap.parse_args()
    root = args.data_root or ROOT / load_yaml(CONFIG_DIR / "cls.yaml")["roi"]["out_dir"] / args.modality
    r = evaluate_val(args.run, root, args.device, out_subdir=args.out_subdir)
    print("aggregation chosen on Val:", r["aggregation"]["chosen"], r["aggregation"]["macro_f1"])
    for key in ("gt_roi", "yolo_roi_matched", "yolo_roi_all"):
        print(f"{key:17s} ROI     {fmt(r.get(key), 'roi')}")
        print(f"{key:17s} patient {fmt(r.get(key), 'patient')}")
    print("pipeline:", json.dumps(r.get("pipeline_patient"), ensure_ascii=False))
    print("written", args.run / args.out_subdir / "metrics_val.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
