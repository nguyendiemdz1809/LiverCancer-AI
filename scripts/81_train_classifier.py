"""GĐ4: train a lesion classifier locally (smoke tests on the 4 GB GPU / CPU). Main training runs on Colab with
notebooks/05_train_classifier_colab.ipynb, which calls the same livercancer.classification.train.train().

  .venv\\Scripts\\python scripts\\81_train_classifier.py --modality ct --model hybrid --smoke          # tiny subset, 1 epoch
  .venv\\Scripts\\python scripts\\81_train_classifier.py --modality mri --model resnet18 --epochs 2 --batch-size 8

--smoke uses random initialisation (no ImageNet download), 2 epochs on a small patient-stratified subset, and writes to
outputs/runs/cls_smoke/. Never touches Test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.classification.train import train  # noqa: E402
from livercancer.config import CONFIG_DIR, load_yaml  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["ct", "mri"], required=True)
    ap.add_argument("--model", default=None, help="resnet18 | resnet50 | swin_t | hybrid | hybrid_r50_swin")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-train-rois", type=int, default=None)
    ap.add_argument("--max-val-rois", type=int, default=None)
    ap.add_argument("--pretrained", choices=["yes", "no"], default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="extra key=value overrides (JSON values), e.g. fusion=\"gated\"")
    args = ap.parse_args()
    allc = load_yaml(CONFIG_DIR / "cls.yaml")
    cfg = dict(allc["train"])
    cfg["model"] = args.model or cfg["model"]
    cfg.update(modality=args.modality, data_root=str(ROOT / allc["roi"]["out_dir"] / args.modality))
    if args.smoke:
        cfg.update(epochs=2, patience=2, batch_size=4, max_train_rois=48, max_val_rois=24, pretrained=False, num_workers=0, resume=False)
    for k, v in (("epochs", args.epochs), ("batch_size", args.batch_size), ("max_train_rois", args.max_train_rois),
                 ("max_val_rois", args.max_val_rois)):
        if v is not None:
            cfg[k] = v
    if args.pretrained:
        cfg["pretrained"] = args.pretrained == "yes"
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    run = f"{args.modality}_{cfg['model']}_{cfg['roi_source']}_s{cfg['seed']}"
    cfg["out_dir"] = str(args.out or ROOT / "outputs" / "runs" / ("cls_smoke" if args.smoke else "cls") / run)
    print(json.dumps(train(cfg), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
