"""Export the architecture of the trained models -> web/src/data/model_arch.json ("Kiến trúc mô hình" on the Models page).

Read-only: loads the locked checkpoints (sha256 checked against configs/yolo_inference.yaml and configs/cls.yaml), runs
one dummy forward pass on CPU to record each block's output shape, and counts layers / parameters. Nothing is trained,
no threshold or checkpoint is changed, no patient data is read.
  YOLO11n CT / MRI : every block of model.model (index, type, inputs, output C x H x W for a 512 x 512 image, parameters)
  Hybrid CT        : ResNet-18 (local crop) + Swin-T (context crop) + projection / fusion / head, from the checkpoint
Run: .venv\\Scripts\\python scripts\\76_export_model_arch.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLO_OFFLINE", "1")          # never download anything

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from livercancer.config import CONFIG_DIR, load_yaml  # noqa: E402
from livercancer.detection.yolo_detector import sha256_file  # noqa: E402

OUT = ROOT / "web" / "src" / "data" / "model_arch.json"

YOLO_VI = {   # short Vietnamese description of each Ultralytics block type
    "Conv": "Tích chập 3×3 bước 2 + BatchNorm + SiLU (giảm kích thước ảnh một nửa)",
    "C3k2": "Khối CSP gồm các bottleneck tích chập (trích đặc trưng)",
    "SPPF": "Gộp đặc trưng nhiều tỉ lệ (max-pooling 5×5 lặp 3 lần)",
    "C2PSA": "Khối có self-attention theo vị trí (attention)",
    "Upsample": "Phóng ảnh đặc trưng ×2 (neck)",
    "Concat": "Nối kênh đặc trưng từ hai nhánh (neck)",
    "Detect": "Đầu ra: dự đoán hộp (DFL) + điểm tin cậy σ(z) trên 3 tỉ lệ",
}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def n_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


def yolo_arch(mod: str) -> dict:
    from ultralytics import YOLO
    cfg = load_yaml(CONFIG_DIR / "yolo_inference.yaml")[mod]
    path = ROOT / cfg["weights"]
    if sha256_file(path) != cfg["sha256"]:
        raise SystemExit(f"sha256 mismatch: {path}")
    net = YOLO(str(path)).model.float().eval()
    blocks = list(net.model)
    imgsz = int(cfg["imgsz"])
    shapes: dict[int, list[int]] = {}
    hooks = [b.register_forward_hook(lambda m, i, o, k=k: shapes.__setitem__(k, list(o.shape[1:])) if torch.is_tensor(o) else None)
             for k, b in enumerate(blocks)]
    with torch.no_grad():
        y = net(torch.zeros(1, 3, imgsz, imgsz))
    for h in hooks:
        h.remove()
    out = y[0] if isinstance(y, (list, tuple)) else y
    n_back = len(net.yaml["backbone"])
    rows = []
    for k, b in enumerate(blocks):
        t = type(b).__name__
        frm = b.f if isinstance(b.f, list) else [b.f]
        rows.append({"i": k, "type": t, "from": ["ảnh" if f == -1 and k == 0 else (str(k - 1) if f == -1 else str(f)) for f in frm],
                     "part": "backbone" if k < n_back else ("head" if k == len(blocks) - 1 else "neck"),
                     "out": list(out.shape[1:]) if k == len(blocks) - 1 else shapes.get(k), "params": n_params(b),
                     "desc": YOLO_VI.get(t, t)})
    c = Counter(type(x).__name__ for x in net.modules())
    return {"modality": mod, "weights": cfg["weights"], "sha256": cfg["sha256"], "imgsz": imgsz, "nc": int(net.yaml.get("nc", 0)),
            "scale": net.yaml.get("scale"), "n_blocks": len(blocks), "n_backbone": n_back, "n_hidden_blocks": len(blocks) - 1,
            "conv2d": c["Conv2d"], "batchnorm": c["BatchNorm2d"], "params": n_params(net),
            "activation": type(blocks[0].act).__name__, "output": list(out.shape[1:]), "blocks": rows}


def hybrid_arch() -> dict:
    from livercancer.classification.evaluate import load_classifier
    inf = load_yaml(CONFIG_DIR / "cls.yaml")["inference"]["ct"]
    path = ROOT / inf["checkpoint"]
    if sha256_file(path) != inf["sha256"]:
        raise SystemExit(f"sha256 mismatch: {path}")
    model, ck, _ = load_classifier(path, "cpu")
    cfg = {**ck["cfg"], "classes": list(ck["classes"])}
    cnn, vit = model.cnn, model.vit
    swin_depths = [len(s) for s in vit.features[1::2]]
    swin_heads = [int(s[0].attn.num_heads) for s in vit.features[1::2]]
    head_parts = n_params(model) - n_params(cnn) - n_params(vit)
    with torch.no_grad():
        z = model.features(torch.zeros(1, 3, 224, 224), torch.zeros(1, 3, 224, 224))
    return {"checkpoint": inf["checkpoint"], "sha256": inf["sha256"], "model": cfg["model"], "fusion": cfg["fusion"],
            "fusion_dim": int(cfg["fusion_dim"]), "dropout": float(cfg["dropout"]), "classes": list(cfg["classes"]),
            "params": n_params(model), "fused_dim": int(z.shape[1]),
            "cnn": {"name": "ResNet-18", "input": "crop cục bộ 224×224", "params": n_params(cnn), "feature": 512,
                    "conv2d": sum(isinstance(x, nn.Conv2d) for x in cnn.modules()), "activation": "ReLU"},
            "transformer": {"name": "Swin-T", "input": "crop ngữ cảnh 224×224", "params": n_params(vit), "feature": 768,
                            "patch": 4, "embed_dim": int(vit.features[0][0].out_channels), "depths": swin_depths,
                            "blocks": sum(swin_depths), "heads": swin_heads, "window": 7, "activation": "GELU"},
            "head": {"params": head_parts, "layers": [
                "Linear 512→256 + LayerNorm + GELU (chiếu nhánh CNN)",
                "Linear 768→256 + LayerNorm + GELU (chiếu nhánh Transformer)",
                f"Nối (concat) → {int(z.shape[1])}",
                f"Dropout {cfg['dropout']} → Linear {int(z.shape[1])}→{int(cfg['fusion_dim'])} → GELU",
                f"Dropout {cfg['dropout']} → Linear {int(cfg['fusion_dim'])}→{len(cfg['classes'])} → softmax"]}}


def main() -> None:
    data = {"generated": datetime.now().isoformat(timespec="seconds"), "source": "scripts/76_export_model_arch.py",
            "yolo": {"ct": yolo_arch("ct"), "mri": yolo_arch("mri")}, "hybrid_ct": hybrid_arch()}
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    y, h = data["yolo"]["ct"], data["hybrid_ct"]
    print(f"YOLO11n: {y['n_blocks']} blocks ({y['n_hidden_blocks']} hidden), {y['conv2d']} Conv2d, {y['params']:,} params, out {y['output']}")
    print(f"MRI params {data['yolo']['mri']['params']:,}; Hybrid {h['params']:,} (ResNet {h['cnn']['params']:,}, Swin {h['transformer']['params']:,})")
    print("->", rel(OUT))


if __name__ == "__main__":
    main()
