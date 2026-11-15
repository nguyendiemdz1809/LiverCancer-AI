"""Lesion classifiers: CNN baseline (local crop), Transformer baseline (context crop), Hybrid CNN + Transformer.

    model = build_model("hybrid", num_classes=3, pretrained=True)
    logits = model(local=x_local, context=x_context)          # every model accepts both inputs, uses what it needs
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tvm

BACKBONES = {  # name -> (constructor, ImageNet weights enum name, feature dim)
    "resnet18": (tvm.resnet18, "ResNet18_Weights", 512),
    "resnet50": (tvm.resnet50, "ResNet50_Weights", 2048),
    "swin_t": (tvm.swin_t, "Swin_T_Weights", 768),
}


def backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Feature extractor without its ImageNet head; returns (module, feature_dim)."""
    ctor, wname, dim = BACKBONES[name]
    weights = getattr(tvm, wname).IMAGENET1K_V1 if pretrained else None
    m = ctor(weights=weights)
    if name.startswith("resnet"):
        m.fc = nn.Identity()
    else:
        m.head = nn.Identity()
    return m, dim


class SingleBranch(nn.Module):
    """Baseline: one backbone on one crop ('local' or 'context') + linear head."""

    def __init__(self, name: str, num_classes: int, pretrained: bool, crop: str, dropout: float = 0.3):
        super().__init__()
        self.backbone, dim = backbone(name, pretrained)
        self.crop = crop
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_classes))

    def forward(self, local: torch.Tensor | None = None, context: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.backbone(local if self.crop == "local" else context))


class HybridCNNTransformer(nn.Module):
    """CNN (ResNet) on the local lesion crop + Transformer (Swin-T) on the context crop; features projected to a common
    size, fused (concat + MLP, or gated sum) and classified."""

    def __init__(self, num_classes: int, pretrained: bool, cnn: str = "resnet18", transformer: str = "swin_t",
                 fusion: str = "concat", fusion_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.cnn, dc = backbone(cnn, pretrained)
        self.vit, dt = backbone(transformer, pretrained)
        proj = lambda d: nn.Sequential(nn.Linear(d, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())  # noqa: E731
        self.proj_cnn, self.proj_vit = proj(dc), proj(dt)
        self.fusion = fusion
        if fusion == "gated":
            self.gate = nn.Sequential(nn.Linear(2 * fusion_dim, fusion_dim), nn.Sigmoid())
            fused = fusion_dim
        elif fusion == "concat":
            fused = 2 * fusion_dim
        else:
            raise ValueError(f"unknown fusion {fusion!r}")
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fused, fusion_dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(fusion_dim, num_classes))

    def features(self, local: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        a, b = self.proj_cnn(self.cnn(local)), self.proj_vit(self.vit(context))
        if self.fusion == "gated":
            g = self.gate(torch.cat([a, b], 1))
            return g * a + (1 - g) * b
        return torch.cat([a, b], 1)

    def forward(self, local: torch.Tensor | None = None, context: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.features(local, context))


def build_model(name: str, num_classes: int, pretrained: bool = True, fusion: str = "concat", fusion_dim: int = 256,
                dropout: float = 0.3) -> nn.Module:
    if name in ("resnet18", "resnet50"):
        return SingleBranch(name, num_classes, pretrained, crop="local", dropout=dropout)
    if name == "swin_t":
        return SingleBranch(name, num_classes, pretrained, crop="context", dropout=dropout)
    if name in ("hybrid", "hybrid_r18_swin", "hybrid_r50_swin"):
        cnn = "resnet50" if name == "hybrid_r50_swin" else "resnet18"
        return HybridCNNTransformer(num_classes, pretrained, cnn, "swin_t", fusion, fusion_dim, dropout)
    raise ValueError(f"unknown model {name!r} (resnet18 | resnet50 | swin_t | hybrid | hybrid_r50_swin)")


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Lower learning rate for pretrained backbones, higher for new projection/fusion/head layers; no decay on norms/biases."""
    groups = {"bb_decay": [], "bb_no": [], "new_decay": [], "new_no": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        pre = n.startswith(("backbone.", "cnn.", "vit."))
        no_decay = p.ndim <= 1
        groups[("bb" if pre else "new") + ("_no" if no_decay else "_decay")].append(p)
    return [g for g in (
        {"params": groups["bb_decay"], "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": groups["bb_no"], "lr": lr_backbone, "weight_decay": 0.0},
        {"params": groups["new_decay"], "lr": lr_head, "weight_decay": weight_decay},
        {"params": groups["new_no"], "lr": lr_head, "weight_decay": 0.0}) if g["params"]]
