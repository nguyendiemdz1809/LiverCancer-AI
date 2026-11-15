"""ROI dataset for lesion classification (manifest from scripts/80_build_cls_dataset.py).

Each sample = one lesion ROI on one slice: `local` (lesion + margin) and `context` (2.5x, surrounding liver) crops,
224x224 uint8 grayscale PNG. For ImageNet-pretrained backbones the gray value is replicated to 3 channels, scaled to
[0, 1] and normalised with the ImageNet mean / std.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def read_manifest(path: str | Path, split: str | None = None, sources: tuple[str, ...] = ("gt",),
                  matched_only: bool = False, labelled_only: bool = True) -> list[dict]:
    """Rows of manifest.csv. `sources`: 'gt' and/or 'yolo'; `matched_only`: YOLO rows overlapping a GT lesion;
    `labelled_only`: drop rows without a classifier class (CT controls, label_id -1)."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if split and r["split"] != split:
            continue
        if r["source"] not in sources:
            continue
        if matched_only and r["source"] == "yolo" and r["matches_gt"] != "1":
            continue
        if labelled_only and int(r["label_id"]) < 0:
            continue
        out.append(r)
    return out


def to_tensor(gray: np.ndarray) -> torch.Tensor:
    """HxW uint8 -> 3xHxW float, ImageNet-normalised."""
    t = torch.from_numpy(np.ascontiguousarray(gray)).float().div_(255.0)
    return (t.unsqueeze(0).expand(3, -1, -1) - IMAGENET_MEAN) / IMAGENET_STD


class Augment:
    """Light, anatomy-preserving augmentation applied independently to each crop (Train only)."""

    def __init__(self, rot_deg: float = 10, scale: tuple = (0.9, 1.1), shift: float = 0.05, flip_p: float = 0.5,
                 contrast: float = 0.1, brightness: float = 0.08, seed: int | None = None):
        self.rot, self.scale, self.shift, self.flip_p = rot_deg, scale, shift, flip_p
        self.contrast, self.brightness = contrast, brightness
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        r = self.rng
        m = cv2.getRotationMatrix2D((w / 2, h / 2), r.uniform(-self.rot, self.rot), r.uniform(*self.scale))
        m[:, 2] += r.uniform(-self.shift, self.shift, 2) * (w, h)
        out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        if r.random() < self.flip_p:
            out = out[:, ::-1]
        a = 1 + r.uniform(-self.contrast, self.contrast)
        b = r.uniform(-self.brightness, self.brightness) * 255
        return np.clip(out.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


class RoiDataset(Dataset):
    def __init__(self, rows: list[dict], root: str | Path, augment: Augment | None = None):
        self.rows, self.root, self.augment = rows, Path(root), augment

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, rel: str) -> np.ndarray:
        img = cv2.imread(str(self.root / rel), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.root / rel)
        return self.augment(img) if self.augment else img

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        return {"local": to_tensor(self._load(r["local_path"])), "context": to_tensor(self._load(r["context_path"])),
                "label": int(r["label_id"]), "index": i}


def worker_init(worker_id: int) -> None:
    """Different, reproducible augmentation streams per DataLoader worker."""
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    if getattr(ds, "augment", None) is not None:
        ds.augment.rng = np.random.default_rng(torch.initial_seed() % 2**32 + worker_id)


def class_weights(labels: list[int], n_classes: int, mode: str = "inverse_sqrt") -> torch.Tensor:
    """Per-class loss weights from Train counts (normalised to mean 1). Classes absent from Train get weight 0."""
    counts = np.bincount(np.asarray(labels, int), minlength=n_classes).astype(float)
    if mode == "none":
        w = np.ones(n_classes)
    else:
        w = np.where(counts > 0, 1 / np.maximum(counts, 1) ** (0.5 if mode == "inverse_sqrt" else 1.0), 0.0)
    w = w / w[counts > 0].mean()
    return torch.tensor(w, dtype=torch.float32)
