"""CT vs MRI recognition, in order of reliability:
  1. DICOM `Modality` tag (CT / MR)                              -> certain
  2. NIfTI header geometry (server/inspect_file.py: CT 512^2 1 mm isotropic; MRI anisotropic) -> certain on the
     datasets' format
  3. image classifier (this module) for PNG / JPG / screenshots / secondary-capture DICOM / inconclusive NIfTI:
     a small CNN trained on Train-patient slices (+ synthetic screenshot variants) and evaluated on Validation
     patients (scripts/87_modality_classifier.py). It decides only when its confidence reaches the threshold chosen on
     Val (configs/modality.yaml). Used only by API callers that send "auto": the website analyses the modality the user
     chose (and only checks it against DICOM / NIfTI information). It is a separate model: the locked AI pipeline is unchanged.
The file name is never used to decide.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from livercancer.config import CONFIG_DIR, PROJECT_ROOT, load_yaml

INPUT = 160
CLASSES = ("ct", "mri")


def build_model():
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
    return nn.Sequential(block(1, 24), block(24, 48), block(48, 96), block(96, 160), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(0.2), nn.Linear(160, 2))


def to_input(gray512: np.ndarray) -> np.ndarray:
    return cv2.resize(gray512, (INPUT, INPUT), interpolation=cv2.INTER_AREA)


class ModalityClassifier:
    def __init__(self, device: str = "auto"):
        import hashlib
        import torch
        cfg = load_yaml(CONFIG_DIR / "modality.yaml")
        self.cfg = cfg
        path = PROJECT_ROOT / cfg["checkpoint"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != cfg["sha256"]:
            raise RuntimeError("modality checkpoint sha256 mismatch")
        self.device = "cuda" if device in ("auto", "cuda", "cuda:0") and torch.cuda.is_available() else "cpu"
        self.model = build_model()
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model.eval().to(self.device)
        self.threshold = float(cfg["threshold"])

    def probs(self, grays: list[np.ndarray]) -> np.ndarray:
        """P(CT), P(MRI) per 512x512 prepared slice."""
        import torch
        x = torch.from_numpy(np.stack([to_input(g) for g in grays]).astype(np.float32) / 255.0)[:, None]
        with torch.no_grad():
            p = torch.softmax(self.model(x.to(self.device)).float(), 1).cpu().numpy()
        return p

    def decide(self, grays: list[np.ndarray]) -> tuple[str | None, float]:
        """Mean probability over the given slices; (modality, confidence) or (None, confidence) below the threshold."""
        p = self.probs(grays).mean(0)
        k = int(p.argmax())
        conf = float(p[k])
        return (CLASSES[k] if conf >= self.threshold else None), conf


def representative_slices(vol: np.ndarray, n: int = 7) -> list[np.ndarray]:
    """n display slices from the central 60 % of a volume (liver level is usually there)."""
    from livercancer.data.slices import slice_image
    k = vol.shape[2]
    idx = np.unique(np.linspace(int(0.2 * k), max(int(0.2 * k), int(0.8 * k) - 1), n).astype(int))
    out = []
    for i in idx:
        s = slice_image(vol, int(i))
        out.append(s if s.shape == (512, 512) else cv2.resize(s, (512, 512), interpolation=cv2.INTER_AREA))
    return out


def available() -> bool:
    try:
        cfg = load_yaml(CONFIG_DIR / "modality.yaml")
        return (PROJECT_ROOT / cfg["checkpoint"]).exists()
    except Exception:  # noqa: BLE001
        return False


__all__ = ["ModalityClassifier", "build_model", "to_input", "representative_slices", "available", "Path"]
