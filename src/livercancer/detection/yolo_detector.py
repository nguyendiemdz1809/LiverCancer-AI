"""Load the CT or MRI YOLO lesion detector and run it on 2D slices (GPU when available, CPU fallback).

    det = LesionDetector.from_config("ct")              # configs/yolo_inference.yaml, sha256-checked weights
    dets = det.predict([gray_slice])[0]                  # gray_slice: HxW uint8 from livercancer.data.slices

CT and MRI are separate models; never feed a CT slice to the MRI detector or vice versa.
"""
from __future__ import annotations

import hashlib
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from livercancer.config import CONFIG_DIR, PROJECT_ROOT, load_yaml
from livercancer.data.slices import to_model_input

MODALITIES = ("ct", "mri")


@dataclass
class Detection:
    """One predicted box in pixel coordinates of the displayed slice (x = column = i, y = row = j)."""
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float
    label: str = "lesion"

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_device(requested: str = "auto") -> str:
    """'auto' -> 'cuda:0' if CUDA works, else 'cpu'. An explicit CUDA request falls back to CPU with a warning."""
    import torch

    req = str(requested).lower()
    if req == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0" if req in ("auto", "cuda", "gpu") else req
    if req != "auto":
        warnings.warn(f"device {requested!r} requested but CUDA is not available -> using CPU", stacklevel=2)
    return "cpu"


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class LesionDetector:
    """Single-class ('lesion') YOLO detector for one modality."""

    def __init__(self, modality: str, weights: str | Path, device: str = "auto", imgsz: int = 512,
                 conf: float = 0.25, iou: float = 0.7, max_det: int = 300, expected_sha256: str | None = None):
        from ultralytics import YOLO

        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
        self.modality, self.weights = modality, Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"{modality.upper()} weights not found: {self.weights}")
        self.sha256 = sha256_file(self.weights)
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError(f"{self.weights}: sha256 {self.sha256} != configured {expected_sha256}")
        self.model = YOLO(str(self.weights), task="detect")
        names = dict(self.model.names)
        if names != {0: "lesion"}:
            raise ValueError(f"{self.weights} is not a single-class lesion detector: names={names}")
        train_imgsz = (getattr(self.model, "ckpt", None) or {}).get("train_args", {}).get("imgsz")
        if train_imgsz and int(train_imgsz) != int(imgsz):
            warnings.warn(f"inference imgsz {imgsz} != training imgsz {train_imgsz}", stacklevel=2)
        self.imgsz, self.conf, self.iou, self.max_det = int(imgsz), float(conf), float(iou), int(max_det)
        self.device = resolve_device(device)
        self.fallbacks: list[str] = []  # e.g. CUDA OOM -> CPU, kept for logging

    @classmethod
    def from_config(cls, modality: str, device: str = "auto", conf: float | None = None,
                    config: str | Path = CONFIG_DIR / "yolo_inference.yaml") -> "LesionDetector":
        c = load_yaml(config)[modality]
        return cls(modality, PROJECT_ROOT / c["weights"], device=device, imgsz=c["imgsz"],
                   conf=c["conf"] if conf is None else conf, iou=c["iou"], max_det=c["max_det"],
                   expected_sha256=c.get("sha256"))

    @property
    def half(self) -> bool:
        return self.device.startswith("cuda")  # fp16 on GPU (4 GB VRAM), fp32 on CPU

    @property
    def quantize(self) -> int | None:
        return 16 if self.half else None  # Ultralytics 8.4: `half` is deprecated, fp16 = quantize 16

    def _run(self, batch: list[np.ndarray], conf: float):
        return self.model.predict(batch, imgsz=self.imgsz, conf=conf, iou=self.iou, max_det=self.max_det,
                                  device=self.device, quantize=self.quantize, verbose=False)

    def predict(self, images: list[np.ndarray], conf: float | None = None, batch_size: int = 8) -> list[list[Detection]]:
        """Grayscale HxW uint8 slices -> detections per slice (sorted by confidence, highest first)."""
        import torch

        conf = self.conf if conf is None else conf
        out: list[list[Detection]] = []
        for i in range(0, len(images), batch_size):
            batch = [to_model_input(im) for im in images[i:i + batch_size]]
            try:
                results = self._run(batch, conf)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.fallbacks.append(f"CUDA out of memory on batch of {len(batch)} -> CPU")
                warnings.warn(self.fallbacks[-1], stacklevel=2)
                self.device = "cpu"
                results = self._run(batch, conf)
            for r in results:
                xyxy, cf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
                dets = [Detection(*map(float, b), conf=float(c)) for b, c in zip(xyxy, cf)]
                out.append(sorted(dets, key=lambda d: -d.conf))
        return out

    def timed_predict(self, images: list[np.ndarray], conf: float | None = None) -> tuple[list[list[Detection]], float]:
        """predict() plus wall time in ms per image (synchronised on GPU)."""
        import torch

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.perf_counter()
        dets = self.predict(images, conf=conf)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return dets, (time.perf_counter() - t) * 1000 / max(1, len(images))

    def info(self) -> dict:
        try:
            weights = self.weights.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:  # weights outside the project
            weights = str(self.weights)
        return {"modality": self.modality, "weights": weights, "sha256": self.sha256, "device": self.device,
                "half": self.half, "imgsz": self.imgsz, "conf": self.conf, "iou_nms": self.iou, "max_det": self.max_det,
                "fallbacks": list(self.fallbacks)}
