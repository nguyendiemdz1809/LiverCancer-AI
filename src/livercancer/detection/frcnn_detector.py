"""Faster R-CNN (CT) with the same interface as `LesionDetector`, so the Tumor Zoom code can use either detector.

    det = FrcnnDetector.from_config()                    # configs/frcnn_inference.yaml, sha256-checked weights
    dets = det.predict([gray_slice])[0]                  # list[Detection], highest confidence first

Comparison model only (website choice "Faster R-CNN"): YOLO11n stays the detector of the locked pipeline.
Input preparation is the one used when the model was trained and evaluated (`frcnn.YoloFolder`): grayscale uint8 / 255,
repeated to 3 channels; fp16 autocast on GPU as in `frcnn.predict_split`.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from livercancer.config import CONFIG_DIR, PROJECT_ROOT, load_yaml
from livercancer.detection.yolo_detector import Detection, resolve_device, sha256_file

NAME = "Faster R-CNN"


class FrcnnDetector:
    modality = "ct"

    def __init__(self, weights: str | Path, device: str = "auto", conf: float = 0.28, max_det: int = 300,
                 batch_size: int = 4, expected_sha256: str | None = None):
        from livercancer.detection import frcnn

        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"Faster R-CNN weights not found: {self.weights}")
        self.sha256 = sha256_file(self.weights)
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError(f"{self.weights}: sha256 {self.sha256} != configured {expected_sha256}")
        self.device = resolve_device(device)
        self.model, _ = frcnn.load_weights(self.weights, self.device)
        self.model.roi_heads.detections_per_img = int(max_det)
        self.conf, self.max_det, self.batch_size = float(conf), int(max_det), int(batch_size)
        self.fallbacks: list[str] = []

    @classmethod
    def from_config(cls, device: str = "auto", config: str | Path = CONFIG_DIR / "frcnn_inference.yaml") -> "FrcnnDetector":
        c = load_yaml(config)["ct"]
        return cls(PROJECT_ROOT / c["weights"], device=device, conf=c["conf"], max_det=c["max_det"],
                   batch_size=c.get("batch_size", 4), expected_sha256=c.get("sha256"))

    def _run(self, batch: list[np.ndarray]):
        import torch
        xs = [torch.from_numpy(im.astype(np.float32) / 255.0)[None].expand(3, -1, -1).contiguous().to(self.device)
              for im in batch]
        with torch.inference_mode(), torch.autocast("cuda", enabled=self.device.startswith("cuda")):
            return self.model(xs)

    def predict(self, images: list[np.ndarray], conf: float | None = None, batch_size: int | None = None) -> list[list[Detection]]:
        """Grayscale HxW uint8 slices -> detections per slice with confidence >= conf (highest first)."""
        import torch

        conf = self.conf if conf is None else conf
        bs = min(batch_size or self.batch_size, self.batch_size)   # never above what fits on a 4 GB GPU
        out: list[list[Detection]] = []
        for i in range(0, len(images), bs):
            batch = images[i:i + bs]
            for im in batch:
                if im.ndim != 2 or im.dtype != np.uint8:
                    raise ValueError(f"expected a 2D uint8 image, got {im.shape} {im.dtype}")
            try:
                res = self._run(batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.fallbacks.append(f"CUDA out of memory on batch of {len(batch)} -> CPU")
                warnings.warn(self.fallbacks[-1], stacklevel=2)
                self.device = "cpu"
                self.model.to("cpu")
                res = self._run(batch)
            for r in res:
                b, s = r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy()
                keep = s >= conf
                dets = [Detection(*map(float, bb), conf=float(c)) for bb, c in zip(b[keep], s[keep])]
                out.append(sorted(dets, key=lambda d: -d.conf))
        return out

    def info(self) -> dict:
        return {"model": NAME, "modality": "ct", "weights": str(self.weights.relative_to(PROJECT_ROOT)) if self.weights.is_relative_to(PROJECT_ROOT) else str(self.weights),
                "sha256": self.sha256, "device": self.device, "conf": self.conf}
