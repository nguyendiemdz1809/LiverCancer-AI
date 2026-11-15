"""Interface between YOLO + Tumor Zoom (livercancer.roi) and the lesion classifier, for the website and scripts.

    clf = LesionClassifier.load("ct")                       # checkpoint from configs/cls.yaml `inference` (or a path)
    result = clf.classify(rois_from_roi_result(roi_result))  # -> ClassificationResult(status, labels, scores, ...)

Status values: ok | no_roi (nothing to classify) | model_unavailable (no trained checkpoint yet) | invalid_input | error.
The result never contains a label when status != ok; messages are in Vietnamese for the user interface.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from livercancer.config import CONFIG_DIR, PROJECT_ROOT, load_yaml

MESSAGES = {
    "ok": "Phân loại thành công.",
    "no_roi": "Không có vùng tổn thương nào để phân loại (YOLO không phát hiện vùng nghi ngờ).",
    "model_unavailable": "Mô hình phân loại chưa được huấn luyện hoặc chưa có checkpoint; chức năng đang phát triển.",
    "invalid_input": "Dữ liệu vùng tổn thương không hợp lệ (cần ảnh xám 224×224, 8-bit).",
    "error": "Đã xảy ra lỗi khi phân loại.",
}


@dataclass
class RoiInput:
    """One lesion ROI on one slice: Tumor Zoom crops (224x224 uint8 grayscale) + provenance."""
    local: np.ndarray
    context: np.ndarray
    roi_id: str = ""
    track_id: int = -1
    slice_k: int = -1
    det_conf: float = float("nan")


@dataclass
class ClassificationResult:
    status: str
    message: str
    modality: str = ""
    classes: list[str] = field(default_factory=list)
    class_names_vi: list[str] = field(default_factory=list)
    label: str | None = None            # patient / volume-level prediction (aggregated over ROIs)
    label_vi: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    aggregation: str = ""
    per_roi: list[dict] = field(default_factory=list)   # roi_id, track_id, slice_k, det_conf, label, scores
    per_track: list[dict] = field(default_factory=list) # track-level aggregation (one lesion = one track)
    model: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def rois_from_roi_result(res, slices_per_track: int = 3) -> list[RoiInput]:
    """Crops for every kept track of a livercancer.roi.pipeline.RoiResult (key slices: area x confidence)."""
    out = []
    for ti, t in enumerate(res.tracks):
        for k in t.key_slices(slices_per_track):
            z = res.zoom(t, k)
            out.append(RoiInput(z["local"], z["context"], f"T{ti}_k{k}", ti, k, t.boxes[k].conf))
    return out


class LesionClassifier:
    def __init__(self, modality: str, model=None, ckpt: dict | None = None, device: str = "cpu", aggregation: str = "mean_prob",
                 reason: str = ""):
        self.modality, self.model, self.ckpt, self.device = modality, model, ckpt or {}, device
        self.aggregation, self.reason = aggregation, reason
        self.reliable, self.checkpoint_path, self.metrics_path = True, None, None
        cfg = load_yaml(CONFIG_DIR / "cls.yaml")[modality]
        self.classes = self.ckpt.get("classes", cfg["classes"])
        self.class_names_vi = cfg.get("class_names_vi", self.classes)

    @property
    def available(self) -> bool:
        return self.model is not None

    @classmethod
    def load(cls, modality: str, checkpoint: str | Path | None = None, device: str = "auto") -> "LesionClassifier":
        """Never raises for a missing model: returns an unavailable classifier (status model_unavailable)."""
        inf = (load_yaml(CONFIG_DIR / "cls.yaml").get("inference") or {}).get(modality) or {}
        path = checkpoint or inf.get("checkpoint")
        agg = inf.get("aggregation") or "mean_prob"
        if not path:
            return cls(modality, reason="no checkpoint configured")
        path = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
        if not path.exists():
            return cls(modality, reason=f"checkpoint not found: {path}")
        if not checkpoint and inf.get("sha256"):   # configured checkpoint: integrity check
            import hashlib
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            if h != inf["sha256"]:
                return cls(modality, reason=f"sha256 mismatch for {path.name}")
        from livercancer.classification.evaluate import load_classifier
        model, ck, dev = load_classifier(path, device)
        if ck["cfg"]["modality"] != modality:
            return cls(modality, reason=f"checkpoint is for {ck['cfg']['modality']}, not {modality}")
        out = cls(modality, model, ck, dev, agg)
        configured = inf.get("checkpoint") and (PROJECT_ROOT / inf["checkpoint"]).resolve() == path.resolve()
        # quality flag only for the checkpoint chosen on Val; any other checkpoint is treated as not validated
        out.reliable = bool(inf.get("reliable", False)) if configured else False
        out.checkpoint_path = path
        out.metrics_path = PROJECT_ROOT / inf["metrics"] if configured and inf.get("metrics") else None
        if not configured:
            out.reason = "checkpoint differs from the one chosen on Validation (configs/cls.yaml)"
        return out

    def classify(self, rois: list[RoiInput], batch_size: int = 16) -> ClassificationResult:
        base = {"modality": self.modality, "classes": list(self.classes), "class_names_vi": list(self.class_names_vi),
                "model": {"model": self.ckpt.get("cfg", {}).get("model"), "epoch": self.ckpt.get("epoch"), "reason": self.reason}}
        if not self.available:
            return ClassificationResult("model_unavailable", MESSAGES["model_unavailable"], **base)
        if not rois:
            return ClassificationResult("no_roi", MESSAGES["no_roi"], **base)
        for r in rois:
            if r.local.shape != (224, 224) or r.context.shape != (224, 224) or r.local.dtype != np.uint8 or r.context.dtype != np.uint8:
                return ClassificationResult("invalid_input", MESSAGES["invalid_input"], **base)
        try:
            import torch

            from livercancer.classification.data import to_tensor
            from livercancer.classification.metrics import aggregate
            probs = []
            with torch.no_grad():
                for i in range(0, len(rois), batch_size):
                    chunk = rois[i:i + batch_size]
                    x1 = torch.stack([to_tensor(r.local) for r in chunk]).to(self.device)
                    x2 = torch.stack([to_tensor(r.context) for r in chunk]).to(self.device)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
                        logits = self.model(local=x1, context=x2)
                    probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
            p = np.concatenate(probs)
            sc = lambda v: {c: round(float(x), 4) for c, x in zip(self.classes, v)}  # noqa: E731
            per_roi = [{"roi_id": r.roi_id, "track_id": r.track_id, "slice_k": r.slice_k, "det_conf": r.det_conf,
                        "label": self.classes[int(v.argmax())], "scores": sc(v)} for r, v in zip(rois, p)]
            per_track = []
            for t in sorted({r.track_id for r in rois}):
                idx = [i for i, r in enumerate(rois) if r.track_id == t]
                v = aggregate(p[idx], self.aggregation)
                per_track.append({"track_id": t, "label": self.classes[int(v.argmax())], "scores": sc(v), "n_rois": len(idx)})
            v = aggregate(p, self.aggregation)
            lab = int(v.argmax())
            return ClassificationResult("ok", MESSAGES["ok"], label=self.classes[lab], label_vi=self.class_names_vi[lab], scores=sc(v),
                                        aggregation=self.aggregation, per_roi=per_roi, per_track=per_track, **base)
        except Exception as e:  # noqa: BLE001 - surfaced to the UI as a status, never as a fake label
            return ClassificationResult("error", f"{MESSAGES['error']} ({type(e).__name__})", **base)
