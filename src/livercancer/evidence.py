"""Evidence gate: refuse to classify when the image evidence for a lesion is too weak (reject option).

The classifier only knows cancer classes (CT: HCC / ICC / CHCC), so every YOLO region - also a false one in a control
patient - receives a cancer label. The gate looks only at DETECTION evidence (never at GT, never at the label):
  max_conf          highest YOLO confidence of any region of the case
  n_volumes_strong  number of input volumes (CT phases / MRI sequences) holding a region with confidence >= conf_min
  longest_strong    longest region (slices) among regions with confidence >= conf_min
A case that fails the rule gets status "insufficient_evidence": no label is returned, and the result says explicitly that
this is NOT a conclusion that the patient is healthy. Rule and thresholds are chosen on Validation only
(scripts/93_limitations_analysis.py); configs/cls.yaml `inference.<mod>.evidence_gate`.
"""
from __future__ import annotations

from dataclasses import dataclass

INSUFFICIENT_VI = ("Chưa đủ bằng chứng hình ảnh để phân loại: các vùng YOLO tìm thấy quá yếu hoặc không lặp lại giữa các "
                   "{unit}. Hệ thống KHÔNG kết luận bệnh nhân khỏe mạnh hay không mắc ung thư; cần bác sĩ đánh giá trên ảnh gốc.")


@dataclass(frozen=True)
class GateRule:
    kind: str            # "max_conf" | "n_volumes" | "longest"
    conf_min: float      # confidence threshold (for max_conf: the threshold itself)
    k: int = 1           # n_volumes: minimum number of volumes; longest: minimum slices

    def describe_vi(self, unit: str = "pha") -> str:
        if self.kind == "max_conf":
            return f"độ tin cậy YOLO cao nhất ≥ {self.conf_min:g}"
        if self.kind == "n_volumes":
            return f"có vùng với độ tin cậy ≥ {self.conf_min:g} ở ít nhất {self.k} {unit}"
        return f"có vùng với độ tin cậy ≥ {self.conf_min:g} kéo dài ít nhất {self.k} lát"


def features(regions: list[dict], conf_min: float) -> dict:
    """regions: dicts with 'volume', 'max_conf', 'n_slices' (PipelineResult.regions or cached tracks)."""
    strong = [r for r in regions if r["max_conf"] >= conf_min]
    return {"max_conf": max((r["max_conf"] for r in regions), default=0.0),
            "n_volumes_strong": len({r["volume"] for r in strong}),
            "longest_strong": max((r["n_slices"] for r in strong), default=0)}


def passes(regions: list[dict], rule: GateRule) -> bool:
    f = features(regions, rule.conf_min)
    if rule.kind == "max_conf":
        return f["max_conf"] >= rule.conf_min
    if rule.kind == "n_volumes":
        return f["n_volumes_strong"] >= rule.k
    if rule.kind == "longest":
        return f["longest_strong"] >= rule.k
    raise ValueError(rule.kind)


def rule_from_config(cfg: dict | None) -> GateRule | None:
    if not cfg or not cfg.get("enabled"):
        return None
    return GateRule(cfg["kind"], float(cfg["conf_min"]), int(cfg.get("k", 1)))
