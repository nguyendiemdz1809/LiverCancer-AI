"""End-to-end AI pipeline: CT / MRI image -> YOLO lesion detection -> Automatic Tumor Zoom -> lesion classification.

    pipe = LiverCancerPipeline("ct")
    res = pipe.run_nifti([path_c1, path_c2])          # one or several volumes (phases / sequences) of ONE patient
    res.status, res.label, res.scores, res.warnings   # res.as_dict() is JSON-serialisable

Every step reuses the validated modules: livercancer.data.slices (training-identical preprocessing),
livercancer.detection (final YOLO, conf chosen on Val), livercancer.roi (Tumor Zoom, configs/roi.yaml) and
livercancer.classification (checkpoint + aggregation chosen on Val, configs/cls.yaml `inference`).
ROI key slices are chosen exactly as for the Val YOLO ROIs of the classification dataset (area x conf, spaced).

Status: ok | no_lesion_detected | insufficient_evidence | classifier_unavailable | error. No label is ever returned unless status is ok, and
"no lesion detected" is never turned into "no cancer". MRI predictions carry an explicit not-reliable warning.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from livercancer.classification.inference import ClassificationResult, LesionClassifier, RoiInput
from livercancer.config import CONFIG_DIR, load_yaml
from livercancer.data.slices import load_volume_uint8
from livercancer.evidence import INSUFFICIENT_VI, features, passes, rule_from_config
from livercancer.roi.pipeline import RoiResult, analyze_image, analyze_volume
from livercancer.roi.tracking import spaced_key_slices

AGG_VI = {"mean_prob": "trung bình xác suất", "conf_weighted": "trung bình có trọng số theo độ tin cậy",
          "majority_vote": "bỏ phiếu đa số", "top3_mean": "trung bình 3 vùng tin cậy nhất"}
DISCLAIMER = "Kết quả do AI tạo ra, chỉ phục vụ nghiên cứu, không thay thế chẩn đoán của bác sĩ."
NO_LESION = ("YOLO không phát hiện vùng nghi ngờ nào ở ngưỡng tin cậy {conf:g}. Điều này KHÔNG có nghĩa là bệnh nhân không mắc "
             "ung thư: mô hình có thể bỏ sót tổn thương (nhất là tổn thương nhỏ). Cần bác sĩ đánh giá trên ảnh gốc.")


@dataclass
class PipelineResult:
    status: str
    message: str
    modality: str
    case_id: str = ""
    label: str | None = None
    label_vi: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    reliable: bool = False
    warnings: list[str] = field(default_factory=list)
    volumes: list[dict] = field(default_factory=list)      # per input volume: name, n_slices, n_regions
    regions: list[dict] = field(default_factory=list)      # per lesion candidate (track): slices, confidence, label, scores
    classification: dict | None = None                    # ClassificationResult.as_dict() (per ROI details)
    timings_s: dict[str, float] = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)              # evidence gate features and decision (when enabled)
    device: str = ""
    roi_results: dict = field(default_factory=dict, repr=False)   # name -> RoiResult (images; not serialised)
    rois: list = field(default_factory=list, repr=False)          # RoiInput list (crops; not serialised)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("roi_results", "rois")}
        return json.loads(json.dumps(d, default=float))


class LiverCancerPipeline:
    def __init__(self, modality: str, device: str = "auto", cls_checkpoint: str | Path | None = None, evidence_gate: str = "config"):
        """evidence_gate: 'config' (configs/cls.yaml inference.<mod>.evidence_gate.enabled), 'on' or 'off'."""
        from livercancer.detection.yolo_detector import LesionDetector
        self.modality = modality
        self.detector = LesionDetector.from_config(modality, device=device)
        self.classifier = LesionClassifier.load(modality, cls_checkpoint, device)
        cfg = load_yaml(CONFIG_DIR / "cls.yaml")
        self.n_slices = int(cfg["inference"]["slices_per_track"][modality])
        self.min_gap = int(cfg[modality]["slice_min_gap"])
        self.quality = self._quality()
        g = dict(cfg["inference"][modality].get("evidence_gate") or {})
        if evidence_gate in ("on", "off"):
            g["enabled"] = evidence_gate == "on" and bool(g.get("kind"))
        self.gate = rule_from_config(g)
        self.gate_single = rule_from_config({**g.get("single_volume", {}), "enabled": g.get("enabled")}) if g.get("single_volume") else self.gate

    def _quality(self) -> dict:
        p = self.classifier.metrics_path
        if not p or not Path(p).exists():
            return {}
        m = json.loads(Path(p).read_text(encoding="utf-8"))
        return {"val_roi_accuracy": m["gt_roi"]["roi"]["accuracy"], "val_roi_macro_f1": m["gt_roi"]["roi"]["macro_f1"],
                "val_pipeline_patient_accuracy": (m.get("pipeline_patient") or {}).get("accuracy_counting_missing_as_wrong"),
                "source": str(p)}

    # ------------------------------------------------------------------ helpers
    def rois_for(self, res: RoiResult, name: str, first_track_id: int = 0) -> tuple[list[RoiInput], list[dict]]:
        rois, regions = [], []
        for i, t in enumerate(res.tracks):
            tid = first_track_id + i
            ks = spaced_key_slices(t, self.n_slices, self.min_gap) if res.n_slices > 1 else t.ks
            for k in ks:
                z = res.zoom(t, k)
                rois.append(RoiInput(z["local"], z["context"], f"{name}_T{i}_k{k:03d}", tid, k, t.boxes[k].conf))
            regions.append({"track_id": tid, "volume": name, "k_first": t.ks[0], "k_last": t.ks[-1], "n_slices": t.n_slices,
                            "max_conf": round(t.max_conf, 4), "key_slices": ks, "best_slice": t.best_slice,
                            "box_best_slice": [round(v, 1) for v in t.boxes[t.best_slice].xyxy]})
        return rois, regions

    def _warnings(self) -> list[str]:
        w = [DISCLAIMER]
        if self.modality == "mri" and not self.classifier.reliable:
            q = self.quality
            nums = (f" (Validation: accuracy {100 * q['val_roi_accuracy']:.1f}%, macro-F1 {100 * q['val_roi_macro_f1']:.1f}%)" if q else "")
            w.append("CẢNH BÁO: mô hình phân loại MRI CHƯA ĐẠT chất lượng mong muốn" + nums +
                     ". Nhãn MRI chỉ để minh họa nghiên cứu, KHÔNG phải chẩn đoán và không đáng tin cậy.")
        elif not self.classifier.reliable:
            w.append("Checkpoint phân loại không phải bản đã chọn trên Validation: kết quả chưa được kiểm chứng.")
        if self.modality == "ct":
            w.append("Mô hình CT chỉ phân biệt HCC / ICC / CHCC, không có lớp 'không tổn thương'. Vùng YOLO phát hiện nhầm "
                     "(kể cả ở người không có bệnh gan) vẫn bị gán một trong 3 nhãn.")
        w.append("YOLO dùng ngưỡng thấp để hạn chế bỏ sót, nên có thể có nhiều vùng giả (trung bình khoảng 5 vùng mỗi volume trên Validation).")
        return w

    # ------------------------------------------------------------------ entry points
    def run(self, volumes: list[tuple[str, np.ndarray]], case_id: str = "", is_2d: bool = False) -> PipelineResult:
        """volumes: [(name, (i, j, k) uint8 volume)] of ONE patient, or [(name, H x W uint8 slice)] with is_2d=True."""
        t0 = time.perf_counter()
        out = PipelineResult("error", "", self.modality, case_id, warnings=self._warnings(), device=self.detector.device)
        try:
            rois, first = [], 0
            for name, arr in volumes:
                res = analyze_image(arr, self.detector) if is_2d else analyze_volume(arr, self.detector)
                r, regs = self.rois_for(res, name, first)
                first += len(res.tracks)
                rois += r
                out.regions += regs
                out.roi_results[name] = res
                out.volumes.append({"name": name, "n_slices": res.n_slices, "n_regions": len(res.tracks),
                                    "slices_with_boxes": len(res.slices_with_boxes())})
            out.rois = rois
            out.timings_s["detection_zoom"] = round(time.perf_counter() - t0, 2)
            out.reliable = self.classifier.reliable
            if not out.regions:
                out.status, out.message = "no_lesion_detected", NO_LESION.format(conf=self.detector.conf)
                return out
            t1 = time.perf_counter()
            c: ClassificationResult = self.classifier.classify(rois)
            out.timings_s["classification"] = round(time.perf_counter() - t1, 2)
            out.classification = c.as_dict()
            if c.status != "ok":
                out.status = "classifier_unavailable" if c.status == "model_unavailable" else "error"
                out.message = c.message
                return out
            per_track = {t["track_id"]: t for t in c.per_track}
            for reg in out.regions:
                t = per_track.get(reg["track_id"], {})
                reg["label"], reg["scores"] = t.get("label"), t.get("scores", {})
            gate = self.gate_single if len(volumes) == 1 else self.gate
            if gate is not None:
                ev = features(out.regions, gate.conf_min)
                out.evidence = {**ev, "rule": gate.describe_vi("pha CT" if self.modality == "ct" else "chuỗi MRI"), "passed": passes(out.regions, gate)}
                if not out.evidence["passed"]:   # reject option: no label, never "healthy"
                    out.status = "insufficient_evidence"
                    out.message = INSUFFICIENT_VI.format(unit="pha CT" if self.modality == "ct" else "chuỗi MRI") + \
                        f" (Điều kiện: {out.evidence['rule']}.)"
                    return out
            out.status, out.label, out.label_vi, out.scores = "ok", c.label, c.label_vi, c.scores
            out.message = ("Đã phân tích xong. Nhãn là dự đoán của mô hình, gộp từ mọi vùng phát hiện (cách gộp: "
                           f"{AGG_VI.get(c.aggregation, c.aggregation)}, chọn trên Validation).")
            return out
        except Exception as e:  # noqa: BLE001 - reported as a status, never as a fake label
            out.status, out.message = "error", f"Lỗi khi chạy pipeline: {type(e).__name__}: {e}"
            return out
        finally:
            out.timings_s["total"] = round(time.perf_counter() - t0, 2)

    def run_nifti(self, paths: list[str | Path], case_id: str = "", names: list[str] | None = None) -> PipelineResult:
        vols = [(names[i] if names else Path(p).name.replace(".nii.gz", "").replace(".nii", ""), load_volume_uint8(p))
                for i, p in enumerate(paths)]
        return self.run(vols, case_id or Path(paths[0]).name.split("_")[0])

    def run_image(self, gray: np.ndarray, name: str = "image") -> PipelineResult:
        return self.run([(name, gray)], name, is_2d=True)
