"""Analysis service: intake -> CT / MRI recognition -> locked AI pipeline -> Vietnamese dashboard result.

Intake (livercancer.intake, separate from the locked pipeline):
  NIfTI  -> livercancer.data.slices.load_volume_uint8 (training-identical)
  DICOM  -> livercancer.intake.dicom_io (one file, several files, or a .zip; Modality tag; source normalisation)
  PNG/JPG/screenshot/secondary-capture DICOM -> livercancer.intake.image2d.prepare_2d (find the slice, remove UI and
            burned-in text, frame like training)
CT / MRI: chosen by the user on the website ("Phân tích CT" / "Phân tích MRI"). The file's own information (DICOM
          Modality tag, NIfTI header geometry) is checked against that choice: on a contradiction nothing is analysed
          until the user confirms (ModalityConflict -> job state needs_confirm). File names are never used.
          API callers may send "auto": metadata, then the image recogniser; otherwise the request is refused.
AI (unchanged, locked before the Test run; reports/final_test/locked_config.json compared at start-up):
  CT  : LiverCancerPipeline = YOLO11n -> Tumor Zoom -> Hybrid -> Evidence Gate
  MRI : YOLO11n -> Tumor Zoom (the MRI classifier did not reach the required quality and is never run)
Every number shown comes from the real checkpoints. Inputs are deleted as soon as they are read; results stay in memory.

3D (NIfTI, DICOM series / .zip): the volume is cut into its axial 2D slices exactly as for training
(load_volume_uint8 / dicom_io -> slice_image) and EVERY slice goes through the 2D YOLO of that modality (this is not a
3D model). Byte-identical slices are detected (livercancer.intake.dedup): YOLO runs once per distinct slice and the
result is copied back to each copy, so no slice is dropped and slice indices are unchanged (LCAI_DEDUP_EXACT=0 turns
it off). The case summary lists slice counts, duplicates and every slice with a suspicious region; it never turns a
detection confidence into a probability of cancer. The CT case label is the locked pipeline's (Validation-chosen,
Test-evaluated aggregation + Evidence Gate); MRI has no label.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from livercancer.config import CONFIG_DIR, PROJECT_ROOT, load_yaml
from livercancer.data.slices import load_volume_uint8
from livercancer.intake import dicom_io
from livercancer.intake.dedup import DedupDetector
from livercancer.intake.image2d import prepare_2d
from livercancer.pipeline import DISCLAIMER, NO_LESION, LiverCancerPipeline, PipelineResult
from livercancer.pipeline_viz import short
from livercancer.roi.display import display_zoom
from livercancer.roi.pipeline import analyze_image, analyze_volume

Progress = Callable[[str, str, float | None], None]      # (stage key, Vietnamese text, real percentage or None)
MAX_REGIONS_SHOWN = 8
LOCK_FILE = PROJECT_ROOT / "reports" / "final_test" / "locked_config.json"
TEST_METRICS = PROJECT_ROOT / "reports" / "final_test" / "metrics_test.json"
AUDIT_3D = PROJECT_ROOT / "outputs" / "case3d_audit" / "volumes.csv"     # scripts/74_case3d_audit.py (Validation)
FOOTNOTE = ("Kết quả AI chỉ phục vụ nghiên cứu và tham khảo, không thay thế đánh giá của bác sĩ. "
            "Không phát hiện vùng nghi ngờ không đồng nghĩa với loại trừ bệnh.")
METHOD_VI = {"dicom_tag": "Đọc từ thông tin DICOM (Modality)", "nifti_header": "Nhận dạng từ thông tin header NIfTI",
             "image_model": "Nhận dạng bằng mô hình ảnh"}
SOURCE_VI = {"dicom_tag": "thông tin DICOM", "nifti_header": "header NIfTI"}


class UserError(Exception):
    """An input problem explained to the user in Vietnamese."""


class ModalityConflict(Exception):
    """The file's own information says the other modality than the one the user chose: ask, never analyse silently."""

    def __init__(self, detected: str, chosen: str, source: str):
        super().__init__(f"{detected} != {chosen}")
        self.detected, self.chosen, self.source = detected, chosen, source

    def need(self) -> dict:
        d, c = self.detected.upper(), self.chosen.upper()
        return {"detected": self.detected, "chosen": self.chosen,
                "message": f"Theo {SOURCE_VI.get(self.source, 'thông tin trong tệp')}, đây là ảnh {d}, khác loại bạn đã chọn ({c})."}


def method_vi(info: dict) -> str:
    """How the modality was set, in Vietnamese (card "Loại ảnh")."""
    m = info.get("method")
    if m == "user":
        src = info.get("meta_source")
        return "Bạn đã chọn" + (f" · khớp {SOURCE_VI.get(src, 'thông tin trong tệp')}" if src else "")
    if m == "user_override":
        return f"Bạn đã xác nhận (theo {SOURCE_VI.get(info.get('meta_source'), 'thông tin trong tệp')}: {str(info.get('meta')).upper()})"
    return METHOD_VI.get(m, "") + (f" ({_vn(100 * info['confidence'])}%)" if info.get("confidence") else "")


BRANCH_VI = {"ct": "YOLO CT", "mri": "YOLO MRI"}


def _vn(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def _png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _thumb_b64(img: np.ndarray, side: int = 900) -> str:
    h, w = img.shape[:2]
    s = min(1.0, side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii")


class _ProgressDetector:
    """Wraps the YOLO detector to report progress. Calls the real predict on chunks that are whole multiples of the
    batch size, so the batches - and therefore the boxes - are exactly those of an unwrapped call."""

    def __init__(self, det, on_volume: Callable[[int, int, int], None]):
        self._det, self._on, self._vol = det, on_volume, -1

    def __getattr__(self, k):
        return getattr(self._det, k)

    def predict(self, images, conf=None, batch_size=8):
        self._vol += 1
        out, step = [], batch_size * 4
        if not images:
            return out
        for s in range(0, len(images), step):
            out += self._det.predict(images[s:s + step], conf=conf, batch_size=batch_size)
            self._on(self._vol, min(len(images), s + step), len(images))
        return out


@dataclass
class Loaded:
    """Inputs read into memory (files already deleted)."""
    kind: str                                    # volume | image
    items: list[tuple[str, np.ndarray]] = field(default_factory=list)   # (display name, (i,j,k) volume | raw 2D image)
    meta_modality: str | None = None             # from DICOM tag / NIfTI header (certain), else None
    meta_method: str | None = None
    prepared: list = field(default_factory=list)  # image2d.Prepared per 2D item (generic framing)
    notes: list[str] = field(default_factory=list)
    vinfo: list[dict] = field(default_factory=list)  # per item: source, spacing, DICOM files / duplicate files


class AnalysisService:
    def __init__(self, device: str = "auto", dedup_exact: bool = True):
        self.device = device
        self.dedup_exact = dedup_exact
        self.ct: LiverCancerPipeline | None = None
        self.mri = None
        self.modality_model = None
        self.state = {"ct": "loading", "mri": "loading"}
        self.errors: dict[str, str] = {}
        self.gpu_lock = threading.Lock()
        self.facts = self._facts()

    # ------------------------------------------------------------------ start-up
    @staticmethod
    def _facts() -> dict:
        f = {}
        try:
            m = json.loads(TEST_METRICS.read_text(encoding="utf-8"))
            d = m["ct"]["detection"]
            f["ct_small"] = (d["small_detected"], d["small"])
            f["mri_acc"] = (m["mri"]["pipeline"]["correct"], m["mri"]["pipeline"]["n"])
        except Exception:  # noqa: BLE001
            pass
        try:                                       # 3D CT audit on Validation: regions per volume, outside liver, split lesions
            import csv
            rows = [r for r in csv.DictReader(open(AUDIT_3D, encoding="utf-8")) if r.get("split") == "val" and not r.get("error")]
            num = lambda r, k: int(float(r[k])) if r.get(k) not in (None, "") else 0  # noqa: E731
            liv = [r for r in rows if r.get("boxes_centre_outside_liver") not in (None, "")]
            canc = [r for r in rows if r.get("gt_lesions") not in (None, "")]
            f["ct_3d"] = {"volumes": len(rows), "regions": sum(num(r, "regions") for r in rows),
                          "outside": sum(num(r, "regions_all_outside_liver") for r in liv), "liv_regions": sum(num(r, "regions") for r in liv),
                          "split": sum(num(r, "lesions_split_in_several_regions") for r in canc),
                          "detected": sum(num(r, "gt_lesions_detected") for r in canc)}
        except Exception:  # noqa: BLE001 - no audit file: the limitation is simply not shown
            pass
        try:                                       # YOLO box area / real lesion box area (Validation, same audit)
            f["box_ratio"] = json.loads((AUDIT_3D.parent / "box_gt_area_ratio.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
        return f

    def load(self) -> None:
        from livercancer.detection.yolo_detector import LesionDetector
        from livercancer.intake import modality
        lock = json.loads(LOCK_FILE.read_text(encoding="utf-8")) if LOCK_FILE.exists() else None
        try:
            pipe = LiverCancerPipeline("ct", self.device, evidence_gate="config")
            if lock:
                x = lock["modalities"]["ct"]
                same = (pipe.detector.conf == x["yolo"]["conf"] and pipe.classifier.aggregation == x["classifier"]["aggregation"]
                        and (pipe.gate is not None) == x["evidence_gate"]["enabled"] and pipe.gate.conf_min == x["evidence_gate"]["conf_min"]
                        and pipe.gate.k == x["evidence_gate"]["k"] and pipe.classifier.reliable)
                if not same:
                    raise RuntimeError("cấu hình CT khác bản đã khóa trước khi chạy Test")
            if pipe.classifier.model is None:
                raise RuntimeError(pipe.classifier.reason or "không nạp được mô hình phân loại CT")
            self.ct, self.state["ct"] = pipe, "ready"
        except Exception as e:  # noqa: BLE001
            self.state["ct"], self.errors["ct"] = "error", str(e)
        try:
            det = LesionDetector.from_config("mri", device=self.device)
            if lock and det.conf != lock["modalities"]["mri"]["yolo"]["conf"]:
                raise RuntimeError("ngưỡng YOLO MRI khác bản đã khóa")
            self.mri, self.state["mri"] = det, "ready"
        except Exception as e:  # noqa: BLE001
            self.state["mri"], self.errors["mri"] = "error", str(e)
        try:
            if modality.available():
                self.modality_model = modality.ModalityClassifier(self.device)
        except Exception as e:  # noqa: BLE001 - without it, 2D images simply ask the user
            self.errors["modality"] = str(e)
        self._warm_up()

    def _warm_up(self) -> None:
        from livercancer.classification.inference import RoiInput
        blank = np.zeros((512, 512), np.uint8)
        try:
            with self.gpu_lock:
                if self.ct:
                    self.ct.detector.predict([blank])
                    z = np.zeros((224, 224), np.uint8)
                    self.ct.classifier.classify([RoiInput(z, z, "warmup", 0, 0)])
                if self.mri:
                    self.mri.predict([blank])
                if self.modality_model:
                    self.modality_model.probs([blank])
        except Exception:  # noqa: BLE001
            pass

    @property
    def device_used(self) -> str:
        d = (self.ct.detector.device if self.ct else self.mri.device if self.mri else "")
        return "GPU" if str(d).startswith("cuda") else "CPU" if d else ""

    # ------------------------------------------------------------------ intake
    def load_inputs(self, files: list[tuple[str, Path, str]], progress: Progress) -> Loaded:
        """files: [(display name, path, kind nifti | image | dicom | zip)]. Every path is deleted right after reading."""
        progress("format", "Đang kiểm tra định dạng", None)
        kinds = {k for _, _, k in files}
        try:
            if kinds <= {"dicom", "zip"}:
                count = [0]

                def blobs():                               # one DICOM at a time; a .zip is read member by member
                    for _, p, k in files:
                        for b in (dicom_io.iter_zip(str(p)) if k == "zip" else [Path(p).read_bytes()]):
                            count[0] += 1
                            if count[0] % 25 == 1:
                                progress("format", f"Đang đọc tệp DICOM ({count[0]} tệp)", None)
                            yield b
                gen = blobs()
                try:
                    series = dicom_io.read_series(gen)
                finally:
                    gen.close()
                return self._from_dicom(series)
            if kinds == {"nifti"}:
                out = Loaded("volume", meta_method="nifti_header")
                for i, (name, p, _) in enumerate(files):
                    progress("format", f"Đang đọc ảnh {i + 1}/{len(files)}", None)
                    vol = self._load_nifti(p)
                    out.items.append((name, vol))
                    sp = self._nifti_spacing(p)          # nibabel zooms (i, j, k) = (x column, y row, z slice)
                    out.vinfo.append({"source": "NIfTI", "spacing": sp, "spacing_xyz": sp, "irregular": False})
                    Path(p).unlink(missing_ok=True)
                return out
            if kinds == {"image"}:
                name, p, _ = files[0]
                img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
                Path(p).unlink(missing_ok=True)
                if img is None:
                    raise UserError("Không đọc được ảnh PNG/JPG. Tệp có thể bị hỏng.")
                return self._from_picture(name, img)
        except dicom_io.DicomError as e:
            raise UserError(str(e)) from None
        finally:
            for _, p, _ in files:
                Path(p).unlink(missing_ok=True)
        raise UserError("Không trộn nhiều loại tệp (NIfTI, DICOM, ảnh 2D) trong cùng một lần phân tích.")

    def _from_picture(self, name: str, img: np.ndarray, note: str | None = None) -> Loaded:
        if max(img.shape[:2]) > 6000:
            raise UserError("Ảnh quá lớn (tối đa khoảng 6000 điểm ảnh mỗi chiều).")
        p = prepare_2d(img)
        if not p.ok:
            raise UserError(p.error_vi)
        out = Loaded("image", items=[(name, img)], prepared=[p])
        if note:
            out.notes.append(note)
        return out

    def _from_dicom(self, series: list) -> Loaded:
        caps = [s for s in series if s.is_capture]
        vols = [s for s in series if not s.is_capture]
        if caps and vols:
            raise UserError("Tệp DICOM vừa có ảnh chụp màn hình vừa có chuỗi ảnh CT/MRI; hãy tải riêng từng loại.")
        if caps:
            if len(caps) > 1:
                raise UserError("Có nhiều ảnh DICOM dạng chụp màn hình; hãy tải từng ảnh một.")
            return self._from_picture("Ảnh 1", caps[0].picture, "DICOM dạng ảnh chụp (Secondary Capture): xử lý như ảnh 2D.")
        mods = {s.modality for s in vols}
        if len(mods) > 1:
            raise UserError("Các chuỗi DICOM thuộc hai loại ảnh khác nhau (CT và MRI). Hãy phân tích riêng từng loại.")
        mod = mods.pop()
        if len(vols) > (4 if mod == "ct" else 8):
            raise UserError(f"Có {len(vols)} chuỗi DICOM; tối đa {4 if mod == 'ct' else 8} chuỗi cho một lần phân tích.")
        single = all(len(s.slices) == 1 for s in vols)
        if single and len(vols) == 1:
            out = Loaded("image", meta_modality=mod, meta_method="dicom_tag")
            out.items = [("Ảnh 1", vols[0].slices[0])]
            out.prepared = [None]                               # a full DICOM slice: no screenshot pre-processing
            return out
        out = Loaded("volume", meta_modality=mod, meta_method="dicom_tag")
        unit = "Pha" if mod == "ct" else "Chuỗi"
        out.items = [(f"{unit} {i + 1}" if len(vols) > 1 else ("Ảnh CT" if mod == "ct" else "Ảnh MRI"), s.volume) for i, s in enumerate(vols)]
        out.vinfo = [{"source": "DICOM", "spacing": list(s.spacing) if s.spacing else None, "files": s.n_files,
                      "duplicate_files": s.duplicates, "irregular": s.irregular,
                      # dicom_io spacing = (row mm, column mm, slice interval) -> (x column, y row, z)
                      "spacing_xyz": [s.spacing[1], s.spacing[0], s.spacing[2]] if s.spacing else None} for s in vols]
        if mod == "ct":
            out.notes.append("DICOM CT đã được chuyển về cửa sổ [−200, 200] HU và 1 mm như dữ liệu huấn luyện.")
        if len(vols) > 1:
            out.notes.append(f"Tệp tải lên có {len(vols)} chuỗi DICOM (SeriesInstanceUID khác nhau); mỗi chuỗi được phân tích "
                             "như một volume riêng, không gộp lại.")
        dup = sum(s.duplicates for s in vols)
        if dup:
            out.notes.append(f"Đã bỏ qua {dup} tệp DICOM trùng hoàn toàn (cùng mã ảnh, hoặc cùng vị trí và cùng dữ liệu điểm ảnh).")
        if any(s.irregular for s in vols):
            out.notes.append("Khoảng cách giữa các lát cắt DICOM không đều (có thể thiếu lát); vị trí lát được giữ theo thứ tự không gian.")
        return out

    @staticmethod
    def _nifti_spacing(path: Path) -> list | None:
        try:
            import nibabel as nib
            z = nib.load(str(path)).header.get_zooms()[:3]
            return [round(float(v), 4) for v in z]
        except Exception:  # noqa: BLE001 - informative only
            return None

    @staticmethod
    def _load_nifti(path: Path) -> np.ndarray:
        try:
            return load_volume_uint8(path)
        except ValueError as e:
            msg = str(e)
            if "orientation" in msg:
                raise UserError("Hướng ảnh khác hướng LPS của dữ liệu huấn luyện, nên không phân tích để tránh kết quả sai.") from None
            if "dtype" in msg:
                raise UserError("Ảnh không cùng định dạng 8 bit (uint8) của dữ liệu huấn luyện.") from None
            raise UserError("Không đọc được ảnh NIfTI.") from None
        except (EOFError, OSError):
            raise UserError("Tệp NIfTI bị hỏng hoặc tải lên chưa đầy đủ.") from None

    # ------------------------------------------------------------------ modality
    def decide_modality(self, loaded: Loaded, header_modality: str | None, requested: str | None,
                        progress: Progress, force: bool = False) -> tuple[str, dict]:
        """The user's choice (ct | mri), checked against the file's own information. A contradiction raises
        ModalityConflict unless the user has confirmed (force). "auto" (API only): metadata, then the image recogniser."""
        progress("modality", "Đang kiểm tra loại ảnh", None)
        meta = loaded.meta_modality or header_modality
        source = loaded.meta_method if loaded.meta_modality else ("nifti_header" if header_modality else None)
        if requested in ("ct", "mri"):
            if meta and meta != requested:
                if not force:
                    raise ModalityConflict(meta, requested, source)
                return requested, {"method": "user_override", "meta": meta, "meta_source": source, "confidence": None}
            return requested, {"method": "user", "meta_source": source if meta else None, "confidence": None}
        if meta:
            return meta, {"method": source, "confidence": None}
        refuse = "Hãy chọn loại ảnh (Phân tích CT hoặc Phân tích MRI) trước khi tải ảnh lên."
        if not self.modality_model:
            raise UserError(refuse)
        if loaded.kind == "image":
            grays = [p.gray if p is not None else cv2.resize(g, (512, 512)) for (_, g), p in zip(loaded.items, loaded.prepared)]
        else:
            from livercancer.intake.modality import representative_slices
            grays = []
            for _, vol in loaded.items:
                for sl in representative_slices(vol, 5):
                    pp = prepare_2d(sl)
                    if pp.ok:
                        grays.append(pp.gray)
        if not grays:
            raise UserError(refuse)
        with self.gpu_lock:
            mod, conf = self.modality_model.decide(grays)
        if mod is None:
            raise UserError(refuse)
        return mod, {"method": "image_model", "confidence": round(conf, 4)}

    # ------------------------------------------------------------------ AI
    def _inputs_for(self, loaded: Loaded, modality: str):
        """(volumes, prep) given to one YOLO. A 2D image is framed like that modality's training slices."""
        if loaded.kind != "image":
            return loaded.items, None
        name, raw = loaded.items[0]
        if loaded.prepared and loaded.prepared[0] is not None:
            prep = prepare_2d(raw, modality)
            if not prep.ok:
                raise UserError(prep.error_vi)
            return [(name, prep.gray)], prep
        return [(name, raw)], None                          # a full DICOM slice: no screenshot pre-processing

    @staticmethod
    def _progress_callbacks(progress: Progress, runs: list[tuple[str, list]], is_2d: bool) -> list:
        """runs = [(label, volumes)] in execution order -> one callback per run; one real percentage over all runs."""
        sizes = [[1 if is_2d else v.shape[2] for _, v in vols] for _, vols in runs]
        total = max(1, sum(map(sum, sizes)))
        cbs = []
        for r, (label, vols) in enumerate(runs):
            before = sum(map(sum, sizes[:r]))

            def cb(vi: int, done: int, n: int, r=r, label=label, vols=vols, before=before) -> None:
                base = before + sum(sizes[r][:vi])
                frac = done / max(1, n)                     # n < slices when exact duplicates are skipped
                progress("detect", f"Đang phát hiện vùng tổn thương{' – ' + label if label else ''}: ảnh {vi + 1}/{len(vols)}, "
                                   f"lát {done}/{n}", 100.0 * (base + frac * sizes[r][vi]) / total)
            cbs.append(cb)
        return cbs

    def run_models(self, loaded: Loaded, modality: str, mod_info: dict, progress: Progress) -> tuple[dict, dict]:
        """CT: YOLO CT -> Tumor Zoom -> Hybrid CT -> Evidence Gate (locked pipeline). MRI: YOLO MRI -> Tumor Zoom."""
        t0 = time.perf_counter()
        is_2d = loaded.kind == "image"
        if self.state.get(modality) != "ready":
            raise UserError("Mô hình " + modality.upper() + " chưa sẵn sàng trên máy chủ. Vui lòng thử lại sau ít phút.")
        vols, prep = self._inputs_for(loaded, modality)
        if not is_2d:
            n = sum(v.shape[2] for _, v in vols)
            progress("extract", f"Đang tách {n} lát cắt 2D (axial) từ {len(vols)} volume"
                     + (" và kiểm tra lát trùng hoàn toàn" if self.dedup_exact else ""), None)
        cb = self._progress_callbacks(progress, [("", vols)], is_2d)[0]
        det_real = self.ct.detector if modality == "ct" else self.mri
        wrapped = _ProgressDetector(det_real, cb)
        dd = DedupDetector(wrapped) if (self.dedup_exact and not is_2d) else None
        with self.gpu_lock:
            res = self._run_ct(vols, is_2d, dd or wrapped) if modality == "ct" else self._run_detect(dd or wrapped, "mri", vols, is_2d)
        progress("zoom", "Đang phóng to vùng nghi ngờ", None)
        progress("summary", "Đang tổng hợp kết quả", None)
        out, viewer = self._result(res, modality, is_2d, mod_info, loaded, prep)
        out["case"] = self._case(res, loaded, dd.stats if dd else None, is_2d, prep)
        out["case"]["regions3d"] = [] if is_2d else self.regions_3d(res, loaded)
        # "strong region" threshold of the Evidence Gate rule that was applied (locked config; display only)
        g = (self.ct.gate_single if len(vols) == 1 else self.ct.gate) if modality == "ct" and self.ct else None
        out["details"]["strong_conf"] = float(g.conf_min) if g is not None else None
        out["timings_s"] = {"total": round(time.perf_counter() - t0, 1)}
        return out, viewer

    def _run_ct(self, vols, is_2d, det) -> PipelineResult:
        pipe = self.ct
        real = pipe.detector
        pipe.detector = det
        try:
            res = pipe.run(vols, "", is_2d=is_2d)
        finally:
            pipe.detector = real
        if res.status == "error":
            raise RuntimeError(res.message)
        return res

    @staticmethod
    def _run_detect(det, modality: str, vols, is_2d) -> PipelineResult:
        """YOLO -> Tumor Zoom only (same analyze_volume / analyze_image calls as the locked pipeline)."""
        detector = det
        t0 = time.perf_counter()
        out = PipelineResult("detection_only", "", modality, warnings=[DISCLAIMER] if modality == "mri" else [], device=detector.device)
        first = 0
        for name, arr in vols:
            rr = analyze_image(arr, det) if is_2d else analyze_volume(arr, det)
            for i, t in enumerate(rr.tracks):
                out.regions.append({"track_id": first + i, "volume": name, "k_first": t.ks[0], "k_last": t.ks[-1],
                                    "n_slices": t.n_slices, "max_conf": round(t.max_conf, 4), "best_slice": t.best_slice,
                                    "box_best_slice": [round(v, 1) for v in t.boxes[t.best_slice].xyxy]})
            first += len(rr.tracks)
            out.roi_results[name] = rr
            out.volumes.append({"name": name, "n_slices": rr.n_slices, "n_regions": len(rr.tracks),
                                "slices_with_boxes": len(rr.slices_with_boxes())})
        out.timings_s["detection_zoom"] = round(time.perf_counter() - t0, 2)
        if not out.regions:
            out.status, out.message = "no_lesion_detected", NO_LESION.format(conf=detector.conf)
        return out

    # ------------------------------------------------------------------ presentation
    def limitations(self, modality: str) -> list[str]:
        """Detailed limits, shown under "Chi tiết nâng cao" (numbers from the official Test run)."""
        w = []
        s = self.facts.get("ct_small")
        if modality == "ct":
            w.append("Mô hình CT phát hiện kém tổn thương nhỏ (dưới khoảng 32×32 điểm ảnh)"
                     + (f": trên tập Test chỉ tìm thấy {s[0]}/{s[1]} tổn thương nhỏ." if s else "."))
        if modality == "ct":
            w.append("Mô hình CT chỉ phân biệt HCC / ICC / CHCC, không có lớp \"không có tổn thương\"; hệ thống không kết luận gan bình thường.")
            w.append("Evidence Gate chỉ cho phân loại khi vùng nghi ngờ đủ mạnh; nếu không, hệ thống không đưa ra nhãn.")
        if modality == "mri":
            a = self.facts.get("mri_acc")
            w.append("Phân loại MRI chưa đạt chất lượng" + (f" (Test: đúng {a[0]}/{a[1]} bệnh nhân)" if a else "")
                     + " nên không được dùng; với MRI hệ thống chỉ phát hiện và phóng to vùng nghi ngờ.")
        w.append("YOLO dùng ngưỡng thấp (0,12) để hạn chế bỏ sót, nên có thể có vùng giả.")
        a3 = self.facts.get("ct_3d")
        if modality == "ct" and a3 and a3["volumes"]:
            w.append(f"Kiểm toán trên {a3['volumes']} volume CT Validation: trung bình {_vn(a3['regions'] / a3['volumes'])} vùng nghi ngờ mỗi volume; "
                     f"{_vn(100 * a3['outside'] / max(1, a3['liv_regions']), 0)}% vùng nằm hoàn toàn ngoài gan; {a3['split']}/{a3['detected']} tổn thương "
                     "được phát hiện bị tách thành nhiều vùng. Vì vậy số vùng nghi ngờ không phải số khối u (reports/case3d_audit.md).")
        br = self.facts.get("box_ratio")
        if modality == "ct" and br:
            w.append(f"Kích thước khung YOLO CT: trên Validation, diện tích khung so với tổn thương thật có trung vị {_vn(br['median'], 2)} lần, "
                     f"nhưng 10% khung lớn hơn {_vn(br['p90'])} lần và {_vn(100 * br['share_over_4x'], 0)}% lớn hơn 4 lần. Khối bao 3D vì vậy "
                     "không đo kích thước khối u và có thể lớn hơn tổn thương nhiều lần ở một số ca.")
        w.append("Ảnh Tumor Zoom trên màn hình được cắt rộng hơn quanh tổn thương và phóng tối đa 4 lần (nội suy bicubic, "
                 "không làm nét, không tạo chi tiết); ảnh đưa vào mô hình phân loại vẫn là vùng cắt 224×224 như lúc huấn luyện.")
        return w

    @staticmethod
    def region_numbers(res: PipelineResult) -> dict[str, dict[int, int]]:
        """volume -> {track index: region number}. Numbers follow the website order (highest detection confidence
        first, the same order as the region chips), over ALL regions of the case."""
        order = sorted(res.regions, key=lambda r: -r["max_conf"])
        first = {name: min((r["track_id"] for r in res.regions if r["volume"] == name), default=0) for name in res.roi_results}
        out: dict[str, dict[int, int]] = {name: {} for name in res.roi_results}
        for i, r in enumerate(order):
            out[r["volume"]][r["track_id"] - first[r["volume"]]] = i + 1
        return out

    @classmethod
    def regions_3d(cls, res: PipelineResult, loaded: Loaded) -> list[dict]:
        """Every 3D region (track of linked per-slice YOLO boxes, locked rule: IoU >= link_iou on consecutive slices,
        at most link_max_gap empty slices) with its boxes per slice and its 3D bounding box. x / y in pixels of the
        displayed slice, z = slice index. Millimetres only when the spacing is known and the slice interval regular.
        The box is the hull of 2D YOLO boxes, NOT a segmentation: its volume is not the lesion volume."""
        nums = cls.region_numbers(res)
        first = {name: min((r["track_id"] for r in res.regions if r["volume"] == name), default=0) for name in res.roi_results}
        out = []
        for vi, (name, rr) in enumerate(res.roi_results.items()):
            info = loaded.vinfo[vi] if vi < len(loaded.vinfo) else {}
            sp = info.get("spacing_xyz")
            reliable = bool(sp and all(v for v in sp) and not info.get("irregular"))
            for ti, t in enumerate(rr.tracks):
                ks = sorted(t.boxes)
                bx = [t.boxes[k].xyxy for k in ks]
                x0, y0 = min(b[0] for b in bx), min(b[1] for b in bx)
                x1, y1 = max(b[2] for b in bx), max(b[3] for b in bx)
                reg = {"id": first[name] + ti, "num": nums[name].get(ti), "volume": name, "k_first": ks[0], "k_last": ks[-1],
                       "best_slice": int(t.best_slice), "n_slices": len(ks), "gap_slices": ks[-1] - ks[0] + 1 - len(ks),
                       "max_conf": round(float(t.max_conf), 4),
                       "boxes": [[int(k), *(round(float(v), 1) for v in t.boxes[k].xyxy), round(float(t.boxes[k].conf), 4)] for k in ks],
                       "bbox_vox": {"x_min": round(x0, 1), "y_min": round(y0, 1), "z_min": ks[0], "x_max": round(x1, 1),
                                    "y_max": round(y1, 1), "z_max": ks[-1]}}
                if reliable:
                    reg["size_mm"] = [round((x1 - x0) * sp[0], 1), round((y1 - y0) * sp[1], 1), round((ks[-1] - ks[0] + 1) * sp[2], 1)]
                out.append(reg)
        out.sort(key=lambda r: r["num"] or 10 ** 6)
        return out

    @staticmethod
    def frames_sprite(rr, start: int, count: int, quality: int = 88) -> bytes:
        """`count` consecutive slices stacked vertically in one JPEG (video playback; plain images, boxes drawn by the
        browser from the region data)."""
        n = rr.n_slices
        start = int(np.clip(start, 0, n - 1))
        ks = range(start, min(n, start + count))
        img = np.vstack([rr.images[k] if rr.images[k].ndim == 2 else cv2.cvtColor(rr.images[k], cv2.COLOR_BGR2GRAY) for k in ks])
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    @staticmethod
    def _case(res: PipelineResult, loaded: Loaded, dstats, is_2d: bool, prep) -> dict:
        """Case summary (3D): real counts per volume and every slice that holds a suspicious region (kept 3D tracks,
        the same boxes as the viewer). Detection confidence only: never a probability of disease."""
        vols, slices = [], []
        for i, (name, rr) in enumerate(res.roi_results.items()):
            vi = loaded.vinfo[i] if i < len(loaded.vinfo) else {}
            st = dstats[i] if dstats and i < len(dstats) else None
            per: dict[int, list] = {}
            for t in rr.tracks:
                for k, b in t.boxes.items():
                    per.setdefault(int(k), []).append(float(b.conf))
            for k in sorted(per):
                slices.append({"volume": name, "k": k, "conf": round(max(per[k]), 4), "n_boxes": len(per[k])})
            h, w = rr.images[0].shape[:2]
            vols.append({"name": name, "source": vi.get("source") or ("Ảnh 2D" if is_2d else ""), "width": int(w), "height": int(h),
                         "n_slices": rr.n_slices, "yolo_slices": st.n_unique if st else rr.n_slices,
                         "duplicate_slices": st.n_duplicates if st else 0, "spacing": vi.get("spacing"),
                         "dicom_files": vi.get("files"), "dicom_duplicate_files": vi.get("duplicate_files", 0),
                         "spacing_xyz": vi.get("spacing_xyz"),
                         "irregular": bool(vi.get("irregular")), "slices_with_regions": len(per), "n_regions": len(rr.tracks)})
        top = max(slices, key=lambda x: x["conf"]) if slices else None
        orig = None
        if is_2d and loaded.items:
            raw = loaded.items[0][1]
            orig = [int(raw.shape[1]), int(raw.shape[0])]
        return {"is_3d": not is_2d, "volumes": vols, "total_slices": sum(v["n_slices"] for v in vols),
                "yolo_slices": sum(v["yolo_slices"] for v in vols), "duplicate_slices": sum(v["duplicate_slices"] for v in vols),
                "dicom_duplicate_files": sum(v["dicom_duplicate_files"] or 0 for v in vols),
                "slices_with_regions": len(slices), "suspicious_slices": slices, "top_slice": top,
                "dedup": dstats is not None, "original_size": orig, "screen_prepared": prep is not None}

    def thumb(self, roi_results: dict, volume: str, k: int, side: int = 128) -> bytes:
        """Small JPEG of one slice with its suspicious-region boxes (thumbnail strip)."""
        rr = roi_results[volume]
        k = int(np.clip(k, 0, rr.n_slices - 1))
        img = cv2.cvtColor(rr.images[k], cv2.COLOR_GRAY2BGR) if rr.images[k].ndim == 2 else rr.images[k].copy()
        for _, b in rr.boxes_on(k):
            x0, y0, x1, y1 = (int(round(v)) for v in b.xyxy)
            cv2.rectangle(img, (x0, y0), (x1, y1), (59, 59, 239), max(2, img.shape[0] // 128))
        h, w = img.shape[:2]
        sc = side / max(h, w)
        img = cv2.resize(img, (max(1, round(w * sc)), max(1, round(h * sc))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()

    @staticmethod
    def _top_slice(res: PipelineResult, regions: list[dict]) -> tuple[str, int]:
        if regions:
            return regions[0]["volume"], int(regions[0]["best_slice"])
        name, rr = next(iter(res.roi_results.items()))
        return name, rr.n_slices // 2

    @staticmethod
    def _boxes_on(rr, k: int) -> list[dict]:
        return [{"box": [round(float(v), 1) for v in b.xyxy], "conf": round(float(b.conf), 3), "track": int(rr.tracks.index(t))}
                for t, b in rr.boxes_on(k)]

    def slice_view(self, roi_results: dict, volume: str, k: int, first_ids: dict) -> dict:
        rr = roi_results[volume]
        k = int(np.clip(k, 0, rr.n_slices - 1))
        boxes = self._boxes_on(rr, k)
        for b in boxes:
            b["region"] = first_ids[volume] + b.pop("track")
        img = rr.images[k]
        return {"volume": volume, "slice": k, "n_slices": rr.n_slices, "width": int(img.shape[1]), "height": int(img.shape[0]),
                "png": _png_b64(img), "boxes": boxes}

    @staticmethod
    def _annotate(img: np.ndarray, boxes: list[dict], main_id) -> np.ndarray:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        for b in boxes:
            x0, y0, x1, y1 = (int(round(v)) for v in b["box"])
            main = b["region"] == main_id
            cv2.rectangle(out, (x0, y0), (x1, y1), (59, 59, 239) if main else (11, 158, 245), 2 if main else 1)
            cv2.putText(out, f"{b['conf']:.2f}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (59, 59, 239), 1, cv2.LINE_AA)
        return out

    @staticmethod
    def _cls_names() -> tuple[list[str], dict]:
        cfg = load_yaml(CONFIG_DIR / "cls.yaml")
        classes = cfg["ct"]["classes"]
        return classes, dict(zip(classes, cfg["ct"]["class_names_vi"]))

    def _branch(self, res: PipelineResult, key: str, is_2d: bool) -> tuple[dict, list[dict], np.ndarray]:
        """Everything one model shows: slice view, regions with display zooms, chart, region table."""
        classes, nvi = self._cls_names()
        regions = sorted(res.regions, key=lambda r: -r["max_conf"])
        first_ids = {name: min((r["track_id"] for r in res.regions if r["volume"] == name), default=0) for name in res.roi_results}
        shown = []
        for i, r in enumerate(regions[:MAX_REGIONS_SHOWN]):
            rr = res.roi_results[r["volume"]]
            tr = rr.tracks[r["track_id"] - first_ids[r["volume"]]]
            k = r["best_slice"]
            box = tuple(float(v) for v in tr.boxes[k].xyxy)
            zl, zc = display_zoom(rr.images[k], box, "local"), display_zoom(rr.images[k], box, "context")
            reg = {"id": r["track_id"], "rank": i + 1, "volume": r["volume"], "slice": int(k), "slice_first": int(r["k_first"]),
                   "slice_last": int(r["k_last"]), "n_slices": int(r["n_slices"]), "confidence": round(float(r["max_conf"]), 4),
                   "box": [round(v, 1) for v in box],
                   "local_png": _png_b64(zl["image"]), "context_png": _png_b64(zc["image"]),
                   "local_window": list(zl["window"]), "context_window": list(zc["window"]),
                   "local_scale": zl["scale"], "context_scale": zc["scale"]}
            if key == "ct" and r.get("scores"):
                reg["label_short"] = short(nvi.get(r["label"], r["label"]))
            shown.append(reg)
        vol0, k0 = self._top_slice(res, regions)
        view = self.slice_view(res.roi_results, vol0, k0, first_ids)
        annotated = self._annotate(res.roi_results[vol0].images[view["slice"]], view["boxes"], regions[0]["track_id"] if regions else None)
        if res.status == "ok":
            chart = {"kind": "class_scores", "title": "Điểm dự đoán từng lớp (Hybrid CT)",
                     "items": [{"label": short(nvi[c]), "full": nvi[c], "value": round(float(res.scores[c]), 4)} for c in classes]}
        elif regions:
            chart = {"kind": "detection", "title": f"Độ tin cậy phát hiện theo vùng ({BRANCH_VI[key]})",
                     "items": [{"label": f"Vùng {i + 1}", "full": f"{r['volume']} · lát {r['best_slice'] + 1}" if not is_2d else r["volume"],
                                "value": round(float(r["max_conf"]), 4)} for i, r in enumerate(regions[:MAX_REGIONS_SHOWN])]}
        else:
            chart = {"kind": "none", "title": "Biểu đồ phân tích", "items": []}
        b = {"key": key, "title": BRANCH_VI[key], "status": res.status, "view": view, "regions": shown, "n_regions": len(regions),
             "max_conf": round(float(regions[0]["max_conf"]), 4) if regions else None, "first_ids": first_ids, "chart": chart,
             "volumes": [{"name": v["name"], "n_slices": v["n_slices"]} for v in res.volumes],
             "all_regions": [{"volume": r["volume"], "slices": f"{r['k_first'] + 1}–{r['k_last'] + 1}", "confidence": round(float(r["max_conf"]), 4),
                              "label": short(nvi.get(r.get("label"), "")) if key == "ct" and r.get("label") else None} for r in regions]}
        return b, regions, annotated

    def _commentary(self, res, modality, regions, nvi, is_2d, n_slices, n_vol, prep) -> list[str]:
        """Case-level comments. Wording: "vùng nghi ngờ" (never a confirmed tumour); detection confidence (YOLO) and the
        Hybrid CT score (research model, softmax) are named as such, never as a probability of cancer."""
        n = len(regions)
        mod = "CT" if modality == "ct" else "MRI"
        if is_2d:
            where = f"trên ảnh {mod} 2D"
        elif n_vol == 1:
            where = f"trong một ca chụp {mod} 3D gồm {n_slices} lát cắt"
        else:
            where = f"trong ca chụp {mod} 3D gồm {n_vol} {'pha' if modality == 'ct' else 'chuỗi'}, tổng {n_slices} lát cắt"
        out = []
        if n == 0:
            out.append(f"Hệ thống không phát hiện vùng nghi ngờ {where} ở ngưỡng phát hiện của mô hình.")
            out.append("Vì mô hình không có lớp \"gan bình thường\", kết quả này không khẳng định gan bình thường.")
        else:
            top = regions[0]
            loc = "" if is_2d else (f" (vùng ở lát {top['k_first'] + 1}–{top['k_last'] + 1}{'' if n_vol == 1 else ', ' + top['volume']};"
                                    f" lát đại diện {top['best_slice'] + 1})")
            out.append(f"Hệ thống phát hiện {n} vùng nghi ngờ {where}. Độ tin cậy phát hiện (YOLO) cao nhất là "
                       f"{_vn(top['max_conf'], 2)}{loc}. Đây là vùng cần xem xét, chưa phải tổn thương đã được xác nhận.")
            if res.status == "ok":
                sc = sorted(res.scores.items(), key=lambda x: -x[1])
                out.append(f"Mô hình nghiên cứu Hybrid CT cho điểm dự đoán cao nhất ở {short(nvi[sc[0][0]])}: {_vn(100 * sc[0][1])}%"
                           f" (tiếp theo {short(nvi[sc[1][0]])} {_vn(100 * sc[1][1])}%). Đây là điểm của mô hình cho toàn ca, "
                           "không phải xác suất người bệnh mắc ung thư.")
            elif res.status == "insufficient_evidence":
                ev = res.evidence or {}
                out.append(f"Bằng chứng hình ảnh chưa đủ để phân loại (độ tin cậy phát hiện cao nhất {_vn(ev.get('max_conf', 0), 2)}"
                           + (f", vùng đủ mạnh xuất hiện ở {ev.get('n_volumes_strong', 0)} pha" if n_vol > 1 else "")
                           + "), nên hệ thống không đưa ra nhãn.")
            else:
                out.append("Mô hình MRI chỉ phát hiện và định vị vùng nghi ngờ, không phân loại loại tổn thương.")
        if prep is not None:
            out.append(self._note_2d(prep))
        return out

    @staticmethod
    def _note_2d(prep) -> str:
        return ("Ảnh đầu vào là ảnh 2D" + (" chụp từ màn hình" if prep.screenshot_like else "")
                + "; mô hình được kiểm chứng trên ảnh y tế gốc, nên kết quả trên ảnh này chỉ mang tính tham khảo.")

    @staticmethod
    def _preprocessing(loaded: Loaded, prep) -> dict:
        raw = loaded.items[0][1]
        return {"original_jpg": _thumb_b64(raw if raw.ndim == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)),
                "prepared_png": _png_b64(prep.gray), "steps": prep.steps_vi, "warnings": prep.warnings_vi}

    def _result(self, res: PipelineResult, modality: str, is_2d: bool, mod_info: dict, loaded: Loaded, prep) -> tuple[dict, dict]:
        classes, nvi = self._cls_names()
        b, regions, annotated = self._branch(res, modality, is_2d)
        n_slices = sum(v["n_slices"] for v in res.volumes)
        ok = res.status == "ok"
        if ok:
            top = max(res.scores, key=res.scores.get)
            card_result = {"title": f"Nghi ngờ {short(nvi[top])}", "sub": nvi[top], "tone": "result"}
            card_conf = {"value": f"{_vn(100 * res.scores[top])}%", "sub": "điểm dự đoán của mô hình nghiên cứu, không phải xác suất mắc bệnh"}
        elif res.status == "insufficient_evidence":
            card_result = {"title": "Chưa đủ cơ sở để phân loại", "sub": "vùng nghi ngờ chưa đủ mạnh để phân loại", "tone": "pending"}
            card_conf = {"value": "—", "sub": "không phân loại"}
        elif res.status == "no_lesion_detected":
            card_result = {"title": "Không phát hiện vùng nghi ngờ trong " + ("ảnh này" if is_2d else "ca chụp này"),
                           "sub": "ở ngưỡng phát hiện của mô hình; không đồng nghĩa với gan bình thường", "tone": "clear"}
            card_conf = {"value": "—", "sub": "không có vùng để đánh giá"}
        else:
            card_result = {"title": f"Phát hiện {len(regions)} vùng nghi ngờ", "sub": "đã khoanh và phóng to (MRI không phân loại)", "tone": "info"}
            card_conf = {"value": _vn(regions[0]["max_conf"], 2), "sub": "độ tin cậy phát hiện cao nhất (YOLO)"}
        ev = res.evidence or {}
        mod_vi = "CT" if modality == "ct" else "MRI"
        notes = list(loaded.notes)
        if mod_info.get("method") == "user_override":
            notes.append(f"Bạn đã xác nhận phân tích như ảnh {mod_vi} dù {SOURCE_VI.get(mod_info.get('meta_source'), 'thông tin trong tệp')} "
                         f"ghi là {str(mod_info.get('meta')).upper()}; mô hình {mod_vi} chỉ được kiểm chứng trên ảnh {mod_vi}.")
        details = {"limitations": self.limitations(modality),
                   "evidence": ({"passed": bool(ev.get("passed")), "rule": ev.get("rule", ""), "max_conf": round(float(ev.get("max_conf", 0)), 4),
                                 "n_volumes_strong": int(ev.get("n_volumes_strong", 0))} if ev else None),
                   "modality": {"value": mod_vi, "method": method_vi(mod_info), "confidence": mod_info.get("confidence")},
                   "volumes": [{"name": v["name"], "n_slices": v["n_slices"], "n_regions": v["n_regions"]} for v in res.volumes],
                   "device": self.device_used}
        if prep is not None:
            details["preprocessing"] = self._preprocessing(loaded, prep)
            notes += prep.warnings_vi
        out = {
            "mode": "single", "modality": modality, "status": res.status, "is_2d": is_2d,
            "cards": {"result": card_result, "confidence": card_conf,
                      "modality": {"value": mod_vi, "sub": method_vi(mod_info)},
                      "regions": {"value": str(len(regions)), "sub": f"trên {len(res.volumes)} ảnh · {n_slices} lát cắt" if not is_2d else "trên ảnh 2D"}},
            "label": res.label if ok else None, "label_name": nvi.get(res.label) if ok else None,
            "scores": [{"code": c, "name": short(nvi[c]), "full_name": nvi[c], "prob": round(float(res.scores[c]), 4)} for c in classes] if ok else [],
            "branches": [b], "n_regions": len(regions), "original": None,
            "commentary": self._commentary(res, modality, regions, nvi, is_2d, n_slices, len(res.volumes), prep),
            "notes": notes, "footnote": FOOTNOTE, "details": details, "annotated_png": _png_b64(annotated),
            "device": self.device_used,
        }
        return out, {modality: res.roi_results}

