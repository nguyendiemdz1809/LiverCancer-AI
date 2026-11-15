"""Upload validation and CT / MRI recognition from the file header (and the file name as a hint).

Only the header is read here (the browser sends the first bytes of each file); voxel data is never needed to decide.
Recognition rule, calibrated on the real datasets (header only):
  CT  PLC-CECT : 512 x 512 in-plane, isotropic 1.0 mm voxels, uint8, sometimes 5D (X, Y, Z, 1, 3) (all files)
  MRI LLD-MMRI : anisotropic voxels: slice spacing 2-11 mm and >= 1.5 x in-plane (0.62-1.93 mm), never isotropic, uint8
When header and name disagree, or neither is conclusive, the modality is "unknown" and the user chooses.
Files that differ from the training format (orientation not LPS, not uint8) are refused with a Vietnamese reason,
because the models were only validated on that format.
"""
from __future__ import annotations

import io
import re
import zlib
from dataclasses import asdict, dataclass, field

import numpy as np

GZIP_MAGIC = b"\x1f\x8b"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
NII_EXT = (".nii.gz", ".nii")
IMG_EXT = (".png", ".jpg", ".jpeg")
DICOM_EXT = (".dcm", ".dicom", ".ima")
ZIP_MAGIC = b"PK"
MAX_VOXELS = 512 * 512 * 1200 * 3            # decompression-bomb guard (largest real CT: 512 x 512 x 222 x 3)

CT_PHASES = ("C1", "C2", "C3", "P")
MRI_SEQS = {"t2wi": "T2WI", "dwi": "DWI", "inphase": "In Phase", "in phase": "In Phase", "outphase": "Out Phase",
            "out phase": "Out Phase", "c+a": "C+A", "c+v": "C+V", "c+delay": "C+Delay", "c-pre": "C-pre"}


@dataclass
class FileInfo:
    index: int
    kind: str = "unknown"                 # nifti | image | unknown
    ok: bool = False
    error_vi: str = ""
    modality: str = "unknown"             # ct | mri | unknown
    modality_reason_vi: str = ""
    series: str = ""                      # display name: CT phase / MRI sequence / "Ảnh n" (never the file name)
    shape: list[int] = field(default_factory=list)
    spacing_mm: list[float] = field(default_factory=list)
    n_slices: int = 0
    warnings_vi: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _name_hints(name: str) -> tuple[str, str]:
    """(modality hint, series) from the file name. Never returns the name itself."""
    low = name.lower()
    stem = re.sub(r"\.(nii\.gz|nii|png|jpe?g)$", "", low)
    for key, seq in sorted(MRI_SEQS.items(), key=lambda x: -len(x[0])):
        if re.search(rf"(^|[_\-. ]){re.escape(key)}([_\-. ]|$)", stem):
            return "mri", seq
    m = re.search(r"(^|[_\-. ])ct[_\-. ](c1|c2|c3|p)([_\-. ]|$)", stem) or re.search(r"(^|[_\-. ])(c1|c2|c3)([_\-. ]|$)", stem)
    if m:
        return "ct", m.group(m.lastindex - 1).upper()
    if re.search(r"(^|[_\-. ])ct([_\-. ]|$)", stem):
        return "ct", ""
    if re.search(r"(^|[_\-. ])(mri?|mr-?\d+)", stem):
        return "mri", ""
    return "unknown", ""


def _parse_nifti_header(head: bytes, gz: bool):
    import nibabel as nib
    raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(head) if gz else head
    if len(raw) < 348:
        raise ValueError("header too short")
    size = int.from_bytes(raw[:4], "little")
    cls = nib.Nifti2Header if size == 540 else nib.Nifti1Header
    if size not in (348, 540) and int.from_bytes(raw[:4], "big") not in (348, 540):
        raise ValueError("not a NIfTI header")
    return cls.from_fileobj(io.BytesIO(raw[:540] if size == 540 else raw[:348]))


def inspect_head(index: int, name: str, head: bytes, size: int, max_bytes: int) -> FileInfo:
    """Validate one file from its first bytes. `size` = full file size in bytes."""
    import nibabel as nib
    info = FileInfo(index)
    low = name.lower()
    if size > max_bytes:
        info.error_vi = f"Tệp quá lớn ({size / 2 ** 20:.0f} MB). Giới hạn là {max_bytes / 2 ** 20:.0f} MB mỗi tệp."
        return info
    if head[128:132] == b"DICM" or low.endswith(DICOM_EXT):
        info.kind = "dicom"
        if head[128:132] != b"DICM":
            info.error_vi = "Tệp không phải DICOM hợp lệ (thiếu phần đầu DICM)."
            return info
        try:
            import pydicom
            ds = pydicom.dcmread(io.BytesIO(head), stop_before_pixels=True, force=False,
                                 specific_tags=["Modality", "NumberOfFrames"])
            tag = str(getattr(ds, "Modality", "")).upper()
        except Exception:  # noqa: BLE001
            tag = ""
        if tag == "CT" or tag == "MR":
            info.modality = "ct" if tag == "CT" else "mri"
            info.modality_reason_vi = f"Thông tin DICOM ghi loại ảnh {tag}."
        elif tag and tag not in ("SC", "OT"):
            info.error_vi = f"Ảnh DICOM loại \"{tag}\" không được hỗ trợ (chỉ hỗ trợ CT và MRI)."
            return info
        info.n_slices = 1
        info.ok = True
        return info
    if low.endswith(".zip"):
        info.kind = "zip"
        if not head.startswith(ZIP_MAGIC):
            info.error_vi = "Tệp .zip không hợp lệ."
            return info
        info.modality_reason_vi = "Tệp .zip (chuỗi DICOM): loại ảnh được đọc khi giải nén."
        info.ok = True
        return info
    if low.endswith(NII_EXT):
        info.kind = "nifti"
        gz = head[:2] == GZIP_MAGIC
        if low.endswith(".nii.gz") != gz:
            info.error_vi = "Nội dung tệp không khớp đuôi .nii / .nii.gz."
            return info
        try:
            hdr = _parse_nifti_header(head, gz)
        except Exception:  # noqa: BLE001
            info.error_vi = "Không đọc được header NIfTI. Tệp có thể bị hỏng hoặc không phải ảnh NIfTI."
            return info
        shape = [int(x) for x in hdr.get_data_shape()]
        zooms = [round(float(z), 3) for z in hdr.get_zooms()[:3]]
        info.shape, info.spacing_mm = shape, zooms
        if len(shape) < 3 or (len(shape) > 3 and not (len(shape) == 5 and shape[3] == 1 and shape[4] == 3)) or min(shape[:3]) < 1:
            info.error_vi = f"Ảnh NIfTI phải là khối 3D (nhận được kích thước {tuple(shape)})."
            return info
        if int(np.prod(shape, dtype=np.int64)) > MAX_VOXELS:
            info.error_vi = "Ảnh quá lớn so với dữ liệu mà mô hình được kiểm chứng."
            return info
        info.n_slices = shape[2]
        dtype = hdr.get_data_dtype()
        slope = hdr.get_slope_inter()[0]
        if dtype != np.uint8 or (slope not in (None, 1.0) and not np.isnan(slope)):
            info.error_vi = ("Ảnh không cùng định dạng dữ liệu huấn luyện: mô hình chỉ được kiểm chứng trên ảnh đã chuẩn hóa sẵn "
                             "về 8 bit (uint8), giống bộ PLC-CECT / LLD-MMRI. Ảnh này dùng kiểu dữ liệu khác.")
            return info
        codes = nib.aff2axcodes(hdr.get_best_affine())
        if codes != ("L", "P", "S"):
            info.error_vi = f"Hướng ảnh {''.join(codes)} khác hướng LPS của dữ liệu huấn luyện, nên không phân tích để tránh kết quả sai."
            return info
        hint, series = _name_hints(name)
        iso = all(abs(z - 1.0) < 0.05 for z in zooms)
        header_mod = ("ct" if (iso and shape[0] == shape[1] == 512) or len(shape) == 5 else
                      "mri" if zooms[2] >= 2.0 and zooms[2] >= 1.5 * max(zooms[:2]) else "unknown")
        if header_mod != "unknown" and hint in (header_mod, "unknown"):
            info.modality = header_mod
            info.modality_reason_vi = ("Header giống ảnh CT của bộ dữ liệu (512×512, voxel 1 mm đẳng hướng)." if header_mod == "ct" else
                                       "Header giống ảnh MRI của bộ dữ liệu (lát cắt dày hơn nhiều so với độ phân giải trong lát).")
        elif header_mod == "unknown" and hint != "unknown":
            info.modality_reason_vi = "Chỉ tên tệp gợi ý loại ảnh; header không đủ để chắc chắn."
        elif header_mod != "unknown":
            info.modality_reason_vi = "Tên tệp và header gợi ý hai loại ảnh khác nhau."
        else:
            info.modality_reason_vi = "Không xác định chắc chắn được ảnh CT hay MRI."
        info.series = series or ""
        info.ok = True
        return info
    if low.endswith(IMG_EXT):
        info.kind = "image"
        if not (head.startswith(PNG_MAGIC) or head.startswith(JPEG_MAGIC)):
            info.error_vi = "Nội dung tệp không phải ảnh PNG/JPG hợp lệ."
            return info
        info.n_slices = 1
        hint, series = _name_hints(name)
        info.series = series
        info.modality_reason_vi = "Ảnh 2D: loại ảnh được nhận dạng bằng mô hình ảnh."
        info.ok = True
        return info
    info.error_vi = "Định dạng không được hỗ trợ. Hãy tải ảnh NIfTI (.nii, .nii.gz), DICOM (.dcm hoặc .zip) hoặc ảnh PNG/JPG."
    return info


def decide_modality(infos: list[FileInfo]) -> tuple[str, str]:
    """Modality of a whole upload: (ct | mri | unknown | mixed, Vietnamese reason)."""
    ok = [i for i in infos if i.ok]
    kinds = {i.kind for i in ok}
    if len(kinds) > 1 and not kinds <= {"dicom", "zip"}:
        return "mixed", "Không trộn nhiều loại tệp (NIfTI, DICOM, ảnh 2D) trong cùng một lần phân tích."
    mods = {i.modality for i in ok}
    if mods == {"ct"} or mods == {"mri"}:
        return mods.pop(), ok[0].modality_reason_vi
    if {"ct", "mri"} <= mods:
        return "mixed", "Các tệp thuộc hai loại ảnh khác nhau (CT và MRI). Hãy phân tích riêng từng loại."
    return "unknown", next((i.modality_reason_vi for i in ok if i.modality == "unknown"), "Không xác định được loại ảnh.")


def display_names(infos: list[FileInfo], modality: str) -> list[str]:
    """Unique display names; the original file name is never used (it can contain identifiers or labels)."""
    out, seen = [], set()
    for n, i in enumerate(infos, 1):
        base = i.series or f"Ảnh {n}"
        if modality == "ct" and i.series in CT_PHASES:
            base = f"Pha {i.series}"
        name, j = base, 2
        while name in seen:
            name, j = f"{base} ({j})", j + 1
        seen.add(name)
        out.append(name)
    return out
