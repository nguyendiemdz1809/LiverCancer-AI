"""DICOM intake: one file, several files, or a .zip of a series -> per-series uint8 axial slices in the training convention.

Modality comes from the DICOM `Modality` tag (CT / MR). Pixel data -> uint8 exactly like the training sources:
  CT  : HU = stored * RescaleSlope + RescaleIntercept, clipped to [-200, 200] HU (PLC-CECT paper and code,
        reports/audit_report.md) and mapped linearly to 0..255; in-plane resampled to 1.0 mm and centred in 512 x 512
        (PLC-CECT volumes are 1.0 mm isotropic, 512 x 512)
  MR  : per-series clip at the 99.5th percentile -> 0..255 (as the LLD-MMRI volumes; no resampling, like the dataset)
Display convention = NIfTI slice.T in LPS: rows go anterior -> posterior, columns patient right -> left. Standard axial
DICOM (ImageOrientationPatient = [1,0,0, 0,1,0]) already is that; flipped axes are flipped back; non-axial is refused.
Privacy: only Modality / geometry / pixel tags are read; no tag value is ever returned, logged or written.
Validated only on DICOM converted from the datasets (tests); scanner DICOM has not been evaluated.

Streaming: files are read one at a time (a .zip member by member, in memory, never extracted to disk) and each is
reduced at once to its display slice (CT) or raw pixel array (MR); the parsed DICOM object is dropped.
Series: one volume per SeriesInstanceUID; series are never merged (several CT phases / MRI sequences stay separate).
Order: along the slice normal (ImagePositionPatient), then InstanceNumber.
Duplicates: a file whose SOPInstanceUID was already seen, or whose position AND pixels equal an earlier slice, is a
true copy and is skipped (counted). Two DIFFERENT images at the same position are refused (mixed acquisitions).
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np

CT_WINDOW = (-200.0, 200.0)
MAX_MEMBERS = 3000
MAX_UNZIPPED = 800 * 2 ** 20
MAX_MEMBER = 64 * 2 ** 20
MAX_RATIO = 200                 # uncompressed / compressed size of one member (zip bomb guard)
SUPPORTED = {"CT": "ct", "MR": "mri"}
CAPTURE = {"SC", "OT"}          # secondary capture / other: usually a screenshot -> treated as a 2D image


class DicomError(Exception):
    """Vietnamese, user-facing."""


@dataclass
class DicomSeries:
    modality_tag: str
    modality: str | None                 # ct | mri | None (capture / unsupported)
    slices: list[np.ndarray] = field(default_factory=list)   # display uint8 slices, inferior -> superior
    is_capture: bool = False             # secondary capture: keep as an RGB/gray 2D picture for image2d
    picture: np.ndarray | None = None
    n_files: int = 0                     # DICOM files of this series in the upload
    duplicates: int = 0                  # true copies skipped (same SOPInstanceUID, or same position and pixels)
    spacing: tuple | None = None         # (row mm, column mm, slice interval mm or None) of the slices given to the AI
    irregular: bool = False              # slice interval not regular (missing slices?)

    @property
    def volume(self) -> np.ndarray:
        """(i, j, k) uint8 like livercancer.data.slices.load_volume_uint8 (display slice = vol[:, :, k].T)."""
        return np.stack([s.T for s in self.slices], axis=2)


def is_dicom(head: bytes) -> bool:
    return len(head) >= 132 and head[128:132] == b"DICM"


def iter_zip(src) -> Iterator[bytes]:
    """DICOM members of a .zip (path or bytes), one at a time. Nothing is extracted to disk, so member names
    (even '../x' or absolute paths) are never used as paths. Limits: members, total / per-member size, compression ratio."""
    try:
        z = zipfile.ZipFile(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
    except (zipfile.BadZipFile, OSError):
        raise DicomError("Tệp .zip bị hỏng hoặc không đọc được.") from None
    with z:
        members = [m for m in z.infolist() if not m.is_dir()]
        if len(members) > MAX_MEMBERS:
            raise DicomError(f"Tệp .zip có quá nhiều tệp (tối đa {MAX_MEMBERS}).")
        if sum(m.file_size for m in members) > MAX_UNZIPPED:
            raise DicomError("Tệp .zip giải nén ra quá lớn.")
        found = 0
        for m in members:
            name = m.filename.replace("\\", "/")
            if m.file_size > MAX_MEMBER or name.startswith("__MACOSX") or name.split("/")[-1].startswith("."):
                continue
            if m.flag_bits & 0x1:
                raise DicomError("Tệp .zip có mật khẩu, không đọc được.")
            if m.file_size > (1 << 20) and m.file_size > MAX_RATIO * max(1, m.compress_size):
                raise DicomError("Tệp .zip có dữ liệu nén bất thường (nghi \"bom giải nén\"), không xử lý.")
            try:
                b = z.read(m)                 # zipfile stops at the declared size and checks the CRC
            except (zipfile.BadZipFile, RuntimeError, OSError, EOFError, ValueError):
                raise DicomError("Tệp .zip bị hỏng hoặc không đọc được.") from None
            if is_dicom(b[:132]):
                found += 1
                yield b
        if not found:
            raise DicomError("Không tìm thấy tệp DICOM nào trong tệp .zip.")


def blobs_from_zip(data: bytes) -> list[bytes]:
    return list(iter_zip(data))


def _pixels(ds) -> np.ndarray:
    try:
        arr = ds.pixel_array
    except Exception:  # noqa: BLE001 - compressed transfer syntax without a decoder, corrupt data
        raise DicomError("Không giải mã được dữ liệu ảnh DICOM (có thể là định dạng nén chưa được hỗ trợ).") from None
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def _orient(img: np.ndarray, ds) -> np.ndarray:
    iop = [float(x) for x in getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]
    r, c = np.array(iop[:3]), np.array(iop[3:])
    n = np.cross(r, c)
    if abs(n[2]) < 0.8:
        raise DicomError("Chỉ hỗ trợ lát cắt ngang (axial); ảnh DICOM này có hướng cắt khác.")
    if r[0] < 0:
        img = img[:, ::-1]
    if c[1] < 0:
        img = img[::-1, :]
    return np.ascontiguousarray(img)


def _ct_slice(ds) -> np.ndarray:
    hu = _pixels(ds).astype(np.float32) * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
    lo, hi = CT_WINDOW
    u8 = np.round((np.clip(hu, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)
    u8 = _orient(u8, ds)
    ps = [float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])]
    if abs(ps[0] - 1) > 0.02 or abs(ps[1] - 1) > 0.02:          # rows / cols spacing -> 1.0 mm like PLC-CECT
        u8 = cv2.resize(u8, (round(u8.shape[1] * ps[1]), round(u8.shape[0] * ps[0])), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((512, 512), np.uint8)                          # centred in 512 x 512 (crop if the FOV is larger)
    h, w = u8.shape
    sy, sx = max(0, (h - 512) // 2), max(0, (w - 512) // 2)
    dy, dx = max(0, (512 - h) // 2), max(0, (512 - w) // 2)
    hh, ww = min(512, h), min(512, w)
    out[dy:dy + hh, dx:dx + ww] = u8[sy:sy + hh, sx:sx + ww]
    return out


def _pixel_hash(ds) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(bytes(ds.PixelData) if "PixelData" in ds else b"")
    return h.hexdigest()


def _capture_picture(ds) -> np.ndarray:
    pic = _pixels(ds)
    if pic.ndim == 3 and pic.shape[-1] == 3:
        return cv2.cvtColor(pic.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if pic.dtype != np.uint8:
        return cv2.normalize(pic.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return pic


def read_series(blobs: Iterable[bytes]) -> list[DicomSeries]:
    """DICOM blobs of ONE upload (list or generator, consumed one at a time) -> series (one per SeriesInstanceUID,
    slices sorted inferior -> superior along the slice normal, true duplicate files skipped)."""
    import pydicom
    groups: dict[str, dict] = {}
    any_dicom = False
    for b in blobs:
        if not is_dicom(b[:132]):
            continue
        any_dicom = True
        try:
            ds = pydicom.dcmread(io.BytesIO(b), force=False)
        except Exception:  # noqa: BLE001
            raise DicomError("Có tệp DICOM bị hỏng, không đọc được.") from None
        del b
        uid = str(getattr(ds, "SeriesInstanceUID", "")) or "single"
        g = groups.setdefault(uid, {"tag": str(getattr(ds, "Modality", "")).upper(), "items": [], "sop": set(),
                                    "n": 0, "dup": 0, "picture": None, "ps": None})
        g["n"] += 1
        tag, mod = g["tag"], SUPPORTED.get(g["tag"])
        if tag in CAPTURE or (mod is None and tag == ""):
            if g["picture"] is None:
                g["picture"] = _capture_picture(ds)
            continue
        if mod is None:
            raise DicomError(f"Ảnh DICOM loại \"{tag}\" không được hỗ trợ (chỉ hỗ trợ CT và MRI).")
        if int(getattr(ds, "NumberOfFrames", 1) or 1) > 1:
            raise DicomError("Ảnh DICOM nhiều khung (multi-frame) chưa được hỗ trợ; hãy xuất từng lát cắt.")
        sop = str(getattr(ds, "SOPInstanceUID", ""))
        if sop and sop in g["sop"]:                       # the very same file twice
            g["dup"] += 1
            continue
        g["sop"].add(sop)
        iop = [float(x) for x in getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]
        n = np.cross(iop[:3], iop[3:])
        n = n if n[2] >= 0 else -n                        # inferior -> superior whatever the in-plane flips
        pos = float(np.dot(n, [float(x) for x in getattr(ds, "ImagePositionPatient", [0, 0, 0])]))
        if g["ps"] is None:
            g["ps"] = [float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])]
        data = _ct_slice(ds) if mod == "ct" else _orient(_pixels(ds).astype(np.float32), ds)
        g["items"].append({"pos": pos, "has_pos": "ImagePositionPatient" in ds, "inst": int(getattr(ds, "InstanceNumber", 0) or 0),
                           "hash": _pixel_hash(ds), "data": data})
        del ds
    if not any_dicom:
        raise DicomError("Tệp không phải DICOM hợp lệ.")
    out = []
    for g in groups.values():
        tag, mod = g["tag"], SUPPORTED.get(g["tag"])
        if g["picture"] is not None:
            out.append(DicomSeries(tag or "?", None, is_capture=True, picture=g["picture"], n_files=g["n"]))
            continue
        items = sorted(g["items"], key=lambda f: (f["pos"], f["inst"]))
        positioned = all(f["has_pos"] for f in items)
        kept, dup, seen = [], g["dup"], set()
        for f in items:
            if positioned and kept and abs(f["pos"] - kept[-1]["pos"]) < 1e-3:
                if f["hash"] == kept[-1]["hash"]:
                    dup += 1                              # same position, same pixels: a copy
                    continue
                raise DicomError("Chuỗi DICOM có hai ảnh khác nhau ở cùng một vị trí lát cắt (có thể trộn nhiều lần chụp "
                                 "trong một chuỗi). Hãy tải riêng từng chuỗi.")
            if not positioned and f["hash"] in seen:      # no position: only identical pixels count as a copy
                dup += 1
                continue
            seen.add(f["hash"])
            kept.append(f)
        if mod == "ct":
            slices = [f["data"] for f in kept]
            ps = (1.0, 1.0)                               # resampled in-plane to 1.0 mm like PLC-CECT
        else:
            raw = [f["data"] for f in kept]
            if len({r.shape for r in raw}) > 1:
                raise DicomError("Các lát cắt trong chuỗi MRI có kích thước khác nhau.")
            stack = np.stack(raw)
            p995 = float(np.percentile(stack, 99.5)) or 1.0
            slices = list(np.clip(np.round(stack / p995 * 255), 0, 255).astype(np.uint8))
            ps = tuple(g["ps"] or (1.0, 1.0))
        if len({s.shape for s in slices}) > 1:
            raise DicomError("Các lát cắt trong chuỗi DICOM có kích thước khác nhau.")
        gaps = np.diff([f["pos"] for f in kept]) if positioned and len(kept) > 1 else np.array([])
        dz = float(np.median(gaps)) if gaps.size else None
        irregular = bool(gaps.size and dz and (gaps.max() > 1.5 * dz or gaps.min() < 0.5 * dz))
        out.append(DicomSeries(tag, mod, slices, n_files=g["n"], duplicates=dup,
                               spacing=(round(ps[0], 4), round(ps[1], 4), round(dz, 4) if dz else None), irregular=irregular))
    return out
