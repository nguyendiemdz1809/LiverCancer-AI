"""DICOM files built from dataset volumes, ONLY to test the DICOM intake (tests, scripts/98_website_e2e.py).

CT : uint8 (= [-200, 200] HU window of the source) -> HU integers (inverse of the window) stored as int16 with
     RescaleSlope 1 / Intercept 0. Reading them back with livercancer.intake.dicom_io gives the same uint8 values.
MR : uint8 x 10 stored as uint16 (dicom_io re-normalises with the 99.5th percentile).
Patient tags hold an obviously fake name / ID so tests can check that no metadata ever reaches the API response.
These are converted copies for testing, never presented as scanner DICOM.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np

FAKE_NAME = "TEST^KHONG^HIEN^THI"
FAKE_ID = "ID-KHONG-HIEN-THI-123"


def _ds(pixels: np.ndarray, modality: str, k: int, series_uid: str, iop=(1, 0, 0, 0, 1, 0), spacing=(1.0, 1.0)):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2" if modality == "CT" else "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.PatientName, ds.PatientID = modality, FAKE_NAME, FAKE_ID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = series_uid + ".1", series_uid
    ds.InstanceNumber = k + 1
    ds.ImageOrientationPatient = [float(x) for x in iop]
    ds.ImagePositionPatient = [0.0, 0.0, float(k)]
    ds.PixelSpacing = [float(spacing[0]), float(spacing[1])]
    ds.SliceThickness = 1.0
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation = 1 if pixels.dtype == np.int16 else 0
    if modality == "CT":
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
    ds.PixelData = pixels.tobytes()
    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, enforce_file_format=True)
    return buf.getvalue()


def series_from_volume(vol: np.ndarray, modality: str, series_uid: str, flip_rows=False) -> list[bytes]:
    """(i, j, k) uint8 volume (training convention) -> one DICOM file per slice (display slice = vol[:, :, k].T)."""
    out = []
    for k in range(vol.shape[2]):
        disp = np.ascontiguousarray(vol[:, :, k].T)
        if modality == "CT":
            px = np.round(disp.astype(np.float32) / 255 * 400 - 200).astype(np.int16)
        else:
            px = (disp.astype(np.uint16) * 10).astype(np.uint16)
        iop = (1, 0, 0, 0, 1, 0)
        if flip_rows:                                      # rows stored bottom-up: reader must flip them back
            px, iop = px[::-1].copy(), (1, 0, 0, 0, -1, 0)
        out.append(_ds(px, modality, k, series_uid, iop))
    return out


def zip_blobs(blobs: list[bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, b in enumerate(blobs):
            z.writestr(f"series/IM{i:04d}", b)
    return buf.getvalue()
