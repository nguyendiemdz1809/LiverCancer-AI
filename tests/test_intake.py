"""Website intake (livercancer.intake): screenshot pre-processing, DICOM reading. Uses Validation slices only."""
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CT_DS = ROOT / "data" / "processed" / "full" / "yolo_ct"
MRI_DS = ROOT / "data" / "processed" / "pilot" / "yolo_mri"


def _val_slices(ds: Path, n: int, seed=0):
    if not (ds / "slices.csv").exists():
        pytest.skip("exported dataset not available")
    with open(ds / "slices.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    random.Random(seed).shuffle(rows)
    return [cv2.imread(str(ds / "images" / "val" / f"{r['file']}.png"), cv2.IMREAD_GRAYSCALE) for r in rows[:n]]


def test_plain_slice_kept_as_is():
    from livercancer.intake.image2d import prepare_2d
    for g in _val_slices(CT_DS, 5):
        p = prepare_2d(g, "ct")
        assert p.ok and p.gray.shape == (512, 512) and not p.screenshot_like and p.body


@pytest.mark.parametrize("ds,mod", [(CT_DS, "ct"), (MRI_DS, "mri")])
def test_screenshot_slice_is_isolated_and_ui_removed(ds, mod):
    from livercancer.intake.image2d import prepare_2d
    from livercancer.intake.synthetic import jpeg, render_screenshot
    ok = 0
    for i, g in enumerate(_val_slices(ds, 12, 1)):
        shot, info = render_screenshot([g], seed=100 + i, panels=1)
        p = prepare_2d(jpeg(shot, 75), mod)
        if not p.ok:
            continue                                   # refusing is acceptable; mis-cropping is not
        vx0, vy0, vx1, vy1 = info["viewport"]
        bx0, by0, bx1, by1 = p.body
        assert bx0 >= vx0 - 4 and by0 >= vy0 - 4 and bx1 <= vx1 + 4 and by1 <= vy1 + 4, "body outside the viewer panel"
        assert p.screenshot_like and p.gray[:8, :8].max() < 60, "viewer UI or text left in the analysed image"
        ok += 1
    assert ok >= 10


def test_non_medical_and_unusable_images_are_refused():
    from livercancer.intake.image2d import prepare_2d
    rng = np.random.default_rng(0)
    colour = rng.integers(0, 255, (400, 600, 3), dtype=np.uint8)               # colour noise photo
    document = np.full((800, 600, 3), 250, np.uint8)
    cv2.putText(document, "BAO CAO", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    flat = np.full((512, 512), 128, np.uint8)
    for img in (colour, document, flat, np.zeros((40, 40), np.uint8)):
        p = prepare_2d(img)
        assert not p.ok and p.error_vi


def test_dicom_ct_and_mri_identical_to_nifti():
    from livercancer.config import load_paths
    from livercancer.data.slices import load_volume_uint8
    from livercancer.data.sources import image_path
    from livercancer.intake import dicom_io
    from livercancer.intake.dicom_fixtures import series_from_volume, zip_blobs
    paths = load_paths()
    f = image_path(paths, "ct", "P0267", "C2")
    if not f.exists():
        pytest.skip("source dataset not available")
    ct = load_volume_uint8(f)[:, :, 30:40]
    s = dicom_io.read_series(dicom_io.blobs_from_zip(zip_blobs(series_from_volume(ct, "CT", "1.2.840.1"))))
    assert len(s) == 1 and s[0].modality == "ct" and np.array_equal(s[0].volume, ct)
    s = dicom_io.read_series(series_from_volume(ct, "CT", "1.2.840.2", flip_rows=True))
    assert np.array_equal(s[0].volume, ct)                                     # flipped rows / normal handled
    mr = load_volume_uint8(image_path(paths, "mri", "MR-400851", "T2WI"))
    s = dicom_io.read_series(series_from_volume(mr, "MR", "1.2.840.3"))
    assert s[0].modality == "mri" and np.abs(s[0].volume.astype(int) - mr.astype(int)).max() <= 1


def test_dicom_guards():
    import pydicom
    from livercancer.intake import dicom_io
    from livercancer.intake.dicom_fixtures import _ds
    px = np.zeros((64, 64), np.int16)
    sag = _ds(px, "CT", 0, "1.2.9", iop=(0, 1, 0, 0, 0, -1))                    # sagittal
    with pytest.raises(dicom_io.DicomError):
        dicom_io.read_series([sag])
    us = pydicom.dcmread(__import__("io").BytesIO(_ds(px, "CT", 0, "1.2.10")))
    us.Modality = "US"
    buf = __import__("io").BytesIO()
    us.save_as(buf, enforce_file_format=True)
    with pytest.raises(dicom_io.DicomError):
        dicom_io.read_series([buf.getvalue()])
    with pytest.raises(dicom_io.DicomError):
        dicom_io.blobs_from_zip(b"PK\x03\x04 not a zip")


def test_display_zoom_is_faithful():
    """Display Tumor Zoom: window contains the box and stays in the image, enlargement <= 4x, pixels come from the
    original slice (reducing the zoom back to the window size gives the original crop), classifier crop unchanged."""
    from livercancer.roi.crop import zoom as cls_zoom
    from livercancer.roi.display import MAX_UPSCALE, display_zoom
    img = _val_slices(CT_DS, 1, seed=5)[0]
    for box in [(250.0, 240.0, 260.0, 251.0), (100.0, 120.0, 220.0, 230.0), (0.0, 3.0, 14.0, 15.0), (500.0, 500.0, 511.0, 511.0)]:
        for kind in ("local", "context"):
            z = display_zoom(img, box, kind, marks=False)
            x0, y0, x1, y1 = z["window"]
            assert x0 <= box[0] and y0 <= box[1] and x1 >= box[2] and y1 >= box[3]
            assert 0 <= x0 and 0 <= y0 and x1 <= img.shape[1] and y1 <= img.shape[0]      # shifted inside, no black padding
            assert z["scale"] <= MAX_UPSCALE + 1e-6
            back = cv2.resize(z["image"][:, :, 0], (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(int)
            assert np.abs(back - img[y0:y1, x0:x1].astype(int)).mean() < 3
    c = cls_zoom(img, (250.0, 240.0, 260.0, 251.0))                  # the classifier's crop is still 224 px
    assert c["local"].shape == (224, 224)


def test_dicom_duplicates_order_series_and_conflicts():
    """True copies are skipped, order comes from the slice position, series stay separate, conflicts are refused."""
    import random
    import zipfile
    from livercancer.config import load_paths
    from livercancer.data.slices import load_volume_uint8
    from livercancer.data.sources import image_path
    from livercancer.intake import dicom_io
    from livercancer.intake.dicom_fixtures import series_from_volume, zip_blobs
    f = image_path(load_paths(), "ct", "P0267", "C2")
    if not f.exists():
        pytest.skip("source dataset not available")
    ct = load_volume_uint8(f)[:, :, 30:36]
    blobs = series_from_volume(ct, "CT", "1.2.840.7")
    s = dicom_io.read_series(dicom_io.iter_zip(zip_blobs(blobs + blobs[:2])))          # the same 2 files twice
    assert len(s) == 1 and s[0].duplicates == 2 and s[0].n_files == 8 and np.array_equal(s[0].volume, ct)
    assert s[0].spacing == (1.0, 1.0, 1.0) and not s[0].irregular
    sh = blobs[:]
    random.Random(1).shuffle(sh)
    assert np.array_equal(dicom_io.read_series(sh)[0].volume, ct)                      # file order does not matter
    s = dicom_io.read_series(blobs + series_from_volume(ct[:, :, :3], "CT", "1.2.840.8"))
    assert sorted(len(x.slices) for x in s) == [3, 6]                                   # two series, never merged
    again = series_from_volume(ct[:, :, :1], "CT", "1.2.840.7")                         # re-export: new UID, same slice
    s = dicom_io.read_series(blobs + again)
    assert s[0].duplicates == 1 and np.array_equal(s[0].volume, ct)
    other = series_from_volume(ct[:, :, ::-1][:, :, :1], "CT", "1.2.840.7")             # other pixels, same position
    with pytest.raises(dicom_io.DicomError):
        dicom_io.read_series(blobs + other)
    gap = [b for i, b in enumerate(blobs) if i != 3]                                    # a missing slice
    assert dicom_io.read_series(gap)[0].irregular


def test_zip_guards():
    import io
    import zipfile
    from livercancer.intake import dicom_io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("../../evil.dcm", b"\0" * (6 << 20))                                 # highly compressible + traversal name
    with pytest.raises(dicom_io.DicomError, match="bom"):
        list(dicom_io.iter_zip(buf.getvalue()))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", b"hello")
    with pytest.raises(dicom_io.DicomError, match="Không tìm thấy"):
        list(dicom_io.iter_zip(buf.getvalue()))
