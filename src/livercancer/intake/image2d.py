"""2D image intake (PNG / JPG / screenshot / photo of a screen) -> one grayscale slice framed like the training data.

The models were trained on full axial slices: body centred, black background around it, no text, uint8 gray.
A screenshot of a hospital viewer adds UI (toolbars, panels), burned-in text (often the patient's name / ID) and a
different scale. This module, without any learned component:
  1. finds the slice: the body is the largest bright region ENCLOSED by the dark background of the viewport
     (connected components of the dark mask; the body is a "hole" of it). If the body touches the image borders
     (a slice that is already tight, e.g. many MRI exports) and the image is grayscale, the whole image is the slice;
  2. blanks everything outside the viewport (UI) and small bright components outside the body (burned-in text:
     identifiers are removed from what is analysed and exported);
  3. crops a square around the body with the median framing of the training slices (body side / image side,
     measured per modality on the training data: FRAMING) and resizes to 512 x 512;
  4. checks quality (too small, blurred, glare, colour cast, body cut by the border) -> Vietnamese messages.
The intensities are NOT re-mapped (no histogram equalisation): only resizing and blanking, so lesion contrast is kept.
Screenshots remain a different domain than the NIfTI data the models were validated on; results are flagged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

OUT_SIZE = 512
# median (body bbox longest side / slice side) on training slices, measured by scripts/87_modality_classifier.py
# (stage `framing`); used to re-frame screenshots like the training images. Generic value before the modality is known.
FRAMING = {"generic": 0.80, "ct": 0.763, "mri": 0.839}
DARK = 35            # gray level below which a pixel counts as background
SAT_GRAY = 40        # saturation below which a pixel counts as grayscale


@dataclass
class Prepared:
    ok: bool
    gray: np.ndarray | None = None              # 512 x 512 uint8, ready for the models
    error_vi: str = ""
    crop: tuple[int, int, int, int] | None = None   # body-centred square in original coordinates (x0, y0, x1, y1)
    body: tuple[int, int, int, int] | None = None   # body bounding box in original coordinates
    steps_vi: list[str] = field(default_factory=list)
    warnings_vi: list[str] = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    n_panels: int = 1
    screenshot_like: bool = False


def to_gray(img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """(gray uint8, saturation uint8 or None)."""
    if img.ndim == 2:
        g = img
        sat = None
    else:
        bgr = img[:, :, :3]
        if img.shape[2] == 4:                                   # transparent PNG: composite on black
            a = img[:, :, 3:4].astype(np.float32) / 255
            bgr = (bgr.astype(np.float32) * a).astype(np.uint8)
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    if g.dtype != np.uint8:
        g = cv2.normalize(g.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return g, sat


def _masks(gray: np.ndarray, sat: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, int]:
    """(grayscale-like mask, dark-background mask, dark threshold). Saturation is meaningless on dark pixels
    (noise gives them random hue), so dark pixels always count as grayscale. The dark threshold adapts to the
    background level (photos of a screen have a lifted, noisy black)."""
    thr = int(np.clip(np.percentile(gray, 5) + 30, DARK, 90))
    grayish = np.ones_like(gray, bool) if sat is None else (sat < SAT_GRAY) | (gray < 70)
    dark = (gray < thr) & grayish
    return grayish, dark, thr


def flat_map(gray: np.ndarray) -> np.ndarray:
    """Pixels whose 5x5 neighbourhood is flat (std < 1.5): typical of viewer UI, rare in body tissue (noise, texture)."""
    g = gray.astype(np.float32)
    m = cv2.boxFilter(g, -1, (5, 5))
    v = cv2.boxFilter(g * g, -1, (5, 5)) - m * m
    return v < 1.5 ** 2


def _components(mask: np.ndarray):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return n, lab, stats


def find_body(gray: np.ndarray, sat: np.ndarray | None) -> dict:
    """Body = largest non-dark, grayscale region enclosed by the dark background (not touching the image border)."""
    h, w = gray.shape
    _, dark, thr = _masks(gray, sat)
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    not_dark = ~dark
    n, lab, st = _components(not_dark)
    flat = flat_map(gray)
    holes = []
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        whole_screen = bw > 0.90 * w and bh > 0.90 * h        # a photographed screen inside its dark bezel, not a body
        if not touches and not whole_screen and area >= 0.01 * h * w:
            if float(flat[y:y + bh, x:x + bw][lab[y:y + bh, x:x + bw] == i].mean()) > 0.6:
                continue                                          # flat UI panel / window, not tissue
            holes.append((area, i, (x, y, x + bw, y + bh)))
    holes.sort(reverse=True)

    def inside(b, c):
        return b[0] >= c[0] and b[1] >= c[1] and b[2] <= c[2] and b[3] <= c[3]
    # a region that ENCLOSES other large bright regions is a container (a screen / window around the viewport), not a body
    holes = [hh for hh in holes if sum(o[0] for o in holes if o is not hh and inside(o[2], hh[2])) < 0.10 * hh[0]]
    out = {"dark_frac": float(dark.mean()), "holes": holes, "lab": lab, "dark_thr": thr, "flat_frac": float((flat & ~dark).mean())}   # flat AND not background: viewer UI
    if holes:
        area, i, box = holes[0]
        comp = lab == i
        med_sat = float(np.median(sat[comp])) if sat is not None else 0.0
        out.update(body=box, body_mask=comp, body_area=int(area), body_sat=med_sat)
    return out


def _viewport_mask(gray: np.ndarray, sat: np.ndarray | None, body_mask: np.ndarray) -> np.ndarray:
    """Dark background connected to the body (the viewer's image area) + the body itself; everything else is UI."""
    dark = _masks(gray, sat)[1].astype(np.uint8)
    ring = cv2.dilate(body_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    n, lab, _ = _components(dark)
    lut = np.zeros(n, bool)
    lut[np.unique(lab[ring & (dark > 0)])] = True
    lut[0] = False
    keep = lut[lab]
    filled = (keep | body_mask).astype(np.uint8)
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vp = np.zeros_like(filled)
    cv2.drawContours(vp, cnts, -1, 1, thickness=-1)           # holes inside the viewport (text, table) stay inside
    return vp > 0


def prepare_2d(img: np.ndarray, modality: str = "generic") -> Prepared:
    if img is None or img.size == 0:
        return Prepared(False, error_vi="Không đọc được ảnh.")
    gray, sat = to_gray(img)
    h, w = gray.shape
    if min(h, w) < 96:
        return Prepared(False, error_vi="Ảnh quá nhỏ để phân tích (cần ít nhất khoảng 100 điểm ảnh mỗi chiều).")
    p = Prepared(True)
    color = sat is not None and float(np.median(sat)) > SAT_GRAY
    fb = find_body(gray, sat)
    if "body" in fb:
        x0, y0, x1, y1 = fb["body"]
        body_mask = fb["body_mask"]
        vp = _viewport_mask(gray, sat, body_mask)
        outside_vp = ~vp
        dark = _masks(gray, sat)[1]
        # a viewer UI is bright: exported slices have <= 2.2% bright pixels outside the viewport (CT table, an arm on the
        # border), synthetic screenshots >= 6.2% (Val, scratch measurement). Clean slices keep every bright pixel.
        screenshot = bool(outside_vp.mean() > 0.03 and (outside_vp & ~dark).mean() > 0.03)
        work = gray.copy()
        work[outside_vp if screenshot else (outside_vp & dark)] = 0   # viewer UI removed / background set to black
        # burned-in text / small marks outside the body (inside the viewport): blanked
        bright = (work >= fb["dark_thr"]) & ~(cv2.dilate(body_mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
        n, lab, st = _components(bright)
        small = st[:, 4] < 0.004 * h * w
        small[0] = False
        work[small[lab]] = 0                                   # one look-up, not one pass per component
        n_text = int(((st[:, 4] >= 20) & small).sum())          # visible marks only (noise specks are not "text")
        p.screenshot_like = screenshot
        p.steps_vi.append("Tìm thấy vùng cơ thể trên nền tối của lát cắt.")
        if screenshot:
            p.steps_vi.append("Đã loại bỏ phần giao diện phần mềm nằm ngoài khung ảnh y tế.")
        if n_text:
            p.steps_vi.append(f"Đã che {n_text} ký tự / dấu nhỏ nằm ngoài vùng cơ thể (có thể là chữ hiển thị thông tin).")
        big = [hh for hh in fb["holes"] if hh[0] >= 0.4 * fb["holes"][0][0]]
        if len(big) >= 2 and screenshot:
            p.n_panels = len(big)
            p.warnings_vi.append(f"Ảnh có {len(big)} khung lát cắt; hệ thống phân tích khung lớn nhất.")
        side = max(x1 - x0, y1 - y0) / FRAMING.get(modality, FRAMING["generic"])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        p.body = (int(x0), int(y0), int(x1), int(y1))
    else:
        if not 0.8 <= w / h <= 1.25:   # a screen-shaped image whose slice could not be isolated: never analysed as a whole
            return Prepared(False, error_vi=("Không tách được lát cắt khỏi ảnh chụp màn hình. Hãy cắt ảnh sát quanh lát cắt "
                                             "(bỏ thanh công cụ, chữ) rồi tải lại, hoặc dùng file NIfTI / DICOM gốc."))
        if color or fb["dark_frac"] < 0.05 or fb["flat_frac"] > 0.35:
            return Prepared(False, error_vi=("Không tìm thấy lát cắt CT/MRI trong ảnh (cần ảnh xám có vùng cơ thể trên nền tối). "
                                             "Hãy dùng ảnh lát cắt rõ ràng hoặc file NIfTI / DICOM gốc."))
        work = gray.copy()
        side, cx, cy = max(h, w), w / 2, h / 2
        p.steps_vi.append("Ảnh là lát cắt đầy khung; dùng toàn bộ ảnh.")
    side = int(round(side))
    ox, oy = int(round(cx - side / 2)), int(round(cy - side / 2))
    p.crop = (ox, oy, ox + side, oy + side)
    canvas = np.zeros((side, side), np.uint8)                   # outside the original image = black background
    sx0, sy0, sx1, sy1 = max(0, ox), max(0, oy), min(w, ox + side), min(h, oy + side)
    canvas[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox] = work[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_AREA if side > OUT_SIZE else cv2.INTER_CUBIC
    p.gray = cv2.resize(canvas, (OUT_SIZE, OUT_SIZE), interpolation=interp)
    if side != OUT_SIZE:
        p.steps_vi.append(f"Căn khung quanh cơ thể và chuẩn hóa kích thước về {OUT_SIZE}×{OUT_SIZE} (không chỉnh độ sáng, độ tương phản).")
    # ---- quality
    if p.body:
        bw = p.body[2] - p.body[0]
        bh = p.body[3] - p.body[1]
        p.quality["body_px"] = int(max(bw, bh))
        if max(bw, bh) < 160:
            p.warnings_vi.append("Vùng ảnh y tế nhỏ (độ phân giải thấp); có thể bỏ sót tổn thương nhỏ.")
        m = cv2.resize(fb["body_mask"].astype(np.uint8), None, fx=1, fy=1) > 0
        body_px = gray[m]
        glare = float((body_px >= 250).mean()) if body_px.size else 0.0
        p.quality["glare_frac"] = round(glare, 4)
        if glare > 0.12:
            p.warnings_vi.append("Ảnh có vùng quá sáng (chói hoặc phản chiếu), kết quả có thể sai lệch.")
        if fb.get("body_sat", 0) > 25:
            p.warnings_vi.append("Ảnh có màu hoặc ánh sáng lệch; đã chuyển sang ảnh xám để phân tích.")
        edge = sum([p.body[0] <= 2, p.body[1] <= 2, p.body[2] >= w - 2, p.body[3] >= h - 2])
        if edge:
            p.warnings_vi.append("Vùng cơ thể chạm mép ảnh; có thể bị mất góc.")
    std = float(p.gray.std())
    if std < 3:
        return Prepared(False, error_vi="Ảnh gần như đồng màu, không có nội dung để phân tích.")
    region = work if p.body is None else work[p.body[1]:p.body[3], p.body[0]:p.body[2]]
    lap = float(cv2.Laplacian(region, cv2.CV_64F).var())      # native resolution: upscaling must not look like blur
    p.quality["sharpness"] = round(lap, 1)
    if lap < 15 and std > 20:
        p.warnings_vi.append("Ảnh có vẻ bị mờ; kết quả có thể kém tin cậy.")
    if color:
        p.warnings_vi.append("Ảnh màu đã được chuyển sang ảnh xám.")
    return p
