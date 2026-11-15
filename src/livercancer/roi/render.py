"""Drawing helpers shared by the ROI report scripts and the website (PIL; Vietnamese text needs a Unicode TTF font)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PRED = (220, 38, 38)        # red: YOLO lesion box
GT = (22, 163, 74)          # green: ground truth (Validation only)
LOCAL = (37, 99, 235)       # blue: local zoom window
CONTEXT = (14, 165, 233)    # light blue: context zoom window


@lru_cache(maxsize=8)
def font(size: int = 14) -> ImageFont.ImageFont:
    import matplotlib
    for p in (Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf", Path("C:/Windows/Fonts/arial.ttf")):
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def to_rgb(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(gray)).convert("RGB")


def draw(gray: np.ndarray, pred: list[tuple] = (), gt: list[tuple] = (), windows: list[tuple] = (),
         labels: list[str] | None = None, scale: int = 1, width: int | None = None, font_size: int | None = None) -> np.ndarray:
    """RGB image with GT boxes (green), predicted boxes (red, with labels) and zoom windows (blue).
    Boxes are (x0, y0, x1, y1) px; windows are (x0, y0, x1, y1, kind) with kind 'local' | 'context'."""
    im = to_rgb(gray)
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(im)
    s = scale
    width = width or max(2, round(im.height / 220))                 # thicker lines on larger images
    font_size = font_size or max(12, round(im.height * 0.035))    # ~18 px on a 512 px slice: readable when shown smaller
    for b in gt:
        d.rectangle([b[0] * s, b[1] * s, b[2] * s, b[3] * s], outline=GT, width=width)
    for w in windows:
        c = LOCAL if w[4] == "local" else CONTEXT
        d.rectangle([w[0] * s, w[1] * s, w[2] * s, w[3] * s], outline=c, width=max(1, width - 1))
    f = font(font_size)
    for i, b in enumerate(pred):
        d.rectangle([b[0] * s, b[1] * s, b[2] * s, b[3] * s], outline=PRED, width=width)
        if labels:
            t = labels[i]
            tx, ty = b[0] * s, max(0, b[1] * s - f.size - 4)
            bb = d.textbbox((tx, ty), t, font=f)
            d.rectangle([bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1], fill=PRED)
            d.text((tx, ty), t, fill=(255, 255, 255), font=f)
    return np.asarray(im)


def panel(images: list[np.ndarray], titles: list[str], height: int = 300, title_size: int = 16, pad: int = 8) -> np.ndarray:
    """Horizontal strip of images (each resized to `height`) with a title above each (white background)."""
    f = font(title_size)
    tiles = []
    for img, t in zip(images, titles):
        im = Image.fromarray(img).convert("RGB") if img.ndim == 3 else to_rgb(img)
        im = im.resize((max(1, round(im.width * height / im.height)), height), Image.BILINEAR)
        tile = Image.new("RGB", (im.width, height + title_size + 2 * pad), "white")
        ImageDraw.Draw(tile).text((2, 2), t, fill=(17, 24, 39), font=f)
        tile.paste(im, (0, title_size + 2 * pad))
        tiles.append(tile)
    out = Image.new("RGB", (sum(t.width for t in tiles) + pad * (len(tiles) - 1), tiles[0].height), "white")
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width + pad
    return np.asarray(out)
