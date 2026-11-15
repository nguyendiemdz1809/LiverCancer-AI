"""Synthetic screenshots / photos of a screen built from real dataset slices.

ONLY for training and evaluating the CT/MRI recognition model and for testing the screenshot pre-processing
(scripts/87_modality_classifier.py, tests). They are labelled synthetic everywhere, never shown as patient data and
never used to score the locked AI pipeline. The fake viewer text is random (no real identifiers).
"""
from __future__ import annotations

import random
import string

import cv2
import numpy as np

UI_THEMES = [((245, 245, 245), (225, 225, 225)), ((30, 30, 30), (55, 55, 55)), ((255, 255, 255), (238, 240, 244)),
             ((40, 44, 52), (60, 64, 72)), ((210, 214, 220), (190, 194, 200))]
ACCENTS = [(214, 120, 30), (60, 150, 40), (40, 40, 200), (150, 90, 20), (90, 90, 90)]


def _rand_text(rng: random.Random, n=10) -> str:
    return "".join(rng.choice(string.ascii_uppercase + string.digits + " :/.-") for _ in range(n))


def _put(img, rng, xy, scale, color):
    cv2.putText(img, _rand_text(rng, rng.randint(6, 18)), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_screenshot(slices: list[np.ndarray], seed: int = 0, panels: int | None = None) -> tuple[np.ndarray, dict]:
    """slices: grayscale uint8 slices (first one = main panel). Returns (BGR screenshot, info with the main viewport box)."""
    rng = random.Random(seed)
    W, H = rng.choice([(1280, 800), (1366, 768), (1600, 900), (1920, 1080), (1024, 768)])
    bg, panel = rng.choice(UI_THEMES)
    img = np.full((H, W, 3), bg, np.uint8)
    top = rng.randint(30, 90)
    cv2.rectangle(img, (0, 0), (W, top), rng.choice(ACCENTS + [panel]), -1)                 # toolbar
    for i in range(rng.randint(3, 8)):
        x = 10 + i * rng.randint(60, 120)
        cv2.rectangle(img, (x, 8), (x + 40, top - 8), panel, -1)
    side = rng.randint(0, 1) * rng.randint(160, 300)                                       # side panel
    if side:
        cv2.rectangle(img, (0, top), (side, H), panel, -1)
        for y in range(top + 20, H - 20, rng.randint(24, 40)):
            _put(img, rng, (10, y), 0.45, (20, 20, 20) if sum(bg) > 300 else (220, 220, 220))
    n = panels or rng.choice([1, 1, 1, 1, 2, 4])
    area = (side + 10, top + 10, W - 10, H - 10)
    aw, ah = area[2] - area[0], area[3] - area[1]
    cols = 1 if n == 1 else 2
    rows = 1 if n <= 2 else 2
    cw, ch = aw // cols, ah // rows
    main = None
    for pi in range(n):
        s = slices[pi % len(slices)]
        r, c = divmod(pi, cols)
        vs = int(min(cw, ch) * (rng.uniform(0.7, 0.98) if n == 1 else 0.96))
        vx = area[0] + c * cw + (cw - vs) // 2
        vy = area[1] + r * ch + (ch - vs) // 2
        if pi == 0:
            vs = int(vs * (1.0 if n == 1 else 1.0))
        view = cv2.resize(s, (vs, vs), interpolation=cv2.INTER_AREA if vs < s.shape[0] else cv2.INTER_CUBIC)
        img[vy:vy + vs, vx:vx + vs] = view[:, :, None]
        for (tx, ty) in ((vx + 6, vy + 18), (vx + 6, vy + 36), (vx + vs - 150, vy + 18), (vx + 6, vy + vs - 10)):
            if rng.random() < 0.8:
                _put(img, rng, (tx, ty), 0.42, (235, 235, 235))                              # burned-in viewer text
        if pi == 0:
            main = (vx, vy, vx + vs, vy + vs)
    info = {"viewport": main, "panels": n, "size": (W, H)}
    return img, info


def photo_of_screen(img: np.ndarray, seed: int = 0) -> np.ndarray:
    """Phone photo of a monitor: slight perspective, blur, brightness gradient, a glare spot, colour cast, noise."""
    rng = random.Random(seed)
    h, w = img.shape[:2]
    d = 0.04
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[rng.uniform(0, d) * w, rng.uniform(0, d) * h], [w - rng.uniform(0, d) * w, rng.uniform(0, d) * h],
                      [w - rng.uniform(0, d) * w, h - rng.uniform(0, d) * h], [rng.uniform(0, d) * w, h - rng.uniform(0, d) * h]])
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(20, 20, 20))
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.6, 1.6))
    gx = np.linspace(rng.uniform(0.85, 1.0), rng.uniform(1.0, 1.12), w, dtype=np.float32)[None, :, None]
    cast = np.array([rng.uniform(0.92, 1.08) for _ in range(3)], np.float32)[None, None, :]
    out = out.astype(np.float32) * gx * cast
    gxy = (rng.randint(0, w), rng.randint(0, h))
    yy, xx = np.mgrid[0:h, 0:w]
    out += 60 * np.exp(-(((xx - gxy[0]) ** 2 + (yy - gxy[1]) ** 2) / (2 * (0.08 * w) ** 2)))[:, :, None] * rng.uniform(0.2, 1)
    out += np.random.default_rng(seed).normal(0, rng.uniform(2, 6), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg(img: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def degrade(gray: np.ndarray, seed: int) -> np.ndarray:
    """Light photometric / compression changes of a plain exported slice (JPG / PNG from another viewer)."""
    rng = random.Random(seed)
    g = gray
    if rng.random() < 0.5:
        s = rng.uniform(0.35, 1.0)
        g = cv2.resize(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), g.shape[::-1], interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        gamma = rng.uniform(0.75, 1.35)
        g = np.clip(255 * (g / 255.0) ** gamma, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.5, 1.2))
    return jpeg(g, rng.randint(35, 95))
