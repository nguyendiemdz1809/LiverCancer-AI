"""Real-browser check (Edge, Playwright) of the display additions requested by the supervisor (2026-11-08):
  1. "Kiến trúc mô hình" on /mo-hinh: numbers equal web/src/data/model_arch.json (exported from the checkpoints)
  2. sigmoid chart in the result panel: one dot per region, each dot exactly on the curve sigma(z)
  3. "Rải màu" (pseudo-colour): the displayed pixels equal the colour table applied to the gray slice; back to gray = original
  4. "Kính lúp" (magnifier): the lens centre shows the image point under the pointer (also when the viewer is zoomed)
Display only: the analysis is the normal locked pipeline (Validation patient P0267, never Test).
Prerequisite: backend serving the built frontend on http://127.0.0.1:8000. Screenshots -> outputs/web_e2e_tools/ (local).
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import load_paths  # noqa: E402
from livercancer.data.sources import image_path  # noqa: E402

OUT = ROOT / "outputs" / "web_e2e_tools"
VAL = ROOT / "outputs" / "predictions" / "pipeline_validation_gate"
ARCH = json.loads((ROOT / "web" / "src" / "data" / "model_arch.json").read_text(encoding="utf-8"))
JET = [  # same 17-sample table as web/src/components/colormaps.tsx
    [0, 0, 0, 0, 0, 0, 0.085, 0.288, 0.49, 0.693, 0.895, 1, 1, 1, 1, 0.767, 0.5],
    [0, 0, 0.002, 0.253, 0.504, 0.755, 1, 1, 1, 1, 1, 0.814, 0.582, 0.349, 0.117, 0, 0],
    [0.5, 0.785, 1, 1, 1, 1, 0.882, 0.68, 0.478, 0.275, 0.073, 0, 0, 0, 0, 0, 0]]


def shot(loc) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(loc.screenshot())).convert("RGB")).astype(np.int16)


def main() -> int:
    from playwright.sync_api import expect, sync_playwright
    OUT.mkdir(parents=True, exist_ok=True)
    paths = load_paths()
    series = [v["series"] for v in json.loads((VAL / "ct" / "cases" / "P0267.json").read_text(encoding="utf-8"))["volumes"]]
    files = [str(image_path(paths, "ct", "P0267", s)) for s in series]
    errors, rep = [], {}
    with sync_playwright() as p:
        br = p.chromium.launch(channel="msedge", headless=True)
        ctx = br.new_context(viewport={"width": 1440, "height": 900}, locale="vi-VN", device_scale_factor=1)
        page = ctx.new_page()
        page.on("console", lambda m: errors.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

        # 1. architecture section
        page.goto("http://127.0.0.1:8000/mo-hinh")
        arch = page.locator("section.arch")
        expect(arch).to_be_visible(timeout=20000)
        y = ARCH["yolo"]["ct"]
        vals = arch.locator(".stat .value").all_inner_texts()
        page.locator("section.arch details summary").click()
        rows = arch.locator("table.arch-table tbody tr").count()
        rep["architecture"] = {"stats": vals, "table_rows": rows, "expected_blocks": y["n_blocks"]}
        if vals[:2] != [str(y["n_blocks"]), str(y["conv2d"])]:
            errors.append(f"architecture stats {vals[:2]} != {[y['n_blocks'], y['conv2d']]}")
        if rows != y["n_blocks"] + 1:
            errors.append(f"architecture table rows {rows} != {y['n_blocks'] + 1}")
        arch.screenshot(path=str(OUT / "kien_truc_mo_hinh.png"))

        # analysis of one Validation case (normal locked pipeline)
        page.goto("http://127.0.0.1:8000/phan-tich")
        page.get_by_role("radio", name="Phân tích ca 3D").click()
        page.get_by_role("radio", name="Phân tích CT").click()
        t0 = time.time()
        page.set_input_files("input[type=file][accept]", files)
        expect(page.locator(".concl-title")).to_be_visible(timeout=600000)
        rep["analysis_s"] = round(time.time() - t0, 1)
        rep["result"] = page.locator(".concl-title").inner_text()

        # 2. sigmoid chart: dots on the curve, one per region
        sg = page.locator(".sigmoid svg")
        expect(sg).to_be_visible()
        n_reg = int(page.locator(".info-row[data-k=regions] .v").inner_text())
        dots = page.evaluate("""() => [...document.querySelectorAll('.sigmoid circle.sg-pt')].map(c => [+c.getAttribute('cx'), +c.getAttribute('cy')])""")
        L, R_, T, B, W, H = 34, 10, 10, 30, 320, 178
        off = []
        for cx, cy in dots:
            z = -6 + (cx - L) / (W - L - R_) * 12
            off.append(abs(cy - (T + (1 - 1 / (1 + np.exp(-z))) * (H - T - B))))
        rep["sigmoid"] = {"dots": len(dots), "regions": n_reg, "max_off_curve_px": round(max(off), 3) if off else None,
                          "selected": page.locator(".sigmoid circle.sg-sel").count(), "caption": page.locator(".sigmoid figcaption").inner_text()[:160]}
        if len(dots) != n_reg:
            errors.append(f"sigmoid dots {len(dots)} != regions {n_reg}")
        if off and max(off) > 0.6:
            errors.append(f"sigmoid dot off the curve by {max(off):.2f} px")
        page.locator(".rp").screenshot(path=str(OUT / "bang_ket_qua_sigmoid.png"))

        # 3. pseudo-colour: boxes hidden so that only image pixels are compared
        stage = page.locator(".viewer-stage")
        page.get_by_role("checkbox", name="Khung", exact=True).uncheck()
        page.mouse.move(5, 5)
        time.sleep(0.4)
        g = shot(stage)
        rep["gray_channel_spread"] = int(np.abs(g[..., 0] - g[..., 2]).max())
        page.locator(".view-tools select").select_option("jet")
        time.sleep(0.4)
        c = shot(stage)
        flt = page.evaluate("getComputedStyle(document.querySelector('.viewer-stage img.slice')).filter")
        lum = g[..., 0].astype(float) / 255
        exp = np.stack([np.interp(lum, np.linspace(0, 1, 17), JET[k]) * 255 for k in range(3)], -1)
        img = np.zeros(lum.shape, bool)
        h_, w_ = img.shape
        img[30:h_ - 10, 10:w_ - 60] = True                  # skip the overlay texts / buttons in the corners
        err = np.abs(c - exp)[img].mean(axis=0)
        rep["jet"] = {"filter": flt, "mean_abs_err_rgb": [round(float(v), 2) for v in err],
                      "coloured_pixels_pct": round(float((np.abs(c[..., 0] - c[..., 2]) > 30)[img].mean() * 100), 1)}
        if "lc-cmap-jet" not in flt or err.max() > 6:
            errors.append(f"jet colouring wrong: filter={flt}, err={err}")
        Image.fromarray(c.astype(np.uint8)).save(OUT / "rai_mau_jet.png")
        for cm in ("hot", "bone", "viridis"):
            page.locator(".view-tools select").select_option(cm)
            time.sleep(0.3)
            Image.fromarray(shot(stage).astype(np.uint8)).save(OUT / f"rai_mau_{cm}.png")
        page.locator(".view-tools select").select_option("gray")
        time.sleep(0.4)
        back = shot(stage)
        rep["back_to_gray_max_diff"] = int(np.abs(back - g)[img].max())
        if rep["back_to_gray_max_diff"] > 2:
            errors.append(f"back to gray differs by {rep['back_to_gray_max_diff']}")

        # 4. magnifier: lens centre == image point under the pointer (unzoomed and zoomed x1.5)
        box = stage.bounding_box()
        lens_err = []
        for zoomed in (False, True):
            if zoomed:
                page.locator(".vt-tr button[aria-label='Phóng to']").click()
                page.locator(".view-tools button", has_text="Tắt").click()
                time.sleep(0.3)
            page.mouse.move(5, 5)
            base = shot(stage)
            page.locator(".view-tools button", has_text="3×").click()
            for fx, fy in ((0.5, 0.5), (0.38, 0.45), (0.62, 0.55)):
                x, y_ = box["x"] + fx * box["width"], box["y"] + fy * box["height"]
                page.mouse.move(x, y_)
                time.sleep(0.25)
                lens = page.locator(".lens")
                if not lens.is_visible():
                    errors.append("lens not visible"); continue
                lb = lens.bounding_box()
                li = np.asarray(Image.open(io.BytesIO(page.screenshot(clip={"x": lb["x"] + lb["width"] / 2 - 2, "y": lb["y"] + lb["height"] / 2 - 2, "width": 4, "height": 4}))).convert("RGB")).astype(float)
                sx, sy = int(x - box["x"]), int(y_ - box["y"])
                ref = base[sy - 1:sy + 2, sx - 1:sx + 2].astype(float).mean(axis=(0, 1))
                lens_err.append(round(float(np.abs(li.mean(axis=(0, 1)) - ref).max()), 1))
                if not zoomed and (fx, fy) == (0.5, 0.5):
                    page.screenshot(path=str(OUT / "kinh_lup_3x.png"), clip={"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]})
            rep[f"lens_tag_{'zoom' if zoomed else 'nozoom'}"] = page.locator(".lens-tag").inner_text()
        rep["lens_centre_err"] = lens_err
        if lens_err and max(lens_err) > 40:
            errors.append(f"lens centre differs from the pointed pixel: {lens_err}")
        page.locator(".view-tools button", has_text="Tắt").click()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        if page.locator(".lens").count():
            errors.append("lens still shown when off")

        # phone width: no horizontal scroll with the new tool row
        page.set_viewport_size({"width": 390, "height": 844})
        time.sleep(0.5)
        if page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1"):
            errors.append("horizontal scroll at 390 px")
        page.screenshot(path=str(OUT / "dien_thoai.png"), full_page=False)
        br.close()
    rep["errors"] = errors
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print("PASS" if not errors else f"FAIL ({len(errors)})")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
