"""Real-browser stability test of the slice viewer (Edge via Playwright), with a screen recording.

For CT 3D, MRI 3D and a 2D image (Validation patients only) it samples every 50 ms: the viewer box, the box of the
image element on screen (PNG <img> or playback <canvas>), its intrinsic size, the zoom/pan transform, the page scroll,
and the screen position of every YOLO box versus the position computed from its image coordinates. Phases:
hold one slice 3 s; play >= 20 consecutive slices; zoom x2.25 + pan, then play again (zoom/pan must be kept).
Any change of the viewer / image box or of the transform that the user did not ask for is an error.
Outputs (local only, dataset images): outputs/viewer_stability/ (report.json, screenshots, video .webm).

    .venv\\Scripts\\python scripts\\web\\viewer_stability.py [--base URL]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import load_paths  # noqa: E402
from livercancer.data.sources import image_path  # noqa: E402

OUT = ROOT / "outputs" / "viewer_stability"

SAMPLE = """(ms) => new Promise((res) => {
  const out = []; const t0 = performance.now();
  const r = (e) => { if (!e) return null; const b = e.getBoundingClientRect(); return [+b.left.toFixed(1), +b.top.toFixed(1), +b.width.toFixed(1), +b.height.toFixed(1)]; };
  const tick = () => {
    const st = document.querySelector('.viewer-stage');
    const img = st.querySelector('img.slice'), cv = st.querySelector('canvas.slice');
    const vis = img && img.offsetParent !== null ? img : (cv && getComputedStyle(cv).display !== 'none' ? cv : null);
    const svg = st.querySelector('svg.boxes');
    let boxErr = 0, nBox = 0;
    if (svg && vis) {
      const vb = svg.viewBox.baseVal, V = vis.getBoundingClientRect();
      for (const rc of svg.querySelectorAll('rect[stroke]')) {
        const R = rc.getBoundingClientRect(); nBox++;
        const ex = V.left + (+rc.getAttribute('x')) / vb.width * V.width, ey = V.top + (+rc.getAttribute('y')) / vb.height * V.height;
        boxErr = Math.max(boxErr, Math.abs(R.left - ex), Math.abs(R.top - ey));
      }
    }
    const lab = (document.querySelector('.vt-tl') || {}).textContent || '';
    out.push({ t: Math.round(performance.now() - t0), slice: (lab.match(/Lát (\\d+)/) || [])[1] || '2D',
      stage: r(st), image: r(vis), kind: vis ? vis.tagName : null,
      intrinsic: vis ? (vis.tagName === 'IMG' ? [vis.naturalWidth, vis.naturalHeight] : [vis.width, vis.height]) : null,
      transform: (st.querySelector('.viewer-img') || {}).style?.transform || '', scrollY: Math.round(window.scrollY), boxes: nBox, boxErr: +boxErr.toFixed(2) });
    if (performance.now() - t0 < ms) setTimeout(tick, 50); else res(out);
  };
  tick();
})"""


def check(samples, phase, errors, expect_transform=None):
    """Viewer box, image box, intrinsic size, transform and scroll must not change (slices may change)."""
    first = samples[0]
    slices = sorted({s["slice"] for s in samples}, key=lambda x: (len(x), x))
    for s in samples:
        for key in ("stage", "image"):
            if s[key] and first[key] and max(abs(a - b) for a, b in zip(s[key], first[key])) > 0.6:
                errors.append(f"{phase}: {key} box changed {first[key]} -> {s[key]} at {s['t']} ms (slice {s['slice']})")
                break
        if s["intrinsic"] and first["intrinsic"] and s["intrinsic"][0] and s["intrinsic"] != first["intrinsic"]:
            errors.append(f"{phase}: image size changed {first['intrinsic']} -> {s['intrinsic']} at slice {s['slice']}")
        if s["transform"] != (expect_transform if expect_transform is not None else first["transform"]):
            errors.append(f"{phase}: zoom/pan changed to {s['transform']} at slice {s['slice']}")
        if s["scrollY"] != first["scrollY"]:
            errors.append(f"{phase}: page scrolled {first['scrollY']} -> {s['scrollY']}")
        if s["boxErr"] > 1.0:
            errors.append(f"{phase}: YOLO box {s['boxErr']} px away from its image position at slice {s['slice']}")
    return {"samples": len(samples), "slices_seen": len(slices), "first": slices[0], "last": slices[-1],
            "stage": first["stage"], "image": first["image"], "intrinsic": first["intrinsic"], "transform": first["transform"],
            "max_box_error_px": max(s["boxErr"] for s in samples), "slices_with_boxes": len({s["slice"] for s in samples if s["boxes"]})}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://livercancer-ai-demo.pages.dev")
    args = ap.parse_args()
    from playwright.sync_api import expect, sync_playwright
    paths = load_paths()
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    png = ROOT / "outputs" / "web_e2e" / "inputs" / "ct_lat_cat.png"
    cases = [("CT_3D", "3d", "Phân tích CT", [image_path(paths, "ct", "P0053", "C1")]),
             ("MRI_3D", "3d", "Phân tích MRI", [image_path(paths, "mri", "MR-400851", "T2WI")]),
             ("CT_2D", "2d", "Phân tích CT", [png])]
    report, errors = {}, []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        for name, dims, radio, files in cases:
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}, locale="vi-VN",
                                      record_video_dir=str(OUT / "video_tmp"), record_video_size={"width": 1440, "height": 900})
            page = ctx.new_page()
            page.goto(args.base + "/phan-tich")
            page.get_by_role("radio", name="Phân tích ảnh 2D" if dims == "2d" else "Phân tích ca 3D").click()
            page.get_by_role("radio", name=radio).click()
            page.set_input_files("input[type=file][accept]", [str(f) for f in files])
            expect(page.locator(".concl-title")).to_be_visible(timeout=300000)
            page.locator(".viewer-stage").scroll_into_view_if_needed()
            rep = {}
            # 1. right after the result: hold the opening slice 3 s (no auto play, no resize)
            s = page.evaluate(SAMPLE, 3000)
            rep["hold_3s"] = check(s, f"{name} hold", errors)
            rep["autoplay"] = len({x["slice"] for x in s}) > 1
            if rep["autoplay"]:
                errors.append(f"{name}: the slice changed by itself before Play")
            page.locator(".viewer-stage").screenshot(path=str(OUT / f"{name}_1_giu_lat.png"))
            if dims == "3d":
                # 2. play >= 20 consecutive slices from slice 1 at 1x (8 slices/s)
                page.locator(".player-bar input[type=range]").fill("0")
                page.wait_for_timeout(1200)
                page.get_by_role("radio", name="1x").click()
                page.get_by_role("button", name="Phát", exact=True).click()
                s = page.evaluate(SAMPLE, 3500)
                rep["play"] = check(s, f"{name} play", errors)
                if rep["play"]["slices_seen"] < 20 and rep["play"]["last"] != str(page.evaluate("document.querySelector('.player-bar input').max|0") + 1):
                    errors.append(f"{name}: only {rep['play']['slices_seen']} slices played")
                page.locator(".viewer-stage").screenshot(path=str(OUT / f"{name}_2_dang_phat.png"))
                b = page.get_by_role("button", name="Tạm dừng")
                if b.count():
                    b.click()
                page.wait_for_timeout(1500)
                page.locator(".viewer-stage").screenshot(path=str(OUT / f"{name}_3_tam_dung.png"))
                # 3. user zoom x2.25 + pan, then play again: the user's zoom/pan must stay
                page.get_by_role("button", name="Phóng to").click()
                page.get_by_role("button", name="Phóng to").click()
                bx = page.locator(".viewer-stage").bounding_box()
                page.mouse.move(bx["x"] + bx["width"] / 2, bx["y"] + bx["height"] / 2)
                page.mouse.down()
                page.mouse.move(bx["x"] + bx["width"] / 2 + 50, bx["y"] + bx["height"] / 2 + 30, steps=4)
                page.mouse.up()
                tr = page.locator(".viewer-img").get_attribute("style")
                page.locator(".player-bar input[type=range]").fill("10")
                page.wait_for_timeout(1000)
                page.get_by_role("button", name="Phát", exact=True).click()
                s = page.evaluate(SAMPLE, 3000)
                rep["play_zoomed"] = check(s, f"{name} play zoomed", errors, expect_transform=s[0]["transform"])
                rep["play_zoomed"]["user_transform"] = tr
                if "scale(2.25)" not in s[0]["transform"]:
                    errors.append(f"{name}: user zoom lost before play ({s[0]['transform']})")
                page.locator(".viewer-stage").screenshot(path=str(OUT / f"{name}_4_phat_khi_phong_to.png"))
                b = page.get_by_role("button", name="Tạm dừng")
                if b.count():
                    b.click()
            report[name] = rep
            ctx.close()
            vids = sorted((OUT / "video_tmp").glob("*.webm"), key=lambda f: f.stat().st_mtime)
            if vids:
                vids[-1].rename(OUT / f"video_{name}.webm")
            print(name, json.dumps(rep, ensure_ascii=False)[:600], flush=True)
        browser.close()
    shutil.rmtree(OUT / "video_tmp", ignore_errors=True)
    (OUT / "report.json").write_text(json.dumps({"cases": report, "errors": errors}, indent=1, ensure_ascii=False), encoding="utf-8")
    for e in errors[:40]:
        print("  ERROR", e)
    print("STABILITY", "PASS" if not errors else f"FAIL ({len(errors)})", "->", OUT)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
