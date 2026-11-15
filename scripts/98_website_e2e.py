"""End-to-end test of the website in a real browser (Microsoft Edge via Playwright) against the real backend.

Prerequisite: backend running (and serving the built frontend):
    .venv\\Scripts\\python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
Inputs: Validation patients only (never Test). 2D inputs are Val slices exported as PNG, and SYNTHETIC screenshots /
photos of a screen built from Val slices (livercancer.intake.synthetic); DICOM is converted from Val NIfTI
(livercancer.intake.dicom_fixtures), once more with duplicated files. No real hospital image is available: those
cases are labelled synthetic.
Flow (workstation page): the user picks "Phân tích ảnh 2D" / "Phân tích ca 3D" and CT / MRI, then uploads; the
analysis starts by itself. A file whose own information contradicts the choice (MRI NIfTI uploaded as CT) must stop
at a confirmation; a PNG in 3D mode must be refused.
Checks: waiting state without any number before an analysis, AI conclusion, Hybrid CT scores (only when classified),
case information (slices, suspicious slices, DICOM duplicates), viewer (boxes, previous / next, slider, thumbnails jump
to the right slice, zoom, pan, full screen), Tumor Zoom pair, downloads (PNG, HTML report, the report printed to PDF),
Vietnamese texts, NO access code, no 401, no horizontal scroll, laptop + phone + projector (1920 x 1080), direct links
(no 404), and the Vietnamese message when the AI server is off. Cloudflare Pages: --base https://<project>.pages.dev
finds the AI server in /runtime-config.json. Screenshots (local only, dataset images: not committed) -> outputs/web_e2e/
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import load_paths  # noqa: E402
from livercancer.data.slices import load_volume_uint8  # noqa: E402
from livercancer.data.sources import image_path  # noqa: E402

OUT = ROOT / "outputs" / "web_e2e"
VAL = ROOT / "outputs" / "predictions" / "pipeline_validation_gate"
PAGES = [("/", "LiverCancer AI"), ("/phan-tich", "Chọn loại phân tích"), ("/mo-hinh", "Từ ảnh gốc đến kết quả"),
         ("/ket-qua", "Chỉ số đo được"), ("/gioi-thieu", "Mục tiêu nghiên cứu")]


def fetch(url: str, timeout=10, tries=4) -> bytes:
    # Cloudflare refuses Python's default User-Agent (error 1010); transient network errors are retried
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "LiverCancerAI-e2e/1.0"}), timeout=timeout).read()
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(3)
    return b""


def wait_ready(base: str, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = json.loads(fetch(base + "/api/health", tries=1))
            if all(v == "ready" for v in h["models"].values()):
                return h
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    raise SystemExit("backend not ready")


def series_of(pid):
    return [v["series"] for v in json.loads((VAL / "ct" / "cases" / f"{pid}.json").read_text(encoding="utf-8"))["volumes"]]


def val_slice(ds: str, seed: int):
    root = ROOT / "data" / "processed" / ds
    rows = [r for r in csv.DictReader(open(root / "slices.csv", encoding="utf-8")) if r["split"] == "val" and r["kind"] == "pos"]
    r = random.Random(seed).choice(rows)
    return cv2.imread(str(root / "images" / "val" / f"{r['file']}.png"), cv2.IMREAD_GRAYSCALE)


def make_inputs(paths) -> list[tuple]:
    from livercancer.intake.dicom_fixtures import series_from_volume, zip_blobs
    from livercancer.intake.synthetic import jpeg, photo_of_screen, render_screenshot
    d = OUT / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    ct2d, mr2d = val_slice("full/yolo_ct", 11), val_slice("pilot/yolo_mri", 12)
    cv2.imwrite(str(d / "ct_lat_cat.png"), ct2d)
    cv2.imwrite(str(d / "mri_lat_cat.png"), mr2d)
    shot, _ = render_screenshot([ct2d], seed=21, panels=1)
    cv2.imwrite(str(d / "ct_chup_man_hinh.jpg"), jpeg(shot, 80))
    shot, _ = render_screenshot([mr2d], seed=22, panels=1)
    cv2.imwrite(str(d / "mri_chup_lai_man_hinh.jpg"), photo_of_screen(shot, 22))
    blobs = []
    for i, s in enumerate(("C2", "P")):
        blobs += series_from_volume(load_volume_uint8(image_path(paths, "ct", "P0267", s)), "CT", f"1.2.826.0.9.{i + 1}")
    (d / "ct_dicom.zip").write_bytes(zip_blobs(blobs))
    (d / "ct_dicom_trung_tep.zip").write_bytes(zip_blobs(blobs + blobs[40:45]))          # 5 files twice
    mri3 = [image_path(paths, "mri", "MR-400851", s) for s in ("T2WI", "C+A", "DWI")]
    return [  # (case, files, expected conclusion, chosen modality, modality after an eventual confirmation, 2d | 3d)
        ("ct_nifti", [image_path(paths, "ct", "P0267", s) for s in series_of("P0267")], "Nghi ngờ", "ct", "ct", "3d"),
        ("ct_1pha_92lat", [image_path(paths, "ct", "P0053", "C1")], "Nghi ngờ|Chưa đủ|Không phát hiện", "ct", "ct", "3d"),
        ("ct_gate", [image_path(paths, "ct", "P0133", s) for s in series_of("P0133")], "Chưa đủ cơ sở để phân loại", "ct", "ct", "3d"),
        ("ct_no_region", [image_path(paths, "ct", "P0304", s) for s in series_of("P0304")], "Không phát hiện vùng nghi ngờ trong ca chụp này", "ct", "ct", "3d"),
        ("mri_nifti", mri3, "vùng nghi ngờ", "mri", "mri", "3d"),
        ("ct_dicom_zip", [d / "ct_dicom.zip"], "Nghi ngờ|Chưa đủ|Không phát hiện", "ct", "ct", "3d"),
        ("ct_dicom_zip_duplicates", [d / "ct_dicom_trung_tep.zip"], "Nghi ngờ|Chưa đủ|Không phát hiện", "ct", "ct", "3d"),
        ("ct_png", [d / "ct_lat_cat.png"], ".+", "ct", "ct", "2d"),
        ("mri_png", [d / "mri_lat_cat.png"], ".+", "mri", "mri", "2d"),
        ("ct_screenshot_synthetic", [d / "ct_chup_man_hinh.jpg"], ".+", "ct", "ct", "2d"),
        ("mri_photo_synthetic", [d / "mri_chup_lai_man_hinh.jpg"], ".+", "mri", "mri", "2d"),
        ("conflict_mri_as_ct", mri3[:1], "vùng nghi ngờ|Không phát hiện", "ct", "mri", "3d"),
        ("png_in_3d_mode", [d / "ct_lat_cat.png"], "REFUSE", "ct", "ct", "3d"),
    ]


def scan3d(page, rec, errors, name, expect):
    """3D scan player, boxes per frame, speed, boxes on/off, 3D regions tab, MP4 export (real browser)."""
    import imageio_ffmpeg
    lab = lambda: int(re.search(r"Lát (\d+) /", text(page, ".vt-tl")).group(1))  # noqa: E731
    total = lambda: int(re.search(r"/ (\d+)", text(page, ".vt-tl")).group(1))  # noqa: E731

    def pause():
        b = page.get_by_role("button", name="Tạm dừng")
        if b.count():
            b.click()
    page.locator(".player-bar input[type=range]").fill("0")
    expect(page.locator(".vt-tl")).to_contain_text("Lát 1 /", timeout=20000)
    page.get_by_role("radio", name="1x").click()
    page.get_by_role("button", name="Phát", exact=True).click()
    page.wait_for_timeout(2500)
    k1 = lab()
    page.wait_for_timeout(1500)
    k2 = lab()
    pause()
    rec["play_advanced"] = k1 > 5 and (k2 > k1 or k2 == total())
    if not rec["play_advanced"]:
        errors.append(f"{name}: playback did not advance ({k1} -> {k2})")
    # paused: the exact PNG and its server boxes must equal the frame's detection list from the 3D regions
    page.wait_for_timeout(1200)
    n_rect = page.locator(".viewer-stage svg.boxes rect[stroke]").count()
    n_info = page.locator(".frame-info .fi-chip").count()
    rec["frame_boxes_match"] = n_rect == n_info
    if n_rect != n_info:
        errors.append(f"{name}: slice {lab()} has {n_rect} boxes on the image but {n_info} detections listed")
    page.get_by_role("checkbox", name="Khung").uncheck()
    rec["boxes_toggle"] = page.locator(".viewer-stage svg.boxes").count() == 0
    page.get_by_role("checkbox", name="Khung").check()
    page.get_by_role("radio", name="4x").click()
    page.get_by_role("button", name="Phát lại từ lát 1").click()
    page.wait_for_timeout(2000)
    pause()
    rec["speed_4x_slices_in_2s"] = lab()
    page.get_by_role("radio", name="1x").click()
    # 3D regions tab: hidden on the user's request (2026-11-08, Viewer.tsx SHOW_3D_TAB = false) -> must be absent
    if page.get_by_role("tab", name=re.compile("Vùng nghi ngờ 3D")).count() == 0:
        rec["three_d_tab"] = "hidden"
    else:
        rec["three_d_tab"] = "shown"
        scan3d_tab(page, rec, errors, name, expect)
    mp4_check(page, rec, errors, name, expect)


def scan3d_tab(page, rec, errors, name, expect):
    """3D regions tab (only when it is enabled in the viewer)."""
    page.get_by_role("tab", name=re.compile("Vùng nghi ngờ 3D")).click()
    host = page.locator(".l3d-host")
    try:
        expect(host.locator("canvas")).to_be_visible(timeout=30000)
        expect(host).to_have_attribute("data-plane-k", re.compile(r"\d+"), timeout=30000)
        rec["three_d_canvas"] = True
    except AssertionError:
        rec["three_d_canvas"] = False
        errors.append(f"{name}: 3D view did not render")
    page.locator(".l3d-bar input[type=range]").fill("10")
    try:
        expect(host).to_have_attribute("data-plane-k", "10", timeout=20000)
        rec["plane_follows_slider"] = True
    except AssertionError:
        rec["plane_follows_slider"] = False
        errors.append(f"{name}: slice plane did not move to slice 11")
    page.wait_for_timeout(600)
    rec["plane_gray_levels"] = page.evaluate("""() => {
      const c = document.querySelector('.l3d-host canvas'); const t = document.createElement('canvas');
      t.width = c.width; t.height = c.height; const g = t.getContext('2d'); g.drawImage(c, 0, 0);
      const d = g.getImageData(Math.floor(c.width * 0.3), Math.floor(c.height * 0.3), Math.floor(c.width * 0.4), Math.floor(c.height * 0.4)).data;
      const s = new Set(); for (let i = 0; i < d.length; i += 4) if (Math.abs(d[i] - d[i + 1]) < 6 && Math.abs(d[i + 1] - d[i + 2]) < 6) s.add(d[i]);
      return s.size; }""")
    if rec["plane_gray_levels"] < 40:
        errors.append(f"{name}: slice plane shows no CT/MRI image ({rec['plane_gray_levels']} gray levels)")
    for v in ("Trục X", "Trục Y", "Trục Z", "Mặc định"):
        page.locator(".l3d-cams").get_by_role("button", name=v).click()
        page.wait_for_timeout(200)
    rows = page.locator(".r3d-table tbody tr")
    n_rows = rows.count() if "Không có" not in rows.nth(0).inner_text() else 0
    rec["three_d_rows"] = n_rows
    page.get_by_role("button", name="Ẩn tất cả").click()
    hid = host.get_attribute("data-visible-regions")
    page.get_by_role("button", name="Hiện tất cả").click()
    shown = host.get_attribute("data-visible-regions")
    rec["three_d_visibility"] = hid == "0" and shown == str(n_rows)
    if not rec["three_d_visibility"]:
        errors.append(f"{name}: region visibility toggles gave {hid} / {shown} of {n_rows}")
    page.screenshot(path=str(OUT / f"vung_3d_{name}.png"), full_page=True)
    if n_rows >= 2:
        best = int(rows.nth(1).locator("td").nth(3).inner_text())
        rows.nth(1).click()
        try:
            expect(host).to_have_attribute("data-plane-k", str(best - 1), timeout=20000)
            expect(page.locator(".l3d-sel .zoom-pair")).to_be_visible(timeout=10000)
            page.get_by_role("button", name="Xem trên video quét lát").click()
            expect(page.locator(".vt-tl")).to_contain_text(f"Lát {best} /", timeout=20000)
            rec["three_d_select_jump"] = True
        except AssertionError:
            rec["three_d_select_jump"] = False
            errors.append(f"{name}: choosing a 3D region did not move the plane / video to its representative slice {best}")
    else:
        page.get_by_role("tab", name="Quét lát cắt").click()


def mp4_check(page, rec, errors, name, expect):
    import imageio_ffmpeg
    # MP4 export of the scan
    with page.expect_download(timeout=180000) as dl:
        page.locator(".frame-info").get_by_role("button", name="Tải video phân tích 3D (MP4)").click()
    path = Path(dl.value.path())
    rd = imageio_ffmpeg.read_frames(str(path))
    meta = next(rd)
    n_frames = sum(1 for _ in rd)
    n_slices = int(re.search(r"/ (\d+)", text(page, ".vt-tl")).group(1))
    rec["video"] = f"{meta.get('codec')} {meta.get('size')} {n_frames} frames / {n_slices} slices, {path.stat().st_size // 1024} KB"
    rec["video_ok"] = meta.get("codec") == "h264" and n_frames == n_slices
    if not rec["video_ok"]:
        errors.append(f"{name}: video {rec['video']}")


def text(page, sel: str) -> str:
    loc = page.locator(sel)
    return " ".join((loc.first.text_content() or "").split()) if loc.count() else ""   # also text inside closed <details>


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000", help="address of the website")
    ap.add_argument("--api", default="", help="force a separate AI server as ?api= (default: /runtime-config.json, else same address)")
    ap.add_argument("--viewports", default="laptop,dien_thoai,may_chieu", help="subset of laptop,dien_thoai,may_chieu")
    ap.add_argument("--cases", default="", help="comma-separated subset of case names (default: all)")
    args = ap.parse_args()
    from playwright.sync_api import expect, sync_playwright
    api = args.api
    if not api:
        try:
            api = json.loads(fetch(args.base + "/runtime-config.json", 15)).get("api_base") or ""
        except Exception:  # noqa: BLE001 - served by the backend itself: no runtime config
            api = ""
    api = api or args.base
    print("website:", args.base, "| AI server:", api)
    h = wait_ready(api)
    print("backend ready:", h["models"], h["device"], "modality model:", h.get("modality_model"))
    paths = load_paths()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = make_inputs(paths)
    if args.cases:
        cases = [c for c in cases if c[0] in args.cases.split(",")]
    errors, report = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        viewports = (("laptop", {"width": 1440, "height": 900}), ("dien_thoai", {"width": 390, "height": 844}),
                     ("may_chieu", {"width": 1920, "height": 1080}))
        for vp_name, vp in [v for v in viewports if v[0] in args.viewports.split(",")]:
            ctx = browser.new_context(viewport=vp, locale="vi-VN", accept_downloads=True)
            page = ctx.new_page()
            page.on("console", lambda m: errors.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("response", lambda r: errors.append(f"HTTP {r.status} {r.url}") if r.status in (401, 403) and "/api/" in r.url else None)
            if args.api:
                page.goto(args.base + "/?api=" + args.api)
            # public website: no access code, CT / MRI choice usable at once, a waiting state without any number
            page.goto(args.base + "/phan-tich")
            expect(page.get_by_role("radio", name="Phân tích CT")).to_be_enabled(timeout=20000)
            expect(page.get_by_role("radio", name="Phân tích MRI")).to_be_enabled(timeout=20000)
            for bad in ("Mã truy cập", "Kiểm tra mã", "mã do nhóm nghiên cứu"):
                if page.get_by_text(bad, exact=False).count():
                    errors.append(f"access code UI still visible: '{bad}' ({vp_name})")
            if page.locator("#code, input[type=password]").count():
                errors.append(f"access code input still present ({vp_name})")
            if not (page.locator(".rp-wait").is_visible() and page.locator(".viewer-empty").is_visible()) or page.locator(".concl-title").count():
                errors.append(f"waiting state wrong before any analysis ({vp_name})")
            if re.search(r"\d+[,.]\d+\s*%", text(page, ".ws-right")):
                errors.append(f"a percentage is shown before any analysis ({vp_name})")
            page.screenshot(path=str(OUT / f"cho_phan_tich_{vp_name}.png"), full_page=True)
            for path, txt in PAGES:
                page.goto(args.base + path)
                expect(page.get_by_text(txt, exact=False).first).to_be_visible(timeout=10000)
                page.screenshot(path=str(OUT / f"trang{path.replace('/', '_') or '_chu'}_{vp_name}.png"), full_page=True)
                if page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1"):
                    errors.append(f"horizontal scroll on {path} ({vp_name})")
                try:
                    expect(page.locator(".ai-status.ready")).to_be_visible(timeout=20000)
                except AssertionError:
                    errors.append(f"AI status not 'ready' on {path} ({vp_name})")
            todo = cases if vp_name == "laptop" else [c for c in cases if c[0] in (("ct_nifti",) if vp_name == "may_chieu" else ("ct_nifti", "mri_png", "conflict_mri_as_ct"))]
            for name, files, expect_text, chosen, final, dims in todo:
                page.goto(args.base + "/phan-tich")
                if page.locator("input[type=file][accept]").is_enabled():
                    errors.append(f"{name}: upload enabled before choosing CT / MRI")
                page.get_by_role("radio", name="Phân tích ảnh 2D" if dims == "2d" else "Phân tích ca 3D").click()
                page.get_by_role("radio", name="Phân tích CT" if chosen == "ct" else "Phân tích MRI").click()
                t0 = time.time()
                page.set_input_files("input[type=file][accept]", [str(f) for f in files])
                first = page.locator(".concl-title").or_(page.locator(".notice.danger")).or_(page.locator(".confirm"))
                expect(first.first).to_be_visible(timeout=600000)
                asked = ""
                if page.locator(".confirm").is_visible():
                    asked = page.locator(".confirm").inner_text()
                    page.screenshot(path=str(OUT / f"xac_nhan_{name}_{vp_name}.png"), full_page=True)
                    page.locator(".confirm-btns").get_by_role("button", name=f"Phân tích như {final.upper()}").click()
                    expect(page.locator(".concl-title").or_(page.locator(".notice.danger")).first).to_be_visible(timeout=600000)
                if page.locator(".notice.danger").is_visible():
                    msg = page.locator(".notice.danger").inner_text()
                    report.append({"case": name, "viewport": vp_name, "state": "refused", "message": msg[:220], "seconds": round(time.time() - t0, 1)})
                    print(f"  {vp_name:10s} {name:26s} REFUSED  {msg[:110]}", flush=True)
                    page.screenshot(path=str(OUT / f"ket_qua_{name}_{vp_name}.png"), full_page=True)
                    continue
                expect(page.locator(".concl-title")).to_have_text(re.compile(expect_text), timeout=60000)
                dt = time.time() - t0
                rec = {"case": name, "viewport": vp_name, "state": "done", "seconds": round(dt, 1), "dims": dims,
                       "result": text(page, ".concl-title"), "badge": text(page, ".badge"),
                       "modality": text(page, ".info-row[data-k=modality] .v"), "modality_how": text(page, ".info-row[data-k=modality] .s"),
                       "regions": text(page, ".info-row[data-k=regions] .v"), "detect_conf": text(page, ".info-row[data-k=conf] .v"),
                       "gate": text(page, ".info-row[data-k=gate] .v"), "total_slices": text(page, ".info-row[data-k=slices] .v"),
                       "suspicious_slices": text(page, ".info-row[data-k=suspicious] .v"), "dicom_dup_files": text(page, ".info-row[data-k=dupfiles] .v"),
                       "time": text(page, ".info-row[data-k=time] .v"),
                       "images_shown": page.locator(".viewer-stage img.slice").count() + page.locator(".zoom-pair img").count(),
                       "boxes_on_view": page.locator(".viewer-stage svg.boxes rect[stroke]").count(),
                       "zoom_placeholder": page.locator(".zoom-pair .zp.empty").count(),
                       "class_bars": page.locator(".rp .bar-row:not(.det)").count(), "thumbs": page.locator(".thumb").count(),
                       "chosen": chosen, "expected_modality": final, "asked_confirmation": asked[:200],
                       "commentary": text(page, ".comment")[:300]}
                if vp_name == "laptop":
                    # boxes drawn exactly at the image coordinates (scale 1): SVG rect vs stage, in screen pixels
                    rec["box_px_error"] = page.evaluate("""() => {
                      const st = document.querySelector('.viewer-stage'), svg = st.querySelector('svg.boxes'), r = svg.querySelector('rect[stroke]');
                      if (!r) return null;
                      const vb = svg.viewBox.baseVal, S = st.getBoundingClientRect(), R = r.getBoundingClientRect();
                      const x = +r.getAttribute('x'), y = +r.getAttribute('y'), w = +r.getAttribute('width'), h = +r.getAttribute('height');
                      const ex = S.left + x / vb.width * S.width, ey = S.top + y / vb.height * S.height;
                      const ew = w / vb.width * S.width, eh = h / vb.height * S.height;
                      const img = st.querySelector('img.slice').getBoundingClientRect();
                      return Math.max(Math.abs(R.left - ex), Math.abs(R.top - ey), Math.abs(R.width - ew), Math.abs(R.height - eh),
                                      Math.abs(img.width / img.height - vb.width / vb.height) * 100);
                    }""")
                    if rec["box_px_error"] is not None and rec["box_px_error"] > 1.5:
                        errors.append(f"{name}: box drawn {rec['box_px_error']:.1f}px away from its image coordinates")
                if vp_name == "laptop" and page.locator(".zoom-head .chips button").count() >= 2:
                    page.locator(".zoom-head .chips button").nth(1).click()
                    cap = text(page, ".zoom-block > p")
                    m = re.search(r"lát đại diện (\d+)", cap)
                    if m:
                        try:
                            expect(page.locator(".vt-tl")).to_contain_text(f"Lát {m.group(1)} /", timeout=20000)
                            rec["region_jump"] = page.locator(".viewer-stage svg.boxes rect[stroke='#ef4444']").count() >= 1
                        except AssertionError:
                            rec["region_jump"] = False
                        if not rec["region_jump"]:
                            errors.append(f"{name}: choosing a region did not open its representative slice with its box")
                    page.locator(".zoom-head .chips button").nth(0).click()
                if vp_name == "laptop" and dims == "3d" and rec["thumbs"] >= 2:
                    th = page.locator(".thumb").nth(1)
                    k = int(re.search(r"\d+", th.locator(".t-k").inner_text()).group())
                    th.click()
                    try:
                        expect(page.locator(".vt-tl")).to_contain_text(f"Lát {k} /", timeout=20000)
                        rec["thumb_jump"] = True
                        page.wait_for_timeout(300)
                        pickable = page.locator(".viewer-stage svg.boxes g.pickable").count()
                        rec["thumb_selects_region"] = page.locator(".viewer-stage svg.boxes rect[stroke='#ef4444']").count() >= 1 or pickable == 0
                        if not rec["thumb_selects_region"]:
                            errors.append(f"{name}: after a thumbnail, no region of that slice is selected")
                    except AssertionError:
                        rec["thumb_jump"] = False
                        errors.append(f"{name}: thumbnail did not open slice {k}")
                    rec["thumb_img_loaded"] = page.evaluate("() => { const i = document.querySelector('.thumb img'); return !!i && i.complete && i.naturalWidth > 0; }")
                    before = text(page, ".vt-tl")
                    page.locator(".vnav.next").click()
                    try:
                        expect(page.locator(".vt-tl")).to_contain_text(f"Lát {k + 1} /", timeout=20000)
                        rec["next_button"] = True
                    except AssertionError:
                        rec["next_button"] = False
                        errors.append(f"{name}: next slice button did not work ({before})")
                    page.locator(".player-bar input[type=range]").fill("5")
                    expect(page.locator(".vt-tl")).to_contain_text("Lát 6 /", timeout=20000)
                    rec["slider"] = text(page, ".slice-label")
                if vp_name == "laptop" and name in ("ct_1pha_92lat", "mri_nifti"):
                    scan3d(page, rec, errors, name, expect)
                if vp_name == "laptop":
                    page.get_by_role("button", name="Phóng to").click()
                    page.get_by_role("button", name="Phóng to").click()
                    tr = page.locator(".viewer-img").get_attribute("style") or ""
                    box = page.locator(".viewer-stage").bounding_box()
                    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                    page.mouse.down()
                    page.mouse.move(box["x"] + box["width"] / 2 + 60, box["y"] + box["height"] / 2 + 40, steps=5)
                    page.mouse.up()
                    tr2 = page.locator(".viewer-img").get_attribute("style") or ""
                    rec["zoom"] = "scale(2.25)" in tr
                    rec["pan"] = "translate(0px, 0px)" not in tr2
                    page.get_by_role("button", name="Toàn màn hình").click()
                    rec["fullscreen"] = page.locator(".viewer-wrap.full").is_visible()
                    page.screenshot(path=str(OUT / f"toan_man_hinh_{name}.png"))
                    page.get_by_role("button", name="Thoát toàn màn hình").click()
                    page.get_by_role("button", name="Vừa khung").click()
                    with page.expect_download() as d1:
                        page.get_by_role("button", name=re.compile(r"^Tải ảnh .*\(PNG\)$")).click()
                    png = Path(d1.value.path()).read_bytes()
                    rec["png_ok"] = png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10000
                    with page.expect_download() as d2:
                        page.get_by_role("button", name="Tải báo cáo (HTML)").click()
                    rep = Path(d2.value.path()).read_text(encoding="utf-8")
                    rec["download_image"] = d1.value.suggested_filename
                    rec["download_report_ok"] = "Báo cáo kết quả phân tích" in rep and rec["result"] in rep and "Thông tin ca phân tích" in rep
                    rp = ctx.new_page()
                    rp.set_content(rep)
                    pdf = rp.pdf(format="A4", print_background=True)
                    rp.close()
                    (OUT / f"bao_cao_{name}.pdf").write_bytes(pdf)
                    rec["report_pdf_bytes"] = len(pdf)
                    rec["report_pdf_ok"] = pdf[:4] == b"%PDF" and len(pdf) > 20000
                    page.evaluate("() => { window.__printed = null; window.addEventListener('lcai-print', (e) => { window.__printed = e.detail; }); }")
                    page.get_by_role("button", name="In / Lưu PDF").click()
                    page.wait_for_function("() => window.__printed !== null", timeout=20000)
                    pr = page.evaluate("() => window.__printed")
                    rec["print_button"] = bool(pr and pr["chars"] > 5000 and pr["images"] >= 2)
                page.screenshot(path=str(OUT / f"ket_qua_{name}_{vp_name}.png"), full_page=True)
                if page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1"):
                    errors.append(f"{name}: horizontal scroll on the result ({vp_name})")
                report.append(rec)
                print(f"  {vp_name:10s} {name:26s} {dt:5.1f}s  {rec['result']} | {rec['modality']} ({rec['modality_how']}) | {rec['regions']} vùng"
                      f" | lát {rec['total_slices'] or '-'} / nghi ngờ {rec['suspicious_slices'] or '-'} | gate {rec['gate'] or '-'}"
                      + (f" | DICOM trùng {rec['dicom_dup_files']}" if rec["dicom_dup_files"] else "") + (" | đã hỏi xác nhận" if asked else ""), flush=True)
                if rec["images_shown"] < 2 and not rec["zoom_placeholder"]:
                    errors.append(f"{name}: images missing")
            ctx.close()
        # AI server off: a dead address must give the Vietnamese message, the information pages still work
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, locale="vi-VN")
        page = ctx.new_page()
        page.goto(args.base + "/phan-tich?api=https://lcai-may-chu-tat-kiem-thu.trycloudflare.com")
        try:
            expect(page.get_by_text("Máy chủ AI hiện đang tắt", exact=False).first).to_be_visible(timeout=60000)
            expect(page.locator(".ai-status.off")).to_be_visible(timeout=60000)
            page.screenshot(path=str(OUT / "may_chu_tat.png"), full_page=True)
            report.append({"case": "server_off_message", "viewport": "laptop", "state": "done", "result": "shown"})
        except AssertionError:
            errors.append("no Vietnamese message when the AI server is off")
        for path, txt in PAGES[2:]:
            page.goto(args.base + path)
            try:
                expect(page.get_by_text(txt, exact=False).first).to_be_visible(timeout=15000)
            except AssertionError:
                errors.append(f"information page {path} broken when the AI server is off")
        ctx.close()
        browser.close()
    (OUT / "e2e_report.json").write_text(json.dumps({"cases": report, "errors": errors}, indent=1, ensure_ascii=False), encoding="utf-8")
    must = {c[0] for c in cases[:7] if c[5] == "3d"} if "laptop" in args.viewports else set()
    ok = not errors and all(r["state"] == "done" for r in report if r["case"] in must)
    ok &= any(r["case"] == "server_off_message" for r in report)
    rs = [r for r in report if r["case"] != "server_off_message"]
    ok &= all(r["state"] == "refused" for r in rs if r["case"] == "png_in_3d_mode")
    done = [r for r in rs if r["state"] == "done"]
    ok &= all(r.get(k, True) for r in done for k in ("download_report_ok", "report_pdf_ok", "fullscreen", "zoom", "pan", "thumb_jump",
                                                         "next_button", "thumb_img_loaded", "print_button", "png_ok", "region_jump",
                                                         "thumb_selects_region", "play_advanced", "frame_boxes_match",
                                                         "boxes_toggle", "three_d_canvas", "three_d_select_jump", "video_ok",
                                                         "plane_follows_slider", "three_d_visibility"))
    ok &= all("trong một ca chụp CT 3D gồm 92 lát cắt" in r["commentary"] for r in done if r["case"] == "ct_1pha_92lat")
    ok &= all(r["zoom_placeholder"] >= 1 for r in done if r["case"] == "ct_no_region")
    ok &= all(r["modality"] == r["expected_modality"].upper() for r in done)            # the chosen model was used
    ok &= all(bool(r["asked_confirmation"]) == (r["case"] == "conflict_mri_as_ct") for r in done)
    ok &= all(r["class_bars"] == 3 for r in done if r["result"].startswith("Nghi ngờ"))  # HCC / ICC / CHCC scores
    ok &= all(r["class_bars"] == 0 for r in done if not r["result"].startswith("Nghi ngờ"))
    d3 = [r for r in done if r["dims"] == "3d"]       # 3D: case counts shown; thumbnails (current volume) iff suspicious slices
    ok &= all(r["total_slices"] and (r["thumbs"] > 0) == (int(r["suspicious_slices"] or 0) > 0) for r in d3)
    ok &= all(r["thumbs"] <= int(r["suspicious_slices"] or 0) for r in d3)
    ok &= all(r["dicom_dup_files"] == "5" for r in done if r["case"] == "ct_dicom_zip_duplicates")
    for e in errors:
        print("  ERROR", e)
    print("E2E", "PASS" if ok else "FAIL", "->", OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
