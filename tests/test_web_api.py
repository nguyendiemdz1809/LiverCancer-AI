"""Web API (server/) on REAL checkpoints and Validation patients (never Test).

CT results must equal the Validation pipeline run with the locked configuration (outputs/predictions/
pipeline_validation_gate). Covered: lesion classified, evidence gate refusal, no YOLO region, MRI detection only,
DICOM .zip (same result as NIfTI, no patient metadata in the response), 2D PNG and synthetic screenshot analysed with
the modality the user chose, confirmation (nothing analysed) when the file's information contradicts that choice,
display Tumor Zoom, slice viewer, chunked upload, public API (no access code) with per-client limits and a time limit,
refusals, temporary files removed.
The website always sends the chosen modality ("ct" / "mri"); the tests do the same.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLO_OFFLINE", "1")
VAL = ROOT / "outputs" / "predictions" / "pipeline_validation_gate"


def _paths():
    from livercancer.config import load_paths
    from livercancer.data.sources import image_path
    paths = load_paths()
    return lambda mod, pid, s: image_path(paths, mod, pid, s)


@pytest.fixture(scope="module")
def client():
    if not (ROOT / "outputs" / "runs" / "cls" / "ct_hybrid_gt_s42" / "best.pt").exists():
        pytest.skip("checkpoints not available")
    from fastapi.testclient import TestClient
    import server.main as m
    with TestClient(m.app) as c:
        t0 = time.time()
        while time.time() - t0 < 300:
            st = c.get("/api/health").json()["models"]
            if all(v != "loading" for v in st.values()):
                break
            time.sleep(1)
        assert st == {"ct": "ready", "mri": "ready"}, st
        time.sleep(1)
        yield c


def _files(mod, pid, series):
    ip = _paths()
    out = []
    for s in series:
        p = ip(mod, pid, s)
        if not p.exists():
            pytest.skip("source dataset not available")
        out.append((p.name, p.read_bytes()))
    return out


def _wait(client, jid):
    for _ in range(2250):                        # 15 min: a 4-phase CT on a busy 4 GB GPU can take ~5 min
        j = client.get(f"/api/jobs/{jid}").json()
        if j["state"] in ("done", "error", "needs_confirm"):
            return j
        time.sleep(0.4)
    raise AssertionError("timeout")


def _upload(client, items, modality="auto", chunk=32 << 20, dims="", detector=""):
    """[(name, bytes)] -> job state after a chunked upload (like the website)."""
    r = client.post("/api/uploads", json={"files": [{"name": n, "size": len(b)} for n, b in items]})
    assert r.status_code == 200, r.text
    uid = r.json()["upload_id"]
    for i, (_, b) in enumerate(items):
        for off in range(0, len(b), chunk):
            r = client.post(f"/api/uploads/{uid}/chunk?index={i}&offset={off}", content=b[off:off + chunk],
                            headers={"Content-Type": "application/octet-stream"})
            assert r.status_code == 200, r.text
    r = client.post(f"/api/uploads/{uid}/analyze", data={"modality": modality, **({"dims": dims} if dims else {}),
                                                         **({"detector": detector} if detector else {})})
    assert r.status_code == 200, r.text
    return _wait(client, r.json()["job_id"]), r.json()["job_id"]


def _val(pid):
    return json.loads((VAL / "ct" / "cases" / f"{pid}.json").read_text(encoding="utf-8"))


def _series(pid):
    return [v["series"] for v in _val(pid)["volumes"]]


def _same_scores(res, ref):
    got = {s["code"]: s["prob"] for s in res["scores"]}
    return set(got) == set(ref) and all(abs(got[c] - ref[c]) < 2e-3 for c in got)


def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["limits"]["max_files"] >= 4 and h["locked_config"]


def test_ct_lesion_identical_to_validation(client):
    ref = _val("P0267")
    j, jid = _upload(client, _files("ct", "P0267", _series("P0267")), "ct", chunk=8 << 20)
    assert j["state"] == "done", j
    res = j["result"]
    assert res["modality"] == "ct" and res["status"] == "ok" and res["label"] == ref["pipeline_label"] and _same_scores(res, ref["pipeline_scores"])
    assert res["n_regions"] == len(ref["tracks"]) and res["cards"]["result"]["title"].startswith("Nghi ngờ")
    assert res["cards"]["modality"]["value"] == "CT" and res["cards"]["modality"]["sub"] == "Bạn đã chọn · khớp header NIfTI"
    assert res["mode"] == "single" and len(res["branches"]) == 1
    b = res["branches"][0]
    assert b["chart"]["kind"] == "class_scores" and abs(sum(i["value"] for i in b["chart"]["items"]) - 1) < 1e-2
    # the chart shows exactly the backend's Hybrid scores (which equal the Validation pipeline run, checked above)
    assert [(i["label"], i["value"]) for i in b["chart"]["items"]] == [(s["name"], s["prob"]) for s in res["scores"]]
    assert any(f"{res['n_regions']} vùng nghi ngờ" in t for t in res["commentary"])
    assert b["view"]["png"] and b["regions"][0]["local_png"] and res["annotated_png"]
    assert len(b["regions"]) > 1 and len(b["all_regions"]) == res["n_regions"]                # several regions
    for r in b["regions"]:                                   # display zoom: window contains the box, at most 4x
        w = r["local_window"]
        assert w[0] <= r["box"][0] and w[1] <= r["box"][1] and w[2] >= r["box"][2] and w[3] >= r["box"][3]
        assert r["local_scale"] <= 4 and r["context_scale"] <= 4
        assert r["context_window"][2] - r["context_window"][0] >= w[2] - w[0]
    assert any("tổn thương nhỏ" in t for t in res["details"]["limitations"])          # moved to advanced details
    assert {v["name"] for v in b["volumes"]} == {f"Pha {s}" for s in _series("P0267")}  # never the file name
    v = client.get(f"/api/jobs/{jid}/slice", params={"branch": "ct", "volume": b["view"]["volume"], "k": 3}).json()
    assert v["slice"] == 3 and v["png"] and isinstance(v["boxes"], list)


def test_ct_evidence_gate_refusal(client):
    j, _ = _upload(client, _files("ct", "P0133", _series("P0133")), "ct")
    res = j["result"]
    assert res["status"] == _val("P0133")["status"] == "insufficient_evidence" and res["label"] is None and not res["scores"]
    assert res["cards"]["result"]["title"] == "Chưa đủ cơ sở để phân loại" and res["branches"][0]["chart"]["kind"] == "detection"
    assert res["details"]["evidence"]["passed"] is False


def test_ct_no_yolo_region(client):
    j, _ = _upload(client, _files("ct", "P0304", _series("P0304")), "ct")
    res = j["result"]
    b = res["branches"][0]
    assert res["status"] == "no_lesion_detected" and res["label"] is None and res["n_regions"] == 0 and not b["regions"]
    assert res["cards"]["result"]["title"] == "Không phát hiện vùng nghi ngờ trong ca chụp này" and b["chart"]["kind"] == "none"
    assert b["view"]["png"] and "không khẳng định gan bình thường" in " ".join(res["commentary"])
    assert "Normal" not in json.dumps(res) and "bình thường 9" not in json.dumps(res)


def test_mri_detection_only(client):
    j, _ = _upload(client, _files("mri", "MR-400851", ["T2WI", "C+A"]), "mri")
    res = j["result"]
    assert res["modality"] == "mri" and res["status"] == "detection_only" and res["label"] is None and not res["scores"]
    b = res["branches"][0]
    assert b["key"] == "mri" and b["regions"] and "label_short" not in b["regions"][0] and b["chart"]["kind"] == "detection"
    assert {v["name"] for v in b["volumes"]} == {"T2WI", "C+A"}


def test_ct_faster_rcnn_detection_only(client):
    """User-chosen comparison detector: Faster R-CNN -> Tumor Zoom, no classification, no Evidence Gate; regions equal a
    direct FrcnnDetector + analyze_volume run on the same volume (threshold 0.28 chosen on Validation)."""
    from livercancer.data.slices import load_volume_uint8
    from livercancer.detection.frcnn_detector import FrcnnDetector
    from livercancer.roi.pipeline import analyze_volume
    pid, s = "P0267", _series("P0267")[0]
    files = _files("ct", pid, [s])
    j, jid = _upload(client, files, "ct", detector="frcnn")
    assert j["state"] == "done", j
    res = j["result"]
    assert res["status"] in ("detection_only", "no_lesion_detected") and res["label"] is None and not res["scores"]
    assert res["details"]["detector"] == {"key": "frcnn", "name": "Faster R-CNN", "conf": 0.28, "main": False}
    assert res["details"]["strong_conf"] is None and res["details"]["evidence"] is None
    b = res["branches"][0]
    assert b["key"] == "ct" and b["title"] == "Faster R-CNN CT" and b["detector"] == "frcnn"
    assert all(r["confidence"] >= 0.28 for r in b["regions"]) and "label_short" not in (b["regions"] or [{}])[0]
    assert any("Faster R-CNN" in t for t in res["commentary"]) and any("đối chứng" in t for t in res["details"]["limitations"])
    rr = analyze_volume(load_volume_uint8(_paths()("ct", pid, s)), FrcnnDetector.from_config())
    assert res["n_regions"] == len(rr.tracks)
    assert [round(t.max_conf, 2) for t in rr.tracks][:3] == [round(r["confidence"], 2) for r in b["regions"][:3]]
    # default detector is still the locked YOLO11n pipeline
    j2, _ = _upload(client, files, "ct")
    assert j2["result"]["details"]["detector"]["key"] == "yolo11n" and j2["result"]["branches"][0]["title"] == "YOLO CT"


def test_faster_rcnn_refusals(client):
    r = client.post("/api/uploads", json={"files": [{"name": "a.nii.gz", "size": 10}]})
    uid = r.json()["upload_id"]
    r = client.post(f"/api/uploads/{uid}/analyze", data={"modality": "mri", "detector": "frcnn"})
    assert r.status_code == 400 and "chỉ được huấn luyện cho ảnh CT" in r.json()["detail"]
    r = client.post(f"/api/uploads/{uid}/analyze", data={"modality": "ct", "detector": "yolov8"})
    assert r.status_code == 400
    client.delete(f"/api/uploads/{uid}")
    assert client.get("/api/health").json()["comparison"]["frcnn"] in ("available", "ready")


def test_dicom_zip_identical_to_nifti_and_no_metadata(client):
    from livercancer.data.slices import load_volume_uint8
    from livercancer.intake.dicom_fixtures import FAKE_ID, FAKE_NAME, series_from_volume, zip_blobs
    ip = _paths()
    ser = ["C2", "P"]
    vols = [load_volume_uint8(ip("ct", "P0267", s)) for s in ser]
    blobs = []
    for i, v in enumerate(vols):
        blobs += series_from_volume(v, "CT", f"1.2.826.0.1.{i + 1}")
    j_d, _ = _upload(client, [("ca_benh.zip", zip_blobs(blobs))], "ct")
    j_n, _ = _upload(client, [(f.name, f.read_bytes()) for f in (ip("ct", "P0267", s) for s in ser)], "ct")
    rd, rn = j_d["result"], j_n["result"]
    assert rd["modality"] == "ct" and "DICOM" in rd["cards"]["modality"]["sub"]
    assert rd["status"] == rn["status"] and rd["label"] == rn["label"] and rd["n_regions"] == rn["n_regions"]
    if rd["scores"]:
        assert _same_scores(rd, {s["code"]: s["prob"] for s in rn["scores"]})
    txt = json.dumps(j_d)
    assert FAKE_NAME not in txt and FAKE_ID not in txt and "TEST^" not in txt


def _val_slice(ds, lesion=True, seed=0):
    import csv
    import random
    root = ROOT / "data" / "processed" / ds
    rows = [r for r in csv.DictReader(open(root / "slices.csv", encoding="utf-8")) if r["split"] == "val" and (r["kind"] == "pos") == lesion]
    r = random.Random(seed).choice(rows)
    return cv2.imread(str(root / "images" / "val" / f"{r['file']}.png"), cv2.IMREAD_GRAYSCALE)


@pytest.mark.parametrize("ds,mod", [("full/yolo_ct", "ct"), ("pilot/yolo_mri", "mri")])
def test_png_and_screenshot_with_chosen_modality(client, ds, mod):
    """PNG / JPG screenshot: analysed with the model the user chose; CT goes through Hybrid + Evidence Gate."""
    from livercancer.intake.synthetic import jpeg, render_screenshot
    g = _val_slice(ds)
    ok, png = cv2.imencode(".png", g)
    shot, _ = render_screenshot([g], seed=7, panels=1)
    ok, jpg = cv2.imencode(".jpg", jpeg(shot, 80))
    for name, data in (("lat.png", png.tobytes()), ("man_hinh.jpg", jpg.tobytes())):
        j, _ = _upload(client, [(name, data)], mod)
        assert j["state"] == "done", j
        res = j["result"]
        assert res["modality"] == mod and res["is_2d"] and res["cards"]["modality"]["sub"] == "Bạn đã chọn"
        assert res["details"]["preprocessing"]["prepared_png"]
        assert any("ảnh 2D" in t for t in res["commentary"])
        if mod == "ct":
            assert res["status"] in ("ok", "insufficient_evidence", "no_lesion_detected")
            assert res["details"]["evidence"] is not None or res["status"] == "no_lesion_detected"
        else:
            assert res["status"] in ("detection_only", "no_lesion_detected") and not res["scores"]


def _raw_upload(client, items, modality):
    r = client.post("/api/uploads", json={"files": [{"name": n, "size": len(d)} for n, d in items]})
    uid = r.json()["upload_id"]
    for i, (_, d) in enumerate(items):
        client.post(f"/api/uploads/{uid}/chunk?index={i}&offset=0", content=d, headers={"Content-Type": "application/octet-stream"})
    return client.post(f"/api/uploads/{uid}/analyze", data={"modality": modality})


@pytest.mark.parametrize("pick", ["mri", "ct"])
def test_conflict_asks_before_analysing(client, pick):
    """MRI NIfTI uploaded as CT: nothing is analysed until the user confirms. Choosing MRI = plain MRI analysis;
    keeping CT = CT pipeline, recorded as an override of the file's information."""
    items = _files("mri", "MR-400851", ["T2WI"])
    r = _raw_upload(client, items, "ct")
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    j = _wait(client, jid)
    assert j["state"] == "needs_confirm" and "result" not in j
    assert j["need"]["detected"] == "mri" and j["need"]["chosen"] == "ct" and "header NIfTI" in j["need"]["message"]
    r = client.post(f"/api/jobs/{jid}/confirm", data={"modality": pick})
    assert r.status_code == 200, r.text
    j = _wait(client, jid)
    assert j["state"] == "done", j
    res = j["result"]
    assert res["modality"] == pick
    if pick == "mri":
        assert res["cards"]["modality"]["sub"] == "Bạn đã chọn · khớp header NIfTI" and res["status"] in ("detection_only", "no_lesion_detected")
    else:
        assert res["cards"]["modality"]["sub"].startswith("Bạn đã xác nhận") and any("Bạn đã xác nhận phân tích như ảnh CT" in n for n in res["notes"])
    assert client.post(f"/api/jobs/{jid}/confirm", data={"modality": "ct"}).status_code == 409


def test_dicom_tag_conflict_and_cancel(client):
    """A CT DICOM series uploaded as MRI: asked (DICOM tag), and a cancelled job leaves nothing behind."""
    from livercancer.data.slices import load_volume_uint8
    from livercancer.intake.dicom_fixtures import series_from_volume, zip_blobs
    vol = load_volume_uint8(_paths()("ct", "P0267", "C2"))[:, :, 40:60]
    r = _raw_upload(client, [("ca.zip", zip_blobs(series_from_volume(vol, "CT", "1.2.826.0.7.1")))], "mri")
    jid = r.json()["job_id"]
    j = _wait(client, jid)
    assert j["state"] == "needs_confirm" and j["need"]["detected"] == "ct" and "DICOM" in j["need"]["message"]
    assert client.delete(f"/api/jobs/{jid}").json()["deleted"]
    assert client.get(f"/api/jobs/{jid}").status_code == 404


def test_refusals(client):
    import cv2 as _cv
    ok, blank = _cv.imencode(".png", np.zeros((128, 128), np.uint8))
    j, _ = _upload(client, [("den.png", blank.tobytes())])
    assert j["state"] == "error" and j["error"]                        # not a CT/MRI slice: refused, not guessed
    mri = _files("mri", "MR-400851", ["T2WI"])
    r = client.post("/api/uploads", json={"files": [{"name": mri[0][0], "size": len(mri[0][1])}]})
    uid = r.json()["upload_id"]
    client.post(f"/api/uploads/{uid}/chunk?index=0&offset=0", content=mri[0][1], headers={"Content-Type": "application/octet-stream"})
    r = client.post(f"/api/uploads/{uid}/analyze", data={"modality": "ct"})   # header says MRI, user chose CT
    assert r.status_code == 200 and _wait(client, r.json()["job_id"])["state"] == "needs_confirm"
    r = client.post("/api/uploads", json={"files": [{"name": "a.txt", "size": 5}]})
    uid = r.json()["upload_id"]
    client.post(f"/api/uploads/{uid}/chunk?index=0&offset=0", content=b"hello", headers={"Content-Type": "application/octet-stream"})
    assert client.post(f"/api/uploads/{uid}/analyze", data={"modality": "auto"}).status_code == 400


def test_chunked_upload_guards(client):
    r = client.post("/api/uploads", json={"files": [{"name": "a.png", "size": 100}]})
    uid = r.json()["upload_id"]
    assert client.post(f"/api/uploads/{uid}/chunk?index=0&offset=5", content=b"x").status_code == 409
    assert client.post(f"/api/uploads/{uid}/analyze", data={"modality": "auto"}).status_code == 400
    assert client.delete(f"/api/uploads/{uid}").json()["deleted"]
    assert client.post("/api/uploads", json={"files": [{"name": "a.nii.gz", "size": 10 << 30}]}).status_code == 413


def test_public_api_no_access_code(client):
    """The analysis API is public: no code, no login; the old code endpoint and header are gone."""
    h = client.get("/api/health").json()
    assert "requires_token" not in h and h["limits"]["analysis_timeout_s"] > 0
    assert client.get("/api/jobs/x").status_code == 404                     # not 401
    assert client.get("/api/session").status_code == 404
    r = client.post("/api/uploads", json={"files": [{"name": "a.png", "size": 100}]})
    assert r.status_code == 200
    assert client.delete(f"/api/uploads/{r.json()['upload_id']}").json()["deleted"]
    pre = client.options("/api/uploads", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
                                                  "Access-Control-Request-Headers": "content-type"})
    assert pre.status_code == 200 and "x-access-token" not in pre.headers.get("access-control-allow-headers", "").lower()


def test_rate_limit_per_client(client, monkeypatch):
    """Too many API requests from one client (CF-Connecting-IP) -> 429; other clients are not affected."""
    import server.main as m
    monkeypatch.setattr(m, "rate_limit", m.SlidingLimit(3, 60))
    a, b = {"CF-Connecting-IP": "203.0.113.7"}, {"CF-Connecting-IP": "203.0.113.8"}
    assert [client.get("/api/health", headers=a).status_code for _ in range(3)] == [200, 200, 200]
    r = client.get("/api/health", headers=a)
    assert r.status_code == 429 and "quá nhiều yêu cầu" in r.json()["detail"]
    assert client.get("/api/health", headers=b).status_code == 200


def test_job_limits_per_client(client, monkeypatch):
    """One client: running analyses at once and analyses per hour are capped (checked before anything is queued)."""
    import server.main as m
    png = cv2.imencode(".png", np.zeros((64, 64), np.uint8))[1].tobytes()
    monkeypatch.setattr(m.jobs, "active_for", lambda c: 2)
    r = client.post("/api/analyze", files=[("files", ("a.png", png, "image/png"))], data={"modality": "ct"})
    assert r.status_code == 429 and "đang có ảnh khác" in r.json()["detail"]
    monkeypatch.setattr(m.jobs, "active_for", lambda c: 0)
    monkeypatch.setattr(m, "job_limit", m.SlidingLimit(1, 3600))
    m._check_new_job("203.0.113.9")
    with pytest.raises(m.HTTPException) as e:
        m._check_new_job("203.0.113.9")
    assert e.value.status_code == 429
    m._check_new_job("203.0.113.10")


def test_upload_limits_per_client(client, monkeypatch):
    import server.main as m
    monkeypatch.setattr(m, "upload_limit", m.SlidingLimit(2, 3600))
    h = {"CF-Connecting-IP": "203.0.113.20"}
    body = {"files": [{"name": "a.png", "size": 100}]}
    ids = [client.post("/api/uploads", json=body, headers=h).json()["upload_id"] for _ in range(2)]
    assert client.post("/api/uploads", json=body, headers=h).status_code == 429          # 2 unfinished uploads per client
    for u in ids:
        client.delete(f"/api/uploads/{u}")
    assert client.post("/api/uploads", json=body, headers=h).status_code == 429          # 2 uploads per hour
    r = client.post("/api/uploads", json=body, headers={"CF-Connecting-IP": "203.0.113.21"})
    assert r.status_code == 200
    assert client.delete(f"/api/uploads/{r.json()['upload_id']}").json()["deleted"]


def test_job_timeout_stops_analysis(tmp_path):
    """An analysis running longer than the limit stops at its next progress report with a plain error, even when the
    pipeline turns the exception into an error status (as the locked CT pipeline does)."""
    from server.jobs import JobManager

    class Slow:
        def load_inputs(self, files, progress):
            time.sleep(0.2)
            try:
                progress("format", "x", None)
            except Exception as e:  # noqa: BLE001 - swallowed like LiverCancerPipeline.run
                raise RuntimeError("pipeline error") from e

    jm = JobManager(Slow(), 3, 60, timeout_s=0)
    job = jm.submit([("a", tmp_path / "a.png", "image")], tmp_path, None, "ct", client="x")
    t0 = time.time()
    while job.state in ("queued", "running") and time.time() - t0 < 10:
        time.sleep(0.05)
    assert job.state == "error" and "vượt quá thời gian" in job.error
    assert jm.active_for("x") == 0


def test_open_upload_limit(client, monkeypatch):
    import server.main as m
    from server.settings import Settings
    monkeypatch.setattr(m, "settings", Settings(max_open_uploads=1))
    r = client.post("/api/uploads", json={"files": [{"name": "a.png", "size": 100}]})
    assert r.status_code == 200
    assert client.post("/api/uploads", json={"files": [{"name": "b.png", "size": 100}]}).status_code == 429
    assert client.delete(f"/api/uploads/{r.json()['upload_id']}").json()["deleted"]


def test_case_summary_3d_matches_viewer(client):
    """3D case summary = real counts, and every listed suspicious slice has exactly those boxes in the slice viewer."""
    j, jid = _upload(client, _files("ct", "P0267", _series("P0267")), "ct", dims="3d")
    res = j["result"]
    cs, b = res["case"], res["branches"][0]
    assert cs["is_3d"] and len(cs["volumes"]) == len(_series("P0267"))
    assert res["details"]["strong_conf"] == 0.4                               # 4 phases: multi-phase Evidence Gate rule
    assert len(cs["regions3d"]) == res["n_regions"]                           # every region of every phase, 3D summary
    assert any("trong ca chụp CT 3D gồm 4 pha" in t for t in res["commentary"])
    assert any("không phải xác suất người bệnh mắc ung thư" in t for t in res["commentary"])
    assert cs["total_slices"] == sum(v["n_slices"] for v in b["volumes"]) == sum(v["n_slices"] for v in cs["volumes"])
    assert cs["duplicate_slices"] == 0 and cs["yolo_slices"] == cs["total_slices"]        # Validation: no exact duplicate
    assert all(v["source"] == "NIfTI" and v["spacing"] and v["width"] == 512 for v in cs["volumes"])
    sus = cs["suspicious_slices"]
    assert cs["slices_with_regions"] == len(sus) > 0 and abs(cs["top_slice"]["conf"] - b["max_conf"]) < 1e-3
    listed = {(x["volume"], x["k"]): x for x in sus}
    vol = cs["top_slice"]["volume"]
    n = next(v["n_slices"] for v in cs["volumes"] if v["name"] == vol)
    for k in sorted({cs["top_slice"]["k"], 0, n // 2, n - 1} | {x["k"] for x in sus if x["volume"] == vol}):
        v = client.get(f"/api/jobs/{jid}/slice", params={"branch": "ct", "volume": vol, "k": k}).json()
        x = listed.get((vol, k))
        assert v["slice"] == k and len(v["boxes"]) == (x["n_boxes"] if x else 0), (k, v["boxes"], x)
        if x:
            assert abs(max(bb["conf"] for bb in v["boxes"]) - x["conf"]) < 1e-3
    t = client.get(f"/api/jobs/{jid}/thumb", params={"branch": "ct", "volume": vol, "k": cs["top_slice"]["k"]})
    assert t.status_code == 200 and t.headers["content-type"] == "image/jpeg" and t.content[:2] == b"\xff\xd8"
    assert client.get(f"/api/jobs/{jid}/thumb", params={"branch": "ct", "volume": "x", "k": 0}).status_code == 404


def test_dims_must_match_files(client):
    """The 2D / 3D choice of the website is checked on the server, before anything is queued."""
    png = cv2.imencode(".png", np.zeros((64, 64), np.uint8))[1].tobytes()
    r = client.post("/api/analyze", files=[("files", ("a.png", png, "image/png"))], data={"modality": "ct", "dims": "3d"})
    assert r.status_code == 400 and "Phân tích ảnh 2D" in r.json()["detail"]
    name, data = _files("mri", "MR-400851", ["T2WI"])[0]
    r = client.post("/api/analyze", files=[("files", (name, data, "application/gzip"))], data={"modality": "mri", "dims": "2d"})
    assert r.status_code == 400 and "Phân tích ca 3D" in r.json()["detail"]


def test_cancel_running_job(tmp_path):
    """Deleting a running job stops it at its next progress report (the website's "Hủy phân tích")."""
    from server.jobs import JobManager

    class Slow:
        def load_inputs(self, files, progress):
            for _ in range(300):
                time.sleep(0.02)
                progress("format", "x", None)
            raise AssertionError("not cancelled")

    jm = JobManager(Slow(), 3, 60, timeout_s=60)
    job = jm.submit([("a", tmp_path / "a.png", "image")], tmp_path, None, "ct", client="y")
    t0 = time.time()
    while job.state != "running" and time.time() - t0 < 5:
        time.sleep(0.01)
    assert jm.delete(job.id)
    while job.state == "running" and time.time() - t0 < 10:
        time.sleep(0.02)
    assert job.state == "error" and job.error == "Đã hủy phân tích." and jm.active_for("y") == 0


def _check_regions3d(client, jid, res, branch):
    """Every 3D region is coherent, and on EVERY slice the viewer boxes are exactly the boxes of the 3D regions."""
    cs = res["case"]
    r3 = cs["regions3d"]
    assert len(r3) == res["n_regions"] and sorted(r["num"] for r in r3) == list(range(1, len(r3) + 1))
    sp = {v["name"]: v["spacing_xyz"] for v in cs["volumes"]}
    for r in r3:
        ks = [bx[0] for bx in r["boxes"]]
        assert ks == sorted(set(ks)) and (r["k_first"], r["k_last"], r["n_slices"]) == (ks[0], ks[-1], len(ks))
        assert r["gap_slices"] == ks[-1] - ks[0] + 1 - len(ks)
        bb = r["bbox_vox"]
        assert abs(bb["x_min"] - min(b[1] for b in r["boxes"])) < 0.11 and abs(bb["y_max"] - max(b[4] for b in r["boxes"])) < 0.11
        assert (bb["z_min"], bb["z_max"]) == (ks[0], ks[-1]) and abs(max(b[5] for b in r["boxes"]) - r["max_conf"]) < 1e-3
        x, y, z = sp[r["volume"]]
        assert r["size_mm"] == pytest.approx([(bb["x_max"] - bb["x_min"]) * x, (bb["y_max"] - bb["y_min"]) * y, (ks[-1] - ks[0] + 1) * z], abs=0.2)
    for v in cs["volumes"]:
        for k in range(v["n_slices"]):
            got = client.get(f"/api/jobs/{jid}/slice", params={"branch": branch, "volume": v["name"], "k": k}).json()["boxes"]
            exp = sorted((bx[1], r["id"]) for r in r3 if r["volume"] == v["name"] for bx in r["boxes"] if bx[0] == k)
            assert sorted((round(b["box"][0], 1), b["region"]) for b in got) == exp, (v["name"], k)
    return r3


def test_regions3d_frames_video_ct(client, tmp_path):
    """Single-phase Validation CT (92 slices, 1 mm): 3D regions, JPEG strips for playback, MP4 (H.264) of every slice."""
    import imageio_ffmpeg
    j, jid = _upload(client, _files("ct", "P0053", ["C1"]), "ct", dims="3d")
    res = j["result"]
    vol = res["case"]["volumes"][0]
    assert vol["n_slices"] == 92 and vol["spacing_xyz"] == pytest.approx([1.0, 1.0, 1.0])
    r3 = _check_regions3d(client, jid, res, "ct")
    assert r3 and all(r["volume"] == vol["name"] for r in r3)
    fr = client.get(f"/api/jobs/{jid}/frames", params={"branch": "ct", "volume": vol["name"], "start": 0, "count": 8})
    img = cv2.imdecode(np.frombuffer(fr.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert fr.status_code == 200 and img.shape == (8 * 512, 512) and fr.headers["x-frame-count"] == "8"
    assert "max-age" in fr.headers["cache-control"]
    fr = client.get(f"/api/jobs/{jid}/frames", params={"branch": "ct", "volume": vol["name"], "start": 89, "count": 8})
    assert cv2.imdecode(np.frombuffer(fr.content, np.uint8), cv2.IMREAD_GRAYSCALE).shape == (3 * 512, 512)
    assert client.get(f"/api/jobs/{jid}/frames", params={"branch": "ct", "volume": "../../x", "start": 0}).status_code == 404
    v = client.post(f"/api/jobs/{jid}/video", json={"volume": vol["name"], "fps": 8})
    assert v.status_code == 200, v.text
    vid = v.json()["id"]
    for _ in range(600):
        st = client.get(f"/api/videos/{vid}").json()
        if st["state"] not in ("queued", "running"):
            break
        time.sleep(0.3)
    assert st["state"] == "done" and st["done"] == st["total"] == 92, st
    f = client.get(f"/api/videos/{vid}/file")
    assert f.status_code == 200 and f.headers["content-type"] == "video/mp4" and f.content[4:8] == b"ftyp"
    out = tmp_path / "v.mp4"
    out.write_bytes(f.content)
    rd = imageio_ffmpeg.read_frames(str(out))
    meta = next(rd)
    assert meta["codec"] == "h264" and sum(1 for _ in rd) == 92                 # one frame per slice, none lost
    assert client.delete(f"/api/videos/{vid}").json()["deleted"]
    assert client.get(f"/api/videos/{vid}").status_code == 404
    # cancel while queued / running
    v2 = client.post(f"/api/jobs/{jid}/video", json={"volume": vol["name"]}).json()
    assert client.delete(f"/api/videos/{v2['id']}").json()["deleted"]


def test_regions3d_mri_sequences(client):
    """MRI: two sequences with different size and spacing stay separate volumes; no CT classification."""
    j, jid = _upload(client, _files("mri", "MR-400851", ["T2WI", "DWI"]), "mri", dims="3d")
    res = j["result"]
    assert res["status"] in ("detection_only", "no_lesion_detected") and not res["scores"]
    vols = {v["name"]: v for v in res["case"]["volumes"]}
    assert len(vols) == 2 and {(v["width"], v["height"]) for v in vols.values()} == {(512, 512), (256, 256)}
    assert {tuple(round(x, 2) for x in v["spacing_xyz"]) for v in vols.values()} == {(0.74, 0.74, 8.0), (1.48, 1.48, 8.0)}
    _check_regions3d(client, jid, res, "mri")


def test_header_recognition_all_sequences():
    """CT / MRI recognised from the header alone (neutral file name) for every MRI sequence and CT files."""
    import glob
    from livercancer.config import load_paths
    from server.inspect_file import inspect_head
    p = load_paths()
    mri = sorted(glob.glob(os.path.join(p["mri"]["images"], "*.nii.gz")))[:8 * 3]
    ct = sorted(glob.glob(os.path.join(p["ct"]["data_dir"], "ct_files", "*.nii.gz")))[::60]
    if not mri or not ct:
        pytest.skip("source dataset not available")
    for files, want in ((mri, "mri"), (ct, "ct")):
        for f in files:
            with open(f, "rb") as fh:
                head = fh.read(1 << 18)
            info = inspect_head(0, "anh.nii.gz", head, os.path.getsize(f), 300 << 20)
            assert info.ok and info.modality == want, (os.path.basename(f), info)


def test_no_upload_left_on_disk(client):
    time.sleep(0.5)
    left = [d for d in Path(tempfile.gettempdir()).glob("lcai_*") if d.is_dir() and any(d.iterdir())]
    assert not left, left
