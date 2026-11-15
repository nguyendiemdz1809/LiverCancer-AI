"""LiverCancer AI web API (FastAPI).

Run (Windows, from the project folder):
    .venv\\Scripts\\python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
Endpoints (all under /api):
    GET    /api/health            models ready?, limits, queue
    POST   /api/inspect           first bytes of each file -> CT / MRI recognition, validation (nothing stored)
    POST   /api/uploads           start a chunked upload (Cloudflare: <= 100 MB per request; CT files reach 160 MB)
    POST   /api/uploads/{id}/chunk?index=&offset=   raw bytes of one file part (<= 32 MB)
    POST   /api/uploads/{id}/analyze                 modality (ct | mri | auto), dims (2d | 3d | "") -> job id
    POST   /api/analyze           files + modality in one request (small uploads, tests)
    GET    /api/jobs/{id}         progress / Vietnamese result (state needs_confirm: the file says the other modality)
    POST   /api/jobs/{id}/confirm              modality ct | mri chosen after a contradiction
    GET    /api/jobs/{id}/slice?branch=&volume=&k=   one slice of a 3D input with its YOLO boxes
    GET    /api/jobs/{id}/thumb?branch=&volume=&k=   small JPEG of one slice with its boxes (thumbnail strip)
    GET    /api/jobs/{id}/frames?branch=&volume=&start=&count=   <= 16 consecutive slices in one JPEG (3D scan playback)
    POST   /api/jobs/{id}/video   volume, fps -> MP4 export job (H.264) of the scan with the YOLO boxes of every slice
    GET    /api/videos/{vid}      export progress;  GET /api/videos/{vid}/file  the MP4;  DELETE /api/videos/{vid}  cancel
    DELETE /api/jobs/{id}         cancel a running analysis / delete a result immediately
The built React frontend (web/dist) is served at / when present, so one local address (or one tunnel) is enough.

The analysis endpoints are PUBLIC (no access code, no login). Protection instead: per-client (IP) request rate,
uploads and analyses per hour, one running analysis per client, global queue, open-upload cap, file size / count /
format checks on the server, a time limit per analysis, CORS limited to the website. There is no admin endpoint.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from server.inspect_file import decide_modality, display_names, inspect_head  # noqa: E402
from server.jobs import JobManager  # noqa: E402
from server.service import AnalysisService  # noqa: E402
from server.settings import get_settings  # noqa: E402
from server.video import VideoManager  # noqa: E402

HEAD_BYTES = 1 << 20          # bytes of each file read to validate its header
CHUNK_MB = 32                 # upload part size, well below the 100 MB request limit of Cloudflare
MAX_DICOM_FILES = 2000        # one DICOM file per slice
MAX_TOTAL_BYTES = 1536 * 2 ** 20
INSPECT_MAX_BYTES = 16 << 20  # /api/inspect only receives the first 256 KB of each file

settings = get_settings()
service = AnalysisService(settings.device, settings.dedup_exact)
jobs = JobManager(service, settings.max_queue, settings.result_ttl_s, settings.job_timeout_s)
videos = VideoManager(max_queue=2, ttl_s=settings.result_ttl_s)


@asynccontextmanager
async def lifespan(_app):
    tmp = Path(settings.tmp_dir or tempfile.gettempdir())
    for d in tmp.glob("lcai_*"):           # leftovers of an interrupted run: never kept
        if d.is_dir() and time.time() - d.stat().st_mtime > 3600:
            shutil.rmtree(d, ignore_errors=True)
    threading.Thread(target=service.load, daemon=True, name="model-loader").start()
    yield


app = FastAPI(title="LiverCancer AI API", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins,
                   allow_origin_regex=settings.allowed_origin_regex or None, allow_methods=["GET", "POST", "DELETE"],
                   allow_headers=["Content-Type"], max_age=600)


def _err(status: int, msg: str, code: str = "error"):
    return JSONResponse({"detail": msg, "code": code}, status_code=status)


@app.exception_handler(HTTPException)
async def http_error(_req: Request, exc: HTTPException):
    return _err(exc.status_code, str(exc.detail), getattr(exc, "code", "error"))


class SlidingLimit:
    """At most `n` events per client in the last `window_s` seconds (in memory; resets when the server restarts)."""

    def __init__(self, n: int, window_s: float):
        self.n, self.window = n, window_s
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.n:
                return False
            q.append(now)
            if len(self.hits) > 10000:                 # forget idle clients
                for k in [k for k, v in self.hits.items() if not v][:5000]:
                    self.hits.pop(k, None)
            return True


rate_limit = SlidingLimit(settings.rate_per_min, 60)
upload_limit = SlidingLimit(settings.uploads_per_hour, 3600)
job_limit = SlidingLimit(settings.jobs_per_hour, 3600)
video_limit = SlidingLimit(6, 3600)                    # video exports per client per hour
BUSY_VI = "Bạn đã gửi quá nhiều yêu cầu trong thời gian ngắn. Vui lòng thử lại sau ít phút."


def _client_ip(request: Request) -> str:
    # Behind Cloudflare Tunnel every request comes from 127.0.0.1; Cloudflare sets CF-Connecting-IP itself.
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")


@app.middleware("http")
async def guard(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/") and request.method != "OPTIONS":
        if not rate_limit.allow(_client_ip(request)):
            return _err(429, BUSY_VI, "rate_limited")
        if request.method == "POST":
            cl = request.headers.get("content-length")
            cap = INSPECT_MAX_BYTES if path == "/api/inspect" else settings.max_files * settings.max_file_bytes + (1 << 20)
            if cl and cl.isdigit() and int(cl) > cap:
                return _err(413, f"Dữ liệu tải lên quá lớn (tối đa {settings.max_files} tệp, mỗi tệp {settings.max_file_mb} MB).")
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Frame-Options"] = "DENY"
    if path.startswith("/api/") and "Cache-Control" not in resp.headers:
        resp.headers["Cache-Control"] = "no-store"
    return resp


def _check_new_job(client: str) -> None:
    """Before queueing an analysis: global queue, one running analysis per client, analyses per hour."""
    if jobs.waiting() >= settings.max_queue:
        raise HTTPException(429, "Máy chủ đang bận phân tích ảnh khác. Vui lòng thử lại sau ít phút.")
    if jobs.active_for(client) >= settings.active_per_client:
        raise HTTPException(429, "Mạng của bạn đang có ảnh khác được phân tích. Vui lòng chờ kết quả rồi gửi ảnh tiếp theo.")
    if not job_limit.allow(client):
        raise HTTPException(429, "Bạn đã gửi quá nhiều lượt phân tích trong một giờ. Vui lòng thử lại sau.")


@app.get("/api/health")
def health():
    lock = ROOT / "reports" / "final_test" / "locked_config.json"
    commit = json.loads(lock.read_text(encoding="utf-8"))["git"]["commit"][:7] if lock.exists() else ""
    return {"status": "ok", "models": service.state, "device": service.device_used,
            "limits": {"max_file_mb": settings.max_file_mb, "max_files": settings.max_files, "max_dicom_files": MAX_DICOM_FILES,
                       "analysis_timeout_s": settings.job_timeout_s},
            "locked_config": commit, "modality_model": service.modality_model is not None,
            "queue": jobs.waiting()}


def _inspect(names: list[str], heads: list[bytes], sizes: list[int]):
    infos = [inspect_head(i, n, h, s, settings.max_file_bytes) for i, (n, h, s) in enumerate(zip(names, heads, sizes))]
    mod, reason = decide_modality(infos)
    return infos, mod, reason


@app.post("/api/inspect")
async def inspect(files: list[UploadFile] = File(...), sizes: str = Form("[]")):
    if len(files) > settings.max_files:
        raise HTTPException(400, f"Tối đa {settings.max_files} tệp cho một lần phân tích.")
    try:
        sz = [int(x) for x in json.loads(sizes)]
    except Exception:  # noqa: BLE001
        sz = []
    heads = [await f.read(HEAD_BYTES) for f in files]
    sz = sz if len(sz) == len(files) else [len(h) for h in heads]
    infos, mod, reason = _inspect([f.filename or "" for f in files], heads, sz)
    names = display_names(infos, mod)
    return {"files": [{**i.as_dict(), "display_name": n} for i, n in zip(infos, names)], "modality": mod, "reason": reason}


DIMS_2D_VI = ("Đây là dữ liệu 3D (NIfTI, chuỗi DICOM hoặc .zip). Hãy chọn \"Phân tích ca 3D\" rồi tải lại.")
DIMS_3D_VI = ("Ảnh PNG/JPG hoặc một tệp DICOM đơn là ảnh 2D (một lát cắt). Hãy chọn \"Phân tích ảnh 2D\", "
              "hoặc tải tệp NIfTI / DICOM .zip cho ca 3D.")


def _check_dims(infos, dims: str) -> None:
    """The 2D / 3D tab chosen on the website must match the files (checked on the server, never trusted)."""
    if dims not in ("2d", "3d"):
        return
    kinds = {i.kind for i in infos}
    is_2d = kinds == {"image"} or (kinds == {"dicom"} and len(infos) == 1)
    if dims == "2d" and not is_2d:
        raise HTTPException(400, DIMS_2D_VI)
    if dims == "3d" and is_2d:
        raise HTTPException(400, DIMS_3D_VI)


def _submit(tmp: Path, paths: list[Path], orig_names: list[str], heads: list[bytes], sizes: list[int], modality: str,
            client: str, dims: str = "") -> dict:
    """Server-side validation of complete uploads (never trusts the browser), then queue the analysis."""
    infos, mod, reason = _inspect(orig_names, heads, sizes)
    bad = [i for i in infos if not i.ok]
    if bad:
        raise HTTPException(400, f"Tệp {bad[0].index + 1}: {bad[0].error_vi}")
    if mod == "mixed":
        raise HTTPException(400, reason)
    _check_dims(infos, dims)
    # a header contradicting the chosen modality is NOT refused here: the job stops before any model and asks the user
    kinds = {i.kind for i in infos}
    if kinds == {"image"} and len(infos) > 1:
        raise HTTPException(400, "Với ảnh 2D (PNG/JPG), hãy tải từng ảnh một.")
    if kinds == {"nifti"} and mod == "ct" and len(infos) > 4:
        raise HTTPException(400, "Một ca CT có tối đa 4 pha (C1, C2, C3, P).")
    if kinds == {"nifti"} and len(infos) > settings.max_files:
        raise HTTPException(400, f"Tối đa {settings.max_files} tệp NIfTI cho một lần phân tích.")
    names = display_names(infos, mod if mod in ("ct", "mri") else "unknown")
    job = jobs.submit([(n, p, i.kind) for n, p, i in zip(names, paths, infos)], tmp,
                      header_modality=mod if mod in ("ct", "mri") else None,
                      requested=None if modality == "auto" else modality, client=client)
    return {"job_id": job.id, "files": [{"display_name": n, "n_slices": i.n_slices} for n, i in zip(names, infos)]}


def _ext(name: str) -> str:
    low = name.lower()
    return ".nii.gz" if low.endswith(".nii.gz") else Path(low).suffix


@app.post("/api/analyze")
async def analyze(request: Request, files: list[UploadFile] = File(...), modality: str = Form("auto"), dims: str = Form("")):
    if modality not in ("auto", "ct", "mri"):
        raise HTTPException(400, "Loại ảnh không hợp lệ.")
    if not files or len(files) > settings.max_files:
        raise HTTPException(400, f"Hãy chọn từ 1 đến {settings.max_files} tệp ảnh của cùng một bệnh nhân.")
    client = _client_ip(request)
    _check_new_job(client)
    tmp = Path(tempfile.mkdtemp(prefix="lcai_", dir=settings.tmp_dir or None))
    try:
        paths, heads, sizes = [], [], []
        for i, f in enumerate(files):
            p = tmp / (str(i) + _ext(f.filename or ""))                       # the original file name is never written anywhere
            n = 0
            with open(p, "wb") as out:
                while chunk := await f.read(1 << 20):
                    n += len(chunk)
                    if n > settings.max_file_bytes:
                        raise HTTPException(413, f"Tệp {i + 1} vượt quá {settings.max_file_mb} MB.")
                    out.write(chunk)
            with open(p, "rb") as fh:
                heads.append(fh.read(HEAD_BYTES))
            paths.append(p)
            sizes.append(n)
        return _submit(tmp, paths, [f.filename or "" for f in files], heads, sizes, modality, client, dims)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


# ----------------------------------------------------------------------------- chunked upload (Cloudflare: <= 100 MB per request)
UPLOAD_TTL_S = 1800
uploads: dict[str, dict] = {}
uploads_lock = threading.Lock()


def _expire_uploads() -> None:
    now = time.time()
    with uploads_lock:
        old = [k for k, u in uploads.items() if now - u["created"] > UPLOAD_TTL_S]
        for k in old:
            shutil.rmtree(uploads.pop(k)["tmp"], ignore_errors=True)


class UploadInit(BaseModel):
    files: list[dict]          # [{"name": str, "size": int}]


@app.post("/api/uploads")
def upload_init(body: UploadInit, request: Request):
    _expire_uploads()
    fs = body.files
    dicom_like = all(_ext(str(f.get("name", ""))) in (".dcm", ".dicom", ".ima", "") for f in fs)
    limit = MAX_DICOM_FILES if dicom_like else settings.max_files      # a DICOM series = one file per slice
    if not fs or len(fs) > limit:
        raise HTTPException(400, f"Hãy chọn từ 1 đến {settings.max_files} tệp ảnh (hoặc tối đa {MAX_DICOM_FILES} tệp DICOM) của cùng một bệnh nhân.")
    if sum(int(f.get("size") or 0) for f in fs) > MAX_TOTAL_BYTES:
        raise HTTPException(413, "Tổng dung lượng tải lên quá lớn (tối đa 1,5 GB).")
    for i, f in enumerate(fs):
        if not isinstance(f.get("size"), int) or not 0 < f["size"] <= settings.max_file_bytes:
            raise HTTPException(413, f"Tệp {i + 1} vượt quá {settings.max_file_mb} MB hoặc rỗng.")
    if jobs.waiting() >= settings.max_queue:
        raise HTTPException(429, "Máy chủ đang bận phân tích ảnh khác. Vui lòng thử lại sau ít phút.")
    client = _client_ip(request)
    with uploads_lock:
        n_open = len(uploads)
        n_mine = sum(u["client"] == client for u in uploads.values())
    if n_open >= settings.max_open_uploads or n_mine >= 2:
        raise HTTPException(429, "Đang có quá nhiều lượt tải ảnh chưa hoàn tất. Vui lòng thử lại sau ít phút.")
    if not upload_limit.allow(client):
        raise HTTPException(429, "Bạn đã tải lên quá nhiều lần trong một giờ. Vui lòng thử lại sau.")
    tmp = Path(tempfile.mkdtemp(prefix="lcai_", dir=settings.tmp_dir or None))
    uid = uuid.uuid4().hex
    with uploads_lock:
        uploads[uid] = {"tmp": tmp, "created": time.time(), "client": client, "names": [str(f.get("name", "")) for f in fs],
                        "sizes": [f["size"] for f in fs], "received": [0] * len(fs)}
    return {"upload_id": uid, "chunk_mb": CHUNK_MB}


@app.post("/api/uploads/{uid}/chunk")
async def upload_chunk(uid: str, request: Request, index: int, offset: int):
    with uploads_lock:
        u = uploads.get(uid)
    if not u:
        raise HTTPException(404, "Phiên tải lên đã hết hạn. Vui lòng chọn lại ảnh.")
    if not 0 <= index < len(u["sizes"]) or offset != u["received"][index]:
        raise HTTPException(409, "Dữ liệu tải lên không liên tục. Vui lòng thử lại.")
    p = u["tmp"] / (str(index) + _ext(u["names"][index]))
    n = 0
    with open(p, "ab") as out:
        async for chunk in request.stream():
            n += len(chunk)
            if n > (CHUNK_MB << 20) or offset + n > u["sizes"][index]:
                raise HTTPException(413, "Phần dữ liệu tải lên vượt quá kích thước đã khai báo.")
            out.write(chunk)
    u["received"][index] = offset + n
    return {"received": u["received"][index]}


@app.post("/api/uploads/{uid}/analyze")
def upload_analyze(uid: str, request: Request, modality: str = Form("auto"), dims: str = Form("")):
    if modality not in ("auto", "ct", "mri"):
        raise HTTPException(400, "Loại ảnh không hợp lệ.")
    with uploads_lock:
        u = uploads.get(uid)
    if not u:
        raise HTTPException(404, "Phiên tải lên đã hết hạn. Vui lòng chọn lại ảnh.")
    if u["received"] != u["sizes"]:
        raise HTTPException(400, "Ảnh chưa được tải lên đầy đủ.")
    client = _client_ip(request)
    _check_new_job(client)
    paths = [u["tmp"] / (str(i) + _ext(n)) for i, n in enumerate(u["names"])]
    heads = []
    for p in paths:
        with open(p, "rb") as fh:
            heads.append(fh.read(HEAD_BYTES))
    try:
        out = _submit(u["tmp"], paths, u["names"], heads, u["sizes"], modality, client, dims)
    except HTTPException:
        with uploads_lock:
            uploads.pop(uid, None)
        shutil.rmtree(u["tmp"], ignore_errors=True)
        raise
    with uploads_lock:
        uploads.pop(uid, None)                              # the job now owns (and deletes) the folder
    return out


@app.delete("/api/uploads/{uid}")
def upload_delete(uid: str):
    with uploads_lock:
        u = uploads.pop(uid, None)
    if u:
        shutil.rmtree(u["tmp"], ignore_errors=True)
    return {"deleted": u is not None}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job, pos = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Không tìm thấy kết quả (kết quả chỉ được giữ tạm thời và có thể đã bị xóa).")
    return job.public(pos)


@app.post("/api/jobs/{job_id}/confirm")
def job_confirm(job_id: str, request: Request, modality: str = Form(...)):
    if modality not in ("ct", "mri"):
        raise HTTPException(400, "Loại ảnh không hợp lệ.")
    if jobs.waiting() >= settings.max_queue or jobs.active_for(_client_ip(request)) >= settings.active_per_client:
        raise HTTPException(429, "Máy chủ đang bận phân tích ảnh khác. Vui lòng thử lại sau ít phút.")
    job = jobs.confirm(job_id, modality)
    if not job:
        raise HTTPException(409, "Phân tích này không chờ xác nhận (có thể đã hết hạn). Vui lòng tải ảnh lên lại.")
    return job.public()


@app.get("/api/jobs/{job_id}/slice")
def job_slice(job_id: str, volume: str, k: int, branch: str = ""):
    job, _ = jobs.get(job_id)
    br = None
    if job and job.state == "done" and job.viewer and job.result:
        br = next((b for b in job.result["branches"] if b["key"] == (branch or job.result["branches"][0]["key"])), None)
    if br is None or br["key"] not in job.viewer or volume not in job.viewer[br["key"]]:
        raise HTTPException(404, "Không còn dữ liệu lát cắt cho kết quả này (chỉ giữ tạm cho 2 kết quả gần nhất).")
    return service.slice_view(job.viewer[br["key"]], volume, k, br["first_ids"])


def _viewer_volume(job_id: str, volume: str, branch: str = ""):
    """(job, RoiResult) of a finished 3D result still kept for the viewer, else 404 (never a file path)."""
    job, _ = jobs.get(job_id)
    br = None
    if job and job.state == "done" and job.viewer and job.result:
        br = next((b for b in job.result["branches"] if b["key"] == (branch or job.result["branches"][0]["key"])), None)
    if br is None or br["key"] not in job.viewer or volume not in job.viewer[br["key"]]:
        raise HTTPException(404, "Không còn dữ liệu lát cắt cho kết quả này (chỉ giữ tạm cho 2 kết quả gần nhất).")
    return job, br, job.viewer[br["key"]][volume]


@app.get("/api/jobs/{job_id}/frames")
def job_frames(job_id: str, volume: str, start: int = 0, count: int = 8, branch: str = ""):
    _, _, rr = _viewer_volume(job_id, volume, branch)
    count = max(1, min(16, count))
    data = service.frames_sprite(rr, start, count)
    n = min(count, rr.n_slices - max(0, min(start, rr.n_slices - 1)))
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=900", "X-Frame-Count": str(n)})


class VideoReq(BaseModel):
    volume: str
    fps: int = 8
    branch: str = ""


@app.post("/api/jobs/{job_id}/video")
def video_start(job_id: str, body: VideoReq, request: Request):
    job, br, rr = _viewer_volume(job_id, body.volume, body.branch)
    if rr.n_slices < 2:
        raise HTTPException(400, "Video chỉ dành cho ca 3D (nhiều lát cắt).")
    client = _client_ip(request)
    if videos.waiting() >= 2:
        raise HTTPException(429, "Máy chủ đang tạo video khác. Vui lòng thử lại sau ít phút.")
    if not video_limit.allow(client):
        raise HTTPException(429, "Bạn đã tạo quá nhiều video trong một giờ. Vui lòng thử lại sau.")
    res = job.result
    mod = "CT" if br["key"] == "ct" else "MRI"
    nums = {}
    for r in res.get("case", {}).get("regions3d", []):
        if r["volume"] == body.volume:
            nums[r["id"] - br["first_ids"].get(body.volume, 0)] = r["num"]
    title = f"LiverCancer AI · {mod} 3D · {body.volume}"
    v = videos.submit(client, title, mod, body.volume, rr, nums, max(2, min(16, int(body.fps))))
    return v.public()


@app.get("/api/videos/{vid}")
def video_status(vid: str):
    v = videos.get(vid)
    if not v:
        raise HTTPException(404, "Không tìm thấy video (có thể đã hết hạn).")
    return v.public()


@app.get("/api/videos/{vid}/file")
def video_file(vid: str):
    v = videos.get(vid)
    if not v or v.state != "done" or not v.path or not v.path.exists():
        raise HTTPException(404, "Video chưa sẵn sàng hoặc đã hết hạn.")
    name = f"livercancer-ai-{v.modality.lower()}-3d-{v.volume.replace(' ', '-').lower()}.mp4"
    return FileResponse(v.path, media_type="video/mp4", filename=name, headers={"Cache-Control": "no-store"})


@app.delete("/api/videos/{vid}")
def video_delete(vid: str):
    return {"deleted": videos.delete(vid)}


@app.get("/api/jobs/{job_id}/thumb")
def job_thumb(job_id: str, volume: str, k: int, branch: str = ""):
    job, _ = jobs.get(job_id)
    br = None
    if job and job.state == "done" and job.viewer and job.result:
        br = next((b for b in job.result["branches"] if b["key"] == (branch or job.result["branches"][0]["key"])), None)
    if br is None or br["key"] not in job.viewer or volume not in job.viewer[br["key"]]:
        raise HTTPException(404, "Không còn dữ liệu lát cắt cho kết quả này.")
    return Response(service.thumb(job.viewer[br["key"]], volume, k), media_type="image/jpeg")


@app.delete("/api/jobs/{job_id}")
def job_delete(job_id: str):
    return {"deleted": jobs.delete(job_id)}


# ----------------------------------------------------------------------------- frontend (built React app)
DIST = Path(settings.frontend_dist) if settings.frontend_dist else None
if DIST and (DIST / "index.html").exists():
    if (DIST / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "Không tìm thấy.")
        f = (DIST / path).resolve()
        if path and f.is_file() and DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
