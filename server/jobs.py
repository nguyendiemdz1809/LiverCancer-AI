"""In-memory analysis jobs: one worker thread (one GPU), bounded queue, results deleted after a TTL.

States: queued -> running -> done | error | needs_confirm.
needs_confirm: the file's own information (DICOM Modality tag, NIfTI header) contradicts the modality the user chose.
Nothing is analysed; the already-read images wait in memory (TTL) until the user picks a modality
(POST /api/jobs/{id}/confirm), or cancels.
Uploaded files live in a private temporary folder per job and are removed as soon as they are read (and in any case
when the job ends). Nothing is written to the project folders; nothing about the patient is logged.
Slices of 3D inputs are kept for the slice viewer of the 2 most recent results only (RAM bound), until the TTL.
A running analysis longer than `timeout_s` is stopped at its next progress report (every few YOLO batches) and ends
with a plain error, so one input cannot hold the GPU indefinitely. Deleting a job (DELETE /api/jobs/{id}, the
"Hủy" button) also cancels it: a queued job is skipped, a running one stops at its next progress report.
"""
from __future__ import annotations

import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from server.service import AnalysisService, ModalityConflict, UserError

KEEP_VIEWERS = 2
TIMEOUT_VI = "Phân tích vượt quá thời gian cho phép ({m} phút) nên đã dừng. Hãy thử với ít tệp hơn hoặc ảnh nhỏ hơn."


class JobTimeout(Exception):
    pass


class JobCancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    files: list[tuple[str, Path, str]]
    tmp: Path
    header_modality: str | None = None
    requested: str | None = None
    client: str = ""
    force: bool = False
    timed_out: bool = False
    cancelled: bool = False
    state: str = "queued"
    stage_key: str = "queued"
    stage: str = "Đang chờ đến lượt phân tích"
    progress: float | None = None
    result: dict | None = None
    error: str = ""
    need: dict | None = None
    loaded: object = None
    viewer: dict | None = None
    created: float = field(default_factory=time.time)
    finished: float = 0.0

    def public(self, position: int | None = None) -> dict:
        d = {"id": self.id, "state": self.state, "stage_key": self.stage_key, "stage": self.stage,
             "progress": None if self.progress is None else round(self.progress, 1)}
        if position:
            d["queue_position"] = position
        if self.state == "done":
            d["result"] = self.result
        if self.state == "error":
            d["error"] = self.error
        if self.state == "needs_confirm":
            d["need"] = self.need
        return d


class JobManager:
    def __init__(self, service: AnalysisService, max_queue: int, ttl_s: int, timeout_s: int = 600):
        self.service, self.max_queue, self.ttl, self.timeout_s = service, max_queue, ttl_s, timeout_s
        self.jobs: dict[str, Job] = {}
        self.q: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True, name="analysis-worker").start()
        threading.Thread(target=self._janitor, daemon=True, name="result-janitor").start()

    def waiting(self) -> int:
        with self.lock:
            return sum(j.state in ("queued", "running") for j in self.jobs.values())

    def active_for(self, client: str) -> int:
        """Analyses of one client (IP) still waiting for or using the GPU."""
        with self.lock:
            return sum(j.state in ("queued", "running") and j.client == client for j in self.jobs.values())

    def submit(self, files: list[tuple[str, Path, str]], tmp: Path, header_modality: str | None, requested: str | None,
               client: str = "") -> Job:
        job = Job(uuid.uuid4().hex, files, tmp, header_modality, requested, client)
        with self.lock:
            self.jobs[job.id] = job
        self.q.put(job.id)
        return job

    def get(self, job_id: str) -> tuple[Job | None, int | None]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None, None
            pos = None
            if job.state == "queued":
                pos = 1 + sum(1 for j in self.jobs.values() if j.state in ("queued", "running") and j.created < job.created)
            return job, pos

    def confirm(self, job_id: str, modality: str) -> Job | None:
        """After a contradiction: analyse as `modality`. Choosing the file's modality is a plain choice; keeping the
        original choice against the file's information is recorded as an override."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job.state != "needs_confirm" or job.loaded is None:
                return None
            job.force = modality != job.need["detected"]
            job.requested, job.state, job.need, job.finished = modality, "queued", None, 0.0
            job.stage_key, job.stage, job.progress = "queued", "Đang chờ đến lượt phân tích", None
        self.q.put(job_id)
        return job

    def delete(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.pop(job_id, None)
        if job:
            job.cancelled = True                     # a running analysis stops at its next progress report
            job.loaded = job.viewer = None
            shutil.rmtree(job.tmp, ignore_errors=True)
        return job is not None

    def _trim_viewers(self) -> None:
        with self.lock:
            done = sorted((j for j in self.jobs.values() if j.viewer is not None), key=lambda j: -j.finished)
            for j in done[KEEP_VIEWERS:]:
                j.viewer = None

    def _worker(self) -> None:
        while True:
            jid = self.q.get()
            with self.lock:
                job = self.jobs.get(jid)
            if not job or job.cancelled:
                continue
            job.state, t_start = "running", time.monotonic()

            def progress(key: str, stage: str, pct: float | None, job=job, t_start=t_start) -> None:
                if job.cancelled:
                    raise JobCancelled()
                if time.monotonic() - t_start > self.timeout_s:
                    job.timed_out = True
                    raise JobTimeout()
                job.stage_key, job.stage, job.progress = key, stage, pct

            try:
                if job.loaded is None:
                    job.loaded = self.service.load_inputs(job.files, progress)
                    job.files = []
                    shutil.rmtree(job.tmp, ignore_errors=True)          # inputs are in memory now
                mod, info = self.service.decide_modality(job.loaded, job.header_modality, job.requested, progress, job.force)
                job.result, job.viewer = self.service.run_models(job.loaded, mod, info, progress)
                job.loaded = None
                job.state, job.stage_key, job.stage, job.progress = "done", "done", "Hoàn thành", None
            except ModalityConflict as e:                               # ask; the images stay in memory until the TTL
                job.state, job.stage_key, job.stage, job.need = "needs_confirm", "modality", "Cần xác nhận loại ảnh", e.need()
            except UserError as e:
                job.state, job.error, job.loaded = "error", str(e), None
            except Exception:  # noqa: BLE001 - details stay in the server console, the user gets a plain message
                job.state, job.loaded = "error", None
                if job.cancelled:
                    job.error = "Đã hủy phân tích."
                elif job.timed_out:     # the locked CT pipeline turns the exception into an error status: use the flag
                    job.error = TIMEOUT_VI.format(m=max(1, round(self.timeout_s / 60)))
                else:
                    traceback.print_exc()
                    job.error = "Lỗi trong quá trình phân tích. Vui lòng thử lại hoặc dùng ảnh khác."
            finally:
                shutil.rmtree(job.tmp, ignore_errors=True)
                job.files = []
                if job.state != "needs_confirm" or job.cancelled:
                    job.loaded = None
                if job.cancelled:
                    job.viewer = None
                job.finished = time.time()
                self._trim_viewers()

    def _janitor(self) -> None:
        while True:
            time.sleep(30)
            now = time.time()
            with self.lock:
                old = [j for j in self.jobs.values() if j.finished and now - j.finished > self.ttl]
                for j in old:
                    self.jobs.pop(j.id, None)
            for j in old:
                j.loaded = j.viewer = None
                shutil.rmtree(j.tmp, ignore_errors=True)
