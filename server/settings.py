"""Backend settings from environment variables (never from files committed with secrets).

LCAI_DEVICE            auto | cuda:0 | cpu                      (default auto: GPU fp16, CPU fallback)
LCAI_ALLOWED_ORIGINS   comma-separated CORS origins             (default: local Vite dev server)
LCAI_ALLOWED_ORIGIN_REGEX  optional regex of extra CORS origins (e.g. preview URLs <hash>.<project>.pages.dev)
LCAI_MAX_FILE_MB       max size of one uploaded file            (default 300)
LCAI_MAX_FILES         max files per analysis                   (default 8: 4 CT phases or 8 MRI sequences)
LCAI_MAX_QUEUE         max waiting analyses                     (default 3)
LCAI_MAX_OPEN_UPLOADS  max unfinished uploads kept at once     (default 6)
LCAI_RATE_PER_MIN      API requests per client (IP) per minute  (default 600; a waiting page polls ~100/min)
LCAI_UPLOADS_PER_HOUR  uploads started per client per hour      (default 40)
LCAI_JOBS_PER_HOUR     analyses started per client per hour     (default 30)
LCAI_ACTIVE_PER_CLIENT analyses of one client waiting / running at once (default 2; one IP can be a shared Wi-Fi)
LCAI_JOB_TIMEOUT_S     max running time of one analysis         (default 600, then stopped with an error)
LCAI_DEDUP_EXACT       1 = YOLO once per byte-identical slice of a volume, result copied back (default 1; 0 = off)
The analysis API is public (no access code): it is protected by these limits, the CORS list and the queue size.
Behind Cloudflare the client is CF-Connecting-IP (set by Cloudflare, not by the browser).
LCAI_RESULT_TTL_S      seconds a finished result stays in memory (default 900, then deleted)
LCAI_FRONTEND_DIST     built frontend served at /               (default web/dist when it exists; "" disables)
LCAI_TMP_DIR           where uploads are written briefly         (default: system temp)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _list(v: str) -> list[str]:
    return [x.strip().rstrip("/") for x in v.split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    device: str = os.environ.get("LCAI_DEVICE", "auto")
    allowed_origins: list[str] = field(default_factory=lambda: _list(os.environ.get(
        "LCAI_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")))
    allowed_origin_regex: str = os.environ.get("LCAI_ALLOWED_ORIGIN_REGEX", "")
    max_file_mb: int = int(os.environ.get("LCAI_MAX_FILE_MB", "300"))
    max_files: int = int(os.environ.get("LCAI_MAX_FILES", "8"))
    max_queue: int = int(os.environ.get("LCAI_MAX_QUEUE", "3"))
    max_open_uploads: int = int(os.environ.get("LCAI_MAX_OPEN_UPLOADS", "6"))
    rate_per_min: int = int(os.environ.get("LCAI_RATE_PER_MIN", "600"))
    uploads_per_hour: int = int(os.environ.get("LCAI_UPLOADS_PER_HOUR", "40"))
    jobs_per_hour: int = int(os.environ.get("LCAI_JOBS_PER_HOUR", "30"))
    active_per_client: int = int(os.environ.get("LCAI_ACTIVE_PER_CLIENT", "2"))
    job_timeout_s: int = int(os.environ.get("LCAI_JOB_TIMEOUT_S", "600"))
    dedup_exact: bool = os.environ.get("LCAI_DEDUP_EXACT", "1") != "0"
    result_ttl_s: int = int(os.environ.get("LCAI_RESULT_TTL_S", "900"))
    frontend_dist: str = os.environ.get("LCAI_FRONTEND_DIST", str(ROOT / "web" / "dist"))
    tmp_dir: str = os.environ.get("LCAI_TMP_DIR", "")

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024


def get_settings() -> Settings:
    return Settings()
