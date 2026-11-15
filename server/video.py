"""3D scan video export: MP4 (H.264, yuv420p) of one analysed volume, slice 1 -> N, with the YOLO boxes of each slice.

Frames come from the volume kept in memory for the slice viewer (the same images the AI analysed) and the boxes of the
locked pipeline's 3D regions (merged YOLO boxes, each belonging to one region). Nothing is re-detected: YOLO is not run
again, and a slice without a detection is shown as it is. Box labels carry the region number and the real detection
confidence of that box; no classification score is written on a box.

Encoder: ffmpeg from imageio-ffmpeg (libx264). One export at a time (CPU only, the GPU is not used), bounded queue,
progress per frame, cancel (DELETE), temporary files removed after download TTL. Frames are streamed to ffmpeg one by
one, never held all in memory.
"""
from __future__ import annotations

import queue
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from livercancer.roi.render import font

OUT_H = 512                                   # output height (px); width follows the slice aspect, both multiples of 16
BAR = 34                                      # header / footer bar height in the output frame
RED, AMBER = (239, 68, 68), (245, 158, 11)


class Cancelled(Exception):
    pass


@dataclass
class VideoJob:
    id: str
    client: str
    title: str
    modality: str
    volume: str
    rr: object                                # RoiResult kept for the slice viewer (images + tracks)
    nums: dict                                # track index -> region number shown on the website
    fps: int
    state: str = "queued"                     # queued | running | done | error
    done: int = 0
    total: int = 0
    error: str = ""
    path: Path | None = None
    tmp: Path | None = None
    cancelled: bool = False
    created: float = field(default_factory=time.time)
    finished: float = 0.0

    def public(self) -> dict:
        d = {"id": self.id, "state": self.state, "done": self.done, "total": self.total,
             "progress": round(100.0 * self.done / self.total, 1) if self.total else 0.0}
        if self.state == "error":
            d["error"] = self.error
        if self.state == "done" and self.path:
            d["size_bytes"] = self.path.stat().st_size
        return d


def _out_size(w: int, h: int) -> tuple[int, int]:
    oh = OUT_H
    ow = int(round(w * oh / h / 16)) * 16
    return max(16, ow), oh


def render_frame(gray: np.ndarray, boxes: list[tuple], k: int, n: int, title: str, ow: int, oh: int) -> np.ndarray:
    """One RGB frame: slice (resized), boxes [(x0, y0, x1, y1, conf, label)], header and footer bars."""
    h, w = gray.shape[:2]
    ih = oh - 2 * BAR
    iw = int(round(w * ih / h))
    img = cv2.resize(gray, (iw, ih), interpolation=cv2.INTER_LINEAR if ih >= h else cv2.INTER_AREA)
    canvas = Image.new("RGB", (ow, oh), (11, 18, 32))
    x_off = (ow - iw) // 2
    canvas.paste(Image.fromarray(img).convert("RGB"), (x_off, BAR))
    d = ImageDraw.Draw(canvas)
    s = ih / h
    f_small, f_box = font(15), font(14)
    for i, (x0, y0, x1, y1, conf, label) in enumerate(sorted(boxes, key=lambda b: -b[4])):
        col = RED if i == 0 else AMBER
        X0, Y0, X1, Y1 = x_off + x0 * s, BAR + y0 * s, x_off + x1 * s, BAR + y1 * s
        d.rectangle([X0, Y0, X1, Y1], outline=col, width=2)
        t = f"{label} · {conf:.2f}".replace(".", ",")
        tw = d.textlength(t, font=f_box) + 8
        ty = Y0 - 19 if Y0 - 19 >= BAR else Y1 + 1
        d.rectangle([X0, ty, X0 + tw, ty + 18], fill=col)
        d.text((X0 + 4, ty + 1), t, fill=(255, 255, 255), font=f_box)
    d.rectangle([0, 0, ow, BAR], fill=(11, 61, 145))
    d.text((10, 8), title, fill=(255, 255, 255), font=f_small)
    right = f"Lát {k + 1}/{n}" + (f" · {len(boxes)} vùng" if boxes else "")
    d.text((ow - 10 - d.textlength(right, font=f_small), 8), right, fill=(255, 255, 255), font=f_small)
    d.rectangle([0, oh - BAR, ow, oh], fill=(11, 18, 32))
    d.text((10, oh - BAR + 9), "YOLO 2D từng lát · số = độ tin cậy phát hiện · không phải chẩn đoán",
           fill=(203, 213, 225), font=font(12))
    return np.asarray(canvas)


class VideoManager:
    def __init__(self, max_queue: int = 2, ttl_s: int = 900):
        self.max_queue, self.ttl = max_queue, ttl_s
        self.jobs: dict[str, VideoJob] = {}
        self.q: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True, name="video-worker").start()
        threading.Thread(target=self._janitor, daemon=True, name="video-janitor").start()

    def waiting(self) -> int:
        with self.lock:
            return sum(j.state in ("queued", "running") for j in self.jobs.values())

    def submit(self, client: str, title: str, modality: str, volume: str, rr, nums: dict, fps: int) -> VideoJob:
        j = VideoJob(uuid.uuid4().hex, client, title, modality, volume, rr, nums, fps, total=rr.n_slices)
        with self.lock:
            self.jobs[j.id] = j
        self.q.put(j.id)
        return j

    def get(self, vid: str) -> VideoJob | None:
        with self.lock:
            return self.jobs.get(vid)

    def delete(self, vid: str) -> bool:
        with self.lock:
            j = self.jobs.pop(vid, None)
        if j:
            j.cancelled = True
            j.rr = None
            if j.tmp:
                shutil.rmtree(j.tmp, ignore_errors=True)
        return j is not None

    def _encode(self, j: VideoJob) -> None:
        import imageio_ffmpeg
        rr = j.rr
        h, w = rr.images[0].shape[:2]
        ow, oh = _out_size(w, h)
        j.tmp = Path(tempfile.mkdtemp(prefix="lcai_video_"))
        out = j.tmp / "video.mp4"
        writer = imageio_ffmpeg.write_frames(str(out), (ow, oh), fps=j.fps, codec="libx264", pix_fmt_out="yuv420p",
                                             macro_block_size=16, quality=7, output_params=["-movflags", "+faststart"])
        writer.send(None)
        try:
            for k in range(rr.n_slices):
                if j.cancelled:
                    raise Cancelled()
                boxes = [(*b.xyxy, float(b.conf), f"Vùng {j.nums.get(rr.tracks.index(t), '?')}") for t, b in rr.boxes_on(k)]
                frame = render_frame(rr.images[k], boxes, k, rr.n_slices, j.title, ow, oh)
                writer.send(np.ascontiguousarray(frame))
                j.done = k + 1
        finally:
            writer.close()
        j.path = out

    def _worker(self) -> None:
        while True:
            vid = self.q.get()
            j = self.get(vid)
            if not j or j.cancelled:
                continue
            j.state = "running"
            try:
                self._encode(j)
                j.state = "done"
            except Cancelled:
                j.state, j.error = "error", "Đã hủy xuất video."
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                j.state, j.error = "error", "Không tạo được video. Vui lòng thử lại."
            finally:
                j.rr = None                    # the job no longer needs the volume
                j.finished = time.time()
                if j.state != "done" and j.tmp:
                    shutil.rmtree(j.tmp, ignore_errors=True)

    def _janitor(self) -> None:
        while True:
            time.sleep(30)
            now = time.time()
            with self.lock:
                old = [j for j in self.jobs.values() if j.finished and now - j.finished > self.ttl]
                for j in old:
                    self.jobs.pop(j.id, None)
            for j in old:
                if j.tmp:
                    shutil.rmtree(j.tmp, ignore_errors=True)
