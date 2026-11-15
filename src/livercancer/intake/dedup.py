"""Exact-duplicate slices of a volume: run YOLO once per distinct slice content, give the result back to every copy.

Only byte-identical slices (same BLAKE2b hash of the uint8 image given to YOLO) are duplicates. Consecutive slices that
merely look alike are NEVER merged: a lesion spans several neighbouring slices, and no similarity threshold has been
validated. Nothing is removed from the volume: every slice keeps its index k, its image and (copied) detections, so
tracking, Tumor Zoom, the classifier and the slice viewer see exactly the same volume as without this step.

Equivalence with the locked pipeline is checked on Validation volumes by scripts/86_slice_duplicates.py
(reports/slice_duplicates.md). Website: on by default, LCAI_DEDUP_EXACT=0 switches it off (server/settings.py).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


def slice_hash(img: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{img.shape}{img.dtype}".encode())
    h.update(np.ascontiguousarray(img).tobytes())
    return h.hexdigest()


@dataclass
class DedupStats:
    n_slices: int
    n_unique: int
    groups: list[list[int]] = field(default_factory=list)   # slice indices with identical content (>= 2 each)
    blank_groups: int = 0                                    # groups whose slice is uniform (e.g. empty padding)

    @property
    def n_duplicates(self) -> int:
        return self.n_slices - self.n_unique


def find_duplicates(images: list[np.ndarray]) -> tuple[list[int], list[int], DedupStats]:
    """-> (indices of the first copy of each content, for every slice the position of its content in that list, stats)."""
    first: dict[str, int] = {}
    uniq, where = [], []
    members: dict[int, list[int]] = {}
    for k, im in enumerate(images):
        h = slice_hash(im)
        if h not in first:
            first[h] = len(uniq)
            uniq.append(k)
        u = first[h]
        where.append(u)
        members.setdefault(u, []).append(k)
    groups = [m for m in members.values() if len(m) > 1]
    blank = sum(1 for g in groups if int(images[g[0]].max()) == int(images[g[0]].min()))
    return uniq, where, DedupStats(len(images), len(uniq), groups, blank)


class DedupDetector:
    """Wraps a detector: predict() runs the real detector on distinct slices only and copies results to duplicates.
    `stats` holds one DedupStats per predict call (one call per volume in analyze_volume)."""

    def __init__(self, det):
        self._det = det
        self.stats: list[DedupStats] = []

    def __getattr__(self, k):
        return getattr(self._det, k)

    def predict(self, images, conf=None, batch_size=8):
        uniq, where, st = find_duplicates(images)
        self.stats.append(st)
        if st.n_duplicates == 0:
            return self._det.predict(images, conf=conf, batch_size=batch_size)
        dets = self._det.predict([images[k] for k in uniq], conf=conf, batch_size=batch_size)
        return [list(dets[u]) for u in where]
