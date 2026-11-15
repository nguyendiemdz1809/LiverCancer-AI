"""Link per-slice boxes of one volume into 3D lesion candidates ("tracks") across consecutive slices."""
from __future__ import annotations

from dataclasses import dataclass, field

from livercancer.roi.boxes import Box, iou


@dataclass
class Track:
    track_id: int
    boxes: dict[int, Box] = field(default_factory=dict)   # slice index k -> merged box

    @property
    def ks(self) -> list[int]:
        return sorted(self.boxes)

    @property
    def n_slices(self) -> int:
        return len(self.boxes)

    @property
    def max_conf(self) -> float:
        return max(b.conf for b in self.boxes.values())

    @property
    def mean_conf(self) -> float:
        return sum(b.conf for b in self.boxes.values()) / len(self.boxes)

    @property
    def union_box(self) -> tuple[float, float, float, float]:
        bs = self.boxes.values()
        return (min(b.x0 for b in bs), min(b.y0 for b in bs), max(b.x1 for b in bs), max(b.y1 for b in bs))

    def key_slices(self, k: int = 3) -> list[int]:
        """Representative slices: highest area x confidence first."""
        return sorted(self.boxes, key=lambda s: -(self.boxes[s].area * self.boxes[s].conf))[:k]

    @property
    def best_slice(self) -> int:
        return self.key_slices(1)[0]


def link_tracks(per_slice: dict[int, list[Box]], iou_thr: float = 0.3, max_gap: int = 1) -> list[Track]:
    """Greedy linking in slice order: a box joins the active track whose last box has the highest IoU (>= iou_thr);
    a track stays active for `max_gap` empty slices. Each track takes at most one box per slice."""
    tracks: list[Track] = []
    for k in sorted(per_slice):
        active = [t for t in tracks if 0 < k - t.ks[-1] <= max_gap + 1]
        pairs = sorted(((iou(t.boxes[t.ks[-1]].xyxy, b.xyxy), ti, bi)
                        for ti, t in enumerate(active) for bi, b in enumerate(per_slice[k])), reverse=True)
        used_t, used_b = set(), set()
        for v, ti, bi in pairs:
            if v < iou_thr:
                break
            if ti in used_t or bi in used_b:
                continue
            active[ti].boxes[k] = per_slice[k][bi]
            used_t.add(ti)
            used_b.add(bi)
        for bi, b in enumerate(per_slice[k]):
            if bi not in used_b:
                tracks.append(Track(len(tracks), {k: b}))
    return tracks


def spaced_key_slices(track: Track, n: int, min_gap: int, score=lambda b: b.area * b.conf) -> list[int]:
    """Up to n slices of a track, best `score` first, at least `min_gap` slices apart, returned in slice order.
    Same rule as the classification dataset (scripts/80_build_cls_dataset.py `choose_slices`): YOLO ROIs use area x conf."""
    out: list[int] = []
    for k in sorted(track.boxes, key=lambda s: -score(track.boxes[s])):
        if all(abs(k - c) >= min_gap for c in out):
            out.append(k)
        if len(out) == n:
            break
    return sorted(out)


def filter_tracks(tracks: list[Track], min_slices: int = 1, min_max_conf: float = 0.0) -> list[Track]:
    return [t for t in tracks if t.n_slices >= min_slices and t.max_conf >= min_max_conf]
