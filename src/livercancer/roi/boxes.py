"""Within-slice merging of YOLO boxes (duplicates / nested boxes of one lesion -> one box).

Boxes are (x0, y0, x1, y1) in pixels of the displayed slice (x = column, y = row), as returned by LesionDetector.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float
    n_merged: int = 1          # how many detector boxes were merged into this one
    confs: list[float] = field(default_factory=list)

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


def iou(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def containment(a: tuple, b: tuple) -> float:
    """Intersection area / area of the smaller box (1.0 when one box lies inside the other)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small if small > 0 else 0.0


def merge_boxes(dets, iou_thr: float = 0.5, contain_thr: float = 0.8) -> list[Box]:
    """Greedy merge by descending confidence: a box joins a group when IoU > iou_thr or containment > contain_thr
    with the group's current box; the group box becomes the union, its confidence the maximum. Repeats until stable."""
    boxes = [Box(d.x0, d.y0, d.x1, d.y1, d.conf, 1, [d.conf]) for d in sorted(dets, key=lambda d: -d.conf)]
    changed = True
    while changed:
        changed = False
        out: list[Box] = []
        for b in boxes:
            for g in out:
                if iou(g.xyxy, b.xyxy) > iou_thr or containment(g.xyxy, b.xyxy) > contain_thr:
                    g.x0, g.y0, g.x1, g.y1 = min(g.x0, b.x0), min(g.y0, b.y0), max(g.x1, b.x1), max(g.y1, b.y1)
                    g.conf, g.n_merged, g.confs = max(g.conf, b.conf), g.n_merged + b.n_merged, g.confs + b.confs
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return sorted(boxes, key=lambda b: -b.conf)
