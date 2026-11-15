"""Box matching used to count TP / FP / FN (same rule as scripts/50_analyze_yolo_run.py)."""
from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes a (N,4) and b (M,4)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(gt: np.ndarray, pred: np.ndarray, conf: np.ndarray, iou_thr: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one matching by descending confidence. Returns (pred_is_tp, gt_is_matched)."""
    gt, pred = np.asarray(gt, float).reshape(-1, 4), np.asarray(pred, float).reshape(-1, 4)
    order = np.argsort(-np.asarray(conf, float))
    tp, used = np.zeros(len(pred), bool), np.zeros(len(gt), bool)
    if len(gt) and len(pred):
        ious = iou_matrix(pred, gt)
        for p in order:
            cand = np.where(~used & (ious[p] >= iou_thr))[0]
            if len(cand):
                g = cand[np.argmax(ious[p, cand])]
                tp[p], used[g] = True, True
    return tp, used


def count_outcomes(gt: np.ndarray, pred: np.ndarray, conf: np.ndarray, iou_thr: float = 0.5) -> dict:
    """TP / FP / FN on one image plus the best IoU of each GT box (for near-miss reporting)."""
    tp, used = match(gt, pred, conf, iou_thr)
    ious = iou_matrix(gt, pred)
    return {"TP": int(tp.sum()), "FP": int((~tp).sum()), "FN": int((~used).sum()),
            "gt_best_iou": [round(float(r.max()), 3) if r.size else 0.0 for r in ious]}
