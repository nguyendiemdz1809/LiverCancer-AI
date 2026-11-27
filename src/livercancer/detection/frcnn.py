"""Faster R-CNN ResNet-50 FPN (torchvision): control detector of the YOLO11n comparison (CT only).

NOT used by the pipeline or the website: YOLO11n stays the detector of the project. This module is self-contained
(torch, torchvision, numpy, opencv; ultralytics only for `ap_per_class`) so the Colab notebook can inline it.

Data   : the YOLO export of GĐ2 (images/<split>/*.png 512x512 grayscale, labels/<split>/*.txt one class `lesion`),
         read exactly like Ultralytics reads it (3 identical channels). Negative slices (empty label) are kept.
Model  : torchvision `fasterrcnn_resnet50_fpn` (v1, COCO pretrained), predictor replaced by 2 classes (background,
         lesion), images kept at 512 (min_size = max_size = 512, no resize).
Metrics: `detection_metrics` re-implements Ultralytics DetectionValidator (greedy matching in descending confidence over
         IoU 0.50:0.95, `ap_per_class`) so mAP of both models comes from the same algorithm; `check_evaluator` in
         scripts/79_frcnn_comparison.py verifies it reproduces Ultralytics `val` on YOLO11n.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np

IMGSZ = 512
IOUV = np.linspace(0.5, 0.95, 10)
SMALL = 32 * 32


# ------------------------------------------------------------------------------------------------------------ data
def read_labels(lab: Path, w: int = IMGSZ, h: int = IMGSZ) -> np.ndarray:
    """YOLO txt (cls cx cy w h, normalised) -> xyxy pixels (N, 4) float32."""
    out = []
    if lab.exists():
        for line in lab.read_text().splitlines():
            if line.strip():
                _, cx, cy, bw, bh = map(float, line.split())
                out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out, np.float32).reshape(-1, 4)


def list_split(root: Path, split: str) -> list[tuple[Path, Path]]:
    imgs = sorted((root / "images" / split).glob("*.png"))
    return [(p, root / "labels" / split / f"{p.stem}.txt") for p in imgs]


def load_image(p: Path) -> np.ndarray:
    import cv2
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(p)
    return im


class YoloFolder:
    """torch Dataset over a YOLO split. Train augmentation: horizontal flip (p 0.5) and brightness gain in
    [1 - v, 1 + v] (the `fliplr` / `hsv_v` of the YOLO runs; mosaic is YOLO-specific and not used)."""

    def __init__(self, root: Path, split: str, train: bool = False, flip: float = 0.5, gain: float = 0.3):
        self.items = list_split(Path(root), split)
        self.train, self.flip, self.gain = train, flip, gain

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        ip, lp = self.items[i]
        im = load_image(ip).astype(np.float32) / 255.0
        h, w = im.shape
        boxes = read_labels(lp, w, h)
        if self.train:
            if random.random() < self.flip:
                im = im[:, ::-1]
                boxes = boxes[:, [2, 1, 0, 3]].copy()
                boxes[:, [0, 2]] = w - boxes[:, [0, 2]]
            if self.gain:
                im = np.clip(im * random.uniform(1 - self.gain, 1 + self.gain), 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(im))[None].expand(3, -1, -1).contiguous()
        tgt = {"boxes": torch.from_numpy(boxes), "labels": torch.ones(len(boxes), dtype=torch.int64)}
        return x, tgt, ip.stem


def collate(batch):
    return tuple(zip(*batch))


# ----------------------------------------------------------------------------------------------------------- model
def build_model(pretrained: bool = True, score_thresh: float = 0.001, detections: int = 300):
    """torchvision Faster R-CNN R50-FPN v1, 2 classes. score_thresh / detections as the YOLO val (conf 0.001, max_det
    300); the RoI-head NMS stays at the torchvision default 0.5 (part of the architecture's standard recipe)."""
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    # pretrained=False: architecture only (weights come from a checkpoint); weights_backbone=None so torchvision does
    # not download the ImageNet ResNet-50 (it would be overwritten by load_state_dict anyway).
    m = fasterrcnn_resnet50_fpn(weights=w, weights_backbone=None, min_size=IMGSZ, max_size=IMGSZ, box_score_thresh=score_thresh,
                                box_detections_per_img=detections)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    return m


def load_weights(path: Path, device="cpu"):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


# ------------------------------------------------------------------------------------------------------ prediction
def predict_split(model, root: Path, split: str, device="cuda", batch: int = 8, amp: bool = True,
                  log_every: int = 2000, limit: int | None = None) -> dict[str, dict]:
    """{stem: {'boxes': (N,4) xyxy, 'conf': (N,), 'gt': (M,4)}} for every image of the split (first `limit` if set)."""
    import torch
    ds = YoloFolder(root, split, train=False)
    if limit:
        ds.items = ds.items[:limit]
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate)
    out, n, t0 = {}, 0, time.time()
    model.eval()
    with torch.inference_mode():
        for xs, tg, stems in dl:
            with torch.autocast("cuda", enabled=amp and str(device).startswith("cuda")):
                res = model([x.to(device, non_blocking=True) for x in xs])
            for r, t, s in zip(res, tg, stems):
                out[s] = {"boxes": r["boxes"].float().cpu().numpy(), "conf": r["scores"].float().cpu().numpy(),
                          "gt": t["boxes"].numpy()}
            n += len(xs)
            if log_every and n // log_every != (n - len(xs)) // log_every:
                print(f"  predict {split}: {n}/{len(ds)} ({n / (time.time() - t0):.1f} img/s)", flush=True)
    return out


# --------------------------------------------------------------------------------------------------------- metrics
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-7)


def correct_matrix(pred: np.ndarray, conf: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Ultralytics `match_predictions` (use_scipy=False): detections in descending confidence, each claims the best
    unclaimed target per IoU threshold. Returns (N, 10) bool in the order of `np.argsort(-conf)`."""
    order = np.argsort(-conf, kind="stable")
    correct = np.zeros((len(pred), len(IOUV)), bool)
    if not len(pred) or not len(gt):
        return correct
    iou = box_iou(gt, pred[order])                       # (M, N)
    matched = np.zeros((len(gt), len(IOUV)), bool)
    for j in np.flatnonzero((iou >= IOUV.min()).any(0)):
        avail = np.where(matched, 0, iou[:, j, None])
        k = avail.argmax(0)
        correct[j] = avail[k, range(len(IOUV))] >= IOUV
        matched[k, range(len(IOUV))] |= correct[j]
    return correct


def detection_metrics(preds: dict[str, dict]) -> dict:
    """P, R, F1 (max-F1 point), mAP50, mAP50-95 with Ultralytics `ap_per_class` on the same statistics it builds."""
    from ultralytics.utils.metrics import ap_per_class
    tps, confs, n_gt = [], [], 0
    for d in preds.values():
        c = d["conf"]
        tps.append(correct_matrix(d["boxes"], c, d["gt"]))
        confs.append(np.sort(c)[::-1])
        n_gt += len(d["gt"])
    tp = np.concatenate(tps) if tps else np.zeros((0, 10), bool)
    conf = np.concatenate(confs) if confs else np.zeros(0)
    r = ap_per_class(tp, conf, np.zeros(len(conf)), np.zeros(n_gt))
    p, rc, ap = float(r[2].mean()), float(r[3].mean()), r[5]
    return {"precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
            "mAP50": float(ap[:, 0].mean()), "mAP50-95": float(ap.mean()), "images": len(preds), "gt_boxes": n_gt,
            "detections": int(len(conf))}


def operating_point(preds: dict[str, dict], thr: float) -> dict:
    """Greedy matching (highest confidence first, IoU >= 0.5) at confidence `thr` (same rule as scripts/53_threshold_sweep.py)."""
    tp = fp = fn = small_tp = small_n = 0
    for d in preds.values():
        keep = d["conf"] >= thr
        pred, conf, gt = d["boxes"][keep], d["conf"][keep], d["gt"]
        small = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) < SMALL
        matched = np.zeros(len(gt), bool)
        if len(pred) and len(gt):
            for row in box_iou(pred[np.argsort(-conf, kind="stable")], gt):
                row = np.where(matched, -1, row)
                j = int(row.argmax())
                if row[j] >= 0.5:
                    matched[j] = True
        tp += int(matched.sum()); fn += int((~matched).sum()); fp += len(pred) - int(matched.sum())
        small_n += int(small.sum()); small_tp += int((matched & small).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"conf": thr, "iou_match": 0.5, "gt_boxes": tp + fn, "tp": tp, "fp": fp, "fn": fn, "precision": p,
            "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
            "f2": 5 * p * rc / (4 * p + rc) if p + rc else 0.0,
            "small_boxes": small_n, "small_recall": small_tp / small_n if small_n else None}


def choose_threshold(preds: dict[str, dict]) -> dict:
    """Operating threshold of the model chosen on Val with the project rule (scripts/53_threshold_sweep.py):
    t* = argmax F2 on a 0.01 grid, ties -> higher t."""
    best = None
    for t in np.round(np.arange(0.01, 0.81, 0.01), 2):
        op = operating_point(preds, float(t))
        if best is None or op["f2"] >= best["f2"]:
            best = op
    return best


# -------------------------------------------------------------------------------------------------------- training
def train(data: Path, out: Path, epochs: int = 12, batch: int = 8, lr: float = 0.01, momentum: float = 0.9,
          weight_decay: float = 1e-4, warmup_iters: int = 1000, patience: int = 5, seed: int = 42, amp: bool = True,
          workers: int = 2, save_every: int = 500, val_batch: int = 8, device: str = "cuda",
          max_train: int | None = None, max_val: int | None = None) -> dict:
    """SGD + linear warm-up + cosine decay (per iteration), AMP. Full Val after every epoch; best.pt = highest
    mAP50-95 (the YOLO `fitness`); early stop after `patience` epochs without improvement. Resumes from out/last.pt
    (also mid-epoch: the epoch permutation is a function of seed + epoch). max_train / max_val: smoke test only."""
    import torch
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    ds = YoloFolder(data, "train", train=True)
    if max_train:
        ds.items = random.Random(seed).sample(ds.items, max_train)
    val_root = Path(data)
    model = build_model(pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    per_epoch = len(ds) // batch
    total = per_epoch * epochs

    def lr_at(it):
        if it < warmup_iters:
            return lr * (0.001 + 0.999 * it / warmup_iters)
        return lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / max(1, total - warmup_iters))))

    state = {"epoch": 0, "iter_in_epoch": 0, "history": [], "best": -1.0, "best_epoch": -1, "bad": 0,
             "config": {"epochs": epochs, "batch": batch, "lr": lr, "momentum": momentum, "weight_decay": weight_decay,
                        "warmup_iters": warmup_iters, "patience": patience, "seed": seed, "amp": amp, "imgsz": IMGSZ,
                        "train_images": len(ds), "augment": {"fliplr": 0.5, "brightness_gain": 0.3},
                        "model": "torchvision fasterrcnn_resnet50_fpn COCO_V1", "data": str(data)},
             "seconds": 0.0}
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        print(f"resume: epoch {state['epoch']} iter {state['iter_in_epoch']} (best mAP50-95 {state['best']:.4f} "
              f"@ epoch {state['best_epoch']})", flush=True)

    def save(path, extra=None):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "state": state, **(extra or {})}, tmp)
        tmp.replace(path)

    while state["epoch"] < epochs and state["bad"] < patience:
        ep = state["epoch"]
        g = torch.Generator().manual_seed(seed * 1000 + ep)
        perm = torch.randperm(len(ds), generator=g).tolist()[: per_epoch * batch]
        start = state["iter_in_epoch"]
        sub = torch.utils.data.Subset(ds, perm[start * batch:])
        dl = torch.utils.data.DataLoader(sub, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate,
                                         drop_last=True, persistent_workers=False, pin_memory=True)
        model.train()
        t0, run, n = time.time(), 0.0, 0
        t_ep = t0                                   # display only (t0 is reset at every checkpoint)
        for i, (xs, tg, _) in enumerate(dl, start=start):
            it = ep * per_epoch + i
            for gp in opt.param_groups:
                gp["lr"] = lr_at(it)
            xs = [x.to(device, non_blocking=True) for x in xs]
            tg = [{k: v.to(device) for k, v in t.items()} for t in tg]
            with torch.autocast("cuda", enabled=amp):
                loss = sum(model(xs, tg).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"loss is {loss.item()} at epoch {ep} iter {i}")
            run += loss.item(); n += 1
            state["iter_in_epoch"] = i + 1
            if (i + 1) % 100 == 0 or i + 1 == per_epoch:
                el = time.time() - t_ep
                print(f"epoch {ep + 1}/{epochs} iter {i + 1}/{per_epoch} loss {run / n:.4f} lr {lr_at(it):.5f} "
                      f"{n * batch / el:.1f} img/s, ETA epoch {(per_epoch - i - 1) * el / n / 60:.1f} min", flush=True)
            if save_every and (i + 1) % save_every == 0:
                state["seconds"] += time.time() - t0; t0 = time.time()
                save(last)
        state["seconds"] += time.time() - t0
        # ---- validation (full Val split)
        tv = time.time()
        vr = Path(val_root)
        pv = predict_split(model, vr, "val", device, val_batch, amp, log_every=0, limit=max_val)
        m = detection_metrics(pv)
        m.update({"epoch": ep + 1, "train_loss": run / max(n, 1), "val_seconds": round(time.time() - tv, 1)})
        state["history"].append(m)
        print(f"epoch {ep + 1} val: P {m['precision']:.3f} R {m['recall']:.3f} mAP50 {m['mAP50']:.3f} "
              f"mAP50-95 {m['mAP50-95']:.3f} ({m['val_seconds']} s)", flush=True)
        state["epoch"] = ep + 1
        state["iter_in_epoch"] = 0
        if m["mAP50-95"] > state["best"]:
            state["best"], state["best_epoch"], state["bad"] = m["mAP50-95"], ep + 1, 0
            torch.save({"model": model.state_dict(), "epoch": ep + 1, "val": m, "config": state["config"]},
                       out / "best.pt")
        else:
            state["bad"] += 1
        save(last)
        with open(out / "results.csv", "w", encoding="utf-8") as f:
            keys = ["epoch", "train_loss", "precision", "recall", "f1", "mAP50", "mAP50-95", "val_seconds"]
            f.write(",".join(keys) + "\n")
            for h in state["history"]:
                f.write(",".join(str(round(h[k], 5) if isinstance(h[k], float) else h[k]) for k in keys) + "\n")
    summary = {"epochs_run": state["epoch"], "best_epoch": state["best_epoch"], "best_mAP50-95": state["best"],
               "stopped_early": state["bad"] >= patience, "train_hours": round(state["seconds"] / 3600, 2),
               "config": state["config"], "history": state["history"]}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
