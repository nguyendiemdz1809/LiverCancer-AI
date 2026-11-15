"""Training loop for the lesion classifiers (local smoke tests and Google Colab use the same code).

    summary = train(cfg)        # cfg = configs/cls.yaml `train` section + run-specific keys (see REQUIRED / defaults)

- Transfer learning (ImageNet backbones, lower LR for backbones), AdamW, warmup + cosine schedule, AMP on CUDA.
- Class-weighted cross-entropy (+ label smoothing). Early stopping on Validation macro-F1 at ROI level.
- Every epoch: history.csv (train/val loss, accuracy, macro-F1, per-class F1, patient-level macro-F1, overfitting gap),
  curves.png, last.pt (full state, resumable, written atomically); best.pt + confusion matrices + predictions on improvement.
- CUDA out of memory: batch size is halved and gradient accumulation doubled (same effective batch), epoch restarted.
- Test is never read here.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from livercancer.classification.data import Augment, RoiDataset, class_weights, read_manifest, worker_init
from livercancer.classification.metrics import compute_metrics, patient_level, plot_confusion
from livercancer.classification.models import build_model, param_groups

DEFAULTS = {"roi_source": "gt", "epochs": 40, "patience": 8, "batch_size": 32, "accum_steps": 1, "lr_backbone": 1e-4,
            "lr_head": 1e-3, "weight_decay": 0.05, "warmup_epochs": 1, "label_smoothing": 0.05, "class_weighting": "inverse_sqrt",
            "fusion": "concat", "fusion_dim": 256, "dropout": 0.3, "amp": True, "pretrained": True, "seed": 42, "num_workers": 2,
            "deterministic": False, "device": "auto", "resume": True, "max_train_rois": 0, "max_val_rois": 0, "min_batch_size": 2}
REQUIRED = ("modality", "model", "data_root", "out_dir")


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def _subset(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Small class-balanced subset for smoke tests: patients taken round-robin over classes, at most `cap` ROIs each
    (whole patients stay in one split; nothing crosses Train/Val)."""
    if not n or len(rows) <= n:
        return rows
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r["label_id"], {}).setdefault(r["patient_id"], []).append(r)
    labels = sorted(by)
    cap = max(1, n // (2 * len(labels)))
    queues = {lab: rng.sample(sorted(by[lab]), len(by[lab])) for lab in labels}
    out = []
    while len(out) < n and any(queues.values()):
        for lab in labels:
            if queues[lab] and len(out) < n:
                pid = queues[lab].pop()
                out += rng.sample(by[lab][pid], min(cap, len(by[lab][pid])))
    return out[:n]


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str, amp: bool, criterion: nn.Module | None = None) -> tuple[np.ndarray, float]:
    """Softmax probabilities in dataset order (and mean loss when a criterion is given)."""
    model.eval()
    probs, loss_sum, n = [], 0.0, 0
    for b in loader:
        x1, x2, y = b["local"].to(device, non_blocking=True), b["context"].to(device, non_blocking=True), b["label"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.startswith("cuda")):
            logits = model(local=x1, context=x2)
        logits = logits.float()
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * len(y)
            n += len(y)
        probs.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros((0, 1)), (loss_sum / n if n else float("nan"))


def _loader(ds, bs, shuffle, workers, seed) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=worker_init if workers else None, generator=g, persistent_workers=False, drop_last=False)


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _curves(hist: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train loss")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="val loss")
    ax[1].plot(ep, [h["train_acc"] for h in hist], label="train accuracy")
    ax[1].plot(ep, [h["val_acc"] for h in hist], label="val accuracy (ROI)")
    ax[1].plot(ep, [h["val_macro_f1"] for h in hist], label="val macro-F1 (ROI)")
    ax[1].plot(ep, [h["val_patient_macro_f1"] for h in hist], label="val macro-F1 (bệnh nhân)", ls="--")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def train(cfg: dict) -> dict:
    cfg = {**DEFAULTS, **cfg}
    for k in REQUIRED:
        if k not in cfg:
            raise KeyError(f"missing config key {k!r}")
    root, out = Path(cfg["data_root"]), Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"], cfg["deterministic"])
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if cfg["device"] == "auto" else cfg["device"]
    amp = bool(cfg["amp"]) and device.startswith("cuda")
    cmap = json.loads((root / "class_map.json").read_text(encoding="utf-8"))
    classes = cmap["classes"]
    k = len(classes)

    sources = ("gt", "yolo") if cfg["roi_source"] == "gt+yolo" else ("gt",)
    tr_rows = _subset(read_manifest(root / "manifest.csv", "train", sources, matched_only=True), cfg["max_train_rois"], cfg["seed"])
    va_rows = _subset(read_manifest(root / "manifest.csv", "val", ("gt",)), cfg["max_val_rois"], cfg["seed"])
    if not tr_rows or not va_rows:
        raise RuntimeError("empty Train or Val set: check manifest.csv / roi_source")
    assert not ({r["patient_id"] for r in tr_rows} & {r["patient_id"] for r in va_rows}), "patient in Train and Val"
    y_tr = [int(r["label_id"]) for r in tr_rows]
    y_va = np.array([int(r["label_id"]) for r in va_rows])

    model = build_model(cfg["model"], k, cfg["pretrained"], cfg["fusion"], cfg["fusion_dim"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg["lr_backbone"], cfg["lr_head"], cfg["weight_decay"]))
    base_lrs = [g["lr"] for g in opt.param_groups]   # before a resume overwrites them with scheduled values
    w = class_weights(y_tr, k, cfg["class_weighting"]).to(device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg["label_smoothing"])
    crit_eval = nn.CrossEntropyLoss()   # unweighted Val loss: comparable across runs, used to watch overfitting
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    bs, accum = int(cfg["batch_size"]), int(cfg["accum_steps"])

    state = {"epoch": 0, "best": -1.0, "best_epoch": 0, "bad": 0, "history": [], "oom_events": []}
    last = out / "last.pt"
    if cfg["resume"] and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        if ck["cfg"]["model"] != cfg["model"] or ck["cfg"]["modality"] != cfg["modality"]:
            raise RuntimeError(f"{last} belongs to another run ({ck['cfg']['modality']}/{ck['cfg']['model']})")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        bs, accum = ck["batch_size"], ck["accum_steps"]
        print(f"resumed from {last}: epoch {state['epoch']}, best macro-F1 {state['best']:.4f} (epoch {state['best_epoch']})", flush=True)

    tr_ds = RoiDataset(tr_rows, root, Augment(seed=cfg["seed"]))
    va_ds = RoiDataset(va_rows, root)
    va_loader = _loader(va_ds, max(bs, 8), False, cfg["num_workers"], cfg["seed"])
    steps_per_epoch = lambda: math.ceil(len(tr_ds) / bs / accum)  # noqa: E731

    def lr_factor(step: int) -> float:
        total = cfg["epochs"] * steps_per_epoch()
        warm = cfg["warmup_epochs"] * steps_per_epoch()
        if step < warm:
            return (step + 1) / max(1, warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

    (out / "config.json").write_text(json.dumps({**cfg, "classes": classes, "n_train_rois": len(tr_rows), "n_val_rois": len(va_rows),
                                                 "n_train_patients": len({r['patient_id'] for r in tr_rows}),
                                                 "n_val_patients": len({r['patient_id'] for r in va_rows}),
                                                 "class_weights": [round(float(x), 4) for x in w.cpu()], "device": device,
                                                 "torch": torch.__version__}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{cfg['modality'].upper()} {cfg['model']}: train {len(tr_rows)} ROIs / val {len(va_rows)} ROIs, classes {classes}, "
          f"device {device}, AMP {amp}, batch {bs} x accum {accum}", flush=True)

    while state["epoch"] < cfg["epochs"] and state["bad"] < cfg["patience"]:
        ep = state["epoch"] + 1
        t0 = time.time()
        try:
            model.train()
            tr_loader = _loader(tr_ds, bs, True, cfg["num_workers"], cfg["seed"] + ep)
            loss_sum = correct = n = 0
            opt.zero_grad(set_to_none=True)
            step0 = (ep - 1) * steps_per_epoch()
            for i, b in enumerate(tr_loader):
                x1, x2, y = b["local"].to(device, non_blocking=True), b["context"].to(device, non_blocking=True), b["label"].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    logits = model(local=x1, context=x2)
                    loss = crit(logits.float(), y)
                scaler.scale(loss / accum).backward()
                if (i + 1) % accum == 0 or i + 1 == len(tr_loader):
                    f = lr_factor(step0 + i // accum)
                    for g, lr0 in zip(opt.param_groups, base_lrs):
                        g["lr"] = lr0 * f
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                loss_sum += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                n += len(y)
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if bs // 2 < cfg["min_batch_size"]:
                raise
            bs, accum = bs // 2, accum * 2
            state["oom_events"].append({"epoch": ep, "new_batch_size": bs, "accum_steps": accum})
            print(f"CUDA out of memory -> batch {bs} x accum {accum} (same effective batch), restarting epoch {ep}", flush=True)
            continue

        prob, val_loss = predict(model, va_loader, device, amp, crit_eval)
        m = compute_metrics(y_va, prob, classes)
        pats, yp, pp = patient_level([r["patient_id"] for r in va_rows], y_va, prob, "mean_prob")
        mp = compute_metrics(yp, pp, classes)
        h = {"epoch": ep, "lr_head": round(opt.param_groups[-1]["lr"], 8), "batch_size": bs, "accum_steps": accum,
             "train_loss": round(loss_sum / n, 5), "train_acc": round(correct / n, 4), "val_loss": round(val_loss, 5),
             "val_acc": round(m["accuracy"], 4), "val_bal_acc": round(m["balanced_accuracy"], 4), "val_macro_f1": round(m["macro_f1"], 4),
             "val_patient_acc": round(mp["accuracy"], 4), "val_patient_macro_f1": round(mp["macro_f1"], 4),
             "gap_acc_train_minus_val": round(correct / n - m["accuracy"], 4), "sec": round(time.time() - t0, 1)}
        h.update({f"val_f1_{c}": round(m["per_class"][c]["f1"], 4) for c in classes})
        hist = state["history"] + [h]
        vl = [x["val_loss"] for x in hist[-4:]]
        tl = [x["train_loss"] for x in hist[-4:]]
        h["overfit_warning"] = int(len(vl) == 4 and all(a < b for a, b in zip(vl, vl[1:])) and tl[-1] < tl[0])
        state["history"] = hist
        improved = m["macro_f1"] > state["best"] + 1e-4
        if improved:
            state.update(best=m["macro_f1"], best_epoch=ep, bad=0)
            _atomic_save({"model": model.state_dict(), "cfg": cfg, "classes": classes, "epoch": ep,
                          "val_metrics": {"roi": m, "patient_mean_prob": mp}}, out / "best.pt")
            with open(out / "val_predictions_best.csv", "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["roi_id", "patient_id", "label_id", "pred_id", *[f"p_{c}" for c in classes]])
                for r, p in zip(va_rows, prob):
                    wr.writerow([r["roi_id"], r["patient_id"], r["label_id"], int(p.argmax()), *np.round(p, 5)])
            (out / "metrics_val_best.json").write_text(json.dumps({"epoch": ep, "roi": m, "patient_mean_prob": mp}, indent=1,
                                                                  ensure_ascii=False), encoding="utf-8")
            plot_confusion(m["confusion_matrix"], classes, f"Val ROI (epoch {ep}) – macro-F1 {m['macro_f1']:.3f}", out / "cm_val_roi_best.png")
            plot_confusion(mp["confusion_matrix"], classes, f"Val bệnh nhân (trung bình xác suất) – macro-F1 {mp['macro_f1']:.3f}",
                           out / "cm_val_patient_best.png")
        else:
            state["bad"] += 1
        state["epoch"] = ep
        _atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "state": state,
                      "cfg": cfg, "classes": classes, "batch_size": bs, "accum_steps": accum}, last)
        with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(hist[-1]))
            wr.writeheader()
            wr.writerows(state["history"])
        _curves(state["history"], out / "curves.png")
        print(f"ep {ep:3d} | train loss {h['train_loss']:.4f} acc {h['train_acc']:.3f} | val loss {h['val_loss']:.4f} "
              f"acc {h['val_acc']:.3f} macro-F1 {h['val_macro_f1']:.3f} (BN {h['val_patient_macro_f1']:.3f})"
              f"{' *' if improved else ''}{' [overfit?]' if h['overfit_warning'] else ''} | {h['sec']}s", flush=True)

    summ = {"run": out.name, "modality": cfg["modality"], "model": cfg["model"], "best_epoch": state["best_epoch"],
            "best_val_macro_f1_roi": round(state["best"], 4), "epochs_run": state["epoch"],
            "stopped_early": state["bad"] >= cfg["patience"], "oom_events": state["oom_events"],
            "finished": datetime.now().isoformat(timespec="seconds")}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    return summ
