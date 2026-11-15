"""Classification metrics at ROI and patient level, and patient-level aggregation of ROI probabilities.

The aggregation method is chosen on Validation only (choose_aggregation); Test is evaluated once with the chosen one.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

AGGREGATIONS = ("mean_prob", "conf_weighted", "majority_vote", "top3_mean")


def compute_metrics(y_true, prob: np.ndarray, class_names: list[str]) -> dict:
    """Accuracy, balanced accuracy, macro precision / recall (sensitivity) / specificity / F1, per-class values,
    confusion matrix (rows = true, cols = predicted) and one-vs-rest ROC-AUC (macro, only when every class is present)."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y_true, int)
    prob = np.asarray(prob, float)
    k = len(class_names)
    pred = prob.argmax(1)
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y, pred), 1)
    per = {}
    for c, name in enumerate(class_names):
        tp, fn, fp = cm[c, c], cm[c].sum() - cm[c, c], cm[:, c].sum() - cm[c, c]
        tn = cm.sum() - tp - fn - fp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[name] = {"precision": p, "recall_sensitivity": r, "specificity": tn / (tn + fp) if tn + fp else 0.0,
                     "f1": 2 * p * r / (p + r) if p + r else 0.0, "support": int(cm[c].sum())}
    present = [n for n in class_names if per[n]["support"] > 0]
    out = {"n": int(len(y)), "accuracy": float((pred == y).mean()) if len(y) else 0.0,
           "balanced_accuracy": float(np.mean([per[n]["recall_sensitivity"] for n in present])) if present else 0.0,
           "macro_precision": float(np.mean([per[n]["precision"] for n in present])) if present else 0.0,
           "macro_recall": float(np.mean([per[n]["recall_sensitivity"] for n in present])) if present else 0.0,
           "macro_specificity": float(np.mean([per[n]["specificity"] for n in present])) if present else 0.0,
           "macro_f1": float(np.mean([per[n]["f1"] for n in present])) if present else 0.0,
           "per_class": per, "confusion_matrix": cm.tolist(), "class_names": list(class_names), "roc_auc_macro_ovr": None}
    if len(present) == k and k > 1 and len(y):
        try:
            if k == 2:
                out["roc_auc_macro_ovr"] = float(roc_auc_score(y, prob[:, 1]))
            else:
                out["roc_auc_macro_ovr"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=list(range(k))))
        except ValueError:
            pass
    return out


def aggregate(prob: np.ndarray, method: str) -> np.ndarray:
    """ROI probabilities (n_roi x k) of one patient -> one probability vector."""
    prob = np.asarray(prob, float)
    if method == "mean_prob":
        return prob.mean(0)
    if method == "conf_weighted":
        w = prob.max(1)
        return (prob * w[:, None]).sum(0) / w.sum()
    if method == "majority_vote":
        v = np.bincount(prob.argmax(1), minlength=prob.shape[1]).astype(float)
        return (v + 1e-3 * prob.mean(0)) / (v.sum() + 1e-3)   # ties broken by mean probability
    if method == "top3_mean":
        idx = np.argsort(-prob.max(1))[:3]
        return prob[idx].mean(0)
    raise ValueError(method)


def patient_level(patient_ids, y_true, prob: np.ndarray, method: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Aggregate ROI predictions per patient. Returns (patients, patient labels, patient probabilities)."""
    by = defaultdict(list)
    for i, p in enumerate(patient_ids):
        by[p].append(i)
    pats = sorted(by)
    labels = []
    for p in pats:
        ls = {int(y_true[i]) for i in by[p]}
        if len(ls) != 1:
            raise ValueError(f"patient {p} has ROIs with different labels {ls}")
        labels.append(ls.pop())
    return pats, np.array(labels), np.stack([aggregate(prob[by[p]], method) for p in pats])


def choose_aggregation(patient_ids, y_true, prob, class_names, methods=AGGREGATIONS) -> tuple[str, dict]:
    """Best patient-level aggregation by macro-F1 (ties -> order in `methods`). Call on Validation only."""
    res = {}
    for m in methods:
        _, yl, pp = patient_level(patient_ids, y_true, prob, m)
        res[m] = compute_metrics(yl, pp, class_names)
    best = max(methods, key=lambda m: (round(res[m]["macro_f1"], 6), -methods.index(m)))
    return best, res


def plot_confusion(cm, class_names, title: str, path, normalize: bool = True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, float)
    show = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(1.1 * len(class_names) + 3, 1.0 * len(class_names) + 2.5))
    im = ax.imshow(show, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=9)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{int(cm[i, j])}\n{show[i, j]:.2f}" if normalize else f"{int(cm[i, j])}", ha="center", va="center",
                    fontsize=8, color="white" if show[i, j] > (0.5 if normalize else show.max() / 2) else "black")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
