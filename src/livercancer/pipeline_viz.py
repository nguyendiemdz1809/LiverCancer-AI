"""Vietnamese presentation of a pipeline result: figure (white background, dark text, large fonts) and JSON summary.

Figure rows: (1) original slice · YOLO boxes · Tumor Zoom local · Tumor Zoom context; (2) conclusion box (prediction,
true label when known, CORRECT / WRONG shown honestly) · probability per class (pipeline, and the GT-ROI "oracle" when
given, Validation only); (3) warnings. No label is ever drawn when the pipeline did not classify.
"""
from __future__ import annotations

import textwrap

import numpy as np

from livercancer.roi.render import draw

STATUS_VI = {"ok": "Đã phân tích", "no_lesion_detected": "Không phát hiện vùng nghi ngờ",
             "insufficient_evidence": "Chưa đủ bằng chứng để phân loại",
             "classifier_unavailable": "Chưa có mô hình phân loại", "error": "Lỗi khi xử lý",
             "detection_only": "Đã khoanh vùng nghi ngờ (không phân loại)"}
INK, BLUE, RED, GREEN, GREY = "#111827", "#0B3D91", "#B91C1C", "#15803D", "#4B5563"


def short(name: str) -> str:
    """'HCC (ung thư ...)' -> 'HCC' for chart axes; names without a code stay as they are."""
    return name.split(" (")[0]


def _names(res, names_vi: dict | None) -> dict:
    return names_vi or {c: c for c in (res.scores or {})}


def verdict(res, true_label: str | None, classes=None) -> str | None:
    """'ĐÚNG' / 'SAI' / 'KHÔNG PHÂN LOẠI' for a lesion class; for a label outside the model classes (CT controls, CN):
    'CN: KHÔNG GÁN NHÃN UNG THƯ' (no label - still not a 'healthy' claim) or 'CN: DƯƠNG TÍNH GIẢ'."""
    if not true_label:
        return None
    if classes is not None and true_label not in classes:
        return "CN: DƯƠNG TÍNH GIẢ" if res.status == "ok" else "CN: KHÔNG GÁN NHÃN UNG THƯ"
    if res.status != "ok":
        return "KHÔNG PHÂN LOẠI"
    return "ĐÚNG" if res.label == true_label else "SAI"


def result_vi(res, true_label: str | None = None, names_vi: dict | None = None) -> dict:
    """JSON-ready Vietnamese summary (+ the technical result under 'Chi tiết kỹ thuật')."""
    nv = _names(res, names_vi)
    out = {
        "Mã ca": res.case_id, "Loại ảnh": res.modality.upper(), "Trạng thái": STATUS_VI.get(res.status, res.status),
        "Mô tả": res.message,
        "Kết luận của hệ thống": (nv.get(res.label, res.label) if res.label else "Không đưa ra nhãn"),
        "Xác suất từng lớp": {nv.get(c, c): round(p, 4) for c, p in sorted(res.scores.items(), key=lambda x: -x[1])},
        "Độ tin cậy của mô hình phân loại": "Đã kiểm chứng trên Validation (chỉ phục vụ nghiên cứu)" if res.reliable
        else "CHƯA ĐẠT chất lượng / chưa kiểm chứng: không dùng như chẩn đoán",
        "Ảnh đã phân tích": [{"Tên": v["name"], "Số lát": v["n_slices"], "Số vùng nghi ngờ": v["n_regions"]} for v in res.volumes],
        "Các vùng nghi ngờ": [{"Vùng": r["track_id"], "Ảnh": r["volume"], "Lát": f"{r['k_first']}–{r['k_last']}",
                                "Độ tin cậy YOLO cao nhất": r["max_conf"], "Lát dùng để phân loại": r["key_slices"],
                                "Nhãn của vùng": nv.get(r.get("label"), r.get("label"))} for r in res.regions],
        "Cảnh báo": res.warnings, "Thời gian xử lý (giây)": res.timings_s, "Thiết bị": res.device,
    }
    if true_label:
        out["Nhãn thực tế (dataset, chỉ để đối chiếu)"] = nv.get(true_label, true_label)
        out["Đánh giá"] = verdict(res, true_label, list(names_vi) if names_vi else None)
    out["Chi tiết kỹ thuật"] = res.as_dict()
    return out


def render_case(res, path, title: str = "", true_label: str | None = None, gt_scores: dict | None = None,
                names_vi: dict | None = None, note: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams["font.family"] = "DejaVu Sans"          # ships with matplotlib, full Vietnamese diacritics
    nv = _names(res, names_vi)

    fig = plt.figure(figsize=(18, 10.2), facecolor="white")
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 0.78, 0.30], hspace=0.38, wspace=0.12)
    img_ax = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for a in img_ax:
        a.axis("off")
    titles = ["Ảnh gốc", "YOLO khoanh vùng tổn thương", "Tumor Zoom: cục bộ", "Tumor Zoom: ngữ cảnh"]
    if res.regions:
        top = max(res.regions, key=lambda r: r["max_conf"])
        rr = res.roi_results[top["volume"]]
        first = min(r["track_id"] for r in res.regions if r["volume"] == top["volume"])
        tr = rr.tracks[top["track_id"] - first]
        k = top["best_slice"]
        z = rr.zoom(tr, k)
        boxes = rr.boxes_on(k)
        img = rr.images[k]
        boxed = draw(img, [b.xyxy for _, b in boxes], [], [(*z["local_window"], "local"), (*z["context_window"], "context")],
                     [f"tổn thương {b.conf:.2f}" for _, b in boxes])
        panels = [img, boxed, z["local"], z["context"]]
        sub = [f"{top['volume']} · lát {k}", "đỏ: YOLO · xanh: khung zoom", "khối u + lề mô", "vùng rộng quanh khối u"]
    else:
        name, rr = next(iter(res.roi_results.items()))
        k = rr.n_slices // 2
        panels = [rr.images[k], rr.images[k], None, None]
        sub = [f"{name} · lát giữa {k}", "không có vùng nào vượt ngưỡng", "", ""]
    for a, p, t, s in zip(img_ax, panels, titles, sub):
        if p is None:
            a.text(0.5, 0.5, "Không có vùng\nđể phóng to", ha="center", va="center", fontsize=15, color=GREY)
        else:
            a.imshow(p, cmap=None if p.ndim == 3 else "gray", vmin=0, vmax=255)
        a.set_title(f"{t}\n{s}", fontsize=13, color=BLUE, fontweight="bold" if s == "" else "normal", loc="center")

    # ---- conclusion box
    box = fig.add_subplot(gs[1, :2])
    box.axis("off")
    box.set_xlim(0, 1)
    box.set_ylim(0, 1)
    v = verdict(res, true_label, list(names_vi) if names_vi else None)
    lines = [("Trạng thái: " + STATUS_VI.get(res.status, res.status), INK, 15, "bold")]
    if res.label:
        lines.append(("Kết luận của hệ thống: " + nv.get(res.label, res.label), BLUE, 16, "bold"))
        lines.append((f"Xác suất lớp dự đoán: {100 * res.scores[res.label]:.1f}%", INK, 14, "normal"))
    else:
        lines.append(("Hệ thống KHÔNG đưa ra nhãn bệnh cho ca này.", RED, 15, "bold"))
    if true_label:
        lines.append(("Nhãn thực tế (dataset): " + nv.get(true_label, true_label), INK, 14, "normal"))
    n_vol = len(res.volumes)
    lines.append((f"Đã phân tích {n_vol} ảnh, {sum(x['n_slices'] for x in res.volumes)} lát; {len(res.regions)} vùng nghi ngờ; "
                  f"{res.timings_s.get('total', 0):.1f} giây ({res.device}).", GREY, 12, "normal"))
    if note:
        lines.append((note, GREY, 12, "normal"))
    y = 0.95
    for t, c, s, w in lines:
        for i, part in enumerate(textwrap.wrap(t, 54 if s >= 15 else 64)):
            box.text(0.0, y, part, fontsize=s, color=c, fontweight=w, va="top")
            y -= s * 0.0082   # line height proportional to the font size
    if v:   # verdict badge in the top-right corner of the figure, next to the title (never over other text)
        col = {"ĐÚNG": GREEN, "SAI": RED, "CN: DƯƠNG TÍNH GIẢ": RED, "CN: KHÔNG GÁN NHÃN UNG THƯ": GREEN}.get(v, GREY)
        fig.text(0.995, 0.995, (f" {v} " if v.startswith("CN") else f" {v} so với nhãn thực tế "), fontsize=17, fontweight="bold", color="white", ha="right", va="top",
                 bbox={"boxstyle": "round,pad=0.35", "facecolor": col, "edgecolor": col})

    # ---- probabilities
    ax = fig.add_subplot(gs[1, 2:])
    if res.scores:
        classes = list(res.scores)
        yy = np.arange(len(classes))
        h = 0.38 if gt_scores else 0.6
        ax.barh(yy - (h / 2 if gt_scores else 0), [100 * res.scores[c] for c in classes], h, color="#DC2626",
                label="Pipeline tự động (vùng do YOLO tìm)")
        if gt_scores:
            ax.barh(yy + h / 2, [100 * gt_scores.get(c, 0) for c in classes], h, color="#16A34A",
                    label="Vùng cắt theo nhãn chuẩn (oracle, chỉ để đối chiếu)")
        ax.set_yticks(yy, [("▶ " if c == true_label else "") + short(nv.get(c, c)) for c in classes], fontsize=13, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Xác suất (%)", fontsize=12, color=INK)
        ax.set_title("Xác suất từng lớp" + (" (▶ = nhãn thực tế)" if true_label else ""), fontsize=14, color=BLUE)
        ax.legend(fontsize=10, loc="lower right", frameon=True)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(colors=INK)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Không có xác suất\n(không phân loại)", ha="center", va="center", fontsize=15, color=GREY)

    # ---- warnings
    wa = fig.add_subplot(gs[2, :])
    wa.axis("off")
    msgs = ([res.message] if res.status != "ok" else []) + res.warnings[1:] + [res.warnings[0]]
    red = res.modality == "mri" or res.status != "ok"
    wa.text(0, 1, "\n".join(textwrap.fill("• " + m, 160) for m in msgs), va="top", fontsize=11, color=RED if red else GREY)

    fig.suptitle(title or res.case_id, fontsize=18, fontweight="bold", color=INK, y=0.995)
    fig.savefig(path, dpi=100, facecolor="white", bbox_inches="tight")
    plt.close(fig)
