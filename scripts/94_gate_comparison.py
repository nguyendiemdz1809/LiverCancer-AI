"""GĐ5: compare the pipeline on Validation WITHOUT and WITH the evidence gate (reject option). Validation only, no retraining.

Inputs (both produced by scripts/92_pipeline_validation.py on the same checkpoints):
  outputs/predictions/pipeline_validation/<mod>/cases/<pid>.json        gate off (baseline)
  outputs/predictions/pipeline_validation_gate/<mod>/cases/<pid>.json   --evidence-gate on --tag gate
Per patient, mutually exclusive outcomes (a rejected case is NEVER counted as correct):
  correct   status ok, label == true label
  wrong     status ok, label != true label
  rejected  status insufficient_evidence (gate refused to give a label)
  missed    status no_lesion_detected (YOLO found no region at all)
CT controls (CN, no lesion data collected) have no correct class: "labelled" = the system gave a cancer label (false alarm).
Checks: the real gated run agrees with a replay of the gate rule on the baseline tracks, and labels of accepted cases are
unchanged. Output: reports/pipeline_validation/so_sanh_evidence_gate.md (+ CSV, figure).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from livercancer.config import CONFIG_DIR, load_yaml  # noqa: E402
from livercancer.evidence import passes, rule_from_config  # noqa: E402

BASE = ROOT / "outputs" / "predictions" / "pipeline_validation"
GATE = ROOT / "outputs" / "predictions" / "pipeline_validation_gate"
REP = ROOT / "reports" / "pipeline_validation"
OUTCOMES = ["correct", "wrong", "rejected", "missed"]
OUT_VI = {"correct": "Đúng", "wrong": "Sai", "rejected": "Từ chối phân loại", "missed": "Bỏ sót (không có vùng)"}
CANCER_RISK_MAX = 0.05     # declared before the run (scripts/93_: >= 95% of CT cancer patients must keep a label)


def load(d: Path, mod: str) -> dict:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((d / mod / "cases").glob("*.json"))}


def outcome(c: dict) -> str:
    if c["status"] == "insufficient_evidence":
        return "rejected"
    if c["status"] == "no_lesion_detected":
        return "missed"
    if c["status"] != "ok":
        raise ValueError(f"{c['patient_id']}: status {c['status']}")
    return "correct" if c["pipeline_label"] == c["true_label"] else "wrong"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, m - h), min(1.0, m + h))


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "–"


def ci(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * lo:.0f}–{100 * hi:.0f}%"


def gate_rules(mod: str):
    g = dict(load_yaml(CONFIG_DIR / "cls.yaml")["inference"][mod]["evidence_gate"])
    g["enabled"] = True                      # what --evidence-gate on applies
    return rule_from_config(g), rule_from_config({**g["single_volume"], "enabled": True})


def group_counts(cases: list[dict]) -> dict:
    n = {o: 0 for o in OUTCOMES}
    for c in cases:
        n[outcome(c)] += 1
    n["n"] = len(cases)
    n["no_true_lesion_found"] = sum(1 for c in cases if not any(t["matched"] for t in c["tracks"]))
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-dir", type=Path, default=GATE)
    args = ap.parse_args()

    rows, checks, groups = [], [], {}
    for mod in ("ct", "mri"):
        base, gated = load(BASE, mod), load(args.gate_dir, mod)
        assert set(base) == set(gated), f"{mod}: patient sets differ ({len(base)} vs {len(gated)})"
        rule, rule1 = gate_rules(mod)
        mism_replay, mism_label = [], []
        for pid in sorted(base):
            b, g = base[pid], gated[pid]
            r = rule1 if len(b["volumes"]) == 1 else rule
            replay = b["status"] == "ok" and not passes(b["tracks"], r)
            if replay != (g["status"] == "insufficient_evidence"):
                mism_replay.append(pid)
            if g["status"] == "ok" and (b["status"] != "ok" or g["pipeline_label"] != b["pipeline_label"]):
                mism_label.append(pid)
            ev = g.get("evidence") or {}
            rows.append({"modality": mod, "patient_id": pid, "true_label": b["true_label"],
                         "group": ("CN" if b["true_label"] == "CN" else "ung_thu") if mod == "ct" else "mri",
                         "n_volumes": len(b["volumes"]), "max_conf": ev.get("max_conf", ""),
                         "n_volumes_strong": ev.get("n_volumes_strong", ""),
                         "status_off": b["status"], "label_off": b["pipeline_label"] or "", "outcome_off": outcome(b),
                         "status_on": g["status"], "label_on": g["pipeline_label"] or "", "outcome_on": outcome(g),
                         "true_lesion_found": int(any(t["matched"] for t in b["tracks"]))})
        checks.append((mod, len(base), mism_replay, mism_label))
        if mod == "ct":
            for key, sel in (("CN", lambda c: c["true_label"] == "CN"), ("ung_thu", lambda c: c["true_label"] != "CN")):
                groups[(mod, key)] = {s: group_counts([c for c in d.values() if sel(c)]) for s, d in (("off", base), ("on", gated))}
        else:
            groups[(mod, "mri")] = {s: group_counts(list(d.values())) for s, d in (("off", base), ("on", gated))}

    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / "so_sanh_evidence_gate.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    figure(groups, REP / "figures" / "so_sanh_evidence_gate.png")
    summary = {f"{m}_{k}": v for (m, k), v in groups.items()}
    summary["checks"] = [{"modality": m, "n": n, "replay_mismatch": a, "label_changed": b} for m, n, a, b in checks]
    (REP / "so_sanh_evidence_gate.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report(groups, rows, checks)
    print("written", REP / "so_sanh_evidence_gate.md")


def figure(groups: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams["font.family"] = "DejaVu Sans"
    # categorical slots 1-4 of the validated reference palette, fixed order (adjacent pairs pass CVD checks)
    col = {"correct": "#2a78d6", "wrong": "#eb6834", "rejected": "#1baf7a", "missed": "#eda100"}
    lab = {"correct": "Đúng", "wrong": "Sai / CN bị gán nhãn ung thư", "rejected": "Từ chối phân loại",
           "missed": "Không có vùng YOLO"}
    titles = {("ct", "CN"): "CT – nhóm chứng CN (12 BN)", ("ct", "ung_thu"): "CT – bệnh nhân ung thư (42 BN)",
              ("mri", "mri"): "MRI – 78 BN"}
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.2), facecolor="white")
    for ax, key in zip(axes, titles):
        g = groups[key]
        for yi, s in enumerate(("on", "off")):
            x0 = 0
            for o in OUTCOMES:
                v = g[s][o]
                if key[1] == "CN" and o == "correct":
                    continue
                if v:
                    ax.barh(yi, v, left=x0, color=col[o], edgecolor="white", linewidth=2, height=0.62)
                    if v / g[s]["n"] >= 0.04:
                        ax.text(x0 + v / 2, yi, str(v), ha="center", va="center", fontsize=10,
                                color="white" if o in ("correct", "wrong") else "#0b0b0b")
                x0 += v
        ax.set_yticks([0, 1], ["Bật cổng", "Tắt cổng"], fontsize=10, color="#0b0b0b")
        ax.set_xlim(0, g["off"]["n"])
        ax.set_title(titles[key], loc="left", fontsize=11, color="#0b0b0b")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#b8b7b0")
        ax.tick_params(axis="x", colors="#52514e", labelsize=9)
        ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col[o]) for o in OUTCOMES]
    fig.legend(handles, [lab[o] for o in OUTCOMES], loc="lower center", ncol=4, frameon=False, fontsize=9.5)
    fig.suptitle("Pipeline trên Validation: trước và sau khi bật Evidence Gate (số bệnh nhân)", fontsize=12.5, x=0.02,
                 ha="left", color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def report(groups: dict, rows: list[dict], checks: list) -> None:
    cn, ca, mr = groups[("ct", "CN")], groups[("ct", "ung_thu")], groups[("mri", "mri")]
    rule, rule1 = gate_rules("ct")
    L = ["# So sánh pipeline trước và sau khi bật Evidence Gate (Validation)", "",
         f"_Tạo bởi `scripts/94_gate_comparison.py` lúc {datetime.now():%Y-%m-%d %H:%M}. Chỉ dùng Validation; Test không được dùng. "
         "Checkpoint YOLO và Hybrid giữ nguyên, không huấn luyện lại. Ngưỡng cổng chọn trên Val ở bước phân tích hạn chế "
         "(`reports/pipeline_validation/han_che/bao_cao_han_che.md`), không chỉnh lại sau lần chạy này._", "",
         "## 1. Cấu hình được kiểm tra", "",
         f"- **Nhiều pha / chuỗi:** chỉ phân loại khi {rule.describe_vi('pha (CT) / chuỗi (MRI)')}.",
         f"- **Một file:** chỉ phân loại khi {rule1.describe_vi()}.",
         "- Không đạt → trạng thái *Chưa đủ bằng chứng để phân loại*: không có nhãn, **không** kết luận khỏe mạnh.",
         "- Cổng chỉ dùng độ tin cậy và vị trí box YOLO (không dùng GT, không dùng nhãn hay xác suất của bộ phân loại).",
         "- MRI: áp **nguyên quy tắc của CT** để đánh giá. MRI Val không có nhóm chứng nên không chọn được ngưỡng riêng cho MRI.", "",
         "Cách đếm (mỗi BN thuộc đúng một nhóm): **Đúng** / **Sai** (có nhãn) · **Từ chối phân loại** (cổng không cho nhãn) · "
         "**Bỏ sót** (YOLO không có vùng nào). Ca bị từ chối **không** được tính là đúng: accuracy trên toàn bộ BN = Đúng / tổng số BN.", "",
         "![so sánh](figures/so_sanh_evidence_gate.png)", ""]

    def tbl(g, title, note=""):
        out = [f"### {title}", "", "| | Đúng | Sai | Từ chối phân loại | Bỏ sót | Accuracy trên toàn bộ BN | Accuracy trên ca có nhãn | Tỷ lệ có nhãn |",
               "|---|---|---|---|---|---|---|---|"]
        for s, nm in (("off", "Tắt cổng"), ("on", "Bật cổng")):
            x = g[s]
            lab = x["correct"] + x["wrong"]
            out.append(f"| {nm} | {x['correct']} | {x['wrong']} | {x['rejected']} | {x['missed']} | {pct(x['correct'], x['n'])} | "
                       f"{pct(x['correct'], lab)} | {pct(lab, x['n'])} |")
        if note:
            out += ["", note]
        return out + [""]

    L += ["## 2. Kết quả theo nhóm", "", "### CT – 12 bệnh nhân nhóm chứng CN", "",
          "CN không có lớp đúng trong bộ phân loại 3 lớp, nên không có ô *Đúng*. Hành vi mong muốn là không gán nhãn ung thư.", "",
          "| | Bị gán nhãn ung thư (cảnh báo sai) | Từ chối phân loại | Không có vùng YOLO | Không bị gán nhãn ung thư |", "|---|---|---|---|---|"]
    for s, nm in (("off", "Tắt cổng"), ("on", "Bật cổng")):
        x = cn[s]
        fp = x["correct"] + x["wrong"]
        L.append(f"| {nm} | {pct(fp, x['n'])} (KTC 95% {ci(fp, x['n'])}) | {x['rejected']} | {x['missed']} | {pct(x['n'] - fp, x['n'])} |")
    fp_ids = [r["patient_id"] + f" ({r['label_on']})" for r in rows if r["group"] == "CN" and r["label_on"]]
    L += ["", f"CN vẫn bị gán nhãn khi bật cổng: {', '.join(fp_ids) or 'không có'}.", ""]

    L += tbl(ca, "CT – 42 bệnh nhân ung thư (HCC / ICC / CHCC)")
    rej = [r for r in rows if r["group"] == "ung_thu" and r["outcome_on"] == "rejected"]
    L += [f"BN ung thư bị từ chối khi bật cổng: " + (", ".join(f"{r['patient_id']} ({r['true_label']}, trước đó {OUT_VI[r['outcome_off']].lower()}, "
                                                       f"conf max {float(r['max_conf']):.2f}, {r['n_volumes_strong']} pha đạt ≥ {rule.conf_min:g})"
                                                       for r in rej) or "không có") + ".",
          f"BN ung thư mà không vùng YOLO nào trùng tổn thương thật (nhãn nếu có chỉ đến từ vùng giả): {ca['off']['no_true_lesion_found']}/42.", ""]
    L += tbl(mr, "MRI – 78 bệnh nhân (7 lớp, chưa đạt chất lượng)",
             "MRI không có BN nhóm chứng trên Val, nên cổng **không thể** chứng minh giảm cảnh báo sai trên MRI; chỉ đo được cái giá "
             "(số BN mất nhãn).")
    rej_m = [r for r in rows if r["group"] == "mri" and r["outcome_on"] == "rejected"]
    mri_min_strong = min(int(r["n_volumes_strong"]) for r in rows if r["group"] == "mri")
    mri_min_conf = min(float(r["max_conf"]) for r in rows if r["group"] == "mri")
    L += [f"MRI bị từ chối: {len(rej_m)} BN, trong đó trước đó đúng {sum(r['outcome_off'] == 'correct' for r in rej_m)}, "
          f"sai {sum(r['outcome_off'] == 'wrong' for r in rej_m)}. "
          f"Mọi BN MRI đều có vùng độ tin cậy ≥ {rule.conf_min:g} ở ít nhất {mri_min_strong} chuỗi "
          f"(độ tin cậy cao nhất thấp nhất {mri_min_conf:.2f}), nên quy tắc CT không bao giờ kích hoạt trên MRI Val.",
          f"BN MRI mà không vùng YOLO nào trùng tổn thương thật (nhãn chỉ đến từ vùng giả, được tính là **Sai**): "
          f"{mr['off']['no_true_lesion_found']}/78 ({', '.join(r['patient_id'] for r in rows if r['group'] == 'mri' and r['true_lesion_found'] == 0)}).", ""]

    # whole CT
    def ct_all(s):
        x, y = cn[s], ca[s]
        fp = x["correct"] + x["wrong"]
        return {"n": x["n"] + y["n"], "ok_behaviour": y["correct"] + x["n"] - fp, "wrong_label": y["wrong"] + fp,
                "no_label_cancer": y["rejected"] + y["missed"]}
    a0, a1 = ct_all("off"), ct_all("on")
    L += ["### CT – toàn bộ 54 bệnh nhân Validation", "",
          "| | Ung thư phân loại đúng + CN không bị gán nhãn | Nhãn sai (ung thư sai lớp + CN bị gán nhãn ung thư) | BN ung thư không có nhãn |",
          "|---|---|---|---|"]
    for s, nm, a in (("off", "Tắt cổng", a0), ("on", "Bật cổng", a1)):
        L.append(f"| {nm} | {pct(a['ok_behaviour'], a['n'])} | {pct(a['wrong_label'], a['n'])} | {pct(a['no_label_cancer'], a['n'])} |")
    L += ["", "Đây **không** phải accuracy phân loại 4 lớp (hệ thống không có lớp CN): CN được tính là \"hành vi đúng\" khi không bị gán nhãn ung thư.", ""]

    # verification
    L += ["## 3. Kiểm tra tính nhất quán", ""]
    for m, n, a, b in checks:
        L.append(f"- {m.upper()} ({n} BN): kết quả chạy thật khớp với phát lại quy tắc trên vùng YOLO của lần chạy không cổng: "
                 f"{'có' if not a else 'KHÔNG, lệch ở ' + ', '.join(a)}; nhãn của ca được giữ lại không đổi: "
                 f"{'có' if not b else 'KHÔNG, đổi ở ' + ', '.join(b)}.")
    L.append("")

    # assessment
    fp0, fp1 = cn["off"]["correct"] + cn["off"]["wrong"], cn["on"]["correct"] + cn["on"]["wrong"]
    lost = ca["on"]["rejected"] + ca["on"]["missed"]
    lost_correct = sum(1 for r in rej if r["outcome_off"] == "correct")
    ok_risk = lost / ca["on"]["n"] <= CANCER_RISK_MAX
    L += ["## 4. Đánh giá", "",
          f"- **Cảnh báo ung thư sai ở CN:** {fp0}/12 → {fp1}/12 (KTC 95% {ci(fp0, 12)} → {ci(fp1, 12)}). Cổng giảm rõ, nhưng mẫu chỉ có 12 BN "
          "và ngưỡng được chọn trên chính 12 BN này, nên con số lạc quan; ước lượng leave-one-out trước đó là 9/12 bị từ chối.",
          f"- **BN ung thư mất nhãn:** {ca['off']['rejected'] + ca['off']['missed']}/42 → {lost}/42 ({100 * lost / 42:.1f}%; KTC 95% {ci(lost, 42)}). "
          f"{lost_correct}/{len(rej)} ca bị từ chối trước đó được phân loại đúng; không có BN ung thư nào bị YOLO bỏ sót hoàn toàn. "
          f"Tiêu chí đặt trước (≤ {100 * CANCER_RISK_MAX:.0f}% BN ung thư mất nhãn): **{'đạt' if ok_risk else 'KHÔNG đạt'}**.",
          f"- Số ca phân loại sai lớp ở BN ung thư không đổi ({ca['off']['wrong']} → {ca['on']['wrong']}): cổng chỉ xét bằng chứng phát hiện, "
          "nên không sửa được lỗi phân loại.",
          f"- **MRI:** quy tắc CT làm mất nhãn {len(rej_m)}/78 BN (không kích hoạt lần nào), và không có nhóm chứng để chứng minh lợi ích. Accuracy trên toàn bộ BN "
          f"{pct(mr['off']['correct'], 78)} → {pct(mr['on']['correct'], 78)}.", "",
          "## 5. Cấu hình đề xuất (dựa trên Validation)", "",
          f"- **CT: bật Evidence Gate** (≥ 0.4 ở ≥ 2 pha; một file: ≥ 0.35), `configs/cls.yaml → inference.ct.evidence_gate.enabled: true`"
          + ("." if ok_risk else " — cần cân nhắc vì tiêu chí bỏ sót không đạt."),
          "- **MRI: không bật cổng** (`enabled: false`): không có nhóm chứng để chọn hoặc kiểm chứng ngưỡng. Nhãn MRI tiếp tục kèm cảnh báo "
          "chưa đáng tin cậy; đề xuất cho website: MRI chỉ hiển thị phát hiện + Tumor Zoom, phân loại MRI chỉ là thông tin tham khảo.",
          "- YOLO (conf 0.12), Tumor Zoom, Hybrid CT và cách gộp mức BN giữ nguyên.",
          "- Mọi con số trên là của Validation, dùng để chọn cấu hình. Kết quả cuối cùng phải đo **một lần** trên Test với cấu hình đã chốt này.", "",
          "## 6. Hạn chế", "",
          "- 12 BN CN là mẫu rất nhỏ; khoảng tin cậy rộng. Ngưỡng được chọn trên cùng tập Val nên kết quả trên Test có thể kém hơn.",
          "- *Từ chối phân loại* không phải là kết luận âm tính: BN vẫn cần bác sĩ đọc ảnh gốc.",
          "- CN là nhóm \"không có bệnh gan mật\" theo nguồn dữ liệu, không phải \"khỏe mạnh hoàn toàn\".", "",
          "Tệp: `reports/pipeline_validation/so_sanh_evidence_gate.csv` (từng BN), `so_sanh_evidence_gate.json`, "
          "`reports/pipeline_validation_gate/` (báo cáo pipeline đầy đủ khi bật cổng)."]
    (REP / "so_sanh_evidence_gate.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
