// Self-contained HTML report of ONE result, generated in the browser from the backend's real values (nothing invented).
// No file name, no DICOM metadata, no original screenshot: only the pre-processed images that were analysed.
// PDF: the same report printed by the browser ("In / Lưu PDF" -> Save as PDF).
import type { AnalysisResult } from "./api";
import { detShort, fmtNum, fmtPct } from "./api";
import type { ViewerState } from "./components/Viewer";

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c] as string));
const vn = (x: number, d = 1) => x.toLocaleString("vi-VN", { maximumFractionDigits: d });

function fig(src: string, cap: string): string {
  return `<figure><img src="${src}" alt=""><figcaption>${esc(cap)}</figcaption></figure>`;
}

export function buildReport(res: AnalysisResult, st: ViewerState): string {
  const b = res.branches[0];
  const cs = res.case;
  const sel = st.sel;
  const now = new Date().toLocaleString("vi-VN");
  const det = detShort(res);
  const where = res.is_2d ? "" : ` (${b.view.volume}, lát ${b.view.slice + 1})`;
  const imgs = [
    fig(`data:image/png;base64,${b.view.png}`, `Ảnh gốc${where}`),
    fig(`data:image/png;base64,${res.annotated_png}`, `${det} khoanh vùng${where}`),
  ];
  if (sel) {
    imgs.push(fig(`data:image/png;base64,${sel.context_png}`, `Ảnh gốc (vùng cắt) – vùng ${sel.rank}${res.is_2d ? "" : `, lát đại diện ${sel.slice + 1}`}`));
    imgs.push(fig(`data:image/png;base64,${sel.local_png}`, `Tumor Zoom – vùng ${sel.rank} (độ tin cậy phát hiện cao nhất của vùng ${fmtNum(sel.confidence)})`));
  }
  const rows: [string, string][] = [
    ["Loại ảnh", `${res.cards.modality.value} – ${res.cards.modality.sub}`],
    ["Dữ liệu", res.is_2d ? "Ảnh 2D" : `Volume 3D${cs?.volumes[0]?.source ? " · " + cs.volumes[0].source : ""}${cs && cs.volumes.length > 1 ? ` · ${cs.volumes.length} ${res.modality === "ct" ? "pha" : "chuỗi"}` : ""}`],
  ];
  if (cs) {
    for (const v of cs.volumes) {
      rows.push([res.is_2d ? "Kích thước ảnh" : `Kích thước ${esc(v.name)}`,
        res.is_2d ? `${v.width} × ${v.height}` : `${v.width} × ${v.height} × ${v.n_slices}` + (v.spacing ? ` (${v.spacing.map((x) => (x == null ? "?" : vn(x, 2))).join(" × ")} mm)` : "")]);
    }
    if (cs.is_3d) {
      rows.push(["Tổng số lát cắt", `${cs.total_slices} (${det} 2D quét mọi lát${cs.duplicate_slices ? `; ${cs.duplicate_slices} lát trùng hoàn toàn dùng lại kết quả` : ""})`]);
      rows.push(["Lát có vùng nghi ngờ", String(cs.slices_with_regions)]);
      if (cs.dicom_duplicate_files) rows.push(["Tệp DICOM trùng đã bỏ", String(cs.dicom_duplicate_files)]);
    }
  }
  rows.push([res.is_2d ? "Số vùng nghi ngờ" : "Số vùng nghi ngờ 3D", String(res.n_regions)]);
  rows.push(["Mô hình phát hiện", res.details.detector ? `${res.details.detector.name}${res.details.detector.main ? " (mô hình chính)" : " (mô hình đối chứng, chỉ khoanh vùng)"} · ngưỡng ${fmtNum(res.details.detector.conf)}` : "YOLO11n (mô hình chính)"]);
  rows.push([`Độ tin cậy phát hiện cao nhất (${det})`, b.max_conf != null ? fmtNum(b.max_conf) + (cs?.top_slice ? ` – ${cs.volumes.length > 1 ? cs.top_slice.volume + ", " : ""}lát ${cs.top_slice.k + 1}` : "") : "—"]);
  if (res.details.evidence) rows.push(["Evidence Gate", `${res.details.evidence.passed ? "Đủ bằng chứng" : "Chưa đủ bằng chứng"} (${res.details.evidence.rule})`]);
  rows.push(["Thời gian phân tích", `${vn(res.timings_s.total ?? 0, 1)} giây trên ${res.device || "máy chủ"}`]);
  const scores = res.scores.length
    ? `<h2>Xác suất phân loại (Hybrid CT)</h2><table>${res.scores.map((s) =>
      `<tr><td style="width:22%">${esc(s.name)}</td><td><div class="bt"><div style="width:${Math.max(1.5, 100 * s.prob)}%"></div></div></td><td class="n">${fmtPct(s.prob)}</td></tr>`).join("")}</table>
      <p class="muted">Đầu ra softmax của mô hình, không phải xác suất y khoa người bệnh mắc bệnh.</p>`
    : "";
  const r3 = cs?.regions3d ?? [];
  const regs3d = r3.length
    ? `<h2>Vùng nghi ngờ 3D (${r3.length})</h2><table><tr><td><b>Vùng</b></td><td><b>${cs && cs.volumes.length > 1 ? "Pha / chuỗi" : ""}</b></td><td><b>Lát</b></td><td><b>Đại diện</b></td><td><b>Số lát có khung</b></td><td><b>Độ tin cậy cao nhất</b></td><td><b>Khối bao</b></td></tr>${
      r3.slice(0, 60).map((r) => `<tr><td>Vùng ${r.num}</td><td>${cs && cs.volumes.length > 1 ? esc(r.volume) : ""}</td><td>${r.k_first + 1}–${r.k_last + 1}</td><td>${r.best_slice + 1}</td><td>${r.n_slices}</td><td>${fmtNum(r.max_conf)}</td><td>${r.size_mm ? r.size_mm.map((x) => vn(x, 0)).join(" × ") + " mm" : `${vn(r.bbox_vox.x_max - r.bbox_vox.x_min, 0)} × ${vn(r.bbox_vox.y_max - r.bbox_vox.y_min, 0)} px × ${r.k_last - r.k_first + 1} lát`}</td></tr>`).join("")}</table>
      <p class="muted">Vùng 3D = các khung ${det} 2D trên những lát liên tiếp được nối theo quy tắc của pipeline đã khóa. Khối bao là bao của các khung 2D, không phải hình dạng hay thể tích khối u; một khối u có thể bị tách thành nhiều vùng. Video quét lát (MP4) được xuất riêng trên website.</p>`
    : "";
  const sus = cs && cs.is_3d && cs.suspicious_slices.length
    ? `<h2>Lát cắt có vùng nghi ngờ (${cs.suspicious_slices.length})</h2><p class="muted">${cs.suspicious_slices.slice(0, 120).map((s) =>
      `${cs.volumes.length > 1 ? esc(s.volume) + " " : ""}lát ${s.k + 1} (${fmtNum(s.conf)})`).join(" · ")}${cs.suspicious_slices.length > 120 ? " · …" : ""}</p>
      <p class="muted">Số trong ngoặc: độ tin cậy phát hiện cao nhất trên lát (${det}), không phải xác suất ung thư.</p>`
    : "";
  return `<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Báo cáo kết quả – LiverCancer AI</title>
<style>body{font-family:"Be Vietnam Pro","Segoe UI",Arial,sans-serif;color:#0f1b2d;max-width:960px;margin:24px auto;padding:0 16px;line-height:1.5}
h1{font-size:22px;margin:0 0 4px;color:#0b3d91}h2{font-size:16px;margin:22px 0 8px;color:#0b3d91;border-bottom:1px solid #d9e3f0;padding-bottom:4px}
.muted{color:#44546a;font-size:12.5px}.concl{border:1px solid #f5c2bd;background:#fdf2f1;border-radius:10px;padding:12px 14px;margin-top:14px}
.concl.clear,.concl.pending,.concl.info{border-color:#d9e3f0;background:#f4f7fc}.concl b{font-size:18px;color:#b42318}.concl.clear b,.concl.info b{color:#0b3d91}.concl.pending b{color:#8a4b00}
.imgs{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;break-inside:avoid}figure img{width:100%;border-radius:8px;background:#000}
figcaption{font-size:12px;color:#44546a;margin-top:4px}table{width:100%;border-collapse:collapse}td{padding:5px 6px;border-bottom:1px solid #edf2f9;font-size:13.5px;vertical-align:top}
td.n{text-align:right;white-space:nowrap}.bt{background:#eaf1fb;border-radius:4px}.bt div{height:12px;background:#1d5fd6;border-radius:4px}
@page{size:A4;margin:14mm}@media print{body{margin:0;max-width:none}h2{break-after:avoid}}@media(max-width:640px){.imgs{grid-template-columns:1fr}}</style></head><body>
<h1>Báo cáo kết quả phân tích – LiverCancer AI</h1><div class="muted">Tạo lúc ${esc(now)} · hệ thống nghiên cứu, không thay thế chẩn đoán của bác sĩ</div>
<div class="concl ${res.status === "ok" ? "" : res.status === "insufficient_evidence" ? "pending" : res.status === "no_lesion_detected" ? "clear" : "info"}">
<div class="muted">Kết luận AI</div><b>${esc(res.cards.result.title || "")}</b>${res.label_name ? `<div>${esc(res.label_name)}</div>` : ""}<div class="muted">${esc(res.cards.result.sub)}</div></div>
${scores}
<h2>Thông tin ca phân tích</h2><table>${rows.map(([k, v]) => `<tr><td style="width:36%">${esc(k)}</td><td>${esc(v)}</td></tr>`).join("")}</table>
<h2>Hình ảnh</h2><div class="imgs">${imgs.join("")}</div>
${regs3d}
${sus}
<h2>Nhận xét</h2>${res.commentary.map((t) => `<p>${esc(t)}</p>`).join("")}${res.notes.map((t) => `<p class="muted">• ${esc(t)}</p>`).join("")}
<h2>Giới hạn</h2>${res.details.limitations.map((t) => `<p class="muted">• ${esc(t)}</p>`).join("")}
<p class="muted" style="margin-top:20px;border-top:1px solid #d9e3f0;padding-top:8px">${esc(res.footnote)}</p></body></html>`;
}

function save(blob: Blob, name: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

// PNG of the slice on screen with exactly the boxes the viewer draws (same coordinates, same labels).
export function downloadSlicePng(res: AnalysisResult, st: ViewerState) {
  const v = st.view;
  const img = new Image();
  img.onload = () => {
    const c = document.createElement("canvas");
    c.width = v.width; c.height = v.height;
    const g = c.getContext("2d");
    if (!g) return;
    g.drawImage(img, 0, 0, v.width, v.height);
    const fs = Math.max(11, Math.round(v.width * 0.026));
    g.font = `600 ${fs}px "Be Vietnam Pro", Arial, sans-serif`;
    const rank = new Map(st.branch.regions.map((r) => [r.id, r.rank]));
    for (const b of v.boxes) {
      const main = st.sel?.id === b.region;
      const col = main ? "#ef4444" : "#f59e0b";
      const [x0, y0, x1, y1] = b.box;
      g.strokeStyle = col; g.lineWidth = main ? 2.5 : 1.8;
      g.setLineDash(main ? [] : [5, 4]);
      g.strokeRect(x0, y0, x1 - x0, y1 - y0);
      g.setLineDash([]);
      const t = `${rank.has(b.region) ? `Vùng ${rank.get(b.region)}` : "Vùng khác"} · ${fmtNum(b.conf)}`;
      const tw = g.measureText(t).width + fs * 0.6;
      const ty = y0 - fs * 1.35 < 0 ? y1 + 2 : y0 - fs * 1.35;
      g.fillStyle = col; g.fillRect(x0, ty, tw, fs * 1.3);
      g.fillStyle = "#fff"; g.fillText(t, x0 + fs * 0.3, ty + fs * 0.98);
    }
    const name = res.is_2d ? "livercancer-ai-anh-2d.png" : `livercancer-ai-${v.volume.replace(/\s+/g, "-").toLowerCase()}-lat-${v.slice + 1}.png`;
    c.toBlob((bl) => bl && save(bl, name), "image/png");
  };
  img.src = `data:image/png;base64,${v.png}`;
}

export function downloadReport(res: AnalysisResult, st: ViewerState) {
  const blob = new Blob([buildReport(res, st)], { type: "text/html;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "bao-cao-livercancer-ai.html";
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

// Print the same report from a hidden frame: the browser's print dialog offers "Save as PDF".
export function printReport(res: AnalysisResult, st: ViewerState) {
  const f = document.createElement("iframe");
  f.setAttribute("aria-hidden", "true");
  f.style.cssText = "position:fixed;right:0;bottom:0;width:0;height:0;border:0";
  document.body.appendChild(f);
  const doc = f.contentWindow?.document;
  if (!doc) { f.remove(); return; }
  doc.open(); doc.write(buildReport(res, st)); doc.close();
  const go = () => {
    window.dispatchEvent(new CustomEvent("lcai-print", { detail: { chars: doc.documentElement.outerHTML.length, images: doc.images.length } }));
    try { f.contentWindow?.focus(); f.contentWindow?.print(); } catch { /* ignore */ }
    setTimeout(() => f.remove(), 60000);
  };
  const imgs = Array.from(doc.images);
  Promise.all(imgs.map((im) => (im.complete ? Promise.resolve() : new Promise((r) => { im.onload = im.onerror = () => r(null); })))).then(() => setTimeout(go, 50));
}
