// Right column: AI conclusion, Hybrid CT scores (only when the backend returned them), case information, Evidence
// Gate, downloads. Every value comes from the backend result; detection confidence is never shown as a probability.
import type { AnalysisResult } from "../api";
import { fmtNum, fmtPct } from "../api";
import { downloadReport, downloadSlicePng, printReport } from "../report";
import type { ViewerState } from "./Viewer";
import VideoExport from "./VideoExport";
import SigmoidChart from "./SigmoidChart";

const YOLO_CONF = 0.12;   // locked detection threshold (configs/yolo_inference.yaml), same value as the "Ngưỡng phát hiện YOLO" row

const vn = (x: number, d = 1) => x.toLocaleString("vi-VN", { maximumFractionDigits: d });

function Row({ k, label, value, sub }: { k: string; label: string; value: string; sub?: string }) {
  return (
    <div className="info-row" data-k={k}>
      <span className="l">{label}</span>
      <span className="r"><span className="v">{value}</span>{sub && <span className="s">{sub}</span>}</span>
    </div>
  );
}

export default function ResultPanel({ res, state, onReset, jobId }: { res: AnalysisResult; state: ViewerState | null; onReset: () => void; jobId: string | null }) {
  const b = res.branches[0];
  const cs = res.case;
  const st: ViewerState = state ?? { branch: b, view: b.view, sel: b.regions[0] ?? null, sel3d: null, volume: b.view.volume };
  const r3 = st.sel3d;
  const n = res.n_regions;
  const tone = res.status === "ok" ? "alert" : res.status === "insufficient_evidence" ? "pending" : res.status === "no_lesion_detected" ? "clear" : "info";
  const badge = { ok: "Có vùng nghi ngờ", insufficient_evidence: "Chưa đủ bằng chứng", no_lesion_detected: "Không phát hiện", detection_only: "Có vùng nghi ngờ" }[res.status];
  const ev = res.details.evidence;
  const v0 = cs?.volumes[0];
  const sameSize = cs ? cs.volumes.every((v) => v.width === v0?.width && v.height === v0?.height) : true;
  const units = res.modality === "ct" ? "pha" : "chuỗi";
  const top = cs?.top_slice;
  const sp = v0?.spacing;
  const strong = res.details.strong_conf ?? null;
  const nStrong = strong != null ? b.all_regions.filter((r) => r.confidence >= strong).length : 0;

  return (
    <section className="card rp" aria-label="Kết quả phân tích">
      <div className="rp-head">
        <h2>Kết quả phân tích</h2>
        <span className={`badge ${tone}`}>{badge}</span>
      </div>

      <h3 className="rp-h">Kết luận AI{res.is_2d ? "" : " (toàn ca)"}</h3>
      <div className={`concl ${tone}`}>
        {(res.status === "ok" || res.status === "detection_only") && <div className="concl-line">● Có vùng nghi ngờ tổn thương gan</div>}
        {res.status === "insufficient_evidence" && <div className="concl-line">● Có vùng nghi ngờ, nhưng chưa đủ bằng chứng để phân loại</div>}
        {res.status === "ok" && <div className="concl-pre">Hybrid CNN + Transformer (CT) phân loại:</div>}
        <div className="concl-title">{res.cards.result.title}</div>
        {res.status === "ok" && res.label_name && <div className="concl-name">{res.label_name}</div>}
        {res.status === "ok" && res.label && (
          <div className="concl-score">Điểm dự đoán của mô hình nghiên cứu: <b>{fmtPct(res.scores.find((s) => s.code === res.label)?.prob ?? 0)}</b></div>
        )}
        <div className="concl-sub">
          {res.status === "ok" && (res.is_2d ? "Nhãn cho ảnh này, không phải chẩn đoán." : `Nhãn cho toàn ca, gộp ${n} vùng nghi ngờ (cách gộp chọn trên Validation, đánh giá trên Test). Không phải chẩn đoán.`)}
          {res.status === "insufficient_evidence" && "Evidence Gate từ chối phân loại; hệ thống không đưa ra nhãn."}
          {res.status === "no_lesion_detected" && `Ở ngưỡng phát hiện của mô hình. Không đồng nghĩa với gan bình thường.`}
          {res.status === "detection_only" && "MRI: chỉ phát hiện và phóng to vùng nghi ngờ, không phân loại loại tổn thương."}
        </div>
      </div>

      {res.scores.length > 0 && (
        <>
          <h3 className="rp-h">Điểm dự đoán của mô hình (Hybrid CT{res.is_2d ? "" : ", toàn ca"})</h3>
          <div className="bars" role="list">
            {res.scores.map((s) => {
              const isTop = s.code === res.label;
              return (
                <div key={s.code} className={`bar-row${isTop ? " top" : ""}`} role="listitem" title={s.full_name}>
                  <span className="bar-name">{s.name}</span>
                  <div className="bar-track"><div className={`bar-fill c-${s.code.toLowerCase()}`} style={{ width: `${Math.max(1.5, 100 * s.prob)}%` }} /></div>
                  <span className="bar-val tnum">{fmtPct(s.prob)}</span>
                </div>
              );
            })}
          </div>
          <p className="small muted note">Điểm softmax của mô hình nghiên cứu, không phải xác suất người bệnh mắc ung thư.</p>
        </>
      )}
      {res.scores.length === 0 && b.chart.kind === "detection" && (
        <>
          <h3 className="rp-h">Độ tin cậy phát hiện theo vùng (YOLO)</h3>
          <div className="bars" role="list">
            {b.chart.items.map((it, i) => (
              <div key={it.label} className={`bar-row det${i === 0 ? " top" : ""}`} role="listitem" title={it.full}>
                <span className="bar-name">{it.label}</span>
                <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.max(1.5, 100 * it.value)}%` }} /></div>
                <span className="bar-val tnum">{fmtNum(it.value)}</span>
              </div>
            ))}
          </div>
          <p className="small muted note">Độ tin cậy phát hiện (0–1) của từng vùng: mức chắc chắn của YOLO, không phải xác suất ung thư.</p>
        </>
      )}

      {b.all_regions.length > 0 && (
        <>
          <h3 className="rp-h">Hàm sigmoid – độ tin cậy phát hiện YOLO</h3>
          <SigmoidChart confs={b.all_regions.map((r) => r.confidence)} threshold={YOLO_CONF} strong={strong}
            sel={r3 ? r3.max_conf : st.sel ? st.sel.confidence : null}
            selLabel={r3 ? `Vùng ${r3.num}` : st.sel ? `Vùng ${st.sel.rank}` : undefined} />
        </>
      )}

      <h3 className="rp-h">Tổng hợp {res.is_2d ? "ảnh" : "theo ca"}</h3>
      <div className="info">
        <Row k="modality" label="Loại ảnh" value={res.cards.modality.value || ""} sub={res.cards.modality.sub} />
        <Row k="data" label="Dữ liệu" value={res.is_2d ? "Ảnh 2D" : `Volume 3D${v0?.source ? ` · ${v0.source}` : ""}`}
          sub={!res.is_2d && cs && cs.volumes.length > 1 ? `${cs.volumes.length} ${units}, phân tích riêng từng ${units}` : undefined} />
        {cs?.is_3d && (
          <>
            <Row k="slices" label="Tổng số lát cắt" value={vn(cs.total_slices, 0)}
              sub={cs.duplicate_slices > 0 ? `YOLO chạy trên ${vn(cs.yolo_slices, 0)} lát khác nhau; ${cs.duplicate_slices} lát trùng hoàn toàn dùng lại kết quả`
                : `YOLO 2D đã quét toàn bộ ${vn(cs.total_slices, 0)} lát`} />
            <Row k="suspicious" label="Lát có vùng nghi ngờ" value={vn(cs.slices_with_regions, 0)} sub="lát có ít nhất một khung YOLO" />
          </>
        )}
        <Row k="regions" label={res.is_2d ? "Vùng nghi ngờ" : "Vùng nghi ngờ 3D"} value={String(n)}
          sub={n === 0 ? undefined : [
            !res.is_2d ? "sau khi nối khung qua các lát liên tiếp" : "",
            strong != null ? `${nStrong} vùng có độ tin cậy ≥ ${fmtNum(strong)} (mức vùng mạnh của Evidence Gate), ${n - nStrong} vùng thấp hơn` : "",
          ].filter(Boolean).join("; ")} />
        <Row k="conf" label={`Độ tin cậy phát hiện cao nhất${res.is_2d ? "" : " (cả ca)"}`} value={b.max_conf != null ? fmtNum(b.max_conf) : "—"}
          sub={top ? `YOLO · ${cs && cs.volumes.length > 1 ? top.volume + ", " : ""}lát ${top.k + 1}` : b.max_conf != null ? "YOLO" : undefined} />
        {res.modality === "ct" && (
          <Row k="gate" label="Evidence Gate" value={ev ? (ev.passed ? "Đủ bằng chứng" : "Chưa đủ bằng chứng") : "—"}
            sub={ev ? ev.rule : "không có vùng để đánh giá"} />
        )}
        <Row k="time" label="Thời gian phân tích" value={`${vn(res.timings_s.total ?? 0, 1)} giây`} sub={res.device ? `trên ${res.device}` : undefined} />
      </div>

      {!res.is_2d && r3 && (
        <>
          <h3 className="rp-h">Vùng đang chọn</h3>
          <div className="info sel3d">
            <Row k="sel_region" label="Vùng" value={`Vùng ${r3.num}${cs && cs.volumes.length > 1 ? ` · ${r3.volume}` : ""}`}
              sub={strong != null && r3.max_conf < strong ? "độ tin cậy thấp" : undefined} />
            <Row k="sel_range" label="Phạm vi lát" value={`${r3.k_first + 1}–${r3.k_last + 1}`}
              sub={`${r3.n_slices} lát có khung${r3.gap_slices ? `, ${r3.gap_slices} lát trống ở giữa` : ""}; lát đại diện ${(st.sel && st.sel.id === r3.id ? st.sel.slice : r3.best_slice) + 1}`} />
            <Row k="sel_conf" label="Độ tin cậy cao nhất" value={fmtNum(r3.max_conf)} sub="YOLO, trên các lát của vùng" />
            <Row k="sel_box" label="Khối bao 3D" value={r3.size_mm ? `${r3.size_mm.map((x) => fmtNum(x, 0)).join(" × ")} mm`
              : `${fmtNum(r3.bbox_vox.x_max - r3.bbox_vox.x_min, 0)} × ${fmtNum(r3.bbox_vox.y_max - r3.bbox_vox.y_min, 0)} px × ${r3.k_last - r3.k_first + 1} lát`}
              sub="bao của các khung YOLO 2D, không phải kích thước hay thể tích khối u" />
          </div>
        </>
      )}

      <details className="mini more">
        <summary>Thông tin kỹ thuật</summary>
        <div className="info">
        {v0 && (
          <Row k="size" label={res.is_2d ? "Kích thước ảnh" : "Kích thước volume"}
            value={res.is_2d ? `${v0.width} × ${v0.height}` : sameSize ? `${v0.width} × ${v0.height} × ${v0.n_slices}` : `${v0.width} × ${v0.height}…`}
            sub={res.is_2d && cs?.original_size && (cs.original_size[0] !== v0.width || cs.original_size[1] !== v0.height)
              ? `ảnh tải lên ${cs.original_size[0]} × ${cs.original_size[1]}, đã chuẩn hóa` : !res.is_2d && cs && cs.volumes.length > 1 ? `${units} 1` : undefined} />
        )}
        {!res.is_2d && sp && sp.some((x) => x != null) && (
          <Row k="spacing" label="Khoảng cách điểm ảnh" value={`${sp.map((x) => (x == null ? "?" : vn(x, 2))).join(" × ")} mm`} sub={`${units} 1`} />
        )}
        {cs && cs.dicom_duplicate_files > 0 && <Row k="dupfiles" label="Tệp DICOM trùng đã bỏ" value={String(cs.dicom_duplicate_files)} />}
        <Row k="thr" label="Ngưỡng phát hiện YOLO" value="0,12 (đã khóa, chọn trên Validation)" />
        </div>
        {cs && cs.volumes.length > 1 && (
          <div className="table-wrap" style={{ marginTop: 8 }}>
            <table>
              <thead><tr><th>{units === "pha" ? "Pha" : "Chuỗi"}</th><th className="num">Lát</th><th className="num">Lát nghi ngờ</th><th className="num">Vùng</th></tr></thead>
              <tbody>{cs.volumes.map((v) => (
                <tr key={v.name}><td>{v.name}</td><td className="num">{v.n_slices}</td><td className="num">{v.slices_with_regions}</td><td className="num">{v.n_regions}</td></tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </details>

      <details className="mini more">
        <summary>Nhận xét</summary>
        <div className="comment small">
          {res.commentary.map((t, i) => <p key={i}>{t}</p>)}
          {res.notes.length > 0 && <ul className="notes">{res.notes.map((t, i) => <li key={i}>{t}</li>)}</ul>}
        </div>
      </details>

      <div className="rp-actions">
        {!res.is_2d && <div className="wide"><VideoExport jobId={jobId} branch={b.key} volume={st.volume} /></div>}
        <button className="btn btn-ghost wide" onClick={() => downloadSlicePng(res, st)}>Tải ảnh {res.is_2d ? "khoanh vùng" : `lát ${st.view.slice + 1}`} (PNG)</button>
        <button className="btn btn-ghost" onClick={() => downloadReport(res, st)}>Tải báo cáo (HTML)</button>
        <button className="btn btn-ghost" onClick={() => printReport(res, st)} title="Mở hộp thoại In của trình duyệt; chọn &quot;Lưu dưới dạng PDF&quot;">In / Lưu PDF</button>
        <button className="btn btn-primary" onClick={onReset}>Phân tích ảnh khác</button>
      </div>
      <p className="footnote">{res.footnote}</p>
    </section>
  );
}
