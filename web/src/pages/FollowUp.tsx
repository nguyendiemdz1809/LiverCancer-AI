// "Hồ sơ theo dõi": results saved in this browser (../history.ts), grouped by record code, and a side-by-side
// comparison of two exams. Only numbers the AI returned are shown; differences are stated neutrally (more / fewer,
// larger / smaller). The page never concludes "ổn định", "tiến triển" or "đáp ứng": that is a clinical judgement
// (RECIST / mRECIST on the radiologist's measurements) and the models were never validated for follow-up.
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { fmtNum, fmtPct } from "../api";
import type { Visit } from "../history";
import { deleteCode, deleteVisit, exportJson, fmtDate, historyAvailable, importJson, listVisits } from "../history";

const vn = (x: number, d = 0) => x.toLocaleString("vi-VN", { maximumFractionDigits: d });
const modVi = (v: Visit) => `${v.modality.toUpperCase()}${v.is_2d ? " 2D" : " 3D"}`;

function concl(v: Visit): string {
  if (v.status === "ok") return `Nghi ngờ ${v.label ?? ""}`.trim();
  if (v.status === "insufficient_evidence") return "Chưa đủ cơ sở để phân loại";
  if (v.status === "no_lesion_detected") return "Không phát hiện vùng nghi ngờ";
  return `Phát hiện ${v.n_regions} vùng nghi ngờ (không phân loại)`;
}
const topRegion = (v: Visit) => v.regions[0] ?? null;
const sizeTxt = (v: Visit) => {
  const r = topRegion(v);
  if (!r) return "—";
  return r.size_mm ? `${r.size_mm.map((x) => vn(x)).join(" × ")} mm` : r.size_vox ?? "—";
};
const maxMm = (v: Visit) => { const r = topRegion(v); return r?.size_mm ? Math.max(...r.size_mm.slice(0, 2)) : null; };

function delta(a: number | null, b: number | null, unit: string, d = 0, pct = false): string {
  if (a == null || b == null) return "—";
  const x = b - a;
  if (Math.abs(x) < (d ? 10 ** -d / 2 : 0.5)) return "không đổi";
  const s = `${x > 0 ? "tăng" : "giảm"} ${vn(Math.abs(x), d)}${unit}`;
  return pct && a > 0 ? `${s} (${x > 0 ? "+" : "−"}${vn((100 * Math.abs(x)) / a)}%)` : s;
}

function Compare({ a, b }: { a: Visit; b: Visit }) {
  const warn: string[] = [];
  if (a.modality !== b.modality || a.is_2d !== b.is_2d) warn.push("Hai lần khác loại ảnh (CT/MRI hoặc 2D/3D): các số đo không so sánh trực tiếp được.");
  if (a.detector !== b.detector) warn.push(`Hai lần dùng mô hình phát hiện khác nhau (${a.detector} và ${b.detector}): chênh lệch có thể do mô hình, không phải do người bệnh.`);
  if (a.modality === b.modality && a.volumes.join("|") !== b.volumes.join("|")) warn.push("Hai lần có pha / chuỗi ảnh khác nhau: số vùng và độ tin cậy có thể khác vì khác dữ liệu đầu vào.");
  if (a.status === "ok" && b.status === "ok" && a.label !== b.label) warn.push("Nhãn phân loại khác nhau giữa hai lần. Thể ung thư thường không đổi theo thời gian; chênh lệch này nhiều khả năng do mô hình, cần bác sĩ đánh giá.");
  const rows: [string, string, string, string][] = [
    ["Ngày khám", fmtDate(a.date), fmtDate(b.date), (() => { const d = (Date.parse(b.date) - Date.parse(a.date)) / 86400000; return Number.isFinite(d) ? `${vn(d)} ngày` : "—"; })()],
    ["Loại ảnh", modVi(a), modVi(b), a.modality === b.modality && a.is_2d === b.is_2d ? "giống nhau" : "khác nhau"],
    ["Mô hình phát hiện", a.detector, b.detector, a.detector === b.detector ? "giống nhau" : "khác nhau"],
    ["Kết luận AI", concl(a), concl(b), a.status === b.status && a.label === b.label ? "giống nhau" : "khác nhau"],
    ["Số vùng nghi ngờ", String(a.n_regions), String(b.n_regions), delta(a.n_regions, b.n_regions, " vùng")],
    ...(a.n_strong != null || b.n_strong != null
      ? [["Vùng mạnh (Evidence Gate)", a.n_strong == null ? "—" : String(a.n_strong), b.n_strong == null ? "—" : String(b.n_strong), delta(a.n_strong, b.n_strong, " vùng")] as [string, string, string, string]]
      : []),
    ["Độ tin cậy phát hiện cao nhất", a.max_conf == null ? "—" : fmtNum(a.max_conf), b.max_conf == null ? "—" : fmtNum(b.max_conf), delta(a.max_conf, b.max_conf, "", 2)],
    ["Khối bao của vùng mạnh nhất", sizeTxt(a), sizeTxt(b), delta(maxMm(a), maxMm(b), " mm (cạnh lớn nhất trong lát)", 0, true)],
  ];
  const codes = [...new Set([...a.scores, ...b.scores].map((s) => s.code))];
  return (
    <section className="card fu-compare" aria-label="So sánh hai lần khám">
      <h2 className="fu-h">So sánh lần {fmtDate(a.date)} và lần {fmtDate(b.date)}</h2>
      {warn.length > 0 && <div className="notice warn" role="note"><ul>{warn.map((w) => <li key={w}>{w}</li>)}</ul></div>}
      <div className="fu-thumbs">
        {[a, b].map((v, i) => (
          <figure key={v.id}>
            {v.thumb ? <img src={v.thumb} alt={`Ảnh khoanh vùng lần ${i + 1}`} /> : <div className="fu-noimg">Không có ảnh</div>}
            <figcaption>{i === 0 ? "Lần trước" : "Lần sau"} · {fmtDate(v.date)} · {v.detector}</figcaption>
          </figure>
        ))}
      </div>
      <div className="table-wrap">
        <table className="fu-table">
          <thead><tr><th>Chỉ số</th><th>Lần trước</th><th>Lần sau</th><th>Thay đổi</th></tr></thead>
          <tbody>
            {rows.map(([k, x, y, d]) => <tr key={k}><td>{k}</td><td>{x}</td><td>{y}</td><td className="fu-delta">{d}</td></tr>)}
            {codes.map((c) => {
              const x = a.scores.find((s) => s.code === c), y = b.scores.find((s) => s.code === c);
              return <tr key={c}><td>Điểm Hybrid {x?.name ?? y?.name}</td><td>{x ? fmtPct(x.prob) : "—"}</td><td>{y ? fmtPct(y.prob) : "—"}</td>
                <td className="fu-delta">{x && y ? delta(100 * x.prob, 100 * y.prob, " điểm %", 1) : "—"}</td></tr>;
            })}
          </tbody>
        </table>
      </div>
      <div className="notice info small fu-disclaimer">
        <p style={{ margin: 0 }}><b>Đọc kết quả so sánh thế nào?</b> Bảng chỉ đặt cạnh nhau các con số do AI trả về ở hai lần phân tích.
          Hệ thống <b>không kết luận</b> bệnh "ổn định", "tiến triển" hay "đáp ứng điều trị", vì:</p>
        <ul>
          <li>mô hình được huấn luyện và kiểm định trên dữ liệu chỉ có <b>một lần chụp</b> mỗi bệnh nhân, chưa từng được kiểm chứng cho việc theo dõi theo thời gian;</li>
          <li>khối bao là bao của các khung phát hiện, <b>không phải kích thước hay thể tích khối u</b> (có thể lớn hơn tổn thương nhiều lần);</li>
          <li>khác máy chụp, khác pha thuốc cản quang hay khác tư thế đều làm số đo thay đổi.</li>
        </ul>
        <p style={{ margin: 0 }}>Đánh giá tiến triển phải do bác sĩ thực hiện trên ảnh gốc theo tiêu chuẩn lâm sàng (ví dụ RECIST hoặc mRECIST).</p>
      </div>
    </section>
  );
}

export default function FollowUp() {
  const [ok, setOk] = useState<boolean | null>(null);
  const [visits, setVisits] = useState<Visit[]>([]);
  const [params, setParams] = useSearchParams();
  const code = params.get("ma") ?? "";
  const [pick, setPick] = useState<string[]>([]);
  const [msg, setMsg] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const reload = async () => setVisits(await listVisits().catch(() => []));
  useEffect(() => { (async () => { const a = await historyAvailable(); setOk(a); if (a) await reload(); })(); }, []);

  const codes = useMemo(() => {
    const m = new Map<string, Visit[]>();
    for (const v of visits) m.set(v.code, [...(m.get(v.code) ?? []), v]);
    return [...m.entries()].sort((x, y) => (y[1][y[1].length - 1].date).localeCompare(x[1][x[1].length - 1].date));
  }, [visits]);
  const mine = useMemo(() => visits.filter((v) => v.code === code), [visits, code]);
  useEffect(() => { setPick(mine.slice(-2).map((v) => v.id)); }, [code, mine.length]);   // default: the two latest exams

  const chosen = mine.filter((v) => pick.includes(v.id));
  const toggle = (id: string) => setPick((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id].slice(-2)));
  const download = async (vs: Visit[], name: string) => {
    const url = URL.createObjectURL(await exportJson(vs));
    const a = document.createElement("a");
    a.href = url; a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  };
  const onImport = async (f: File | undefined) => {
    if (!f) return;
    try { const r = await importJson(f); setMsg(`Đã nhập ${r.added} lần khám${r.skipped ? `, bỏ qua ${r.skipped} (trùng hoặc không hợp lệ)` : ""}.`); await reload(); }
    catch (e) { setMsg((e as Error).message || "Không đọc được tệp."); }
    if (fileRef.current) fileRef.current.value = "";
  };

  return (
    <div className="page fu-page">
      <div className="container wide">
        <div className="fu-head">
          <div>
            <h1>Hồ sơ theo dõi</h1>
            <p className="muted">Lưu kết quả phân tích theo <b>mã hồ sơ</b> và so sánh các lần khám. Dữ liệu chỉ nằm trên trình duyệt của máy này
              (không gửi lên máy chủ); xóa dữ liệu trình duyệt sẽ mất hồ sơ, nên hãy <b>xuất tệp</b> để lưu trữ hoặc chuyển sang máy khác.</p>
          </div>
          <div className="fu-tools">
            <button className="btn btn-ghost small" disabled={!visits.length} onClick={() => download(visits, `ho-so-theo-doi-${new Date().toISOString().slice(0, 10)}.json`)}>Xuất tất cả (.json)</button>
            <button className="btn btn-ghost small" disabled={ok === false} onClick={() => fileRef.current?.click()}>Nhập tệp</button>
            <input ref={fileRef} type="file" accept=".json,application/json" hidden onChange={(e) => onImport(e.target.files?.[0])} />
          </div>
        </div>
        {msg && <div className="notice info small" role="status" style={{ marginBottom: 12 }}>{msg}</div>}
        {ok === false && <div className="notice danger">Trình duyệt này không cho lưu dữ liệu (chế độ ẩn danh hoặc đã chặn lưu trữ trang web), nên không dùng được hồ sơ theo dõi.</div>}
        {ok && !visits.length && (
          <div className="card fu-empty">
            <b>Chưa có hồ sơ nào.</b>
            <p className="muted">Sau khi phân tích một ca ở trang <Link to="/phan-tich">Phân tích ảnh</Link>, nhập mã hồ sơ và ngày khám ở mục
              “Lưu vào hồ sơ theo dõi” bên dưới kết quả. Lần khám sau, lưu vào cùng mã để so sánh.</p>
          </div>
        )}
        {visits.length > 0 && (
          <div className="fu-grid">
            <aside className="card fu-codes" aria-label="Danh sách mã hồ sơ">
              <h2 className="fu-h">Mã hồ sơ ({codes.length})</h2>
              <ul>
                {codes.map(([c, vs]) => (
                  <li key={c}><button className={c === code ? "on" : ""} onClick={() => setParams({ ma: c })}>
                    <b>{c}</b><span className="small muted">{vs.length} lần khám · gần nhất {fmtDate(vs[vs.length - 1].date)}</span>
                  </button></li>
                ))}
              </ul>
            </aside>
            <div className="fu-main">
              {!code || !mine.length ? (
                <div className="card fu-empty"><p className="muted" style={{ margin: 0 }}>Chọn một mã hồ sơ bên trái để xem các lần khám.</p></div>
              ) : (
                <>
                  <section className="card">
                    <div className="fu-row">
                      <h2 className="fu-h">Hồ sơ {code}: {mine.length} lần khám</h2>
                      <div className="fu-tools">
                        <button className="btn btn-ghost small" onClick={() => download(mine, `ho-so-${code.replace(/[^\w-]+/g, "_")}.json`)}>Xuất hồ sơ này</button>
                        <button className="btn btn-ghost small" onClick={async () => {
                          if (!window.confirm(`Xóa toàn bộ ${mine.length} lần khám của hồ sơ ${code} trên trình duyệt này?`)) return;
                          await deleteCode(code); setParams({}); await reload();
                        }}>Xóa hồ sơ</button>
                      </div>
                    </div>
                    <p className="small muted" style={{ margin: "0 0 8px" }}>Chọn hai lần khám để so sánh (mặc định: hai lần gần nhất).</p>
                    <div className="table-wrap">
                      <table className="fu-table">
                        <thead><tr><th>So sánh</th><th>Ngày khám</th><th>Loại ảnh</th><th>Mô hình</th><th>Kết luận AI</th><th className="num">Số vùng</th><th className="num">Độ tin cậy cao nhất</th><th>Khối bao vùng mạnh nhất</th><th>Ghi chú</th><th></th></tr></thead>
                        <tbody>
                          {mine.map((v) => (
                            <tr key={v.id} className={pick.includes(v.id) ? "picked" : ""}>
                              <td><input type="checkbox" aria-label={`Chọn lần khám ${fmtDate(v.date)}`} checked={pick.includes(v.id)} onChange={() => toggle(v.id)} /></td>
                              <td>{fmtDate(v.date)}</td><td>{modVi(v)}</td><td>{v.detector}</td><td>{concl(v)}</td>
                              <td className="num">{v.n_regions}</td><td className="num">{v.max_conf == null ? "—" : fmtNum(v.max_conf)}</td>
                              <td>{sizeTxt(v)}</td><td className="small">{v.note || ""}</td>
                              <td><button className="linklike small" onClick={async () => {
                                if (!window.confirm(`Xóa lần khám ngày ${fmtDate(v.date)}?`)) return;
                                await deleteVisit(v.id); await reload();
                              }}>Xóa</button></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </section>
                  {chosen.length === 2 ? <Compare a={chosen[0]} b={chosen[1]} />
                    : <div className="card fu-empty"><p className="muted" style={{ margin: 0 }}>{mine.length < 2 ? "Hồ sơ mới có một lần khám. Lưu thêm lần khám sau vào cùng mã để so sánh." : "Chọn đúng hai lần khám để so sánh."}</p></div>}
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
