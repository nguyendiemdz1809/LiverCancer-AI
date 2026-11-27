// "Lưu vào hồ sơ theo dõi": saves a summary of this result in the browser (see ../history.ts), for comparison with
// later or earlier exams on the "Hồ sơ theo dõi" page. Nothing is sent to the server.
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import type { AnalysisResult } from "../api";
import { historyAvailable, listVisits, normCode, saveVisit, today, visitFrom } from "../history";

export default function SaveVisit({ res }: { res: AnalysisResult }) {
  const [codes, setCodes] = useState<string[]>([]);
  const [code, setCode] = useState("");
  const [date, setDate] = useState(today());
  const [note, setNote] = useState("");
  const [state, setState] = useState<"idle" | "saving" | "saved" | "error" | "off">("idle");
  const [msg, setMsg] = useState("");

  useEffect(() => {
    let alive = true;
    (async () => {
      if (!(await historyAvailable())) { if (alive) setState("off"); return; }
      const v = await listVisits().catch(() => []);
      if (alive) setCodes([...new Set(v.map((x) => x.code))]);
    })();
    return () => { alive = false; };
  }, []);
  useEffect(() => { setState((s) => (s === "off" ? s : "idle")); }, [res]);

  if (state === "off") {
    return <p className="small muted" style={{ margin: 0 }}>Trình duyệt này không cho lưu dữ liệu (chế độ ẩn danh hoặc đã chặn lưu trữ), nên không lưu được hồ sơ theo dõi.</p>;
  }
  const c = normCode(code);
  const save = async () => {
    if (!c) { setMsg("Hãy nhập mã hồ sơ."); setState("error"); return; }
    setState("saving");
    try {
      await saveVisit(await visitFrom(res, c, date || today(), note));
      setCodes((x) => (x.includes(c) ? x : [...x, c]));
      setState("saved"); setMsg("");
    } catch {
      setState("error"); setMsg("Không lưu được (trình duyệt hết dung lượng hoặc chặn lưu trữ).");
    }
  };

  return (
    <div className="save-visit">
      <p className="small muted" style={{ margin: "0 0 8px" }}>
        Lưu tóm tắt kết quả này <b>trên trình duyệt của máy bạn</b> để so sánh với lần khám khác. Máy chủ không lưu gì.
        Dùng <b>mã hồ sơ</b>, không nhập họ tên người bệnh.
      </p>
      <div className="sv-grid">
        <label><span>Mã hồ sơ</span>
          <input list="lcai-codes" value={code} maxLength={40} placeholder="VD: BN-001" onChange={(e) => { setCode(e.target.value); setState("idle"); }} />
          <datalist id="lcai-codes">{codes.map((x) => <option key={x} value={x} />)}</datalist>
        </label>
        <label><span>Ngày khám</span>
          <input type="date" value={date} max={today()} onChange={(e) => { setDate(e.target.value); setState("idle"); }} />
        </label>
      </div>
      <label className="sv-note"><span>Ghi chú (không bắt buộc)</span>
        <input value={note} maxLength={300} placeholder="VD: lần khám đầu, sau điều trị…" onChange={(e) => setNote(e.target.value)} />
      </label>
      <div className="sv-actions">
        <button className="btn btn-primary small" onClick={save} disabled={state === "saving" || state === "saved"}>
          {state === "saving" ? "Đang lưu…" : state === "saved" ? "Đã lưu" : "Lưu vào hồ sơ theo dõi"}
        </button>
        {state === "saved" && <Link className="small" to={`/theo-doi?ma=${encodeURIComponent(c)}`}>Mở hồ sơ {c} để so sánh →</Link>}
      </div>
      {state === "error" && msg && <p className="small" style={{ color: "var(--danger, #b42318)", margin: "6px 0 0" }}>{msg}</p>}
    </div>
  );
}
