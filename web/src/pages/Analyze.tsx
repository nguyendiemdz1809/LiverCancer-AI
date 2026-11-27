import { useCallback, useEffect, useRef, useState } from "react";
import type { AnalysisResult, Detector, Dims, Health, JobState, Modality } from "../api";
import { ApiError, DETECTOR_NAME, analyzeUpload, confirmModality, deleteJob, deleteUpload, getHealth, getJob, uploadFiles } from "../api";
import ResultPanel from "../components/ResultPanel";
import Viewer, { type ViewerState } from "../components/Viewer";

type Phase = "idle" | "busy" | "confirm" | "done" | "error";
type Run = { jobId: string; result: AnalysisResult; elapsed: number };
type Runs = Partial<Record<Detector, Run>>;
const DETS: Detector[] = ["yolo11n", "frcnn"];
const EXT_2D = /\.(png|jpe?g|dcm|dicom|ima)$/i;
const EXT_3D = /\.(nii\.gz|nii|zip|dcm|dicom|ima)$/i;
const STEPS: { key: string; label: string; d3?: boolean }[] = [
  { key: "upload", label: "Tải ảnh lên máy chủ" },
  { key: "format", label: "Đọc và kiểm tra tệp" },
  { key: "modality", label: "Kiểm tra loại ảnh CT / MRI" },
  { key: "extract", label: "Tách volume thành lát cắt 2D", d3: true },
  { key: "detect", label: "Phát hiện vùng nghi ngờ" },
  { key: "zoom", label: "Tumor Zoom vùng nghi ngờ" },
  { key: "summary", label: "Tổng hợp kết quả" },
];

function Svg({ d, size = 22 }: { d: string; size?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={d} /></svg>
  );
}
const P = {
  upload: "M12 16V4m0 0l-4 4m4-4l4 4M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2",
  ct: "M12 3a9 9 0 100 18 9 9 0 000-18zM12 8a4 4 0 100 8 4 4 0 000-8zM12 3v2M12 19v2M3 12h2M19 12h2",
  mri: "M7 5h10a6 6 0 010 12H7A6 6 0 017 5zM12 8.5a3.5 2.5 0 100 5 3.5 2.5 0 000-5z",
  info: "M12 8h.01M11 12h1v5h1M12 3a9 9 0 100 18 9 9 0 000-18z",
  image: "M4 5h16v14H4zM4 15l4-4 4 4 3-3 5 5M15 9h.01",
  result: "M9 12l2 2 4-4M12 3a9 9 0 100 18 9 9 0 000-18z",
};

function Stepper({ stage, pct, detail, dims, det }: { stage: string; pct: number | null; detail: string; dims: Dims; det: string }) {
  const steps = STEPS.filter((s) => !s.d3 || dims === "3d");
  const cur = Math.max(0, steps.findIndex((s) => s.key === stage));
  return (
    <div className="stepper v" aria-live="polite">
      <ol>
        {steps.map((s, i) => (
          <li key={s.key} className={i < cur ? "done" : i === cur ? "now" : ""}>
            <span className="dot">{i < cur ? "✓" : i + 1}</span>
            <span>{s.label}{s.key === "detect" ? ` (${det})` : ""}{i === cur && pct != null && (s.key === "upload" || s.key === "detect") ? ` · ${Math.round(pct)}%` : ""}</span>
          </li>
        ))}
      </ol>
      {(stage === "upload" || stage === "detect") && pct != null && (
        <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct)}><div style={{ width: `${Math.max(2, pct)}%` }} /></div>
      )}
      {detail && <p className="small muted" style={{ margin: "8px 0 0" }}>{detail}</p>}
    </div>
  );
}

export default function Analyze() {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [files, setFiles] = useState<File[]>([]);
  const [stage, setStage] = useState("upload");
  const [pct, setPct] = useState<number | null>(null);
  const [detail, setDetail] = useState("");
  const [error, setError] = useState("");
  const [dims, setDims] = useState<Dims>("2d");
  const [mod, setMod] = useState<Modality | null>(null);
  const [need, setNeed] = useState<JobState["need"] | null>(null);
  const modRef = useRef<Modality | null>(null);
  const [detector, setDetector] = useState<Detector>("yolo11n");
  const detRef = useRef<Detector>("yolo11n");
  const dimsRef = useRef<Dims>("2d");
  // Finished results of the SAME files, one per detector (CT: YOLO11n and Faster R-CNN), kept side by side so both can
  // be shown in turn; new files, another modality or "Phân tích ảnh khác" clear them.
  const [runs, setRuns] = useState<Runs>({});
  const runsRef = useRef<Runs>({});
  const [shown, setShown] = useState<Detector>("yolo11n");
  const [vstate, setVstate] = useState<ViewerState | null>(null);
  const keepAt = useRef<{ volume: string; k: number } | null>(null);   // slice to reopen at after switching detector
  const [drag, setDrag] = useState(false);
  const t0 = useRef(0);
  const jobRef = useRef<string | null>(null);                 // the job being analysed (not yet in `runs`)
  const uploadRef = useRef<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const dirRef = useRef<HTMLInputElement>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const h = await getHealth();
        if (!alive.current) return;
        setHealth(h); setOffline("");
        if (Object.values(h.models).some((s) => s === "loading")) timer = window.setTimeout(poll, 3000);
      } catch (e) {
        if (!alive.current) return;
        setOffline((e as Error).message);
        timer = window.setTimeout(poll, 5000);
      }
    };
    poll();
    return () => {
      alive.current = false; window.clearTimeout(timer);
      if (jobRef.current) deleteJob(jobRef.current);
      Object.values(runsRef.current).forEach((r) => r && deleteJob(r.jobId));
      if (uploadRef.current) deleteUpload(uploadRef.current);
    };
  }, []);

  const clearInputs = () => {
    if (inputRef.current) inputRef.current.value = "";
    if (dirRef.current) dirRef.current.value = "";
  };

  const putRuns = (r: Runs) => { runsRef.current = r; setRuns(r); };

  // Stops the analysis in progress (if any); finished results stay.
  const cancelActive = useCallback(() => {
    if (jobRef.current) deleteJob(jobRef.current);          // also cancels a running analysis on the server
    if (uploadRef.current) deleteUpload(uploadRef.current);
    jobRef.current = uploadRef.current = null;
  }, []);

  const reset = useCallback(() => {
    cancelActive();
    Object.values(runsRef.current).forEach((r) => r && deleteJob(r.jobId));
    putRuns({});
    setPhase("idle"); setFiles([]); setVstate(null); setError(""); setPct(null); setStage("upload"); setDetail(""); setNeed(null);
    clearInputs();
  }, [cancelActive]);

  const fail = (msg: string) => { setError(msg); setPhase("error"); };

  const follow = useCallback(async (jobId: string, det: Detector) => {
    for (;;) {
      await new Promise((r) => setTimeout(r, 600));
      if (!alive.current || jobRef.current !== jobId) return;
      const j = await getJob(jobId);
      if (jobRef.current !== jobId) return;
      setStage(j.state === "queued" ? "format" : j.stage_key || "format");
      setPct(j.progress);
      setDetail(j.queue_position ? `Đang chờ đến lượt (vị trí ${j.queue_position}).` : j.stage);
      if (j.state === "needs_confirm") { setNeed(j.need ?? null); setPhase("confirm"); return; }
      if (j.state === "done" && j.result) {
        const run: Run = { jobId, result: j.result, elapsed: Math.round((Date.now() - t0.current) / 1000) };
        jobRef.current = null;
        putRuns({ ...runsRef.current, [det]: run }); setShown(det); setVstate(null); setPhase("done"); return;
      }
      if (j.state === "error") { fail(j.error || "Không phân tích được ảnh."); return; }
    }
  }, []);

  const start = useCallback(async (fs: File[]) => {
    setPhase("busy"); setStage("upload"); setPct(0); setDetail(""); t0.current = Date.now();
    try {
      const uid = await uploadFiles(fs, (p) => setPct(p));
      uploadRef.current = uid;
      setStage("format"); setPct(null);
      const det: Detector = modRef.current === "ct" ? detRef.current : "yolo11n";
      const { job_id } = await analyzeUpload(uid, modRef.current ?? "auto", dimsRef.current, det);
      uploadRef.current = null;
      jobRef.current = job_id;
      await follow(job_id, det);
    } catch (e) {
      if (uploadRef.current) { deleteUpload(uploadRef.current); uploadRef.current = null; }
      if (alive.current) fail((e as ApiError).message);
    }
  }, [follow]);

  const confirm = useCallback(async (m: Modality) => {
    if (!jobRef.current) return;
    setPhase("busy"); setStage("modality"); setNeed(null); setMod(m); modRef.current = m;
    try { await confirmModality(jobRef.current, m); await follow(jobRef.current, m === "ct" ? detRef.current : "yolo11n"); }
    catch (e) { fail((e as ApiError).message); }
  }, [follow]);

  const busy = phase === "busy" || phase === "confirm";
  const cur = runs[shown] ?? null;
  const nRuns = DETS.filter((d) => runs[d]).length;
  const chooseMod = (m: Modality) => {
    if (busy) return; if (phase !== "idle") reset(); setMod(m); modRef.current = m;
    if (m === "mri") { setDetector("yolo11n"); detRef.current = "yolo11n"; }      // Faster R-CNN exists for CT only
  };
  // "So sánh": show the other detector's result if it exists; otherwise upload the same files (still in this page) again
  // and analyse them with the other detector. The result already shown is kept.
  const rerun = (d: Detector) => {
    if (busy) return;
    if (vstate && !runsRef.current[shown]?.result.is_2d) keepAt.current = { volume: vstate.view.volume, k: vstate.view.slice };
    setDetector(d); detRef.current = d;
    if (runsRef.current[d]) { cancelActive(); setShown(d); setVstate(null); setError(""); setPhase("done"); return; }
    if (!files.length) return;
    cancelActive(); setError(""); setVstate(null); start(files);
  };
  const chooseDet = (d: Detector) => {
    if (busy) return;
    if (files.length && Object.keys(runsRef.current).length) return rerun(d);       // same case: keep both results
    if (phase !== "idle") reset();
    setDetector(d); detRef.current = d;
  };
  const chooseDims = (d: Dims) => { if (busy) return; if (phase !== "idle") reset(); setDims(d); dimsRef.current = d; clearInputs(); };

  const onFiles = useCallback((list: FileList | File[] | null) => {
    const fs = Array.from(list || []).filter((f) => f.size > 0 && !f.name.startsWith("."));
    if (!fs.length || !modRef.current) return;
    reset();
    setFiles(fs);
    const d = dimsRef.current;
    const maxMb = health?.limits.max_file_mb ?? 300;
    const dicomLike = fs.every((f) => /\.(dcm|dicom|ima)$/i.test(f.name) || !f.name.includes("."));
    if (d === "2d") {
      if (fs.length > 1) return fail("Chế độ ảnh 2D phân tích từng ảnh một. Với chuỗi DICOM hoặc volume, hãy chọn \"Phân tích ca 3D\".");
      if (/\.(nii|nii\.gz|zip)$/i.test(fs[0].name)) return fail("Tệp NIfTI / .zip là dữ liệu 3D. Hãy chọn \"Phân tích ca 3D\" rồi tải lại.");
      if (!EXT_2D.test(fs[0].name) && fs[0].name.includes(".")) return fail("Chế độ 2D nhận ảnh PNG, JPG hoặc một tệp DICOM.");
    } else {
      if (fs.some((f) => /\.(png|jpe?g)$/i.test(f.name))) return fail("Ảnh PNG/JPG là ảnh 2D. Hãy chọn \"Phân tích ảnh 2D\", hoặc tải NIfTI / DICOM .zip cho ca 3D.");
      const max = dicomLike ? health?.limits.max_dicom_files ?? 2000 : health?.limits.max_files ?? 8;
      if (fs.length > max) return fail(`Tối đa ${max} tệp cho một lần phân tích. Với chuỗi DICOM lớn, hãy nén thành một tệp .zip.`);
      const bad = fs.find((f) => !EXT_3D.test(f.name) && f.name.includes("."));
      if (bad) return fail("Chế độ 3D nhận NIfTI (.nii, .nii.gz), DICOM (.dcm, thư mục) hoặc DICOM nén .zip.");
    }
    if (fs.some((f) => f.size > maxMb * 1024 * 1024)) return fail(`Có tệp lớn hơn giới hạn ${maxMb} MB.`);
    start(fs);
  }, [health, reset, start]);

  const locked = busy || !mod;
  const loading = health && Object.values(health.models).some((s) => s === "loading");
  const accept = dims === "2d" ? ".png,.jpg,.jpeg,.dcm,.dicom,.ima" : ".nii,.gz,.zip,.dcm,.dicom,.ima";

  return (
    <div className="ws-page">
      <div className="container wide">
        {offline && <div className="notice danger" style={{ margin: "0 0 14px" }}>{offline}</div>}
        {loading && !offline && (
          <div className="notice info" style={{ margin: "0 0 14px", display: "flex", gap: 12, alignItems: "center" }}>
            <span className="spinner" aria-hidden="true" /> Máy chủ đang khởi động mô hình AI, vui lòng chờ trong giây lát…
          </div>
        )}
        <h1 className="sr-only">Tải ảnh lên, AI tự phân tích</h1>
        <div className="ws">
          {/* ---------------- left: choices, upload, progress */}
          <aside className="ws-left">
            <section className="card step-card">
              <h2 className="sc-h"><span className="num">1</span>Chọn loại phân tích</h2>
              <div className="seg2" role="radiogroup" aria-label="Ảnh 2D hay ca 3D">
                <button role="radio" aria-checked={dims === "2d"} className={dims === "2d" ? "on" : ""} disabled={busy} onClick={() => chooseDims("2d")}>Phân tích ảnh 2D</button>
                <button role="radio" aria-checked={dims === "3d"} className={dims === "3d" ? "on" : ""} disabled={busy} onClick={() => chooseDims("3d")}>Phân tích ca 3D</button>
              </div>
              <p className="small muted sc-note">{dims === "2d" ? "Một lát cắt: PNG, JPG, ảnh chụp màn hình hoặc một tệp DICOM." : "Volume CT/MRI: NIfTI, chuỗi DICOM hoặc DICOM nén .zip. Hệ thống tự tách lát cắt."}</p>
              <div className="mod2" role="radiogroup" aria-label="Chọn loại ảnh cần phân tích">
                <button role="radio" aria-checked={mod === "ct"} aria-label="Phân tích CT" className={mod === "ct" ? "on" : ""} disabled={busy} onClick={() => chooseMod("ct")}>
                  <Svg d={P.ct} /> <span>CT Scan</span>
                </button>
                <button role="radio" aria-checked={mod === "mri"} aria-label="Phân tích MRI" className={mod === "mri" ? "on" : ""} disabled={busy} onClick={() => chooseMod("mri")}>
                  <Svg d={P.mri} /> <span>MRI</span>
                </button>
              </div>
              {mod === "ct" && (
                <div className="det2" role="radiogroup" aria-label="Chọn mô hình phát hiện">
                  <button role="radio" aria-checked={detector === "yolo11n"} className={detector === "yolo11n" ? "on" : ""} disabled={busy} onClick={() => chooseDet("yolo11n")}>
                    <b>YOLO11n{runs.yolo11n && <span className="det-ok" title="Đã có kết quả cho ca này"> ✓</span>}</b><small>mô hình chính, có phân loại</small>
                  </button>
                  <button role="radio" aria-checked={detector === "frcnn"} className={detector === "frcnn" ? "on" : ""} disabled={busy || health?.comparison?.frcnn === "missing"} onClick={() => chooseDet("frcnn")}>
                    <b>Faster R-CNN{runs.frcnn && <span className="det-ok" title="Đã có kết quả cho ca này"> ✓</span>}</b><small>đối chứng, chỉ khoanh vùng</small>
                  </button>
                </div>
              )}
              <p className="small muted sc-note">{mod === "ct" ? (detector === "frcnn" ? "CT: Faster R-CNN → Tumor Zoom (mô hình đối chứng, không phân loại; chậm hơn YOLO11n khoảng 7 lần)." : "CT: YOLO11n → Tumor Zoom → Hybrid CT (HCC / ICC / CHCC) → Evidence Gate.") : mod === "mri" ? "MRI: YOLO MRI → Tumor Zoom (không phân loại)." : "Chọn CT hoặc MRI trước khi tải ảnh."}</p>
            </section>

            <section className="card step-card">
              <h2 className="sc-h"><span className="num">2</span>Tải ảnh lên</h2>
              <label className={`dropzone${drag ? " drag" : ""}${locked ? " busy" : ""}`} tabIndex={0}
                onDragOver={(e) => { e.preventDefault(); if (!locked) setDrag(true); }}
                onDragLeave={() => setDrag(false)}
                onDrop={(e) => { e.preventDefault(); setDrag(false); if (!locked) onFiles(e.dataTransfer.files); }}
                onKeyDown={(e) => { if ((e.key === "Enter" || e.key === " ") && !locked) inputRef.current?.click(); }}
                aria-disabled={locked}>
                <span className="dz-icon"><Svg d={P.upload} size={30} /></span>
                <span className="big">{phase === "confirm" ? "Đang chờ bạn xác nhận loại ảnh" : busy ? "Đang phân tích…" : !mod ? "Chọn CT hoặc MRI ở bước 1 trước khi tải ảnh"
                  : `Kéo thả ${dims === "2d" ? "ảnh" : "tệp"} ${mod === "ct" ? "CT" : "MRI"} vào đây`}</span>
                {!busy && mod && <span className="or small muted">hoặc</span>}
                {!busy && mod && <span className="btn btn-primary dz-btn">Chọn {dims === "2d" ? "ảnh" : "tệp"}</span>}
                <input ref={inputRef} type="file" multiple={dims === "3d"} hidden disabled={locked} accept={accept} onChange={(e) => onFiles(e.target.files)} />
              </label>
              <p className="small muted sc-note">
                Định dạng: {dims === "2d" ? "PNG, JPG, DICOM (một lát)" : "NIfTI (.nii, .nii.gz), DICOM .zip, tệp .dcm"} · tối đa {health?.limits.max_file_mb ?? 300} MB/tệp.
                {dims === "3d" && mod === "ct" && " Có thể chọn cùng lúc tối đa 4 pha CT của một bệnh nhân."}
                {dims === "3d" && mod === "mri" && " Có thể chọn cùng lúc nhiều chuỗi MRI của một bệnh nhân."}
              </p>
              {dims === "3d" && mod && !busy && (
                <p className="small sc-note"><button className="linklike" onClick={() => dirRef.current?.click()}>Chọn thư mục DICOM</button>
                  <input ref={dirRef} type="file" hidden multiple onChange={(e) => onFiles(e.target.files)} {...({ webkitdirectory: "", directory: "" } as Record<string, string>)} /></p>
              )}
              {files.length > 0 && (
                <p className="small muted sc-note">{files.length === 1 ? "1 tệp" : `${files.length} tệp`} · {(files.reduce((a, f) => a + f.size, 0) / 1048576).toLocaleString("vi-VN", { maximumFractionDigits: 1 })} MB</p>
              )}
            </section>

            <section className="card step-card">
              <h2 className="sc-h"><span className="num">3</span>Trạng thái xử lý</h2>
              {phase === "idle" && <p className="small muted" style={{ margin: 0 }}>Hệ thống tự phân tích ngay sau khi tải ảnh lên, không cần bấm thêm nút.</p>}
              {phase === "busy" && (
                <>
                  <Stepper stage={stage} pct={pct} detail={detail} dims={dims} det={mod === "ct" ? DETECTOR_NAME[detector] : "YOLO11n"} />
                  <button className="btn btn-ghost small" style={{ marginTop: 12 }} onClick={reset}>Hủy phân tích</button>
                </>
              )}
              {phase === "confirm" && need && (
                <div className="confirm" role="alertdialog" aria-label="Xác nhận loại ảnh">
                  <b>Vui lòng xác nhận loại ảnh.</b> <span>{need.message}</span>
                  <p className="small muted" style={{ margin: "6px 0 0" }}>Hệ thống chưa phân tích ảnh này. Mô hình {need.chosen.toUpperCase()} chỉ được kiểm chứng trên ảnh {need.chosen.toUpperCase()}.</p>
                  <div className="confirm-btns">
                    <button className="btn btn-primary small" onClick={() => confirm(need.detected)}>Phân tích như {need.detected.toUpperCase()}</button>
                    <button className="btn btn-ghost small" onClick={() => confirm(need.chosen)}>Vẫn phân tích như {need.chosen.toUpperCase()}</button>
                    <button className="btn btn-ghost small" onClick={reset}>Hủy</button>
                  </div>
                </div>
              )}
              {phase === "error" && (
                <div className="notice danger" role="alert">
                  <b>Không thể phân tích ảnh này.</b> {error}
                  <div style={{ marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {DETS.filter((d) => runs[d]).map((d) => (
                      <button key={d} className="btn btn-ghost small" onClick={() => rerun(d)}>Xem lại kết quả {DETECTOR_NAME[d]}</button>
                    ))}
                    <button className="btn btn-ghost small" onClick={reset}>Chọn ảnh khác</button>
                  </div>
                </div>
              )}
              {phase === "done" && cur && (
                <p className="small done-line" style={{ margin: 0 }}><span className="ok-dot">✓</span> Hoàn thành{nRuns > 1 ? ` (${DETECTOR_NAME[shown]})` : ""}. AI xử lý {(cur.result.timings_s.total ?? 0).toLocaleString("vi-VN")} giây
                  {cur.elapsed ? `; tổng thời gian kể cả tải lên ${cur.elapsed} giây` : ""}.</p>
              )}
            </section>

            <section className="note-card">
              <h3><Svg d={P.info} size={18} /> Lưu ý</h3>
              <ul>
                <li>Kết quả chỉ có giá trị hỗ trợ nghiên cứu.</li>
                <li>Không thay thế chẩn đoán của bác sĩ.</li>
                <li>Ảnh tải lên được xóa ngay sau khi đọc; kết quả chỉ giữ tạm 15 phút.</li>
              </ul>
            </section>
          </aside>

          {/* ---------------- centre: viewer */}
          <section className="card ws-center" aria-label="Ảnh phân tích">
            <h2 className="ws-h"><Svg d={P.image} size={20} /> Ảnh phân tích</h2>
            {phase === "done" && cur ? (
              <>
                {nRuns > 1 && (
                  <div className="det-switch" role="tablist" aria-label="Kết quả theo mô hình phát hiện">
                    <span className="small muted">Cùng ca, xem kết quả của:</span>
                    {DETS.filter((d) => runs[d]).map((d) => (
                      <button key={d} role="tab" aria-selected={shown === d} className={shown === d ? "on" : ""} onClick={() => rerun(d)}>
                        {DETECTOR_NAME[d]}<small>{runs[d]!.result.case?.regions3d?.length ?? runs[d]!.result.branches[0]?.regions.length ?? 0} vùng</small>
                      </button>
                    ))}
                  </div>
                )}
                <Viewer key={cur.jobId} res={cur.result} jobId={cur.jobId} onState={setVstate} startAt={keepAt.current} />
              </>
            ) : (
              <>
              {dims === "3d" && (
                <div className="vtabs skeleton" aria-hidden="true"><span>Quét lát cắt</span><span>Vùng nghi ngờ 3D</span></div>
              )}
              <div className="viewer-empty">
                {busy ? (
                  <><span className="spinner light" aria-hidden="true" /><b>Đang phân tích…</b><span>Ảnh và vùng nghi ngờ sẽ hiển thị ngay khi AI chạy xong.</span></>
                ) : (
                  <><Svg d={P.image} size={44} /><b>Chưa có ảnh</b><span>Chọn loại phân tích, chọn CT hoặc MRI rồi tải ảnh lên. Trình xem sẽ hiển thị lát cắt, khung phát hiện và Tumor Zoom.</span></>
                )}
              </div>
              </>
            )}
          </section>

          {/* ---------------- right: results */}
          <aside className="ws-right">
            {phase === "done" && cur ? (
              <ResultPanel key={cur.jobId} res={cur.result} state={vstate} onReset={reset} jobId={cur.jobId}
                onRerun={files.length ? rerun : undefined} otherDone={nRuns > 1} />
            ) : (
              <section className="card rp rp-empty" aria-label="Kết quả phân tích">
                <div className="rp-head"><h2>Kết quả phân tích</h2></div>
                <div className="rp-wait">
                  <Svg d={P.result} size={40} />
                  <b>{busy ? "Đang chờ kết quả AI" : "Chưa có kết quả"}</b>
                  <span>Kết luận, xác suất phân loại (CT), thông tin ca và Evidence Gate sẽ hiển thị ở đây sau khi phân tích xong.</span>
                </div>
              </section>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}
