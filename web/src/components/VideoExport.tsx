// "Tải video phân tích 3D": MP4 (H.264) rendered on the server from the analysed volume and the real YOLO boxes of every
// slice (no new detection). Progress per frame, cancel, then the file is saved by the browser.
import { useEffect, useRef, useState } from "react";
import type { Modality, VideoJob } from "../api";
import { ApiError, deleteVideo, fetchVideo, getVideo, startVideo } from "../api";

export default function VideoExport({ jobId, branch, volume, compact = false }: {
  jobId: string | null; branch: Modality; volume: string; compact?: boolean;
}) {
  const [job, setJob] = useState<VideoJob | null>(null);
  const [msg, setMsg] = useState("");
  const vidRef = useRef<string | null>(null);
  const alive = useRef(true);

  useEffect(() => () => { alive.current = false; if (vidRef.current) deleteVideo(vidRef.current); }, []);

  const save = (blob: Blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `livercancer-ai-${branch}-3d-${volume.replace(/\s+/g, "-").toLowerCase()}.mp4`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
  };

  const start = async () => {
    if (!jobId) return;
    setMsg("");
    try {
      let j = await startVideo(jobId, volume, 8, branch);
      vidRef.current = j.id;
      setJob(j);
      while (alive.current && vidRef.current === j.id && (j.state === "queued" || j.state === "running")) {
        await new Promise((r) => setTimeout(r, 700));
        j = await getVideo(j.id);
        if (vidRef.current === j.id) setJob(j);
      }
      if (!alive.current || vidRef.current !== j.id) return;
      if (j.state === "done") {
        save(await fetchVideo(j.id));
        setMsg(`Đã tải video: ${j.total} lát, ${((j.size_bytes ?? 0) / 1048576).toLocaleString("vi-VN", { maximumFractionDigits: 1 })} MB (MP4, H.264).`);
      } else {
        setMsg(j.error || "Không tạo được video.");
      }
      deleteVideo(j.id);
      vidRef.current = null;
      setJob(null);
    } catch (e) {
      setMsg((e as ApiError).message);
      setJob(null);
      vidRef.current = null;
    }
  };

  const cancel = () => { if (vidRef.current) deleteVideo(vidRef.current); vidRef.current = null; setJob(null); setMsg("Đã hủy xuất video."); };

  const busy = !!job;
  return (
    <div className={`video-export${compact ? " compact" : ""}`}>
      {!busy ? (
        <button className="btn btn-ghost wide" onClick={start} disabled={!jobId} title="Video MP4 quét toàn bộ lát của pha / chuỗi đang xem, kèm khung YOLO thật của từng lát">
          Tải video phân tích 3D (MP4)
        </button>
      ) : (
        <div className="ve-busy" role="status">
          <div className="ve-line">
            <span>{job.state === "queued" ? "Đang chờ tạo video…" : `Đang tạo video: lát ${job.done}/${job.total}`}</span>
            <button className="linklike" onClick={cancel}>Hủy</button>
          </div>
          <div className="progress"><div style={{ width: `${Math.max(2, job.progress)}%` }} /></div>
        </div>
      )}
      {msg && <p className="small muted ve-msg">{msg}</p>}
    </div>
  );
}
