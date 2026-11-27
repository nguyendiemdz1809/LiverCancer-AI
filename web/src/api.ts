// Client for the FastAPI backend (server/main.py). All texts shown to the user come in Vietnamese from the backend.

export type Modality = "ct" | "mri";
export type Dims = "2d" | "3d";
// Detection model chosen on the website: YOLO11n = locked main pipeline; Faster R-CNN = CT comparison model (detection only).
export type Detector = "yolo11n" | "frcnn";
export const DETECTOR_NAME: Record<Detector, string> = { yolo11n: "YOLO11n", frcnn: "Faster R-CNN" };

export interface Health {
  status: string;
  models: Record<Modality, "loading" | "ready" | "error">;
  device: string;
  limits: { max_file_mb: number; max_files: number; max_dicom_files?: number; analysis_timeout_s?: number };
  modality_model?: boolean;
  locked_config: string;
  queue: number;
  comparison?: { frcnn: "available" | "ready" | "missing" | "error" };
}

export interface FileCheck {
  index: number;
  kind: "nifti" | "image" | "unknown";
  ok: boolean;
  error_vi: string;
  modality: Modality | "unknown";
  modality_reason_vi: string;
  display_name: string;
  n_slices: number;
  warnings_vi: string[];
}

export interface InspectResult {
  files: FileCheck[];
  modality: Modality | "unknown" | "mixed";
  reason: string;
}

export interface Score { code: string; name: string; full_name?: string; prob: number }

export interface SliceBox { box: number[]; conf: number; region: number }
export interface SliceView { volume: string; slice: number; n_slices: number; width: number; height: number; png: string; boxes: SliceBox[] }

export interface Region {
  id: number;
  rank: number;
  volume: string;
  slice: number;
  slice_first: number;
  slice_last: number;
  n_slices: number;
  confidence: number;
  box: number[];
  local_png: string;
  context_png: string;
  local_window: number[];      // display zoom windows in slice pixels
  context_window: number[];
  local_scale: number;         // enlargement factor of the display zoom (<= 4)
  context_scale: number;
  label_short?: string;
}

export interface Card { title?: string; value?: string; sub: string; tone?: "result" | "pending" | "clear" | "info" }

export interface Chart { kind: "class_scores" | "detection" | "none"; title: string; items: { label: string; full: string; value: number }[] }

// What the chosen model shows (one branch: YOLO CT or YOLO MRI)
export interface Branch {
  key: Modality;
  title: string;
  detector?: Detector;
  status: string;
  view: SliceView;
  regions: Region[];
  n_regions: number;
  max_conf: number | null;
  first_ids: Record<string, number>;
  chart: Chart;
  volumes: { name: string; n_slices: number }[];
  all_regions: { volume: string; slices: string; confidence: number; label: string | null }[];
}

// Real counts of the case (3D: per volume / series / phase). Detection confidence only, never a probability of disease.
export interface CaseVolume {
  name: string; source: string; width: number; height: number; n_slices: number; yolo_slices: number; duplicate_slices: number;
  spacing: (number | null)[] | null; dicom_files: number | null; dicom_duplicate_files: number | null; irregular: boolean;
  slices_with_regions: number; n_regions: number; spacing_xyz?: (number | null)[] | null;
}
// One suspicious region in 3D = YOLO boxes of consecutive slices linked by the locked rule (backend regions_3d).
// boxes: [k, x0, y0, x1, y1, conf]; x / y in pixels of the displayed slice, z = slice index. size_mm only when the
// spacing is known and regular. The 3D box is the hull of 2D boxes, not a segmentation.
export interface Region3D {
  id: number; num: number | null; volume: string; k_first: number; k_last: number; best_slice: number; n_slices: number;
  gap_slices: number; max_conf: number; boxes: number[][];
  bbox_vox: { x_min: number; y_min: number; z_min: number; x_max: number; y_max: number; z_max: number };
  size_mm?: number[];
}
export interface SuspiciousSlice { volume: string; k: number; conf: number; n_boxes: number }
export interface CaseSummary {
  is_3d: boolean; volumes: CaseVolume[]; total_slices: number; yolo_slices: number; duplicate_slices: number;
  dicom_duplicate_files: number; slices_with_regions: number; suspicious_slices: SuspiciousSlice[];
  top_slice: SuspiciousSlice | null; dedup: boolean; original_size: number[] | null; screen_prepared: boolean;
  regions3d?: Region3D[];
}

export interface AnalysisResult {
  mode: "single";
  modality: Modality;
  status: "ok" | "insufficient_evidence" | "no_lesion_detected" | "detection_only";
  is_2d: boolean;
  cards: { result: Card; confidence: Card; modality: Card; regions: Card };
  label: string | null;
  label_name: string | null;
  scores: Score[];
  branches: Branch[];
  n_regions: number;
  commentary: string[];
  notes: string[];
  footnote: string;
  details: {
    limitations: string[];
    evidence: { passed: boolean; rule: string; max_conf: number; n_volumes_strong: number } | null;
    strong_conf?: number | null;     // CT: confidence of a "strong" region in the applied Evidence Gate rule
    detector?: { key: Detector; name: string; conf: number; main: boolean };
    modality: { value: string; method: string; confidence: number | null };
    volumes: { name: string; n_slices: number; n_regions: number }[];
    device: string;
    preprocessing?: { original_jpg: string; prepared_png: string; steps: string[]; warnings: string[] };
  };
  annotated_png: string;
  device: string;
  timings_s: Record<string, number>;
  case?: CaseSummary;
}

export interface JobState {
  id: string;
  state: "queued" | "running" | "done" | "error" | "needs_confirm";
  stage_key: string;
  stage: string;
  progress: number | null;
  queue_position?: number;
  result?: AnalysisResult;
  error?: string;
  need?: { detected: Modality; chosen: Modality; message: string };   // the file's information contradicts the choice
}

export class ApiError extends Error {
  status: number;
  code: string;
  constructor(message: string, status = 0, code = "error") {
    super(message);
    this.status = status;
    this.code = code;
  }
}

const API_KEY = "lcai_api_base";

function safeGet(store: Storage | undefined, key: string): string {
  try { return store?.getItem(key) ?? ""; } catch { return ""; }
}
function safeSet(store: Storage | undefined, key: string, value: string) {
  try { value ? store?.setItem(key, value) : store?.removeItem(key); } catch { /* storage unavailable */ }
}

const clean = (v: string) => v.trim().replace(/\/+$/, "");
const okUrl = (v: unknown): v is string => typeof v === "string" && /^https:\/\/[a-z0-9.-]+(:\d+)?$/i.test(v.trim().replace(/\/+$/, ""));
let runtimeApi = "";                 // from /runtime-config.json, written by MO_WEBSITE_ONLINE / MO_DEMO_ONLINE when the tunnel starts

// Called once before the first render. On Cloudflare Pages the file holds the current tunnel address; when the
// site is served by the backend itself the file does not exist and the API is on the same address.
export async function loadRuntimeConfig(timeoutMs = 3000): Promise<void> {
  try { window.localStorage.removeItem(API_KEY); } catch { /* old builds kept ?api= forever: never reuse it */ }
  try { window.sessionStorage.removeItem("lcai_access_code"); } catch { /* access codes are gone: drop old ones */ }
  try {
    const q = new URLSearchParams(window.location.search).get("api");
    if (q && okUrl(clean(q))) safeSet(window.sessionStorage, API_KEY, clean(q));
  } catch { /* ignore */ }
  const ctl = new AbortController();
  const timer = window.setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch("/runtime-config.json", { cache: "no-store", signal: ctl.signal });
    if (r.ok && (r.headers.get("content-type") || "").includes("json")) {
      const c = await r.json();
      if (okUrl(c?.api_base)) runtimeApi = clean(c.api_base);
    }
  } catch { /* no config: same address */ } finally { window.clearTimeout(timer); }
}

// Backend address: ?api=... (this tab only), then the runtime config, then the build-time VITE_API_BASE, else the same address.
export function apiBase(): string {
  return safeGet(window.sessionStorage, API_KEY) || runtimeApi || clean((import.meta.env.VITE_API_BASE as string | undefined) ?? "");
}
export function setApiBase(v: string) { safeSet(window.sessionStorage, API_KEY, clean(v)); }

export const OFFLINE =
  "Máy chủ AI hiện đang tắt hoặc chưa kết nối, nên chưa phân tích ảnh được. Bản demo online chỉ hoạt động khi máy tính chạy AI " +
  "của nhóm nghiên cứu đang bật (MO_WEBSITE_ONLINE). Các trang Giới thiệu, Thông tin mô hình và Kết quả nghiên cứu vẫn xem được bình thường.";

async function parse<T>(r: Response): Promise<T> {
  let body: any = null;
  try { body = await r.json(); } catch { /* not JSON */ }
  if (!r.ok) {
    // Cloudflare answers 502/530 (HTML, no JSON) when the AI computer or its tunnel is off.
    if (body === null && [502, 503, 504, 521, 522, 523, 524, 530].includes(r.status)) throw new ApiError(OFFLINE, 0, "offline");
    throw new ApiError(body?.detail || `Máy chủ trả lỗi (${r.status}).`, r.status, body?.code || "error");
  }
  if (body === null) throw new ApiError(OFFLINE, 0, "offline");     // e.g. a static host answering with index.html
  return body as T;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await fetch(apiBase() + path, init);
  } catch {
    throw new ApiError(OFFLINE, 0, "offline");
  }
  return parse<T>(r);
}

export const getHealth = () => call<Health>("/api/health");

export async function inspectFiles(files: File[]): Promise<InspectResult> {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f.slice(0, 256 * 1024), f.name));   // header only: nothing else leaves the browser here
  fd.append("sizes", JSON.stringify(files.map((f) => f.size)));
  return call<InspectResult>("/api/inspect", { method: "POST", body: fd });
}

// One raw-bytes request with upload progress (XMLHttpRequest reports upload progress, fetch does not).
function xhrPost(url: string, body: Blob, onProgress: (loaded: number) => void): Promise<any> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => onProgress(e.loaded);
    xhr.onerror = () => reject(new ApiError("Mất kết nối khi tải ảnh lên máy chủ.", 0, "offline"));
    xhr.onload = () => {
      let b: any = null;
      try { b = JSON.parse(xhr.responseText); } catch { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(b);
      else reject(new ApiError(b?.detail || `Máy chủ trả lỗi (${xhr.status}).`, xhr.status, b?.code || "error"));
    };
    xhr.send(body);
  });
}

// Chunked upload: Cloudflare accepts at most 100 MB per request, a single CT phase can be ~160 MB.
export async function uploadFiles(files: File[], onProgress: (pct: number) => void): Promise<string> {
  const init = await call<{ upload_id: string; chunk_mb: number }>("/api/uploads", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ files: files.map((f) => ({ name: f.name, size: f.size })) }),
  });
  const chunk = init.chunk_mb * 1024 * 1024;
  const total = files.reduce((a, f) => a + f.size, 0) || 1;
  let done = 0;
  for (let i = 0; i < files.length; i++) {
    for (let off = 0; off < files[i].size; off += chunk) {
      const part = files[i].slice(off, Math.min(off + chunk, files[i].size));
      const url = `${apiBase()}/api/uploads/${init.upload_id}/chunk?index=${i}&offset=${off}`;
      await xhrPost(url, part, (loaded) => onProgress((100 * (done + loaded)) / total));
      done += part.size;
      onProgress((100 * done) / total);
    }
  }
  return init.upload_id;
}

export function analyzeUpload(uploadId: string, modality: Modality | "auto", dims?: Dims, detector: Detector = "yolo11n") {
  const fd = new FormData();
  fd.append("modality", modality);
  if (dims) fd.append("dims", dims);
  fd.append("detector", detector);
  return call<{ job_id: string; modality: Modality }>(`/api/uploads/${uploadId}/analyze`, { method: "POST", body: fd });
}

export const deleteUpload = (id: string) => call<{ deleted: boolean }>(`/api/uploads/${id}`, { method: "DELETE" }).catch(() => undefined);

export const getJob = (id: string) => call<JobState>(`/api/jobs/${id}`);
export function confirmModality(id: string, modality: Modality) {
  const fd = new FormData();
  fd.append("modality", modality);
  return call<JobState>(`/api/jobs/${id}/confirm`, { method: "POST", body: fd });
}
export const getSlice = (id: string, branch: Modality, volume: string, k: number) =>
  call<SliceView>(`/api/jobs/${id}/slice?branch=${branch}&volume=${encodeURIComponent(volume)}&k=${k}`);
// Thumbnail strip: plain <img> URLs (lazy-loaded by the browser), small JPEG with the region boxes.
export const thumbUrl = (id: string, branch: Modality, volume: string, k: number) =>
  `${apiBase()}/api/jobs/${id}/thumb?branch=${branch}&volume=${encodeURIComponent(volume)}&k=${k}`;
// 3D scan playback: `count` consecutive slices stacked vertically in one JPEG (browser-cached).
export const framesUrl = (id: string, branch: Modality, volume: string, start: number, count: number) =>
  `${apiBase()}/api/jobs/${id}/frames?branch=${branch}&volume=${encodeURIComponent(volume)}&start=${start}&count=${count}`;

export interface VideoJob { id: string; state: "queued" | "running" | "done" | "error"; done: number; total: number; progress: number; error?: string; size_bytes?: number }
export const startVideo = (id: string, volume: string, fps: number, branch: Modality) =>
  call<VideoJob>(`/api/jobs/${id}/video`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ volume, fps, branch }) });
export const getVideo = (vid: string) => call<VideoJob>(`/api/videos/${vid}`);
export const deleteVideo = (vid: string) => call<{ deleted: boolean }>(`/api/videos/${vid}`, { method: "DELETE" }).catch(() => undefined);
export async function fetchVideo(vid: string): Promise<Blob> {
  let r: Response;
  try { r = await fetch(`${apiBase()}/api/videos/${vid}/file`); } catch { throw new ApiError(OFFLINE, 0, "offline"); }
  if (!r.ok) throw new ApiError(`Không tải được video (${r.status}).`, r.status);
  return r.blob();
}

export const deleteJob = (id: string) => call<{ deleted: boolean }>(`/api/jobs/${id}`, { method: "DELETE" }).catch(() => undefined);

// Short name of the detector that produced a result ("YOLO" for the main pipeline, as everywhere on the site).
export const detShort = (res: AnalysisResult) => (res.details.detector?.key === "frcnn" ? "Faster R-CNN" : "YOLO");

export const fmtPct = (x: number, d = 1) => `${(100 * x).toLocaleString("vi-VN", { minimumFractionDigits: d, maximumFractionDigits: d })}%`;
export const fmtNum = (x: number, d = 2) => x.toLocaleString("vi-VN", { minimumFractionDigits: d, maximumFractionDigits: d });
