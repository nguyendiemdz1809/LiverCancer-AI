// Medical image viewer of ONE result (axial slices): real slices and YOLO boxes from the backend, nothing drawn that the
// model did not return.
// 3D: a scan player (play / pause / replay / speed / previous / next / boxes on-off / full screen) that shows slice 1 -> N
// from browser-cached JPEG strips (8 slices per request, LRU cache) with the boxes of the displayed slice taken from the
// backend's 3D regions; when paused the lossless PNG of the slice is loaded. Thumbnails of every slice with a box,
// region numbers shared with the 3D view, the video, the report. Tab "Vùng nghi ngờ 3D": box of every region (Three.js).
import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AnalysisResult, Branch, Region, Region3D, SliceView } from "../api";
import { detShort, fmtNum, framesUrl, getSlice, thumbUrl } from "../api";
import VideoExport from "./VideoExport";
import { CMAPS, CmapDefs, cmapFilter, cmapGradient } from "./colormaps";
import type { CmapName } from "./colormaps";
import type { CamView } from "./Lesion3D";

const Lesion3D = lazy(() => import("./Lesion3D"));
// Tab "Vùng nghi ngờ 3D" (Three.js boxes) hidden on the user's request (2026-11-08): it added clutter without new
// information. Code kept; set to true to show it again. Scan video, MP4, region numbers and the 3D region data are unchanged.
const SHOW_3D_TAB = false;
const PALETTE = ["#ef4444", "#3b82f6", "#f59e0b", "#10b981", "#a855f7", "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16"];

export interface ViewerState { branch: Branch; view: SliceView; sel: Region | null; sel3d: Region3D | null; volume: string }

type Bx = { box: number[]; conf: number; region: number };
const RED = "#ef4444";
const AMBER = "#f59e0b";
const MAX_SCALE = 6;
const LENS_MAGS = [2, 3, 4];   // magnifier ("kính lúp") levels, display only
const CHUNK = 8;                 // slices per JPEG strip
const BASE_FPS = 8;              // 1x speed
const SPEEDS = [0.5, 1, 2, 4];
const CACHE_CHUNKS = 24;         // <= 192 slices decoded in the browser at once

function Icon({ d, label }: { d: string; label?: string }) {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"
      strokeLinejoin="round" aria-hidden={label ? undefined : true} aria-label={label}><path d={d} /></svg>
  );
}
const I = {
  zin: "M11 4a7 7 0 100 14 7 7 0 000-14zM21 21l-5-5M11 8v6M8 11h6",
  zout: "M11 4a7 7 0 100 14 7 7 0 000-14zM21 21l-5-5M8 11h6",
  fit: "M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5",
  full: "M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5M9 9h6v6H9z",
  exit: "M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5",
  prev: "M15 18l-6-6 6-6",
  next: "M9 18l6-6-6-6",
  play: "M7 4l13 8-13 8z",
  pause: "M7 4h4v16H7zM14 4h4v16h-4z",
  replay: "M3 12a9 9 0 109-9 9 9 0 00-6.4 2.6L3 8M3 3v5h5",
};

function Boxes({ w, h, boxes, labels, selected, onPick }: {
  w: number; h: number; boxes: Bx[]; labels: Map<number, string>; selected: number | null; onPick: (regionId: number) => void;
}) {
  const fs = Math.max(11, w * 0.026);
  return (
    <svg className="boxes" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      {boxes.map((b, i) => {
        const main = b.region === selected;
        const [x0, y0, x1, y1] = b.box;
        const txt = `${labels.get(b.region) ?? "Vùng"} · ${fmtNum(b.conf)}`;
        const tw = txt.length * fs * 0.56 + fs * 0.6;
        const ty = y0 - fs * 1.35 < 0 ? y1 + 2 : y0 - fs * 1.35;
        return (
          <g key={i} className="pickable" onClick={() => onPick(b.region)}>
            <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill="none" stroke={main ? RED : AMBER}
              strokeWidth={main ? 2.5 : 1.8} strokeDasharray={main ? undefined : "5 4"} vectorEffect="non-scaling-stroke" />
            <rect x={x0} y={ty} width={tw} height={fs * 1.3} rx={fs * 0.2} fill={main ? RED : AMBER} />
            <text x={x0 + fs * 0.3} y={ty + fs * 0.98} fontSize={fs} fill="#fff" fontWeight={600}>{txt}</text>
          </g>
        );
      })}
    </svg>
  );
}

function ZoomPair({ sel, r3, isTwoD, onOpen, det = "YOLO" }: {
  sel: Region | null; r3: Region3D | null; isTwoD: boolean; onOpen: (src: string, title: string) => void; det?: string;
}) {
  if (!sel) {
    return (
      <div className="zoom-pair">
        <div className="zp empty"><span>{r3 ? `Vùng ${r3.num} không có ảnh Tumor Zoom` : "Không có vùng nghi ngờ cần phóng to"}</span></div>
        <div className="zp empty"><span>{r3 ? "Tumor Zoom chỉ tạo cho 8 vùng có độ tin cậy phát hiện cao nhất" : `Tumor Zoom chỉ hiển thị khi ${det} phát hiện vùng nghi ngờ`}</span></div>
      </div>
    );
  }
  const w = sel.context_window;
  const ctx = `data:image/png;base64,${sel.context_png}`;
  const loc = `data:image/png;base64,${sel.local_png}`;
  const where = isTwoD ? "" : ` · ${sel.volume} · lát ${sel.slice + 1}`;
  return (
    <div className="zoom-pair">
      <figure className="zp">
        <button className="zp-img" onClick={() => onOpen(ctx, `Ảnh gốc (vùng cắt) – vùng ${sel.rank}${where}`)} aria-label="Phóng lớn ảnh vùng cắt">
          <img src={ctx} alt={`Vùng cắt quanh vùng nghi ngờ ${sel.rank}`} />
          <svg className="boxes" viewBox={`${w[0]} ${w[1]} ${w[2] - w[0]} ${w[3] - w[1]}`} preserveAspectRatio="none" aria-hidden="true">
            <rect x={sel.box[0]} y={sel.box[1]} width={sel.box[2] - sel.box[0]} height={sel.box[3] - sel.box[1]} fill="none"
              stroke={RED} strokeWidth={2.5} vectorEffect="non-scaling-stroke" />
          </svg>
        </button>
        <figcaption><span className="tag">Ảnh gốc (vùng cắt){isTwoD ? "" : ` · lát ${sel.slice + 1}`}</span></figcaption>
      </figure>
      <figure className="zp zp-zoom">
        <button className="zp-img" onClick={() => onOpen(loc, `Tumor Zoom – vùng ${sel.rank}${where}`)} aria-label="Phóng lớn ảnh Tumor Zoom">
          <img src={loc} alt={`Tumor Zoom vùng nghi ngờ ${sel.rank}`} />
        </button>
        <figcaption><span className="tag">Tumor Zoom{sel.local_scale > 1.01 ? ` ×${fmtNum(sel.local_scale, 1)}` : ""}</span></figcaption>
      </figure>
    </div>
  );
}

export default function Viewer({ res, jobId, onState, startAt }: {
  res: AnalysisResult; jobId: string | null; onState?: (s: ViewerState) => void;
  startAt?: { volume: string; k: number } | null;   // open at this slice (switching between two detectors' results of one case)
}) {
  const b = res.branches[0];
  const det = detShort(res);
  const isTwoD = res.is_2d;
  const strong = res.details.strong_conf ?? null;
  const r3all = useMemo(() => res.case?.regions3d ?? [], [res.case]);
  const [view, setView] = useState<SliceView>(b.view);
  const [selId, setSelId] = useState<number | null>(b.regions[0]?.id ?? null);
  const [err, setErr] = useState("");
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [full, setFull] = useState(false);
  const [big, setBig] = useState<{ src: string; title: string } | null>(null);
  const [tab, setTab] = useState<"scan" | "3d">("scan");
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showBoxes, setShowBoxes] = useState(true);
  const [buffering, setBuffering] = useState(false);
  const [frame, setFrame] = useState<{ volume: string; k: number } | null>(null);
  const [sortBy, setSortBy] = useState<"k" | "conf">("k");
  const [hidden3d, setHidden3d] = useState<Set<number>>(new Set());
  const [onlySel, setOnlySel] = useState(false);
  const [cam, setCam] = useState<{ v: CamView; n: number }>({ v: "default", n: 0 });
  const [cmap, setCmap] = useState<CmapName>("gray");             // pseudo-colour of the displayed slice (display only)
  const [lens, setLens] = useState(0);                             // magnifier level, 0 = off
  const [lensAt, setLensAt] = useState<{ x: number; y: number; u: number; v: number; w: number; h: number } | null>(null);
  const timer = useRef<number | undefined>();
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const stage = useRef<HTMLDivElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const strip = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const cache = useRef(new Map<string, Promise<HTMLImageElement>>());
  const drawSeq = useRef(0);
  const multi = !isTwoD && b.volumes.some((v) => v.n_slices > 1);

  const [target, setTarget] = useState(() => {
    const v = !isTwoD && startAt ? b.volumes.find((x) => x.name === startAt.volume) : undefined;
    return v ? { volume: v.name, k: Math.max(0, Math.min(v.n_slices - 1, startAt!.k)), n: v.n_slices }
      : { volume: b.view.volume, k: b.view.slice, n: b.view.n_slices };
  });
  const targetRef = useRef(target);
  targetRef.current = target;
  const tvol = target.volume;

  const labels = useMemo(() => {
    const m = new Map<number, string>(b.regions.map((r) => [r.id, `Vùng ${r.rank}`]));
    r3all.forEach((r) => m.set(r.id, `Vùng ${r.num ?? "?"}`));
    return m;
  }, [b.regions, r3all]);
  const boxIndex = useMemo(() => {                     // `${volume}|${k}` -> boxes of that slice (all 3D regions)
    const m = new Map<string, Bx[]>();
    r3all.forEach((r) => r.boxes.forEach(([k, x0, y0, x1, y1, c]) => {
      const key = `${r.volume}|${k}`;
      if (!m.has(key)) m.set(key, []);
      m.get(key)!.push({ box: [x0, y0, x1, y1], conf: c, region: r.id });
    }));
    return m;
  }, [r3all]);
  const boxesAt = useCallback((vol: string, k: number) => boxIndex.get(`${vol}|${k}`) ?? [], [boxIndex]);
  const dims = useCallback((vol: string) => {
    const v = res.case?.volumes.find((x) => x.name === vol);
    return { w: v?.width ?? view.width, h: v?.height ?? view.height, n: v?.n_slices ?? b.volumes.find((x) => x.name === vol)?.n_slices ?? 1 };
  }, [res.case, view.width, view.height, b.volumes]);
  const sel = b.regions.find((r) => r.id === selId) ?? null;
  const sel3d = r3all.find((r) => r.id === selId) ?? null;
  const volRegions = useMemo(() => r3all.filter((r) => r.volume === tvol), [r3all, tvol]);
  const colors = useMemo(() => new Map(volRegions.map((r, i) => [r.id, PALETTE[i % PALETTE.length]])), [volRegions]);
  const suspicious = useMemo(() => {
    const l = (res.case?.suspicious_slices ?? []).filter((s) => s.volume === tvol);
    return sortBy === "k" ? l : [...l].sort((a, c) => c.conf - a.conf);
  }, [res.case, tvol, sortBy]);
  const topConf = useMemo(() => Math.max(0, ...suspicious.map((s) => s.conf)), [suspicious]);

  useEffect(() => { onState?.({ branch: b, view, sel, sel3d, volume: tvol }); }, [b, view, sel, sel3d, tvol, onState]);

  // ------------------------------------------------------------------ exact PNG (paused) and JPEG strips (playback)
  const seq = useRef(0);                       // only the latest slice request may update the view (no stale responses)
  const loadSlice = useCallback((volume: string, k: number, delay = 110) => {
    window.clearTimeout(timer.current);
    const my = ++seq.current;
    timer.current = window.setTimeout(async () => {
      if (!jobId) return;
      try {
        const v = await getSlice(jobId, b.key, volume, k);
        if (my === seq.current) { setView(v); setErr(""); }
      } catch (e) { if (my === seq.current) setErr((e as Error).message); }
    }, delay);
  }, [jobId, b.key]);

  const loadChunk = useCallback((vol: string, start: number): Promise<HTMLImageElement> => {
    const key = `${vol}|${start}`;
    const c = cache.current;
    const hit = c.get(key);
    if (hit) { c.delete(key); c.set(key, hit); return hit; }          // LRU order
    const p = new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";            // CORS image: may also be drawn on the WebGL slice plane of the 3D view
      img.onload = () => resolve(img);
      img.onerror = () => { c.delete(key); reject(new Error("Không tải được khung hình (máy chủ AI có thể đang tắt)." )); };
      img.src = framesUrl(jobId ?? "", b.key, vol, start, CHUNK);
    });
    c.set(key, p);
    while (c.size > CACHE_CHUNKS) c.delete(c.keys().next().value as string);
    return p;
  }, [jobId, b.key]);

  const drawFrame = useCallback(async (vol: string, k: number) => {
    const my = ++drawSeq.current;
    const start = Math.floor(k / CHUNK) * CHUNK;
    const known = cache.current.has(`${vol}|${start}`);
    if (!known) setBuffering(true);
    try {
      const img = await loadChunk(vol, start);
      if (my !== drawSeq.current || !canvas.current) return false;
      const d = dims(vol);
      const cv = canvas.current;
      if (cv.width !== d.w || cv.height !== d.h) { cv.width = d.w; cv.height = d.h; }
      cv.getContext("2d")!.drawImage(img, 0, (k - start) * d.h, d.w, d.h, 0, 0, d.w, d.h);
      setFrame({ volume: vol, k });
      return true;
    } catch (e) {
      setErr((e as Error).message);
      return false;
    } finally {
      if (my === drawSeq.current) setBuffering(false);
    }
  }, [loadChunk, dims]);

  // every target change: draw the frame (fast, cached); when paused also load the exact PNG
  useEffect(() => {
    if (isTwoD) return;
    drawFrame(target.volume, target.k);
    if (!playing && (view.volume !== target.volume || view.slice !== target.k)) loadSlice(target.volume, target.k, 180);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target.volume, target.k, playing, isTwoD]);

  // playback: one slice per tick, never past the last slice; strips prefetched ahead
  useEffect(() => {
    if (!playing) return;
    let stop = false;
    let t: number | undefined;
    const tick = async () => {
      if (stop) return;
      const cur = targetRef.current;
      if (cur.k >= cur.n - 1) { setPlaying(false); return; }
      const k = cur.k + 1;
      const ok = await drawFrame(cur.volume, k);
      if (stop) return;
      if (!ok) { setPlaying(false); return; }
      setTarget({ ...cur, k });
      for (let a = 1; a <= 3; a++) {
        const s = (Math.floor(k / CHUNK) + a) * CHUNK;
        if (s < cur.n) loadChunk(cur.volume, s).catch(() => undefined);
      }
      t = window.setTimeout(tick, 1000 / (BASE_FPS * speed));
    };
    t = window.setTimeout(tick, 1000 / (BASE_FPS * speed));
    return () => { stop = true; window.clearTimeout(t); };
  }, [playing, speed, drawFrame, loadChunk]);

  const go = useCallback((k: number, volume = targetRef.current.volume) => {
    const n = dims(volume).n;
    setTarget({ volume, k: Math.max(0, Math.min(n - 1, k)), n });
  }, [dims]);

  const play = () => {
    if (target.k >= target.n - 1) setTarget({ ...target, k: 0 });
    setPlaying(true);
  };
  const replay = () => { setTarget({ ...target, k: 0 }); setPlaying(true); };

  const selectRegion = useCallback((id: number, jump = true) => {
    setSelId(id);
    const r = r3all.find((x) => x.id === id);
    const z = b.regions.find((x) => x.id === id);
    if (jump && !isTwoD && (r || z)) { setPlaying(false); go(z ? z.slice : r!.best_slice, (r ?? z)!.volume); }
  }, [r3all, b.regions, isTwoD, go]);

  const openThumb = (vol: string, k: number) => {
    setPlaying(false);
    const here = boxesAt(vol, k);
    if (here.length && !here.some((x) => x.region === selId)) setSelId(here.reduce((a, c) => (c.conf > a.conf ? c : a)).region);
    go(k, vol);
  };

  // keep the active thumbnail visible
  // horizontal scroll of the strip ONLY: element.scrollIntoView would also scroll the page and move the viewer
  const stripReady = useRef(false);
  useEffect(() => {
    const st = strip.current;
    const el = st?.querySelector<HTMLElement>(".thumb.on");
    if (!st || !el) return;
    const left = el.offsetLeft - st.clientWidth / 2 + el.clientWidth / 2;
    st.scrollTo({ left: Math.max(0, left), behavior: playing || !stripReady.current ? "auto" : "smooth" });
    stripReady.current = true;
  }, [target.k, target.volume, playing]);
  // the 3D view is a separate chunk: fetch it now so the tab opens at once
  useEffect(() => { if (multi && SHOW_3D_TAB) import("./Lesion3D").catch(() => undefined); }, [multi]);

  const zoomBy = (f: number) => setScale((s) => {
    const n = Math.max(1, Math.min(MAX_SCALE, s * f));
    if (n === 1) setPan({ x: 0, y: 0 });
    return n;
  });
  const reset = () => { setScale(1); setPan({ x: 0, y: 0 }); };
  const toggleFull = () => {
    const next = !full;
    setFull(next);
    try {
      if (next && wrap.current?.requestFullscreen) wrap.current.requestFullscreen().catch(() => undefined);
      else if (!next && document.fullscreenElement) document.exitFullscreen().catch(() => undefined);
    } catch { /* CSS full screen still applies */ }
  };
  useEffect(() => {
    const onFs = () => { if (!document.fullscreenElement) setFull(false); };
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);
  useEffect(() => {
    const el = stage.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (!full && !e.ctrlKey) return;
      e.preventDefault();
      zoomBy(e.deltaY < 0 ? 1.2 : 1 / 1.2);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [full]);

  const onKey = (e: React.KeyboardEvent) => {
    if (multi && (e.key === "ArrowLeft" || e.key === "ArrowDown")) { e.preventDefault(); setPlaying(false); go(target.k - 1); }
    else if (multi && (e.key === "ArrowRight" || e.key === "ArrowUp")) { e.preventDefault(); setPlaying(false); go(target.k + 1); }
    else if (multi && e.key === " ") { e.preventDefault(); if (playing) setPlaying(false); else play(); }
    else if (e.key === "+" || e.key === "=") zoomBy(1.5);
    else if (e.key === "-") zoomBy(1 / 1.5);
    else if (e.key === "0") reset();
    else if (e.key.toLowerCase() === "f") toggleFull();
    else if (e.key === "Escape" && full) toggleFull();
  };
  // magnifier: (x, y) pointer in the stage; (u, v) the same point in the unzoomed image (0..1), undoing pan + zoom
  const lensMove = (e: React.PointerEvent) => {
    const r = stage.current?.getBoundingClientRect();
    if (!r) return;
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const u = ((x - r.width / 2 - pan.x) / scale + r.width / 2) / r.width;
    const v = ((y - r.height / 2 - pan.y) / scale + r.height / 2) / r.height;
    setLensAt(u < 0 || u > 1 || v < 0 || v > 1 ? null : { x, y, u, v, w: r.width, h: r.height });
  };
  const down = (e: React.PointerEvent) => {
    if (lens) { lensMove(e); return; }
    if (scale <= 1) return;
    drag.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
    (e.target as Element).setPointerCapture?.(e.pointerId);
  };
  const move = (e: React.PointerEvent) => {
    if (lens) { lensMove(e); return; }
    const d = drag.current;
    if (!d || !stage.current) return;
    const r = stage.current.getBoundingClientRect();
    const lim = (v: number, size: number) => Math.max(-size * (scale - 1) / 2, Math.min(size * (scale - 1) / 2, v));
    setPan({ x: lim(d.px + e.clientX - d.x, r.width), y: lim(d.py + e.clientY - d.y, r.height) });
  };
  const up = () => { drag.current = null; };
  const scrollStrip = (dir: number) => strip.current?.scrollBy({ left: dir * strip.current.clientWidth * 0.8, behavior: "smooth" });

  // what is on screen: the exact PNG when paused on the asked slice, else the canvas frame (its own boxes)
  const showImg = isTwoD || (!playing && view.volume === target.volume && view.slice === target.k);
  const shown = showImg ? { volume: view.volume, k: view.slice } : (frame ?? { volume: view.volume, k: view.slice });
  const sd = isTwoD ? { w: view.width, h: view.height, n: 1 } : dims(shown.volume);
  const shownBoxes: Bx[] = showImg ? view.boxes.map((x) => ({ box: x.box, conf: x.conf, region: x.region })) : boxesAt(shown.volume, shown.k);
  const regionNum = (id: number) => labels.get(id) ?? "Vùng";
  const getFrame3d = useCallback(async (k: number) => {
    const start = Math.floor(k / CHUNK) * CHUNK;
    const d = dims(tvol);
    return { img: await loadChunk(tvol, start), sy: (k - start) * d.h, w: d.w, h: d.h };
  }, [loadChunk, dims, tvol]);
  const boxesAt3d = useCallback((k: number) => boxesAt(tvol, k), [boxesAt, tvol]);
  const spacing3d = (() => { const v = res.case?.volumes.find((x) => x.name === tvol); const sp = v?.spacing_xyz; return sp && sp.every((x) => x) && !v?.irregular ? (sp as number[]) : null; })();
  const toggleRegion = (id: number) => setHidden3d((h) => { const n = new Set(h); if (n.has(id)) n.delete(id); else n.add(id); return n; });

  const player = (
    <div ref={wrap} className={`viewer-wrap${full ? " full" : ""}`}>
      <div ref={stage} className={`viewer-stage${scale > 1 ? " zoomed" : ""}${lens ? " lens-on" : ""}`} tabIndex={0} onKeyDown={onKey}
        onPointerLeave={() => setLensAt(null)}
        aria-label={`Trình xem ảnh${multi ? `, lát ${shown.k + 1} trên ${sd.n}` : ""}. Phím cách: phát / dừng; mũi tên: đổi lát; + / −: phóng to; F: toàn màn hình.`}
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onDoubleClick={reset}
        style={{ ["--ar" as string]: String(sd.w / sd.h) } as React.CSSProperties}>
        <div className="viewer-img" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})` }}>
          {!isTwoD && <canvas ref={canvas} className="slice" style={{ display: showImg ? "none" : "block", filter: cmapFilter(cmap) }} aria-hidden="true" />}
          {showImg && <img className="slice" style={{ filter: cmapFilter(cmap) }} src={`data:image/png;base64,${view.png}`} alt={`${b.title}: ${isTwoD ? "ảnh được phân tích" : `${view.volume}, lát ${view.slice + 1}`} với khung vùng nghi ngờ`} draggable={false} />}
          {showBoxes && <Boxes w={sd.w} h={sd.h} boxes={shownBoxes} labels={labels} selected={selId} onPick={(id) => selectRegion(id, false)} />}
        </div>
        {lens > 0 && lensAt && showImg && (() => {
          const m = scale * lens, cw = lensAt.w * m, ch = lensAt.h * m;
          return (
            <div className="lens" style={{ left: lensAt.x, top: lensAt.y }} aria-hidden="true">
              <div className="lens-in" style={{ width: cw, height: ch, transform: `translate(calc(var(--ls) / 2 - ${lensAt.u * cw}px), calc(var(--ls) / 2 - ${lensAt.v * ch}px))` }}>
                <img className="slice" style={{ filter: cmapFilter(cmap) }} src={`data:image/png;base64,${view.png}`} alt="" draggable={false} />
                {showBoxes && <Boxes w={sd.w} h={sd.h} boxes={shownBoxes} labels={labels} selected={selId} onPick={() => undefined} />}
              </div>
              <span className="lens-tag tnum">×{fmtNum(m, 1)}</span>
            </div>
          );
        })()}
        <div className="vt vt-tl tnum">
          {isTwoD ? "Ảnh 2D" : `Lát ${shown.k + 1} / ${sd.n}`}{!isTwoD && b.volumes.length > 1 ? ` · ${shown.volume}` : ""}
          {shownBoxes.length > 0 ? <span className="vt-n"> · {shownBoxes.length} vùng trên lát này</span>
            : !isTwoD && <span className="vt-0"> · không có vùng</span>}
        </div>
        <div className="vt vt-tr" onPointerDown={(e) => e.stopPropagation()}>
          <button onClick={() => zoomBy(1.5)} aria-label="Phóng to" title="Phóng to (+)"><Icon d={I.zin} /></button>
          <button onClick={() => zoomBy(1 / 1.5)} aria-label="Thu nhỏ" title="Thu nhỏ (−)" disabled={scale <= 1}><Icon d={I.zout} /></button>
          <button onClick={reset} aria-label="Vừa khung" title="Vừa khung (0)" disabled={scale <= 1}><Icon d={I.fit} /></button>
          <button onClick={toggleFull} aria-label={full ? "Thoát toàn màn hình" : "Toàn màn hình"} title="Toàn màn hình (F)"><Icon d={full ? I.exit : I.full} /></button>
        </div>
        {scale > 1 && <div className="vt vt-bl tnum">×{fmtNum(scale, 1)} · kéo để di chuyển</div>}
        {buffering && playing && <div className="vt vt-br">Đang tải khung hình…</div>}
        {multi && (
          <>
            <button className="vnav prev" onPointerDown={(e) => e.stopPropagation()} onClick={() => { setPlaying(false); go(target.k - 1); }} disabled={target.k <= 0} aria-label="Lát trước"><Icon d={I.prev} /></button>
            <button className="vnav next" onPointerDown={(e) => e.stopPropagation()} onClick={() => { setPlaying(false); go(target.k + 1); }} disabled={target.k >= target.n - 1} aria-label="Lát sau"><Icon d={I.next} /></button>
          </>
        )}
      </div>
      <div className="view-tools" onPointerDown={(e) => e.stopPropagation()}>
        <label className="vtool">Rải màu
          <select value={cmap} onChange={(e) => setCmap(e.target.value as CmapName)} aria-label="Bảng màu hiển thị">
            {CMAPS.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
          </select>
        </label>
        <span className="cbar" style={{ background: cmapGradient(cmap) }} title="Thang màu: trái = tối (0), phải = sáng (255)" aria-hidden="true" />
        <span className="vtool" role="radiogroup" aria-label="Kính lúp">Kính lúp
          {[0, ...LENS_MAGS].map((m) => (
            <button key={m} role="radio" aria-checked={lens === m} className={lens === m ? "on" : ""}
              onClick={() => { setLens(m); setLensAt(null); }}>{m ? `${m}×` : "Tắt"}</button>
          ))}
        </span>
        <span className="vtool-note small">
          {lens > 0 && !showImg ? "Kính lúp hoạt động khi dừng phát. " : ""}Chỉ đổi cách hiển thị; AI phân tích trên ảnh xám gốc.
        </span>
      </div>
      {multi && (
        <div className="player-bar">
          <button className="pb-main" onClick={() => (playing ? setPlaying(false) : play())} aria-label={playing ? "Tạm dừng" : "Phát"} title="Phát / dừng (phím cách)">
            <Icon d={playing ? I.pause : I.play} />
          </button>
          <button onClick={replay} aria-label="Phát lại từ lát 1" title="Phát lại từ lát 1"><Icon d={I.replay} /></button>
          <div className="range-wrap" title="Vạch đỏ: lát có vùng nghi ngờ. Phím ← →: đổi lát.">
            <input type="range" min={0} max={target.n - 1} value={target.k} aria-label="Chọn lát cắt" onChange={(e) => go(+e.target.value)} />
            <div className="marks" aria-hidden="true">
              {(res.case?.suspicious_slices ?? []).filter((s) => s.volume === tvol).map((s) => (
                <i key={s.k} style={{ left: `${target.n > 1 ? (100 * s.k) / (target.n - 1) : 0}%` }} className={s.conf === topConf ? "top" : ""} />
              ))}
            </div>
          </div>
          <span className="slice-label tnum">Lát {target.k + 1}/{target.n}</span>
          <div className="speeds" role="radiogroup" aria-label="Tốc độ phát">
            {SPEEDS.map((s) => (
              <button key={s} role="radio" aria-checked={speed === s} className={speed === s ? "on" : ""} onClick={() => setSpeed(s)}>{s.toLocaleString("vi-VN")}x</button>
            ))}
          </div>
          <label className="pb-toggle"><input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} /> Khung</label>
        </div>
      )}
    </div>
  );

  return (
    <div className="viewer-block">
      <CmapDefs />
      {multi && (SHOW_3D_TAB || b.volumes.length > 1) && (
        <div className="vtabs" role={SHOW_3D_TAB ? "tablist" : undefined} aria-label="Chế độ xem">
          {SHOW_3D_TAB && (
            <>
              <button role="tab" aria-selected={tab === "scan"} className={tab === "scan" ? "on" : ""} onClick={() => setTab("scan")}>Quét lát cắt</button>
              <button role="tab" aria-selected={tab === "3d"} className={tab === "3d" ? "on" : ""} onClick={() => { setPlaying(false); setTab("3d"); }}>
                Vùng nghi ngờ 3D <span className="tab-n">{r3all.length}</span>
              </button>
            </>
          )}
          {b.volumes.length > 1 && (
            <select aria-label="Chọn pha / chuỗi" value={tvol} onChange={(e) => {
              const v = e.target.value;
              setPlaying(false);
              const first = (res.case?.suspicious_slices ?? []).find((s) => s.volume === v);
              go(first ? first.k : Math.floor(dims(v).n / 2), v);
            }}>
              {b.volumes.map((v) => <option key={v.name} value={v.name}>{v.name}</option>)}
            </select>
          )}
        </div>
      )}

      <div className="vpane" hidden={tab !== "scan"}>
        {player}
        {multi && (
          <div className="frame-info small" aria-live="off">
            <b>Lát {shown.k + 1}/{sd.n}</b>
            {shownBoxes.length === 0 ? <span className="muted"> · không có phát hiện {det} trên lát này</span> : (
              <> · {shownBoxes.length} phát hiện:{" "}
                {[...shownBoxes].sort((a, c) => c.conf - a.conf).map((x, i) => (
                  <button key={i} className={`fi-chip${x.region === selId ? " on" : ""}`} onClick={() => selectRegion(x.region, false)}>
                    {regionNum(x.region)} · {fmtNum(x.conf)}
                  </button>
                ))}
              </>
            )}
            <VideoExport jobId={jobId} branch={b.key} volume={tvol} compact />
          </div>
        )}
        {err && <p className="small muted" role="alert" style={{ margin: "6px 0 0" }}>{err}</p>}

        {multi && (
          <div className="thumbs-block">
            <div className="thumbs-head">
              <b>Lát có vùng nghi ngờ{b.volumes.length > 1 ? ` · ${tvol}` : ""}</b>
              <span className="muted small">{suspicious.length ? `${suspicious.length} lát · số = độ tin cậy phát hiện (${det})` : "không có"}</span>
              {suspicious.length > 1 && (
                <span className="sortby small">Xếp theo:
                  <button className={sortBy === "k" ? "on" : ""} onClick={() => setSortBy("k")}>số lát</button>
                  <button className={sortBy === "conf" ? "on" : ""} onClick={() => setSortBy("conf")}>độ tin cậy</button>
                </span>
              )}
            </div>
            {suspicious.length > 0 ? (
              <div className="thumbs-row">
                <button className="strip-nav" onClick={() => scrollStrip(-1)} aria-label="Cuộn trái"><Icon d={I.prev} /></button>
                <div className="strip" ref={strip} role="list">
                  {suspicious.map((s) => {
                    const regs = [...new Set(boxesAt(s.volume, s.k).map((x) => x.region))];
                    return (
                      <button key={`${s.volume}|${s.k}`} role="listitem" className={`thumb${s.k === target.k ? " on" : ""}`} onClick={() => openThumb(s.volume, s.k)}
                        aria-label={`Lát ${s.k + 1}, độ tin cậy phát hiện ${fmtNum(s.conf)}`}>
                        {jobId && <img loading="lazy" src={thumbUrl(jobId, b.key, s.volume, s.k)} alt="" />}
                        <span className="t-k tnum">Lát {s.k + 1}</span>
                        <span className={`t-c tnum${s.conf === topConf ? " top" : ""}`}>{fmtNum(s.conf)}</span>
                        <span className="t-r">{regs.map((id) => regionNum(id).replace("Vùng ", "V")).join(" ")}</span>
                      </button>
                    );
                  })}
                </div>
                <button className="strip-nav" onClick={() => scrollStrip(1)} aria-label="Cuộn phải"><Icon d={I.next} /></button>
              </div>
            ) : <p className="small muted" style={{ margin: 0 }}>{det} không phát hiện vùng nghi ngờ nào trong {tvol}.</p>}
          </div>
        )}

        <div className="zoom-block">
          <div className="zoom-head">
            <b>Zoom vùng nghi ngờ</b>
            {b.regions.length > 1 && (
              <div className="chips" role="tablist" aria-label="Chọn vùng nghi ngờ">
                {b.regions.map((r) => (
                  <button key={r.id} role="tab" aria-selected={selId === r.id} onClick={() => selectRegion(r.id)}
                    className={`${selId === r.id ? "active" : ""}${strong != null && r.confidence < strong ? " weak" : ""}`}
                    title={strong != null && r.confidence < strong ? `Độ tin cậy thấp (< ${fmtNum(strong)}, mức vùng mạnh của Evidence Gate)` : undefined}>
                    Vùng {r.rank} · {fmtNum(r.confidence)}{!isTwoD && b.volumes.length > 1 ? ` · ${r.volume}` : ""}{strong != null && r.confidence < strong ? " · thấp" : ""}
                  </button>
                ))}
              </div>
            )}
          </div>
          <ZoomPair sel={sel} r3={sel3d} isTwoD={isTwoD} det={det} onOpen={(src, title) => setBig({ src, title })} />
          {(sel || sel3d) && (
            <p className="small muted" style={{ margin: "8px 0 0" }}>
              {sel ? `Vùng ${sel.rank}` : `Vùng ${sel3d!.num}`}: độ tin cậy phát hiện cao nhất {fmtNum(sel ? sel.confidence : sel3d!.max_conf)}
              {!isTwoD && sel3d ? `, lát ${sel3d.k_first + 1}–${sel3d.k_last + 1}` : ""}
              {!isTwoD && sel ? `; ảnh Zoom lấy trên lát đại diện ${sel.slice + 1}` : ""}
              {!isTwoD && sel && (sel.volume !== target.volume || sel.slice !== target.k) && (
                <> · <button className="linklike" onClick={() => { setPlaying(false); go(sel.slice, sel.volume); }}>Xem lát {sel.slice + 1}</button></>
              )}
            </p>
          )}
        </div>
      </div>

      {multi && SHOW_3D_TAB && (
        <div className="vpane" hidden={tab !== "3d"}>
          <div className="l3d-wrap">
            {tab === "3d" && (
              <Suspense fallback={<div className="l3d-host loading">Đang tải trình xem 3D…</div>}>
                <Lesion3D key={tvol} regions={volRegions} width={dims(tvol).w} height={dims(tvol).h} nSlices={dims(tvol).n} spacing={spacing3d}
                  selectedId={selId} currentK={target.k} hidden={hidden3d} onlySelected={onlySel} strong={strong} colors={colors}
                  getFrame={getFrame3d} boxesAt={boxesAt3d} onSelect={(id) => selectRegion(id)} view={cam.v} viewSeq={cam.n} />
              </Suspense>
            )}
            <div className="l3d-cams" role="group" aria-label="Góc nhìn">
              <button onClick={() => setCam((c) => ({ v: "default", n: c.n + 1 }))} title="Góc nhìn mặc định">Mặc định</button>
              <button onClick={() => setCam((c) => ({ v: "x", n: c.n + 1 }))} title="Nhìn dọc trục X">Trục X</button>
              <button onClick={() => setCam((c) => ({ v: "y", n: c.n + 1 }))} title="Nhìn dọc trục Y">Trục Y</button>
              <button onClick={() => setCam((c) => ({ v: "z", n: c.n + 1 }))} title="Nhìn dọc trục Z (mặt cắt ngang)">Trục Z</button>
            </div>
          </div>
          <div className="l3d-bar">
            <span className="small"><b>Mặt phẳng ảnh:</b></span>
            <div className="range-wrap">
              <input type="range" min={0} max={target.n - 1} value={target.k} aria-label="Lát cắt trong không gian 3D" onChange={(e) => go(+e.target.value)} />
              <div className="marks" aria-hidden="true">
                {(res.case?.suspicious_slices ?? []).filter((x) => x.volume === tvol).map((x) => (
                  <i key={x.k} style={{ left: `${target.n > 1 ? (100 * x.k) / (target.n - 1) : 0}%` }} />
                ))}
              </div>
            </div>
            <span className="slice-label tnum">Lát {target.k + 1}/{target.n}</span>
            <label className="pb-toggle"><input type="checkbox" checked={onlySel} onChange={(e) => setOnlySel(e.target.checked)} /> Chỉ vùng đang chọn</label>
            <button className="linklike small" onClick={() => setHidden3d(new Set())}>Hiện tất cả</button>
            <button className="linklike small" onClick={() => setHidden3d(new Set(volRegions.map((r) => r.id)))}>Ẩn tất cả</button>
          </div>
          <p className="small muted l3d-note">
            Mặt phẳng = ảnh lát cắt thật kèm khung YOLO của lát đó; kéo thanh trên để di chuyển. Mỗi khối = bao các khung YOLO 2D của một vùng qua các lát
            (không phải hình dạng hay thể tích khối u; ở một số ca khung YOLO rộng hơn tổn thương nhiều lần; không dựng gan vì chưa có mô hình phân đoạn gan).
            {" "}Viền đứt / "(thấp)": độ tin cậy dưới {strong != null ? fmtNum(strong) : "mức"} của Evidence Gate.
            {" "}Đơn vị: {spacing3d ? "mm" : "điểm ảnh / lát (không có khoảng cách điểm ảnh đáng tin)"}. Kéo để xoay, cuộn để phóng to, bấm vào khối để chọn vùng.
          </p>
          <div className="table-wrap r3d-table">
            <table>
              <thead><tr><th aria-label="Hiện"></th><th>Vùng</th><th>Lát</th><th className="num">Đại diện</th><th className="num">Số lát có khung</th><th className="num">Độ tin cậy cao nhất</th><th>Khối bao</th></tr></thead>
              <tbody>
                {volRegions.length === 0 && <tr><td colSpan={7} className="muted">Không có vùng nghi ngờ trong {tvol}.</td></tr>}
                {volRegions.map((r) => (
                  <tr key={r.id} className={`${r.id === selId ? "on" : ""}${strong != null && r.max_conf < strong ? " weak" : ""}`} onClick={() => selectRegion(r.id)} tabIndex={0}
                    onKeyDown={(e) => { if (e.key === "Enter") selectRegion(r.id); }}>
                    <td onClick={(e) => e.stopPropagation()}>
                      <input type="checkbox" checked={!hidden3d.has(r.id)} onChange={() => toggleRegion(r.id)} aria-label={`Hiện vùng ${r.num}`} />
                    </td>
                    <td><span className="swatch" style={{ background: colors.get(r.id) }} /><b>Vùng {r.num}</b>{strong != null && r.max_conf < strong && <span className="weak-tag"> thấp</span>}</td>
                    <td className="tnum">{r.k_first + 1}–{r.k_last + 1}{r.gap_slices ? ` (${r.gap_slices} trống)` : ""}</td>
                    <td className="num">{(b.regions.find((x) => x.id === r.id)?.slice ?? r.best_slice) + 1}</td>
                    <td className="num">{r.n_slices}</td>
                    <td className="num">{fmtNum(r.max_conf)}</td>
                    <td className="tnum">{r.size_mm ? `${r.size_mm.map((x) => fmtNum(x, 0)).join(" × ")} mm` : `${fmtNum(r.bbox_vox.x_max - r.bbox_vox.x_min, 0)} × ${fmtNum(r.bbox_vox.y_max - r.bbox_vox.y_min, 0)} px × ${r.k_last - r.k_first + 1} lát`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {sel3d && sel3d.volume === tvol && (
            <div className="l3d-sel">
              <p className="small" style={{ margin: "12px 0 8px" }}>
                Đang chọn <b>Vùng {sel3d.num}</b> · lát {sel3d.k_first + 1}–{sel3d.k_last + 1} · mặt phẳng đang ở lát {target.k + 1}.{" "}
                <button className="btn btn-ghost small" onClick={() => setTab("scan")}>Xem trên video quét lát</button>
              </p>
              <ZoomPair sel={sel} r3={sel3d} isTwoD={false} det={det} onOpen={(src, title) => setBig({ src, title })} />
            </div>
          )}
        </div>
      )}

      {big && (
        <div className="lightbox" role="dialog" aria-modal="true" aria-label={big.title} onClick={() => setBig(null)}
          onKeyDown={(e) => e.key === "Escape" && setBig(null)}>
          <div className="lightbox-inner" onClick={(e) => e.stopPropagation()}>
            <div className="lightbox-head"><b>{big.title}</b><button className="btn btn-ghost small" autoFocus onClick={() => setBig(null)}>Đóng ✕</button></div>
            <img className="lightbox-img" src={big.src} alt={big.title} />
          </div>
        </div>
      )}
    </div>
  );
}
