// "Hồ sơ theo dõi": analysis summaries saved in THIS browser only (IndexedDB), grouped by a record code the user types
// (never a name: the page asks for a code). Nothing is sent to the AI server, which still keeps no patient data.
// A saved visit holds only numbers the backend returned plus one small JPEG of the annotated slice; it is a copy of the
// AI output at that date, not a medical record. Export / import as a JSON file to move records to another computer.
import type { AnalysisResult } from "./api";

export interface VisitRegion { num: number; volume: string; slices: string; conf: number; size_mm: number[] | null; size_vox: string | null }

export interface Visit {
  id: string;
  code: string;                 // record code chosen by the user (e.g. "BN-001"); not a name
  date: string;                 // exam date, yyyy-mm-dd (chosen by the user)
  saved_at: string;             // ISO time of saving
  note: string;
  modality: "ct" | "mri";
  is_2d: boolean;
  detector: string;             // "YOLO11n" | "Faster R-CNN"
  detector_main: boolean;
  status: AnalysisResult["status"];
  result_title: string;
  label: string | null;
  label_name: string | null;
  scores: { code: string; name: string; prob: number }[];
  n_regions: number;
  max_conf: number | null;
  n_strong: number | null;      // regions >= Evidence Gate "strong" confidence (CT, YOLO11n), else null
  total_slices: number | null;
  n_volumes: number;
  volumes: string[];
  regions: VisitRegion[];       // up to 5, highest detection confidence first
  thumb: string;                // data:image/jpeg;base64 of the annotated slice (<= 360 px)
  app: string;                  // version marker of the saved format
}

const DB = "livercancer_ai";
const STORE = "visits";
const FORMAT = "lcai-visit-1";

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    let req: IDBOpenDBRequest;
    try { req = indexedDB.open(DB, 1); } catch (e) { reject(e); return; }
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: "id" }).createIndex("code", "code");
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T> | void): Promise<T | undefined> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const r = fn(t.objectStore(STORE));
    t.oncomplete = () => { db.close(); resolve(r ? (r as IDBRequest<T>).result : undefined); };
    t.onerror = () => { db.close(); reject(t.error); };
    t.onabort = () => { db.close(); reject(t.error); };
  });
}

export async function historyAvailable(): Promise<boolean> {
  try { await (await open()).close(); return true; } catch { return false; }
}

export async function listVisits(): Promise<Visit[]> {
  const all = (await tx<Visit[]>("readonly", (s) => s.getAll())) ?? [];
  return all.filter((v) => v && v.app === FORMAT).sort((a, b) => (a.date === b.date ? a.saved_at.localeCompare(b.saved_at) : a.date.localeCompare(b.date)));
}

export const saveVisit = (v: Visit) => tx("readwrite", (s) => s.put(v));
export const deleteVisit = (id: string) => tx("readwrite", (s) => s.delete(id));
export async function deleteCode(code: string) {
  for (const v of await listVisits()) if (v.code === code) await deleteVisit(v.id);
}

export const normCode = (c: string) => c.trim().replace(/\s+/g, " ").slice(0, 40);
export const today = () => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
};
export const fmtDate = (iso: string) => { const [y, m, d] = iso.split("-"); return d && m && y ? `${d}/${m}/${y}` : iso; };

function shrink(b64png: string, side = 360): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const s = Math.min(1, side / Math.max(img.width, img.height));
      const c = document.createElement("canvas");
      c.width = Math.round(img.width * s); c.height = Math.round(img.height * s);
      c.getContext("2d")?.drawImage(img, 0, 0, c.width, c.height);
      resolve(c.toDataURL("image/jpeg", 0.8));
    };
    img.onerror = () => resolve("");
    img.src = `data:image/png;base64,${b64png}`;
  });
}

// Summary of one backend result (numbers exactly as returned; nothing recomputed except the count of strong regions).
export async function visitFrom(res: AnalysisResult, code: string, date: string, note: string): Promise<Visit> {
  const b = res.branches[0];
  const cs = res.case;
  const strong = res.details.strong_conf ?? null;
  const r3 = cs?.regions3d ?? [];
  const regions: VisitRegion[] = r3.length
    ? [...r3].sort((x, y) => y.max_conf - x.max_conf).slice(0, 5).map((r) => ({
        num: r.num ?? 0, volume: r.volume, slices: `${r.k_first + 1}–${r.k_last + 1}`, conf: r.max_conf, size_mm: r.size_mm ?? null,
        size_vox: r.size_mm ? null : `${Math.round(r.bbox_vox.x_max - r.bbox_vox.x_min)} × ${Math.round(r.bbox_vox.y_max - r.bbox_vox.y_min)} px × ${r.k_last - r.k_first + 1} lát`,
      }))
    : b.regions.slice(0, 5).map((r) => ({
        num: r.rank, volume: r.volume, slices: res.is_2d ? "" : `${r.slice_first + 1}–${r.slice_last + 1}`, conf: r.confidence, size_mm: null,
        size_vox: `${Math.round(r.box[2] - r.box[0])} × ${Math.round(r.box[3] - r.box[1])} px`,
      }));
  return {
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    code: normCode(code), date, saved_at: new Date().toISOString(), note: note.trim().slice(0, 300),
    modality: res.modality, is_2d: res.is_2d,
    detector: res.details.detector?.name ?? "YOLO11n", detector_main: res.details.detector?.main ?? true,
    status: res.status, result_title: res.cards.result.title ?? "", label: res.label, label_name: res.label_name,
    scores: res.scores.map((s) => ({ code: s.code, name: s.name, prob: s.prob })),
    n_regions: res.n_regions, max_conf: b.max_conf,
    n_strong: strong != null ? b.all_regions.filter((r) => r.confidence >= strong).length : null,
    total_slices: cs?.is_3d ? cs.total_slices : null, n_volumes: b.volumes.length, volumes: b.volumes.map((v) => v.name),
    regions, thumb: await shrink(res.annotated_png), app: FORMAT,
  };
}

export async function exportJson(visits: Visit[]): Promise<Blob> {
  return new Blob([JSON.stringify({ format: FORMAT, exported_at: new Date().toISOString(), visits }, null, 1)], { type: "application/json" });
}

// Import: only well-formed visits of this format; ids already present are skipped (no silent overwrite).
export async function importJson(file: File): Promise<{ added: number; skipped: number }> {
  const data = JSON.parse(await file.text());
  if (data?.format !== FORMAT || !Array.isArray(data.visits)) throw new Error("Tệp không phải hồ sơ theo dõi của LiverCancer AI.");
  const have = new Set((await listVisits()).map((v) => v.id));
  let added = 0, skipped = 0;
  for (const v of data.visits as Visit[]) {
    const ok = v && typeof v.id === "string" && typeof v.code === "string" && /^\d{4}-\d{2}-\d{2}$/.test(v.date) && v.app === FORMAT
      && (v.modality === "ct" || v.modality === "mri") && typeof v.n_regions === "number"
      && (typeof v.thumb !== "string" || v.thumb === "" || v.thumb.startsWith("data:image/jpeg;base64,"));
    if (!ok || have.has(v.id)) { skipped++; continue; }
    await saveVisit({ ...v, code: normCode(v.code), note: String(v.note ?? "").slice(0, 300) });
    added++;
  }
  return { added, skipped };
}
