// 3D view of ONE volume: the volume frame, X / Y / Z axes, the REAL slice image (with its YOLO boxes) on a plane that
// moves with the slice slider, and one semi-transparent box per suspicious region. Each box is the hull of the region's
// 2D YOLO boxes over its slices (backend regions_3d): not a segmentation, not the lesion shape, not the liver (no liver
// model). Regions below the Evidence Gate "strong" confidence are drawn with dashed edges. Units: millimetres when the
// voxel spacing is known and regular, otherwise pixels / slices. Drag to rotate, wheel to zoom, click a box to select.
// Camera presets only rotate the view; no oblique re-slicing is done.
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Region3D } from "../api";

export const PALETTE = ["#ef4444", "#3b82f6", "#f59e0b", "#10b981", "#a855f7", "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16"];
export type CamView = "default" | "x" | "y" | "z";
export interface FrameSrc { img: CanvasImageSource; sy: number; w: number; h: number }
export interface SliceBox { box: number[]; conf: number; region: number }

function label(text: string, color: string, size = 44): THREE.Sprite {
  const c = document.createElement("canvas");
  const g = c.getContext("2d")!;
  g.font = `600 ${size}px Arial`;
  const w = Math.ceil(g.measureText(text).width) + 16;
  c.width = w; c.height = size + 16;
  g.font = `600 ${size}px Arial`;
  g.fillStyle = "rgba(17,24,39,0.82)"; g.fillRect(0, 0, w, c.height);
  g.fillStyle = color; g.textBaseline = "middle"; g.fillText(text, 8, c.height / 2);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, depthTest: false, transparent: true }));
  s.userData.aspect = w / c.height;
  return s;
}

export default function Lesion3D({ regions, width, height, nSlices, spacing, selectedId, currentK, hidden, onlySelected, strong, colors,
  getFrame, boxesAt, onSelect, view, viewSeq }: {
  regions: Region3D[]; width: number; height: number; nSlices: number; spacing: number[] | null;
  selectedId: number | null; currentK: number; hidden: Set<number>; onlySelected: boolean; strong: number | null; colors: Map<number, string>;
  getFrame: (k: number) => Promise<FrameSrc>; boxesAt: (k: number) => SliceBox[]; onSelect: (id: number) => void;
  view: CamView; viewSeq: number;
}) {
  const host = useRef<HTMLDivElement>(null);
  const api = useRef<{ setSel: (id: number | null) => void; setK: (k: number) => void; setVis: () => void; cam: (v: CamView) => void } | null>(null);
  const [err, setErr] = useState("");
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;
  const live = useRef({ selectedId, hidden, onlySelected, getFrame, boxesAt, colors });
  live.current = { selectedId, hidden, onlySelected, getFrame, boxesAt, colors };
  const sx = spacing?.[0] ?? 1, sy = spacing?.[1] ?? 1, sz = spacing?.[2] ?? 1;

  // layout effect: the canvas gets its final size before the browser paints (no small-then-large jump)
  useLayoutEffect(() => {
    const el = host.current;
    if (!el) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    } catch {
      setErr("Trình duyệt không hỗ trợ WebGL nên không hiển thị được khối 3D. Bảng vùng bên dưới vẫn đầy đủ.");
      return;
    }
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setClearColor(0x2b313b);
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const W = width * sx, H = height * sy, D = nSlices * sz;
    const R = Math.max(W, H, D);
    const camera = new THREE.PerspectiveCamera(40, 1, R / 100, R * 20);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    // volume coordinates -> scene (centred): X = image column, Y = up = image row reversed (top of the slice up), Z = slice
    const P = (x: number, y: number, k: number) => new THREE.Vector3(x * sx - W / 2, H / 2 - y * sy, k * sz - D / 2);
    const cam = (v: CamView) => {
      const d = R * 2.1;
      const pos = { default: new THREE.Vector3(R * 1.15, R * 0.75, R * 1.35), x: new THREE.Vector3(d, 0, 0), y: new THREE.Vector3(0, d, 0.0001), z: new THREE.Vector3(0, 0, d) }[v];
      camera.up.set(0, v === "y" ? 0 : 1, v === "y" ? -1 : 0);
      camera.position.copy(pos);
      controls.target.set(0, 0, 0);
      camera.lookAt(0, 0, 0);
      controls.update();
    };
    cam("default");

    scene.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(W, H, D)), new THREE.LineBasicMaterial({ color: 0x9ca3af })));
    const origin = P(0, 0, 0);
    const axes: [THREE.Vector3, string, number, number][] = [
      [new THREE.Vector3(1, 0, 0), "X (cột ảnh)", 0xf87171, W], [new THREE.Vector3(0, -1, 0), "Y (hàng ảnh)", 0x4ade80, H], [new THREE.Vector3(0, 0, 1), "Z (lát cắt)", 0x60a5fa, D]];
    for (const [dir, name, col, len] of axes) {
      scene.add(new THREE.ArrowHelper(dir, origin, len * 1.08, col, R * 0.03, R * 0.018));
      const s = label(name, "#" + col.toString(16).padStart(6, "0"), 40);
      s.position.copy(origin.clone().add(dir.clone().multiplyScalar(len * 1.16)));
      s.scale.set(R * 0.07 * s.userData.aspect, R * 0.07, 1);
      scene.add(s);
    }

    // real slice image (with its YOLO boxes) on the current slice plane
    const tex = document.createElement("canvas");
    tex.width = width; tex.height = height;
    const tg = tex.getContext("2d")!;
    tg.fillStyle = "#000"; tg.fillRect(0, 0, width, height);
    const texture = new THREE.CanvasTexture(tex);
    texture.colorSpace = THREE.SRGBColorSpace;
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(W, H), new THREE.MeshBasicMaterial({ map: texture, transparent: true, opacity: 0.88, side: THREE.DoubleSide, depthWrite: false }));
    const planeEdge = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(W, H)), new THREE.LineBasicMaterial({ color: 0x93c5fd }));
    scene.add(plane, planeEdge);
    let kSeq = 0;
    const setK = (k: number) => {
      const z = P(0, 0, k + 0.5).z;
      plane.position.set(0, 0, z);
      planeEdge.position.set(0, 0, z);
      const my = ++kSeq;
      live.current.getFrame(k).then((f) => {
        if (my !== kSeq) return;
        tg.drawImage(f.img, 0, f.sy, f.w, f.h, 0, 0, width, height);
        const lw = Math.max(2, width / 170);
        for (const b of live.current.boxesAt(k)) {
          const on = b.region === live.current.selectedId;
          tg.strokeStyle = live.current.colors.get(b.region) ?? "#f59e0b";
          tg.lineWidth = on ? lw * 1.8 : lw;
          tg.setLineDash(on ? [] : [lw * 3, lw * 2]);
          tg.strokeRect(b.box[0], b.box[1], b.box[2] - b.box[0], b.box[3] - b.box[1]);
        }
        tg.setLineDash([]);
        texture.needsUpdate = true;
        el.dataset.planeK = String(k);
      }).catch(() => undefined);
    };

    const meshes: THREE.Mesh[] = [];
    const boxes = new Map<number, { mesh: THREE.Mesh; edge: THREE.LineSegments; tag: THREE.Sprite; weak: boolean }>();
    regions.forEach((r) => {
      const b = r.bbox_vox;
      const a = P(b.x_min, b.y_min, b.z_min), c = P(b.x_max, b.y_max, b.z_max + 1);
      const size = new THREE.Vector3(Math.abs(c.x - a.x), Math.abs(c.y - a.y), Math.abs(c.z - a.z)).max(new THREE.Vector3(R * 0.004, R * 0.004, R * 0.004));
      const hex = live.current.colors.get(r.id) ?? "#f59e0b";
      const color = new THREE.Color(hex);
      const weak = strong != null && r.max_conf < strong;
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(size.x, size.y, size.z), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.3, depthWrite: false }));
      mesh.position.set((a.x + c.x) / 2, (a.y + c.y) / 2, (a.z + c.z) / 2);
      mesh.userData.id = r.id;
      const edge = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry),
        weak ? new THREE.LineDashedMaterial({ color, dashSize: R * 0.012, gapSize: R * 0.008 }) : new THREE.LineBasicMaterial({ color }));
      edge.position.copy(mesh.position);
      if (weak) edge.computeLineDistances();
      const tag = label(`Vùng ${r.num ?? "?"}${weak ? " (thấp)" : ""}`, hex, 40);
      tag.position.set(mesh.position.x, mesh.position.y + size.y / 2 + R * 0.035, mesh.position.z);
      tag.scale.set(R * 0.06 * tag.userData.aspect, R * 0.06, 1);
      scene.add(mesh, edge, tag);
      meshes.push(mesh);
      boxes.set(r.id, { mesh, edge, tag, weak });
    });

    const setVis = () => {
      const { hidden: hid, onlySelected: only, selectedId: sel } = live.current;
      let n = 0;
      boxes.forEach((v, id) => {
        const on = !hid.has(id) && (!only || id === sel);
        v.mesh.visible = v.edge.visible = v.tag.visible = on;
        n += on ? 1 : 0;
      });
      el.dataset.visibleRegions = String(n);
    };
    const setSel = (id: number | null) => {
      boxes.forEach((v, k) => {
        const on = k === id;
        (v.mesh.material as THREE.MeshBasicMaterial).opacity = on ? 0.55 : id == null ? 0.3 : v.weak ? 0.12 : 0.2;
      });
      setVis();
    };
    api.current = { setSel, setK, setVis, cam };
    setSel(live.current.selectedId);
    setK(currentK);

    const ray = new THREE.Raycaster();
    let downAt: [number, number] | null = null;
    const onDown = (e: PointerEvent) => { downAt = [e.clientX, e.clientY]; };
    const onUp = (e: PointerEvent) => {
      if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
      const rc = renderer.domElement.getBoundingClientRect();
      ray.setFromCamera(new THREE.Vector2(((e.clientX - rc.left) / rc.width) * 2 - 1, -((e.clientY - rc.top) / rc.height) * 2 + 1), camera);
      const hit = ray.intersectObjects(meshes.filter((m) => m.visible))[0];
      if (hit) onSelectRef.current(hit.object.userData.id as number);
    };
    renderer.domElement.addEventListener("pointerdown", onDown);
    renderer.domElement.addEventListener("pointerup", onUp);

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;      // fixed by CSS (aspect-ratio), identical to the loading box
      if (!w || !h) return;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();
    let raf = 0;
    const loop = () => { controls.update(); renderer.render(scene, camera); raf = requestAnimationFrame(loop); };
    loop();
    el.dataset.ready = "1";
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.domElement.removeEventListener("pointerdown", onDown);
      renderer.domElement.removeEventListener("pointerup", onUp);
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose?.();
        const mat = m.material as THREE.Material | THREE.Material[] | undefined;
        (Array.isArray(mat) ? mat : mat ? [mat] : []).forEach((x) => { (x as THREE.SpriteMaterial).map?.dispose(); x.dispose(); });
      });
      texture.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      api.current = null;
      delete el.dataset.ready;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [regions, width, height, nSlices, sx, sy, sz, strong]);

  useEffect(() => { api.current?.setSel(selectedId); api.current?.setK(currentK); }, [selectedId]);   // redraw boxes in the selected colour
  useEffect(() => { api.current?.setK(currentK); }, [currentK]);
  useEffect(() => { api.current?.setVis(); }, [hidden, onlySelected]);
  useEffect(() => { if (viewSeq) api.current?.cam(view); }, [view, viewSeq]);

  if (err) return <p className="notice info small">{err}</p>;
  return <div ref={host} className="l3d-host" aria-label="Không gian 3D: ảnh lát cắt thật, khối bao các vùng nghi ngờ. Kéo để xoay, cuộn để phóng to, bấm vào khối để chọn vùng." />;
}
