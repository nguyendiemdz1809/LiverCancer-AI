// Pseudo-colour ("rải màu") for DISPLAY ONLY: a gray CT/MRI slice (R = G = B) is mapped to an RGB colour scale by an
// SVG filter (luminance -> per-channel lookup table). The AI always analyses the original gray image; downloads and
// reports keep the original gray image. Tables: 17 samples of the matplotlib colormaps (linear interpolation between).
export type CmapName = "gray" | "jet" | "hot" | "bone" | "viridis";

export const CMAPS: { id: CmapName; label: string; rgb?: [string, string, string] }[] = [
  { id: "gray", label: "Xám (gốc)" },
  { id: "jet", label: "Jet", rgb: [
    "0 0 0 0 0 0 0.085 0.288 0.49 0.693 0.895 1 1 1 1 0.767 0.5",
    "0 0 0.002 0.253 0.504 0.755 1 1 1 1 1 0.814 0.582 0.349 0.117 0 0",
    "0.5 0.785 1 1 1 1 0.882 0.68 0.478 0.275 0.073 0 0 0 0 0 0"] },
  { id: "hot", label: "Hot", rgb: [
    "0.042 0.206 0.371 0.536 0.7 0.865 1 1 1 1 1 1 1 1 1 1 1",
    "0 0 0 0 0 0 0.03 0.195 0.359 0.524 0.689 0.853 1 1 1 1 1",
    "0 0 0 0 0 0 0 0 0 0 0 0 0.027 0.274 0.521 0.768 1"] },
  { id: "bone", label: "Bone", rgb: [
    "0 0.055 0.11 0.165 0.22 0.275 0.329 0.384 0.439 0.494 0.549 0.604 0.662 0.748 0.834 0.92 1",
    "0 0.055 0.11 0.165 0.22 0.275 0.333 0.409 0.484 0.56 0.635 0.711 0.784 0.839 0.894 0.949 1",
    "0 0.076 0.153 0.229 0.306 0.382 0.454 0.509 0.564 0.619 0.674 0.729 0.784 0.839 0.894 0.949 1"] },
  { id: "viridis", label: "Viridis", rgb: [
    "0.267 0.282 0.279 0.259 0.23 0.199 0.173 0.149 0.128 0.121 0.158 0.246 0.369 0.516 0.678 0.846 0.993",
    "0.005 0.095 0.175 0.252 0.322 0.388 0.449 0.508 0.567 0.626 0.684 0.739 0.789 0.831 0.864 0.887 0.906",
    "0.329 0.417 0.483 0.525 0.546 0.555 0.558 0.557 0.551 0.533 0.502 0.452 0.383 0.294 0.19 0.1 0.144"] },
];

export const cmapFilter = (id: CmapName) => (id === "gray" ? undefined : `url(#lc-cmap-${id})`);

/** CSS gradient of a colour scale (colour bar next to the selector). */
export function cmapGradient(id: CmapName): string {
  const c = CMAPS.find((x) => x.id === id);
  if (!c?.rgb) return "linear-gradient(90deg, #000, #fff)";
  const ch = c.rgb.map((s) => s.split(" ").map(Number));
  const stops = ch[0].map((_, i) => `rgb(${ch.map((v) => Math.round(255 * v[i])).join(",")}) ${((100 * i) / (ch[0].length - 1)).toFixed(1)}%`);
  return `linear-gradient(90deg, ${stops.join(", ")})`;
}

/** Hidden SVG with one filter per colour scale; rendered once by the viewer. sRGB space, otherwise the browser applies
 *  the table in linearRGB and the gray levels shift. */
export function CmapDefs() {
  return (
    <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden="true" focusable="false">
      <defs>
        {CMAPS.filter((c) => c.rgb).map((c) => (
          <filter key={c.id} id={`lc-cmap-${c.id}`} colorInterpolationFilters="sRGB">
            <feColorMatrix type="matrix" values="0.299 0.587 0.114 0 0  0.299 0.587 0.114 0 0  0.299 0.587 0.114 0 0  0 0 0 1 0" />
            <feComponentTransfer>
              <feFuncR type="table" tableValues={c.rgb![0]} />
              <feFuncG type="table" tableValues={c.rgb![1]} />
              <feFuncB type="table" tableValues={c.rgb![2]} />
            </feComponentTransfer>
          </filter>
        ))}
      </defs>
    </svg>
  );
}
