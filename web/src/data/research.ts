// Measured numbers exported by scripts/97_export_web_metrics.py from the real result files (no hand-typed values).
import raw from "./research.json";

export const R: any = raw;
export const CT: any = R.modalities.ct;
export const MRI: any = R.modalities.mri;

export const pct = (k: number, n: number, d = 1) =>
  n ? `${((100 * k) / n).toLocaleString("vi-VN", { minimumFractionDigits: d, maximumFractionDigits: d })}%` : "–";
export const frac = (k: number, n: number) => `${k}/${n}`;
export const p100 = (x: number, d = 1) => `${(100 * x).toLocaleString("vi-VN", { minimumFractionDigits: d, maximumFractionDigits: d })}%`;
export const dec = (x: number, d = 3) => x.toLocaleString("vi-VN", { minimumFractionDigits: d, maximumFractionDigits: d });
export const ci = (a: number[], asPct = true, d = 1) => (asPct ? `${p100(a[0], d)} – ${p100(a[1], d)}` : `${dec(a[0])} – ${dec(a[1])}`);
