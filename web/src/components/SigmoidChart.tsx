// Sigmoid of the YOLO detection score (display only). YOLO's class branch outputs a logit z for the "lesion" class and
// the confidence shown everywhere on the site is sigma(z) = 1 / (1 + e^-z) (Ultralytics Detect head). Here z is
// recovered from each region's confidence: z = ln(c / (1 - c)). Nothing is recomputed by a model.
import { fmtNum } from "../api";

const W = 320, H = 178, L = 34, R = 10, T = 10, B = 30;
const ZMIN = -6, ZMAX = 6;
const sig = (z: number) => 1 / (1 + Math.exp(-z));
const logit = (c: number) => { const p = Math.min(0.9975, Math.max(0.0025, c)); return Math.log(p / (1 - p)); };
const X = (z: number) => L + ((Math.max(ZMIN, Math.min(ZMAX, z)) - ZMIN) / (ZMAX - ZMIN)) * (W - L - R);
const Y = (p: number) => T + (1 - p) * (H - T - B);
const zs = (x: number) => x.toLocaleString("vi-VN", { maximumFractionDigits: 2, minimumFractionDigits: 2, signDisplay: "exceptZero" });

export default function SigmoidChart({ confs, threshold, strong, sel, selLabel }: {
  confs: number[];                 // detection confidence of every region of the case (sigma(z))
  threshold: number;               // locked YOLO threshold
  strong?: number | null;          // Evidence Gate "strong region" level (CT), display only
  sel?: number | null;             // confidence of the selected region
  selLabel?: string;
}) {
  const curve = Array.from({ length: 97 }, (_, i) => ZMIN + (i * (ZMAX - ZMIN)) / 96)
    .map((z, i) => `${i ? "L" : "M"}${X(z).toFixed(1)},${Y(sig(z)).toFixed(1)}`).join(" ");
  const zt = logit(threshold);
  const zSel = sel != null ? logit(sel) : null;
  return (
    <figure className="sigmoid" aria-label="Đồ thị hàm sigmoid của độ tin cậy YOLO">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-hidden="true">
        <rect x={X(zt)} y={T} width={X(ZMAX) - X(zt)} height={H - T - B} className="sg-keep" />
        {[0, 0.5, 1].map((p) => (
          <g key={p}><line x1={L} x2={W - R} y1={Y(p)} y2={Y(p)} className="sg-grid" /><text x={L - 5} y={Y(p) + 3} className="sg-ax" textAnchor="end">{p.toLocaleString("vi-VN")}</text></g>
        ))}
        {[-6, -4, -2, 0, 2, 4, 6].map((z) => (
          <text key={z} x={X(z)} y={H - B + 13} className="sg-ax" textAnchor="middle">{z}</text>
        ))}
        <text x={(L + W - R) / 2} y={H - 3} className="sg-ax" textAnchor="middle">logit z</text>
        <line x1={L} x2={W - R} y1={Y(threshold)} y2={Y(threshold)} className="sg-thr" />
        <line x1={X(zt)} x2={X(zt)} y1={T} y2={H - B} className="sg-thr" />
        <text x={X(zt) + 3} y={T + 9} className="sg-lb">ngưỡng {fmtNum(threshold, 2)}</text>
        {strong != null && (
          <><line x1={L} x2={W - R} y1={Y(strong)} y2={Y(strong)} className="sg-strong" />
            <text x={W - R - 2} y={Y(strong) - 3} className="sg-lb" textAnchor="end">vùng mạnh {fmtNum(strong, 2)}</text></>
        )}
        <path d={curve} className="sg-curve" />
        {confs.map((c, i) => <circle key={i} cx={X(logit(c))} cy={Y(c)} r={3.2} className="sg-pt" />)}
        {zSel != null && sel != null && <circle cx={X(zSel)} cy={Y(sel)} r={5.5} className="sg-sel" />}
      </svg>
      <figcaption className="small muted">
        <b>σ(z) = 1 / (1 + e<sup>−z</sup>)</b>: YOLO tính logit z cho lớp “tổn thương”, độ tin cậy là σ(z) ∈ (0, 1).
        Chỉ giữ vùng có σ(z) ≥ {fmtNum(threshold, 2)}, tức z ≥ {zs(zt)} (vùng tô xanh).
        {zSel != null && sel != null && <> {selLabel ?? "Vùng đang chọn"}: z = {zs(zSel)}, σ(z) = {fmtNum(sel)}.</>}
        {" "}Chấm xám: {confs.length} vùng của ca. Đây là độ chắc chắn phát hiện, không phải xác suất ung thư.
      </figcaption>
    </figure>
  );
}
