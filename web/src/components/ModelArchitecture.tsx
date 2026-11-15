// "Kiến trúc mô hình" on the Models page. Every number comes from web/src/data/model_arch.json, exported from the
// locked checkpoints by scripts/76_export_model_arch.py (sha256 checked, one dummy forward pass, nothing trained).
import raw from "../data/model_arch.json";

const A: any = raw;
const n = (x: number) => x.toLocaleString("vi-VN");
const shape = (s: number[] | null) => (s ? s.join(" × ") : "–");
const PART = { backbone: "Backbone", neck: "Neck", head: "Đầu ra" } as Record<string, string>;

export default function ModelArchitecture() {
  const y = A.yolo.ct;
  const ym = A.yolo.mri;
  const h = A.hybrid_ct;
  const nNeck = y.blocks.filter((b: any) => b.part === "neck").length;
  return (
    <section className="arch" aria-labelledby="arch-h">
      <div className="eyebrow">Kiến trúc mô hình</div>
      <h2 id="arch-h">Bên trong hai mô hình học sâu</h2>
      <p className="muted" style={{ maxWidth: "52em" }}>
        Số liệu đo trực tiếp từ checkpoint đang chạy trên website (kiểm tra SHA256), bằng một lần chạy thử ảnh trống trên CPU.
        Không huấn luyện lại, không thay đổi mô hình.
      </p>

      <div className="stats" style={{ margin: "18px 0 22px" }}>
        <div className="stat"><div className="value">{y.n_blocks}</div><div className="label">khối của YOLO11n</div>
          <div className="denom">{y.n_hidden_blocks} khối ẩn ({y.n_backbone} backbone + {nNeck} neck) + 1 khối Detect (đầu ra)</div></div>
        <div className="stat"><div className="value">{y.conv2d}</div><div className="label">lớp tích chập Conv2d</div>
          <div className="denom">{y.batchnorm} lớp BatchNorm; kích hoạt {y.activation}(x) = x·σ(x)</div></div>
        <div className="stat"><div className="value">{(y.params / 1e6).toLocaleString("vi-VN", { maximumFractionDigits: 2 })} tr</div>
          <div className="label">tham số YOLO11n</div><div className="denom">{n(y.params)} (CT), {n(ym.params)} (MRI, cùng kiến trúc)</div></div>
        <div className="stat"><div className="value">{(h.params / 1e6).toLocaleString("vi-VN", { maximumFractionDigits: 1 })} tr</div>
          <div className="label">tham số Hybrid CT</div><div className="denom">ResNet-18 + Swin-T + đầu hợp nhất</div></div>
      </div>

      <div className="grid grid-2">
        <div className="card">
          <div className="eyebrow">Phát hiện · CT và MRI</div>
          <h3>YOLO11n ({y.scale === "n" ? "nano" : y.scale}) – {y.nc} lớp «tổn thương»</h3>
          <div className="arch-flow" aria-label="Luồng dữ liệu YOLO11n">
            <div className="af in">Lát cắt {y.imgsz}×{y.imgsz}<small>xám → 3 kênh R = G = B</small></div>
            <div className="af bb">Backbone<small>khối 0–{y.n_backbone - 1}: Conv, C3k2, SPPF, C2PSA</small></div>
            <div className="af nk">Neck (FPN + PAN)<small>khối {y.n_backbone}–{y.n_blocks - 2}: Upsample, Concat, C3k2</small></div>
            <div className="af hd">Detect<small>3 tỉ lệ 64², 32², 16²; điểm tin cậy = σ(z)</small></div>
            <div className="af out">Đầu ra {shape(y.output)}<small>4 tọa độ hộp + 1 điểm tin cậy</small></div>
          </div>
          <p className="small muted" style={{ margin: "10px 0 0" }}>
            “Lớp ẩn” là các tầng nằm giữa đầu vào và đầu ra: {y.n_hidden_blocks} khối (0–{y.n_blocks - 2}), gồm {y.conv2d} lớp Conv2d.
            Khối {y.n_blocks - 1} (Detect) là tầng đầu ra.
          </p>
        </div>
        <div className="card">
          <div className="eyebrow">Phân loại · chỉ CT</div>
          <h3>Hybrid CNN + Transformer – {h.classes.join(" / ")}</h3>
          <div className="arch-two" aria-label="Luồng dữ liệu Hybrid">
            <div className="af cnn">{h.cnn.name}<small>{h.cnn.input} · {h.cnn.conv2d} Conv2d · {h.cnn.activation}</small>
              <small>{n(h.cnn.params)} tham số → vector {h.cnn.feature}</small></div>
            <div className="af vit">{h.transformer.name}<small>{h.transformer.input} · {h.transformer.blocks} khối Transformer ({h.transformer.depths.join("–")})</small>
              <small>đầu attention {h.transformer.heads.join("–")}, cửa sổ {h.transformer.window}×{h.transformer.window} · {n(h.transformer.params)} tham số → {h.transformer.feature}</small></div>
          </div>
          <div className="arch-flow" style={{ marginTop: 8 }}>
            <div className="af fu">Hợp nhất ({h.fusion}) → {h.fused_dim}<small>{n(h.head.params)} tham số</small></div>
            <div className="af out">softmax {h.classes.length} lớp<small>tổng xác suất = 1</small></div>
          </div>
          <ul className="small muted arch-list">
            {h.head.layers.map((t: string) => <li key={t}>{t}</li>)}
          </ul>
        </div>
      </div>

      <details className="mini more" style={{ marginTop: 18 }}>
        <summary>Bảng {y.n_blocks} khối của YOLO11n (đầu ra với ảnh {y.imgsz}×{y.imgsz})</summary>
        <div className="table-wrap" style={{ marginTop: 10 }}>
          <table className="arch-table">
            <thead><tr><th className="num">Khối</th><th>Phần</th><th>Loại</th><th>Nhận từ</th><th>Đầu ra (C × H × W)</th><th className="num">Tham số</th><th>Chức năng</th></tr></thead>
            <tbody>
              {y.blocks.map((b: any) => (
                <tr key={b.i} className={`p-${b.part}`}>
                  <td className="num">{b.i}</td><td>{PART[b.part]}</td><td><b>{b.type}</b></td><td>{b.from.join(", ")}</td>
                  <td className="tnum">{shape(b.out)}</td><td className="num">{n(b.params)}</td><td className="small">{b.desc}</td>
                </tr>
              ))}
              <tr className="tot"><td colSpan={5}><b>Tổng</b> · {y.n_hidden_blocks} khối ẩn + 1 khối đầu ra</td><td className="num"><b>{n(y.params)}</b></td><td /></tr>
            </tbody>
          </table>
        </div>
        <p className="small muted" style={{ margin: "8px 0 0" }}>Nguồn: {A.source} từ {y.weights}.</p>
      </details>
    </section>
  );
}
