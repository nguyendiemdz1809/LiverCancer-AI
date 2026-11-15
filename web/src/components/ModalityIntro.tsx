import type { Modality } from "../api";

function CtIcon() {
  return (
    <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
      <circle cx="24" cy="24" r="17" /><circle cx="24" cy="24" r="9" /><path d="M24 7v4M24 37v4M7 24h4M37 24h4" />
      <circle cx="28.5" cy="20.5" r="2.6" fill="currentColor" stroke="none" />
    </svg>
  );
}
function MriIcon() {
  return (
    <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="6" y="10" width="36" height="28" rx="14" /><ellipse cx="24" cy="24" rx="8" ry="6" />
      <path d="M13 24h3M32 24h3M24 14v2M24 32v2" />
    </svg>
  );
}

const CARDS: { key: Modality; title: string; pick: string; text: string; icon: JSX.Element }[] = [
  { key: "ct", title: "Phân tích ảnh CT", pick: "Phân tích CT", icon: <CtIcon />,
    text: "Ứng dụng trí tuệ nhân tạo để phát hiện, khoanh vùng và phóng to tổn thương gan trên ảnh CT, hỗ trợ nhận diện và phân loại các dạng tổn thương HCC, ICC và CHCC." },
  { key: "mri", title: "Phân tích ảnh MRI", pick: "Phân tích MRI", icon: <MriIcon />,
    text: "Ứng dụng mô hình học sâu để phát hiện và định vị vùng tổn thương gan trên ảnh cộng hưởng từ, kết hợp công nghệ phóng to thông minh giúp quan sát chi tiết vùng nghi ngờ." },
];

// Without onChange: the two description cards (home page). With onChange: the CT / MRI choice before upload.
export default function ModalityIntro({ value, onChange, disabled }: { value?: Modality | null; onChange?: (m: Modality) => void; disabled?: boolean }) {
  if (!onChange) {
    return (
      <div className="intro-grid">
        {CARDS.map((c) => (
          <div key={c.key} className="intro-card">
            <div className={`intro-icon${c.key === "mri" ? " mri" : ""}`}>{c.icon}</div>
            <div><h3>{c.title}</h3><p>{c.text}</p></div>
          </div>
        ))}
      </div>
    );
  }
  return (
    <div className="intro-grid picker" role="radiogroup" aria-label="Chọn loại ảnh cần phân tích">
      {CARDS.map((c) => (
        <button key={c.key} type="button" role="radio" aria-checked={value === c.key} disabled={disabled}
          className={`intro-card pick${value === c.key ? " on" : ""}`} onClick={() => onChange(c.key)}>
          <div className={`intro-icon${c.key === "mri" ? " mri" : ""}`}>{c.icon}</div>
          <div>
            <h3>{c.pick}</h3>
            <p>{c.text}</p>
          </div>
          <span className="pick-mark" aria-hidden="true">{value === c.key ? "✓" : ""}</span>
        </button>
      ))}
    </div>
  );
}
