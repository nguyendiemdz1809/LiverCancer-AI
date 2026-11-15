import { Link } from "react-router-dom";
import ModalityIntro from "../components/ModalityIntro";
import { CT, frac, pct } from "../data/research";

export default function Home() {
  const t = CT.test;
  const c = t.controls;
  return (
    <>
      <section className="hero">
        <div className="container hero-grid">
          <div>
            <div className="eyebrow">Đồ án tốt nghiệp · Học sâu trong y tế</div>
            <h1>LiverCancer AI</h1>
            <p className="lead">
              Hệ thống nghiên cứu hỗ trợ <b>phát hiện</b> và <b>phân loại</b> ung thư gan trên ảnh CT và MRI:
              YOLO11n tìm vùng nghi ngờ, Tumor Zoom tự phóng to, mô hình Hybrid CNN + Transformer phân loại tổn thương trên CT.
            </p>
            <div className="hero-actions">
              <Link className="btn btn-primary" to="/phan-tich">Bắt đầu phân tích ảnh</Link>
              <Link className="btn btn-ghost" to="/ket-qua">Xem kết quả nghiên cứu</Link>
            </div>
          </div>
          <div className="steps" aria-label="Quy trình">
            <div className="step"><span className="step-num">1</span><div><b>Tải ảnh CT/MRI</b><span className="muted small">NIfTI, DICOM, PNG/JPG hoặc ảnh chụp màn hình.</span></div></div>
            <div className="step"><span className="step-num">2</span><div><b>AI tự phân tích</b><span className="muted small">Tự nhận dạng CT/MRI, phát hiện và phóng to vùng nghi ngờ.</span></div></div>
            <div className="step"><span className="step-num">3</span><div><b>Xem kết quả</b><span className="muted small">Hình ảnh, biểu đồ và nhận xét bằng tiếng Việt.</span></div></div>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="container">
          <h2>Hệ thống làm được gì</h2>
          <div style={{ marginTop: 16 }}><ModalityIntro /></div>
        </div>
      </section>

      <section className="section band">
        <div className="container">
          <h2>Kết quả trên tập Test độc lập (CT)</h2>
          <p className="muted">Đo một lần với cấu hình đã khóa trước khi chạy. Đơn vị: bệnh nhân. Ca bị từ chối phân loại không được tính là đúng.</p>
          <div className="stats" style={{ marginTop: 16 }}>
            <div className="stat"><div className="value">{pct(t.counts.correct, t.n_lesion_patients)}</div>
              <div className="label">Bệnh nhân ung thư được phân loại đúng</div><div className="denom">{frac(t.counts.correct, t.n_lesion_patients)} bệnh nhân</div></div>
            <div className="stat"><div className="value">{frac(t.counts.rejected, t.n_lesion_patients)}</div>
              <div className="label">Bệnh nhân ung thư bị từ chối phân loại</div><div className="denom">do bằng chứng hình ảnh yếu</div></div>
            <div className="stat"><div className="value">{frac(c.cancer_label, c.n)}</div>
              <div className="label">Nhóm chứng bị gán nhãn ung thư</div><div className="denom">khi tắt cổng bằng chứng: {frac(c.ungated_cancer_label, c.n)}</div></div>
          </div>
          <p style={{ marginTop: 16 }}><Link to="/ket-qua">Xem đầy đủ chỉ số Validation và Test, kèm mẫu số và giới hạn →</Link></p>
        </div>
      </section>

      <section className="section-tight">
        <div className="container">
          <p className="footnote">Kết quả AI chỉ phục vụ nghiên cứu và tham khảo, không thay thế đánh giá của bác sĩ.
            Không phát hiện vùng nghi ngờ không đồng nghĩa với loại trừ bệnh.</p>
        </div>
      </section>
    </>
  );
}
