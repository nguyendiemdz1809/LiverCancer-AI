import { CT, MRI, R, ci, dec, frac, pct } from "../data/research";

function Cell({ k, n }: { k: number; n: number }) {
  return <td className="num">{frac(k, n)}<span className="sub">{pct(k, n)}</span></td>;
}

function DetectionRows({ m, label }: { m: any; label: string }) {
  return (
    <>
      {[["val", "Validation"], ["test", "Test"]].map(([s, name]) => {
        const d = m[s].detection;
        return (
          <tr key={s}>
            <td>{label} · {name}</td>
            <Cell k={d.volumes_lesion_detected} n={d.volumes_with_lesion} />
            <Cell k={d.lesions_detected} n={d.lesions} />
            <Cell k={d.small_detected} n={d.small} />
            <td className="num">{dec(d.false_tracks_per_volume, 1)}</td>
          </tr>
        );
      })}
    </>
  );
}

const VARIANT_VI: Record<string, string> = {
  clean: "Lát cắt xuất gốc (PNG)", degraded: "Ảnh xuất bị nén / đổi kích thước / độ sáng", screenshot: "Ảnh chụp màn hình / chụp lại màn hình (tổng hợp)",
};

function ModalitySection({ m }: { m: any }) {
  const y = m.yolo_on_synthetic_screenshots;
  return (
    <section>
      <h2>Tự nhận dạng ảnh CT / MRI (bước tiếp nhận)</h2>
      <p className="muted">Mô hình riêng của website, không thuộc pipeline đã khóa. DICOM dùng thẻ Modality; NIfTI dùng header; ảnh 2D dùng một CNN nhỏ.
        CNN chỉ tự quyết khi độ tin cậy ≥ <b>{dec(m.threshold, 3)}</b> (ngưỡng chọn trên nửa A của Validation); thấp hơn thì từ chối. Website không dùng CNN này để quyết định (người dùng chọn CT hoặc MRI); số liệu giữ lại để báo cáo.
        Số liệu dưới đây đo trên nửa B ({m.n_patients_valB} bệnh nhân Validation không dùng để chọn gì).</p>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Loại ảnh 2D (Validation, nửa B)</th><th className="num">Số ảnh</th><th className="num">Tự quyết</th><th className="num">Đúng khi tự quyết</th><th className="num">Nhầm CT↔MRI</th></tr></thead>
          <tbody>
            {Object.entries(m.valB).map(([k, b]: [string, any]) => (
              <tr key={k}><td>{VARIANT_VI[k] || k}</td><td className="num">{b.n}</td><td className="num">{frac(b.accepted, b.n)}<span className="sub">{pct(b.accepted, b.n)}</span></td>
                <td className="num">{b.accepted_accuracy == null ? "–" : `${dec(100 * b.accepted_accuracy, 1)}%`}</td><td className="num">{b.accepted_errors}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted" style={{ marginTop: 10 }}>
        Quy tắc header NIfTI trên mọi tệp Train + Validation: CT nhận đúng {m.nifti_header_rule.ct.ct ?? 0}, MRI nhận đúng {m.nifti_header_rule.mri.mri ?? 0},
        nhận nhầm {(m.nifti_header_rule.ct.mri ?? 0) + (m.nifti_header_rule.mri.ct ?? 0)}.
      </p>
      {y && (
        <>
          <h3 style={{ marginTop: 18 }}>YOLO trên ảnh chụp màn hình (tổng hợp từ lát cắt Validation)</h3>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Ảnh</th><th className="num">Lát có tổn thương</th><th className="num">Ảnh gốc</th><th className="num">Ảnh chụp màn hình</th><th className="num">Chụp lại màn hình</th></tr></thead>
              <tbody>{["ct", "mri"].map((k) => (
                <tr key={k}><td>{k.toUpperCase()}</td><td className="num">{y[k].n_lesion_slices}</td><Cell k={y[k].detected.original} n={y[k].n_lesion_slices} />
                  <Cell k={y[k].detected.screenshot} n={y[k].n_lesion_slices} /><Cell k={y[k].detected.photo} n={y[k].n_lesion_slices} /></tr>
              ))}</tbody>
            </table>
          </div>
          <p className="small muted" style={{ marginTop: 10 }}>Tỷ lệ lát mà YOLO (đã khóa, ngưỡng 0,12) tìm thấy tổn thương chuẩn; ảnh bị từ chối tính là không tìm thấy.
            Ảnh chụp màn hình do chương trình tạo, <b>chưa kiểm chứng trên ảnh bệnh viện thật</b>.</p>
        </>
      )}
    </section>
  );
}

export default function Research() {
  const v = CT.val, t = CT.test;
  return (
    <div className="section">
      <div className="container grid" style={{ gap: 34 }}>
        <div>
          <div className="eyebrow">Kết quả nghiên cứu</div>
          <h1>Chỉ số đo được trên Validation và Test</h1>
          <div className="notice info">
            <b>Cách đọc.</b>
            <ul>
              <li><b>Validation</b> dùng để chọn mô hình, ngưỡng YOLO, cách gộp và Evidence Gate. <b>Test</b> được chạy <b>đúng một lần</b> với cấu hình đã khóa
                (mã phiên bản <code>{R.locked_commit}</code>), không chỉnh gì theo Test.</li>
              <li>Đơn vị là <b>bệnh nhân</b>. Ca <b>bị từ chối phân loại</b> hoặc <b>YOLO không tìm thấy vùng</b> được tính là <b>không đúng</b>.</li>
              <li>Mỗi ô ghi <b>số ca / mẫu số</b> và tỷ lệ. Tập dữ liệu nhỏ nên khoảng tin cậy rộng; chênh lệch vài ca giữa Validation và Test là dao động bình thường.</li>
            </ul>
          </div>
        </div>

        <section>
          <h2>CT – bệnh nhân ung thư (HCC / ICC / CHCC)</h2>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Tập</th><th className="num">Đúng</th><th className="num">Sai</th><th className="num">Từ chối phân loại</th><th className="num">YOLO không có vùng</th>
                <th className="num">Accuracy (trên mọi BN)</th><th className="num">Macro-F1</th><th className="num">Đúng và tìm thấy tổn thương thật</th></tr></thead>
              <tbody>
                <tr><td>Validation</td><td className="num">{v.counts.correct}</td><td className="num">{v.counts.wrong}</td><td className="num">{v.counts.rejected}</td><td className="num">{v.counts.no_region}</td>
                  <Cell k={v.counts.correct} n={v.n_lesion_patients} /><td className="num">{dec(v.classification.macro_f1)}</td><Cell k={v.strict_correct} n={v.n_lesion_patients} /></tr>
                <tr><td><b>Test</b></td><td className="num">{t.counts.correct}</td><td className="num">{t.counts.wrong}</td><td className="num">{t.counts.rejected}</td><td className="num">{t.counts.no_region}</td>
                  <td className="num">{frac(t.counts.correct, t.n_lesion_patients)}<span className="sub">{pct(t.counts.correct, t.n_lesion_patients)} · KTC 95% {ci(t.accuracy_ci95)}</span></td>
                  <td className="num">{dec(t.classification.macro_f1)}<span className="sub">KTC 95% {ci(t.macro_f1_ci95, false)}</span></td><Cell k={t.strict_correct} n={t.n_lesion_patients} /></tr>
              </tbody>
            </table>
          </div>
          <p className="small muted" style={{ marginTop: 10 }}>
            Cột cuối chỉ tính đúng khi YOLO tìm thấy tổn thương thật: trên Test có {t.counts.correct - t.strict_correct} ca đúng nhưng nhãn đến từ vùng giả (tổn thương thật quá nhỏ),
            nên theo cách tính chặt Test và Validation bằng nhau. Đối chiếu (oracle, vùng cắt theo nhãn chuẩn, <b>không phải kết quả hệ thống</b>): Test {frac(t.oracle.correct, t.oracle.n)}.
          </p>
          <h3 style={{ marginTop: 20 }}>Theo từng lớp (Test)</h3>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Lớp</th><th className="num">Số BN</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th></tr></thead>
              <tbody>{t.classification.per_class.map((c: any) => (
                <tr key={c.code}><td>{c.name}</td><td className="num">{c.support}</td><td className="num">{dec(c.precision)}</td><td className="num">{dec(c.recall)}</td><td className="num">{dec(c.f1)}</td></tr>
              ))}</tbody>
            </table>
          </div>
        </section>

        <section>
          <h2>CT – nhóm chứng (không có bệnh gan mật)</h2>
          <p className="muted">Mô hình 3 lớp không có lớp "không tổn thương". Hành vi mong muốn là <b>không gán nhãn ung thư</b>; điều này vẫn không có nghĩa hệ thống kết luận "khỏe mạnh".</p>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Tập</th><th className="num">Bị gán nhãn ung thư (cảnh báo sai)</th><th className="num">Từ chối phân loại</th><th className="num">YOLO không có vùng</th></tr></thead>
              <tbody>
                <tr><td>Validation</td><Cell k={v.controls.cancer_label} n={v.controls.n} /><td className="num">{v.controls.rejected}</td><td className="num">{v.controls.no_region}</td></tr>
                <tr><td><b>Test</b></td><td className="num">{frac(t.controls.cancer_label, t.controls.n)}<span className="sub">{pct(t.controls.cancer_label, t.controls.n)} · KTC 95% {ci(t.controls.cancer_label_ci95)}</span></td>
                  <td className="num">{t.controls.rejected}</td><td className="num">{t.controls.no_region}</td></tr>
                <tr><td className="muted">Test, nếu tắt Evidence Gate (chỉ để mô tả)</td><Cell k={t.controls.ungated_cancer_label} n={t.controls.n} /><td className="num">—</td><td className="num">—</td></tr>
              </tbody>
            </table>
          </div>
          <p className="small muted" style={{ marginTop: 10 }}>Đổi lại, khi bật cổng trên Test, {t.counts.rejected}/{t.n_lesion_patients} bệnh nhân ung thư bị từ chối phân loại
            (tắt cổng: đúng {t.ungated_correct}/{t.n_lesion_patients}).</p>
        </section>

        <section>
          <h2>Phát hiện tổn thương (YOLO11n)</h2>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Ảnh · tập</th><th className="num">Ảnh có tổn thương được tìm thấy</th><th className="num">Tổn thương 3D được tìm thấy</th>
                <th className="num">Tổn thương nhỏ (&lt; 32×32 px)</th><th className="num">Vùng giả / ảnh</th></tr></thead>
              <tbody><DetectionRows m={CT} label="CT" /><DetectionRows m={MRI} label="MRI" /></tbody>
            </table>
          </div>
          <p className="small muted" style={{ marginTop: 10 }}>
            Validation mức hộp (Ultralytics): CT precision {dec(CT.yolo.precision)}, recall {dec(CT.yolo.recall)}, mAP50 {dec(CT.yolo.map50)}, mAP50-95 {dec(CT.yolo.map50_95)};
            MRI precision {dec(MRI.yolo.precision)}, recall {dec(MRI.yolo.recall)}, mAP50 {dec(MRI.yolo.map50)}, mAP50-95 {dec(MRI.yolo.map50_95)}.
            Không so sánh CT với MRI bằng mAP vì khác dữ liệu và cách gán nhãn.
          </p>
        </section>

        <section>
          <h2>MRI – thí nghiệm phân loại 7 lớp (chưa đạt yêu cầu)</h2>
          <div className="notice warn">Kết quả dưới đây chỉ công bố như một thí nghiệm nghiên cứu. Phân loại MRI <b>không</b> được dùng trên website.</div>
          <div className="table-wrap" style={{ marginTop: 14 }}>
            <table>
              <thead><tr><th>Tập</th><th className="num">Đúng</th><th className="num">Accuracy</th><th className="num">Macro-F1</th><th className="num">Đúng nhóm lành / ác tính</th></tr></thead>
              <tbody>
                <tr><td>Validation</td><td className="num">{MRI.val.counts.correct}</td><Cell k={MRI.val.counts.correct} n={MRI.val.n_patients} /><td className="num">{dec(MRI.val.classification.macro_f1)}</td><td className="num">—</td></tr>
                <tr><td><b>Test</b></td><td className="num">{MRI.test.counts.correct}</td>
                  <td className="num">{frac(MRI.test.counts.correct, MRI.test.n_patients)}<span className="sub">{pct(MRI.test.counts.correct, MRI.test.n_patients)} · KTC 95% {ci(MRI.test.accuracy_ci95)}</span></td>
                  <td className="num">{dec(MRI.test.classification.macro_f1)}</td><Cell k={MRI.test.groups.group_correct} n={MRI.test.groups.n} /></tr>
              </tbody>
            </table>
          </div>
          <p className="small muted" style={{ marginTop: 10 }}>Test: {MRI.test.groups.benign_as_malignant}/{MRI.test.groups.n_benign} tổn thương lành tính bị gán lớp ác tính;
            nhiều tổn thương bị hút về lớp HCC.</p>
        </section>

        {R.modality_recognition && <ModalitySection m={R.modality_recognition} />}

        <section>
          <h2>Giới hạn</h2>
          <ul className="muted">
            <li>Tập Test nhỏ: CT {t.n_patients} bệnh nhân ({t.n_lesion_patients} ung thư, {t.controls.n} nhóm chứng), MRI {MRI.test.n_patients} bệnh nhân; một lần huấn luyện, một seed.</li>
            <li>YOLO bỏ sót phần lớn tổn thương nhỏ trên CT; không thấy vùng nghi ngờ không loại trừ được bệnh.</li>
            <li>Nhóm chứng CT là "không có bệnh gan mật" theo nguồn dữ liệu, không phải "khỏe mạnh hoàn toàn". Từ chối phân loại không phải kết luận âm tính.</li>
            <li>Mô hình chỉ được kiểm chứng trên định dạng của hai bộ dữ liệu (PLC-CECT, LLD-MMRI); ảnh từ máy chụp hoặc quy trình khác có thể cho kết quả khác.</li>
          </ul>
        </section>

        <section>
          <h3>Nguồn số liệu</h3>
          <p className="small muted" style={{ margin: 0 }}>
            Sinh tự động bởi <code>scripts/97_export_web_metrics.py</code> từ: <code>{CT.test.source}</code>, <code>{CT.val.source}</code>, <code>{MRI.val.source}</code>,
            <code> {CT.yolo.source}</code>, <code>{MRI.yolo.source}</code>, <code>{CT.roi_classifier_val.source}</code>, <code>{MRI.roi_classifier_val.source}</code>.
            Lần chạy Test: {R.test_run.started.replace("T", " ")} – {(R.test_run.finished || "").replace("T", " ")}.
          </p>
        </section>
      </div>
    </div>
  );
}
