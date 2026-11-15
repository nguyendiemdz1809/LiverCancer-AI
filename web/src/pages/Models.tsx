import { CT, MRI, dec, p100 } from "../data/research";
import ModelArchitecture from "../components/ModelArchitecture";

export default function Models() {
  const g = CT.evidence_gate;
  return (
    <div className="section">
      <div className="container">
        <div className="eyebrow">Thông tin mô hình</div>
        <h1>Từ ảnh gốc đến kết quả</h1>
        <p className="muted" style={{ maxWidth: "50em" }}>
          Mọi bước dùng đúng mô hình và ngưỡng đã chọn trên tập Validation, sau đó khóa lại và đánh giá một lần trên tập Test.
          Website dùng các checkpoint thật, không có kết quả mẫu.
        </p>

        <div className="flow" style={{ margin: "26px 0 34px" }} aria-label="Sơ đồ pipeline">
          <div className="node"><b>Ảnh đầu vào</b><span>NIfTI, DICOM, PNG/JPG, ảnh chụp màn hình</span></div><span className="arrow">→</span>
          <div className="node"><b>Nhận dạng CT/MRI</b><span>DICOM, header hoặc mô hình ảnh</span></div><span className="arrow">→</span>
          <div className="node"><b>YOLO11n</b><span>tìm vùng nghi ngờ trên từng lát</span></div><span className="arrow">→</span>
          <div className="node"><b>Tumor Zoom</b><span>nối vùng qua các lát, phóng to</span></div><span className="arrow">→</span>
          <div className="node"><b>Hybrid</b><span>ResNet18 + Swin-T (chỉ CT)</span></div><span className="arrow">→</span>
          <div className="node"><b>Evidence Gate</b><span>chỉ kết luận khi đủ bằng chứng</span></div>
        </div>

        <div className="grid grid-2">
          <div className="card">
            <div className="eyebrow">Bước 0 · tiếp nhận ảnh</div>
            <h3>Người dùng chọn CT hoặc MRI</h3>
            <p className="muted">Người dùng chọn <b>Phân tích CT</b> hoặc <b>Phân tích MRI</b>; hệ thống dùng đúng mô hình của loại ảnh đó.
              Thông tin trong tệp (thẻ <b>Modality</b> của DICOM, header NIfTI) được đối chiếu với lựa chọn: nếu mâu thuẫn, website hỏi lại trước khi phân tích.
              Với ảnh PNG/JPG hoặc ảnh chụp màn hình, hệ thống cắt bỏ giao diện phần mềm và che chữ ngoài vùng cơ thể trước khi phân tích.</p>
            <p className="small muted" style={{ margin: 0 }}>Bước tiếp nhận không thay đổi mô hình phát hiện và phân loại đã khóa. Không dùng tên tệp để quyết định.</p>
          </div>
          <div className="card">
            <div className="eyebrow">Bước 1</div>
            <h3>YOLO11n – phát hiện vùng nghi ngờ</h3>
            <p className="muted">Mô hình phát hiện đối tượng một lớp ("tổn thương"), huấn luyện riêng cho CT và MRI, ảnh {CT.yolo.imgsz}×{CT.yolo.imgsz}.
              Ngưỡng tin cậy <b>{dec(CT.yolo.conf, 2)}</b> được chọn trên Validation theo tiêu chí F2 (ưu tiên không bỏ sót), nên có thể có vùng giả.</p>
            <p className="small muted" style={{ margin: 0 }}>Validation mức hộp: CT mAP50 {dec(CT.yolo.map50)}, MRI mAP50 {dec(MRI.yolo.map50)}.</p>
          </div>
          <div className="card">
            <div className="eyebrow">Bước 2</div>
            <h3>Automatic Tumor Zoom</h3>
            <p className="muted" style={{ margin: 0 }}>Gộp các hộp trùng trên cùng lát, nối thành vùng 3D qua các lát liền kề, rồi cắt hai ảnh 224×224:
              <b> ảnh cục bộ</b> (khối u và lề mô) và <b>ảnh ngữ cảnh</b> (vùng rộng xung quanh). Người dùng không cần bấm zoom.</p>
          </div>
          <div className="card">
            <div className="eyebrow">Bước 3 · chỉ CT</div>
            <h3>Hybrid CNN + Transformer</h3>
            <p className="muted">Nhánh CNN (ResNet18) học chi tiết từ ảnh cục bộ, nhánh Transformer (Swin-T) học ngữ cảnh. Hai đặc trưng được ghép lại để phân loại
              <b> HCC</b>, <b>ICC</b>, <b>CHCC</b>. Kết luận của bệnh nhân là trung bình xác suất của mọi vùng YOLO tìm được.</p>
            <p className="small muted" style={{ margin: 0 }}>Validation mức vùng cắt theo nhãn chuẩn: macro-F1 {dec(CT.roi_classifier_val.macro_f1)} trên {CT.roi_classifier_val.n_rois.toLocaleString("vi-VN")} vùng.</p>
          </div>
          <div className="card">
            <div className="eyebrow">Bước 4 · chỉ CT</div>
            <h3>Evidence Gate – cổng bằng chứng</h3>
            <p className="muted">Mô hình 3 lớp không có lớp "không tổn thương", nên vùng giả cũng bị gán nhãn. Cổng chỉ cho phép kết luận khi có vùng với
              độ tin cậy ≥ {dec(g.conf_min, 2)} ở ít nhất {g.k} pha (một ảnh: ≥ {dec(g.single_conf_min, 2)}). Nếu không, hệ thống báo
              <b> "Chưa đủ cơ sở để phân loại"</b>.</p>
            <p className="small muted" style={{ margin: 0 }}>Test: nhóm chứng bị gán nhãn ung thư {CT.test.controls.cancer_label}/{CT.test.controls.n} khi bật cổng, {CT.test.controls.ungated_cancer_label}/{CT.test.controls.n} khi tắt.</p>
          </div>
        </div>

        <div style={{ marginTop: 40 }}><ModelArchitecture /></div>

        <div className="notice info" style={{ marginTop: 24 }}>
          <b>MRI:</b> mô hình phân loại 7 lớp MRI chỉ đạt accuracy {p100(MRI.roi_classifier_val.accuracy)} trên Validation (mức vùng) nên không được dùng trên website.
          Với MRI, hệ thống chỉ phát hiện và phóng to vùng nghi ngờ.
        </div>
      </div>
    </div>
  );
}
