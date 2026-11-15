export default function About() {
  return (
    <div className="section">
      <div className="container grid" style={{ gap: 30, maxWidth: 900 }}>
        <div>
          <div className="eyebrow">Giới thiệu đề tài</div>
          <h1>Ứng dụng mô hình học sâu trong phát hiện và phân loại ung thư gan từ ảnh CT/MRI</h1>
          <p className="muted">Đồ án tốt nghiệp. Website trình bày hệ thống AI đã xây dựng và chạy mô hình thật trên ảnh người dùng tải lên.</p>
        </div>

        <section>
          <h2>Mục tiêu nghiên cứu</h2>
          <ul>
            <li>Tự động phát hiện vùng tổn thương gan trên ảnh CT và MRI bằng YOLO11n.</li>
            <li>Tự động phóng to vùng tổn thương (Automatic Tumor Zoom) để phân tích chi tiết.</li>
            <li>Phân loại ung thư gan nguyên phát trên CT (HCC, ICC, CHCC) bằng mô hình lai CNN + Transformer.</li>
            <li>Đánh giá trung thực: chọn mọi tham số trên Validation, đánh giá một lần trên Test độc lập, công bố cả giới hạn.</li>
          </ul>
        </section>

        <section>
          <h2>Dữ liệu</h2>
          <div className="grid grid-2">
            <div className="card flat">
              <h3>CT – PLC-CECT</h3>
              <p className="muted small" style={{ margin: 0 }}>CT có thuốc cản quang nhiều pha, chẩn đoán HCC / ICC / CHCC xác nhận mô học hoặc tế bào học, cùng nhóm chứng
                không có bệnh gan mật. Giấy phép CC BY 4.0.</p>
            </div>
            <div className="card flat">
              <h3>MRI – LLD-MMRI</h3>
              <p className="muted small" style={{ margin: 0 }}>MRI 8 chuỗi, 7 loại tổn thương gan khu trú. Giấy phép CC BY-NC 4.0: chỉ dùng phi thương mại,
                không phân phối lại dữ liệu. Website không chứa và không công bố ảnh của bộ dữ liệu.</p>
            </div>
          </div>
          <p className="small muted" style={{ marginTop: 10 }}>Chia dữ liệu theo bệnh nhân: mọi ảnh của một bệnh nhân chỉ nằm trong một tập (huấn luyện, Validation hoặc Test).</p>
        </section>

        <section>
          <h2>Lưu ý khi sử dụng AI trong y tế</h2>
          <div className="notice warn">
            <ul style={{ marginTop: 0 }}>
              <li>Đây là <b>hệ thống nghiên cứu</b>, chưa được kiểm định lâm sàng. Kết quả <b>không thay thế chẩn đoán của bác sĩ</b>.</li>
              <li>Hệ thống <b>không bao giờ</b> kết luận bệnh nhân khỏe mạnh. "Không phát hiện vùng nghi ngờ" hay "Chưa đủ cơ sở để phân loại" không loại trừ bệnh.</li>
              <li>Mô hình có thể bỏ sót tổn thương nhỏ và có thể khoanh nhầm mô bình thường.</li>
              <li>Phân loại MRI chưa đạt chất lượng nên không được cung cấp.</li>
            </ul>
          </div>
        </section>

        <section>
          <h2>Bảo vệ dữ liệu</h2>
          <ul className="muted">
            <li>Ảnh tải lên chỉ dùng để phân tích, được xóa khỏi máy chủ ngay sau khi đọc; kết quả chỉ giữ tạm trong bộ nhớ (tối đa 15 phút).</li>
            <li>Không lưu tên tệp, không hiển thị thông tin định danh; ảnh được gọi theo pha/chuỗi (ví dụ "Pha C2", "T2WI").</li>
            <li>Không tải ảnh có thông tin định danh bệnh nhân lên bản demo công khai.</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
