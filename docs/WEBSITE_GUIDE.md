# Website LiverCancer AI – hướng dẫn chạy và kiểm thử (Windows 11)

Website chính thức gồm hai phần tách biệt:

| Phần | Thư mục | Công nghệ | Vai trò |
|---|---|---|---|
| Giao diện | `web/` | React 18 + Vite + TypeScript | 5 trang tiếng Việt, light mode, responsive; build ra `web/dist` (tĩnh, đưa được lên Cloudflare Pages) |
| Máy chủ AI | `server/` | FastAPI + pipeline có sẵn trong `src/livercancer` | nhận ảnh, chạy YOLO11n → Tumor Zoom → Hybrid + Evidence Gate (CT) hoặc YOLO11n → Tumor Zoom (MRI) |

Thuật toán AI **không** được viết lại. Máy chủ gọi đúng `LiverCancerPipeline` và các checkpoint đã khóa trước khi chạy Test.
Khi khởi động, máy chủ so cấu hình đang chạy (ngưỡng YOLO, cách gộp, Evidence Gate) với `reports/final_test/locked_config.json`.
Nếu khác, CT bị tắt. Website cũ (Streamlit, `app/`) vẫn còn nhưng đã được thay thế.

## 1. Chạy website trên máy

Cách nhanh nhất (PowerShell, trong thư mục dự án):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\web\run_website.ps1
```

- Mở **http://127.0.0.1:8000**. Script tự build giao diện nếu chưa có và tự mở trình duyệt.
- Mô hình cần khoảng 20–60 giây để khởi động. Trong lúc đó, trang Phân tích hiện "Máy chủ đang khởi động mô hình AI…".
- Muốn build lại giao diện sau khi sửa `web/src`, chạy lại script với tham số `-Rebuild`.
- Trong VS Code: `Ctrl+Shift+D` → **🌐 Website LiverCancer AI chính thức (http://127.0.0.1:8000)** → `F5`.
  Cấu hình F5 dùng `web/dist` đã build; nếu chưa có, chạy `npm run build` trong `web/` trước.

Chế độ phát triển giao diện (tự tải lại khi sửa code), dùng hai terminal:

```powershell
# terminal 1: máy chủ AI
$env:YOLO_OFFLINE="1"; .venv\Scripts\python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
# terminal 2: giao diện tại http://localhost:5173 (tự chuyển /api sang cổng 8000)
cd web; npm run dev
```

## 2. Dùng website

Trang **Phân tích ảnh** (giao diện v5, 2026-10-30) có ba cột: cột trái là các bước thao tác, cột giữa là trình xem ảnh, cột phải là kết quả.
Màn hình rộng từ 1200 px hiện đủ ba cột. Máy tính bảng hiện hai cột, kết quả nằm dưới trình xem ảnh. Điện thoại hiện một cột.

**Cột trái: các bước thao tác**

1. **Chọn loại phân tích:**
   - **Phân tích ảnh 2D:** một lát cắt PNG/JPG, ảnh chụp màn hình hoặc một tệp DICOM.
   - **Phân tích ca 3D:** NIfTI `.nii` / `.nii.gz`, nhiều tệp DICOM, thư mục DICOM hoặc DICOM nén `.zip`.

   Sau đó chọn **CT Scan** hoặc **MRI**. Chưa chọn CT/MRI thì không tải ảnh lên được.
   Máy chủ kiểm tra lại lựa chọn 2D/3D: ví dụ PNG ở chế độ 3D, hoặc NIfTI ở chế độ 2D, sẽ bị từ chối kèm hướng dẫn.
2. **Tải ảnh lên:** kéo thả ảnh hoặc bấm *Chọn ảnh / Chọn tệp*. Ở chế độ 3D, CT nhận tối đa 4 pha của một bệnh nhân, MRI nhận nhiều chuỗi.
3. **Trạng thái xử lý:** tự chạy, không cần bấm thêm nút. Các bước hiện lần lượt:
   tải lên → đọc và kiểm tra tệp → kiểm tra CT/MRI → *(3D)* tách volume thành lát cắt 2D → YOLO → Tumor Zoom → tổng hợp.
   - Chỉ hai bước tải lên và YOLO có phần trăm, vì đó là tiến độ đo thật.
   - Nút **Hủy phân tích** dừng tác vụ trên máy chủ, kể cả khi đang chạy trên GPU.

**Cột giữa: trình xem ảnh** (`web/src/components/Viewer.tsx`)

- Ảnh lát cắt hiển thị trên nền đen. Box YOLO vẽ đúng trên lát của nó: đỏ là vùng đang chọn, cam là vùng khác.
- Thao tác:
  - Phóng to / thu nhỏ: nút, phím +/−, hoặc Ctrl + cuộn chuột.
  - Kéo ảnh khi đã phóng to.
  - Toàn màn hình: nút hoặc phím F.
- Với ca 3D có thêm:
  - Nhãn "Lát 78 / 132".
  - Nút lát trước/sau và phím ← →.
  - Thanh trượt có vạch đỏ đánh dấu các lát có vùng nghi ngờ.
  - Chọn pha/chuỗi.
  - **Dải thumbnail**: mọi lát có vùng nghi ngờ, kèm độ tin cậy phát hiện. Bấm vào thumbnail để mở đúng lát đó.
- Chỉ có mặt cắt **axial**. Chưa làm coronal/sagittal, vì box 2D không được chuyển sang mặt cắt khác khi chưa có phép chuyển tọa độ đã kiểm chứng.
- **Zoom vùng nghi ngờ:** *Ảnh gốc (vùng cắt)* có khung tổn thương, đặt cạnh *Tumor Zoom*. Có nút chọn từng vùng; bấm vào ảnh để xem lớn.
  Ngay dưới hai ảnh (ca 3D) là 4 ô **Vùng đang chọn**: vùng (và pha/chuỗi), phạm vi lát, độ tin cậy cao nhất, khối bao 3D (không phải kích thước khối u). Trước 2026-11-28 các ô này nằm ở cuối cột phải.
- **Tumor Zoom trên màn hình** (`src/livercancer/roi/display.py`):
  - Cắt rộng quanh tổn thương: cận cảnh 2,2 lần cạnh box (tối thiểu 72 px), xung quanh 3 lần (tối thiểu 192 px).
  - Phóng tối đa 4 lần bằng nội suy bicubic. Không làm nét, không tạo chi tiết.
  - Ảnh đưa vào Hybrid vẫn là vùng cắt 224×224 đã khóa.

**Cột phải: kết quả** (`web/src/components/ResultPanel.tsx`)

- **Kết luận AI**, theo trạng thái:
  - **CT đủ bằng chứng:** "Nghi ngờ HCC/ICC/CHCC", kèm tên đầy đủ.
  - **CT không đủ bằng chứng:** "Chưa đủ cơ sở để phân loại" (Evidence Gate từ chối).
  - **Không có vùng:** "Không phát hiện vùng nghi ngờ trong ảnh này / ca chụp này". Không đồng nghĩa với gan bình thường, vì mô hình không có lớp "bình thường".
  - **MRI:** "Phát hiện N vùng nghi ngờ", không phân loại.
- **Xác suất phân loại (Hybrid CT):** chỉ hiện khi backend thực sự phân loại. Đây là softmax của mô hình, không phải xác suất mắc bệnh.
  Khi không phân loại, mục này hiện độ tin cậy phát hiện (YOLO) của từng vùng.
- **Thông tin chi tiết:**
  - Loại ảnh và cách xác định; dữ liệu 2D hay 3D (NIfTI/DICOM, số pha/chuỗi).
  - Kích thước và khoảng cách điểm ảnh (mm).
  - Tổng số lát; số lát YOLO đã quét; lát trùng hoàn toàn (nếu có); số lát có vùng nghi ngờ; lát đang xem.
  - Số vùng nghi ngờ; độ tin cậy phát hiện cao nhất và lát của nó.
  - Evidence Gate (CT); số tệp DICOM trùng đã bỏ; thời gian xử lý.
- **Nút tải:**
  - *Tải ảnh lát đang xem (PNG)*: đúng lát và đúng các khung đang hiển thị.
  - *Tải báo cáo (HTML)*: gồm ảnh gốc, ảnh khoanh vùng, vùng cắt và Tumor Zoom, thông tin ca, lát nghi ngờ, nhận xét, giới hạn.
  - *In / Lưu PDF*: in chính báo cáo đó bằng trình duyệt, chọn "Lưu dưới dạng PDF".
- Trước khi phân tích, cột giữa và cột phải chỉ hiện trạng thái chờ. Không có ảnh hay con số minh họa nào.
- **Phân biệt các con số:**
  - *Độ tin cậy phát hiện* (YOLO, 0–1, theo từng khung / vùng).
  - *Điểm dự đoán của mô hình* (Hybrid CT, softmax, toàn ca; không phải xác suất mắc ung thư).
  - *Evidence Gate* (đủ / chưa đủ bằng chứng).
  - *Lát có vùng nghi ngờ* (số lát có ít nhất một khung).
  - *Vùng nghi ngờ* (khung nối qua các lát liên tiếp).
- **Vùng yếu:** với CT, vùng có độ tin cậy dưới mức "vùng mạnh" của quy tắc Evidence Gate đang áp dụng (0,4 khi có nhiều pha, 0,35 khi chỉ có một file)
  được ghi "thấp" (nút viền đứt). Ngưỡng phát hiện 0,12 không đổi.
- **Kiểm toán số vùng** (`scripts/74_case3d_audit.py` → `reports/case3d_audit.md`, CT Validation):
  - Trung bình 6,5 vùng mỗi volume; 26% vùng nằm hoàn toàn ngoài gan.
  - 75/156 tổn thương được phát hiện bị tách thành nhiều vùng; không có vùng nào gộp nhầm hai tổn thương.
  - Vì vậy số vùng nghi ngờ **không phải** số khối u.
  - Ca 92 lát P0002 C1 hay dùng để demo thuộc tập **Train**; nên demo bằng ca Validation.
- **Đồng bộ viewer:**
  - Bấm thumbnail: mở đúng lát và tự chọn vùng có trên lát đó (nếu là một trong 8 vùng có Tumor Zoom). Khung của các vùng còn lại ghi "Vùng khác".
  - Bấm chọn vùng: mở lát đại diện của vùng.
  - Chỉ phản hồi của yêu cầu lát mới nhất được hiển thị.

**Đầu ra 3D (v6, 2026-10-31)**
- **Video quét lát** (tab *Quét lát cắt*):
  - Có Phát / Tạm dừng / Phát lại, lát trước/sau, tốc độ 0,5x–4x (1x = 8 lát/giây), bật/tắt khung, toàn màn hình.
  - Ảnh lấy từ `GET /api/jobs/{id}/frames`: dải JPEG 8 lát mỗi lần, trình duyệt lưu đệm tối đa 24 dải. YOLO **không** chạy lại.
  - Khung của lát đang hiện vẽ từ dữ liệu vùng 3D; khi tạm dừng, viewer tải ảnh PNG gốc của lát kèm khung từ máy chủ, và hai nguồn khung trùng nhau (có test).
  - Dòng dưới ảnh liệt kê các phát hiện của lát đang xem, kèm số vùng 3D.
- **Tải video phân tích 3D:**
  - MP4 H.264 tạo trên máy chủ (`server/video.py`, ffmpeg của `imageio-ffmpeg`), cho pha/chuỗi đang xem.
  - Mỗi lát một frame (8 fps): ảnh thật, khung YOLO thật kèm số vùng và độ tin cậy phát hiện, số lát.
  - Không ghi điểm HCC lên khung. Có tiến độ và nút hủy; tối đa 6 video/giờ mỗi IP; file tạm xóa sau 15 phút.
- **Vùng nghi ngờ 3D** (`result.case.regions3d`):
  - Mỗi vùng = các khung YOLO trên những lát liên tiếp, nối theo quy tắc đã khóa (`configs/roi.yaml`), mỗi pha/chuỗi riêng.
  - Thông tin: số vùng, lát đầu/cuối, lát đại diện, số lát có khung, khung trên từng lát, độ tin cậy cao nhất, khối bao (x, y, z min/max).
  - Kích thước mm chỉ khi khoảng cách điểm ảnh đã biết và khoảng cách lát đều.
  - Khối bao **không** phải thể tích khối u. Một khối u có thể bị tách thành nhiều vùng (`reports/case3d_audit.md`).
- **Tab *Vùng nghi ngờ 3D*** (Three.js, chỉ tải khi mở tab):
  - Khung volume, trục X/Y/Z, mặt phẳng lát đang xem, khối bao bán trong suốt; xoay, phóng to, bấm chọn vùng.
  - Bảng vùng bên dưới. Chọn vùng thì video nhảy về lát đại diện.
  - Không dựng hình gan, vì chưa có mô hình phân đoạn gan.
- **Đánh số vùng thống nhất** (theo độ tin cậy giảm dần) ở viewer, thumbnail (V1, V2…), tab 3D, video và báo cáo HTML/PDF.

**Pipeline 3D** (`server/service.py`, `src/livercancer/intake/`). Đây **không phải mô hình 3D**: volume được tách thành lát 2D rồi chạy YOLO 2D đã train.

1. **Đọc volume.**
   - NIfTI dùng `load_volume_uint8`, giống lúc train; spacing lấy từ header.
   - DICOM (`dicom_io`):
     - Đọc từng tệp; `.zip` đọc từng phần tử trong RAM, không giải nén ra đĩa, nên tên phần tử như `../` không bao giờ thành đường dẫn.
     - Mỗi SeriesInstanceUID là một volume riêng, không gộp series.
     - Thứ tự lát theo ImagePositionPatient, sau đó InstanceNumber.
     - CT: cửa sổ [−200, 200] HU, 1 mm, 512² như dữ liệu train.
2. **Tách lát cắt axial** (`slice_image`, giống lúc train). Chỉ số lát được giữ nguyên để ánh xạ về volume gốc.
3. **Lát trùng:**
   - **DICOM:** tệp trùng hoàn toàn (cùng SOPInstanceUID, hoặc cùng vị trí và cùng dữ liệu điểm ảnh) bị bỏ và được đếm.
     Hai ảnh *khác nhau* ở cùng vị trí thì bị từ chối.
   - **Trong volume** (`intake/dedup.py`): chỉ lát giống nhau **từng byte** (hash) mới là trùng. YOLO chạy một lần rồi gán lại kết quả;
     không lát nào bị bỏ. Lát chỉ *giống nhau* không bao giờ bị gộp. Tắt bằng `LCAI_DEDUP_EXACT=0`.
   - Khảo sát Validation (`scripts/86_slice_duplicates.py` → `reports/slice_duplicates.md`):
     - 58.559 lát (211 volume CT, 624 volume MRI), **0 lát trùng**. Vì vậy với dữ liệu thật, website chạy y hệt pipeline đã khóa.
     - Kiểm tra cơ chế trên volume có chèn bản sao: vùng 3D trùng khớp. CT lệch ở mức làm tròn fp16 (box ≤ 0,375 px, độ tin cậy ≤ 0,002).
4. **YOLO 2D** của CT hoặc MRI quét **mọi** lát (ngưỡng 0,12 đã khóa) → nối box thành vùng 3D → Tumor Zoom →
   CT: Hybrid + Evidence Gate (đã khóa). MRI: dừng ở Tumor Zoom.
5. **Tổng hợp ca:**
   - Số lát, lát trùng, lát nghi ngờ, lát có độ tin cậy phát hiện cao nhất.
   - Độ tin cậy phát hiện **không** được trình bày là xác suất ung thư.
   - Nhãn cả ca chỉ có với CT, theo cách gộp đã chọn trên Validation và đánh giá trên Test.
6. **Giới hạn hạ tầng:**
   - Tải lên chia phần 32 MB (Cloudflare ≤ 100 MB/request).
   - Tối đa 300 MB/tệp, 8 tệp (DICOM 2000 tệp, .zip giải nén ≤ 800 MB, tổng 1,5 GB).
   - Mỗi ca tối đa 10 phút; GPU chạy từng ca một; hàng đợi 3.
   - Website không hỗ trợ tệp 2 GB.

Ảnh chụp màn hình được xử lý như sau (`src/livercancer/intake/image2d.py`):
- Tìm lát cắt là vùng cơ thể nằm trên nền tối.
- Bỏ giao diện phần mềm và **che chữ nằm ngoài vùng cơ thể**, vì có thể là tên hoặc mã bệnh nhân.
- Căn khung giống ảnh huấn luyện và đưa về 512×512, không chỉnh độ sáng hay độ tương phản.
- Báo ảnh mờ, chói, quá nhỏ hoặc bị mất góc.
- Nếu không tách được lát cắt, website từ chối và hướng dẫn người dùng, thay vì đưa nguyên ảnh vào mô hình.

Ảnh tải về và báo cáo chỉ dùng ảnh đã tiền xử lý. Số liệu kiểm chứng ở `reports/modality_recognition.md`.

## 3. Bảo mật và dữ liệu

- File tải lên được ghi vào một thư mục tạm riêng, tên file gốc **không** được ghi lại. File bị **xóa ngay** sau khi đọc vào bộ nhớ.
- Kết quả chỉ nằm trong RAM và bị xóa khi người dùng bấm "Phân tích ảnh khác", hoặc sau 15 phút.
- Máy chủ không ghi log tên tệp hay thông tin bệnh nhân. Ảnh được gọi theo pha/chuỗi ("Pha C2", "T2WI", "Chuỗi 1"), không theo tên tệp.
- Với DICOM, máy chủ chỉ đọc thẻ loại ảnh, hình học và dữ liệu điểm ảnh. Tên, mã bệnh nhân và các thẻ khác **không bao giờ** được trả về giao diện (có test kiểm tra).
- Máy chủ kiểm tra lại mọi tệp, không tin dữ liệu trình duyệt gửi:
  - Nội dung tệp phải khớp đuôi tệp.
  - Header NIfTI phải đọc được.
  - Kích thước tối đa 300 MB/tệp, 8 tệp (DICOM: tối đa 2000 tệp, tổng 1,5 GB; .zip có giới hạn giải nén).
  - Có giới hạn chống "bom giải nén".
- Hàng đợi chỉ cho 3 phân tích chờ; GPU chạy từng ca một.
- Có header bảo mật (`nosniff`, chặn nhúng iframe, `no-referrer`). Kết quả API không được cache.
- **Không có mã truy cập hay đăng nhập** (người dùng chốt 2026-10-30). API phân tích công khai, được bảo vệ bằng giới hạn theo IP:
  600 yêu cầu/phút, 40 lượt tải lên/giờ, 30 lượt phân tích/giờ, tối đa 2 ca chạy cùng lúc; mỗi ca tối đa 10 phút. Không có endpoint quản trị.
- Không có checkpoint, khóa bí mật hay ảnh bệnh nhân nào trong `web/`. Thư mục `web/dist` chỉ chứa HTML/CSS/JS và số liệu tổng hợp.

- Tối đa 6 lượt tải lên dở dang cùng lúc (2 cho mỗi IP).

Biến môi trường của máy chủ: xem đầu file `server/settings.py`. Gồm `LCAI_DEVICE`, `LCAI_ALLOWED_ORIGINS`, `LCAI_ALLOWED_ORIGIN_REGEX`,
`LCAI_MAX_FILE_MB`, `LCAI_MAX_FILES`, `LCAI_MAX_QUEUE`, `LCAI_MAX_OPEN_UPLOADS`, `LCAI_RATE_PER_MIN`, `LCAI_UPLOADS_PER_HOUR`,
`LCAI_JOBS_PER_HOUR`, `LCAI_ACTIVE_PER_CLIENT`, `LCAI_JOB_TIMEOUT_S`, `LCAI_DEDUP_EXACT`, `LCAI_RESULT_TTL_S`, `LCAI_FRONTEND_DIST` và `LCAI_TMP_DIR`.

## 4. Kiểm thử

```powershell
$env:YOLO_OFFLINE="1"
.venv\Scripts\python -m pytest tests\test_web_api.py -q          # API với checkpoint thật, ca Validation
# website đang chạy ở cổng 8000, rồi kiểm thử trên trình duyệt Edge thật (laptop, điện thoại, máy chiếu 1920×1080):
.venv\Scripts\python scripts\98_website_e2e.py                    # ảnh chụp: outputs\web_e2e\
```

Mọi ca kiểm thử chỉ dùng bệnh nhân **Validation**. Test đã được đánh giá một lần và không chạy lại.

| Ca kiểm thử | Kỳ vọng |
|---|---|
| CT có tổn thương (4 pha) | phân loại; nhãn và điểm **trùng** kết quả pipeline Validation đã lưu |
| CT bị Evidence Gate từ chối | "Chưa đủ cơ sở để phân loại", không có nhãn |
| CT nhóm chứng, YOLO không có vùng | "Không phát hiện vùng nghi ngờ trong ca chụp này", không kết luận gan bình thường |
| MRI (3 chuỗi) | khoanh vùng, không phân loại, có cảnh báo MRI |
| Lát cắt PNG (chế độ 2D) | phân tích theo CT/MRI đã chọn |
| Ca 3D (NIfTI, DICOM .zip, .zip có tệp trùng) | tổng hợp ca khớp từng lát trong trình xem; tệp DICOM trùng được đếm và bỏ |
| Chọn sai 2D/3D | từ chối kèm hướng dẫn |
| Từ chối | sai định dạng, sai loại ảnh đã chọn, tải lên không liên tục, vượt giới hạn theo IP (429), quá 10 phút |

## 5. Số liệu trang "Kết quả nghiên cứu"

Trang này đọc `web/src/data/research.json`. File được sinh tự động từ các file kết quả thật, chỉ chứa số tổng hợp, không có mã bệnh nhân:

```powershell
.venv\Scripts\python scripts\97_export_web_metrics.py
cd web; npm run build
```

Đưa website lên Internet: nhấp đúp `MO_WEBSITE_ONLINE.bat` (tắt: `TAT_WEBSITE_ONLINE.bat`) (giao diện https://livercancer-ai-demo.pages.dev + máy chủ AI qua Cloudflare Tunnel).
Chi tiết: `docs/DEPLOY_CLOUDFLARE.md`.
