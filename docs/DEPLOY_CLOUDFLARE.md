# Website LiverCancer AI trên Internet (Cloudflare Free)

## Địa chỉ

| Thành phần | Địa chỉ | Cố định? |
|---|---|---|
| Giao diện (Cloudflare Pages) | **https://livercancer-ai-demo-7u5.pages.dev** | Có |
| Máy chủ AI (Quick Tunnel tới máy này) | `https://<4-từ-ngẫu-nhiên>.trycloudflare.com` | **Không**, đổi mỗi lần bật |

Người xem chỉ cần địa chỉ Pages. Mỗi lần bật demo, script tự ghi địa chỉ tunnel mới vào `runtime-config.json` trên Pages,
nên giao diện tự nối tới máy chủ AI. Không ai phải nhập địa chỉ máy chủ.

## Kiến trúc

```
Trình duyệt ──HTTPS──► Cloudflare Pages: HTML/CSS/JS + runtime-config.json { api_base }
     │
     └──HTTPS──► Cloudflare Quick Tunnel ──► máy Windows này: FastAPI + PyTorch + GPU (127.0.0.1:8000)
                                               YOLO11n CT/MRI → Tumor Zoom → Hybrid CT + Evidence Gate
```

- **Pages chỉ chứa giao diện tĩnh, không chạy được PyTorch.** Mô hình AI chạy trên máy này. Máy phải bật, cắm sạc,
  không ngủ và có Internet trong suốt buổi demo.
- `cloudflared` mở kết nối **đi ra** từ máy tới Cloudflare. Không mở cổng router, không lộ IP, máy chủ chỉ nghe ở `127.0.0.1`.
- Mô hình, checkpoint, ngưỡng và Evidence Gate giữ nguyên như bản đã khóa (`reports/final_test/locked_config.json`, commit `2e42271`).
  Khi khởi động, máy chủ đối chiếu cấu hình đang chạy với bản khóa.

## 1. Bật website online

Nhấp đúp **`MO_WEBSITE_ONLINE.bat`** ở thư mục dự án. Script tự làm mọi bước, không cần thao tác trên Dashboard:

1. Khởi động máy chủ AI và chờ mô hình CT/MRI sẵn sàng.
2. Mở Quick Tunnel và kiểm tra tunnel truy cập được từ Internet.
3. Đưa địa chỉ tunnel lên Pages bằng Wrangler, mất khoảng 10–20 giây.
4. In **trạng thái** và địa chỉ trang Phân tích ảnh (cũng chép vào clipboard), rồi mở trình duyệt.

Tổng thời gian khoảng 1–2 phút.

- **Chạy ngầm:** đóng cửa sổ đó thì máy chủ AI, tunnel và tiến trình theo dõi vẫn chạy.
  Muốn xem lại trạng thái và địa chỉ, chạy lại `MO_WEBSITE_ONLINE.bat` (lần này không bật thêm gì).
- **Tự phục hồi:** tiến trình theo dõi kiểm tra tunnel mỗi phút.
  - Tunnel rớt (tiến trình chết, hoặc 3 lần liên tiếp không truy cập được): tự mở tunnel **mới** (địa chỉ mới) và cập nhật Pages, mất khoảng 1 phút, tối đa 6 lần mỗi lần bật.
  - Máy chủ AI dừng: website online tự tắt và gỡ địa chỉ khỏi Pages.
- **Thanh trạng thái** dưới menu ở mọi trang cho biết máy chủ AI đang hoạt động, đang khởi động hay đang tắt.
- Gửi giảng viên địa chỉ `https://livercancer-ai-demo-7u5.pages.dev/phan-tich`. **Không cần mã truy cập hay đăng nhập.**
- **Chế độ có cửa sổ theo dõi:** `MO_DEMO_ONLINE.bat` làm cùng việc nhưng giữ cửa sổ mở để xem log trực tiếp; bấm Q để tắt.
  Tham số: `-NoBrowser`, `-NoPages` (chỉ dùng địa chỉ tunnel, tunnel cũng phục vụ đầy đủ website).
- Trạng thái của lần bật hiện tại nằm ở `outputs\web_deploy\run\session.json`. Thư mục này không đưa lên Git.

## 1b. Tự khởi động khi đăng nhập Windows (không cần VS Code / Terminal)

- **Tác vụ Task Scheduler** `LiverCancerAI_TuKhoiDong`:
  - Chạy dưới tài khoản Windows hiện tại, quyền thường (không cần Administrator).
  - Kích hoạt khi đăng nhập, chờ 30 giây.
  - Chạy cả khi dùng pin; mỗi lần chỉ một bản.
- **Chuỗi chạy:** `wscript.exe scripts\web\an_cua_so.vbs tu_khoi_dong.ps1` (ẩn hoàn toàn, không nháy cửa sổ) → `tu_khoi_dong.ps1`.
  - Nếu bộ điều khiển đã chạy thì không mở thêm, nên không bao giờ có 2 bộ YOLO.
  - Ngược lại, chạy `mo_demo_online.ps1 -Headless -Auto`.
- **Trình tự khởi động:**
  1. Máy chủ AI (Python của `.venv`, đường dẫn tuyệt đối) nạp YOLO CT/MRI + Hybrid; kiểm tra `/api/health` trước khi đi tiếp.
  2. Chờ có Internet.
  3. Mở Quick Tunnel, chờ truy cập được từ Internet.
  4. `deploy_pages.ps1` ghi địa chỉ mới vào `runtime-config.json` trên Pages. Dùng đăng nhập Wrangler của người dùng; token nằm ngoài dự án, không ghi vào file nào.
- **Tự phục hồi** (theo dõi mỗi phút):
  - Máy chủ AI dừng → khởi động lại, tối đa 3 lần/giờ. Vượt mức thì tắt hẳn và ghi log.
  - Tunnel rớt hoặc mất mạng → chờ có mạng (không tính vào giới hạn) → tunnel mới + cập nhật Pages, tối đa 6 lần/giờ. Vượt mức thì chờ 15 phút rồi thử lại.
  - Cập nhật Pages lỗi → thử lại mỗi phút.
- **Chế độ ngủ:** chế độ tự khởi động **không** giữ máy thức và không đổi cài đặt nguồn. Khi máy ngủ, website báo "Máy chủ AI đang tắt"; máy thức dậy thì tunnel tự mở lại.
  Muốn giữ máy thức khi máy chủ chạy, tạo `configs\autostart.json` với `{"keep_awake": true}`. Chỉ có tác dụng khi tiến trình chạy, không đổi cài đặt Windows.
- **Quản lý:** `QUAN_LY_MAY_CHU_AI.bat` có menu: trạng thái, khởi động / khởi động lại, dừng, xem log, bật/tắt tự khởi động.
  Dùng không cần menu: `scripts\web\quan_ly_may_chu_ai.ps1 -Action trang_thai|khoi_dong_lai|dung|log|bat_tu_khoi_dong|tat_tu_khoi_dong`.
- **Log:** `outputs\web_deploy\logs\may_chu_ai_YYYY-MM-DD.log` (ghi nối, giữ 14 ngày), `server_loi_*.log` (lỗi máy chủ trước mỗi lần khởi động lại);
  `outputs\web_deploy
un\` (server.log, tunnel.log, controller.log của lần chạy hiện tại).
- **Kiểm thử 2026-11-01:**
  - Kích hoạt tác vụ: sẵn sàng sau khoảng 3 phút; tiến trình không thuộc VS Code.
  - Chạy lại tác vụ: không mở thêm.
  - Tắt đột ngột máy chủ: tự khởi động lại sau 32 giây.
  - Tắt đột ngột tunnel: tunnel mới + Pages cập nhật sau khoảng 1,5 phút.
  - E2E Production (CT/MRI 2D/3D): PASS.

## 2. Tắt website online

Nhấp đúp **`TAT_WEBSITE_ONLINE.bat`**. Script dừng máy chủ AI, tunnel và tiến trình theo dõi, rồi gỡ địa chỉ tunnel khỏi Pages.
Sau đó website vẫn mở được, thanh trạng thái báo "Máy chủ AI: đang tắt", và các trang thông tin vẫn xem được.
Script chỉ dừng đúng các tiến trình đã ghi trong `outputs\web_deploy\run\pids.json`.

**Khởi động lại:** chạy `TAT_WEBSITE_ONLINE.bat`, rồi chạy `MO_WEBSITE_ONLINE.bat`. Địa chỉ tunnel sẽ đổi (địa chỉ Pages giữ nguyên).

## 3. Cập nhật giao diện

Sau khi sửa `web/src` hoặc số liệu (`scripts/97_export_web_metrics.py`):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\web\deploy_pages.ps1 -Rebuild
```
- Lệnh này đưa giao diện lên Pages với `api_base = null`, tức là chưa nối máy chủ AI. Lần bật `MO_WEBSITE_ONLINE.bat` tiếp theo sẽ nối lại.
- Nếu website online đang chạy, cách đơn giản nhất là chạy `TAT_WEBSITE_ONLINE.bat`, rồi `deploy_pages.ps1 -Rebuild`, rồi `MO_WEBSITE_ONLINE.bat`.
  Hoặc chạy `deploy_pages.ps1 -Rebuild -ApiBase <địa chỉ tunnel hiện tại, xem trong session.json>`.

Trước khi tải lên, `deploy_pages.ps1` kiểm tra thư mục `outputs\web_deploy\pages` và **dừng** nếu gặp một trong các trường hợp:
- Loại tệp ngoài HTML/CSS/JS/SVG/JSON.
- Tệp `.pt`, `.nii`, `.gz`, `.dcm`, `.zip`, `.env` hoặc khóa bí mật.
- Tệp lớn hơn 5 MB.
- Nội dung chứa mã bệnh nhân (`P0xxx`, `MR-xxxxxx`), `localhost`, `127.0.0.1` hoặc `LCAI_ACCESS_TOKEN` / `X-Access-Token`.

Bản hiện tại có 7 tệp tĩnh, khoảng 0,3 MB.

## 4. Kiểm tra máy chủ AI

| Cần kiểm tra | Cách làm |
|---|---|
| Máy chủ trên máy | Mở `http://127.0.0.1:8000/api/health`: `models.ct` và `models.mri` là `ready`, `locked_config` là `2e42271` |
| Từ Internet | Chạy lại `MO_WEBSITE_ONLINE.bat` (in trạng thái), hoặc mở `https://<tunnel>.trycloudflare.com/api/health` |
| Pages đang trỏ đi đâu | Mở `https://livercancer-ai-demo-7u5.pages.dev/runtime-config.json` |
| Không có mã | Trang **Phân tích ảnh** mở ra là chọn CT/MRI và tải ảnh được ngay, không có ô mã truy cập |
| Kiểm thử đầy đủ trên trình duyệt | `.venv\Scripts\python scripts\98_website_e2e.py --base https://livercancer-ai-demo-7u5.pages.dev` (chỉ dùng ca Validation) |
| Nhật ký | `outputs\web_deploy\run\controller.log` (các bước, tunnel tự mở lại), `server.log`, `tunnel.log`. Không chứa tên tệp hay ảnh; ghi đè mỗi lần chạy |

## 5. Bảo mật và giới hạn

- **Không có mã truy cập, không đăng nhập** (người dùng chốt 2026-10-30): ai mở website cũng chọn CT/MRI, tải ảnh lên và nhận kết quả.
  Không có endpoint quản trị. Máy chủ được bảo vệ bằng các giới hạn sau (mặc định, đổi bằng biến môi trường trong `server/settings.py`):
  - Theo IP (`CF-Connecting-IP`, do Cloudflare đặt): 600 yêu cầu API/phút, 40 lượt tải lên/giờ, 30 lượt phân tích/giờ,
    tối đa 2 ca đang chờ/chạy và 2 lượt tải dở dang cùng lúc. Vượt giới hạn: HTTP 429 kèm thông báo tiếng Việt.
  - Toàn máy chủ: 3 ca chờ, 6 lượt tải lên dở dang; GPU xử lý từng ca một.
  - Mỗi ca chạy tối đa 10 phút, quá thì dừng và báo lỗi.
  - Tệp: tối đa 300 MB/tệp, 8 tệp (DICOM 2000 tệp, tổng 1,5 GB), có giới hạn "bom giải nén". Định dạng được kiểm tra lại trên máy chủ.
- **CORS** chỉ cho phép `https://livercancer-ai-demo-7u5.pages.dev` và các bản xem trước `https://<mã>.livercancer-ai-demo-7u5.pages.dev`.
- **Không lưu ảnh:** tệp tải lên ghi vào thư mục tạm và bị xóa ngay sau khi đọc vào bộ nhớ. Kết quả chỉ nằm trong RAM 15 phút.
  Không ghi lại tên tệp gốc. Với DICOM, máy chủ không trả thông tin bệnh nhân.
- **Không có trên Pages hay GitHub:** checkpoint `.pt`, dataset, ảnh bệnh nhân, `.env`, khóa bí mật, token Cloudflare (giao diện không chứa khóa nào).
  Token đăng nhập Wrangler nằm ở `%APPDATA%\xdg.config\.wrangler`, ngoài dự án. `tools/` (chứa cloudflared.exe) và `outputs/` có trong `.gitignore`.
- **Giới hạn 100 MB/request** của gói Free: giao diện luôn chia tệp thành từng phần 32 MB, nên ca CT 4 pha vài trăm MB vẫn tải lên được.
  Kiểm tra ngày 2026-10-30: Quick Tunnel `trycloudflare.com` cho request 101 MB đi qua, nhưng website không dựa vào điều đó;
  với tên miền riêng trên gói Free, giới hạn 100 MB được áp dụng.
- Header bảo mật trên Pages: `nosniff`, chặn nhúng iframe, `no-referrer`, `noindex`. `runtime-config.json` có `no-store`.
- **Giấy phép dữ liệu:** bộ MRI LLD-MMRI là CC BY-NC 4.0, có điều khoản không phân phối lại. Không đăng công khai ảnh hoặc ảnh chụp màn hình của bộ dữ liệu.
  Không tải ảnh có thông tin định danh bệnh nhân lên bản demo.

## 6. Điều kiện để demo chạy khi bảo vệ

1. Máy tính này bật, **cắm sạc**, có Internet. Script giữ máy không ngủ khi đang chạy, nhưng **gập nắp** vẫn có thể làm máy ngủ:
   vào *Control Panel → Power Options → Choose what closing the lid does* và chọn *Do nothing* khi cắm sạc.
2. Chạy `MO_WEBSITE_ONLINE.bat` **trước buổi bảo vệ khoảng 5–10 phút**, chờ "ĐANG HOẠT ĐỘNG", rồi thử phân tích một ca trên điện thoại.
3. Không chạy `run_website.ps1` hay F5 cùng lúc, vì cùng dùng cổng 8000.
4. Mạng ở trường phải cho phép kết nối ra Cloudflare Tunnel (cổng 7844). Nếu bước [2/3] báo lỗi, thử mạng khác, ví dụ điểm phát Wi-Fi từ điện thoại.
5. Phương án dự phòng không cần Internet: `scripts\web\run_website.ps1` rồi mở http://127.0.0.1:8000 ngay trên máy.
6. Quick Tunnel là dịch vụ thử nghiệm miễn phí, không có cam kết hoạt động liên tục. Vì vậy luôn chuẩn bị phương án dự phòng ở bước 5.

## 7. Địa chỉ máy chủ AI cố định (Named Tunnel) – chưa triển khai

Tài khoản Cloudflare dùng cho website: **`74ac12d2f4d87d2f159d80db714b73f9`** (`email tài khoản Cloudflare của chủ dự án`), project `livercancer-ai-demo`,
địa chỉ `livercancer-ai-demo-7u5.pages.dev`, branch production `main`. Account ID được ghi trong `configs/web_deploy.json`.
`deploy_pages.ps1` ép Wrangler dùng đúng tài khoản này và **dừng** nếu đang đăng nhập tài khoản khác.

Lý do có mục này: ngày 2026-10-30, Wrangler trên máy từng đăng nhập nhầm tài khoản `f8f557ed…` (một tài khoản cá nhân khác).
8 deployment đầu tiên đã vào project trùng tên trong tài khoản đó, chiếm địa chỉ `livercancer-ai-demo.pages.dev`, nên project đúng nhận hậu tố `-7u5`.
Project cũ chỉ còn giao diện, đã gỡ địa chỉ máy chủ AI; chưa xóa (xóa là quyết định của chủ tài khoản).

Kiểm tra qua API Cloudflare (chỉ đọc): cả hai tài khoản đều có **0 tên miền** và chưa có Named Tunnel.
Vì vậy hiện **không** tạo được URL backend cố định.

Quick Tunnel đủ cho demo, vì Pages tự nhận địa chỉ mới. Named Tunnel cho địa chỉ AI cố định, ví dụ `https://api.<ten-mien>`.
Điều kiện:
- **Có một tên miền riêng** đã thêm vào tài khoản Cloudflare (đổi nameserver sang Cloudflare). Cloudflare miễn phí, nhưng tên miền phải mua
  (thường vài USD đến khoảng 15 USD/năm). Không dùng được `*.pages.dev` hay `*.trycloudflare.com` cho Named Tunnel.
- Khi đã có tên miền, các bước là:
  ```powershell
  tools\cloudflared\cloudflared.exe tunnel login                        # chọn tên miền trên trình duyệt
  tools\cloudflared\cloudflared.exe tunnel create livercancer-api
  tools\cloudflared\cloudflared.exe tunnel route dns livercancer-api api.<ten-mien>
  ```
  Tạo file `config.yml` trỏ `api.<ten-mien>` tới `http://127.0.0.1:8000`, rồi đặt `pages_url` / `api_base` tương ứng.
- Named Tunnel vẫn chạy trên máy này, nên máy vẫn phải bật. Địa chỉ cố định không làm website chạy 24/7.

## 8. Muốn chạy 24/7 (chưa làm, không tự mua dịch vụ)

Pages chạy 24/7 miễn phí, nhưng phần AI cần một máy luôn bật có PyTorch. Các phương án (giá là ước lượng, cần kiểm tra giá hiện tại trước khi quyết định):

| Phương án | Chi phí | Ghi chú |
|---|---|---|
| Máy này bật liên tục + Named Tunnel | Tên miền (~vài–15 USD/năm) | Rẻ nhất; phụ thuộc điện, mạng và máy cá nhân |
| VPS CPU (4 vCPU, 8 GB RAM) | khoảng 10–25 USD/tháng | Chạy `server/` với `LCAI_DEVICE=cpu`. Ca CT 4 pha chậm hơn GPU nhiều (cần đo lại) |
| Máy chủ GPU thuê theo giờ | khoảng 0,2–0,6 USD/giờ | Chỉ bật quanh buổi bảo vệ |
| Hugging Face Spaces / Google Cloud Run | gói free có giới hạn RAM/thời gian | Cần đóng gói lại, cần kiểm tra giới hạn bộ nhớ khi tải 3 mô hình |

Mọi phương án đều phải đưa checkpoint lên máy chủ đó, nên cần bạn quyết định. Mã nguồn `server/` chạy được ở đó không cần sửa:
chỉ đổi `api_base` trong `runtime-config.json` (`deploy_pages.ps1 -ApiBase https://<máy-chủ>`) và `LCAI_ALLOWED_ORIGINS`.

## Gỡ lỗi nhanh

| Hiện tượng | Cách xử lý |
|---|---|
| Thanh trạng thái "Máy chủ AI: đang tắt" | Chạy `MO_WEBSITE_ONLINE.bat` rồi tải lại trang. Nếu đang bật mà vẫn tắt: chờ 1–2 phút (tunnel tự mở lại), hoặc xem `controller.log` |
| "Website online ĐANG CHẠY rồi" | Đã bật trước đó (chạy ngầm). Muốn bật lại: `TAT_WEBSITE_ONLINE.bat` rồi `MO_WEBSITE_ONLINE.bat` |
| "Bạn đã gửi quá nhiều yêu cầu / lượt phân tích…" | Giới hạn theo IP; chờ ít phút (lượt/giờ: tối đa 1 giờ), hoặc bật lại demo |
| "Mạng của bạn đang có ảnh khác được phân tích" | Mỗi IP tối đa 2 ca cùng lúc; chờ ca trước xong |
| `Cổng 8000 đang được dùng` | Tắt website local (run_website.ps1 / F5), rồi chạy lại |
| Bước [3/3] lỗi Wrangler đăng nhập | `cd web; npx wrangler login`, đăng nhập bằng tài khoản Cloudflare của bạn trên trình duyệt. Không dán token vào đâu cả |
| Bước [2/3] lỗi tunnel | Mạng chặn Cloudflare Tunnel: đổi mạng. Xem `outputs\web_deploy\run\tunnel.log` |
| "Máy chủ đang bận" | GPU đang phân tích ca khác (tối đa 3 ca chờ). Thử lại sau ít phút |
