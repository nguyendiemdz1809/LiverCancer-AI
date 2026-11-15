# Bật website LiverCancer AI online. Xem docs/DEPLOY_CLOUDFLARE.md.
#   MO_WEBSITE_ONLINE.bat   -> chạy file này ở chế độ ngầm (-Headless): cửa sổ đóng được, website vẫn chạy.
#   MO_DEMO_ONLINE.bat      -> chạy file này trong cửa sổ có theo dõi; bấm Q để tắt.
#   Tự khởi động (Task Scheduler, khi đăng nhập Windows) -> scripts\web\tu_khoi_dong.ps1 -> file này với -Headless -Auto.
# Các bước:
#   1. Máy chủ AI (FastAPI + PyTorch) trên máy này, chỉ nghe ở 127.0.0.1. Phân tích ảnh CÔNG KHAI (không mã, không đăng nhập),
#      được bảo vệ bằng giới hạn theo IP, hàng đợi, kích thước tệp và thời gian xử lý (server/settings.py).
#   2. Cloudflare Quick Tunnel -> https://<ngẫu-nhiên>.trycloudflare.com (TẠM THỜI, đổi sau mỗi lần bật / mỗi lần tunnel khởi động lại).
#   3. Ghi địa chỉ tunnel vào runtime-config.json và đưa lại giao diện lên Cloudflare Pages: https://livercancer-ai-demo.pages.dev
#      tự nối tới máy chủ AI, không ai phải nhập địa chỉ máy chủ.
#   4. Theo dõi mỗi phút (tự phục hồi, có giới hạn):
#      - máy chủ AI dừng -> khởi động lại (tối đa 3 lần / giờ);
#      - mất Internet -> chờ có mạng lại (không tính vào giới hạn) rồi mở tunnel mới;
#      - tunnel rớt -> mở tunnel mới + cập nhật Pages (tối đa 6 lần / giờ; vượt thì chờ rồi thử lại, không tắt hẳn);
#      - cập nhật Pages lỗi -> thử lại mỗi phút.
# Tham số: -NoPages  -NoBrowser  -Headless  -Auto (chờ mạng khi khởi động, không giữ máy thức)  -KeepAwake / -NoKeepAwake
param([switch]$NoPages, [switch]$NoBrowser, [switch]$Headless, [switch]$Auto, [switch]$KeepAwake, [switch]$NoKeepAwake)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$cfg = Get-Content (Join-Path $root "configs\web_deploy.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$port = [int]$cfg.port
$run = Join-Path $root "outputs\web_deploy\run"
$logDir = Join-Path $root "outputs\web_deploy\logs"
New-Item -ItemType Directory -Force $run, $logDir | Out-Null
$pidFile = Join-Path $run "pids.json"
$sessFile = Join-Path $run "session.json"
$stopFlag = Join-Path $run "stop.flag"
$py = Join-Path $root ".venv\Scripts\python.exe"
$cf = Join-Path $root $cfg.cloudflared
$tat = Join-Path $PSScriptRoot "tat_demo_online.ps1"
# giữ máy thức: mặc định CÓ khi bật tay (như trước), KHÔNG khi tự khởi động (không đổi chế độ ngủ khi chưa được phép)
$keep = if ($KeepAwake) { $true } elseif ($NoKeepAwake -or $Auto) { $false } else { $true }

# nhật ký lâu dài (thêm vào cuối, mỗi ngày một tệp, giữ 14 ngày); controller.log vẫn là đầu ra của lần chạy hiện tại
Get-ChildItem $logDir -Filter "may_chu_ai_*.log" -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-14) } | Remove-Item -ErrorAction SilentlyContinue
function Say([string]$m, [string]$c = "Gray") {
    $line = "{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $m
    Write-Host ("{0:HH:mm:ss} {1}" -f (Get-Date), $m) -ForegroundColor $c
    try { [IO.File]::AppendAllText((Join-Path $logDir ("may_chu_ai_{0:yyyy-MM-dd}.log" -f (Get-Date))), "$line`r`n", (New-Object Text.UTF8Encoding $false)) } catch { }
}
function Health([string]$base, [int]$timeout = 10) {
    try { return Invoke-RestMethod "$base/api/health" -TimeoutSec $timeout -Headers @{ "Cache-Control" = "no-cache" } } catch { return $null }
}
function Test-Internet {
    # Cloudflare trả lời trang trace khi có Internet; không gửi dữ liệu gì của dự án
    try { $null = Invoke-WebRequest "https://www.cloudflare.com/cdn-cgi/trace" -UseBasicParsing -TimeoutSec 8; return $true } catch { return $false }
}
$session = [ordered]@{ state = "starting"; stage = ""; message = ""; site = ""; tunnel = ""; started = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"); device = "";
                       tunnel_restarts = 0; server_restarts = 0; mode = $(if ($Auto) { "tu_khoi_dong" } else { "bat_tay" }); keep_awake = $keep }
function Save-Session { [IO.File]::WriteAllText($sessFile, ($session | ConvertTo-Json), (New-Object Text.UTF8Encoding $false)) }
function Save-Pids { [ordered]@{ controller = $PID; server = $(if ($server) { $server.Id }); tunnel = $(if ($tunnel) { $tunnel.Id }) } | ConvertTo-Json | Set-Content $pidFile -Encoding UTF8 }
function Stage([string]$m) { $session.stage = $m; Save-Session; Say $m Cyan }
function Fail([string]$m) { $session.state = "error"; $session.message = $m; Save-Session; Say "LỖI: $m" Red; throw $m }
function Kill-Pid([int]$id) {
    # taskkill writes to stderr when the process is already gone: never fatal here (Windows PowerShell 5.1)
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & taskkill.exe /PID $id /T /F 2>&1 | Out-Null } catch { } finally { $ErrorActionPreference = $old }
}
# giới hạn khởi động lại theo cửa sổ 1 giờ
$restarts = @{ server = New-Object System.Collections.ArrayList; tunnel = New-Object System.Collections.ArrayList }
function Allow-Restart([string]$kind, [int]$max) {
    $now = Get-Date
    $l = $restarts[$kind]
    for ($i = $l.Count - 1; $i -ge 0; $i--) { if (($now - $l[$i]).TotalMinutes -gt 60) { $l.RemoveAt($i) } }
    if ($l.Count -ge $max) { return $false }
    [void]$l.Add($now)
    return $true
}

function Start-Server {
    $env:YOLO_OFFLINE = "1"; $env:PYTHONIOENCODING = "utf-8"
    $env:LCAI_ALLOWED_ORIGINS = $cfg.pages_url
    $env:LCAI_ALLOWED_ORIGIN_REGEX = $cfg.pages_preview_origin_regex
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        throw "Cổng $port đang được dùng (có thể website local đang chạy bằng run_website.ps1 hoặc F5 trong VS Code). Hãy tắt nó rồi chạy lại."
    }
    $p = Start-Process $py -ArgumentList "-m", "uvicorn", "server.main:app", "--host", "127.0.0.1", "--port", "$port", "--no-server-header" `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $run "server.out.log") -RedirectStandardError (Join-Path $run "server.log")
    $t0 = Get-Date
    while ($true) {
        if ($p.HasExited) { throw "Máy chủ AI đã dừng khi khởi động (xem outputs\web_deploy\run\server.log)." }
        $h = Health "http://127.0.0.1:$port" 5
        if ($h -and -not ($h.models.PSObject.Properties.Value -contains "loading")) { return @{ Proc = $p; Health = $h } }
        if (((Get-Date) - $t0).TotalSeconds -gt 300) { Kill-Pid $p.Id; throw "Mô hình AI chưa sẵn sàng sau 5 phút." }
        Start-Sleep -Seconds 3
    }
}

function Start-QuickTunnel {
    $tlog = Join-Path $run "tunnel.log"
    Remove-Item $tlog -ErrorAction SilentlyContinue
    $p = Start-Process $cf -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:$port" -WindowStyle Hidden -PassThru `
        -RedirectStandardError $tlog -RedirectStandardOutput (Join-Path $run "tunnel.out.log")
    $url = ""; $t0 = Get-Date
    while (-not $url) {
        if ($p.HasExited) { throw "cloudflared đã dừng (mạng có thể chặn Cloudflare Tunnel, xem outputs\web_deploy\run\tunnel.log)." }
        if (((Get-Date) - $t0).TotalSeconds -gt 90) { Kill-Pid $p.Id; throw "Không nhận được địa chỉ Quick Tunnel sau 90 giây." }
        Start-Sleep -Seconds 1
        $m = Select-String -Path $tlog -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($m) { $url = $m.Matches[0].Value }
    }
    $t0 = Get-Date
    while (-not (Health $url 10)) {
        if (((Get-Date) - $t0).TotalSeconds -gt 120) { Kill-Pid $p.Id; throw "Tunnel đã có địa chỉ nhưng chưa truy cập được từ Internet sau 2 phút." }
        Start-Sleep -Seconds 3
    }
    return @{ Proc = $p; Url = $url }
}

function Publish-Api([string]$url) {
    if ($NoPages) { return $false }
    try {
        $d = & (Join-Path $PSScriptRoot "deploy_pages.ps1") -ApiBase $url -Quiet
        if ($d.Live) { Say "Giao diện $($cfg.pages_url) đã nối với máy chủ AI ($($d.DeploymentId))." Green } else { Say "Đã tải lên Pages; Cloudflare có thể cần thêm ít giây." Yellow }
        return $true
    } catch {
        Say "Không cập nhật được Pages: $($_.Exception.Message). Sẽ thử lại sau 1 phút." Yellow
        return $false
    }
}

function Wait-Internet([string]$why) {
    # chờ có mạng (không giới hạn thời gian; không tính vào giới hạn khởi động lại). Trả về $false nếu nhận lệnh tắt.
    $n = 0
    while (-not (Test-Internet)) {
        if (Test-Path $stopFlag) { return $false }
        if ($n % 5 -eq 0) { Say "Chưa có Internet ($why). Đang chờ có mạng..." Yellow }
        $session.state = "offline"; $session.message = "Chưa có Internet"; Save-Session
        $n++
        Start-Sleep -Seconds 60
    }
    if ($n) { Say "Đã có Internet trở lại." Green }
    return $true
}

Say "LiverCancer AI - BẬT WEBSITE ONLINE (Cloudflare Pages + Quick Tunnel) · chế độ: $($session.mode)" Cyan

# ---- 0. đang chạy rồi? (không bao giờ chạy 2 bộ YOLO cùng lúc) / dọn tiến trình sót lại
if (Test-Path $pidFile) {
    $old = Get-Content $pidFile -Raw | ConvertFrom-Json
    $c = if ($old.controller) { Get-Process -Id $old.controller -ErrorAction SilentlyContinue }
    if ($c -and $c.ProcessName -eq "powershell" -and $c.Id -ne $PID) {
        Say "Website online ĐANG CHẠY rồi (tiến trình $($c.Id)). Không mở thêm. Xem trạng thái: QUAN_LY_MAY_CHU_AI.bat." Yellow
        exit 2
    }
    & $tat -Quiet -FromController
}
Remove-Item $stopFlag -ErrorAction SilentlyContinue
Save-Session

$server = $null; $tunnel = $null; $tunnelUrl = ""; $pagesSet = $false; $pagesPending = ""
try {
    # ---- 1. kiểm tra
    if (-not (Test-Path $py)) { Fail "Không thấy .venv\Scripts\python.exe. Môi trường Python của dự án chưa được cài." }
    if (-not (Test-Path $cf) -or (Get-FileHash $cf -Algorithm SHA256).Hash.ToLower() -ne $cfg.cloudflared_sha256) {
        if (-not (Wait-Internet "tải cloudflared")) { throw "Nhận lệnh tắt." }
        Stage "Tải cloudflared $($cfg.cloudflared_version) (bản chính thức của Cloudflare, kiểm tra SHA256)..."
        New-Item -ItemType Directory -Force (Split-Path $cf) | Out-Null
        $ProgressPreference = "SilentlyContinue"
        Invoke-WebRequest $cfg.cloudflared_url -OutFile "$cf.download" -UseBasicParsing
        if ((Get-FileHash "$cf.download" -Algorithm SHA256).Hash.ToLower() -ne $cfg.cloudflared_sha256) { Remove-Item "$cf.download"; Fail "cloudflared tải về sai SHA256, đã xóa." }
        Move-Item -Force "$cf.download" $cf
    }
    if (-not (Test-Path "web\dist\index.html")) { Fail "Chưa có web\dist. Chạy: cd web; npm run build" }

    if ($keep) {
        # Giữ máy không ngủ khi website online đang chạy (màn hình vẫn có thể tắt). Gập nắp máy vẫn có thể làm máy ngủ.
        Add-Type -Namespace LCAI -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
        [void][LCAI.Power]::SetThreadExecutionState([uint32]2147483649)      # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        Say "Giữ máy không ngủ trong lúc chạy (chỉ khi tiến trình này còn chạy; không đổi cài đặt nguồn của Windows)." DarkGray
    }

    # ---- 2. máy chủ AI (không cần Internet)
    Stage "[1/3] Khởi động máy chủ AI (YOLO CT/MRI, Tumor Zoom, Hybrid CT + Evidence Gate)..."
    try { $s = Start-Server } catch { Fail $_.Exception.Message }
    $server = $s.Proc; Save-Pids
    $h = $s.Health
    $session.device = $h.device
    Say ("Mô hình: CT = {0}, MRI = {1}; thiết bị: {2}; cấu hình khóa: {3}" -f $h.models.ct, $h.models.mri, $h.device, $h.locked_config) Green
    if ($h.models.ct -ne "ready") { Say "CẢNH BÁO: mô hình CT không sẵn sàng ($($h.models.ct))." Yellow }

    # ---- 3. Quick Tunnel (chờ có Internet; thử lại khi lỗi)
    Stage "[2/3] Mở Cloudflare Quick Tunnel..."
    while (-not $tunnel) {
        if (-not (Wait-Internet "mở tunnel")) { throw "Nhận lệnh tắt." }
        try { $t = Start-QuickTunnel; $tunnel = $t.Proc; $tunnelUrl = $t.Url }
        catch {
            if (-not $Auto) { Fail $_.Exception.Message }
            Say "Chưa mở được tunnel: $($_.Exception.Message). Thử lại sau 1 phút." Yellow
            Start-Sleep -Seconds 60
        }
    }
    Save-Pids
    Say "Tunnel hoạt động: $tunnelUrl  (địa chỉ TẠM THỜI)" Green

    # ---- 4. nối giao diện Pages với tunnel
    Stage "[3/3] Cập nhật địa chỉ máy chủ AI cho $($cfg.pages_url)..."
    $pagesSet = Publish-Api $tunnelUrl
    if (-not $pagesSet -and -not $NoPages) { $pagesPending = $tunnelUrl }
    $site = if ($NoPages) { $tunnelUrl } else { $cfg.pages_url }

    $session.state = "ready"; $session.stage = "Sẵn sàng"; $session.message = ""; $session.site = $site; $session.tunnel = $tunnelUrl
    Save-Session
    Say "Sẵn sàng. Website: $site" Green
    if (-not $Headless) {
        Write-Host ""
        Write-Host "  Website:             $site   (ai cũng dùng được, không cần mã)" -ForegroundColor Green
        Write-Host "  Phân tích ảnh:       $site/phan-tich" -ForegroundColor Green
        Write-Host "  Máy chủ AI (tunnel): $tunnelUrl  (tạm thời)" -ForegroundColor DarkGray
        Write-Host ""
        Write-Host "  Giữ cửa sổ này mở, máy tính bật, cắm sạc. Bấm Q để TẮT website online." -ForegroundColor DarkGray
        Write-Host ""
        if (-not $NoBrowser) { Start-Process "$site/phan-tich" }
    }

    # ---- 5. theo dõi + tự phục hồi
    $last = Get-Date; $fails = 0
    while ($true) {
        $q = $false
        if (-not $Headless) { try { if ([Console]::KeyAvailable) { $q = ([Console]::ReadKey($true).Key -eq "Q") } } catch { } }
        if ($q) { break }
        if (Test-Path $stopFlag) { Remove-Item $stopFlag -ErrorAction SilentlyContinue; Say "Nhận lệnh tắt." Cyan; break }

        if ($server.HasExited) {
            Say "Máy chủ AI đã dừng bất ngờ (xem outputs\web_deploy\run\server.log)." Red
            if (-not (Allow-Restart "server" 3)) {
                $session.state = "error"; $session.message = "Máy chủ AI dừng quá 3 lần trong 1 giờ."; Save-Session
                Say "Máy chủ AI dừng quá 3 lần trong 1 giờ: tắt website online để tránh lặp lỗi. Xem log rồi khởi động lại bằng QUAN_LY_MAY_CHU_AI.bat." Red
                break
            }
            Copy-Item (Join-Path $run "server.log") (Join-Path $logDir ("server_loi_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))) -ErrorAction SilentlyContinue
            $session.state = "recovering"; $session.message = "Đang khởi động lại máy chủ AI"; Save-Session
            try {
                $s = Start-Server; $server = $s.Proc; Save-Pids
                $session.server_restarts++; $session.state = "ready"; $session.message = ""; Save-Session
                Say "Đã khởi động lại máy chủ AI ($($s.Health.device))." Green
            } catch { Say "Khởi động lại máy chủ AI thất bại: $($_.Exception.Message)" Red; Start-Sleep -Seconds 30 }
            $last = Get-Date; continue
        }

        $broken = $tunnel.HasExited
        if (-not $broken -and ((Get-Date) - $last).TotalSeconds -ge 60) {
            $last = Get-Date
            $hp = Health $tunnelUrl 15
            if ($hp) {
                $fails = 0
                if ($session.state -ne "ready") { $session.state = "ready"; $session.message = ""; Save-Session }
                Say ("AI trực tuyến qua tunnel, hàng đợi {0}" -f $hp.queue) DarkGray
            } elseif (Health "http://127.0.0.1:$port" 5) { $fails++; Say "CẢNH BÁO: không truy cập được máy chủ AI qua tunnel ($fails/3)." Yellow }
            $broken = $fails -ge 3
            if ($pagesPending -and -not $broken -and (Test-Internet)) {
                if (Publish-Api $pagesPending) { $pagesSet = $true; $pagesPending = "" }
            }
        }
        if ($broken) {
            # tunnel rớt hoặc mất mạng: chờ có mạng (không tính giới hạn), rồi mở tunnel mới (địa chỉ MỚI) và cập nhật Pages
            Kill-Pid $tunnel.Id
            $session.state = "recovering"; $session.message = "Tunnel bị rớt"; Save-Session
            if (-not (Wait-Internet "tunnel bị rớt")) { break }
            if (-not (Allow-Restart "tunnel" 6)) {
                Say "Tunnel đã mở lại 6 lần trong 1 giờ; chờ 15 phút rồi thử lại." Yellow
                for ($i = 0; $i -lt 900 -and -not (Test-Path $stopFlag); $i++) { Start-Sleep -Seconds 1 }
                continue
            }
            Say "Tunnel bị rớt, đang mở tunnel mới..." Yellow
            try {
                $t = Start-QuickTunnel; $tunnel = $t.Proc; $tunnelUrl = $t.Url; Save-Pids
                $session.tunnel_restarts++; $session.tunnel = $tunnelUrl; $session.state = "ready"; $session.message = ""; Save-Session
                Say "Tunnel mới: $tunnelUrl" Green
                if (-not $NoPages) {
                    if (Publish-Api $tunnelUrl) { $pagesSet = $true; $pagesPending = "" } else { $pagesPending = $tunnelUrl }
                } else { $session.site = $tunnelUrl; Save-Session }
                $fails = 0; $last = Get-Date
            } catch {
                Say "Chưa mở lại được tunnel: $($_.Exception.Message). Thử lại sau 1 phút." Yellow
                $tunnel = Start-Process cmd.exe -ArgumentList "/c", "exit" -WindowStyle Hidden -PassThru   # placeholder đã thoát: lần sau thử lại
                Start-Sleep -Seconds 60
            }
        }
        Start-Sleep -Milliseconds 500
    }
}
catch {
    if ($session.state -ne "error") { $session.state = "error"; $session.message = "Lỗi bất ngờ: $($_.Exception.Message)"; Save-Session }
    Say "LỖI: $($_.Exception.Message) (dòng $($_.InvocationInfo.ScriptLineNumber))" Red
}
finally {
    Say "Đang tắt website online..." Cyan
    try { [void][LCAI.Power]::SetThreadExecutionState([uint32]2147483648) } catch { }
    & $tat -Quiet -FromController
    if ($pagesSet -or $pagesPending) {
        try { [void](& (Join-Path $PSScriptRoot "deploy_pages.ps1") -Quiet); Say "Đã gỡ địa chỉ tunnel khỏi $($cfg.pages_url) (trang sẽ báo máy chủ AI đang tắt)." Green }
        catch { Say "Không gỡ được địa chỉ tunnel khỏi Pages: $($_.Exception.Message)" Yellow }
    }
    if ($session.state -ne "error") { $session.state = "stopped" }
    Save-Session
    Say "Đã tắt máy chủ AI và tunnel. Ảnh tải lên không được lưu lại." Green
}
