# Tự khởi động LiverCancer AI khi đăng nhập Windows (Task Scheduler "LiverCancerAI_TuKhoiDong" -> wscript an_cua_so.vbs -> file này).
# Không cần VS Code hay Terminal. Chạy bộ điều khiển mo_demo_online.ps1 ở chế độ ngầm + tự động (-Headless -Auto):
#   máy chủ AI (FastAPI + YOLO CT/MRI + Hybrid CT) -> chờ có Internet -> Quick Tunnel -> cập nhật địa chỉ trên Cloudflare Pages
#   -> theo dõi và tự phục hồi có giới hạn. Không giữ máy thức (chế độ ngủ của Windows giữ nguyên) trừ khi configs\autostart.json bật.
# Nếu bộ điều khiển đang chạy: không làm gì (không bao giờ có 2 bộ YOLO).
param([switch]$Now)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$run = Join-Path $root "outputs\web_deploy\run"
$logDir = Join-Path $root "outputs\web_deploy\logs"
New-Item -ItemType Directory -Force $run, $logDir | Out-Null
$log = Join-Path $logDir ("may_chu_ai_{0:yyyy-MM-dd}.log" -f (Get-Date))
function Log([string]$m) { [IO.File]::AppendAllText($log, ("{0:yyyy-MM-dd HH:mm:ss} [tự khởi động] {1}`r`n" -f (Get-Date), $m), (New-Object Text.UTF8Encoding $false)) }

$pidFile = Join-Path $run "pids.json"
if (Test-Path $pidFile) {
    try {
        $ids = Get-Content $pidFile -Raw | ConvertFrom-Json
        $c = if ($ids.controller) { Get-Process -Id $ids.controller -ErrorAction SilentlyContinue }
        if ($c -and $c.ProcessName -eq "powershell") { Log "Bộ điều khiển đang chạy (PID $($c.Id)); không mở thêm."; exit 0 }
    } catch { }
}
$keep = $false
$cfgFile = Join-Path $root "configs\autostart.json"
if (Test-Path $cfgFile) { try { $keep = [bool]((Get-Content $cfgFile -Raw -Encoding UTF8 | ConvertFrom-Json).keep_awake) } catch { } }

Log "Đăng nhập Windows: khởi động máy chủ AI + Cloudflare Tunnel (giữ máy thức: $keep)."
$args2 = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "mo_demo_online.ps1"), "-Headless", "-Auto")
if ($keep) { $args2 += "-KeepAwake" }
$p = Start-Process powershell.exe -ArgumentList $args2 -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $run "controller.log") -RedirectStandardError (Join-Path $run "controller.err.log")
Log "Bộ điều khiển đã chạy (PID $($p.Id))."
