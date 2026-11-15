# MO_WEBSITE_ONLINE.bat: bật website LiverCancer AI online bằng một thao tác. Xem docs/DEPLOY_CLOUDFLARE.md.
# Chạy mo_demo_online.ps1 ở chế độ NGẦM (cửa sổ ẩn): máy chủ AI, Quick Tunnel và tiến trình theo dõi vẫn chạy sau khi
# cửa sổ này đóng. Chạy lại file này bất cứ lúc nào để xem trạng thái và địa chỉ. Tắt: TAT_WEBSITE_ONLINE.bat.
# Website không có mã truy cập: ai mở trang cũng chọn CT/MRI và tải ảnh lên được.
param([switch]$NoBrowser)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$run = Join-Path $root "outputs\web_deploy\run"
New-Item -ItemType Directory -Force $run | Out-Null
$pidFile = Join-Path $run "pids.json"
$sessFile = Join-Path $run "session.json"
$log = Join-Path $run "controller.log"

function Running {
    if (-not (Test-Path $pidFile)) { return $false }
    $ids = Get-Content $pidFile -Raw | ConvertFrom-Json
    return [bool]($ids.controller -and (Get-Process -Id $ids.controller -ErrorAction SilentlyContinue))
}
function Read-Session { try { return Get-Content $sessFile -Raw -Encoding UTF8 | ConvertFrom-Json } catch { return $null } }

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  LiverCancer AI - WEBSITE ONLINE (Cloudflare Pages + Tunnel)" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

if (-not (Running)) {
    Remove-Item $sessFile, $log -ErrorAction SilentlyContinue
    Write-Host "Đang bật máy chủ AI, Cloudflare Tunnel và cập nhật Pages (khoảng 1-2 phút)..." -ForegroundColor Cyan
    [void](Start-Process powershell -WindowStyle Hidden -WorkingDirectory $root -PassThru `
        -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "mo_demo_online.ps1"), "-Headless" `
        -RedirectStandardOutput $log -RedirectStandardError (Join-Path $run "controller.err.log"))
    $shown = 0; $t0 = Get-Date
    while ($true) {
        Start-Sleep -Seconds 2
        $lines = @(Get-Content $log -Encoding UTF8 -ErrorAction SilentlyContinue)
        for (; $shown -lt $lines.Count; $shown++) { Write-Host "  $($lines[$shown])" -ForegroundColor DarkGray }
        $s = Read-Session
        if ($s -and $s.state -in @("ready", "error", "stopped")) { break }
        if (((Get-Date) - $t0).TotalMinutes -gt 10) { Write-Host "Quá 10 phút chưa sẵn sàng. Xem outputs\web_deploy\run\controller.log" -ForegroundColor Red; exit 1 }
    }
}

$s = Read-Session
if (-not $s -or $s.state -ne "ready" -or -not (Running)) {
    Write-Host ""
    Write-Host "Website online CHƯA chạy được." -ForegroundColor Red
    if ($s -and $s.message) { Write-Host "Lý do: $($s.message)" -ForegroundColor Red }
    Write-Host "Chi tiết: outputs\web_deploy\run\controller.log, server.log, tunnel.log. Cách xử lý: docs\DEPLOY_CLOUDFLARE.md (Gỡ lỗi nhanh)." -ForegroundColor Yellow
    exit 1
}

$h = $null
try { $h = Invoke-RestMethod "$($s.tunnel)/api/health" -TimeoutSec 15 } catch { }
Write-Host ""
Write-Host "  Trạng thái:          $(if ($h) { "ĐANG HOẠT ĐỘNG (AI chạy trên $($h.device) của máy này, hàng đợi $($h.queue))" } else { 'tunnel chưa phản hồi; tiến trình theo dõi sẽ tự mở tunnel mới trong vài phút' })" -ForegroundColor $(if ($h) { "Green" } else { "Yellow" })
Write-Host "  Bật từ:              $($s.started)$(if ($s.tunnel_restarts) { "   (tunnel đã tự mở lại $($s.tunnel_restarts) lần)" })"
Write-Host "  Website:             $($s.site)   (ai cũng dùng được, không cần mã)" -ForegroundColor Green
Write-Host "  Phân tích ảnh:       $($s.site)/phan-tich" -ForegroundColor Green
Write-Host "  Máy chủ AI (tunnel): $($s.tunnel)   (TẠM THỜI, không cố định)" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  - Có thể đóng cửa sổ này: website vẫn chạy ngầm. Chạy lại MO_WEBSITE_ONLINE.bat để xem lại trạng thái." -ForegroundColor DarkGray
Write-Host "  - Phân tích ảnh chỉ hoạt động khi máy tính này bật, có Internet, không ngủ (cắm sạc, không gập nắp)." -ForegroundColor DarkGray
Write-Host "  - Tắt: TAT_WEBSITE_ONLINE.bat. Mỗi IP bị giới hạn số lượt phân tích để máy không quá tải." -ForegroundColor DarkGray
try { Set-Clipboard -Value "$($s.site)/phan-tich" } catch { }
if (-not $NoBrowser) { Start-Process "$($s.site)/phan-tich" }
