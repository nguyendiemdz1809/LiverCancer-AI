# Quản lý máy chủ AI của LiverCancer AI (QUAN_LY_MAY_CHU_AI.bat). Không cần VS Code.
#   menu:  powershell -ExecutionPolicy Bypass -File scripts\web\quan_ly_may_chu_ai.ps1
#   lệnh:  ... -Action trang_thai | khoi_dong_lai | dung | log | bat_tu_khoi_dong | tat_tu_khoi_dong
# Tự khởi động = tác vụ Task Scheduler "LiverCancerAI_TuKhoiDong" của tài khoản Windows hiện tại (khi đăng nhập, chạy ẩn,
# không cần quyền Administrator). Không đổi cài đặt nguồn / chế độ ngủ / Windows Update / Windows Security.
param([string]$Action = "")
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$run = Join-Path $root "outputs\web_deploy\run"
$logDir = Join-Path $root "outputs\web_deploy\logs"
$task = "LiverCancerAI_TuKhoiDong"
$cfg = Get-Content (Join-Path $root "configs\web_deploy.json") -Raw -Encoding UTF8 | ConvertFrom-Json

function Running {
    $f = Join-Path $run "pids.json"
    if (-not (Test-Path $f)) { return $null }
    try { $ids = Get-Content $f -Raw | ConvertFrom-Json } catch { return $null }
    $c = if ($ids.controller) { Get-Process -Id $ids.controller -ErrorAction SilentlyContinue }
    if ($c -and $c.ProcessName -eq "powershell") { return $ids }
    return $null
}
function Health([string]$base) { try { return Invoke-RestMethod "$base/api/health" -TimeoutSec 10 } catch { return $null } }

function Show-Status {
    $ids = Running
    $s = $null
    try { $s = Get-Content (Join-Path $run "session.json") -Raw -Encoding UTF8 | ConvertFrom-Json } catch { }
    $local = Health "http://127.0.0.1:$($cfg.port)"
    $pub = if ($s -and $s.tunnel) { Health $s.tunnel } else { $null }
    $t = Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
    $stateVi = @{ ready = "ĐANG HOẠT ĐỘNG"; starting = "đang khởi động"; recovering = "đang tự phục hồi"; offline = "chờ có Internet"; error = "LỖI"; stopped = "đã tắt" }
    Write-Host ""
    Write-Host "=== Trạng thái máy chủ AI – LiverCancer AI ===" -ForegroundColor Cyan
    Write-Host ("Bộ điều khiển:   " + $(if ($ids) { "đang chạy (PID $($ids.controller); máy chủ $($ids.server); tunnel $($ids.tunnel))" } else { "KHÔNG chạy" })) -ForegroundColor $(if ($ids) { "Green" } else { "Yellow" })
    if ($s) {
        Write-Host ("Trạng thái:      " + $(if ($stateVi[$s.state]) { $stateVi[$s.state] } else { $s.state }) + $(if ($s.message) { " – $($s.message)" } else { "" }))
        Write-Host "Bật từ:          $($s.started)   (chế độ: $($s.mode); tunnel mở lại $($s.tunnel_restarts) lần; máy chủ khởi động lại $($s.server_restarts) lần)"
    }
    Write-Host ("Máy chủ AI:      " + $(if ($local) { "sẵn sàng – CT $($local.models.ct), MRI $($local.models.mri), $($local.device), cấu hình khóa $($local.locked_config)" } else { "không phản hồi" })) -ForegroundColor $(if ($local) { "Green" } else { "Yellow" })
    Write-Host ("Qua Internet:    " + $(if ($pub) { "OK – $($s.tunnel) (địa chỉ tạm thời)" } else { "không truy cập được" })) -ForegroundColor $(if ($pub) { "Green" } else { "Yellow" })
    Write-Host "Website:         $($cfg.pages_url)/phan-tich"
    Write-Host ("Tự khởi động:    " + $(if ($t) { "BẬT (khi đăng nhập Windows; tác vụ: $($t.State))" } else { "TẮT" }))
    Write-Host ""
}
function Start-Ai {
    if (Running) { Write-Host "Máy chủ AI đang chạy." -ForegroundColor Yellow; return }
    Write-Host "Đang khởi động máy chủ AI (chạy ẩn, khoảng 1-2 phút)..." -ForegroundColor Cyan
    & wscript.exe (Join-Path $PSScriptRoot "an_cua_so.vbs") "tu_khoi_dong.ps1"
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalMinutes -lt 6) {
        Start-Sleep -Seconds 5
        try { $s = Get-Content (Join-Path $run "session.json") -Raw -Encoding UTF8 | ConvertFrom-Json } catch { continue }
        if ($s.state -eq "ready" -and (Running)) { Write-Host "Sẵn sàng." -ForegroundColor Green; return }
        if ($s.state -eq "error") { Write-Host "Lỗi: $($s.message)" -ForegroundColor Red; return }
        if ($s.state -eq "offline") { Write-Host "Máy chủ AI đã chạy, đang chờ có Internet để mở tunnel." -ForegroundColor Yellow; return }
    }
    Write-Host "Chưa sẵn sàng sau 6 phút; xem log (mục 4)." -ForegroundColor Yellow
}
function Stop-Ai { & (Join-Path $PSScriptRoot "tat_demo_online.ps1") -ClearPages }
function Show-Log {
    $f = Get-ChildItem $logDir -Filter "may_chu_ai_*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
    Write-Host "=== Nhật ký: $($f.FullName) (40 dòng cuối) ===" -ForegroundColor Cyan
    if ($f) { Get-Content $f.FullName -Tail 40 -Encoding UTF8 }
    $e = Join-Path $run "server.log"
    if (Test-Path $e) {
        $err = Select-String -Path $e -Pattern "Traceback|Error|Exception" -ErrorAction SilentlyContinue | Select-Object -Last 8
        if ($err) { Write-Host "--- Lỗi gần nhất trong server.log ---" -ForegroundColor Yellow; $err | ForEach-Object { $_.Line } }
    }
    Write-Host "Thư mục log: $logDir" -ForegroundColor DarkGray
}
function Enable-Auto {
    $a = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\wscript.exe" `
        -Argument ('"{0}" tu_khoi_dong.ps1' -f (Join-Path $PSScriptRoot "an_cua_so.vbs")) -WorkingDirectory $root
    $tr = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    $tr.Delay = "PT30S"                                   # chờ 30 giây sau khi đăng nhập
    $st = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
        -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
    $pr = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $task -Action $a -Trigger $tr -Settings $st -Principal $pr -Force `
        -Description "LiverCancer AI: tu khoi dong may chu AI (FastAPI + YOLO + Hybrid) va Cloudflare Tunnel khi dang nhap. Quan ly: QUAN_LY_MAY_CHU_AI.bat" | Out-Null
    Write-Host "Đã BẬT tự khởi động khi đăng nhập Windows (tác vụ '$task', tài khoản $env:USERNAME, không cần quyền Administrator)." -ForegroundColor Green
}
function Disable-Auto {
    if (Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $task -Confirm:$false
        Write-Host "Đã TẮT tự khởi động (máy chủ AI đang chạy, nếu có, vẫn chạy cho tới khi dừng)." -ForegroundColor Green
    } else { Write-Host "Tự khởi động đang tắt sẵn." }
}

switch ($Action) {
    "trang_thai" { Show-Status; exit }
    "khoi_dong_lai" { if (Running) { Stop-Ai }; Start-Ai; Show-Status; exit }
    "khoi_dong" { Start-Ai; exit }
    "dung" { Stop-Ai; exit }
    "log" { Show-Log; exit }
    "bat_tu_khoi_dong" { Enable-Auto; exit }
    "tat_tu_khoi_dong" { Disable-Auto; exit }
}
while ($true) {
    Show-Status
    Write-Host "1. Xem lại trạng thái     2. Khởi động / khởi động lại máy chủ AI     3. Dừng máy chủ AI"
    Write-Host "4. Xem log lỗi            5. Bật tự khởi động khi đăng nhập           6. Tắt tự khởi động      0. Thoát"
    $c = Read-Host "Chọn"
    switch ($c) {
        "1" { }
        "2" { if (Running) { Stop-Ai }; Start-Ai }
        "3" { Stop-Ai }
        "4" { Show-Log; Read-Host "Nhấn Enter để quay lại" | Out-Null }
        "5" { Enable-Auto }
        "6" { Disable-Auto }
        "0" { exit }
    }
}
