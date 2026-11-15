# Tắt website online: dừng máy chủ AI, Cloudflare Tunnel và tiến trình theo dõi do mo_demo_online.ps1 mở
# (chỉ các tiến trình đã ghi trong outputs\web_deploy\run\pids.json), rồi gỡ địa chỉ tunnel khỏi Pages.
#   powershell -ExecutionPolicy Bypass -File scripts\web\tat_demo_online.ps1 -ClearPages     (= TAT_WEBSITE_ONLINE.bat)
param([switch]$ClearPages, [switch]$Quiet, [switch]$FromController)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$run = Join-Path $root "outputs\web_deploy\run"
$pidFile = Join-Path $run "pids.json"
function Say([string]$m, [string]$c = "Gray") { if (-not $Quiet) { Write-Host $m -ForegroundColor $c } }
function Kill-Tree([int]$id, [string[]]$names) {
    $p = Get-Process -Id $id -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -in $names -and $id -ne $PID) {
        $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        & taskkill.exe /PID $id /T /F 2>&1 | Out-Null
        $ErrorActionPreference = $old
        return 1
    }
    return 0
}

$stopped = 0; $handled = $false
if ((Test-Path $pidFile) -and -not $FromController) {
    # Tiến trình theo dõi còn chạy thì để nó tự tắt (nó cũng gỡ địa chỉ tunnel khỏi Pages).
    $ids = Get-Content $pidFile -Raw | ConvertFrom-Json
    if ($ids.controller -and (Get-Process -Id $ids.controller -ErrorAction SilentlyContinue)) {
        Set-Content (Join-Path $run "stop.flag") "stop"
        Say "Đang yêu cầu website online tự tắt..." Cyan
        for ($i = 0; $i -lt 40 -and (Test-Path $pidFile); $i++) { Start-Sleep -Milliseconds 500 }
        $handled = -not (Test-Path $pidFile)
        if ($handled) {
            # chờ nó gỡ xong địa chỉ tunnel khỏi Pages
            for ($i = 0; $i -lt 90 -and (Get-Process -Id $ids.controller -ErrorAction SilentlyContinue); $i++) { Start-Sleep -Milliseconds 500 }
            Say "Đã tắt máy chủ AI, tunnel và gỡ địa chỉ máy chủ AI khỏi Pages." Green
        }
        Remove-Item (Join-Path $run "stop.flag") -ErrorAction SilentlyContinue
    }
}
if (Test-Path $pidFile) {
    $ids = Get-Content $pidFile -Raw | ConvertFrom-Json
    if ($ids.tunnel) { $stopped += Kill-Tree $ids.tunnel @("cloudflared") }
    if ($ids.server) { $stopped += Kill-Tree $ids.server @("python") }
    if (-not $FromController -and $ids.controller) { $stopped += Kill-Tree $ids.controller @("powershell") }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}
if ($stopped) { Say "Đã dừng $stopped tiến trình của website online." Green }
elseif (-not $handled) { Say "Không có tiến trình website online nào đang chạy." Gray }
if ($ClearPages -and -not $handled) {
    [void](& (Join-Path $PSScriptRoot "deploy_pages.ps1") -Quiet)
    Say "Đã gỡ địa chỉ máy chủ AI khỏi giao diện Pages (trang sẽ báo máy chủ AI đang tắt)." Green
}
$sess = Join-Path $run "session.json"
if (-not $FromController -and (Test-Path $sess)) {
    try { $s = Get-Content $sess -Raw -Encoding UTF8 | ConvertFrom-Json; if ($s.state -ne "error") { $s.state = "stopped" }; [IO.File]::WriteAllText($sess, ($s | ConvertTo-Json), (New-Object Text.UTF8Encoding $false)) } catch { }
}
