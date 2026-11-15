# Chạy website LiverCancer AI trên máy (Windows 11): backend FastAPI + giao diện React đã build, một địa chỉ duy nhất.
#   powershell -ExecutionPolicy Bypass -File scripts\web\run_website.ps1            (mở http://127.0.0.1:8000)
#   powershell -ExecutionPolicy Bypass -File scripts\web\run_website.ps1 -Rebuild   (build lại giao diện trước)
param([int]$Port = 8000, [switch]$Rebuild, [switch]$NoBrowser)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

if ($Rebuild -or -not (Test-Path "web\dist\index.html")) {
    Write-Host "Đang build giao diện (web\dist)..." -ForegroundColor Cyan
    Push-Location web
    if (-not (Test-Path "node_modules")) { npm install --no-audit --no-fund }
    npm run build
    Pop-Location
}

$env:YOLO_OFFLINE = "1"          # không bao giờ để Ultralytics tải gì từ mạng
$env:PYTHONIOENCODING = "utf-8"
if (-not $NoBrowser) {
    Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile", "-Command", "Start-Sleep 6; Start-Process http://127.0.0.1:$Port/"
}
Write-Host "Website: http://127.0.0.1:$Port/  (Ctrl+C để dừng). Mô hình AI cần khoảng 20-60 giây để khởi động." -ForegroundColor Green
& ".venv\Scripts\python.exe" -m uvicorn server.main:app --host 127.0.0.1 --port $Port
