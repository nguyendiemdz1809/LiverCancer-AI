# Đã thay bằng MO_DEMO_ONLINE.bat (scripts\web\mo_demo_online.ps1): Pages + Quick Tunnel, tự nối giao diện (không mã truy cập).
# Giữ file này để lệnh cũ vẫn chạy:
#   powershell -ExecutionPolicy Bypass -File scripts\web\start_public_demo.ps1          (= MO_DEMO_ONLINE.bat)
#   powershell -ExecutionPolicy Bypass -File scripts\web\start_public_demo.ps1 -NoPages (chỉ địa chỉ trycloudflare)
param([switch]$NoPages, [switch]$NoBrowser)
& (Join-Path $PSScriptRoot "mo_demo_online.ps1") -NoPages:$NoPages -NoBrowser:$NoBrowser