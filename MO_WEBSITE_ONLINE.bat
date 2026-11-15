@echo off
rem LiverCancer AI - bat website online bang mot thao tac: may chu AI + Cloudflare Quick Tunnel + cap nhat Cloudflare Pages.
rem Chay ngam: dong cua so nay website van chay. Chay lai file nay de xem trang thai. Tu khoi dong + quan ly: QUAN_LY_MAY_CHU_AI.bat Tat: TAT_WEBSITE_ONLINE.bat
rem Huong dan: docs\DEPLOY_CLOUDFLARE.md
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\web\mo_website_online.ps1" %*
echo.
pause
