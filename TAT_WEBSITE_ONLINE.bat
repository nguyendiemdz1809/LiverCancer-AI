@echo off
rem LiverCancer AI - tat website online: dung may chu AI, tunnel, tien trinh theo doi; go dia chi tunnel khoi Pages.
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\web\tat_demo_online.ps1" -ClearPages
echo.
pause
