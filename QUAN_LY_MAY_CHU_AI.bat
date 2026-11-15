@echo off
rem LiverCancer AI - quan ly may chu AI: trang thai, khoi dong lai, dung, xem log, bat/tat tu khoi dong khi dang nhap Windows.
rem Khong can VS Code. Huong dan: docs\DEPLOY_CLOUDFLARE.md
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\web\quan_ly_may_chu_ai.ps1" %*
