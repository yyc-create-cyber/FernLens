@echo off
rem FernLens Web launcher (ASCII wrapper; real logic is in start-web.ps1)
rem Usage: start-web.bat [--port 8642]
setlocal
cd /d "%~dp0"
set "PORT=8642"
if /i "%~1"=="--port" if not "%~2"=="" set "PORT=%~2"
powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0start-web.ps1" -Port %PORT%
exit /b 0
