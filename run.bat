@echo off
chcp 65001 >nul
cd /d "%~dp0"
set TORCH_HOME=%~dp0cache
python\python.exe src\main.py %*
if errorlevel 1 pause
