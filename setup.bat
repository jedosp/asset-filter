@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [Setup] Installing dependencies...
python\python.exe -m pip install --upgrade pip
python\python.exe -m pip install -r requirements.txt --target python\Lib\site-packages
echo [Setup] Done. Run 'run.bat' to start.
pause
