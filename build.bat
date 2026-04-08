@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [Build] Installing PyInstaller...
pip install pyinstaller

echo [Build] Building AssetFilter.exe...
pyinstaller ^
    --onefile ^
    --windowed ^
    --name AssetFilter ^
    --add-data "src;src" ^
    --hidden-import onnxruntime ^
    --hidden-import huggingface_hub ^
    --hidden-import PIL ^
    --collect-all onnxruntime ^
    src\main.py

echo [Build] Done. Output: dist\AssetFilter.exe
pause
