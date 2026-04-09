@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [Build] Installing PyInstaller...
pip install pyinstaller

echo [Build] Cleaning previous build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist AssetFilter.spec del AssetFilter.spec

echo [Build] Building AssetFilter.exe...
python -m PyInstaller ^
    --onefile ^
    --console ^
    --name AssetFilter ^
    --add-data "src\filename_parser.py;." ^
    --add-data "src\gui.py;." ^
    --add-data "src\filter.py;." ^
    --add-data "src\report.py;." ^
    --add-data "src\wd_scorer.py;." ^
    --add-data "src\aesthetic_scorer.py;." ^
    --add-data "src\face_scorer.py;." ^
    --hidden-import onnxruntime ^
    --hidden-import huggingface_hub ^
    --hidden-import PIL ^
    --hidden-import mediapipe ^
    --hidden-import cv2 ^
    --collect-all onnxruntime ^
    --exclude-module torch ^
    --exclude-module open_clip ^
    --exclude-module PyQt6 ^
    --exclude-module scipy ^
    --exclude-module pandas ^
    --exclude-module matplotlib ^
    src\main.py

echo [Build] Done. Output: dist\AssetFilter.exe
pause
