"""Entry point for Emotion Image Filter."""

import logging
import os
import sys
import tkinter as tk

# For PyInstaller: ensure bundled modules are on sys.path
if getattr(sys, 'frozen', False):
    sys.path.insert(0, sys._MEIPASS)
    # exe 실행 시 exe 파일이 있는 폴더 기준으로 캐시 저장
    app_dir = os.path.dirname(sys.executable)
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set portable model directories under the app-local models folder
os.environ.setdefault("HF_HOME", os.path.join(app_dir, "models", "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(app_dir, "models", "torch"))
os.environ.setdefault("ASSET_FILTER_APP_DIR", app_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from gui import App


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
