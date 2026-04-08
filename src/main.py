"""Entry point for CLIP Emotion Image Filter."""

import logging
import os
import sys
import tkinter as tk

# Set TORCH_HOME for portable model caching
app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TORCH_HOME", os.path.join(app_dir, "cache"))
os.environ.setdefault("HF_HOME", os.path.join(app_dir, "cache", "huggingface"))

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
