"""tkinter GUI for Emotion Image Filter."""

import logging
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

from parser import parse_filename, scan_folder
from filter import filter_and_copy
from report import generate_report

logger = logging.getLogger(__name__)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Emotion Image Filter")
        self.root.resizable(False, False)

        self.queue: queue.Queue = queue.Queue()
        self.emotion_groups: dict | None = None
        self.output_dir: Path | None = None
        self.running = False

        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 4}
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(sticky="nsew")

        # --- Input folder ---
        row = 0
        ttk.Label(frame, text="Input Folder:").grid(row=row, column=0, sticky="w", **pad)
        self.folder_var = tk.StringVar()
        self.folder_entry = ttk.Entry(frame, textvariable=self.folder_var, width=50)
        self.folder_entry.grid(row=row, column=1, sticky="ew", **pad)
        self.browse_btn = ttk.Button(frame, text="Browse…", command=self._browse_folder)
        self.browse_btn.grid(row=row, column=2, **pad)

        # --- Folder info ---
        row += 1
        self.folder_info_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.folder_info_var, foreground="gray").grid(
            row=row, column=0, columnspan=3, sticky="w", **pad
        )

        # --- Top N ---
        row += 1
        ttk.Label(frame, text="Top N per emotion:").grid(row=row, column=0, sticky="w", **pad)
        self.topn_var = tk.IntVar(value=10)
        self.topn_spin = ttk.Spinbox(frame, from_=1, to=30, textvariable=self.topn_var, width=6)
        self.topn_spin.grid(row=row, column=1, sticky="w", **pad)

        # --- Separator ---
        row += 1
        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8
        )

        # --- Progress ---
        row += 1
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            frame, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.progress_bar.grid(row=row, column=0, columnspan=3, sticky="ew", **pad)

        # --- Status labels ---
        row += 1
        self.status_var = tk.StringVar(value="Ready.")
        self.status_label = ttk.Label(frame, textvariable=self.status_var)
        self.status_label.grid(row=row, column=0, columnspan=3, sticky="w", **pad)

        row += 1
        self.emo_status_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.emo_status_var).grid(
            row=row, column=0, columnspan=3, sticky="w", **pad
        )

        # --- Buttons ---
        row += 1
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)
        self.run_btn = ttk.Button(btn_frame, text="▶ Run Filter", command=self._run_filter)
        self.run_btn.pack(side="left", padx=8)
        self.run_btn.state(["disabled"])
        self.open_btn = ttk.Button(btn_frame, text="Open Output", command=self._open_output)
        self.open_btn.pack(side="left", padx=8)
        self.open_btn.state(["disabled"])

    def _set_running(self, running: bool):
        self.running = running
        state = ["disabled"] if running else ["!disabled"]
        self.browse_btn.state(state)
        self.run_btn.state(state)
        self.topn_spin.state(state)
        if not running and self.output_dir and self.output_dir.exists():
            self.open_btn.state(["!disabled"])

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select emotion image folder")
        if not folder:
            return

        folder_path = Path(folder)
        self.folder_var.set(str(folder_path))

        self.emotion_groups = scan_folder(folder_path)
        if not self.emotion_groups:
            self.folder_info_var.set("No valid .png files found in this folder.")
            self.run_btn.state(["disabled"])
            return

        total_images = sum(len(v) for v in self.emotion_groups.values())
        num_emotions = len(self.emotion_groups)

        # Try to detect character name
        first_file = next(iter(next(iter(self.emotion_groups.values()))))[0]
        result = parse_filename(first_file.name)
        char_str = f" (character: {result[0]})" if result else ""

        self.folder_info_var.set(
            f"Found {num_emotions} emotions, {total_images:,} images{char_str}"
        )
        self.run_btn.state(["!disabled"])

    def _run_filter(self):
        if self.emotion_groups is None:
            return

        self._set_running(True)
        self.progress_var.set(0)
        self.emo_status_var.set("")

        input_folder = Path(self.folder_var.get())
        self.output_dir = input_folder.parent / f"{input_folder.name}_filtered"
        self.status_var.set("Loading WD Tagger v3 model...")

        thread = threading.Thread(
            target=self._worker,
            args=(
                self.emotion_groups,
                self.topn_var.get(),
                self.output_dir,
            ),
            daemon=True,
        )
        thread.start()

    def _worker(self, emotion_groups, top_n, output_dir):
        try:
            from wd_scorer import WDTaggerScorer
            total_images = sum(len(v) for v in emotion_groups.values())

            scorer = WDTaggerScorer()
            self.queue.put(("status", "Analyzing images with WD Tagger..."))

            def progress_cb(current, total):
                pct = current / total * 100
                self.queue.put(("progress", pct))
                self.queue.put(("status", f"Analyzing images... {current}/{total}"))

            scored = scorer.score_all(
                emotion_groups, batch_size=16, progress_callback=progress_cb,
            )

            self.queue.put(("status", "Copying files..."))
            output_dir.mkdir(parents=True, exist_ok=True)
            total_copied = filter_and_copy(scored, top_n, output_dir)

            config = {
                "top_n": top_n,
                "model": "WD-Tagger-v3",
            }
            generate_report(scored, top_n, config, output_dir)

            self.queue.put(("progress", 100))
            self.queue.put(("done", f"Done! {total_copied} images copied."))

        except Exception as e:
            logger.exception("Worker error")
            self.queue.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                msg_type, msg_data = self.queue.get_nowait()
                if msg_type == "progress":
                    self.progress_var.set(msg_data)
                elif msg_type == "status":
                    self.status_var.set(msg_data)
                elif msg_type == "emo_status":
                    self.emo_status_var.set(msg_data)
                elif msg_type == "done":
                    self._set_running(False)
                    self.status_var.set(msg_data)
                    self.emo_status_var.set("")
                    messagebox.showinfo("Complete", msg_data)
                elif msg_type == "error":
                    self._set_running(False)
                    self.status_var.set(f"Error: {msg_data}")
                    self.status_label.configure(foreground="red")
        except queue.Empty:
            pass

        self.root.after(100, self._poll_queue)

    def _open_output(self):
        if self.output_dir and self.output_dir.exists():
            os.startfile(str(self.output_dir))
