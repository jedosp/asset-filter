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

from PIL import Image, ImageTk

from filename_parser import parse_filename, scan_folder, extract_exif_tags_by_emotion
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
        self.reference_paths: list[Path] = []
        self.reference_thumbnails: list[ImageTk.PhotoImage] = []
        self.consistency_model_available = False
        self.consistency_cache_dir: Path | None = None

        try:
            from consistency_scorer import ConsistencyScorer

            self.consistency_cache_dir = ConsistencyScorer.get_expected_cache_dir(
                cache_dir=os.environ.get("HF_HOME")
            )
            self.consistency_model_available = ConsistencyScorer.is_runtime_available()
        except Exception as exc:
            logger.warning("Failed to probe consistency model availability: %s", exc)

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused

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

        # --- Reference Consistency ---
        row += 1
        ref_frame = ttk.LabelFrame(frame, text="Reference Consistency", padding=8)
        ref_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=10, pady=8)
        ref_frame.columnconfigure(2, weight=1)

        rrow = 0
        self.consistency_var = tk.BooleanVar(value=False)
        self.consistency_check = ttk.Checkbutton(
            ref_frame,
            text="Enable Reference Consistency",
            variable=self.consistency_var,
            command=self._update_scoring_ui,
        )
        self.consistency_check.grid(row=rrow, column=0, columnspan=3, sticky="w", pady=2)

        rrow += 1
        self.reference_add_btn = ttk.Button(
            ref_frame,
            text="Add Images...",
            command=self._add_reference_images,
        )
        self.reference_add_btn.grid(row=rrow, column=0, sticky="w", pady=2)
        self.reference_clear_btn = ttk.Button(
            ref_frame,
            text="Clear",
            command=self._clear_reference_images,
        )
        self.reference_clear_btn.grid(row=rrow, column=1, sticky="w", padx=(8, 0), pady=2)

        rrow += 1
        self.reference_status_var = tk.StringVar(value="No reference images selected.")
        self.reference_status_label = ttk.Label(ref_frame, textvariable=self.reference_status_var)
        self.reference_status_label.grid(row=rrow, column=0, columnspan=3, sticky="w", pady=2)

        rrow += 1
        self.reference_hint = ttk.Label(
            ref_frame,
            text="Select 1-5 images of the correct design. Tip: run once, then reuse your best outputs as references.",
            foreground="gray",
            font=("Segoe UI", 8),
        )
        self.reference_hint.grid(row=rrow, column=0, columnspan=3, sticky="w", pady=(0, 2))

        rrow += 1
        if self.consistency_model_available:
            model_text = (
                "DINOv2 will be downloaded automatically on first use and cached in "
                f"{self.consistency_cache_dir}"
            )
            model_color = "gray"
        else:
            model_text = (
                "Runtime not available. Reference consistency is disabled because torch or transformers "
                "could not be imported."
            )
            model_color = "darkred"
        self.consistency_model_var = tk.StringVar(value=model_text)
        self.consistency_model_label = ttk.Label(
            ref_frame,
            textvariable=self.consistency_model_var,
            foreground=model_color,
            font=("Segoe UI", 8),
            wraplength=420,
            justify="left",
        )
        self.consistency_model_label.grid(row=rrow, column=0, columnspan=3, sticky="w", pady=(0, 2))

        rrow += 1
        self.reference_preview_frame = ttk.Frame(ref_frame)
        self.reference_preview_frame.grid(row=rrow, column=0, columnspan=3, sticky="ew", pady=4)

        rrow += 1
        self.consistency_mode_label = ttk.Label(ref_frame, text="Mode:")
        self.consistency_mode_label.grid(row=rrow, column=0, sticky="w", pady=2)
        self.consistency_mode_var = tk.StringVar(value="Hard Filter")
        self.consistency_mode_combo = ttk.Combobox(
            ref_frame,
            textvariable=self.consistency_mode_var,
            values=["Weighted", "Hard Filter"],
            state="readonly",
            width=12,
        )
        self.consistency_mode_combo.grid(row=rrow, column=1, sticky="w", pady=2)
        self.consistency_mode_combo.bind("<<ComboboxSelected>>", lambda e: self._update_scoring_ui())

        rrow += 1
        self.consistency_threshold_label = ttk.Label(ref_frame, text="Threshold:")
        self.consistency_threshold_label.grid(row=rrow, column=0, sticky="w", pady=2)
        self.consistency_threshold_var = tk.DoubleVar(value=0.60)
        self.consistency_threshold_spin = ttk.Spinbox(
            ref_frame,
            from_=-1.0,
            to=1.0,
            increment=0.05,
            textvariable=self.consistency_threshold_var,
            width=6,
        )
        self.consistency_threshold_spin.grid(row=rrow, column=1, sticky="w", pady=2)

        if not self.consistency_model_available:
            self.consistency_var.set(False)

        # --- Scoring Options ---
        row += 1
        scoring_frame = ttk.LabelFrame(frame, text="Scoring Options", padding=8)
        scoring_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=10, pady=8)
        scoring_frame.columnconfigure(1, weight=1)

        srow = 0
        self.aesthetic_var = tk.BooleanVar(value=True)
        self.aesthetic_check = ttk.Checkbutton(
            scoring_frame, text="Enable Aesthetic Score",
            variable=self.aesthetic_var, command=self._update_scoring_ui,
        )
        self.aesthetic_check.grid(row=srow, column=0, columnspan=3, sticky="w", pady=2)

        srow += 1
        self.min_aes_label = ttk.Label(scoring_frame, text="Min Quality:")
        self.min_aes_label.grid(row=srow, column=0, sticky="w", padx=(20, 5), pady=2)
        self.min_aes_var = tk.DoubleVar(value=3.0)
        self.min_aes_spin = ttk.Spinbox(
            scoring_frame, from_=1.0, to=10.0, increment=0.5,
            textvariable=self.min_aes_var, width=6,
        )
        self.min_aes_spin.grid(row=srow, column=1, sticky="w", pady=2)
        self.min_aes_hint = ttk.Label(scoring_frame, text="(1.0~10.0)", foreground="gray")
        self.min_aes_hint.grid(row=srow, column=2, sticky="w", padx=5, pady=2)

        srow += 1
        self.face_var = tk.BooleanVar(value=False)
        self.face_check = ttk.Checkbutton(
            scoring_frame, text="Enable Face Framing",
            variable=self.face_var, command=self._update_scoring_ui,
        )
        self.face_check.grid(row=srow, column=0, columnspan=3, sticky="w", pady=2)

        srow += 1
        self.face_mode_label = ttk.Label(scoring_frame, text="Mode:")
        self.face_mode_label.grid(row=srow, column=0, sticky="w", padx=(20, 5), pady=2)
        self.face_mode_var = tk.StringVar(value="Hard Filter")
        self.face_mode_combo = ttk.Combobox(
            scoring_frame, textvariable=self.face_mode_var,
            values=["Hard Filter", "Weighted"], state="readonly", width=12,
        )
        self.face_mode_combo.grid(row=srow, column=1, sticky="w", pady=2)
        self.face_mode_combo.bind("<<ComboboxSelected>>", lambda e: self._update_scoring_ui())

        srow += 1
        self.face_threshold_label = ttk.Label(scoring_frame, text="Threshold:")
        self.face_threshold_label.grid(row=srow, column=0, sticky="w", padx=(20, 5), pady=2)
        self.face_threshold_var = tk.DoubleVar(value=0.30)
        self.face_threshold_spin = ttk.Spinbox(
            scoring_frame, from_=0.0, to=1.0, increment=0.05,
            textvariable=self.face_threshold_var, width=6,
        )
        self.face_threshold_spin.grid(row=srow, column=1, sticky="w", pady=2)

        srow += 1
        self.exclude_label = ttk.Label(scoring_frame, text="Exclude Tags:")
        self.exclude_label.grid(row=srow, column=0, sticky="w", pady=2)
        self.exclude_tags_var = tk.StringVar(value="")
        self.exclude_tags_entry = ttk.Entry(
            scoring_frame, textvariable=self.exclude_tags_var, width=30,
        )
        self.exclude_tags_entry.grid(row=srow, column=1, columnspan=2, sticky="ew", pady=2)

        srow += 1
        self.exclude_hint = ttk.Label(
            scoring_frame, text="comma-separated, e.g. glasses, red_eyes",
            foreground="gray", font=("Segoe UI", 8),
        )
        self.exclude_hint.grid(row=srow, column=1, columnspan=2, sticky="w", pady=(0, 2))

        srow += 1
        self.weight_sep = ttk.Separator(scoring_frame, orient="horizontal")
        self.weight_sep.grid(row=srow, column=0, columnspan=3, sticky="ew", pady=6)

        srow += 1
        self.weight_section_label = ttk.Label(scoring_frame, text="Weights:")
        self.weight_section_label.grid(row=srow, column=0, sticky="w", pady=2)

        self._updating_weights = False

        srow += 1
        self.emotion_weight_var = tk.DoubleVar(value=0.65)
        self.emotion_weight_label = ttk.Label(scoring_frame, text="Emotion:")
        self.emotion_weight_label.grid(row=srow, column=0, sticky="w", padx=(20, 5), pady=2)
        self.emotion_weight_spin = ttk.Spinbox(
            scoring_frame, from_=0.0, to=1.0, increment=0.05,
            textvariable=self.emotion_weight_var, width=6,
            command=lambda: self._couple_weights("emotion"),
        )
        self.emotion_weight_spin.grid(row=srow, column=1, sticky="w", pady=2)
        self.emotion_weight_spin.bind("<Return>", lambda e: self._couple_weights("emotion"))
        self.emotion_weight_spin.bind("<FocusOut>", lambda e: self._couple_weights("emotion"))

        srow += 1
        self.aesthetic_weight_var = tk.DoubleVar(value=0.35)
        self.aesthetic_weight_label = ttk.Label(scoring_frame, text="Aesthetic:")
        self.aesthetic_weight_label.grid(row=srow, column=0, sticky="w", padx=(20, 5), pady=2)
        self.aesthetic_weight_spin = ttk.Spinbox(
            scoring_frame, from_=0.0, to=1.0, increment=0.05,
            textvariable=self.aesthetic_weight_var, width=6,
            command=lambda: self._couple_weights("aesthetic"),
        )
        self.aesthetic_weight_spin.grid(row=srow, column=1, sticky="w", pady=2)
        self.aesthetic_weight_spin.bind("<Return>", lambda e: self._couple_weights("aesthetic"))
        self.aesthetic_weight_spin.bind("<FocusOut>", lambda e: self._couple_weights("aesthetic"))

        srow += 1
        self.face_weight_var = tk.DoubleVar(value=0.15)
        self.face_weight_label = ttk.Label(scoring_frame, text="Face:")
        self.face_weight_label.grid(row=srow, column=0, sticky="w", padx=(20, 5), pady=2)
        self.face_weight_spin = ttk.Spinbox(
            scoring_frame, from_=0.0, to=1.0, increment=0.05,
            textvariable=self.face_weight_var, width=6,
            command=lambda: self._couple_weights("face"),
        )
        self.face_weight_spin.grid(row=srow, column=1, sticky="w", pady=2)
        self.face_weight_spin.bind("<Return>", lambda e: self._couple_weights("face"))
        self.face_weight_spin.bind("<FocusOut>", lambda e: self._couple_weights("face"))

        srow += 1
        self.consistency_weight_var = tk.DoubleVar(value=0.20)
        self.consistency_weight_label = ttk.Label(scoring_frame, text="Consistency:")
        self.consistency_weight_label.grid(row=srow, column=0, sticky="w", padx=(20, 5), pady=2)
        self.consistency_weight_spin = ttk.Spinbox(
            scoring_frame, from_=0.0, to=1.0, increment=0.05,
            textvariable=self.consistency_weight_var, width=6,
            command=lambda: self._couple_weights("consistency"),
        )
        self.consistency_weight_spin.grid(row=srow, column=1, sticky="w", pady=2)
        self.consistency_weight_spin.bind("<Return>", lambda e: self._couple_weights("consistency"))
        self.consistency_weight_spin.bind("<FocusOut>", lambda e: self._couple_weights("consistency"))

        self._update_scoring_ui()

        # --- Separator ---
        row += 1
        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8
        )

        # --- Phase label ---
        row += 1
        self.phase_var = tk.StringVar(value="")
        self.phase_label = ttk.Label(frame, textvariable=self.phase_var, font=("Segoe UI", 9, "bold"))
        self.phase_label.grid(row=row, column=0, columnspan=3, sticky="w", **pad)

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
        self.pause_btn = ttk.Button(btn_frame, text="⏸ Pause", command=self._toggle_pause)
        self.pause_btn.pack(side="left", padx=8)
        self.pause_btn.state(["disabled"])
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop", command=self._stop_filter)
        self.stop_btn.pack(side="left", padx=8)
        self.stop_btn.state(["disabled"])
        self.open_btn = ttk.Button(btn_frame, text="Open Output", command=self._open_output)
        self.open_btn.pack(side="left", padx=8)
        self.open_btn.state(["disabled"])

    def _set_running(self, running: bool):
        self.running = running
        state = ["disabled"] if running else ["!disabled"]
        self.browse_btn.state(state)
        self.run_btn.state(state)
        self.topn_spin.state(state)
        self.aesthetic_check.state(state)
        self.face_check.state(state)
        self.face_mode_combo.state(["disabled"] if running else ["readonly"])
        self.face_threshold_spin.state(state)
        self.min_aes_spin.state(state)
        self.exclude_tags_entry.state(state)
        self.emotion_weight_spin.state(state)
        self.aesthetic_weight_spin.state(state)
        self.face_weight_spin.state(state)
        self.consistency_check.state(state)
        self.reference_add_btn.state(state)
        self.reference_clear_btn.state(state)
        self.consistency_mode_combo.state(["disabled"] if running else ["readonly"])
        self.consistency_threshold_spin.state(state)
        self.consistency_weight_spin.state(state)
        if running:
            self.pause_btn.state(["!disabled"])
            self.stop_btn.state(["!disabled"])
        else:
            self.pause_btn.state(["disabled"])
            self.stop_btn.state(["disabled"])
            self.pause_btn.configure(text="⏸ Pause")
        if not running and self.output_dir and self.output_dir.exists():
            self.open_btn.state(["!disabled"])

    def _update_scoring_ui(self):
        """Show/hide controls and set weight defaults based on scorer toggles."""
        aes_on = self.aesthetic_var.get()
        face_on = self.face_var.get()
        face_mode = self.face_mode_var.get()
        face_weighted = face_on and face_mode == "Weighted"
        consistency_toggle_available = self.consistency_model_available
        consistency_ready = consistency_toggle_available and self.consistency_var.get() and bool(self.reference_paths)
        consistency_mode = self.consistency_mode_var.get()
        consistency_weighted = consistency_ready and consistency_mode == "Weighted"
        consistency_hard_filter = consistency_ready and consistency_mode == "Hard Filter"

        # Min aesthetic quality
        for w in (self.min_aes_label, self.min_aes_spin, self.min_aes_hint):
            if aes_on:
                w.grid()
            else:
                w.grid_remove()

        # Face sub-controls
        for w in (self.face_mode_label, self.face_mode_combo):
            if face_on:
                w.grid()
            else:
                w.grid_remove()

        show_threshold = face_on and face_mode == "Hard Filter"
        for w in (self.face_threshold_label, self.face_threshold_spin):
            if show_threshold:
                w.grid()
            else:
                w.grid_remove()

        # Weight section
        active_keys = self._get_active_weight_keys()
        need_weights = len(active_keys) > 1
        for w in (self.weight_sep, self.weight_section_label):
            if need_weights:
                w.grid()
            else:
                w.grid_remove()

        for w in (self.emotion_weight_label, self.emotion_weight_spin):
            if need_weights:
                w.grid()
            else:
                w.grid_remove()

        show_aes_weight = need_weights and aes_on
        for w in (self.aesthetic_weight_label, self.aesthetic_weight_spin):
            if show_aes_weight:
                w.grid()
            else:
                w.grid_remove()

        show_face_weight = face_weighted
        for w in (self.face_weight_label, self.face_weight_spin):
            if show_face_weight:
                w.grid()
            else:
                w.grid_remove()

        for w in (self.consistency_mode_label, self.consistency_mode_combo):
            if consistency_ready:
                w.grid()
            else:
                w.grid_remove()

        for w in (self.consistency_threshold_label, self.consistency_threshold_spin):
            if consistency_hard_filter:
                w.grid()
            else:
                w.grid_remove()

        show_consistency_weight = consistency_weighted or consistency_hard_filter
        for w in (self.consistency_weight_label, self.consistency_weight_spin):
            if show_consistency_weight:
                w.grid()
            else:
                w.grid_remove()

        if consistency_toggle_available:
            self.consistency_check.state(["!disabled"])
        else:
            self.consistency_check.state(["disabled"])

        consistency_controls_state = ["!disabled"] if consistency_ready or (self.consistency_var.get() and consistency_toggle_available) else ["disabled"]
        self.reference_add_btn.state(consistency_controls_state)
        self.reference_clear_btn.state(consistency_controls_state)

        # Set default weights
        self._updating_weights = True
        self._set_default_weights(active_keys)
        self._updating_weights = False

    def _couple_weights(self, changed: str):
        """Auto-adjust other weights to maintain sum = 1.0 on 0.05 grid."""
        if self._updating_weights:
            return
        self._updating_weights = True
        try:
            active_keys = self._get_active_weight_keys()
            if len(active_keys) <= 1:
                self._set_default_weights(active_keys)
                return

            if changed not in active_keys:
                changed = "emotion"

            weights = self._get_current_weight_values()
            # Snap changed value to 0.05 grid
            changed_value = round(max(0.0, min(1.0, weights[changed])) * 20) / 20
            other_keys = [key for key in active_keys if key != changed]

            if not other_keys:
                self._set_weight_values({changed: 1.0})
                return

            remainder = round(max(0.0, 1.0 - changed_value), 10)
            old_sum = sum(weights[key] for key in other_keys)

            # Determine proportional shares
            if old_sum > 0:
                shares = {key: weights[key] / old_sum for key in other_keys}
            else:
                defaults = self._default_weights_for_keys(active_keys)
                default_sum = sum(defaults[key] for key in other_keys)
                if default_sum > 0:
                    shares = {key: defaults[key] / default_sum for key in other_keys}
                else:
                    shares = {key: 1.0 / len(other_keys) for key in other_keys}

            # Distribute remainder in 0.05 units using largest-remainder method
            total_units = round(remainder * 20)
            raw_units = {key: shares[key] * total_units for key in other_keys}
            floored = {key: int(raw_units[key]) for key in other_keys}
            deficit = total_units - sum(floored.values())
            if deficit > 0:
                ranked = sorted(other_keys, key=lambda k: -(raw_units[k] - floored[k]))
                for i in range(int(deficit)):
                    floored[ranked[i]] += 1

            normalized = {changed: changed_value}
            for key in other_keys:
                normalized[key] = floored[key] * 0.05

            self._set_weight_values(normalized)
        except (tk.TclError, ValueError):
            pass
        finally:
            self._updating_weights = False

    def _get_active_weight_keys(self) -> list[str]:
        active = ["emotion"]
        if self.aesthetic_var.get():
            active.append("aesthetic")
        if self.face_var.get() and self.face_mode_var.get() == "Weighted":
            active.append("face")
        if (
            self.consistency_model_available
            and self.consistency_var.get()
            and self.reference_paths
            and self.consistency_mode_var.get() == "Weighted"
        ):
            active.append("consistency")
        return active

    def _default_weights_for_keys(self, keys: list[str]) -> dict[str, float]:
        preset_map: dict[tuple[str, ...], dict[str, float]] = {
            ("emotion",): {"emotion": 1.0},
            ("emotion", "aesthetic"): {"emotion": 0.65, "aesthetic": 0.35},
            ("emotion", "face"): {"emotion": 0.80, "face": 0.20},
            ("emotion", "consistency"): {"emotion": 0.60, "consistency": 0.40},
            ("emotion", "aesthetic", "face"): {"emotion": 0.55, "aesthetic": 0.30, "face": 0.15},
            ("emotion", "aesthetic", "consistency"): {"emotion": 0.50, "aesthetic": 0.25, "consistency": 0.25},
            ("emotion", "face", "consistency"): {"emotion": 0.60, "face": 0.15, "consistency": 0.25},
            ("emotion", "aesthetic", "face", "consistency"): {
                "emotion": 0.45,
                "aesthetic": 0.25,
                "face": 0.10,
                "consistency": 0.20,
            },
        }
        return preset_map.get(tuple(keys), {key: 1.0 / len(keys) for key in keys})

    def _set_default_weights(self, active_keys: list[str]):
        self._set_weight_values(self._default_weights_for_keys(active_keys))

    def _get_current_weight_values(self) -> dict[str, float]:
        return {
            "emotion": max(0.0, min(1.0, self.emotion_weight_var.get())),
            "aesthetic": max(0.0, min(1.0, self.aesthetic_weight_var.get())),
            "face": max(0.0, min(1.0, self.face_weight_var.get())),
            "consistency": max(0.0, min(1.0, self.consistency_weight_var.get())),
        }

    def _set_weight_values(self, weights: dict[str, float]):
        self.emotion_weight_var.set(round(weights.get("emotion", 0.0), 2))
        self.aesthetic_weight_var.set(round(weights.get("aesthetic", 0.0), 2))
        self.face_weight_var.set(round(weights.get("face", 0.0), 2))
        self.consistency_weight_var.set(round(weights.get("consistency", 0.0), 2))

    def _refresh_reference_preview(self):
        for child in self.reference_preview_frame.winfo_children():
            child.destroy()

        self.reference_thumbnails = []

        if not self.reference_paths:
            self.reference_status_var.set("No reference images selected.")
            self._update_scoring_ui()
            return

        self.reference_status_var.set(f"{len(self.reference_paths)} reference image(s) selected.")

        for index, path in enumerate(self.reference_paths):
            thumb_frame = ttk.Frame(self.reference_preview_frame)
            thumb_frame.grid(row=0, column=index, padx=4, pady=2, sticky="n")

            try:
                with Image.open(path) as img:
                    preview = img.convert("RGB")
                    preview.thumbnail((80, 80), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(preview)
                    self.reference_thumbnails.append(photo)
                    ttk.Label(thumb_frame, image=photo).grid(row=0, column=0, pady=(0, 2))
            except Exception as exc:
                logger.warning("Failed to build thumbnail for %s: %s", path, exc)
                ttk.Label(thumb_frame, text="Preview\nunavailable", justify="center").grid(row=0, column=0, pady=(0, 2))

            ttk.Label(thumb_frame, text=path.name, width=14, justify="center", wraplength=100).grid(
                row=1, column=0, pady=(0, 2)
            )
            ttk.Button(
                thumb_frame,
                text="Remove",
                command=lambda idx=index: self._remove_reference_image(idx),
            ).grid(row=2, column=0)

        self._update_scoring_ui()

    def _add_reference_images(self):
        selected = filedialog.askopenfilenames(
            title="Select reference images",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.webp"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("WebP", "*.webp"),
            ],
        )
        if not selected:
            return

        merged: list[Path] = list(self.reference_paths)
        for selected_path in [Path(path) for path in selected]:
            if selected_path not in merged:
                merged.append(selected_path)

        if len(merged) > 5:
            messagebox.showwarning("Reference Images", "You can select up to 5 reference images.")
            merged = merged[:5]

        self.reference_paths = merged
        self._refresh_reference_preview()

    def _remove_reference_image(self, index: int):
        if 0 <= index < len(self.reference_paths):
            del self.reference_paths[index]
            self._refresh_reference_preview()

    def _clear_reference_images(self):
        self.reference_paths = []
        self._refresh_reference_preview()

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
        self.phase_var.set("")
        self.emo_status_var.set("")
        self.progress_bar.configure(mode="determinate")
        self._stop_event.clear()
        self._pause_event.set()

        input_folder = Path(self.folder_var.get())
        base_output = input_folder.parent / f"{input_folder.name}_filtered"
        if (base_output / "report.json").exists():
            idx = 2
            while (input_folder.parent / f"{input_folder.name}_filtered_{idx}" / "report.json").exists():
                idx += 1
            self.output_dir = input_folder.parent / f"{input_folder.name}_filtered_{idx}"
        else:
            self.output_dir = base_output
        self.status_var.set("Reading EXIF tags from images...")

        face_on = self.face_var.get()
        face_mode_display = self.face_mode_var.get()
        face_mode = (
            {"Hard Filter": "hard_filter", "Weighted": "weighted"}.get(
                face_mode_display, "off"
            )
            if face_on
            else "off"
        )
        consistency_mode_display = self.consistency_mode_var.get()
        consistency_mode = (
            {"Hard Filter": "hard_filter", "Weighted": "weighted"}.get(
                consistency_mode_display, "weighted"
            )
            if self.consistency_model_available and self.consistency_var.get() and self.reference_paths
            else "off"
        )

        scoring_config = {
            "aesthetic_enabled": self.aesthetic_var.get(),
            "face_enabled": face_on,
            "face_mode": face_mode,
            "face_threshold": self.face_threshold_var.get(),
            "emotion_weight": self.emotion_weight_var.get(),
            "aesthetic_weight": self.aesthetic_weight_var.get(),
            "face_weight": (
                self.face_weight_var.get()
                if face_on and face_mode == "weighted"
                else 0.0
            ),
            "consistency_enabled": self.consistency_model_available and self.consistency_var.get() and bool(self.reference_paths),
            "consistency_weight": (
                self.consistency_weight_var.get()
                if consistency_mode == "weighted"
                else 0.0
            ),
            "consistency_mode": consistency_mode,
            "reference_paths": [Path(path) for path in self.reference_paths],
            "consistency_gate_threshold": self.consistency_threshold_var.get(),
            "consistency_penalty_power": 3.0,
            "min_aesthetic_quality": self.min_aes_var.get() if self.aesthetic_var.get() else 0.0,
            "exclude_tags": [t.strip() for t in self.exclude_tags_var.get().split(",") if t.strip()],
        }

        thread = threading.Thread(
            target=self._worker,
            args=(
                self.emotion_groups,
                self.topn_var.get(),
                self.output_dir,
                scoring_config,
            ),
            daemon=True,
        )
        thread.start()

    # Phase progress weight constants (must sum to 100)
    _PHASE_W_PREPARE = 10   # model download + loading
    _PHASE_W_ANALYZE = 90   # batch scoring + ranking + export

    def _worker(self, emotion_groups, top_n, output_dir, scoring_config):
        try:
            from PIL import Image
            from wd_scorer import CamieTaggerScorer, compute_combined_scores

            aes_enabled = scoring_config["aesthetic_enabled"]
            face_enabled = scoring_config["face_enabled"]
            face_mode = scoring_config["face_mode"]
            face_threshold = scoring_config["face_threshold"]
            emotion_weight = scoring_config["emotion_weight"]
            aesthetic_weight = scoring_config["aesthetic_weight"]
            face_weight = scoring_config["face_weight"]
            consistency_enabled = scoring_config["consistency_enabled"]
            consistency_weight = scoring_config["consistency_weight"]
            consistency_mode = scoring_config["consistency_mode"]
            reference_paths = scoring_config["reference_paths"]
            consistency_gate_threshold = scoring_config["consistency_gate_threshold"]
            consistency_penalty_power = scoring_config["consistency_penalty_power"]

            # ======== Phase 1/3: Model Preparation ========
            self.queue.put(("phase", "Step 1/2 — Preparing models"))
            self.queue.put(("progress", 0))

            self.queue.put(("status", "Reading EXIF tags from images..."))
            exif_tags = extract_exif_tags_by_emotion(emotion_groups)
            found = sum(1 for v in exif_tags.values() if v)
            self.queue.put(("status", f"EXIF tags found for {found}/{len(exif_tags)} emotions. Loading models..."))

            # Switch to indeterminate during model download
            self.queue.put(("progress_mode", "indeterminate"))

            def download_cb(msg_type, msg_data):
                self.queue.put((msg_type, msg_data))

            scorer = CamieTaggerScorer(download_callback=download_cb)

            aes_scorer = None
            if aes_enabled:
                try:
                    from aesthetic_scorer import AestheticScorer
                    self.queue.put(("status", "Loading Aesthetic Predictor V2.5..."))
                    aes_scorer = AestheticScorer()
                    aes_scorer.load_model(
                        progress_callback=lambda msg: self.queue.put(("status", msg))
                    )
                except Exception as e:
                    logger.warning("Aesthetic scoring unavailable: %s", e)
                    self.queue.put(("status", f"Aesthetic scoring unavailable: {e}"))
                    aes_scorer = None

            face_scorer_inst = None
            if face_enabled:
                try:
                    from face_scorer import FaceFramingScorer
                    face_scorer_inst = FaceFramingScorer()
                except Exception as e:
                    logger.warning("Face scoring unavailable: %s", e)
                    self.queue.put(("status", f"Face scoring unavailable: {e}"))

            consistency_scorer = None
            reference_embedding = None
            consistency_raw_scores: dict = {}
            consistency_normalization_stats: dict | None = None
            if consistency_enabled and reference_paths:
                try:
                    from consistency_scorer import ConsistencyScorer

                    self.queue.put(("status", "Loading DINOv2 consistency model..."))
                    consistency_scorer = ConsistencyScorer(cache_dir=os.environ.get("HF_HOME"))
                    consistency_scorer.load_model(
                        progress_callback=lambda msg: self.queue.put(("status", msg))
                    )
                    self.queue.put(("status", "Computing reference embedding..."))
                    reference_embedding = consistency_scorer.compute_reference_embedding(reference_paths)
                except Exception as e:
                    logger.warning("Consistency scoring unavailable: %s", e)
                    self.queue.put(("status", f"Consistency scoring unavailable: {e}"))
                    consistency_scorer = None
                    reference_embedding = None

            # Infer reference images through Camie Tagger for tag deviation filter
            reference_tag_profile = None
            if consistency_enabled and reference_paths:
                self.queue.put(("status", "Building reference tag profile..."))
                ref_items = []
                for rp in reference_paths:
                    try:
                        ref_img = Image.open(rp).convert("RGB")
                        ref_items.append((rp, ref_img))
                    except Exception as e:
                        logger.warning("Failed to load reference image %s: %s", rp, e)
                if ref_items:
                    scorer.infer_batch_pil(ref_items)
                    for _, img in ref_items:
                        img.close()
                    try:
                        reference_tag_profile = scorer.compute_reference_tag_profile(
                            [rp for rp, _ in ref_items]
                        )
                    except Exception as e:
                        logger.warning("Reference tag profile failed: %s", e)

            # Back to determinate, mark phase 1 done
            self.queue.put(("progress_mode", "determinate"))
            self.queue.put(("progress", self._PHASE_W_PREPARE))

            # ======== Phase 2/2: Image Analysis ========
            self.queue.put(("phase", "Step 2/2 — Analyzing images"))

            all_paths = [path for items in emotion_groups.values() for path, _ in items]
            total = len(all_paths)

            aesthetic_scores: dict = {}
            face_scores: dict = {}
            consistency_scores: dict = {}
            camie_batch_size = 16
            aes_batch_size = 16

            face_first = (
                face_scorer_inst is not None
                and face_mode == "hard_filter"
                and aes_scorer is not None
            )

            phase2_base = self._PHASE_W_PREPARE
            phase2_span = self._PHASE_W_ANALYZE

            for i in range(0, total, camie_batch_size):
                if self._stop_event.is_set():
                    self.queue.put(("cancelled", "Processing stopped by user."))
                    return
                self._pause_event.wait()

                chunk_paths = all_paths[i : i + camie_batch_size]

                pil_images: dict[Path, Image.Image] = {}
                for p in chunk_paths:
                    try:
                        pil_images[p] = Image.open(p).convert("RGB")
                    except Exception as e:
                        logger.warning("Failed to load %s: %s", p, e)

                batch_items = [(p, pil_images[p]) for p in chunk_paths if p in pil_images]
                try:
                    scorer.infer_batch_pil(batch_items)

                    consistency_candidates = list(chunk_paths)
                    if consistency_scorer is not None and reference_embedding is not None:
                        consistency_items = [(p, pil_images[p]) for p in chunk_paths if p in pil_images]
                        if consistency_items:
                            consistency_scores.update(
                                consistency_scorer.score_batch_pil(consistency_items, reference_embedding)
                            )

                    if face_scorer_inst is not None and face_first:
                        face_items = [(p, pil_images[p]) for p in consistency_candidates if p in pil_images]
                        if face_items:
                            face_scores.update(face_scorer_inst.score_batch_pil(face_items))

                    if aes_scorer is not None:
                        aes_candidates = list(consistency_candidates)
                        if face_first:
                            aes_candidates = [
                                p for p in aes_candidates
                                if face_scores.get(p, 0.0) >= face_threshold
                            ]
                        for j in range(0, len(aes_candidates), aes_batch_size):
                            sub_paths = aes_candidates[j : j + aes_batch_size]
                            sub_items = [(p, pil_images[p]) for p in sub_paths if p in pil_images]
                            if sub_items:
                                batch_result = aes_scorer.score_batch_pil(sub_items)
                                aesthetic_scores.update(batch_result)

                    if face_scorer_inst is not None and not face_first:
                        face_items = [(p, pil_images[p]) for p in consistency_candidates if p in pil_images]
                        if face_items:
                            face_scores.update(face_scorer_inst.score_batch_pil(face_items))
                finally:
                    for img in pil_images.values():
                        img.close()
                    del pil_images

                processed = min(i + camie_batch_size, total)
                pct = phase2_base + (processed / total) * phase2_span
                self.queue.put(("progress", pct))
                self.queue.put(("status", f"Analyzing images... {processed}/{total}"))

            if face_scorer_inst is not None:
                face_scorer_inst.close()
                face_scorer_inst = None

            if consistency_scorer is not None and consistency_scores:
                from consistency_scorer import normalize_score_map

                consistency_raw_scores = dict(consistency_scores)
                consistency_scores, consistency_normalization_stats = normalize_score_map(consistency_scores)
                if consistency_normalization_stats.get("collapsed"):
                    self.queue.put(("status", "Consistency scores were tightly collapsed; using raw scores fallback normalization."))
                else:
                    self.queue.put((
                        "status",
                        "Consistency normalized using global percentiles "
                        f"({consistency_normalization_stats['scale_min']:.4f} -> {consistency_normalization_stats['scale_max']:.4f}).",
                    ))

            self.queue.put(("status", "Ranking & exporting..."))

            # Save original groups for recovery pass
            original_emotion_groups = {emo: list(items) for emo, items in emotion_groups.items()}
            exclude_tag_excluded: set[Path] = set()

            # Global Aesthetic Floor
            min_aes = scoring_config.get("min_aesthetic_quality", 0.0)
            if aes_scorer is not None and min_aes > 1.0:
                before_count = sum(len(v) for v in emotion_groups.values())
                emotion_groups = {
                    emo: [(p, n) for p, n in items if aesthetic_scores.get(p, 0.0) >= min_aes]
                    for emo, items in emotion_groups.items()
                }
                emotion_groups = {k: v for k, v in emotion_groups.items() if v}
                after_count = sum(len(v) for v in emotion_groups.values())
                removed = before_count - after_count
                if removed > 0:
                    self.queue.put(("status", f"Quality floor: {removed} images below {min_aes:.1f} removed."))

            # Exclude tag filter (EXIF-aware)
            exclude_tags = scoring_config.get("exclude_tags", [])
            if exclude_tags:
                excluded_paths = scorer.get_excluded_paths(
                    exclude_tags, emotion_groups, exif_tags_by_emotion=exif_tags,
                )
                if excluded_paths:
                    exclude_tag_excluded = excluded_paths
                    before_count = sum(len(v) for v in emotion_groups.values())
                    emotion_groups = {
                        emo: [(p, n) for p, n in items if p not in excluded_paths]
                        for emo, items in emotion_groups.items()
                    }
                    emotion_groups = {k: v for k, v in emotion_groups.items() if v}
                    after_count = sum(len(v) for v in emotion_groups.values())
                    removed = before_count - after_count
                    self.queue.put(("status", f"Exclude tags: {removed} images removed ({', '.join(exclude_tags)})."))

            # Tag deviation auto-exclude (reference-based)
            tag_deviation_details: dict = {}
            tag_dev_excluded: set[Path] = set()
            if reference_tag_profile is not None:
                self.queue.put(("status", "Checking tag deviations against reference..."))
                tag_dev_excluded, tag_deviation_details = scorer.get_tag_deviation_excluded_paths(
                    reference_tag_profile, emotion_groups,
                )
                if tag_dev_excluded:
                    before_count = sum(len(v) for v in emotion_groups.values())
                    emotion_groups = {
                        emo: [(p, n) for p, n in items if p not in tag_dev_excluded]
                        for emo, items in emotion_groups.items()
                    }
                    emotion_groups = {k: v for k, v in emotion_groups.items() if v}
                    after_count = sum(len(v) for v in emotion_groups.values())
                    removed = before_count - after_count
                    self.queue.put(("status", f"Tag deviation filter: {removed} images removed."))

            scored = scorer.compute_emotion_scores(emotion_groups, exif_tags)

            score_meta = {}
            actual_face_mode = face_mode if face_scorer_inst is not None else "off"
            if aes_scorer is not None or face_scorer_inst is not None or consistency_scorer is not None:
                scored, score_meta = compute_combined_scores(
                    scored,
                    aesthetic_scores=aesthetic_scores if aes_scorer is not None else None,
                    face_scores=face_scores if face_scorer_inst is not None else None,
                    consistency_scores=consistency_scores if consistency_scorer is not None else None,
                    consistency_raw_scores=consistency_raw_scores if consistency_scorer is not None else None,
                    emotion_weight=emotion_weight,
                    aesthetic_weight=aesthetic_weight,
                    face_mode=actual_face_mode,
                    face_threshold=face_threshold,
                    face_weight=face_weight,
                    consistency_mode=consistency_mode,
                    consistency_weight=consistency_weight,
                    consistency_gate_threshold=consistency_gate_threshold,
                    consistency_penalty_power=consistency_penalty_power,
                )

            # --- Recovery pass for deficit emotions ---
            if top_n > 1:
                deficit_emotions: dict[str, int] = {}
                for emo in original_emotion_groups:
                    current = len(scored.get(emo, []))
                    if current < top_n:
                        deficit_emotions[emo] = top_n - current

                if deficit_emotions:
                    # Build recovery groups: original candidates minus exclude-tag-excluded
                    # and tag-deviation-excluded paths, minus already-scored paths
                    recovery_groups: dict[str, list[tuple[Path, int]]] = {}
                    for emo, needed in deficit_emotions.items():
                        existing_paths = {item["path"] for item in scored.get(emo, [])}
                        recovery = [
                            (p, n) for p, n in original_emotion_groups[emo]
                            if p not in existing_paths
                            and p not in exclude_tag_excluded
                            and p not in tag_dev_excluded
                        ]
                        if recovery:
                            recovery_groups[emo] = recovery

                    if recovery_groups:
                        logger.info(
                            "Recovery pass: %d emotions need more candidates (total deficit: %d)",
                            len(recovery_groups),
                            sum(deficit_emotions[e] for e in recovery_groups),
                        )
                        recovery_scored = scorer.compute_emotion_scores(recovery_groups, exif_tags)

                        # Re-score with relaxed settings: hard_filter → weighted
                        if aes_scorer is not None or face_scorer_inst is not None or consistency_scorer is not None:
                            recovery_face_mode = "weighted" if actual_face_mode == "hard_filter" else actual_face_mode
                            recovery_consistency_mode = "weighted" if consistency_mode == "hard_filter" else consistency_mode
                            recovery_scored, _recovery_meta = compute_combined_scores(
                                recovery_scored,
                                aesthetic_scores=aesthetic_scores if aes_scorer is not None else None,
                                face_scores=face_scores if face_scorer_inst is not None else None,
                                consistency_scores=consistency_scores if consistency_scorer is not None else None,
                                consistency_raw_scores=consistency_raw_scores if consistency_scorer is not None else None,
                                emotion_weight=emotion_weight,
                                aesthetic_weight=aesthetic_weight,
                                face_mode=recovery_face_mode,
                                face_threshold=face_threshold,
                                face_weight=face_weight,
                                consistency_mode=recovery_consistency_mode,
                                consistency_weight=consistency_weight,
                                consistency_gate_threshold=consistency_gate_threshold,
                                consistency_penalty_power=consistency_penalty_power,
                            )

                        # Merge recovery into scored (existing items first, fill deficit)
                        recovered_total = 0
                        for emo in recovery_groups:
                            existing = scored.get(emo, [])
                            existing_paths = {item["path"] for item in existing}
                            recovery_items = [
                                item for item in recovery_scored.get(emo, [])
                                if item["path"] not in existing_paths
                            ]
                            for item in recovery_items:
                                item["recovered_from_filter"] = True
                            needed = deficit_emotions[emo]
                            fill = recovery_items[:needed]
                            if fill:
                                merged = existing + fill
                                merged.sort(key=lambda x: x["score"], reverse=True)
                                scored[emo] = merged
                                recovered_total += len(fill)
                                # Update meta
                                if emo not in score_meta:
                                    score_meta[emo] = {}
                                score_meta[emo]["recovery_filled"] = len(fill)

                        if recovered_total > 0:
                            logger.info("Recovery pass: %d images recovered across %d emotions",
                                        recovered_total, len([e for e in recovery_groups if score_meta.get(e, {}).get("recovery_filled")]))
                            self.queue.put(("status", f"Recovery pass: {recovered_total} images recovered for deficit emotions."))

            self.queue.put(("status", "Copying files..."))
            output_dir.mkdir(parents=True, exist_ok=True)
            total_copied = filter_and_copy(scored, top_n, output_dir)

            config = {"top_n": top_n, "model": "Camie-Tagger-v2"}
            if aes_scorer is not None:
                config["aesthetic_model"] = "aesthetic-predictor-v2-5"
                config["emotion_weight"] = emotion_weight
                config["aesthetic_weight"] = aesthetic_weight
            if face_scorer_inst is not None:
                config["face_framing_enabled"] = True
                config["face_framing_mode"] = face_mode
                if face_mode == "hard_filter":
                    config["face_threshold"] = face_threshold
                elif face_mode == "weighted":
                    config["face_weight"] = face_weight
            if consistency_scorer is not None:
                config["consistency_enabled"] = True
                config["consistency_model"] = consistency_scorer.model_name
                config["consistency_cache_dir"] = str(consistency_scorer.cache_dir)
                config["consistency_mode"] = consistency_mode
                config["consistency_weight"] = consistency_weight
                config["consistency_gate_threshold"] = consistency_gate_threshold
                config["consistency_penalty_power"] = consistency_penalty_power
                config["reference_images"] = [str(path) for path in reference_paths]
                if consistency_normalization_stats is not None:
                    config["consistency_normalization"] = {
                        "method": "global_percentile_minmax",
                        **consistency_normalization_stats,
                    }
            if reference_tag_profile is not None:
                config["tag_deviation_filter"] = {
                    "enabled": True,
                    "candidate_threshold": 0.7,
                    "deviation_threshold": 0.3,
                    "images_excluded": len(tag_deviation_details),
                }

            # Release model resources
            if hasattr(scorer, 'session'):
                del scorer.session
            del scorer
            if aes_scorer is not None:
                del aes_scorer.model, aes_scorer.preprocessor
                del aes_scorer
            if consistency_scorer is not None:
                del consistency_scorer.model, consistency_scorer.processor
                del consistency_scorer

            generate_report(scored, top_n, config, output_dir, score_meta=score_meta)

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
                elif msg_type == "phase":
                    self.phase_var.set(msg_data)
                elif msg_type == "progress_mode":
                    self.progress_bar.configure(mode=msg_data)
                    if msg_data == "indeterminate":
                        self.progress_bar.start(15)
                    else:
                        self.progress_bar.stop()
                elif msg_type == "status":
                    self.status_var.set(msg_data)
                elif msg_type == "emo_status":
                    self.emo_status_var.set(msg_data)
                elif msg_type == "done":
                    self._set_running(False)
                    self.progress_bar.stop()
                    self.progress_bar.configure(mode="determinate")
                    self.phase_var.set("Complete")
                    self.status_var.set(msg_data)
                    self.emo_status_var.set("")
                    messagebox.showinfo("Complete", msg_data)
                elif msg_type == "cancelled":
                    self._set_running(False)
                    self.progress_bar.stop()
                    self.progress_bar.configure(mode="determinate")
                    self.phase_var.set("Stopped")
                    self.status_var.set(msg_data)
                    self.emo_status_var.set("")
                elif msg_type == "error":
                    self._set_running(False)
                    self.progress_bar.stop()
                    self.progress_bar.configure(mode="determinate")
                    self.phase_var.set("Error")
                    self.status_var.set(f"Error: {msg_data}")
                    self.status_label.configure(foreground="red")
        except queue.Empty:
            pass

        self.root.after(100, self._poll_queue)

    def _toggle_pause(self):
        if self._pause_event.is_set():
            self._pause_event.clear()
            self.pause_btn.configure(text="▶ Resume")
            self.status_var.set("Paused.")
        else:
            self._pause_event.set()
            self.pause_btn.configure(text="⏸ Pause")
            self.status_var.set("Resumed.")

    def _stop_filter(self):
        self._stop_event.set()
        self._pause_event.set()  # unblock if paused
        self.stop_btn.state(["disabled"])
        self.pause_btn.state(["disabled"])

    def _open_output(self):
        if self.output_dir and self.output_dir.exists():
            os.startfile(str(self.output_dir))
