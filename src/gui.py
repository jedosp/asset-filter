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
        self.emotion_weight_spin.state(state)
        self.aesthetic_weight_spin.state(state)
        self.face_weight_spin.state(state)
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
        need_weights = aes_on or face_weighted
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

        # Set default weights
        self._updating_weights = True
        if aes_on and face_weighted:
            self.emotion_weight_var.set(0.55)
            self.aesthetic_weight_var.set(0.30)
            self.face_weight_var.set(0.15)
        elif aes_on:
            self.emotion_weight_var.set(0.65)
            self.aesthetic_weight_var.set(0.35)
            self.face_weight_var.set(0.0)
        elif face_weighted:
            self.emotion_weight_var.set(0.80)
            self.aesthetic_weight_var.set(0.0)
            self.face_weight_var.set(0.20)
        else:
            self.emotion_weight_var.set(1.0)
            self.aesthetic_weight_var.set(0.0)
            self.face_weight_var.set(0.0)
        self._updating_weights = False

    def _couple_weights(self, changed: str):
        """Auto-adjust other weights to maintain sum = 1.0."""
        if self._updating_weights:
            return
        self._updating_weights = True
        try:
            aes_on = self.aesthetic_var.get()
            face_weighted = self.face_var.get() and self.face_mode_var.get() == "Weighted"

            em = max(0.0, min(1.0, self.emotion_weight_var.get()))
            ae = max(0.0, min(1.0, self.aesthetic_weight_var.get()))
            fa = max(0.0, min(1.0, self.face_weight_var.get()))

            if aes_on and face_weighted:
                # 3-weight coupling
                if changed == "emotion":
                    remainder = 1.0 - em
                    old_sum = ae + fa
                    if old_sum > 0:
                        ae = remainder * (ae / old_sum)
                        fa = remainder * (fa / old_sum)
                    else:
                        ae = remainder * 0.67
                        fa = remainder * 0.33
                elif changed == "aesthetic":
                    remainder = 1.0 - ae
                    old_sum = em + fa
                    if old_sum > 0:
                        em = remainder * (em / old_sum)
                        fa = remainder * (fa / old_sum)
                    else:
                        em = remainder * 0.8
                        fa = remainder * 0.2
                else:
                    remainder = 1.0 - fa
                    old_sum = em + ae
                    if old_sum > 0:
                        em = remainder * (em / old_sum)
                        ae = remainder * (ae / old_sum)
                    else:
                        em = remainder * 0.65
                        ae = remainder * 0.35
                self.emotion_weight_var.set(round(em, 2))
                self.aesthetic_weight_var.set(round(ae, 2))
                self.face_weight_var.set(round(fa, 2))

            elif aes_on:
                # 2-weight: emotion + aesthetic
                if changed == "emotion":
                    self.emotion_weight_var.set(round(em, 2))
                    self.aesthetic_weight_var.set(round(1.0 - em, 2))
                else:
                    self.aesthetic_weight_var.set(round(ae, 2))
                    self.emotion_weight_var.set(round(1.0 - ae, 2))

            elif face_weighted:
                # 2-weight: emotion + face
                if changed == "emotion":
                    self.emotion_weight_var.set(round(em, 2))
                    self.face_weight_var.set(round(1.0 - em, 2))
                else:
                    self.face_weight_var.set(round(fa, 2))
                    self.emotion_weight_var.set(round(1.0 - fa, 2))
        except (tk.TclError, ValueError):
            pass
        finally:
            self._updating_weights = False

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
        self.output_dir = input_folder.parent / f"{input_folder.name}_filtered"
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
            "min_aesthetic_quality": self.min_aes_var.get() if self.aesthetic_var.get() else 0.0,
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
    _PHASE_W_ANALYZE = 85   # batch scoring loop
    _PHASE_W_FINALIZE = 5   # quality floor + ranking + copy

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

            # ======== Phase 1/3: Model Preparation ========
            self.queue.put(("phase", "Step 1/3 — Preparing models"))
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

            face_scorer_inst = None
            if face_enabled:
                try:
                    from face_scorer import FaceFramingScorer
                    face_scorer_inst = FaceFramingScorer()
                except Exception as e:
                    logger.warning("Face scoring unavailable: %s", e)
                    self.queue.put(("status", f"Face scoring unavailable: {e}"))

            # Back to determinate, mark phase 1 done
            self.queue.put(("progress_mode", "determinate"))
            self.queue.put(("progress", self._PHASE_W_PREPARE))

            # ======== Phase 2/3: Image Analysis ========
            self.queue.put(("phase", "Step 2/3 — Analyzing images"))

            all_paths = [path for items in emotion_groups.values() for path, _ in items]
            total = len(all_paths)

            aesthetic_scores: dict = {}
            face_scores: dict = {}
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
                scorer.infer_batch_pil(batch_items)

                if face_scorer_inst is not None and face_first:
                    face_items = [(p, pil_images[p]) for p in chunk_paths if p in pil_images]
                    if face_items:
                        face_scores.update(face_scorer_inst.score_batch_pil(face_items))

                if aes_scorer is not None:
                    aes_candidates = list(chunk_paths)
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
                    face_items = [(p, pil_images[p]) for p in chunk_paths if p in pil_images]
                    if face_items:
                        face_scores.update(face_scorer_inst.score_batch_pil(face_items))

                for img in pil_images.values():
                    img.close()
                del pil_images

                processed = min(i + camie_batch_size, total)
                pct = phase2_base + (processed / total) * phase2_span
                self.queue.put(("progress", pct))
                self.queue.put(("status", f"Analyzing images... {processed}/{total}"))

            if face_scorer_inst is not None:
                face_scorer_inst.close()

            # ======== Phase 3/3: Ranking & Export ========
            self.queue.put(("phase", "Step 3/3 — Ranking & exporting"))
            self.queue.put(("progress", phase2_base + phase2_span))

            # Global Aesthetic Floor
            min_aes = scoring_config.get("min_aesthetic_quality", 0.0)
            if aes_scorer is not None and min_aes > 1.0:
                before_count = sum(len(v) for v in emotion_groups.values())
                emotion_groups = {
                    emo: [(p, n) for p, n in items if aesthetic_scores.get(p, min_aes) >= min_aes]
                    for emo, items in emotion_groups.items()
                }
                emotion_groups = {k: v for k, v in emotion_groups.items() if v}
                after_count = sum(len(v) for v in emotion_groups.values())
                removed = before_count - after_count
                if removed > 0:
                    self.queue.put(("status", f"Quality floor: {removed} images below {min_aes:.1f} removed."))

            scored = scorer.compute_emotion_scores(emotion_groups, exif_tags)

            score_meta = {}
            if aes_scorer is not None or face_scorer_inst is not None:
                actual_face_mode = face_mode if face_scorer_inst is not None else "off"
                scored, score_meta = compute_combined_scores(
                    scored,
                    aesthetic_scores=aesthetic_scores if aes_scorer is not None else None,
                    face_scores=face_scores if face_scorer_inst is not None else None,
                    emotion_weight=emotion_weight,
                    aesthetic_weight=aesthetic_weight,
                    face_mode=actual_face_mode,
                    face_threshold=face_threshold,
                    face_weight=face_weight,
                )

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
