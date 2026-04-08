"""WD Tagger v3 scoring engine for emotion images."""

import csv
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_REPO = "SmilingWolf/wd-vit-tagger-v3"
MODEL_FILENAME = "model.onnx"
TAGS_FILENAME = "selected_tags.csv"
IMAGE_SIZE = 448


class WDTaggerScorer:
    def __init__(self):
        import onnxruntime as ort

        logger.info("Downloading WD Tagger v3 model...")
        model_path = hf_hub_download(MODEL_REPO, MODEL_FILENAME)
        tags_path = hf_hub_download(MODEL_REPO, TAGS_FILENAME)

        logger.info("Loading ONNX model...")
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        # Load tag vocabulary
        self.tag_names: list[str] = []
        self.tag_to_index: dict[str, int] = {}
        with open(tags_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                tag = row["name"]
                self.tag_names.append(tag)
                self.tag_to_index[tag] = i

        logger.info("Loaded %d tags from vocabulary.", len(self.tag_names))

    @staticmethod
    def _preprocess(image_path: Path) -> np.ndarray:
        """Preprocess image for WD Tagger: RGBA→RGB on white, pad to square, resize, BGR."""
        img = Image.open(image_path).convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img)
        img = bg.convert("RGB")

        # Pad to square
        max_dim = max(img.size)
        canvas = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        canvas.paste(img, ((max_dim - img.width) // 2, (max_dim - img.height) // 2))
        img = canvas.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)

        arr = np.array(img, dtype=np.float32)
        arr = arr[:, :, ::-1]  # RGB → BGR
        return arr

    def _find_tag_indices(self, emotion: str) -> list[int]:
        """
        Find tag indices for an emotion keyword.

        Tries: exact match (space→underscore) → individual words.
        """
        tag = emotion.replace(" ", "_").lower()
        if tag in self.tag_to_index:
            return [self.tag_to_index[tag]]

        # Split into individual words and match each
        indices = []
        for word in emotion.lower().split():
            word_tag = word.replace(" ", "_")
            if word_tag in self.tag_to_index:
                indices.append(self.tag_to_index[word_tag])
        return indices

    def score_all(
        self,
        emotion_groups: dict[str, list[tuple[Path, int]]],
        batch_size: int = 16,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Score all images in a single inference pass, then assign per-emotion scores.

        WD Tagger classifies all tags at once, so we only need one pass
        through every image regardless of how many emotions there are.

        progress_callback(current_image, total_images)
        """
        emotions = sorted(emotion_groups.keys())

        # Pre-resolve tag indices
        emotion_tag_indices: dict[str, list[int]] = {}
        for emotion in emotions:
            indices = self._find_tag_indices(emotion)
            if not indices:
                logger.warning("No matching WD tag for emotion: '%s'", emotion)
            else:
                matched = [self.tag_names[i] for i in indices]
                logger.info("Emotion '%s' → WD tags: %s", emotion, matched)
            emotion_tag_indices[emotion] = indices

        # Flatten all images for single-pass inference
        all_items: list[tuple[str, Path, int]] = []
        for emotion in emotions:
            for path, number in emotion_groups[emotion]:
                all_items.append((emotion, path, number))

        total_images = len(all_items)
        image_probs: dict[Path, np.ndarray] = {}

        for i in range(0, total_images, batch_size):
            batch = all_items[i : i + batch_size]
            batch_arrays = []
            for _, p, _ in batch:
                try:
                    arr = self._preprocess(p)
                except Exception as e:
                    logger.warning("Failed to preprocess %s: %s", p, e)
                    arr = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255.0, dtype=np.float32)
                batch_arrays.append(arr)

            batch_input = np.stack(batch_arrays)
            raw = self.session.run(None, {self.input_name: batch_input})[0]

            # Apply sigmoid if model outputs raw logits
            if raw.max() > 1.0 or raw.min() < 0.0:
                raw = 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))

            for j, (_, p, _) in enumerate(batch):
                image_probs[p] = raw[j]

            if progress_callback:
                progress_callback(min(i + batch_size, total_images), total_images)

        # Assign scores per emotion from cached probabilities
        results: dict[str, list[dict[str, Any]]] = {}
        for emotion in emotions:
            tag_indices = emotion_tag_indices[emotion]
            scored_items = []
            for path, number in emotion_groups[emotion]:
                probs = image_probs.get(path)
                if probs is not None and tag_indices:
                    score = float(np.mean([probs[idx] for idx in tag_indices]))
                else:
                    score = 0.0
                scored_items.append({"path": path, "score": score, "number": number})

            scored_items.sort(key=lambda x: x["score"], reverse=True)
            results[emotion] = scored_items

        return results
