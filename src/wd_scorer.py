"""Camie Tagger v2 scoring engine for emotion images."""

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_REPO = "Camais03/camie-tagger-v2"
MODEL_FILENAME = "camie-tagger-v2.onnx"
METADATA_FILENAME = "camie-tagger-v2-metadata.json"
IMAGE_SIZE = 512

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Padding color derived from ImageNet mean (int RGB)
PAD_COLOR = (124, 116, 104)

# Tags indicating anatomical defects or low quality
NEGATIVE_TAGS = [
    "bad_anatomy", "bad_hands", "bad_proportions", "bad_feet",
    "extra_fingers", "fewer_fingers", "extra_limbs", "missing_fingers",
    "extra_arms", "extra_legs", "missing_arms", "missing_legs",
    "fused_fingers", "too_many_fingers",
    "mutated_hands", "mutation", "deformed",
    "ugly", "blurry", "anatomical_nonsense",
    "extra_digit", "fewer_digits",
]


class CamieTaggerScorer:
    def __init__(self, download_callback=None):
        import onnxruntime as ort

        def _notify(status=None, progress=None):
            if download_callback:
                if status is not None:
                    download_callback("status", status)
                if progress is not None:
                    download_callback("progress", progress)

        # Check if model is already cached
        needs_download = False
        try:
            hf_hub_download(MODEL_REPO, MODEL_FILENAME, local_files_only=True)
        except Exception:
            needs_download = True

        if needs_download:
            _notify(status="Downloading metadata...", progress=5)
            metadata_path = hf_hub_download(MODEL_REPO, METADATA_FILENAME)
            _notify(status="Downloading model (~789MB), please wait...", progress=10)
            model_path = hf_hub_download(MODEL_REPO, MODEL_FILENAME)
            _notify(progress=90)
        else:
            _notify(status="Loading model from cache...")
            model_path = hf_hub_download(MODEL_REPO, MODEL_FILENAME)
            metadata_path = hf_hub_download(MODEL_REPO, METADATA_FILENAME)

        _notify(status="Initializing ONNX runtime...", progress=95)
        providers = []
        available = ort.get_available_providers()
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(model_path, providers=providers)
        logger.info("ONNX providers: %s", self.session.get_providers())
        self.input_name = self.session.get_inputs()[0].name
        _notify(progress=100)

        # Load tag vocabulary from metadata JSON
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        dataset_info = metadata["dataset_info"]
        tag_mapping = dataset_info["tag_mapping"]
        idx_to_tag = tag_mapping["idx_to_tag"]  # str(idx) → tag_name
        self.tag_to_category: dict[str, str] = tag_mapping["tag_to_category"]

        # Build ordered tag list and reverse lookup
        self.tag_names: list[str] = []
        self.tag_to_index: dict[str, int] = {}
        for i in range(len(idx_to_tag)):
            tag = idx_to_tag[str(i)]
            self.tag_names.append(tag)
            self.tag_to_index[tag] = i

        logger.info("Loaded %d tags from vocabulary.", len(self.tag_names))

        # Pre-resolve negative tag indices
        self.negative_indices: list[int] = []
        matched_neg = []
        for tag in NEGATIVE_TAGS:
            if tag in self.tag_to_index:
                self.negative_indices.append(self.tag_to_index[tag])
                matched_neg.append(tag)
        logger.info("Negative tags matched: %d/%d (%s)", len(matched_neg), len(NEGATIVE_TAGS), matched_neg)

    @staticmethod
    def _preprocess(image_path: Path) -> np.ndarray:
        """Preprocess image for Camie Tagger: RGBA→RGB, pad to square, resize 512, ImageNet normalize."""
        img = Image.open(image_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        else:
            img = img.convert("RGB")

        # Resize maintaining aspect ratio
        w, h = img.size
        if w > h:
            new_w, new_h = IMAGE_SIZE, int(IMAGE_SIZE * h / w)
        else:
            new_w, new_h = int(IMAGE_SIZE * w / h), IMAGE_SIZE
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Pad to square with ImageNet mean color
        canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), PAD_COLOR)
        canvas.paste(img, ((IMAGE_SIZE - new_w) // 2, (IMAGE_SIZE - new_h) // 2))

        # Convert to float32 [0, 1], then apply ImageNet normalization
        arr = np.array(canvas, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        # HWC → CHW
        arr = arr.transpose(2, 0, 1)
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
        exif_tags: dict[str, list[str]] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Score all images in a single inference pass, then assign per-emotion scores.

        If exif_tags is provided, the final score is a weighted combination:
          score = 0.5 * emotion_score + 0.5 * exif_tag_score

        progress_callback(current_image, total_images)
        """
        emotions = sorted(emotion_groups.keys())

        # Pre-resolve emotion tag indices
        emotion_tag_indices: dict[str, list[int]] = {}
        for emotion in emotions:
            indices = self._find_tag_indices(emotion)
            if not indices:
                logger.warning("No matching tag for emotion: '%s'", emotion)
            else:
                matched = [self.tag_names[i] for i in indices]
                logger.info("Emotion '%s' → tags: %s", emotion, matched)
            emotion_tag_indices[emotion] = indices

        # Pre-resolve EXIF tag indices per emotion (excluding emotion tag itself)
        exif_tag_indices: dict[str, list[int]] = {}
        if exif_tags:
            for emotion in emotions:
                emotion_tag = emotion.replace(" ", "_").lower()
                indices = []
                tags_used = []
                for tag in exif_tags.get(emotion, []):
                    if tag == emotion_tag:
                        continue  # emotion tag is already scored separately
                    if tag in self.tag_to_index:
                        indices.append(self.tag_to_index[tag])
                        tags_used.append(tag)
                exif_tag_indices[emotion] = indices
                if tags_used:
                    logger.info("Emotion '%s' EXIF tags: %s", emotion, tags_used)
                else:
                    logger.info("Emotion '%s': no matching EXIF tags in vocabulary", emotion)

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
                    arr = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
                batch_arrays.append(arr)

            batch_input = np.stack(batch_arrays)
            outputs = self.session.run(None, {self.input_name: batch_input})

            # Camie v2 outputs: initial(70527), refined(70527), candidates(256)
            # refined only updates top-256 candidates; unselected tags stay near 0.
            # Use element-wise max of initial and refined for full coverage.
            initial = outputs[0]
            if len(outputs) >= 2:
                refined = outputs[1]
                raw = np.maximum(initial, refined)
            else:
                raw = initial

            # Apply sigmoid (model outputs raw logits)
            raw = 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))

            for j, (_, p, _) in enumerate(batch):
                image_probs[p] = raw[j]

            if progress_callback:
                progress_callback(min(i + batch_size, total_images), total_images)

        # Assign scores per emotion from cached probabilities
        results: dict[str, list[dict[str, Any]]] = {}
        for emotion in emotions:
            tag_indices = emotion_tag_indices[emotion]
            exif_indices = exif_tag_indices.get(emotion, []) if exif_tags else []
            scored_items = []
            for path, number in emotion_groups[emotion]:
                probs = image_probs.get(path)
                if probs is not None and tag_indices:
                    emotion_score = float(np.mean([probs[idx] for idx in tag_indices]))
                    if exif_indices:
                        exif_score = float(np.mean([probs[idx] for idx in exif_indices]))
                        score = 0.5 * emotion_score + 0.5 * exif_score
                    else:
                        score = emotion_score

                    # Apply negative tag penalty
                    if self.negative_indices:
                        neg_score = float(np.mean([probs[idx] for idx in self.negative_indices]))
                        score = score * (1.0 - neg_score)
                else:
                    score = 0.0
                scored_items.append({"path": path, "score": score, "number": number})

            scored_items.sort(key=lambda x: x["score"], reverse=True)
            results[emotion] = scored_items

        return results
