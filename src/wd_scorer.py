"""Camie Tagger v2 scoring engine for emotion images."""

import json
import logging
from pathlib import Path
from typing import Any

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

        self.image_probs: dict[Path, np.ndarray] = {}

    @staticmethod
    def _preprocess_pil(img: Image.Image) -> np.ndarray:
        """Preprocess PIL image for Camie Tagger: pad to square, resize 512, ImageNet normalize."""
        img = img.convert("RGB")

        w, h = img.size
        if w > h:
            new_w, new_h = IMAGE_SIZE, int(IMAGE_SIZE * h / w)
        else:
            new_w, new_h = int(IMAGE_SIZE * w / h), IMAGE_SIZE
        img = img.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), PAD_COLOR)
        canvas.paste(img, ((IMAGE_SIZE - new_w) // 2, (IMAGE_SIZE - new_h) // 2))

        arr = np.array(canvas, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
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

    def infer_batch_pil(self, items: list[tuple[Path, Image.Image]]) -> None:
        """Run inference on pre-loaded PIL images, accumulate results in image_probs."""
        if not items:
            return
        batch_arrays = []
        paths = []
        for path, img in items:
            try:
                arr = self._preprocess_pil(img)
            except Exception as e:
                logger.warning("Failed to preprocess %s: %s", path, e)
                arr = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            batch_arrays.append(arr)
            paths.append(path)

        batch_input = np.stack(batch_arrays)
        outputs = self.session.run(None, {self.input_name: batch_input})

        initial = outputs[0]
        if len(outputs) >= 2:
            refined = outputs[1]
            raw = np.maximum(initial, refined)
        else:
            raw = initial
        raw = 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))

        for j, path in enumerate(paths):
            self.image_probs[path] = raw[j]

    def compute_emotion_scores(
        self,
        emotion_groups: dict[str, list[tuple[Path, int]]],
        exif_tags: dict[str, list[str]] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Compute per-emotion scores from accumulated image_probs."""
        emotions = sorted(emotion_groups.keys())

        emotion_tag_indices: dict[str, list[int]] = {}
        for emotion in emotions:
            indices = self._find_tag_indices(emotion)
            if not indices:
                logger.warning("No matching tag for emotion: '%s'", emotion)
            else:
                matched = [self.tag_names[i] for i in indices]
                logger.info("Emotion '%s' → tags: %s", emotion, matched)
            emotion_tag_indices[emotion] = indices

        exif_tag_indices: dict[str, list[int]] = {}
        if exif_tags:
            for emotion in emotions:
                emotion_tag = emotion.replace(" ", "_").lower()
                indices = []
                tags_used = []
                for tag in exif_tags.get(emotion, []):
                    if tag == emotion_tag:
                        continue
                    if tag in self.tag_to_index:
                        indices.append(self.tag_to_index[tag])
                        tags_used.append(tag)
                exif_tag_indices[emotion] = indices
                if tags_used:
                    logger.info("Emotion '%s' EXIF tags: %s", emotion, tags_used)
                else:
                    logger.info("Emotion '%s': no matching EXIF tags in vocabulary", emotion)

        results: dict[str, list[dict[str, Any]]] = {}
        for emotion in emotions:
            tag_indices = emotion_tag_indices[emotion]
            exif_indices = exif_tag_indices.get(emotion, []) if exif_tags else []
            scored_items = []
            for path, number in emotion_groups[emotion]:
                probs = self.image_probs.get(path)
                neg_score = 0.0
                if probs is not None and tag_indices:
                    emotion_score = float(np.mean([probs[idx] for idx in tag_indices]))
                    if exif_indices:
                        exif_score = float(np.mean([probs[idx] for idx in exif_indices]))
                        score = 0.5 * emotion_score + 0.5 * exif_score
                    else:
                        score = emotion_score
                    if self.negative_indices:
                        neg_score = float(np.mean([probs[idx] for idx in self.negative_indices]))
                else:
                    score = 0.0
                scored_items.append({"path": path, "score": score, "number": number, "neg_score": neg_score})

            scored_items.sort(key=lambda x: x["score"], reverse=True)
            results[emotion] = scored_items

        return results

    def get_excluded_paths(
        self,
        exclude_tags: list[str],
        emotion_groups: dict[str, list[tuple[Path, int]]],
        exif_tags_by_emotion: dict[str, list[str]] | None = None,
        threshold: float = 0.5,
    ) -> set[Path]:
        """Return paths where an exclude tag is detected but NOT in EXIF (unintended).

        If EXIF contains the exclude tag, the feature is intentional → skip.
        If EXIF does not contain the exclude tag and Camie detects it → exclude.
        """
        exclude_indices: dict[str, int] = {}
        for tag in exclude_tags:
            tag_clean = tag.strip().replace(" ", "_").lower()
            if tag_clean in self.tag_to_index:
                exclude_indices[tag_clean] = self.tag_to_index[tag_clean]
        if not exclude_indices:
            logger.warning("No exclude tags matched in vocabulary: %s", exclude_tags)
            return set()
        logger.info("Exclude tag indices resolved: %s", list(exclude_indices.keys()))

        excluded = set()
        for emotion, items in emotion_groups.items():
            # Get EXIF tags for this emotion group
            emo_exif = set(exif_tags_by_emotion.get(emotion, [])) if exif_tags_by_emotion else set()

            # Determine which exclude tags to check (only those NOT in EXIF)
            active_tags = {
                tag: idx for tag, idx in exclude_indices.items()
                if tag not in emo_exif
            }
            if not active_tags:
                continue

            for path, _ in items:
                probs = self.image_probs.get(path)
                if probs is None:
                    continue
                for tag, idx in active_tags.items():
                    if probs[idx] > threshold:
                        logger.info("Excluding %s: '%s' detected (%.3f > %.3f), not in EXIF",
                                    path.name, tag, probs[idx], threshold)
                        excluded.add(path)
                        break
        return excluded


def compute_combined_scores(
    emotion_scores: dict[str, list[dict[str, Any]]],
    aesthetic_scores: dict[Path, float] | None = None,
    face_scores: dict[Path, float] | None = None,
    emotion_weight: float = 0.65,
    aesthetic_weight: float = 0.35,
    face_mode: str = "off",
    face_threshold: float = 0.3,
    face_weight: float = 0.15,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """
    Combine emotion, aesthetic, and face scores into a unified ranking.

    Returns (scored_results, meta) where meta has per-emotion metadata
    such as filtered_by_face count.
    """
    results: dict[str, list[dict[str, Any]]] = {}
    meta: dict[str, dict[str, Any]] = {}

    for emotion, items in emotion_scores.items():
        new_items = []
        filtered_count = 0

        # Detect if emotion scoring is effectively dead (all zeros)
        all_emotion_zero = all(item["score"] == 0.0 for item in items)
        if all_emotion_zero and aesthetic_scores is not None:
            # Fallback: use aesthetic 100% when Camie can't score this emotion
            eff_emotion_w = 0.0
            eff_aesthetic_w = 1.0
            eff_face_w = 0.0
            logger.info("Emotion '%s': no Camie tag match, falling back to aesthetic-only ranking", emotion)
        else:
            eff_emotion_w = emotion_weight
            eff_aesthetic_w = aesthetic_weight
            eff_face_w = face_weight

        for item in items:
            path = item["path"]
            em_score = item["score"]

            new_item = {
                "path": path,
                "number": item["number"],
                "emotion_score": em_score,
            }

            # Aesthetic score (normalize 1-10 → 0-1)
            aes_norm = None
            if aesthetic_scores is not None:
                aes_raw = aesthetic_scores.get(path, 0.0)
                new_item["aesthetic_score"] = aes_raw
                aes_norm = max(0.0, min(1.0, (aes_raw - 1.0) / 9.0))

            # Face score
            f_score = None
            if face_scores is not None:
                f_score = face_scores.get(path, 0.3)
                new_item["face_score"] = f_score

            # Hard filter: discard images below threshold
            if face_mode == "hard_filter" and f_score is not None:
                if f_score < face_threshold:
                    filtered_count += 1
                    continue

            # Compute combined score
            if aes_norm is not None and face_mode == "weighted" and f_score is not None:
                combined = (
                    em_score * eff_emotion_w
                    + aes_norm * eff_aesthetic_w
                    + f_score * eff_face_w
                )
            elif aes_norm is not None:
                combined = em_score * eff_emotion_w + aes_norm * eff_aesthetic_w
            elif face_mode == "weighted" and f_score is not None:
                combined = em_score * eff_emotion_w + f_score * eff_face_w
            else:
                combined = em_score

            # Apply negative tag penalty at combined level
            neg = item.get("neg_score", 0.0)
            if neg > 0:
                combined = combined * (1.0 - neg)

            new_item["combined_score"] = combined
            new_item["score"] = combined
            new_items.append(new_item)

        new_items.sort(key=lambda x: x["score"], reverse=True)
        results[emotion] = new_items

        emotion_meta: dict[str, Any] = {}
        if filtered_count > 0:
            emotion_meta["filtered_by_face"] = filtered_count
        if all_emotion_zero and aesthetic_scores is not None:
            emotion_meta["fallback_aesthetic_only"] = True
        meta[emotion] = emotion_meta

    return results, meta
