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

# Appearance tags for tag deviation filter — exact matches
_APPEARANCE_TAGS = frozenset([
    # Eyewear
    "glasses", "sunglasses", "goggles", "eyepatch", "monocle",
    "semi-rimless_eyewear", "rimless_eyewear", "round_eyewear",
    # Head accessories
    "hat", "beret", "hood", "helmet", "crown", "tiara",
    "headband", "hairband", "headphones", "earphones",
    "hair_ornament", "hairclip", "hair_ribbon", "hair_bow",
    "hair_flower", "hair_tie", "scrunchie", "hairpin", "multiple_hairpins",
    # Body accessories
    "earrings", "necklace", "choker", "pendant",
    "bracelet", "gloves", "scarf", "necktie", "bowtie", "piercing",
    # Hair style
    "bob_cut", "ponytail", "side_ponytail", "twintails", "twin_braids",
    "braid", "single_braid", "french_braid",
    "hair_bun", "double_bun", "low_twintails", "high_ponytail",
    "low_ponytail", "drill_hair", "ringlets", "hime_cut",
    "pixie_cut", "afro", "mohawk", "undercut",
    "short_hair", "long_hair", "medium_hair", "very_long_hair",
    "absurdly_long_hair", "hair_over_one_eye", "hair_over_eyes",
    "sidelocks", "ahoge", "blunt_bangs", "swept_bangs",
    "parted_bangs", "hair_between_eyes",
    "messy_hair", "wavy_hair", "curly_hair", "straight_hair",
    "hair_up", "hair_down", "tied_hair",
    # Hair color
    "black_hair", "brown_hair", "blonde_hair", "red_hair",
    "blue_hair", "green_hair", "pink_hair", "purple_hair",
    "white_hair", "grey_hair", "silver_hair", "orange_hair",
    "aqua_hair", "multicolored_hair", "gradient_hair",
    "streaked_hair", "two-tone_hair", "colored_inner_hair",
    # Eye anomalies
    "heterochromia",
    # Non-human features
    "horns", "wings", "tail", "halo",
    "animal_ears", "cat_ears", "dog_ears", "fox_ears", "rabbit_ears",
    "fang", "fangs", "elf_ears", "pointy_ears",
])

# Suffix patterns — any general tag ending with these is appearance-related
_APPEARANCE_TAG_SUFFIXES = (
    "_hair_ornament", "_earrings", "_hairband", "_choker",
    "_necklace", "_glasses", "_eyewear", "_hairclip",
)


def _is_appearance_tag(tag: str) -> bool:
    """Check if a tag is appearance-related (accessories, hair, eyes, etc.)."""
    if tag in _APPEARANCE_TAGS:
        return True
    return any(tag.endswith(s) for s in _APPEARANCE_TAG_SUFFIXES)


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
                        neg_score = float(np.max([probs[idx] for idx in self.negative_indices]))
                else:
                    score = 0.0
                scored_items.append({"path": path, "score": score, "number": number, "neg_score": neg_score})

            scored_items.sort(key=lambda x: x["score"], reverse=True)
            results[emotion] = scored_items

        return results

    def compute_reference_tag_profile(
        self,
        reference_paths: list[Path],
    ) -> np.ndarray:
        """Compute mean tag probabilities across reference images.

        Returns a 1-D array of shape (n_tags,) with the average probability
        for every tag in the vocabulary.  Only ``general`` category tags are
        used later for deviation checks, but we store the full vector so the
        caller can slice as needed.
        """
        ref_probs_list: list[np.ndarray] = []
        for path in reference_paths:
            probs = self.image_probs.get(path)
            if probs is not None:
                ref_probs_list.append(probs)
            else:
                logger.warning("Reference image %s has no tag probs (not inferred yet).", path)
        if not ref_probs_list:
            raise ValueError("No reference images have been inferred by Camie Tagger.")
        profile = np.mean(ref_probs_list, axis=0).astype(np.float32)
        logger.info(
            "Reference tag profile computed from %d image(s).",
            len(ref_probs_list),
        )
        return profile

    def get_tag_deviation_excluded_paths(
        self,
        reference_profile: np.ndarray,
        emotion_groups: dict[str, list[tuple[Path, int]]],
        candidate_threshold: float = 0.7,
        deviation_threshold: float = 0.3,
    ) -> tuple[set[Path], dict[Path, list[tuple[str, float, float]]]]:
        """Exclude images where an appearance tag is prominently present but absent in references.

        Only appearance-related ``general`` category tags are checked
        (accessories, hair, eyes, non-human features).  Expression effects
        (squiggle, sweatdrop) and meta tags (virtual_youtuber) are ignored.

        A tag triggers exclusion when **both** conditions are met:
        1. candidate probability >= *candidate_threshold* (visually prominent)
        2. (candidate prob - reference avg) >= *deviation_threshold* (not in reference)

        Returns (excluded_paths, details) where *details* maps each excluded
        path to a list of (tag, candidate_prob, deviation) tuples.
        """
        # Build index mask for appearance-related general tags only
        appearance_mask = np.zeros(len(self.tag_names), dtype=bool)
        for tag, idx in self.tag_to_index.items():
            if self.tag_to_category.get(tag) == "general" and _is_appearance_tag(tag):
                appearance_mask[idx] = True

        excluded: set[Path] = set()
        details: dict[Path, list[tuple[str, float, float]]] = {}

        for _emotion, items in emotion_groups.items():
            for path, _ in items:
                probs = self.image_probs.get(path)
                if probs is None:
                    continue

                deviations = probs - reference_profile
                # Conditions: appearance tag, high prob, high positive deviation
                hits = np.where(
                    appearance_mask
                    & (probs >= candidate_threshold)
                    & (deviations >= deviation_threshold)
                )[0]

                if hits.size > 0:
                    tag_details = [
                        (self.tag_names[idx], float(probs[idx]), float(deviations[idx]))
                        for idx in hits
                    ]
                    excluded.add(path)
                    details[path] = tag_details
                    logger.info(
                        "Tag deviation excluding %s: %s",
                        path.name,
                        ", ".join(f"{t}={p:.3f}(+{d:.3f})" for t, p, d in tag_details),
                    )

        logger.info(
            "Tag deviation filter: %d images excluded out of %d total.",
            len(excluded),
            sum(len(items) for items in emotion_groups.values()),
        )
        return excluded, details

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
    consistency_scores: dict[Path, float] | None = None,
    consistency_raw_scores: dict[Path, float] | None = None,
    emotion_weight: float = 0.65,
    aesthetic_weight: float = 0.35,
    face_mode: str = "off",
    face_threshold: float = 0.3,
    face_weight: float = 0.15,
    consistency_mode: str = "weighted",
    consistency_weight: float = 0.20,
    consistency_gate_threshold: float = 0.35,
    consistency_penalty_power: float = 3.0,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """
    Combine emotion, aesthetic, face, and consistency scores into a unified ranking.

    Returns (scored_results, meta) where meta has per-emotion metadata
    such as filtered_by_face count.
    """
    results: dict[str, list[dict[str, Any]]] = {}
    meta: dict[str, dict[str, Any]] = {}

    for emotion, items in emotion_scores.items():
        new_items = []
        hard_filtered_items = []  # items rejected by hard filter, kept for fallback
        filtered_count = 0
        consistency_filtered_count = 0

        # Detect if emotion scoring is effectively dead (all zeros)
        all_emotion_zero = all(item["score"] == 0.0 for item in items)
        has_aesthetic = aesthetic_scores is not None
        has_face = face_mode == "weighted" and face_scores is not None
        has_consistency = consistency_mode in ("weighted", "hard_filter") and consistency_scores is not None

        if all_emotion_zero and (has_aesthetic or has_face or has_consistency):
            eff_emotion_w = 0.0
            eff_aesthetic_w = aesthetic_weight if has_aesthetic else 0.0
            eff_face_w = face_weight if has_face else 0.0
            eff_consistency_w = consistency_weight if has_consistency else 0.0
            aux_total = eff_aesthetic_w + eff_face_w + eff_consistency_w
            if aux_total > 0:
                eff_aesthetic_w /= aux_total
                eff_face_w /= aux_total
                eff_consistency_w /= aux_total
            logger.info("Emotion '%s': no Camie tag match, falling back to auxiliary ranking", emotion)
        else:
            eff_emotion_w = emotion_weight
            eff_aesthetic_w = aesthetic_weight
            eff_face_w = face_weight
            eff_consistency_w = consistency_weight

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

            c_score = None
            if consistency_scores is not None:
                c_score = consistency_scores.get(path, 0.0)
                new_item["consistency_score"] = c_score
                if consistency_raw_scores is not None:
                    new_item["consistency_raw_score"] = consistency_raw_scores.get(path, c_score)

            # Hard filter: discard images below threshold
            hard_rejected = False
            if face_mode == "hard_filter" and f_score is not None:
                if f_score < face_threshold:
                    filtered_count += 1
                    hard_rejected = True
            if not hard_rejected and consistency_mode == "hard_filter" and c_score is not None:
                if c_score < consistency_gate_threshold:
                    consistency_filtered_count += 1
                    hard_rejected = True

            # Compute combined score
            combined = em_score * eff_emotion_w
            if aes_norm is not None:
                combined += aes_norm * eff_aesthetic_w
            if face_mode == "weighted" and f_score is not None:
                combined += f_score * eff_face_w
            if has_consistency and c_score is not None:
                combined += c_score * eff_consistency_w
                if consistency_mode == "weighted" and c_score < consistency_gate_threshold:
                    consistency_filtered_count += 1
                    penalty_ratio = max(0.0, c_score / consistency_gate_threshold)
                    combined *= penalty_ratio ** consistency_penalty_power

            # Apply negative tag penalty at combined level
            neg = item.get("neg_score", 0.0)
            if neg > 0:
                combined = combined * (1.0 - neg)

            new_item["combined_score"] = combined
            new_item["score"] = combined

            if hard_rejected:
                hard_filtered_items.append(new_item)
            else:
                new_items.append(new_item)

        # Fallback: if hard filter removed ALL candidates, pick the best rejected one
        if not new_items and hard_filtered_items:
            hard_filtered_items.sort(key=lambda x: x["score"], reverse=True)
            best = hard_filtered_items[0]
            best["hard_filter_fallback"] = True
            new_items.append(best)
            logger.info(
                "Emotion '%s': hard filter removed all %d candidates, "
                "keeping best fallback (score=%.4f)",
                emotion, len(hard_filtered_items), best["score"],
            )

        new_items.sort(key=lambda x: x["score"], reverse=True)
        results[emotion] = new_items

        emotion_meta: dict[str, Any] = {}
        if filtered_count > 0:
            emotion_meta["filtered_by_face"] = filtered_count
        if consistency_filtered_count > 0 and consistency_mode == "weighted":
            emotion_meta["penalized_by_consistency_gate"] = consistency_filtered_count
        if consistency_filtered_count > 0 and consistency_mode == "hard_filter":
            emotion_meta["filtered_by_consistency"] = consistency_filtered_count
        if all_emotion_zero and (has_aesthetic or has_face or has_consistency):
            emotion_meta["fallback_auxiliary_only"] = True
        if not new_items or (len(new_items) == 1 and new_items[0].get("hard_filter_fallback")):
            emotion_meta["hard_filter_fallback_used"] = True
        meta[emotion] = emotion_meta

    return results, meta
