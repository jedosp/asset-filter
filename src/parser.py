"""Filename parsing, folder scanning, and PNG metadata extraction."""

import json
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def parse_filename(filename: str) -> tuple[str, str, int] | None:
    """
    Parse emotion image filename.

    "gabriel.acting coy.14.png" → ("gabriel", "acting coy", 14)

    1. Remove extension (.png)
    2. rsplit('.', 1) → extract number from last segment
    3. split('.', 1) on remainder → character name, emotion keyword
    """
    stem = Path(filename).stem
    parts = stem.rsplit(".", 1)
    if len(parts) != 2:
        return None

    remainder, num_str = parts
    if not num_str.isdigit():
        return None

    char_parts = remainder.split(".", 1)
    if len(char_parts) != 2:
        return None

    character, emotion = char_parts
    if not character or not emotion:
        return None

    return character, emotion, int(num_str)


def scan_folder(folder: Path) -> dict[str, list[tuple[Path, int]]]:
    """
    Scan a folder and group images by emotion keyword.

    Returns: {
        "acting coy": [(Path("gabriel.acting coy.1.png"), 1), ...],
        "angry": [(Path("gabriel.angry.1.png"), 1), ...],
    }
    """
    groups: dict[str, list[tuple[Path, int]]] = {}

    for p in sorted(folder.glob("*.png")):
        result = parse_filename(p.name)
        if result is None:
            logger.warning("Failed to parse filename: %s", p.name)
            continue

        _character, emotion, number = result
        groups.setdefault(emotion, []).append((p, number))

    return groups


def extract_prompt(image_path: Path) -> str | None:
    """
    Extract generation prompt from NovelAI PNG metadata.

    NovelAI stores prompt in PNG text chunks:
    - 'Description' field: the main prompt
    - 'Comment' field: JSON with generation parameters
    - 'parameters' field: alternative location

    Returns the prompt string, or None if not found.
    """
    try:
        img = Image.open(image_path)
        info = img.info

        # Try Comment field first for char_captions (character-only prompt)
        if "Comment" in info:
            try:
                comment = json.loads(info["Comment"])
                if isinstance(comment, dict):
                    char_captions = comment.get("char_captions")
                    if char_captions and isinstance(char_captions, list):
                        captions = [
                            c["char_caption"].strip()
                            for c in char_captions
                            if isinstance(c, dict) and c.get("char_caption")
                        ]
                        if captions:
                            return ", ".join(captions)
                    if "prompt" in comment:
                        return comment["prompt"].strip()
            except (json.JSONDecodeError, TypeError):
                pass

        # NovelAI primary: 'Description' chunk
        if "Description" in info:
            return info["Description"].strip()

        # Alternative: 'parameters' chunk (common in other generators)
        if "parameters" in info:
            text = info["parameters"]
            # Often "prompt\nNegative prompt: ..." format
            return text.split("\n")[0].strip()

    except Exception as e:
        logger.warning("Failed to read metadata from %s: %s", image_path, e)

    return None


def extract_prompts_by_emotion(
    emotion_groups: dict[str, list[tuple[Path, int]]],
) -> dict[str, str | None]:
    """
    For each emotion group, extract the prompt from the first image.

    Returns: {"acting coy": "1girl, acting coy, ...", ...}
    """
    prompts: dict[str, str | None] = {}

    for emotion, items in emotion_groups.items():
        prompt = None
        for path, _ in items:
            prompt = extract_prompt(path)
            if prompt:
                break
        prompts[emotion] = prompt

    return prompts
