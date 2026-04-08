"""Filename parsing, folder scanning, and EXIF tag extraction."""

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


def extract_exif_tags(image_path: Path) -> list[str]:
    """
    Extract character tags from NovelAI PNG metadata (char_captions).

    Returns list of individual tags, e.g.:
    ["girl", "tsuyuri_kumin", "school_uniform", "brown_hair"]

    Spaces are converted to underscores to match WD Tagger vocabulary.
    """
    try:
        img = Image.open(image_path)
        info = img.info

        if "Comment" in info:
            comment = json.loads(info["Comment"])
            if isinstance(comment, dict):
                char_captions = comment.get("char_captions")
                if char_captions and isinstance(char_captions, list):
                    tags = []
                    for c in char_captions:
                        if isinstance(c, dict) and c.get("char_caption"):
                            for part in c["char_caption"].split(","):
                                tag = part.strip().replace(" ", "_").lower()
                                if tag:
                                    tags.append(tag)
                    return tags
    except Exception as e:
        logger.warning("Failed to read EXIF from %s: %s", image_path, e)

    return []


def extract_exif_tags_by_emotion(
    emotion_groups: dict[str, list[tuple[Path, int]]],
) -> dict[str, list[str]]:
    """
    For each emotion group, extract EXIF tags from the first image.

    Returns: {"acting coy": ["girl", "tsuyuri_kumin", "school_uniform", ...], ...}
    """
    result: dict[str, list[str]] = {}

    for emotion, items in emotion_groups.items():
        tags = []
        for path, _ in items:
            tags = extract_exif_tags(path)
            if tags:
                break
        result[emotion] = tags

    return result
