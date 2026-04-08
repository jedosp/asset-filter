"""Filename parsing and folder scanning."""

import logging
from pathlib import Path

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
        prompts[emotion] = prompt

    return prompts
