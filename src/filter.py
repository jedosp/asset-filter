"""Filtering, sorting, and file copy logic."""

import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def filter_and_copy(
    scored: dict[str, list[dict[str, Any]]],
    top_n: int,
    output_dir: Path,
    min_score: float = 0.0,
) -> int:
    """
    For each emotion, take top N images by score and copy to output subfolder.

    Returns total number of images copied.
    """
    total_copied = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for emotion, items in sorted(scored.items()):
        selected = [item for item in items[:top_n] if item["score"] >= min_score]
        if not selected:
            logger.warning("No images above min_score for emotion: %s", emotion)
            continue

        for item in selected:
            src = item["path"]
            dst = output_dir / src.name
            try:
                shutil.copy2(src, dst)
                total_copied += 1
            except OSError as e:
                logger.error("Failed to copy %s: %s", src, e)

    return total_copied
