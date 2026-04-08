"""JSON report generation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def generate_report(
    scored: dict[str, list[dict[str, Any]]],
    top_n: int,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write report.json with full scoring results."""
    total_images = sum(len(items) for items in scored.values())
    total_selected = 0

    emotions_detail: dict[str, Any] = {}
    for emotion, items in sorted(scored.items()):
        selected_count = min(top_n, len(items))
        total_selected += selected_count
        emotions_detail[emotion] = {
            "total": len(items),
            "selected": selected_count,
            "scores": [
                {"filename": item["path"].name, "score": round(item["score"], 4)}
                for item in items
            ],
        }

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "summary": {
            "total_images": total_images,
            "emotions_found": len(scored),
            "images_selected": total_selected,
        },
        "emotions": emotions_detail,
    }

    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
