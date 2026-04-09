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
    score_meta: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Write report.json with full scoring results."""
    total_images = sum(len(items) for items in scored.values())
    total_selected = 0

    emotions_detail: dict[str, Any] = {}
    for emotion, items in sorted(scored.items()):
        selected_count = min(top_n, len(items))
        total_selected += selected_count

        emotion_data: dict[str, Any] = {
            "total": len(items),
            "selected": selected_count,
        }

        # Add face filter metadata if present
        if score_meta and emotion in score_meta:
            em_meta = score_meta[emotion]
            if "filtered_by_face" in em_meta:
                emotion_data["filtered_by_face"] = em_meta["filtered_by_face"]

        scores_list = []
        for item in items:
            entry: dict[str, Any] = {"filename": item["path"].name}
            if "emotion_score" in item:
                entry["emotion_score"] = round(item["emotion_score"], 4)
            if "aesthetic_score" in item:
                entry["aesthetic_score"] = round(item["aesthetic_score"], 2)
            if "face_score" in item:
                entry["face_score"] = round(item["face_score"], 4)
            if "combined_score" in item:
                entry["combined_score"] = round(item["combined_score"], 4)
            else:
                entry["score"] = round(item["score"], 4)
            scores_list.append(entry)

        emotion_data["scores"] = scores_list
        emotions_detail[emotion] = emotion_data

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
