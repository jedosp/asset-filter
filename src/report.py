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
            for meta_key in (
                "filtered_by_face",
                "fallback_auxiliary_only",
                "penalized_by_consistency_gate",
                "filtered_by_consistency",
                "hard_filter_fallback_used",
                "recovery_filled",
            ):
                if meta_key in em_meta:
                    emotion_data[meta_key] = em_meta[meta_key]

        scores_list = []
        for item in items:
            entry: dict[str, Any] = {"filename": item["path"].name}
            if "emotion_score" in item:
                entry["emotion_score"] = round(item["emotion_score"], 4)
            if "aesthetic_score" in item:
                entry["aesthetic_score"] = round(item["aesthetic_score"], 2)
            if "face_score" in item:
                entry["face_score"] = round(item["face_score"], 4)
            if "consistency_score" in item:
                entry["consistency_score"] = round(item["consistency_score"], 4)
            if "consistency_raw_score" in item:
                entry["consistency_raw_score"] = round(item["consistency_raw_score"], 4)
            if "neg_score" in item and item["neg_score"] > 0:
                entry["neg_score"] = round(item["neg_score"], 4)
            if "combined_score" in item:
                entry["combined_score"] = round(item["combined_score"], 4)
            else:
                entry["score"] = round(item["score"], 4)
            if item.get("recovered_from_filter"):
                entry["recovered_from_filter"] = True
            if item.get("hard_filter_fallback"):
                entry["hard_filter_fallback"] = True
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
