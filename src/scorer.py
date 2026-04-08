"""CLIP scoring engine for emotion images."""

import logging
from pathlib import Path
from typing import Any, Callable

import open_clip
import torch
from PIL import Image

logger = logging.getLogger(__name__)

MODELS = {
    "ViT-B-32": {"pretrained": "laion2b_s34b_b79k"},
    "ViT-L-14": {"pretrained": "laion2b_s32b_b82k"},
}


class CLIPScorer:
    def __init__(self, model_name: str = "ViT-B-32"):
        pretrained = MODELS[model_name]["pretrained"]
        self.device = "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def score_batch(self, image_paths: list[Path], text: str) -> list[float]:
        """Compute cosine similarity between images and a text prompt."""
        text_tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        scores: list[float] = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    img_features = self.model.encode_image(img_tensor)
                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                    sim = (img_features @ text_features.T).item()
                scores.append(float(sim))
            except Exception as e:
                logger.warning("Failed to process image %s: %s", p, e)
                scores.append(0.0)

        return scores

    def score_all(
        self,
        emotion_groups: dict[str, list[tuple[Path, int]]],
        prompt_template: str,
        batch_size: int = 16,
        progress_callback: Callable[[str, int, int, int, int], None] | None = None,
        exif_prompts: dict[str, str | None] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Score all images across all emotions.

        progress_callback(emotion, img_current, img_total, emo_current, emo_total)
        exif_prompts: {emotion: prompt_from_exif} — if provided and non-None for
                      an emotion, uses the EXIF prompt instead of the template.
        """
        results: dict[str, list[dict[str, Any]]] = {}
        emotions = sorted(emotion_groups.keys())
        emo_total = len(emotions)

        for emo_idx, emotion in enumerate(emotions, 1):
            items = emotion_groups[emotion]

            # Use EXIF prompt if available, otherwise fall back to template
            exif_prompt = exif_prompts.get(emotion) if exif_prompts else None
            if exif_prompt:
                prompt = exif_prompt
                logger.info("Using EXIF prompt for '%s': %.80s...", emotion, prompt)
            else:
                prompt = prompt_template.format(emotion=emotion)

            paths = [p for p, _ in items]
            numbers = [n for _, n in items]
            img_total = len(paths)

            all_scores: list[float] = []
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i : i + batch_size]
                batch_scores = self.score_batch(batch_paths, prompt)
                all_scores.extend(batch_scores)

                if progress_callback:
                    progress_callback(
                        emotion,
                        min(i + batch_size, img_total),
                        img_total,
                        emo_idx,
                        emo_total,
                    )

            scored_items = [
                {"path": p, "score": s, "number": n}
                for p, s, n in zip(paths, all_scores, numbers)
            ]
            scored_items.sort(key=lambda x: x["score"], reverse=True)
            results[emotion] = scored_items

        return results
