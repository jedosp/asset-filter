"""Aesthetic Predictor V2.5 scoring engine (SigLIP-based)."""

import logging
import os
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class AestheticScorer:
    """
    Wraps aesthetic-predictor-v2-5 for batch scoring.

    Lifecycle:
    1. __init__(): Set cache dir, no model loading yet
    2. load_model(): Downloads (if needed) and loads SigLIP + MLP head
    3. score_batch(): Preprocess images → extract features → predict scores
    4. score_all(): Score all images with progress callback
    """

    def __init__(self, cache_dir: str | None = None):
        if cache_dir:
            os.environ.setdefault("TORCH_HOME", cache_dir)
            os.environ.setdefault("HF_HOME", cache_dir)
        self.model = None
        self.preprocessor = None
        self.device = "cpu"

    def load_model(self, progress_callback: Callable[[str], None] | None = None):
        """Load SigLIP + MLP aesthetic head. Downloads on first run (~900MB)."""
        if progress_callback:
            progress_callback("Downloading Aesthetic Predictor model (~3.5GB, first run only)...")

        import torch
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

        self.model, self.preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()

        # Move to GPU if available
        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = self.model.to("cuda")
            logger.info("Aesthetic Predictor using CUDA GPU: %s", torch.cuda.get_device_name(0))
        else:
            self.device = "cpu"
            logger.info("Aesthetic Predictor using CPU (no CUDA GPU found).")

        if progress_callback:
            progress_callback(f"Aesthetic Predictor loaded ({self.device}).")

    def score_batch(self, image_paths: list[Path]) -> list[float]:
        """Score a batch of images. Returns list of scores (1-10 scale)."""
        import torch
        from PIL import Image

        images = []
        valid_indices = []

        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_indices.append(i)
            except Exception as e:
                logger.warning("Failed to load image %s: %s", path, e)

        result = [0.0] * len(image_paths)

        if not images:
            return result

        try:
            inputs = self.preprocessor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.no_grad():
                logits = self.model(pixel_values).logits.squeeze(-1).float()
            batch_scores = logits.cpu().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
        except Exception as e:
            logger.warning("Batch inference failed: %s", e)
            return result

        for idx, score in zip(valid_indices, batch_scores):
            result[idx] = max(1.0, min(10.0, score))

        return result

    def score_batch_pil(self, items: list[tuple]) -> dict[Path, float]:
        """Score a batch of pre-loaded PIL images. items: [(Path, PIL.Image), ...]"""
        import torch

        results: dict[Path, float] = {}
        images = []
        paths = []

        for path, img in items:
            try:
                images.append(img.convert("RGB"))
                paths.append(path)
            except Exception as e:
                logger.warning("Failed to process image %s: %s", path, e)
                results[path] = 0.0

        if not images:
            return results

        try:
            inputs = self.preprocessor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.no_grad():
                logits = self.model(pixel_values).logits.squeeze(-1).float()
            batch_scores = logits.cpu().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            for path, score in zip(paths, batch_scores):
                results[path] = max(1.0, min(10.0, score))
        except Exception as e:
            logger.warning("Batch inference failed: %s", e)
            for path in paths:
                results[path] = 0.0

        return results

    def score_all(
        self,
        image_paths: list[Path],
        batch_size: int = 4,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[Path, float]:
        """Score all images. Returns {Path: score} mapping."""
        results = {}
        total = len(image_paths)

        for i in range(0, total, batch_size):
            batch = image_paths[i : i + batch_size]
            scores = self.score_batch(batch)
            for path, score in zip(batch, scores):
                results[path] = score
            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        return results
