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
    3. score_batch_pil(): Score pre-loaded PIL images → predict scores (1-10)
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
