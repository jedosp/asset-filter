"""Reference-based visual consistency scoring using DINOv2."""

import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "facebook/dinov2-base"


def _resolve_cache_dir(cache_dir: str | None = None) -> Path:
    if cache_dir:
        return Path(cache_dir)

    app_dir = os.environ.get("ASSET_FILTER_APP_DIR")
    if app_dir:
        return Path(app_dir) / "models" / "huggingface"

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)

    return Path("models") / "huggingface"


def normalize_score_map(
    score_map: dict[Path, float],
    lower_percentile: float = 10.0,
    upper_percentile: float = 90.0,
) -> tuple[dict[Path, float], dict[str, float | bool]]:
    """Spread compressed cosine scores onto a more usable 0-1 scale."""
    if not score_map:
        return {}, {
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "raw_min": 0.0,
            "raw_max": 0.0,
            "scale_min": 0.0,
            "scale_max": 0.0,
            "collapsed": True,
        }

    paths = list(score_map.keys())
    raw_scores = np.asarray([score_map[path] for path in paths], dtype=np.float32)

    scale_min = float(np.percentile(raw_scores, lower_percentile))
    scale_max = float(np.percentile(raw_scores, upper_percentile))
    collapsed = False
    if scale_max - scale_min < 1e-6:
        scale_min = float(raw_scores.min())
        scale_max = float(raw_scores.max())
    if scale_max - scale_min < 1e-6:
        collapsed = True
        normalized = {path: 1.0 for path in paths}
    else:
        normalized = {}
        scale = scale_max - scale_min
        for path in paths:
            score = (score_map[path] - scale_min) / scale
            normalized[path] = float(np.clip(score, 0.0, 1.0))

    stats: dict[str, float | bool] = {
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
        "raw_min": float(raw_scores.min()),
        "raw_max": float(raw_scores.max()),
        "scale_min": scale_min,
        "scale_max": scale_max,
        "collapsed": collapsed,
    }
    return normalized, stats

class ConsistencyScorer:
    """Scores candidate images against user-provided reference images."""

    def __init__(
        self,
        cache_dir: str | None = None,
        model_name: str = DEFAULT_MODEL_ID,
    ):
        if cache_dir:
            os.environ.setdefault("TORCH_HOME", cache_dir)
            os.environ.setdefault("HF_HOME", cache_dir)
        self.model_name = model_name
        self.cache_dir = _resolve_cache_dir(cache_dir=cache_dir)
        self.device = "cpu"
        self.model = None
        self.processor = None

    @classmethod
    def get_expected_cache_dir(cls, cache_dir: str | None = None) -> Path:
        return _resolve_cache_dir(cache_dir=cache_dir)

    @classmethod
    def is_runtime_available(cls) -> bool:
        try:
            import torch  # noqa: F401
            from transformers import AutoImageProcessor, AutoModel  # noqa: F401
        except Exception:
            return False
        return True

    def load_model(self, progress_callback: Callable[[str], None] | None = None):
        """Load the DINOv2 backbone from the Hugging Face cache."""
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if progress_callback:
            progress_callback(f"Loading DINOv2 consistency model into cache {self.cache_dir}...")

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
        )
        self.model.eval()

        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = self.model.to(self.device)
            logger.info("DINOv2 using CUDA GPU: %s", torch.cuda.get_device_name(0))
        else:
            self.device = "cpu"
            logger.info("DINOv2 using CPU.")

        if progress_callback:
            progress_callback(f"DINOv2 loaded ({self.device}).")

    def _embed_batch_pil(self, images: list[Image.Image]):
        import torch
        import torch.nn.functional as functional

        if self.model is None or self.processor is None:
            raise RuntimeError("ConsistencyScorer.load_model() must be called before scoring.")

        if not images:
            return np.empty((0, 0), dtype=np.float32)

        processed_images = [img.convert("RGB") for img in images]
        batch = self.processor(images=processed_images, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}

        with torch.inference_mode():
            outputs = self.model(**batch)
            embeddings = getattr(outputs, "pooler_output", None)
            if embeddings is None:
                embeddings = outputs.last_hidden_state[:, 0]
            embeddings = functional.normalize(embeddings.float(), dim=-1)

        return embeddings.detach().cpu().numpy()

    def compute_reference_embedding(self, reference_paths: list[Path]) -> np.ndarray:
        """Average reference embeddings into a single normalized target vector."""
        valid_paths: list[Path] = []
        images: list[Image.Image] = []

        for path in reference_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                valid_paths.append(path)
            except Exception as exc:
                logger.warning("Failed to load reference image %s: %s", path, exc)

        if not valid_paths:
            raise ValueError("No valid reference images were available for consistency scoring.")

        try:
            embeddings = self._embed_batch_pil(images)
        finally:
            for img in images:
                img.close()

        reference_embedding = embeddings.mean(axis=0)
        norm = np.linalg.norm(reference_embedding)
        if norm <= 0:
            raise ValueError("Reference embedding norm was zero.")
        return (reference_embedding / norm).astype(np.float32)

    def score_batch_pil(
        self,
        items: list[tuple[Path, Image.Image]],
        reference_embedding: np.ndarray,
    ) -> dict[Path, float]:
        """Score pre-loaded PIL images against the reference embedding."""
        results: dict[Path, float] = {}
        if not items:
            return results

        paths = [path for path, _ in items]
        images = [img for _, img in items]

        try:
            embeddings = self._embed_batch_pil(images)
            scores = np.matmul(embeddings, reference_embedding)
            for path, score in zip(paths, scores):
                results[path] = float(score)
        except Exception as exc:
            logger.warning("Consistency batch inference failed: %s", exc)
            for path in paths:
                results[path] = 0.0

        return results

    def score_batch(
        self,
        image_paths: list[Path],
        reference_embedding: np.ndarray,
    ) -> list[float]:
        """Score candidate image paths against the reference embedding."""
        images: list[Image.Image] = []
        valid_paths: list[Path] = []
        for path in image_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                valid_paths.append(path)
            except Exception as exc:
                logger.warning("Failed to load candidate image %s: %s", path, exc)

        try:
            scores = self.score_batch_pil(list(zip(valid_paths, images)), reference_embedding)
        finally:
            for img in images:
                img.close()

        return [scores.get(path, 0.0) for path in image_paths]

    def score_all(
        self,
        image_paths: list[Path],
        reference_paths: list[Path],
        batch_size: int = 16,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[Path, float]:
        """Convenience path-based scoring pipeline."""
        reference_embedding = self.compute_reference_embedding(reference_paths)
        results: dict[Path, float] = {}
        total = len(image_paths)
        for start in range(0, total, batch_size):
            chunk = image_paths[start:start + batch_size]
            scores = self.score_batch(chunk, reference_embedding)
            results.update({path: score for path, score in zip(chunk, scores)})
            if progress_callback:
                progress_callback(f"Consistency scoring... {min(start + batch_size, total)}/{total}")
        return results