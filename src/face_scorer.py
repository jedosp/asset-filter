"""Face framing scorer using MediaPipe Face Detection."""

import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


class FaceFramingScorer:
    """
    Evaluates face visibility and framing quality using MediaPipe Face Detection.

    This scorer is OPTIONAL and toggleable because:
    - Some emotion assets are face-focused (happy smile, angry, crying)
    - Some assets are pose-focused (stretching, cozy, exhausted)
      where the face may be partially hidden by design.
    """

    NO_FACE_SCORE = 0.3
    IDEAL_FACE_RATIO = 0.30
    MIN_USEFUL_FACE_RATIO = 0.08
    CENTER_TOLERANCE = 0.25

    def __init__(self):
        import mediapipe as mp

        self._mp_face = mp.solutions.face_detection
        self.detector = self._mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.3,
        )

    def score_single_pil(self, pil_image) -> float:
        """Compute face framing score from a pre-loaded PIL image (0.0-1.0)."""
        import numpy as np

        try:
            img_rgb = np.array(pil_image.convert("RGB"))
            if img_rgb is None or img_rgb.size == 0:
                return self.NO_FACE_SCORE

            results = self.detector.process(img_rgb)

            if not results.detections:
                return self.NO_FACE_SCORE

            best = max(
                results.detections,
                key=lambda d: (
                    d.location_data.relative_bounding_box.width
                    * d.location_data.relative_bounding_box.height
                ),
            )

            bbox = best.location_data.relative_bounding_box
            face_ratio = bbox.width * bbox.height

            face_cx = bbox.xmin + bbox.width / 2
            face_cy = bbox.ymin + bbox.height / 2
            center_offset = math.sqrt((face_cx - 0.5) ** 2 + (face_cy - 0.5) ** 2)

            confidence = best.score[0] if best.score else 0.5

            size_score = math.exp(
                -((face_ratio - self.IDEAL_FACE_RATIO) ** 2) / (2 * 0.25**2)
            )
            if face_ratio < self.MIN_USEFUL_FACE_RATIO:
                size_score *= face_ratio / self.MIN_USEFUL_FACE_RATIO

            if center_offset <= self.CENTER_TOLERANCE:
                position_score = 1.0
            else:
                position_score = max(
                    0.0, 1.0 - (center_offset - self.CENTER_TOLERANCE) / 0.5
                )

            return max(0.0, min(1.0, size_score * position_score * confidence))

        except Exception as e:
            logger.warning("Face detection failed: %s", e)
            return self.NO_FACE_SCORE

    def score_batch_pil(self, items: list[tuple]) -> dict:
        """Score multiple pre-loaded PIL images in parallel using threads.
        items: [(Path, PIL.Image), ...]
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}
        if not items:
            return results

        def _score_one(path_img):
            path, img = path_img
            return path, self.score_single_pil(img)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_score_one, item): item[0] for item in items}
            for future in as_completed(futures):
                try:
                    path, score = future.result()
                    results[path] = score
                except Exception as e:
                    p = futures[future]
                    logger.warning("Face scoring failed for %s: %s", p, e)
                    results[p] = self.NO_FACE_SCORE
        return results

    def close(self):
        if hasattr(self, "detector"):
            self.detector.close()
