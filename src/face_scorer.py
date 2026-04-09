"""Face framing scorer using MediaPipe Face Detection."""

import logging
import math
from pathlib import Path
from typing import Callable

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

    def score_single(self, image_path: Path) -> float:
        """
        Compute face framing score for a single image (0.0-1.0).

        Scoring considers face size ratio, center offset, and detection confidence.
        Returns NO_FACE_SCORE (0.3) when no face detected — a mild penalty,
        not a hard fail, since MediaPipe misses ~20-30% of anime faces.
        """
        import cv2

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return self.NO_FACE_SCORE

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.detector.process(img_rgb)

            if not results.detections:
                return self.NO_FACE_SCORE

            # Use the largest detected face
            best = max(
                results.detections,
                key=lambda d: (
                    d.location_data.relative_bounding_box.width
                    * d.location_data.relative_bounding_box.height
                ),
            )

            bbox = best.location_data.relative_bounding_box
            face_ratio = bbox.width * bbox.height

            # Face center (normalized 0-1)
            face_cx = bbox.xmin + bbox.width / 2
            face_cy = bbox.ymin + bbox.height / 2
            center_offset = math.sqrt((face_cx - 0.5) ** 2 + (face_cy - 0.5) ** 2)

            confidence = best.score[0] if best.score else 0.5

            # Size score: gaussian around ideal ratio
            size_score = math.exp(
                -((face_ratio - self.IDEAL_FACE_RATIO) ** 2) / (2 * 0.25**2)
            )
            if face_ratio < self.MIN_USEFUL_FACE_RATIO:
                size_score *= face_ratio / self.MIN_USEFUL_FACE_RATIO

            # Position score
            if center_offset <= self.CENTER_TOLERANCE:
                position_score = 1.0
            else:
                position_score = max(
                    0.0, 1.0 - (center_offset - self.CENTER_TOLERANCE) / 0.5
                )

            return max(0.0, min(1.0, size_score * position_score * confidence))

        except Exception as e:
            logger.warning("Face detection failed for %s: %s", image_path, e)
            return self.NO_FACE_SCORE

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

    def score_all(
        self,
        image_paths: list[Path],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[Path, float]:
        """Score all images. MediaPipe is fast (~10-20ms/image), no batching needed."""
        results = {}
        total = len(image_paths)

        for i, path in enumerate(image_paths):
            results[path] = self.score_single(path)
            if progress_callback and ((i + 1) % 10 == 0 or i + 1 == total):
                progress_callback(i + 1, total)

        return results

    def close(self):
        if hasattr(self, "detector"):
            self.detector.close()
