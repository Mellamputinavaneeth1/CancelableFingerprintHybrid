"""
Minutiae Extractor Service.

Extracts minutiae coordinates, orientations, and quality metrics from
enhanced fingerprint ridge skeleton, enforcing operational minimum yield thresholds.
"""

from typing import Optional, List
import numpy as np

from src.minutiae.extractor import extract_minutiae, Minutia
from src.minutiae.onnx_detector import ONNXMinutiaeDetector, DEFAULT_ONNX_MODEL_PATH
from src.services.types import (
    PreprocessedImage, 
    ExtractedMinutiae, 
    BiometricErrorCode, 
    InsufficientMinutiaeError
)


class MinutiaeExtractor:
    """Production service for classical or ONNX-accelerated minutiae feature extraction."""
    def __init__(
        self,
        min_minutiae: int = 3,
        engine: str = "classical",
        binarize_method: str = "otsu",
        prune_artifacts: bool = False,
        quality_thresh: Optional[float] = None,
        model_path: Optional[str] = None,
        conf_thresh: float = 0.35,
        nms_radius: int = 8,
        border_margin: int = 15,
        max_minutiae: int = 80,
    ):
        if engine not in ("classical", "deep"):
            raise ValueError(f"Unknown extraction engine '{engine}'. Must be 'classical' or 'deep'.")
        self.min_minutiae = min_minutiae
        self.engine = engine
        self.binarize_method = binarize_method
        self.prune_artifacts = prune_artifacts
        self.quality_thresh = quality_thresh
        self.border_margin = border_margin
        self.max_minutiae = max_minutiae
        self.onnx_detector: Optional[ONNXMinutiaeDetector] = None
        if engine == "deep":
            resolved = model_path if model_path is not None else DEFAULT_ONNX_MODEL_PATH
            self.onnx_detector = ONNXMinutiaeDetector(
                model_path=resolved,
                conf_thresh=conf_thresh,
                nms_radius=nms_radius,
                border_margin=border_margin,
                max_minutiae=max_minutiae,
            )

    def extract(
        self,
        preprocessed: PreprocessedImage,
        raw_image: Optional[np.ndarray] = None,
    ) -> ExtractedMinutiae:
        """
        Extracts minutiae points from enhanced ridge pattern or raw image via ONNX.
        """
        if self.engine == "deep" and self.onnx_detector is not None and self.onnx_detector.is_model_loaded:
            src_img = raw_image if raw_image is not None else preprocessed.enhanced
            minutiae, skeleton = self.onnx_detector.detect(
                src_img,
                preprocessed.mask,
                orientation_map=preprocessed.orientations,
                coherence_map=preprocessed.coherence,
            )
        else:
            kwargs = {}
            if self.binarize_method != "otsu":
                kwargs["binarize_method"] = self.binarize_method
            if self.prune_artifacts:
                kwargs["prune_artifacts"] = True
            if self.quality_thresh is not None:
                kwargs["quality_thresh"] = self.quality_thresh
            if self.border_margin != 15:
                kwargs["border_margin"] = self.border_margin
            if self.max_minutiae != 80:
                kwargs["max_minutiae"] = self.max_minutiae

            minutiae, skeleton = extract_minutiae(
                preprocessed.enhanced,
                preprocessed.mask,
                preprocessed.orientations,
                preprocessed.coherence,
                **kwargs,
            )

        if len(minutiae) < self.min_minutiae:
            raise InsufficientMinutiaeError(
                BiometricErrorCode.INSUFFICIENT_MINUTIAE,
                f"Extracted {len(minutiae)} minutiae, which is below the operational minimum of {self.min_minutiae}"
            )

        return ExtractedMinutiae(
            minutiae=minutiae,
            skeleton=skeleton,
            count=len(minutiae)
        )
