"""
Preprocessor Service.

Wraps foreground segmentation, orientation field estimation, coherence analysis,
and contextual Gabor ridge enhancement.
"""

from typing import Optional
import numpy as np

from src.preprocessing.segmenter import segment_fingerprint
from src.preprocessing.enhancer import enhance_fingerprint
from src.services.types import (
    RawCapture, 
    PreprocessedImage, 
    BiometricErrorCode, 
    InvalidCaptureError
)


class Preprocessor:
    """Production service for fingerprint segmentation and ridge enhancement."""
    def __init__(self,
                 block_size: int = 16,
                 threshold_factor: float = 0.1,
                 segmentation_method: str = "variance",
                 mask_enhancer: bool = False):
        self.block_size = block_size
        self.threshold_factor = threshold_factor
        self.segmentation_method = segmentation_method
        self.mask_enhancer = mask_enhancer

    def process(self, capture: RawCapture) -> PreprocessedImage:
        """
        Executes segmentation and enhancement on raw capture.

        Args:
            capture: Validated RawCapture object.

        Returns:
            PreprocessedImage with enhanced ridges and orientation fields.
        """
        try:
            if self.segmentation_method == "variance":
                segmented, mask, bbox = segment_fingerprint(capture.image)
            else:
                segmented, mask, bbox = segment_fingerprint(
                    capture.image,
                    block_size=self.block_size,
                    method=self.segmentation_method,
                )
            if self.mask_enhancer:
                enhanced, orientations, coherence = enhance_fingerprint(
                    capture.image, use_gabor=True, mask=mask
                )
            else:
                enhanced, orientations, coherence = enhance_fingerprint(segmented)
            
            return PreprocessedImage(
                segmented=segmented,
                mask=mask,
                bbox=bbox,
                enhanced=enhanced,
                orientations=orientations,
                coherence=coherence
            )
        except Exception as e:
            raise InvalidCaptureError(
                BiometricErrorCode.SEGMENTATION_FAILED,
                f"Preprocessing failed during enhancement: {str(e)}"
            )
