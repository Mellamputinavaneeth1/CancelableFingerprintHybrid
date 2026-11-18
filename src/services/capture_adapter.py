"""
Capture Adapter Service.

Ingests raw images, byte buffers, or file paths, applies dimension, format,
and bit-depth validation, and handles Failure-to-Acquire (FTA) / Failure-to-Enroll (FTE) boundaries.
"""

import os
from typing import Union, Dict, Any, Optional
import numpy as np
import cv2

from src.services.types import (
    RawCapture, 
    BiometricErrorCode, 
    InvalidCaptureError
)


class CaptureAdapter:
    """
    Standardized adapter for ingesting biometric image data.
    Validates sensor parameters, dimensions, and channel counts.
    """
    def __init__(self, min_width: int = 100, min_height: int = 100, default_dpi: float = 500.0):
        self.min_width = min_width
        self.min_height = min_height
        self.default_dpi = default_dpi

    def ingest(self, 
               source: Union[np.ndarray, bytes, str], 
               dpi: Optional[float] = None,
               sensor_id: str = "unknown",
               metadata: Optional[Dict[str, Any]] = None) -> RawCapture:
        """
        Ingests and validates raw capture from array, bytes, or file path.

        Args:
            source: Numpy array, raw file path, or image byte buffer.
            dpi: Sensor resolution in DPI (defaults to self.default_dpi).
            sensor_id: Identifier of physical capture sensor.
            metadata: Optional sensor metadata dict.

        Returns:
            RawCapture object containing validated 2-D grayscale uint8 image.

        Raises:
            InvalidCaptureError: If image is missing, corrupt, too small, or invalid bit depth.
        """
        meta = dict(metadata or {})
        eff_dpi = float(dpi) if dpi is not None else self.default_dpi

        if source is None:
            raise InvalidCaptureError(
                BiometricErrorCode.INVALID_IMAGE,
                "Image source cannot be None"
            )

        # 1. Handle file path
        if isinstance(source, str):
            if not os.path.exists(source):
                raise InvalidCaptureError(
                    BiometricErrorCode.INVALID_IMAGE,
                    f"Capture file not found: {source}"
                )
            img = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise InvalidCaptureError(
                    BiometricErrorCode.INVALID_IMAGE,
                    f"Failed to decode image file: {source}"
                )
            meta["source_path"] = source

        # 2. Handle raw bytes buffer
        elif isinstance(source, (bytes, bytearray)):
            if len(source) == 0:
                raise InvalidCaptureError(
                    BiometricErrorCode.INVALID_IMAGE,
                    "Empty image byte buffer provided"
                )
            try:
                nparr = np.frombuffer(source, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            except Exception as e:
                raise InvalidCaptureError(
                    BiometricErrorCode.INVALID_IMAGE,
                    f"Failed to decode raw image byte buffer: {str(e)}"
                )
            if img is None:
                raise InvalidCaptureError(
                    BiometricErrorCode.INVALID_IMAGE,
                    "Failed to decode raw image byte buffer"
                )
            meta["source_type"] = "bytes"

        # 3. Handle numpy ndarray
        elif isinstance(source, np.ndarray):
            if source.size == 0:
                raise InvalidCaptureError(
                    BiometricErrorCode.INVALID_IMAGE,
                    "Empty image array provided"
                )
            if np.isnan(source).any() or np.isinf(source).any():
                raise InvalidCaptureError(
                    BiometricErrorCode.INVALID_IMAGE,
                    "Image array contains NaN or Inf values"
                )
            if source.ndim == 3:
                # Convert color to grayscale
                if source.shape[2] == 3:
                    img = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
                elif source.shape[2] == 4:
                    img = cv2.cvtColor(source, cv2.COLOR_BGRA2GRAY)
                else:
                    raise InvalidCaptureError(
                        BiometricErrorCode.INVALID_IMAGE,
                        f"Unsupported channel dimension: {source.shape[2]}"
                    )
            elif source.ndim == 2:
                img = source.copy()
            else:
                raise InvalidCaptureError(
                    BiometricErrorCode.INVALID_IMAGE,
                    f"Invalid array dimension: {source.ndim}"
                )
            meta["source_type"] = "ndarray"
        else:
            raise InvalidCaptureError(
                BiometricErrorCode.INVALID_IMAGE,
                f"Unsupported source type: {type(source).__name__}"
            )

        # Ensure uint8
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                # Scale float in [0, 1] or [0, 255]
                if img.max() <= 1.0:
                    img = (img * 255.0).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        h, w = img.shape
        if w < self.min_width or h < self.min_height:
            raise InvalidCaptureError(
                BiometricErrorCode.IMAGE_TOO_SMALL,
                f"Image dimensions ({w}x{h}) below minimum required ({self.min_width}x{self.min_height})"
            )

        return RawCapture(
            image=img,
            dpi=eff_dpi,
            width=w,
            height=h,
            sensor_id=sensor_id,
            metadata=meta
        )
