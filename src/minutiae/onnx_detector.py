"""
ONNX Deep Neural Minutiae Detector Service.
Provides deep convolutional minutiae detection and orientation regression via lightweight CPU ONNX Runtime.

Integrates with the cancelable biometric pipeline:
- Acts solely as a high-repeatability front-end feature extractor.
- Automatically falls back to contextual Gabor classical extraction if ONNX runtime or model weights are unavailable.
- Deep minutiae (x, y, theta, quality) feed directly into the mathematically non-invertible cancelable backend.
"""

import os
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

from src.minutiae.extractor import Minutia, extract_minutiae
from src.preprocessing.enhancer import enhance_fingerprint

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

DEFAULT_ONNX_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "minutiae_net_v1.onnx")
)


class ONNXMinutiaeDetector:
    """
    Lightweight CPU ONNX Minutiae Detector with Non-Maximum Suppression (NMS)
    and automatic graceful fallback to classical Gabor extraction.
    """
    def __init__(self, 
                 model_path: Optional[str] = None,
                 conf_thresh: float = 0.35,
                 nms_radius: int = 8,
                 border_margin: int = 15,
                 max_minutiae: int = 80):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.nms_radius = nms_radius
        self.border_margin = border_margin
        self.max_minutiae = max_minutiae
        self.session: Optional[Any] = None

        if _ONNX_AVAILABLE and model_path is not None and os.path.exists(model_path):
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(
                    model_path, sess_options=opts, providers=['CPUExecutionProvider']
                )
            except Exception as e:
                self.session = None

    @property
    def is_model_loaded(self) -> bool:
        return self.session is not None

    def detect(self, image: np.ndarray, mask: np.ndarray,
               orientation_map: Optional[np.ndarray] = None,
               coherence_map: Optional[np.ndarray] = None) -> Tuple[List[Minutia], np.ndarray]:
        """
        Detects minutiae using the deep ONNX model if loaded, else falls back to classical Gabor.

        Args:
            image: Grayscale uint8 image.
            mask: Binary foreground mask.
            orientation_map: Optional precomputed orientation map.
            coherence_map: Optional precomputed coherence map.

        Returns:
            minutiae: List of Minutia objects.
            skeleton_or_heatmap: Binary skeleton or detection heatmap.
        """
        if not self.is_model_loaded:
            # Graceful fallback: Gabor-enhanced classical extraction with structural artifact pruning
            if orientation_map is None or coherence_map is None:
                enhanced, orientation_map, coherence_map = enhance_fingerprint(
                    image, use_gabor=True, mask=mask
                )
            else:
                enhanced = image

            return extract_minutiae(
                enhanced, mask, orientation_map, coherence_map,
                border_margin=self.border_margin,
                quality_thresh=self.conf_thresh,
                max_minutiae=self.max_minutiae,
                binarize_method="adaptive",
                prune_artifacts=True
            )

        # Deep ONNX Inference Path
        h_orig, w_orig = image.shape
        # Normalize and pad/resize to multiple of 32 for CNN feature maps
        pad_h = (32 - (h_orig % 32)) % 32
        pad_w = (32 - (w_orig % 32)) % 32
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        norm_img = (padded.astype(np.float32) / 255.0)[None, None, :, :]  # (1, 1, H_pad, W_pad)

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: norm_img})

        # Expected model outputs:
        # outputs[0]: Minutiae probability heatmap (1, 1, H_pad, W_pad)
        # outputs[1]: Orientation vector map (1, 2, H_pad, W_pad) representing (cos 2theta, sin 2theta)
        prob_map = outputs[0][0, 0, :h_orig, :w_orig]
        cos_map = outputs[1][0, 0, :h_orig, :w_orig]
        sin_map = outputs[1][0, 1, :h_orig, :w_orig]

        # Mask probability map by foreground ROI & suppress mask-boundary artifacts
        if isinstance(mask, (tuple, list)):
            mask = mask[1] if len(mask) >= 2 and isinstance(mask[1], np.ndarray) else mask[0]
        fg_u8 = (mask > 0).astype(np.uint8)
        prob_map = prob_map * fg_u8.astype(np.float32)
        if np.any(fg_u8 == 0):
            dist_fg = cv2.distanceTransform(fg_u8, cv2.DIST_L2, 3)
            prob_map[dist_fg < 6.0] = 0.0

        # Suppress image borders
        if self.border_margin > 0:
            prob_map[:self.border_margin, :] = 0
            prob_map[-self.border_margin:, :] = 0
            prob_map[:, :self.border_margin] = 0
            prob_map[:, -self.border_margin:] = 0

        # Non-Maximum Suppression (NMS) via max-pooling
        kernel_size = 2 * self.nms_radius + 1
        dilated = cv2.dilate(prob_map, np.ones((kernel_size, kernel_size), dtype=np.uint8))
        peaks = (prob_map == dilated) & (prob_map >= self.conf_thresh)

        peak_ys, peak_xs = np.where(peaks)
        candidates = []
        for y, x in zip(peak_ys, peak_xs):
            q = float(prob_map[y, x])
            px, py = float(x), float(y)
            if 1 <= y < h_orig - 1 and 1 <= x < w_orig - 1:
                denom_x = prob_map[y, x - 1] - 2.0 * q + prob_map[y, x + 1]
                if denom_x < -1e-6:
                    px += float(np.clip(0.5 * (prob_map[y, x - 1] - prob_map[y, x + 1]) / denom_x, -0.5, 0.5))
                denom_y = prob_map[y - 1, x] - 2.0 * q + prob_map[y + 1, x]
                if denom_y < -1e-6:
                    py += float(np.clip(0.5 * (prob_map[y - 1, x] - prob_map[y + 1, x]) / denom_y, -0.5, 0.5))

            cos2th = float(cos_map[y, x])
            sin2th = float(sin_map[y, x])
            th = 0.5 * np.arctan2(sin2th, cos2th)
            th = float(th % np.pi)
            candidates.append(Minutia(px, py, th, 0, q))

        # Sort by confidence descending and truncate to max_minutiae
        candidates.sort(key=lambda m: m.quality, reverse=True)
        candidates = candidates[:self.max_minutiae]

        heatmap_u8 = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
        return candidates, heatmap_u8
