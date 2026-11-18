import numpy as np
import cv2

from typing import Tuple, Optional
import numpy as np
import cv2

try:
    from src.preprocessing.gabor import GaborFilterBank, estimate_ridge_frequency
    _GABOR_AVAILABLE = True
except ImportError:
    _GABOR_AVAILABLE = False


def estimate_orientation_field(image: np.ndarray, block_size: int = 16, continuous: bool = False, smooth_sigma: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates ridge orientation field using squared gradients.
    
    Args:
        image: Grayscale image.
        block_size: Size of analysis block.
        continuous: If True, uses continuous 2D Gaussian vector field smoothing.
                   If False, uses legacy block-wise tensor averaging for exact reproducibility.
        smooth_sigma: Gaussian sigma for continuous mode.
        
    Returns:
        orientations: Angle theta in radians in [0, pi) for each pixel.
        coherence: Confidence/certainty measure in [0, 1].
    """
    sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    
    h, w = image.shape
    
    if continuous:
        # Double-angle continuous vector representation: Phi = (G_x^2 - G_y^2, 2 * G_x * G_y)
        phi_x = sobel_x ** 2 - sobel_y ** 2
        phi_y = 2.0 * sobel_x * sobel_y
        
        ksize = int(block_size) if int(block_size) % 2 == 1 else int(block_size) + 1
        phi_x_smooth = cv2.GaussianBlur(phi_x, (ksize, ksize), smooth_sigma)
        phi_y_smooth = cv2.GaussianBlur(phi_y, (ksize, ksize), smooth_sigma)
        
        denom = np.sqrt(phi_x_smooth ** 2 + phi_y_smooth ** 2) + 1e-8
        grad_energy = cv2.GaussianBlur(sobel_x ** 2 + sobel_y ** 2, (ksize, ksize), smooth_sigma) + 1e-8
        coherence = np.clip(denom / grad_energy, 0.0, 1.0)
        
        theta = 0.5 * np.arctan2(phi_y_smooth, phi_x_smooth) + np.pi / 2.0
        orientations = (theta % np.pi).astype(np.float32)
        return orientations, coherence

    # Canonical block-wise averaging for legacy pipeline reproducibility
    gxx = sobel_x ** 2
    gyy = sobel_y ** 2
    gxy = sobel_x * sobel_y
    
    orientations = np.zeros((h, w), dtype=np.float32)
    coherence = np.zeros((h, w), dtype=np.float32)
    
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            b_gxx = np.sum(gxx[y:min(y+block_size, h), x:min(x+block_size, w)])
            b_gyy = np.sum(gyy[y:min(y+block_size, h), x:min(x+block_size, w)])
            b_gxy = np.sum(gxy[y:min(y+block_size, h), x:min(x+block_size, w)])
            
            denom = np.sqrt((b_gxx - b_gyy)**2 + 4 * b_gxy**2) + 1e-8
            coh = denom / (b_gxx + b_gyy + 1e-8)
            
            theta = 0.5 * np.arctan2(2 * b_gxy, b_gxx - b_gyy) + np.pi / 2
            theta = theta % np.pi
            
            orientations[y:min(y+block_size, h), x:min(x+block_size, w)] = theta
            coherence[y:min(y+block_size, h), x:min(x+block_size, w)] = np.clip(coh, 0.0, 1.0)
            
    return orientations, coherence


_DEFAULT_GABOR_BANK: Optional["GaborFilterBank"] = None


def _get_default_gabor_bank() -> "GaborFilterBank":
    global _DEFAULT_GABOR_BANK
    if _DEFAULT_GABOR_BANK is None:
        _DEFAULT_GABOR_BANK = GaborFilterBank()
    return _DEFAULT_GABOR_BANK


def enhance_fingerprint(gray_image: np.ndarray, 
                        clip_limit: float = 3.0, 
                        tile_grid: tuple = (8, 8),
                        use_gabor: bool = False,
                        mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enhances fingerprint ridge contrast using CLAHE, Gaussian smoothing,
    and optional contextual Gabor bandpass filtering.
    
    Args:
        gray_image: Grayscale uint8 image.
        clip_limit: Threshold for CLAHE contrast limiting.
        tile_grid: Grid size for histogram equalization.
        use_gabor: If True, applies 2D contextual Gabor filter bank (Hong-Wan-Jain 1998).
        mask: Optional binary foreground mask for Gabor filtering.
        
    Returns:
        enhanced: Enhanced image.
        orientations: Dense orientation map (radians in [0, pi)).
        coherence: Local coherence map (0 to 1).
    """
    # 1. Intensity Normalization
    norm = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # 2. CLAHE Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(norm)
    
    # 3. Gaussian smoothing to reduce sensor noise
    smoothed = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 4. Orientation & Coherence Estimation (continuous for Gabor, block-wise for legacy)
    orientations, coherence = estimate_orientation_field(smoothed, continuous=use_gabor)
    
    # 5. Contextual Gabor Filter Bank Enhancement
    if use_gabor and _GABOR_AVAILABLE:
        effective_mask = mask if mask is not None else np.full(gray_image.shape, 255, dtype=np.uint8)
        norm_float = smoothed.astype(np.float32) / 255.0
        frequencies = estimate_ridge_frequency(norm_float, orientations, effective_mask)
        gabor_bank = _get_default_gabor_bank()
        gabor_enhanced = gabor_bank.filter_image(norm_float, orientations, frequencies, effective_mask)
        return gabor_enhanced, orientations, coherence
        
    return smoothed, orientations, coherence
