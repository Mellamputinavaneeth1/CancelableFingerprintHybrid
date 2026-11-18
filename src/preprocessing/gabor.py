"""
Contextual 2D Gabor Filter Bank for Fingerprint Ridge Enhancement.
Implements the Hong, Wan, Jain (IEEE TPAMI 1998) algorithm.

Applies frequency-selective, orientation-tuned 2D Gabor bandpass filters:
    h(x, y; theta, f) = exp( -0.5 * (x_theta^2 / sigma_x^2 + y_theta^2 / sigma_y^2) ) * cos(2 * pi * f * y_theta)
where theta is the ridge tangent orientation in [0, pi), so:
    x_theta =  x * cos(theta) + y * sin(theta)   (coordinate ALONG the ridge tangent)
    y_theta = -x * sin(theta) + y * cos(theta)   (coordinate NORMAL to the ridge, across ridge-valley cycles)
"""

from typing import Tuple, Dict, Optional
import numpy as np
import cv2


def estimate_ridge_frequency(image: np.ndarray, 
                             orientations: np.ndarray, 
                             mask: np.ndarray,
                             block_size: int = 32, 
                             min_wave: float = 4.0, 
                             max_wave: float = 20.0,
                             default_freq: float = 0.11) -> np.ndarray:
    """
    Estimates the local ridge frequency f(x, y) in cycles/pixel using 1D orthogonal
    projection profiles oriented along the ridge normal.

    Args:
        image: Grayscale float32 or uint8 normalized image.
        orientations: Dense orientation map theta (ridge tangent) in radians modulo pi in [0, pi).
        mask: Binary foreground mask (values 0 or 255 / 1).
        block_size: Size of local window for projection analysis.
        min_wave: Minimum expected ridge wavelength in pixels.
        max_wave: Maximum expected ridge wavelength in pixels.
        default_freq: Fallback frequency (cycles/pixel) if estimation is unresolvable.

    Returns:
        freq_map: Dense 2D frequency map f(x, y) in cycles/pixel.
    """
    h, w = image.shape
    img_f = image.astype(np.float32, copy=False)
    freq_acc = np.zeros((h, w), dtype=np.float32)
    weight_acc = np.zeros((h, w), dtype=np.float32)
    binary_mask = (mask > 0)

    half_block = block_size // 2
    step = max(block_size // 2, 8)
    valid_freqs = []

    for y in range(half_block, h - half_block, step):
        for x in range(half_block, w - half_block, step):
            if not binary_mask[y, x]:
                continue

            theta = float(orientations[y, x])
            # Rotate patch so ridge tangent aligns vertically (column axis) and normal runs horizontally
            theta_norm_deg = float(np.degrees(theta + np.pi / 2.0))
            rot_mat = cv2.getRotationMatrix2D((float(x), float(y)), theta_norm_deg, 1.0)
            # Translate local center (x, y) directly into (half_block, half_block) of output block
            rot_mat[0, 2] += float(half_block) - float(x)
            rot_mat[1, 2] += float(half_block) - float(y)

            sub = cv2.warpAffine(
                img_f, rot_mat, (block_size, block_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

            # 1D projection profile averaged along the rotated vertical ridge axis
            proj = np.mean(sub, axis=0)
            ptp = float(np.ptp(proj))
            if ptp < 1e-4:
                continue

            # Gentle 3-tap smoothing to suppress single-pixel aliasing ripples
            proj_s = np.convolve(proj, np.array([0.25, 0.5, 0.25], dtype=np.float32), mode='same')
            min_prom = 0.08 * ptp

            # Detect genuine local maxima separated by >= min_wave
            peaks = []
            for i in range(1, len(proj_s) - 1):
                if proj_s[i] > proj_s[i - 1] and proj_s[i] >= proj_s[i + 1]:
                    lo = max(0, i - 3)
                    hi = min(len(proj_s), i + 4)
                    if proj_s[i] - np.min(proj_s[lo:hi]) >= min_prom:
                        if not peaks or (i - peaks[-1]) >= max(2, int(min_wave * 0.7)):
                            peaks.append(i)
                        elif proj_s[i] > proj_s[peaks[-1]]:
                            peaks[-1] = i

            if len(peaks) >= 2:
                peak_dists = np.diff(peaks)
                valid_dists = peak_dists[(peak_dists >= min_wave) & (peak_dists <= max_wave)]
                if len(valid_dists) > 0:
                    wavelength = float(np.median(valid_dists))
                    freq = float(np.clip(1.0 / wavelength, 1.0 / max_wave, 1.0 / min_wave))
                    valid_freqs.append(freq)
                    y0, y1 = max(0, y - half_block), min(h, y + half_block)
                    x0, x1 = max(0, x - half_block), min(w, x + half_block)
                    freq_acc[y0:y1, x0:x1] += freq
                    weight_acc[y0:y1, x0:x1] += 1.0

    fallback = float(np.median(valid_freqs)) if valid_freqs else float(default_freq)

    # Normalized Gaussian interpolation across valid ridge-frequency blocks
    freq_blur = cv2.GaussianBlur(freq_acc, (25, 25), 5.0)
    weight_blur = cv2.GaussianBlur(weight_acc, (25, 25), 5.0)
    freq_map = np.where(weight_blur > 1e-4, freq_blur / np.maximum(weight_blur, 1e-4), fallback)
    freq_map = np.clip(freq_map, 1.0 / max_wave, 1.0 / min_wave).astype(np.float32)
    return freq_map


def build_gabor_kernel(ksize: int, theta: float, freq: float, 
                       sigma_x: float, sigma_y: float) -> np.ndarray:
    """
    Constructs an even-symmetric 2D Gabor filter kernel tuned to ridge tangent `theta`.
    Since `x_th` runs ALONG the ridge and `y_th` runs NORMAL to the ridge, the harmonic
    carrier oscillates along `y_th`: cos(2 * pi * freq * y_th).
    """
    half = ksize // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1]

    # Coordinate rotation into local ridge frame (x_th along ridge, y_th across ridge)
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    x_th = x * cos_th + y * sin_th
    y_th = -x * sin_th + y * cos_th

    # Anisotropic Gaussian envelope and orthogonal harmonic carrier
    gaussian = np.exp(-0.5 * ((x_th ** 2) / (sigma_x ** 2) + (y_th ** 2) / (sigma_y ** 2)))
    carrier = np.cos(2.0 * np.pi * freq * y_th)

    gabor = gaussian * carrier
    # Zero DC bias (mean zero)
    gabor = gabor - np.mean(gabor)
    return gabor.astype(np.float32)


class GaborFilterBank:
    """
    Precomputed 2D contextual Gabor filter bank covering discrete orientation
    and frequency channels for high-throughput vectorized filtering.
    """
    def __init__(self, 
                 num_orientations: int = 16, 
                 frequencies: Optional[Tuple[float, ...]] = (0.075, 0.090, 0.105, 0.120, 0.140, 0.165),
                 ksize: int = 21, 
                 sigma_x: Optional[float] = None, 
                 sigma_y: Optional[float] = None):
        self.num_orientations = num_orientations
        self.frequencies = frequencies if frequencies is not None else (0.08, 0.10, 0.12, 0.15)
        self.ksize = ksize if ksize % 2 == 1 else ksize + 1

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

        # Discretize orientations in [0, pi)
        self.angles = np.linspace(0, np.pi, num_orientations, endpoint=False)

        # Precompute kernels: bank[(angle_idx, freq_idx)] -> 2D kernel
        # When sigma_x/sigma_y are unspecified, scale envelope proportionally to ridge wavelength (Hong-Wan-Jain)
        self.bank: Dict[Tuple[int, int], np.ndarray] = {}
        for a_idx, th in enumerate(self.angles):
            for f_idx, frq in enumerate(self.frequencies):
                sx = sigma_x if sigma_x is not None else float(np.clip(0.55 / max(frq, 1e-4), 3.0, 7.0))
                sy = sigma_y if sigma_y is not None else float(np.clip(0.45 / max(frq, 1e-4), 2.5, 6.0))
                kernel = build_gabor_kernel(self.ksize, th, frq, sx, sy)
                self.bank[(a_idx, f_idx)] = kernel

    def filter_image(self, image: np.ndarray, 
                     orientations: np.ndarray, 
                     frequencies: np.ndarray, 
                     mask: np.ndarray) -> np.ndarray:
        """
        Applies contextual Gabor filtering to image according to pixel-wise
        ridge tangent orientation theta and frequency f.

        Args:
            image: Grayscale float32 normalized image [0, 1].
            orientations: Dense orientation map in [0, pi).
            frequencies: Dense frequency map in cycles/pixel.
            mask: Binary foreground mask.

        Returns:
            enhanced: Enhanced grayscale image in uint8 [0, 255].
        """
        h, w = image.shape
        enhanced = np.zeros((h, w), dtype=np.float32)
        binary_mask = (mask > 0)
        if not np.any(binary_mask):
            return enhanced.astype(np.uint8)

        # Restrict orientation/frequency quantization to foreground bounding box for efficiency
        fg_ys, fg_xs = np.nonzero(binary_mask)
        fy0, fy1 = int(fg_ys.min()), int(fg_ys.max()) + 1
        fx0, fx1 = int(fg_xs.min()), int(fg_xs.max()) + 1

        angle_step = np.pi / self.num_orientations
        angle_roi = np.clip(
            np.round(orientations[fy0:fy1, fx0:fx1] / angle_step).astype(np.int32) % self.num_orientations,
            0, self.num_orientations - 1
        )

        freq_arr = np.asarray(self.frequencies, dtype=np.float32)
        freq_roi = np.argmin(np.abs(frequencies[fy0:fy1, fx0:fx1, None] - freq_arr[None, None, :]), axis=2)
        mask_roi = binary_mask[fy0:fy1, fx0:fx1]

        half = self.ksize // 2
        if half > 0:
            img_padded = cv2.copyMakeBorder(
                image.astype(np.float32, copy=False),
                half, half, half, half,
                borderType=cv2.BORDER_REFLECT_101,
            )
        else:
            img_padded = image.astype(np.float32, copy=False)

        enh_roi = enhanced[fy0:fy1, fx0:fx1]
        # Filter channel-by-channel strictly over each channel's bounding box (bit-exact to full image)
        for a_idx in range(self.num_orientations):
            a_mask = (angle_roi == a_idx) & mask_roi
            if not np.any(a_mask):
                continue
            for f_idx in range(len(self.frequencies)):
                pixel_selector = a_mask & (freq_roi == f_idx)
                if not np.any(pixel_selector):
                    continue

                kernel = self.bank[(a_idx, f_idx)]
                ys, xs = np.nonzero(pixel_selector)
                r0, r1 = int(ys.min()), int(ys.max()) + 1
                c0, c1 = int(xs.min()), int(xs.max()) + 1

                if half > 0:
                    patch = img_padded[
                        fy0 + r0 : fy0 + r1 + 2 * half,
                        fx0 + c0 : fx0 + c1 + 2 * half,
                    ]
                    filtered_sub = cv2.filter2D(patch, cv2.CV_32F, kernel)[half:-half, half:-half]
                else:
                    patch = img_padded[fy0 + r0 : fy0 + r1, fx0 + c0 : fx0 + c1]
                    filtered_sub = cv2.filter2D(patch, cv2.CV_32F, kernel)

                sub_sel = pixel_selector[r0:r1, c0:c1]
                enh_roi[r0:r1, c0:c1][sub_sel] = filtered_sub[sub_sel]

        # Normalize symmetrically around zero-response (127.5) so dark ridges (<0) map <127.5 and valleys (>0) map >127.5
        fg_vals = enhanced[binary_mask]
        if fg_vals.size > 0:
            scale = float(np.percentile(np.abs(fg_vals), 99))
            if scale > 1e-6:
                enhanced = 127.5 + 127.5 * np.clip(enhanced / scale, -1.0, 1.0)
            else:
                enhanced = np.clip(enhanced, 0.0, 1.0) * 255.0

        enhanced[~binary_mask] = 0.0
        return np.clip(enhanced, 0, 255).astype(np.uint8)

