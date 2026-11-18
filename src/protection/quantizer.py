from typing import Optional
import numpy as np


def quantize_descriptors(projected: np.ndarray, thresholds: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Quantizes continuous projected descriptors into binary bitstrings {0, 1}^M.
    
    Canonical Tie-Breaking Rule (P0.6):
        b_j = 1 if (W f_tilde)_j >= 0 else 0
        b_j = I((W f_tilde)_j >= 0)
        
    Under mathematical sign(x) where sign(0) = 0, the naive formula
    b = 0.5 * (1 + sign(W f_tilde)) produces 0.5 for exact zero, which is not a bit.
    This function enforces deterministic Heaviside step quantization with exact zero
    mapping to 1.
    
    Dimension Support & Signed Zero Handling:
        - Supports both 1D vector inputs of shape (M,) and 2D batch inputs of shape (N, M).
        - Sanitizes signed floating point zeros (+0.0 and -0.0) via `projected + 0.0`
          so that IEEE 754 negative zero (-0.0) behaves identically to positive zero (+0.0)
          and deterministically maps to bit 1 under zero-thresholding.
          
    Args:
        projected: (N, M) array of projected descriptors or (M,) 1D vector.
        thresholds: Optional threshold array matching feature dimension, or scalar.
                    If None, defaults to 0.0.
                    
    Returns:
        bitstrings: uint8 array of same shape (or (N, M)) containing bits in {0, 1}.
    """
    arr = np.asarray(projected)
    if arr.size == 0:
        return np.empty(arr.shape, dtype=np.uint8)

    # Reject non-finite values (NaN and Inf) to prevent undefined biometric representations
    if not np.all(np.isfinite(arr)):
        raise ValueError("Cannot quantize non-finite (NaN or infinite) values")
        
    # Sanitize signed zeros: in IEEE 754 arithmetic, (-0.0 + 0.0) produces +0.0
    arr_clean = arr + 0.0
    
    if thresholds is None:
        bitstrings = (arr_clean >= 0.0).astype(np.uint8)
    else:
        thresh = np.asarray(thresholds) + 0.0
        bitstrings = (arr_clean >= thresh).astype(np.uint8)
        
    return bitstrings
