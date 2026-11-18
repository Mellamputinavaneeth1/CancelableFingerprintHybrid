import numpy as np

def calculate_minutia_quality(x: float, y: float, mask: np.ndarray, coherence_map: np.ndarray, border_margin: int = 15):
    """
    Computes a composite quality score q in [0, 1] for a detected minutia.
    
    Factors considered:
    1. Distance from the boundary of the segmented fingerprint mask.
    2. Local ridge orientation coherence around (x, y).
    """
    h, w = mask.shape
    ix, iy = int(round(x)), int(round(y))
    
    # Check bounds
    if ix < border_margin or ix >= w - border_margin or iy < border_margin or iy >= h - border_margin:
        return 0.0
        
    if mask[iy, ix] == 0:
        return 0.0
        
    # Local coherence in a 9x9 neighborhood
    half = 4
    y1, y2 = max(0, iy - half), min(h, iy + half + 1)
    x1, x2 = max(0, ix - half), min(w, ix + half + 1)
    local_coh = np.mean(coherence_map[y1:y2, x1:x2])
    
    # Boundary distance factor (penalize points near outer edge of mask)
    # Check 4-connected margins
    dist_to_bg = border_margin
    for d in range(1, border_margin + 1):
        if (mask[iy - d, ix] == 0 or mask[iy + d, ix] == 0 or 
            mask[iy, ix - d] == 0 or mask[iy, ix + d] == 0):
            dist_to_bg = d
            break
            
    margin_factor = dist_to_bg / border_margin
    quality = float(np.clip(0.6 * local_coh + 0.4 * margin_factor, 0.0, 1.0))
    return quality
