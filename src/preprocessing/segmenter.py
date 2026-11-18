import numpy as np
import cv2

def segment_fingerprint(image: np.ndarray, block_size: int = 16, var_threshold: float = 120.0,
                        method: str = "variance"):
    """
    Segments the fingerprint foreground (ROI) from the background.
    
    Args:
        image: Grayscale uint8 fingerprint image.
        block_size: Size of the local analysis block in pixels.
        var_threshold: Variance threshold above which a block is marked as foreground.
        method: 'variance' (legacy fixed-threshold) or 'adaptive' (gradient-energy +
                relative local variance with hole filling).
        
    Returns:
        segmented_image: Image with background zeroed out.
        mask: Binary foreground mask (uint8, values 0 or 255).
        roi_bbox: (x, y, w, h) bounding box of the active fingerprint region.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if method == "adaptive":
        img_f = gray.astype(np.float32)
        mean_map = cv2.boxFilter(img_f, cv2.CV_32F, (block_size, block_size), normalize=True)
        sqmean_map = cv2.boxFilter(img_f * img_f, cv2.CV_32F, (block_size, block_size), normalize=True)
        var_map = np.maximum(0.0, sqmean_map - mean_map * mean_map)

        gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
        grad_energy = cv2.boxFilter(np.sqrt(gx * gx + gy * gy), cv2.CV_32F, (block_size, block_size), normalize=True)

        # Combine local contrast variance and ridge gradient energy, scaled relative to image ridge contrast
        ref_var = float(np.percentile(var_map, 90))
        ref_grad = float(np.percentile(grad_energy, 90))
        eff_var_th = max(40.0, min(var_threshold, 0.12 * ref_var))
        eff_grad_th = max(6.0, 0.15 * ref_grad)

        fg_pixels = (var_map >= eff_var_th) & (grad_energy >= eff_grad_th)
        mask = (fg_pixels.astype(np.uint8) * 255)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block_size + 1, block_size + 1))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    else:
        # Calculate local block variance (legacy canonical loop)
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = gray[y:min(y+block_size, h), x:min(x+block_size, w)]
                if block.size == 0:
                    continue
                std = np.std(block)
                if std * std >= var_threshold:
                    mask[y:min(y+block_size, h), x:min(x+block_size, w)] = 255
                    
        # Morphological closing and opening to eliminate small holes and isolated noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Extract largest connected component to ensure contiguous ROI
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_idx).astype(np.uint8) * 255
        if method == "adaptive":
            # Fill interior holes inside the primary fingerprint ROI
            inv = cv2.bitwise_not(mask)
            n_h, l_h, s_h, _ = cv2.connectedComponentsWithStats(inv)
            if n_h > 2:
                bg_idx = 1 + np.argmax(s_h[1:, cv2.CC_STAT_AREA])
                holes = (l_h > 0) & (l_h != bg_idx)
                mask[holes] = 255
        x = stats[largest_idx, cv2.CC_STAT_LEFT]
        y = stats[largest_idx, cv2.CC_STAT_TOP]
        w_box = stats[largest_idx, cv2.CC_STAT_WIDTH]
        h_box = stats[largest_idx, cv2.CC_STAT_HEIGHT]
        roi_bbox = (x, y, w_box, h_box)
    else:
        roi_bbox = (0, 0, w, h)
        
    segmented_image = cv2.bitwise_and(gray, gray, mask=mask)
    return segmented_image, mask, roi_bbox
