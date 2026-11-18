import numpy as np
import pytest
from src.minutiae.extractor import extract_minutiae, Minutia, binarize_and_thin
from src.minutiae.quality import calculate_minutia_quality

def test_minutia_quality():
    mask = np.ones((100, 100), dtype=np.uint8) * 255
    coherence = np.ones((100, 100), dtype=np.float32) * 0.8
    
    # Center minutia
    q_center = calculate_minutia_quality(50, 50, mask, coherence, border_margin=15)
    assert q_center > 0.5
    
    # Boundary minutia (within border margin)
    q_border = calculate_minutia_quality(5, 5, mask, coherence, border_margin=15)
    assert q_border == 0.0

def test_minutiae_extraction():
    # Construct a synthetic ridge ending
    skeleton = np.zeros((64, 64), dtype=np.uint8)
    skeleton[32, 10:32] = 255  # Horizontal ridge terminating at (32, 31)
    
    mask = np.ones((64, 64), dtype=np.uint8) * 255
    orientation = np.zeros((64, 64), dtype=np.float32)
    coherence = np.ones((64, 64), dtype=np.float32)
    
    # Test on enhanced image simulating the skeleton
    enhanced = skeleton.copy()
    minutiae, skel = extract_minutiae(enhanced, mask, orientation, coherence, 
                                      min_dist=5, border_margin=5, quality_thresh=0.1)
    
    # Should detect at least one termination
    assert len(minutiae) >= 1
    m = minutiae[0]
    assert isinstance(m, Minutia)
    assert m.type in (0, 1)
    assert m.quality > 0.0
