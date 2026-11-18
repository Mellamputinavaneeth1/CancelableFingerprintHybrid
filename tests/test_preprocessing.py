import numpy as np
import pytest
from src.preprocessing.segmenter import segment_fingerprint
from src.preprocessing.enhancer import enhance_fingerprint, estimate_orientation_field

def test_segment_fingerprint():
    # Create synthetic image with a central textured block and zero borders
    img = np.zeros((100, 100), dtype=np.uint8)
    # Add random high variance in the center
    rng = np.random.default_rng(42)
    img[20:80, 20:80] = rng.integers(50, 200, size=(60, 60), dtype=np.uint8)
    
    segmented, mask, bbox = segment_fingerprint(img, block_size=8, var_threshold=100.0)
    assert mask.shape == (100, 100)
    assert np.any(mask > 0)
    # Background corners should be 0
    assert mask[5, 5] == 0
    assert mask[95, 95] == 0
    # Center should be foreground (255)
    assert mask[50, 50] == 255
    assert bbox[2] > 0 and bbox[3] > 0

def test_enhance_fingerprint():
    # Create synthetic ridge pattern
    x, y = np.meshgrid(np.linspace(0, 10, 128), np.linspace(0, 10, 128))
    synthetic_ridges = (127.5 * (1.0 + np.sin(x + y))).astype(np.uint8)
    
    enhanced, orientations, coherence = enhance_fingerprint(synthetic_ridges)
    assert enhanced.shape == (128, 128)
    assert orientations.shape == (128, 128)
    assert coherence.shape == (128, 128)
    assert np.all(orientations >= 0.0) and np.all(orientations <= np.pi)
    assert np.all(coherence >= 0.0) and np.all(coherence <= 1.0)


def test_gabor_filter_bank_enhancement():
    # Synthetic oriented sinusoidal grating
    x, y = np.meshgrid(np.linspace(0, 20, 128), np.linspace(0, 20, 128))
    grating = (127.5 * (1.0 + np.sin(x))).astype(np.uint8)
    mask = np.full((128, 128), 255, dtype=np.uint8)

    enhanced, orientations, coherence = enhance_fingerprint(grating, use_gabor=True, mask=mask)
    assert enhanced.shape == (128, 128)
    assert orientations.shape == (128, 128)
    assert coherence.shape == (128, 128)
    assert enhanced.dtype == np.uint8
    # Filtered output must have ridge-valley contrast
    assert np.max(enhanced) > np.min(enhanced)
