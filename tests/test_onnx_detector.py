import numpy as np
import pytest
from src.minutiae.onnx_detector import ONNXMinutiaeDetector
from src.minutiae.extractor import Minutia


def test_onnx_detector_fallback():
    # When no model path is supplied, detector must gracefully fall back to Gabor-enhanced classical extraction
    detector = ONNXMinutiaeDetector(model_path=None, conf_thresh=0.35)
    assert not detector.is_model_loaded

    # Synthetic fingerprint image
    x, y = np.meshgrid(np.linspace(0, 20, 128), np.linspace(0, 20, 128))
    img = (127.5 * (1.0 + np.sin(x))).astype(np.uint8)
    mask = np.full((128, 128), 255, dtype=np.uint8)

    minutiae, skeleton = detector.detect(img, mask)
    assert isinstance(minutiae, list)
    assert skeleton.shape == (128, 128)
    for m in minutiae:
        assert isinstance(m, Minutia)
        assert 0.0 <= m.x < 128.0
        assert 0.0 <= m.y < 128.0
        assert 0.0 <= m.theta <= np.pi
        assert 0.0 <= m.quality <= 1.0


def test_onnx_detector_loaded_model_real_image():
    import os
    import cv2
    from src.preprocessing.segmenter import segment_fingerprint

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "minutiae_net_v1.onnx")
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found at {model_path}")

    detector = ONNXMinutiaeDetector(model_path=model_path, conf_thresh=0.35)
    assert detector.is_model_loaded

    img_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "fvc2002", "DB1_B", "101_1.tif")
    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found at {img_path}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, mask, _ = segment_fingerprint(img)

    # First inference
    minutiae_1, skel_1 = detector.detect(img, mask)
    assert len(minutiae_1) >= 15, f"Expected at least 15 minutiae, got {len(minutiae_1)}"
    assert skel_1.shape == img.shape

    # Orientation and boundary checks
    h, w = img.shape
    for m in minutiae_1:
        assert 0.0 <= m.x < float(w)
        assert 0.0 <= m.y < float(h)
        assert 0.0 <= m.theta < np.pi
        assert 0.35 <= m.quality <= 1.0

    # Deterministic inference check
    minutiae_2, _ = detector.detect(img, mask)
    assert len(minutiae_1) == len(minutiae_2)
    for m1, m2 in zip(minutiae_1, minutiae_2):
        assert m1.x == pytest.approx(m2.x, abs=1e-5)
        assert m1.y == pytest.approx(m2.y, abs=1e-5)
        assert m1.theta == pytest.approx(m2.theta, abs=1e-5)
        assert m1.quality == pytest.approx(m2.quality, abs=1e-5)


def test_rff_theoretical_kernel_mathematical_invariants():
    """
    Verifies that the closed-form Fourier square-wave mixture formula
    K_RFF(Delta) = (8 / pi^2) * sum_{k=0}^inf (1 / (2k+1)^2) exp(- (2k+1)^2 ||Delta||^2 / 2)
    accurately matches Monte Carlo 1-bit cosine RFF simulation across distances.
    """
    rng = np.random.default_rng(42)
    d = 7
    n_projections = 40000

    def theoretical_kernel(norm_val, max_k=20):
        total = 0.0
        for k in range(max_k):
            n = 2 * k + 1
            total += (1.0 / (n**2)) * np.exp(-0.5 * (n**2) * (norm_val**2))
        return (8.0 / (np.pi**2)) * total

    for delta_norm in [0.0, 0.4, 0.8, 1.2, 1.8, 2.5]:
        delta_vec = np.zeros(d, dtype=np.float64)
        delta_vec[0] = delta_norm

        W = rng.normal(0.0, 1.0, size=(n_projections, d))
        b = rng.uniform(0.0, 2.0 * np.pi, size=n_projections)

        u1 = (np.cos(b) >= 0.0).astype(np.int8)
        u2 = (np.cos(W @ delta_vec + b) >= 0.0).astype(np.int8)

        emp_ham = float(np.mean(u1 != u2))
        emp_sim = 1.0 - 2.0 * emp_ham
        theo_sim = theoretical_kernel(delta_norm)

        # Simulation error should be strictly bounded by 0.04 across all tested distances
        assert abs(emp_sim - theo_sim) < 0.04, f"Mismatch at norm {delta_norm}: empirical {emp_sim:.4f} vs theo {theo_sim:.4f}"
