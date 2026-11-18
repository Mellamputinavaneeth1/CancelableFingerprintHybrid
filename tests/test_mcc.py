import numpy as np
import pytest
from src.minutiae.extractor import Minutia
from src.descriptors.mcc import MinutiaeCylinderCode, flip_mcc_dual


def test_mcc_descriptor_shape_and_norm():
    mcc = MinutiaeCylinderCode(radius=80.0, num_spatial_cells=8, num_directional_slices=4)
    assert mcc.descriptor_dim == 256

    np.random.seed(42)
    minutiae = [
        Minutia(float(x), float(y), float(th), 0, 0.9)
        for (x, y), th in zip(
            np.random.uniform(50, 150, (10, 2)),
            np.random.uniform(0, np.pi, 10)
        )
    ]

    descs, weights = mcc.extract_descriptors(minutiae)
    assert descs.shape == (10, 256)
    assert weights.shape == (10,)
    for d in descs:
        norm = np.linalg.norm(d)
        if norm > 1e-4:
            assert np.isclose(norm, 1.0, atol=1e-5)


def test_mcc_se2_rigid_invariance():
    """Validates that MCC descriptors achieve exact SE(2) invariance under rotation and translation."""
    mcc = MinutiaeCylinderCode(radius=80.0, num_spatial_cells=8, num_directional_slices=4)
    
    np.random.seed(1234)
    pts = np.random.uniform(80, 180, (12, 2))
    thetas = np.random.uniform(0, np.pi, 12)
    
    minutiae_orig = [
        Minutia(float(x), float(y), float(th), 0, 0.9)
        for (x, y), th in zip(pts, thetas)
    ]

    # Rigid transformation: 35-degree rotation + translation
    alpha = np.radians(35.0)
    R = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha),  np.cos(alpha)]
    ])
    trans = np.array([60.0, -40.0])

    minutiae_trans = []
    for m in minutiae_orig:
        p_new = R @ np.array([m.x, m.y]) + trans
        th_new = (m.theta + alpha) % np.pi
        minutiae_trans.append(Minutia(float(p_new[0]), float(p_new[1]), float(th_new), 0, 0.9))

    descs_orig, _ = mcc.extract_descriptors(minutiae_orig)
    descs_trans, _ = mcc.extract_descriptors(minutiae_trans)
    descs_trans_dual = mcc.get_dual_descriptors(descs_trans)

    # Invariance check: maximum similarity between base and dual must be 1.0
    similarities = []
    for d1, d2, d2_dual in zip(descs_orig, descs_trans, descs_trans_dual):
        norm1 = np.linalg.norm(d1)
        norm2 = np.linalg.norm(d2)
        if norm1 > 1e-3 and norm2 > 1e-3:
            cos_sim = max(
                float(np.dot(d1, d2) / (norm1 * norm2)),
                float(np.dot(d1, d2_dual) / (norm1 * norm2))
            )
            similarities.append(cos_sim)

    assert len(similarities) >= 8
    mean_sim = np.mean(similarities)
    assert mean_sim > 0.999, f"Expected mean cosine similarity > 0.999, got {mean_sim}"
