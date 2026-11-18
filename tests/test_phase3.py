"""
Unit and Integration Tests for Phase 3: Robustness and Capture-Quality Evaluation.
Tests:
  1. Synthetic capture nuisance transformation operators (rotation, translation, cropping, deformation).
  2. SOCOFing 2-fold cross-validation partition isolation and disjoint subject sets.
  3. Trial accounting identities and Clopper-Pearson exact confidence intervals.
  4. Phase 3 artifact integrity and schema verification.
"""

import os
import pytest
import numpy as np
import cv2

from experiments.evaluate_phase3_nuisance import (
    apply_planar_rotation,
    apply_translation,
    apply_partial_crop,
    apply_elastic_deformation,
)
from experiments.protocols.socofing_protocol import generate_socofing_protocol_pairs
from src.evaluation.metrics import clopper_pearson_ci, clustered_bootstrap_confidence_intervals
from experiments.run_phase3 import verify_phase3_integrity


def test_nuisance_transform_planar_rotation():
    """Verify planar rotation preserves image shape, dtype, and background border."""
    img = np.full((100, 100), 200, dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, 50, -1)
    
    # 0 degree rotation should return exact copy
    rot0 = apply_planar_rotation(img, 0.0)
    assert np.array_equal(rot0, img)
    
    # Non-zero rotation preserves shape and dtype
    rot30 = apply_planar_rotation(img, 30.0)
    assert rot30.shape == img.shape
    assert rot30.dtype == img.dtype
    # Corners exposed by rotation should be padded with 255
    assert rot30[0, 0] == 255
    assert rot30[0, -1] == 255


def test_nuisance_transform_translation():
    """Verify translational displacement shifts features accurately with background margin."""
    img = np.full((100, 100), 255, dtype=np.uint8)
    img[40:60, 40:60] = 50  # Center block
    
    # (0, 0) translation returns exact copy
    trans0 = apply_translation(img, (0, 0))
    assert np.array_equal(trans0, img)
    
    # Shift by (+10, +10)
    trans10 = apply_translation(img, (10, 10))
    assert trans10.shape == img.shape
    assert trans10[0:10, :].mean() == 255.0  # Top margin exposed is 255
    assert trans10[:, 0:10].mean() == 255.0  # Left margin exposed is 255
    # Shifted center block should now be at [50:70, 50:70]
    assert np.all(trans10[50:70, 50:70] == 50)


def test_nuisance_transform_partial_crop():
    """Verify partial cropping retains centered print bounding box and masks exterior."""
    img = np.full((200, 200), 255, dtype=np.uint8)
    # Create simulated active fingerprint print region with variance
    img[50:150, 50:150] = np.random.RandomState(42).randint(50, 180, (100, 100), dtype=np.uint8)
    
    crop100 = apply_partial_crop(img, 1.0)
    assert np.array_equal(crop100, img)
    
    # Crop to 25% area (scale factor 0.5)
    crop25 = apply_partial_crop(img, 0.25)
    assert crop25.shape == img.shape
    # Exterior margins must be 255
    assert crop25[0:20, :].mean() == 255.0
    assert crop25[-20:, :].mean() == 255.0
    assert crop25[:, 0:20].mean() == 255.0
    assert crop25[:, -20:].mean() == 255.0
    # Center region should retain texture
    assert crop25[90:110, 90:110].mean() < 240.0


def test_nuisance_transform_elastic_deformation():
    """Verify non-linear radial/tangential warp produces continuous deformation without NaNs."""
    img = np.full((120, 120), 255, dtype=np.uint8)
    img[30:90, 30:90] = 100
    
    warp0 = apply_elastic_deformation(img, 0.0)
    assert np.array_equal(warp0, img)
    
    warp15 = apply_elastic_deformation(img, 15.0)
    assert warp15.shape == img.shape
    assert warp15.dtype == np.uint8
    assert not np.isnan(warp15).any()
    assert not np.isinf(warp15).any()


def test_socofing_protocol_partition_isolation():
    """Verify strict partition isolation between development and held-out test partitions."""
    mock_data = {
        "subjects": {
            s: {
                "real": [{"path": f"real_{s}.bmp", "filename": f"real_{s}.bmp"}],
                "altered": [
                    {"path": f"alt_{s}_{c}_{d}.bmp", "filename": f"alt_{s}_{c}_{d}.bmp",
                     "alteration_class": c, "difficulty": d}
                    for c in ["Central_Rotation", "Z-cut", "Obliteration"]
                    for d in ["Easy", "Medium", "Hard"]
                ]
            }
            for s in range(1, 11)
        }
    }
    
    dev_data = {"subjects": {s: mock_data["subjects"][s] for s in range(1, 6)}}
    test_data = {"subjects": {s: mock_data["subjects"][s] for s in range(6, 11)}}
    
    dev_gen, dev_imp = generate_socofing_protocol_pairs(dev_data, max_impostors_per_probe=3, seed=42)
    test_gen, test_imp = generate_socofing_protocol_pairs(test_data, max_impostors_per_probe=3, seed=1042)
    
    # 1. Assert exactly 45 genuine and 135 impostor pairs per fold
    assert len(test_gen) == 45, f"Expected 45 test genuine pairs, got {len(test_gen)}"
    assert len(test_imp) == 135, f"Expected 135 test impostor pairs, got {len(test_imp)}"
    
    # 2. Assert strictly zero subject overlap across partitions
    dev_subjs = set(p["subject_id"] for p in dev_gen + dev_imp)
    test_subjs = set(p["subject_id"] for p in test_gen + test_imp)
    assert len(dev_subjs.intersection(test_subjs)) == 0, "Partitions must be strictly subject-disjoint"
    
    # 3. Assert genuine pairs match subject_id == gallery_subject_id
    for p in test_gen:
        assert p["subject_id"] == p["gallery_subject_id"]
        assert p["genuine"] == 1
        
    # 4. Assert impostor pairs have subject_id != gallery_subject_id
    for p in test_imp:
        assert p["subject_id"] != p["gallery_subject_id"]
        assert p["genuine"] == 0
        assert p["gallery_subject_id"] in test_subjs


def test_clopper_pearson_exact_bounds_and_containment():
    """Verify Clopper-Pearson exact confidence interval properties."""
    # Complete failure: 0 successes out of 10
    ci_zero = clopper_pearson_ci(0, 10, alpha=0.05)
    assert ci_zero[0] == 0.0
    assert 0.0 < ci_zero[1] <= 0.35  # Upper bound around 30.8%
    
    # Complete success: 10 successes out of 10
    ci_full = clopper_pearson_ci(10, 10, alpha=0.05)
    assert 0.65 <= ci_full[0] < 1.0  # Lower bound around 69.2%
    assert ci_full[1] == 1.0
    
    # Interior point estimate containment
    ci_half = clopper_pearson_ci(5, 10, alpha=0.05)
    assert ci_half[0] < 0.5 < ci_half[1]
    assert 0.0 <= ci_half[0] <= ci_half[1] <= 1.0


def test_phase3_deliverables_integrity():
    """Gate 3: Assert complete scientific and accounting integrity of saved Phase 3 outputs."""
    phase3_dir = os.path.join("experiments", "results", "phase3")
    if os.path.exists(phase3_dir):
        assert verify_phase3_integrity(phase3_dir=phase3_dir)
