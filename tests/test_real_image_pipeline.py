import os
import cv2
import numpy as np
import pytest

from src.pipeline import CancelableFingerprintPipeline
from src.graph.delaunay import build_delaunay_triangles, build_knn_graphs
from src.graph.descriptors import extract_all_descriptors

DB1_SAMPLE = os.path.join("dataset", "FVC2002", "DB1_B", "101_1.tif")
DB2_SAMPLE = os.path.join("dataset", "FVC2002", "DB2_B", "101_1.tif")

@pytest.mark.skipif(not os.path.exists(DB1_SAMPLE), reason="FVC2002 DB1_B sample not found")
def test_db1_real_image_pipeline():
    """Gate 7.2: Verify real fingerprint pipeline processing on FVC2002 DB1_B (500 DPI)."""
    img = cv2.imread(DB1_SAMPLE, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    assert img.shape == (374, 388)
    
    pipe = CancelableFingerprintPipeline(in_dim=12, dpi=500.0)
    features = pipe.extract_features(img)
    
    assert features["mask"].sum() > 0, "Segmentation mask must cover positive area"
    assert features["enhanced"].shape == img.shape
    assert features["skeleton"].sum() > 0, "Skeletonization must produce non-zero ridge pixels"
    assert len(features["minutiae"]) >= 10, f"Expected >= 10 minutiae, got {len(features['minutiae'])}"
    assert len(features["triangles"]) >= 5, f"Expected >= 5 Delaunay triangles, got {len(features['triangles'])}"
    
    descs = features["descriptors"]
    weights = features["weights"]
    
    assert descs.ndim == 2 and descs.shape[1] == 12, f"Descriptors shape {descs.shape} must be (N, 12)"
    assert len(weights) == len(descs)
    assert np.all(weights > 0.0) and np.all(weights <= 1.0), "Quality weights must be in (0, 1]"
    
    # Check physical perimeter bounds (P_mm should be realistic, e.g., 1.5mm to 30mm)
    p_mm = descs[:, 2]
    assert np.all(p_mm > 0.5) and np.all(p_mm < 40.0), f"Physical perimeters out of range: {p_mm.min()} to {p_mm.max()}"


@pytest.mark.skipif(not os.path.exists(DB2_SAMPLE), reason="FVC2002 DB2_B sample not found")
def test_db2_real_image_pipeline():
    """Gate 7.2: Verify real fingerprint pipeline processing on FVC2002 DB2_B (569 DPI) with 9D descriptors."""
    img = cv2.imread(DB2_SAMPLE, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    assert img.shape == (560, 296)
    
    pipe = CancelableFingerprintPipeline(in_dim=9, dpi=569.0)
    features = pipe.extract_features(img)
    
    assert len(features["minutiae"]) >= 10
    assert len(features["triangles"]) >= 5
    
    descs = features["descriptors"]
    assert descs.ndim == 2 and descs.shape[1] == 9, f"Descriptors shape {descs.shape} must be (N, 9)"


@pytest.mark.skipif(not os.path.exists(DB1_SAMPLE), reason="FVC2002 DB1_B sample not found")
def test_knn_topology_on_real_image():
    """Verifies k-NN minutiae graph generation on real image."""
    img = cv2.imread(DB1_SAMPLE, cv2.IMREAD_GRAYSCALE)
    pipe = CancelableFingerprintPipeline(in_dim=12, dpi=500.0)
    features = pipe.extract_features(img)
    
    knn_triplets = build_knn_graphs(features["minutiae"], k=5, max_edge_length=120.0)
    assert len(knn_triplets) > len(features["triangles"]), "k-NN should form redundant local triplets"
    
    descs, weights = extract_all_descriptors(knn_triplets, dpi=500.0, include_types=True)
    assert descs.shape == (len(knn_triplets), 12)
    assert len(weights) == len(knn_triplets)


def test_fte_fta_handling():
    """Gate 7.2: Verify pipeline gracefully accounts for Failure-to-Acquire and Failure-to-Enroll."""
    pipe = CancelableFingerprintPipeline(in_dim=12, dpi=500.0)
    
    # 1. Blank image (FTA: zero fingerprint signal)
    blank_img = np.zeros((300, 300), dtype=np.uint8)
    feat = pipe.extract_features(blank_img)
    assert len(feat["minutiae"]) == 0 or len(feat["triangles"]) == 0
    assert len(feat["descriptors"]) == 0
    
    # Enrollment should return error status without crashing
    enr_res = pipe.enroll(blank_img, salt="salt1", app_id="test_app")
    assert enr_res["status"] == "error"
    assert "Insufficient minutiae" in enr_res["message"]
    
    # Verification with empty features should return REJECT with score 0.0
    ver_res = pipe.verify_from_features(feat["descriptors"], feat["weights"], 
                                        {"salt": "s", "app_id": "a", "bitstrings": np.zeros((5, 32)), "weights": np.ones(5)})
    assert ver_res["decision"] == "REJECT"
    assert ver_res["score"] == 0.0


@pytest.mark.skipif(not os.path.exists(DB1_SAMPLE), reason="FVC2002 DB1_B sample not found")
def test_end_to_end_real_image_enroll_verify():
    """Verifies end-to-end enrollment and verification on real prints."""
    pipe = CancelableFingerprintPipeline(in_dim=12, dpi=500.0)
    img_enr = cv2.imread(DB1_SAMPLE, cv2.IMREAD_GRAYSCALE)
    
    enr = pipe.enroll(img_enr, salt="test_salt_fvc", app_id="fvc_app", subject_id="subj_101")
    assert enr["status"] == "success"
    assert "bitstrings" in enr["template"]
    assert enr["template"]["bitstrings"].shape[1] == 32
    
    # Self-verification must produce high score and ACCEPT
    ver_self = pipe.verify(img_enr, enr["template"])
    assert ver_self["decision"] == "ACCEPT"
    assert ver_self["score"] > 0.50
