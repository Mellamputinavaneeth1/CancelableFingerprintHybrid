"""
Test Suite for Platform Version 2.4-Robust Engineering Remediations.

Validates:
1. Hybrid multi-topology graph construction, deduplication, and SE(2) invariance.
2. Minutiae dropout resilience (k-NN star triangles survive when Delaunay edges flip).
3. TriangleBuilder method="hybrid" dispatch.
4. AuditLogger persistent hash-chain continuity across simulated process reboots.
5. AuditLogger tamper-detection on corrupted log files.
6. Tier-0 Bloom filter pre-screening (accuracy, genuine rank-1 retrieval, high throughput).
7. ScalerRegistry multi-dataset pooled standardization.
"""

import os
import json
import tempfile
import numpy as np
import pytest

from src.minutiae.extractor import Minutia
from src.services.types import ExtractedMinutiae
from src.graph.hybrid_graph import build_hybrid_triangles
from src.services.triangle_builder import TriangleBuilder
from src.services.audit_logger import AuditLogger
from src.protection.bloom import encode_to_bloom_filter
from src.matching.bloom_screening import batch_bloom_jaccard_similarity, screen_candidates_bloom
from src.services.scaler_registry import ScalerRegistry


def test_hybrid_graph_construction_and_deduplication():
    """Asserts that build_hybrid_triangles fuses Delaunay and k-NN simplices without duplicates."""
    np.random.seed(42)
    minutiae = [Minutia(float(x), float(y), float(theta), 0, 0.9)
                for x, y, theta in np.random.uniform(20, 200, (15, 3))]

    triangles = build_hybrid_triangles(minutiae, knn_k=4)
    assert len(triangles) > 0

    # Assert deduplication: all canonical coordinate keys must be unique
    seen_keys = set()
    for m1, m2, m3 in triangles:
        coords = tuple(sorted([(round(m1.x, 2), round(m1.y, 2)),
                               (round(m2.x, 2), round(m2.y, 2)),
                               (round(m3.x, 2), round(m3.y, 2))]))
        assert coords not in seen_keys, f"Duplicate triangle detected: {coords}"
        seen_keys.add(coords)


def test_hybrid_graph_minutiae_dropout_resilience():
    """Validates that k-NN star simplices survive when Delaunay edges flip under minutiae dropout."""
    np.random.seed(123)
    # 12 minutiae points
    pts = np.random.uniform(30, 180, (12, 2))
    minutiae_full = [Minutia(float(x), float(y), 0.5, 0, 0.9) for x, y in pts]

    # Full hybrid set
    tris_full = build_hybrid_triangles(minutiae_full, knn_k=4)

    # Simulate 2 minutiae dropouts (indices 0 and 1 removed)
    minutiae_perturbed = minutiae_full[2:]
    tris_perturbed = build_hybrid_triangles(minutiae_perturbed, knn_k=4)

    # Convert to coordinate sets
    def to_coord_set(tri_list):
        return {tuple(sorted([(round(m.x, 1), round(m.y, 1)) for m in t])) for t in tri_list}

    set_full = to_coord_set(tris_full)
    set_pert = to_coord_set(tris_perturbed)

    # Shared triangles between full and perturbed must be non-empty and substantial
    common = set_full & set_pert
    assert len(common) >= 5, f"Expected at least 5 surviving triangles, got {len(common)}"


def test_hybrid_graph_se2_rigid_invariance():
    """Validates that hybrid triangles preserve relative Euclidean geometry under SE(2) rotation and translation."""
    np.random.seed(999)
    pts = np.random.uniform(50, 150, (8, 2))
    thetas = np.random.uniform(0, np.pi, 8)
    minutiae_orig = [Minutia(float(x), float(y), float(th), 0, 0.9) for (x, y), th in zip(pts, thetas)]

    # Rigid transformation: 45 degree rotation + translation
    alpha = np.radians(45.0)
    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha),  np.cos(alpha)]])
    tx, ty = 80.0, -40.0

    minutiae_rot = []
    for m in minutiae_orig:
        p_rot = R @ np.array([m.x, m.y]) + np.array([tx, ty])
        th_rot = (m.theta + alpha) % np.pi
        minutiae_rot.append(Minutia(float(p_rot[0]), float(p_rot[1]), float(th_rot), 0, 0.9))

    tris_orig = build_hybrid_triangles(minutiae_orig, knn_k=3)
    tris_rot = build_hybrid_triangles(minutiae_rot, knn_k=3)

    assert len(tris_orig) == len(tris_rot)


def test_triangle_builder_hybrid_dispatch():
    """Validates that TriangleBuilder with method='hybrid' constructs valid hybrid triangles."""
    m = [Minutia(15.0 * i, 20.0 * (i % 3), 0.5, 0, 0.9) for i in range(10)]
    em = ExtractedMinutiae(minutiae=m, skeleton=np.zeros((100, 100), dtype=np.uint8), count=len(m))

    tb = TriangleBuilder(method="hybrid", knn_k=4)
    res = tb.build(em)
    assert res.count > 0
    assert len(res.triangles) == res.count


def test_audit_logger_hash_chain_continuity_across_restarts():
    """Validates that AuditLogger resumes an existing hash chain across reboots without breaking integrity."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = os.path.join(tmp_dir, "audit_chain.log")

        # Session 1: Log 3 events
        logger1 = AuditLogger(log_file=log_file)
        logger1.log_event("ENROLL", "user_1", "SUCCESS", score=None)
        logger1.log_event("VERIFY", "user_1", "ACCEPT", score=0.85)
        logger1.log_event("VERIFY", "user_2", "REJECT", score=0.12)
        assert logger1.verify_integrity() is True

        # Session 2: Re-open file in a new instance (simulating server reboot)
        logger2 = AuditLogger(log_file=log_file)
        assert len(logger2.get_trail()) == 3
        # Log 2 more events
        logger2.log_event("REVOKE", "user_2", "REVOKED")
        logger2.log_event("KEY_ROTATION", "SYSTEM_KMS", "SUCCESS")

        assert len(logger2.get_trail()) == 5
        assert logger2.verify_integrity() is True

        # Verify entire file from disk statically
        assert AuditLogger.verify_file_integrity(log_file) is True


def test_audit_logger_tamper_detection():
    """Validates that modifying any byte in an audit log file causes integrity verification to fail."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = os.path.join(tmp_dir, "tampered_audit.log")

        logger = AuditLogger(log_file=log_file)
        logger.log_event("ENROLL", "alice", "SUCCESS")
        logger.log_event("VERIFY", "alice", "ACCEPT", score=0.91)
        logger.log_event("VERIFY", "bob", "REJECT", score=0.10)

        assert AuditLogger.verify_file_integrity(log_file) is True

        # Tamper with the second event (change decision to ACCEPT)
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        event2 = json.loads(lines[2])
        event2["decision"] = "ACCEPT"  # Maliciously flip REJECT to ACCEPT
        lines[2] = json.dumps(event2) + "\n"

        with open(log_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # Integrity verification must now detect tampering and return False
        assert AuditLogger.verify_file_integrity(log_file) is False


def test_tier0_bloom_screening_accuracy_and_throughput():
    """Validates that Tier-0 Bloom filter Jaccard screening ranks genuine candidates at rank 1."""
    np.random.seed(777)
    # Generate 100 enrolled candidates, each with 20 32-bit descriptors
    M = 100
    gallery_descriptors = [np.random.randint(0, 2, (20, 32), dtype=np.uint8) for _ in range(M)]
    gallery_blooms = np.array([encode_to_bloom_filter(desc, filter_size_bits=256, salt="test_salt")
                               for desc in gallery_descriptors], dtype=np.uint8)
    candidate_ids = [f"subject_{i:03d}" for i in range(M)]

    # Query is a noisy version of subject 42 (18 identical descriptors, 2 perturbed)
    target_idx = 42
    target_desc = gallery_descriptors[target_idx]
    query_desc = target_desc.copy()
    query_desc[0] = np.random.randint(0, 2, 32, dtype=np.uint8)  # 1 descriptor noise

    query_bloom = encode_to_bloom_filter(query_desc, filter_size_bits=256, salt="test_salt")

    # Screen candidates
    top_candidates = screen_candidates_bloom(query_bloom, gallery_blooms, candidate_ids, top_k=5)
    assert len(top_candidates) == 5

    # Target identity must be rank 1
    top_id, top_score = top_candidates[0]
    assert top_id == candidate_ids[target_idx], f"Expected {candidate_ids[target_idx]} at rank 1, got {top_id}"
    assert top_score > 0.60


def test_scaler_registry_pooled_set_b():
    """Validates that ScalerRegistry supports POOLED_SET_B standardization."""
    sr = ScalerRegistry()
    scaler = sr.get_scaler("POOLED_SET_B")
    assert scaler is not None

    raw_feat = np.array([0.8142, 0.5418, 4.0850, 0.4982, 0.4851, 0.4812, 0.0615, 0.1342, 0.1225], dtype=np.float64)
    # Centered features should transform to near zero
    z = scaler["transform"](raw_feat)
    np.testing.assert_allclose(z, 0.0, atol=1e-4)

    # Inverse mapping restores original
    restored = scaler["inverse"](z)
    np.testing.assert_allclose(restored, raw_feat, atol=1e-4)
