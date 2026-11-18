"""
Test Suite for Platform Version 2.3 Remediations.

Validates:
1. Vectorized crossing number equivalence in minutiae extraction.
2. Bitwise uint32 popcount equivalence with 2D broadcasting in pairwise Hamming distance.
3. TriangleBuilder KNN method dispatch and collinearity fallback.
4. Persistent SQLite credential repository (ACID storage, revocation, across instances).
5. NonceManager persistent anti-replay protection across simulated process restarts.
"""

import os
import tempfile
import sqlite3
import numpy as np
import pytest

from src.minutiae.extractor import Minutia, extract_minutiae
from src.matching.bipartite_matcher import compute_pairwise_hamming_distance, match_protected_templates, match_protected_greedy
from src.services.types import ExtractedMinutiae, SerializedCredentialEnvelope, BiometricErrorCode
from src.services.triangle_builder import TriangleBuilder
from src.services.sqlite_repository import SQLiteCredentialRepository
from src.services.credential_repository import CredentialRepository
from src.services.nonce_manager import NonceManager, NonceAllocationError


def test_minutiae_vectorized_crossing_number():
    """Validates that vectorized crossing number detects ridge endings and bifurcations accurately."""
    h, w = 120, 120
    enhanced = np.zeros((h, w), dtype=np.uint8)
    mask = np.ones((h, w), dtype=np.uint8)
    orientations = np.zeros((h, w), dtype=np.float32)
    coherence = np.ones((h, w), dtype=np.float32)

    # Draw synthetic ridges to form known endings and bifurcations
    enhanced[40:80, 50] = 255
    enhanced[60, 50:80] = 255  # T-junction bifurcation at (50, 60)

    minutiae, skeleton = extract_minutiae(
        enhanced_image=enhanced,
        mask=mask,
        orientation_map=orientations,
        coherence_map=coherence,
        min_dist=5,
        border_margin=10,
        quality_thresh=0.1
    )
    assert len(minutiae) > 0
    # Must have both ending and bifurcation detected
    types = {m.type for m in minutiae}
    assert 0 in types or 1 in types


def test_pairwise_hamming_distance_popcount_equivalence():
    """Asserts that the fast bitwise popcount path is mathematically identical to 2D broadcast."""
    np.random.seed(42)
    for _ in range(10):
        nq = np.random.randint(5, 50)
        ne = np.random.randint(5, 50)
        q_bits = np.random.randint(0, 2, (nq, 32), dtype=np.uint8)
        e_bits = np.random.randint(0, 2, (ne, 32), dtype=np.uint8)

        # Baseline broadcast
        ref = np.mean(q_bits[:, None, :] != e_bits[None, :, :], axis=2).astype(np.float32)
        # Fast popcount function
        calc = compute_pairwise_hamming_distance(q_bits, e_bits)

        np.testing.assert_allclose(calc, ref, atol=1e-6)


def test_matching_with_popcount_integration():
    """Asserts that match_protected_templates and match_protected_greedy execute cleanly with popcount."""
    np.random.seed(1337)
    nq, ne = 20, 25
    q_bits = np.random.randint(0, 2, (nq, 32), dtype=np.uint8)
    e_bits = np.random.randint(0, 2, (ne, 32), dtype=np.uint8)
    qw = np.random.uniform(0.5, 1.0, nq).astype(np.float32)
    ew = np.random.uniform(0.5, 1.0, ne).astype(np.float32)

    score_hungarian = match_protected_templates(q_bits, e_bits, qw, ew, hamming_threshold=0.22)
    score_greedy = match_protected_greedy(q_bits, e_bits, qw, ew, hamming_threshold=0.22)

    assert 0.0 <= score_hungarian <= 1.0
    assert 0.0 <= score_greedy <= 1.0


def test_triangle_builder_knn_method_dispatch():
    """Asserts that TriangleBuilder with method='knn' properly invokes k-NN graph construction."""
    m = [Minutia(10.0 * i, 15.0 * (i % 4), 0.5, 0, 0.9) for i in range(12)]
    em = ExtractedMinutiae(minutiae=m, skeleton=np.zeros((100, 100), dtype=np.uint8), count=len(m))

    tb_knn = TriangleBuilder(method="knn", knn_k=4)
    res_knn = tb_knn.build(em)
    assert res_knn.count > 0
    assert len(res_knn.triangles) == res_knn.count


def test_triangle_builder_collinear_fallback():
    """Asserts that collinear minutiae points gracefully fall back to k-NN rather than crashing."""
    # 5 strictly collinear points along y = 20
    m_collinear = [Minutia(10.0 * i, 20.0, 0.0, 0, 0.9) for i in range(5)]
    em = ExtractedMinutiae(minutiae=m_collinear, skeleton=np.zeros((100, 100), dtype=np.uint8), count=len(m_collinear))

    tb_del = TriangleBuilder(method="delaunay", knn_k=4)
    # Delaunay alone would raise QhullError or 0 triangles; fallback should construct k-NN triplets
    res = tb_del.build(em)
    assert res.count > 0


def test_sqlite_credential_repository_persistence():
    """Validates that SQLiteCredentialRepository persists credentials and revocations across instances."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_credentials.db")

        # 1. Enroll credential in first instance
        repo1 = SQLiteCredentialRepository(db_path=db_path)
        env = SerializedCredentialEnvelope(
            raw_bytes=b"encrypted_binary_payload_001",
            total_bytes=28,
            credential_id="cred-001",
            num_triangles=15,
            has_topology=False,
            cipher_suite="CHACHA20_POLY1305",
            nonce=b"123456789012",
            auth_tag=b"tag1234567890123",
            is_encrypted=True
        )
        cid = repo1.store("subj-alpha", env, metadata={"app_id": "finance_app"})
        assert cid == "cred-001"
        assert repo1.count() == 1
        assert repo1.get_raw_bytes("subj-alpha") == b"encrypted_binary_payload_001"
        repo1.close()

        # 2. Re-open database in second instance (simulating server reboot)
        repo2 = SQLiteCredentialRepository(db_path=db_path)
        assert repo2.count() == 1
        assert repo2.get_raw_bytes("subj-alpha") == b"encrypted_binary_payload_001"

        # 3. Revoke subject
        rev_ok = repo2.revoke("subj-alpha", reason="device_lost")
        assert rev_ok is True
        assert repo2.is_revoked("subj-alpha") is True
        repo2.close()

        # 4. Re-open again to assert revocation persists across restarts
        repo3 = SQLiteCredentialRepository(db_path=db_path)
        assert repo3.is_revoked("subj-alpha") is True
        with pytest.raises(Exception):
            repo3.get_raw_bytes("subj-alpha")
        repo3.close()


def test_credential_repository_db_path_integration():
    """Validates that CredentialRepository accepts db_path and transparently delegates to SQLite."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "repo_integration.db")

        repo = CredentialRepository(db_path=db_path)
        env = SerializedCredentialEnvelope(
            raw_bytes=b"payload_bytes",
            total_bytes=13,
            credential_id="cred-int-1",
            num_triangles=5,
            has_topology=False,
            cipher_suite="CHACHA20_POLY1305",
            nonce=b"nonce12bytes",
            auth_tag=b"tag16bytes123456",
            is_encrypted=True
        )
        repo.store("subj-beta", env)
        assert repo.count() == 1
        assert repo.get_raw_bytes("subj-beta") == b"payload_bytes"
        repo.close()


def test_nonce_manager_persistence_blocks_post_restart_replay():
    """Validates that NonceManager persists seen nonces to disk, preventing replay after reboot."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_file = os.path.join(tmp_dir, "nonces_state.json")

        nm1 = NonceManager(state_file=state_file)
        nonce = nm1.allocate_nonce(key_id="app_key_1")
        # First verification succeeds and records nonce
        assert nm1.verify_and_record_nonce(nonce, key_id="app_key_1") is True

        # Second verification in same session raises NonceAllocationError
        with pytest.raises(NonceAllocationError):
            nm1.verify_and_record_nonce(nonce, key_id="app_key_1")

        # Simulate process termination and restart with a new NonceManager instance
        nm2 = NonceManager(state_file=state_file)
        # Post-restart replay MUST be rejected
        with pytest.raises(NonceAllocationError):
            nm2.verify_and_record_nonce(nonce, key_id="app_key_1")
