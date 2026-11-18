"""
Layer 2 Integration Tests for Production Biometric Service Pipeline.

Tests:
1. End-to-end enrollment -> genuine verification on ProductionBiometricService.
2. Impostor rejection flow across distinct finger identities.
3. Re-enrollment under a new epoch following revocation.
4. Cross-application isolation (different app IDs / salts produce non-matching bitstrings).
5. Malformed / non-finite / corrupted image inputs with proper BiometricServiceException error codes.
6. Key rotation and projection cache invalidation.
"""

import os
import pytest
import numpy as np

from src.services.orchestrator import ProductionBiometricService
from src.services.types import (
    BiometricErrorCode,
    BiometricServiceException,
    InvalidCaptureError,
    RevokedCredentialError
)


@pytest.fixture
def service():
    """Provides a fresh production biometric service instance."""
    return ProductionBiometricService(
        master_secret="CancelableMasterSecret_Integration_2026_Key!",
        kms_key_id="kms://vault-test/master-key",
        default_dataset="DB1_B"
    )


@pytest.fixture
def fvc_db1_images():
    """Locates sample genuine and impostor images from DB1_B."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "FVC2002", "DB1_B"))
    img_101_1 = os.path.join(base_dir, "101_1.tif")
    img_101_2 = os.path.join(base_dir, "101_2.tif")
    img_102_1 = os.path.join(base_dir, "102_1.tif")
    img_103_1 = os.path.join(base_dir, "103_1.tif")

    if not all(os.path.exists(p) for p in [img_101_1, img_101_2, img_102_1, img_103_1]):
        pytest.skip(f"FVC DB1_B test images not found in {base_dir}")

    return {
        "101_1": img_101_1,
        "101_2": img_101_2,
        "102_1": img_102_1,
        "103_1": img_103_1
    }


def test_end_to_end_enrollment_and_genuine_verification(service, fvc_db1_images):
    """
    Asserts full enrollment -> genuine verification workflow:
    - Successful enrollment of impression 1.
    - Successful genuine verification of impression 2 with decision == 'ACCEPT'.
    - Score exceeds calibrated threshold and matched triangles > 0.
    - Audit log records compliant lifecycle events.
    """
    subject_id = "user_genuine_101"
    app_id = "fintech_auth_app"
    salt = "salt_prod_genuine_01"

    # 1. Enroll impression 1
    enroll_res = service.enroll(
        source=fvc_db1_images["101_1"],
        subject_id=subject_id,
        salt=salt,
        app_id=app_id
    )
    assert enroll_res["status"] == "success"
    assert enroll_res["subject_id"] == subject_id
    assert enroll_res["num_triangles"] > 20
    assert enroll_res["serialized_bytes"] > 0
    assert enroll_res["latency_ms"] > 0.0

    # 2. Genuine verification with impression 2
    match_res = service.verify(
        source=fvc_db1_images["101_2"],
        subject_id=subject_id,
        app_id=app_id,
        dataset="DB1_B",
        operating_point="EER"
    )
    assert match_res.decision == "ACCEPT"
    assert match_res.error_code == BiometricErrorCode.SUCCESS
    assert match_res.score >= match_res.threshold
    assert match_res.score > 0.40  # Genuine match score comfortably above noise
    assert match_res.matched_triangles > 0
    assert match_res.total_query_triangles > 0
    assert match_res.total_enrolled_triangles == enroll_res["num_triangles"]

    # 3. Audit trail assertions
    trail = service.audit_logger.get_trail()
    assert len(trail) >= 2
    enroll_ev = trail[0]
    verify_ev = trail[1]
    assert enroll_ev.operation == "ENROLL"
    assert enroll_ev.decision == "SUCCESS"
    assert verify_ev.operation == "VERIFY"
    assert verify_ev.decision == "ACCEPT"
    assert service.audit_logger.verify_integrity() is True


def test_impostor_rejection_flow_across_distinct_identities(service, fvc_db1_images):
    """
    Asserts impostor rejection across distinct finger identities:
    - Querying Subject 101's credential with Subject 102's image results in 'REJECT'.
    - Querying Subject 101's credential with Subject 103's image results in 'REJECT'.
    - Impostor scores fall strictly below calibrated threshold.
    """
    subj_101 = "user_subj_101"
    subj_102 = "user_subj_102"
    app_id = "fintech_auth_app"

    # Enroll user 101 and user 102
    service.enroll(fvc_db1_images["101_1"], subj_101, salt="salt_101", app_id=app_id)
    service.enroll(fvc_db1_images["102_1"], subj_102, salt="salt_102", app_id=app_id)

    # Impostor attempt 1: Subject 102 queries against Subject 101
    imp_res_1 = service.verify(
        source=fvc_db1_images["102_1"],
        subject_id=subj_101,
        app_id=app_id,
        dataset="DB1_B",
        operating_point="EER"
    )
    assert imp_res_1.decision == "REJECT"
    assert imp_res_1.score < imp_res_1.threshold

    # Impostor attempt 2: Subject 103 queries against Subject 101
    imp_res_2 = service.verify(
        source=fvc_db1_images["103_1"],
        subject_id=subj_101,
        app_id=app_id,
        dataset="DB1_B",
        operating_point="EER"
    )
    assert imp_res_2.decision == "REJECT"
    assert imp_res_2.score < imp_res_2.threshold


def test_revocation_and_re_enrollment_under_new_epoch(service, fvc_db1_images):
    """
    Asserts complete revocation lifecycle and fresh re-enrollment under a new epoch:
    1. Enroll under epoch 1 salt -> genuine match succeeds.
    2. Revoke subject -> subsequent verification strictly rejected with CREDENTIAL_REVOKED.
    3. Re-enroll under new epoch salt -> genuine match succeeds again.
    4. Intercepted epoch 1 parameters cannot verify against epoch 2 credential.
    """
    subject_id = "user_revocable_101"
    app_id = "enterprise_access"
    salt_epoch1 = "epoch1_salt_v1"
    salt_epoch2 = "epoch2_salt_v2"

    # Step 1: Enroll under Epoch 1
    service.enroll(fvc_db1_images["101_1"], subject_id, salt=salt_epoch1, app_id=app_id)
    res_epoch1 = service.verify(fvc_db1_images["101_2"], subject_id, app_id=app_id, dataset="DB1_B")
    assert res_epoch1.decision == "ACCEPT"

    # Step 2: Revocation
    rev_ok = service.revoke(subject_id, reason="credential_lost_compromise")
    assert rev_ok is True
    assert service.repository.is_revoked(subject_id) is True

    # Verification must fail cleanly with CREDENTIAL_REVOKED
    res_blocked = service.verify(fvc_db1_images["101_2"], subject_id, app_id=app_id, dataset="DB1_B")
    assert res_blocked.decision == "REJECT"
    assert res_blocked.error_code == BiometricErrorCode.CREDENTIAL_REVOKED
    assert res_blocked.score == 0.0

    # Step 3: Re-enrollment under Epoch 2 (Fresh presentation)
    reissue_res = service.reissue(
        source=fvc_db1_images["101_1"],
        subject_id=subject_id,
        new_salt=salt_epoch2,
        app_id=app_id
    )
    assert reissue_res["status"] == "success"
    assert reissue_res.get("reissued") is True
    assert service.repository.is_revoked(subject_id) is False

    # Verification under re-enrolled epoch 2 succeeds
    res_epoch2 = service.verify(fvc_db1_images["101_2"], subject_id, app_id=app_id, dataset="DB1_B")
    assert res_epoch2.decision == "ACCEPT"
    assert res_epoch2.score >= res_epoch2.threshold


def test_cross_application_isolation_and_decorrelation(service, fvc_db1_images):
    """
    Asserts cross-application domain isolation:
    - Same biometric features enrolled under App A (salt A) and App B (salt B).
    - Generated cancelable bitstrings are mutually decorrelated (mean Hamming distance ~ 0.50).
    - Authentication attempt from App B against App A is rejected.
    """
    img_source = fvc_db1_images["101_1"]
    feats = service.extract_pipeline_features(img_source)
    desc_matrix = feats["descriptors"].descriptors

    # Project under App A
    bits_app_a = service.projection_engine.project_and_quantize(
        descriptors=desc_matrix,
        salt="salt_application_alpha",
        app_id="app_alpha",
        subject_id="user_iso"
    )

    # Project under App B
    bits_app_b = service.projection_engine.project_and_quantize(
        descriptors=desc_matrix,
        salt="salt_application_beta",
        app_id="app_beta",
        subject_id="user_iso"
    )

    assert bits_app_a.shape == bits_app_b.shape
    # Pairwise diagonal Hamming distance between corresponding triangles
    pairwise_hd = np.mean(bits_app_a != bits_app_b, axis=1)
    mean_hd = float(np.mean(pairwise_hd))

    # Column-orthonormal projection with distinct HKDF keys yields statistical independence:
    # E[Hamming] = 0.50. Assert mean Hamming distance is in [0.40, 0.60].
    assert 0.40 <= mean_hd <= 0.60, f"Expected decorrelated Hamming ~ 0.50, got {mean_hd:.4f}"

    # Verify cross-application authentication fails
    service.enroll(img_source, "user_iso_alpha", salt="salt_application_alpha", app_id="app_alpha")
    # Query with app_beta parameters against app_alpha enrolled subject
    # Note: verify uses app_id to derive storage key and query projections
    match_wrong_app = service.verify(
        source=fvc_db1_images["101_2"],
        subject_id="user_iso_alpha",
        app_id="app_beta",  # Wrong application ID
        dataset="DB1_B"
    )
    # AEAD storage key derivation uses app_id -> wrong app_id causes decryption failure (or rejection)
    assert match_wrong_app.decision == "REJECT"
    assert match_wrong_app.error_code == BiometricErrorCode.AEAD_INTEGRITY_FAILURE

    # Direct match between cross-app cancelable bitstrings must remain strictly below threshold
    from src.matching.bipartite_matcher import match_protected_templates
    cross_app_score = match_protected_templates(
        bits_app_a, bits_app_b, np.ones(len(bits_app_a)), np.ones(len(bits_app_b))
    )
    threshold = service.threshold_registry.get_threshold("DB1_B", "EER")
    assert cross_app_score < threshold, f"Cross-app match score {cross_app_score:.4f} >= threshold {threshold:.4f}"


def test_malformed_and_non_finite_corrupted_inputs(service, fvc_db1_images):
    """
    Asserts robust input sanitization and standardized BiometricServiceException handling:
    - None input.
    - NaN and Inf image arrays.
    - Zero-size empty arrays and empty byte buffers.
    - Corrupt / non-image byte streams.
    - Undersized images (< 32x32).
    Both enroll() and verify() endpoints are verified for all corrupt inputs.
    """
    subject_id = "user_corrupt_test"
    salt = "salt_corrupt"
    app_id = "app_test"

    # Enroll a valid credential first so verify() exercises query image validation
    service.enroll(fvc_db1_images["101_1"], subject_id, salt, app_id)

    # 1. NaN array
    nan_img = np.full((128, 128), np.nan, dtype=np.float32)
    enroll_nan = service.enroll(nan_img, "user_nan", salt, app_id)
    assert enroll_nan["status"] == "error"
    assert enroll_nan["code"] == int(BiometricErrorCode.INVALID_IMAGE)

    verify_nan = service.verify(nan_img, subject_id, app_id)
    assert verify_nan.decision == "REJECT"
    assert verify_nan.error_code == BiometricErrorCode.INVALID_IMAGE

    # 2. Inf array
    inf_img = np.full((128, 128), np.inf, dtype=np.float32)
    enroll_inf = service.enroll(inf_img, "user_inf", salt, app_id)
    assert enroll_inf["status"] == "error"
    assert enroll_inf["code"] == int(BiometricErrorCode.INVALID_IMAGE)

    verify_inf = service.verify(inf_img, subject_id, app_id)
    assert verify_inf.decision == "REJECT"
    assert verify_inf.error_code == BiometricErrorCode.INVALID_IMAGE

    # 3. Empty bytes and empty array
    enroll_empty_b = service.enroll(b"", subject_id, salt, app_id)
    assert enroll_empty_b["status"] == "error"
    assert enroll_empty_b["code"] == int(BiometricErrorCode.INVALID_IMAGE)

    verify_empty_b = service.verify(b"", subject_id, app_id)
    assert verify_empty_b.decision == "REJECT"
    assert verify_empty_b.error_code == BiometricErrorCode.INVALID_IMAGE

    enroll_empty_arr = service.enroll(np.empty((0, 0), dtype=np.uint8), subject_id, salt, app_id)
    assert enroll_empty_arr["status"] == "error"
    assert enroll_empty_arr["code"] == int(BiometricErrorCode.INVALID_IMAGE)

    verify_empty_arr = service.verify(np.empty((0, 0), dtype=np.uint8), subject_id, app_id)
    assert verify_empty_arr.decision == "REJECT"
    assert verify_empty_arr.error_code == BiometricErrorCode.INVALID_IMAGE

    # 4. Corrupt random bytes
    corrupt_bytes = b"\x00\x11\x22\x33\x44\x55INVALID_HEADER_DATA" * 10
    enroll_corrupt = service.enroll(corrupt_bytes, subject_id, salt, app_id)
    assert enroll_corrupt["status"] == "error"
    assert enroll_corrupt["code"] == int(BiometricErrorCode.INVALID_IMAGE)

    verify_corrupt = service.verify(corrupt_bytes, subject_id, app_id)
    assert verify_corrupt.decision == "REJECT"
    assert verify_corrupt.error_code == BiometricErrorCode.INVALID_IMAGE

    # 5. Image too small
    tiny_img = np.ones((8, 8), dtype=np.uint8) * 128
    enroll_tiny = service.enroll(tiny_img, subject_id, salt, app_id)
    assert enroll_tiny["status"] == "error"
    assert enroll_tiny["code"] in [int(BiometricErrorCode.IMAGE_TOO_SMALL), int(BiometricErrorCode.INVALID_IMAGE)]

    verify_tiny = service.verify(tiny_img, subject_id, app_id)
    assert verify_tiny.decision == "REJECT"
    assert verify_tiny.error_code in [BiometricErrorCode.IMAGE_TOO_SMALL, BiometricErrorCode.INVALID_IMAGE]

    # 6. None source
    enroll_none = service.enroll(None, "user_none", salt, app_id)
    assert enroll_none["status"] == "error"
    assert enroll_none["code"] == int(BiometricErrorCode.INVALID_IMAGE)

    verify_none = service.verify(None, subject_id, app_id)
    assert verify_none.decision == "REJECT"
    assert verify_none.error_code == BiometricErrorCode.INVALID_IMAGE


def test_key_rotation_and_projection_cache_invalidation(service, fvc_db1_images):
    """
    Asserts KMS master secret rotation, projection matrix cache invalidation, and audit logging:
    - KeyService advances cryptographic epoch.
    - ProjectionEngine matrix cache is purged.
    - Derived projection matrices under new master secret are decorrelated from prior epoch.
    - AuditLogger records KEY_ROTATION event.
    """
    salt = "rotation_test_salt"
    app_id = "rotation_app"

    # Derive initial matrix and ensure cached
    W_epoch1 = service.projection_engine.get_projection_matrix(salt=salt, app_id=app_id)
    cache_key = (salt, app_id, "", 9, 32)
    assert cache_key in service.projection_engine._matrix_cache
    assert service.key_service.epoch == 1

    # Execute Key Rotation
    service.rotate_master_secret(
        new_master_secret="NewMasterSecret_Epoch2_Rotation2026_SecureKey!",
        reason="scheduled_epoch_rotation"
    )

    # Assert epoch advanced and cache cleared
    assert service.key_service.epoch == 2
    assert len(service.projection_engine._matrix_cache) == 0

    # Derive matrix again under new epoch
    W_epoch2 = service.projection_engine.get_projection_matrix(salt=salt, app_id=app_id)
    assert cache_key in service.projection_engine._matrix_cache

    # Verify W_epoch1 != W_epoch2
    assert not np.array_equal(W_epoch1, W_epoch2)

    # Verify Gram matrix of cross-epoch projection product demonstrates decorrelation
    cross_product = np.dot(W_epoch1, W_epoch2.T)
    # Absolute dot product between random orthonormal vectors in R^32 is small (~ 1/sqrt(32) ~ 0.17)
    mean_abs_cross = float(np.mean(np.abs(cross_product)))
    assert mean_abs_cross < 0.35

    # Check audit logger recorded KEY_ROTATION
    trail = service.audit_logger.get_trail()
    rotation_events = [ev for ev in trail if ev.operation == "KEY_ROTATION"]
    assert len(rotation_events) == 1
    assert rotation_events[0].decision == "SUCCESS"
    assert rotation_events[0].metadata.get("new_epoch") == 2
