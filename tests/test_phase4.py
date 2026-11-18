"""
Phase 4 Automated Test Suite: Storage, Speed, and Deployability Engineering.

Tests:
1. RFC 8439 ChaCha20-Poly1305 AEAD cipher specification compliance (test vectors).
2. Strict AEAD tamper detection (ciphertext, header AAD, nonce, tag bit flips).
3. Serialization & deserialization round-trip fidelity (descriptor-only and topology).
4. Strictly like-for-like continuous baseline accounting assertions.
5. KeyService KMS/HSM trust boundaries and secret isolation.
6. Modular production service pipeline integration.
7. CredentialRepository lifecycle, revocation blocking, and fresh reissuance.
8. Privacy-minimised audit logging and cryptographic hash-chain integrity.
9. ThresholdRegistry operating point lookups and error handling.
"""

import os
import pytest
import numpy as np

from src.services.types import (
    BiometricErrorCode,
    BiometricServiceException,
    AeadIntegrityError,
    MasterSecretAccessViolation,
    RevokedCredentialError,
    ThresholdNotFoundError,
    InvalidCaptureError,
    ProtectedCredentialDTO
)
from src.services.crypto_aead import (
    chacha20_poly1305_encrypt,
    chacha20_poly1305_decrypt
)
from src.services.capture_adapter import CaptureAdapter
from src.services.preprocessor import Preprocessor
from src.services.minutiae_extractor import MinutiaeExtractor
from src.services.triangle_builder import TriangleBuilder
from src.services.descriptor_builder import DescriptorBuilder
from src.services.scaler_registry import ScalerRegistry
from src.services.key_service import KeyService
from src.services.projection_engine import ProjectionEngine
from src.services.credential_serializer import CredentialSerializer
from src.services.credential_repository import CredentialRepository
from src.services.matcher import Matcher
from src.services.threshold_registry import ThresholdRegistry
from src.services.audit_logger import AuditLogger
from src.services.orchestrator import ProductionBiometricService
from src.matching.graph_consensus import pack_adjacency_matrix, unpack_adjacency_matrix


def test_rfc8439_chacha20_poly1305_test_vector():
    """Verifies RFC 8439 Section 2.8.2 AEAD test vector."""
    key = bytes.fromhex("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f")
    nonce = bytes.fromhex("070000004041424344454647")
    aad = bytes.fromhex("50515253c0c1c2c3c4c5c6c7")
    pt = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it."

    ct, tag = chacha20_poly1305_encrypt(key, nonce, pt, aad)
    expected_ct = bytes.fromhex(
        "d31a8d34648e60db7b86afbc53ef7ec2"
        "a4aded51296e08fea9e2b5a736ee62d6"
        "3dbea45e8ca9671282fafb69da92728b"
        "1a71de0a9e060b2905d6a5b67ecd3b36"
        "92ddbd7f2d778b8c9803aee328091b58"
        "fab324e4fad675945585808b4831d7bc"
        "3ff4def08e4b7a9de576d26586cec64b"
        "6116"
    )
    expected_tag = bytes.fromhex("1ae10b594f09e26a7e902ecbd0600691")

    assert ct == expected_ct, "Ciphertext did not match RFC 8439 test vector!"
    assert tag == expected_tag, "Authentication tag did not match RFC 8439 test vector!"

    dec = chacha20_poly1305_decrypt(key, nonce, ct, tag, aad)
    assert dec == pt, "Decrypted plaintext did not match original input!"


def test_aead_tampering_detection():
    """Asserts that 1-bit modifications in ciphertext, header AAD, nonce, or tag fail AEAD verification."""
    key = os.urandom(32)
    nonce = os.urandom(12)
    aad = b"HEADER_AAD_METADATA_V1"
    plaintext = b"CancelableBiometricPayloadConfidentialData"

    ct, tag = chacha20_poly1305_encrypt(key, nonce, plaintext, aad)

    # 1. Tamper ciphertext
    bad_ct = bytearray(ct)
    bad_ct[0] ^= 0x01
    with pytest.raises(AeadIntegrityError):
        chacha20_poly1305_decrypt(key, nonce, bytes(bad_ct), tag, aad)

    # 2. Tamper AAD
    bad_aad = bytearray(aad)
    bad_aad[-1] ^= 0x01
    with pytest.raises(AeadIntegrityError):
        chacha20_poly1305_decrypt(key, nonce, ct, tag, bytes(bad_aad))

    # 3. Tamper tag
    bad_tag = bytearray(tag)
    bad_tag[5] ^= 0x80
    with pytest.raises(AeadIntegrityError):
        chacha20_poly1305_decrypt(key, nonce, ct, bytes(bad_tag), aad)

    # 4. Tamper nonce
    bad_nonce = bytearray(nonce)
    bad_nonce[0] ^= 0x02
    with pytest.raises(AeadIntegrityError):
        chacha20_poly1305_decrypt(key, bytes(bad_nonce), ct, tag, aad)


def test_serialization_roundtrip_descriptor_only():
    """Tests byte-exact serialization round-trip for production default descriptor-only credential."""
    serializer = CredentialSerializer()
    storage_key = os.urandom(32)
    n = 120

    rng = np.random.default_rng(123)
    bitstrings = rng.integers(0, 2, size=(n, 32), dtype=np.uint8)
    weights = rng.uniform(0.1, 1.0, size=(n,)).astype(np.float32)

    dto = ProtectedCredentialDTO(
        credential_id="12345678-1234-5678-1234-567812345678",
        subject_id="subj_test_01",
        salt="test_salt_prod_01",
        app_id="fvc_app",
        bitstrings=bitstrings,
        weights=weights,
        num_triangles=n,
        packed_topology=None
    )

    env = serializer.serialize_protected(dto, storage_key, encrypt=True)
    expected_bytes = 64 + (n * 4) + (n * 2) + 16  # 80 + 6N = 800 bytes
    assert env.total_bytes == expected_bytes
    assert len(env.raw_bytes) == expected_bytes

    recovered = serializer.deserialize_protected(
        raw_bytes=env.raw_bytes,
        storage_key=storage_key,
        subject_id="subj_test_01",
        app_id="fvc_app"
    )

    assert recovered.num_triangles == n
    assert recovered.credential_id == dto.credential_id
    assert np.array_equal(recovered.bitstrings, bitstrings)
    # Weights stored as float16: assert precision within float16 tolerance
    np.testing.assert_allclose(recovered.weights, weights, atol=1e-3)
    assert recovered.packed_topology is None


def test_serialization_roundtrip_with_topology():
    """Tests byte-exact serialization round-trip with 2-bit packed topology matrix."""
    serializer = CredentialSerializer()
    storage_key = os.urandom(32)
    n = 80

    rng = np.random.default_rng(456)
    bitstrings = rng.integers(0, 2, size=(n, 32), dtype=np.uint8)
    weights = rng.uniform(0.1, 1.0, size=(n,)).astype(np.float32)

    # Synthetic symmetric dual incidence matrix A in {0, 1, 2, 3}
    adj = rng.integers(0, 4, size=(n, n), dtype=np.uint8)
    adj = (adj + adj.T) // 2
    np.fill_diagonal(adj, 3)

    packed_topo = pack_adjacency_matrix(adj)
    expected_packed_len = ((n * (n - 1) // 2) + 3) // 4
    assert len(packed_topo) == expected_packed_len

    dto = ProtectedCredentialDTO(
        credential_id="87654321-4321-8765-4321-876543210987",
        subject_id="subj_topo_02",
        salt="test_salt_topo",
        app_id="fvc_app",
        bitstrings=bitstrings,
        weights=weights,
        num_triangles=n,
        packed_topology=packed_topo
    )

    env = serializer.serialize_protected(dto, storage_key, encrypt=True)
    expected_bytes = 80 + (6 * n) + expected_packed_len
    assert env.total_bytes == expected_bytes

    recovered = serializer.deserialize_protected(
        raw_bytes=env.raw_bytes,
        storage_key=storage_key,
        subject_id="subj_topo_02",
        app_id="fvc_app"
    )

    assert recovered.num_triangles == n
    assert recovered.packed_topology is not None
    unpacked_adj = unpack_adjacency_matrix(recovered.packed_topology, n)
    assert np.array_equal(unpacked_adj, adj), "Unpacked matrix A does not match original!"


def test_continuous_unprotected_baseline_accounting():
    """Asserts that continuous baseline produces strictly 40 B/tri + header."""
    serializer = CredentialSerializer()
    n = 144
    feats = np.zeros((n, 9), dtype=np.float32)
    weights = np.ones((n,), dtype=np.float32)

    raw_base = serializer.serialize_unprotected_baseline(feats, weights)
    expected_len = 64 + (144 * 40)  # 64 + 5760 = 5,824 bytes
    assert len(raw_base) == expected_len

    # Assert like-for-like storage reduction:
    # Protected descriptor-only is 80 + 6*144 = 944 bytes
    prot_len = 80 + (6 * 144)
    reduction = (1.0 - (prot_len / expected_len)) * 100.0
    assert reduction > 80.0, f"Expected >80% reduction, got {reduction:.2f}%"


def test_key_service_trust_boundary():
    """Verifies that KeyService isolates master secret and denies client bypass."""
    ks = KeyService(master_secret="KeyServiceSecurityTestSecretMasterKey2026!", kms_key_id="kms://secure-01")

    # 1. Direct access forbidden (both public and private attribute names)
    with pytest.raises(MasterSecretAccessViolation):
        _ = ks.master_secret
    with pytest.raises(MasterSecretAccessViolation):
        _ = ks._master_secret

    # 2. String representation hides secret
    rep = repr(ks)
    assert "PROTECTED" in rep
    assert "KeyServiceSecurityTest" not in rep
    assert "[REDACTED]" in str(ks)

    # 3. Derivation works properly
    W = ks.derive_projection_matrix("salt1", "app1", in_dim=9, out_dim=32)
    assert W.shape == (32, 9)
    # Check column orthonormality: W^T W = I
    gram = np.dot(W.T, W)
    np.testing.assert_allclose(gram, np.eye(9), atol=1e-5)

    # 4. Key rotation advances epoch
    old_epoch = ks.epoch
    ks.rotate_master_secret("NewKeyServiceSecurityTestSecretMasterKey2026_Epoch2!")
    assert ks.epoch == old_epoch + 1


def test_modular_service_pipeline_end_to_end():
    """Tests end-to-end enrollment and verification using ProductionBiometricService."""
    service = ProductionBiometricService()
    img_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "FVC2002", "DB1_B", "101_1.tif")
    if not os.path.exists(img_path):
        pytest.skip("DB1_B test image not found.")

    subj_id = "test_user_db1"
    app_id = "fvc_e2e_app"
    salt = "salt_e2e_test_2026"

    # Enroll
    enroll_res = service.enroll(img_path, subj_id, salt, app_id, with_topology=False)
    assert enroll_res["status"] == "success"
    assert enroll_res["num_triangles"] > 20
    assert enroll_res["serialized_bytes"] > 0

    # Verify same image (Self-match should yield high score >= threshold)
    match_res = service.verify(img_path, subj_id, app_id, dataset="DB1_B")
    assert match_res.decision == "ACCEPT"
    assert match_res.score > 0.80
    assert match_res.error_code == BiometricErrorCode.SUCCESS


def test_revocation_lifecycle_and_fresh_reissuance():
    """Tests that revoked credentials cannot be verified and fresh presentation succeeds."""
    service = ProductionBiometricService()
    img1_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "FVC2002", "DB1_B", "101_1.tif")
    img2_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "FVC2002", "DB1_B", "101_2.tif")
    if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
        pytest.skip("DB1_B test images not found.")

    subj = "revocable_user_01"
    app_id = "app_revocation"
    salt_old = "salt_epoch_1"
    salt_new = "salt_epoch_2"

    # 1. Enroll under salt_old
    service.enroll(img1_path, subj, salt_old, app_id)
    res_before = service.verify(img2_path, subj, app_id, dataset="DB1_B")
    assert res_before.decision == "ACCEPT"

    # 2. Revoke credential
    revoked = service.revoke(subj, reason="user_lost_credential")
    assert revoked is True
    assert service.repository.is_revoked(subj) is True

    # 3. Verification must fail
    res_after = service.verify(img2_path, subj, app_id, dataset="DB1_B")
    assert res_after.decision == "REJECT"
    assert res_after.error_code == BiometricErrorCode.CREDENTIAL_REVOKED

    # 4. Fresh Presentation Reissuance under salt_new
    reissue_res = service.reissue(img2_path, subj, salt_new, app_id)
    assert reissue_res["status"] == "success"
    assert service.repository.is_revoked(subj) is False

    # 5. Authenticate under reissued credential
    res_reissued = service.verify(img2_path, subj, app_id, dataset="DB1_B")
    assert res_reissued.decision == "ACCEPT"
    assert res_reissued.score > 0.80


def test_privacy_audit_logger_boundaries():
    """Asserts that audit logs satisfy privacy-minimisation and hash-chain integrity."""
    logger = AuditLogger()
    logger.log_event("ENROLL", "subj_sensitive_999", "SUCCESS", score=None, latency_ms=12.5,
                     metadata={"image": "SECRET_IMAGE_ARRAY", "num_triangles": 140})
    logger.log_event("VERIFY", "subj_sensitive_999", "ACCEPT", score=0.420, latency_ms=4.2)

    trail = logger.get_trail()
    assert len(trail) == 2

    for ev in trail:
        # Subject pseudonymized
        assert "subj_sensitive_999" not in ev.subject_pseudonym
        assert ev.subject_pseudonym.startswith("subj_hash_")
        # Forbidden keys redacted
        if "image" in ev.metadata:
            assert ev.metadata["image"] == "[REDACTED_PRIVACY_BOUNDARY]"

    # Hash chain integrity passes
    assert logger.verify_integrity() is True


def test_threshold_registry_operating_points():
    """Verifies threshold registry looks up calibrated operating points and raises for invalid ones."""
    reg = ThresholdRegistry()
    t_db1_a = reg.get_threshold("DB1_A", "EER")
    assert t_db1_a == 0.3751

    t_socofing = reg.get_threshold("SOCOFing", "BENCHMARK_ACCEPT")
    assert t_socofing == 0.3891

    with pytest.raises(ThresholdNotFoundError):
        reg.get_threshold("DB1_A", "NON_EXISTENT_POINT")


def test_unknown_subject_lifecycle_not_found():
    """Verifies that querying an unknown subject cleanly raises CREDENTIAL_NOT_FOUND rather than CREDENTIAL_REVOKED."""
    repo = CredentialRepository()
    with pytest.raises(BiometricServiceException) as exc_info:
        repo.get_raw_bytes("never_enrolled_user_123")
    assert exc_info.value.code == BiometricErrorCode.CREDENTIAL_NOT_FOUND
    assert repo.is_revoked("never_enrolled_user_123") is False

    service = ProductionBiometricService()
    img_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "FVC2002", "DB1_B", "101_1.tif")
    if os.path.exists(img_path):
        res = service.verify(img_path, "never_enrolled_user_456", "test_app")
        assert res.decision == "REJECT"
        assert res.error_code == BiometricErrorCode.CREDENTIAL_NOT_FOUND


def test_credential_serializer_truncated_and_corrupt_payload():
    """Verifies that truncated or malformed payloads cleanly raise BiometricServiceException."""
    import struct
    import zlib
    serializer = CredentialSerializer()
    key = os.urandom(32)

    # 1. Payload shorter than header + tag
    with pytest.raises(BiometricServiceException) as exc_info:
        serializer.deserialize_protected(b"CFPB" + b"\x00" * 20, key)
    assert exc_info.value.code == BiometricErrorCode.SERIALIZATION_ERROR

    # 2. Valid header claiming N=50 triangles, but truncated ciphertext
    header_pre = struct.pack(
        "<4sHHBB2s16s16sI12s",
        b"CFPB", 1, 0, 9, 0, b"\x00\x00",
        b"0" * 16, b"1" * 16, 50, b"0" * 12
    )
    crc = zlib.crc32(header_pre) & 0xffffffff
    header = header_pre + struct.pack("<I", crc)
    truncated_raw = header + b"x" * 10 + b"\x00" * 16  # Expected 50*6 = 300 bytes, only got 10

    with pytest.raises(BiometricServiceException) as exc_info:
        serializer.deserialize_protected(truncated_raw, key)
    assert exc_info.value.code == BiometricErrorCode.SERIALIZATION_ERROR


def test_capture_adapter_empty_and_nan_inputs():
    """Verifies that empty byte buffers and non-finite image arrays raise InvalidCaptureError."""
    adapter = CaptureAdapter()

    # Empty bytes
    with pytest.raises(InvalidCaptureError) as exc_info:
        adapter.ingest(b"")
    assert exc_info.value.code == BiometricErrorCode.INVALID_IMAGE

    # Empty array
    with pytest.raises(InvalidCaptureError) as exc_info:
        adapter.ingest(np.array([], dtype=np.uint8))
    assert exc_info.value.code == BiometricErrorCode.INVALID_IMAGE

    # Array containing NaN
    nan_img = np.full((120, 120), np.nan, dtype=np.float32)
    with pytest.raises(InvalidCaptureError) as exc_info:
        adapter.ingest(nan_img)
    assert exc_info.value.code == BiometricErrorCode.INVALID_IMAGE


def test_matcher_dimension_mismatch_validation():
    """Verifies that mismatched projection dimensions and weight lengths raise DIMENSION_MISMATCH."""
    matcher = Matcher()
    q_bits = np.ones((10, 32), dtype=np.uint8)
    e_bits_wrong_dim = np.ones((10, 64), dtype=np.uint8)
    q_w = np.ones((10,), dtype=np.float32)
    e_w = np.ones((10,), dtype=np.float32)
    q_w_wrong_len = np.ones((5,), dtype=np.float32)

    # Dimension mismatch
    with pytest.raises(BiometricServiceException) as exc_info:
        matcher.match_protected(q_bits, e_bits_wrong_dim, q_w, e_w)
    assert exc_info.value.code == BiometricErrorCode.DIMENSION_MISMATCH

    # Weight length mismatch
    with pytest.raises(BiometricServiceException) as exc_info:
        matcher.match_protected(q_bits, q_bits, q_w_wrong_len, e_w)
    assert exc_info.value.code == BiometricErrorCode.DIMENSION_MISMATCH


def test_audit_logger_nested_secret_redaction():
    """Verifies that nested dictionaries containing sensitive keys are recursively redacted."""
    logger = AuditLogger()
    logger.log_event(
        operation="ENROLL",
        subject_id="test_nested_subj",
        decision="SUCCESS",
        metadata={
            "app": "vault",
            "nested": {
                "safe_info": 42,
                "master_secret": "TOP_SECRET_MASTER_KEY_STRING",
                "inner": {
                    "private_key": "INNER_PRIVATE_KEY_BYTES"
                }
            }
        }
    )
    trail = logger.get_trail()
    assert len(trail) == 1
    nested_meta = trail[0].metadata["nested"]
    assert nested_meta["safe_info"] == 42
    assert nested_meta["master_secret"] == "[REDACTED_PRIVACY_BOUNDARY]"
    assert nested_meta["inner"]["private_key"] == "[REDACTED_PRIVACY_BOUNDARY]"

