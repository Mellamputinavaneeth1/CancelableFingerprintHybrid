"""
Unit, Integration, Security, and Unlinkability Tests for Modular Dual-Tier MCC + Keyed RFF SE(2) Spectral Consensus.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.descriptors.mcc import flip_mcc_dual
from src.pipeline import CancelableFingerprintPipeline
from src.minutiae.onnx_detector import ONNXMinutiaeDetector, DEFAULT_ONNX_MODEL_PATH
from src.matching.bipartite_matcher import compute_pairwise_hamming_distance
from src.services.credential_repository import CredentialRepository
from src.services.credential_serializer import CredentialSerializer
from src.services.key_service import MasterSecretAccessViolation
from src.services.orchestrator import ProductionBiometricService
from src.services.scaler_registry import ScalerRegistry
from src.services.threshold_registry import ThresholdRegistry
from src.services.types import (
    BiometricErrorCode,
    BiometricServiceException,
    ProtectedCredentialDTO,
)


def _synthetic_ridge_image(h: int = 128, w: int = 128) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    phi = 2.0 * np.pi * 0.11 * yy + 0.8 * np.sin(2.0 * np.pi * xx / 48.0)
    img = np.clip(127.0 + 75.0 * np.cos(phi), 0, 255).astype(np.uint8)
    return img


def test_scaler_and_threshold_registries_mcc():
    sr = ScalerRegistry()
    for key in ("POOLED_SET_B_MCC_256", "DEFAULT_MCC_256", "DB1_A_MCC_256", "DB1_B_MCC_256", "DB2_B_MCC_256"):
        sc = sr.get_scaler(key)
        assert sc["mean"].shape == (256,)
        assert sc["std"].shape == (256,)
        assert np.all(sc["std"] > 0.0)
        # Symmetrized stats must be invariant under 180-degree dual channel flip
        assert np.allclose(sc["mean"], flip_mcc_dual(sc["mean"], 8, 4), atol=1e-6)
        assert np.allclose(sc["std"], flip_mcc_dual(sc["std"], 8, 4), atol=1e-6)

    tr = ThresholdRegistry()
    for ds_key in ("DEFAULT_MCC", "DEFAULT_MCC_CLASSICAL", "DB1_A_MCC_CLASSICAL", "DB1_B_MCC_CLASSICAL"):
        t_eer = tr.get_threshold(ds_key, "EER")
        t_1 = tr.get_threshold(ds_key, "FAR_1_PERCENT")
        t_01 = tr.get_threshold(ds_key, "FAR_0_1_PERCENT")
        assert 0.0 < t_eer <= t_1 <= t_01 < 1.0


def test_credential_serializer_256_192_rff128_roundtrip_and_tamper():
    ser = CredentialSerializer()
    rng = np.random.default_rng(7)
    n = 28
    bits = rng.integers(0, 2, size=(n, 192), dtype=np.uint8)
    weights = rng.uniform(0.3, 1.0, size=(n,)).astype(np.float32)
    edges = rng.integers(0, 2**32, size=(n, n, 2), dtype=np.uint64)

    cred = ProtectedCredentialDTO(
        credential_id="cred_test_1",
        subject_id="subj_mcc_1",
        salt="salt_mcc_01",
        app_id="app_mcc",
        bitstrings=bits,
        weights=weights,
        num_triangles=n,
        packed_topology=None,
        edge_tokens=edges,
        in_dim=256,
    )
    key = bytes(range(32))
    env = ser.serialize_protected(cred, storage_key=key, encrypt=True)
    assert env.has_edge_tokens is True
    assert env.total_bytes > 0

    dec = ser.deserialize_protected(
        env.raw_bytes, storage_key=key, subject_id="subj_mcc_1", app_id="app_mcc"
    )
    assert dec.in_dim == 256
    assert dec.bitstrings.shape == (n, 192)
    assert dec.edge_tokens is not None and dec.edge_tokens.shape == (n, n, 2)
    assert np.array_equal(dec.bitstrings, bits)
    assert np.array_equal(dec.edge_tokens, edges)
    assert np.allclose(dec.weights, weights, atol=1e-2)

    # Wrong storage key (e.g. derived for wrong app/subject) must fail AEAD strictly
    wrong_key = bytes(reversed(range(32)))
    with pytest.raises(BiometricServiceException):
        ser.deserialize_protected(
            env.raw_bytes, storage_key=wrong_key, subject_id="subj_mcc_1", app_id="wrong_app"
        )

    bad_payload = bytearray(env.raw_bytes)
    bad_payload[-5] ^= 0xFF
    with pytest.raises(BiometricServiceException) as exc_info:
        ser.deserialize_protected(
            bytes(bad_payload), storage_key=key, subject_id="subj_mcc_1", app_id="app_mcc"
        )
    assert exc_info.value.code == BiometricErrorCode.AEAD_INTEGRITY_FAILURE


def test_mcc_spectral_classical_service_lifecycle(tmp_path):
    db_file = str(tmp_path / "mcc_test.db")
    svc = ProductionBiometricService(
        feature_mode="mcc_spectral",
        engine="classical",
        default_dataset="DB1_B",
        out_dim=192,
        rff_bits=128,
    )
    svc.repository = CredentialRepository(db_path=db_file)

    img1 = "dataset/FVC2002/DB1_B/101_1.tif"
    img2 = "dataset/FVC2002/DB1_B/101_2.tif"
    img_imp = "dataset/FVC2002/DB1_B/102_1.tif"
    if not os.path.exists(img1):
        pytest.skip("DB1_B fixture images unavailable")

    res_enr = svc.enroll(img1, subject_id="u101", salt="salt_ep1", app_id="app_a", dpi=500.0)
    assert res_enr["status"] == "success"
    assert res_enr["num_triangles"] >= 10

    res_gen = svc.verify(img2, subject_id="u101", app_id="app_a", dataset="DB1_B", dpi=500.0)
    res_imp = svc.verify(img_imp, subject_id="u101", app_id="app_a", dataset="DB1_B", dpi=500.0)
    assert res_gen.error_code == BiometricErrorCode.SUCCESS
    assert res_gen.decision == "ACCEPT"
    assert res_gen.score >= res_gen.threshold
    assert res_imp.decision == "REJECT"
    assert res_imp.score < res_gen.score

    # Revoke & Reissue
    assert svc.revoke("u101", reason="test_compromise") is True
    res_rev = svc.verify(img2, subject_id="u101", app_id="app_a", dataset="DB1_B", dpi=500.0)
    assert res_rev.decision == "REJECT"
    assert res_rev.error_code == BiometricErrorCode.CREDENTIAL_REVOKED

    res_re = svc.reissue(img1, subject_id="u101", new_salt="salt_ep2", app_id="app_a", dpi=500.0)
    assert res_re["status"] == "success"
    assert res_re["reissued"] is True
    res_gen2 = svc.verify(img2, subject_id="u101", app_id="app_a", dataset="DB1_B", dpi=500.0)
    assert res_gen2.decision == "ACCEPT"


def test_mcc_spectral_pipeline_wrapper():
    import cv2

    img1 = "dataset/FVC2002/DB1_B/101_1.tif"
    img2 = "dataset/FVC2002/DB1_B/101_2.tif"
    img_imp = "dataset/FVC2002/DB1_B/102_1.tif"
    if not os.path.exists(img1):
        pytest.skip("DB1_B fixture images unavailable")

    im1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
    im_imp = cv2.imread(img_imp, cv2.IMREAD_GRAYSCALE)

    pipe = CancelableFingerprintPipeline(
        feature_mode="mcc_spectral", engine="classical", in_dim=256, out_dim=192, rff_bits=128, dpi=500.0
    )
    t_enr = pipe.enroll(im1, salt="pipe_s1", app_id="pipe_app", subject_id="s101")
    tpl = t_enr["template"]
    assert tpl["bitstrings"].shape[1] == 192
    assert tpl["edge_tokens"].shape[-1] == 2
    assert "coords" not in tpl and "thetas" not in tpl
    assert "coords" not in t_enr and "thetas" not in t_enr

    s_gen = pipe.verify(im2, tpl)
    s_imp = pipe.verify(im_imp, tpl)
    assert s_gen["score"] > s_imp["score"]


def test_onnx_detector_inference_and_fallback():
    det_fallback = ONNXMinutiaeDetector(model_path=None, conf_thresh=0.35)
    assert det_fallback.is_model_loaded is False
    synth = _synthetic_ridge_image()
    mask = np.ones_like(synth, dtype=np.uint8)
    mins_fb, _ = det_fallback.detect(synth, mask)
    assert isinstance(mins_fb, list)

    if os.path.exists(DEFAULT_ONNX_MODEL_PATH):
        import cv2
        from src.preprocessing.segmenter import segment_fingerprint
        from src.preprocessing.enhancer import enhance_fingerprint

        det_onnx = ONNXMinutiaeDetector(model_path=DEFAULT_ONNX_MODEL_PATH)
        assert det_onnx.is_model_loaded is True
        img = cv2.imread("dataset/FVC2002/DB1_B/101_1.tif", cv2.IMREAD_GRAYSCALE)
        _, seg_mask, _ = segment_fingerprint(img, block_size=16, method="adaptive")
        _, ori, coh = enhance_fingerprint(img, use_gabor=True, mask=seg_mask)
        mins_nn, _ = det_onnx.detect(img, seg_mask, orientation_map=ori, coherence_map=coh)
        assert len(mins_nn) >= 10
        dist_fg = cv2.distanceTransform((seg_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        for m in mins_nn:
            assert dist_fg[int(round(m.y)), int(round(m.x))] >= 5.5


def test_mcc_cross_app_unlinkability_and_kms_isolation(tmp_path):
    svc = ProductionBiometricService(
        feature_mode="mcc_spectral",
        engine="classical",
        default_dataset="DB1_B",
        out_dim=192,
        rff_bits=128,
    )
    svc.repository = CredentialRepository(db_path=str(tmp_path / "unlink.db"))

    with pytest.raises(MasterSecretAccessViolation):
        _ = svc.key_service._master_secret

    img = "dataset/FVC2002/DB1_B/101_1.tif"
    if not os.path.exists(img):
        pytest.skip("DB1_B fixture images unavailable")

    svc.enroll(img, subject_id="u_app_a", salt="salt_a", app_id="bank_a", dpi=500.0)
    svc.enroll(img, subject_id="u_app_b", salt="salt_b", app_id="health_b", dpi=500.0)

    raw_a = svc.repository.get_raw_bytes("u_app_a")
    raw_b = svc.repository.get_raw_bytes("u_app_b")
    salt_a = raw_a[28:44].rstrip(b"\x00").decode("utf-8", errors="replace")
    salt_b = raw_b[28:44].rstrip(b"\x00").decode("utf-8", errors="replace")

    c_a = svc.serializer.deserialize_protected(
        raw_a,
        storage_key=svc.key_service.derive_storage_key(salt_a, "bank_a"),
        subject_id="u_app_a",
        app_id="bank_a",
    )
    c_b = svc.serializer.deserialize_protected(
        raw_b,
        storage_key=svc.key_service.derive_storage_key(salt_b, "health_b"),
        subject_id="u_app_b",
        app_id="health_b",
    )

    mean_hd = float(np.mean(compute_pairwise_hamming_distance(c_a.bitstrings, c_b.bitstrings)))
    assert 0.40 <= mean_hd <= 0.60, f"Cross-app MCC bitstrings must be decorrelated, got {mean_hd}"

    edge_xor = np.bitwise_xor(c_a.edge_tokens, c_b.edge_tokens)
    edge_hd = float(np.mean(np.bitwise_count(edge_xor)) / 64.0)
    assert 0.42 <= edge_hd <= 0.58, f"Cross-app RFF edge tokens must be decorrelated, got {edge_hd}"


def test_mcc_spectral_deep_engine_lifecycle(tmp_path):
    if not os.path.exists(DEFAULT_ONNX_MODEL_PATH):
        pytest.skip("Trained ONNX weights absent; fallback tested separately")
    img1 = "dataset/FVC2002/DB1_B/101_1.tif"
    img2 = "dataset/FVC2002/DB1_B/101_2.tif"
    img_imp = "dataset/FVC2002/DB1_B/102_1.tif"
    if not os.path.exists(img1):
        pytest.skip("DB1_B fixture images unavailable")

    svc = ProductionBiometricService(
        feature_mode="mcc_spectral",
        engine="deep",
        default_dataset="DB1_B",
        out_dim=192,
        rff_bits=128,
    )
    svc.repository = CredentialRepository(db_path=str(tmp_path / "deep.db"))
    res_enr = svc.enroll(img1, subject_id="d101", salt="salt_d1", app_id="app_deep", dpi=500.0)
    assert res_enr["status"] == "success"
    assert res_enr["num_triangles"] >= 10

    res_gen = svc.verify(img2, subject_id="d101", app_id="app_deep", dataset="DB1_B", dpi=500.0)
    res_imp = svc.verify(img_imp, subject_id="d101", app_id="app_deep", dataset="DB1_B", dpi=500.0)
    assert res_gen.error_code == BiometricErrorCode.SUCCESS
    assert res_gen.score > res_imp.score


def test_sparse_edge_tokens_wire_compression_and_service_roundtrip(tmp_path):
    """
    Verifies HAS_SPARSE_EDGES wire framing, bit-exact masked AEAD round-trip,
    >= 50% credential size reduction over dense (N, N, 2), and end-to-end matching.
    """
    from src.matching.spectral_consensus import (
        decode_sparse_edge_rff,
        derive_rff_edge_params,
        encode_edge_rff,
        encode_sparse_edge_rff,
    )

    rng = np.random.default_rng(2026)
    n = 50
    coords = rng.uniform(20.0, 320.0, size=(n, 2))
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=(n,))
    W_rff, b_rff = derive_rff_edge_params(
        "TestSparseWireCompressionMasterSecret_32Bytes_2026!",
        "salt_cmp",
        "app_cmp",
        "subj_cmp",
        edge_dim=7,
        rff_bits=128,
    )

    dense_edges = encode_edge_rff(coords, thetas, W_rff, b_rff)
    compact_toks, mask = encode_sparse_edge_rff(coords, thetas, W_rff, b_rff, top_k=16, mode="topk")
    reconstructed = decode_sparse_edge_rff(compact_toks, mask)
    assert mask.shape == (n, n)
    assert int(mask.sum()) == n * 16
    assert np.array_equal(reconstructed[mask], dense_edges[mask])
    assert np.all(reconstructed[~mask] == 0)

    bits = rng.integers(0, 2, size=(n, 192), dtype=np.uint8)
    weights = rng.uniform(0.3, 1.0, size=(n,)).astype(np.float32)
    key = bytes(range(32))
    ser = CredentialSerializer()

    cred_dense = ProtectedCredentialDTO(
        credential_id="cred_dense_cmp",
        subject_id="subj_cmp",
        salt="salt_cmp",
        app_id="app_cmp",
        bitstrings=bits,
        weights=weights,
        num_triangles=n,
        edge_tokens=dense_edges,
        in_dim=256,
    )
    cred_sparse = ProtectedCredentialDTO(
        credential_id="cred_sparse_cmp",
        subject_id="subj_cmp",
        salt="salt_cmp",
        app_id="app_cmp",
        bitstrings=bits,
        weights=weights,
        num_triangles=n,
        edge_tokens=reconstructed,
        in_dim=256,
        edge_mask=mask,
    )
    env_dense = ser.serialize_protected(cred_dense, storage_key=key, encrypt=True)
    env_sparse = ser.serialize_protected(cred_sparse, storage_key=key, encrypt=True, compact_edges=True)

    assert env_dense.has_sparse_edges is False
    assert env_sparse.has_sparse_edges is True
    assert env_sparse.total_bytes <= int(0.50 * env_dense.total_bytes)

    dec = ser.deserialize_protected(env_sparse.raw_bytes, storage_key=key, subject_id="subj_cmp", app_id="app_cmp")
    assert dec.edge_mask is not None and np.array_equal(dec.edge_mask, mask)
    assert np.array_equal(dec.edge_tokens, reconstructed)

    bad_payload = bytearray(env_sparse.raw_bytes)
    bad_payload[-7] ^= 0xFF
    with pytest.raises(BiometricServiceException) as exc_info:
        ser.deserialize_protected(bytes(bad_payload), storage_key=key, subject_id="subj_cmp", app_id="app_cmp")
    assert exc_info.value.code == BiometricErrorCode.AEAD_INTEGRITY_FAILURE

    img1 = "dataset/FVC2002/DB1_B/101_1.tif"
    img2 = "dataset/FVC2002/DB1_B/101_2.tif"
    img_imp = "dataset/FVC2002/DB1_B/102_1.tif"
    if os.path.exists(img1):
        svc = ProductionBiometricService(
            feature_mode="mcc_spectral",
            engine="classical",
            default_dataset="DB1_B",
            compact_edges=True,
            edge_top_k=16,
        )
        svc.repository = CredentialRepository(db_path=str(tmp_path / "sparse.db"))
        enr = svc.enroll(img1, subject_id="u_sp", salt="salt_sp", app_id="app_sp", dpi=500.0)
        assert enr["status"] == "success"
        res_gen = svc.verify(img2, subject_id="u_sp", app_id="app_sp", dataset="DB1_B", dpi=500.0)
        res_imp = svc.verify(img_imp, subject_id="u_sp", app_id="app_sp", dataset="DB1_B", dpi=500.0)
        assert res_gen.score > res_imp.score
