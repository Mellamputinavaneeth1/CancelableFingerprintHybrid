import numpy as np
import pytest
from src.protection.kdf import derive_application_key, validate_master_secret, generate_master_secret
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors
from src.protection.bloom import encode_to_bloom_filter, bloom_jaccard_similarity

def test_master_secret_validation():
    # CSPRNG generated 32-byte secret
    secret = generate_master_secret()
    assert len(secret) == 32
    assert validate_master_secret(secret) == secret
    
    # 64-character hex string
    hex_secret = "0123456789abcdef" * 4
    assert len(hex_secret) == 64
    assert len(validate_master_secret(hex_secret)) == 32
    
    # Secret too short (< 32 bytes)
    with pytest.raises(ValueError, match="below the required 32 bytes"):
        validate_master_secret("ShortSecret123")

def test_hkdf_derivation_and_domain_separation():
    master = generate_master_secret()
    k1 = derive_application_key(master, salt="salt_app1", app_id="banking")
    k2 = derive_application_key(master, salt="salt_app2", app_id="banking")
    k3 = derive_application_key(master, salt="salt_app1", app_id="healthcare")
    
    assert len(k1) == 32
    assert k1 != k2
    assert k1 != k3
    
    # Exact reproducibility given identical inputs
    k1_repeat = derive_application_key(master, salt="salt_app1", app_id="banking")
    assert k1 == k1_repeat

def test_orthonormal_embedding():
    master = generate_master_secret()
    
    # 1. Test 12-D security ablation branch
    W12 = derive_orthonormal_matrix(master, "saltA", "appA", in_dim=12, out_dim=32)
    assert W12.shape == (32, 12)
    gram12 = W12.T @ W12
    max_err12 = np.max(np.abs(gram12 - np.eye(12)))
    assert max_err12 < 1e-6, f"Gram matrix tolerance exceeded: {max_err12}"
    
    # Verify column-orthonormal, NOT row-orthonormal (rank <= 12 < 32)
    assert not np.allclose(W12 @ W12.T, np.eye(32), atol=1e-3)
    
    # 2. Test 9-D primary candidate
    W9 = derive_orthonormal_matrix(master, "saltB", "appB", in_dim=9, out_dim=32)
    assert W9.shape == (32, 9)
    gram9 = W9.T @ W9
    max_err9 = np.max(np.abs(gram9 - np.eye(9)))
    assert max_err9 < 1e-6, f"Gram matrix tolerance exceeded: {max_err9}"
    
    # 3. Explicit binary indicator quantization definition: b_j = I(p_j >= 0)
    dummy_desc = np.random.uniform(0, 1, size=(5, 9)).astype(np.float32)
    proj = project_descriptors(dummy_desc, W9)
    assert proj.shape == (5, 32)
    bits = quantize_descriptors(proj)
    expected_bits = (proj >= 0.0).astype(np.uint8)
    np.testing.assert_array_equal(bits, expected_bits)

def test_cross_key_decorrelation():
    master = generate_master_secret()
    desc = np.random.uniform(0, 1, size=(20, 12)).astype(np.float32)
    W1 = derive_orthonormal_matrix(master, "saltA", "app1", in_dim=12, out_dim=32)
    W2 = derive_orthonormal_matrix(master, "saltB", "app2", in_dim=12, out_dim=32)
    
    b1 = quantize_descriptors(project_descriptors(desc, W1))
    b2 = quantize_descriptors(project_descriptors(desc, W2))
    
    hamming_dist = np.mean(b1 != b2)
    assert 0.35 <= hamming_dist <= 0.65

def test_bloom_filter_encoding():
    bitstrings = np.array([
        [1, 0, 1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0, 1, 1]
    ], dtype=np.uint8)
    
    bloom1 = encode_to_bloom_filter(bitstrings, filter_size_bits=128, num_hashes=3, salt="salt1")
    bloom2 = encode_to_bloom_filter(bitstrings, filter_size_bits=128, num_hashes=3, salt="salt1")
    bloom_diff = encode_to_bloom_filter(bitstrings, filter_size_bits=128, num_hashes=3, salt="salt2")
    
    assert bloom1.shape == (128,)
    assert np.array_equal(bloom1, bloom2)
    assert bloom_jaccard_similarity(bloom1, bloom2) == 1.0
    assert bloom_jaccard_similarity(bloom1, bloom_diff) < 0.6

def test_crypto_gaussian_generator():
    """
    Tests that crypto_gaussian_matrix is purely deterministic from key_bytes
    using HMAC-SHA256 PRF + Box-Muller, without relying on NumPy's PCG64 PRNG.
    """
    from src.protection.projection import crypto_gaussian_matrix
    key1 = b"TestKeyForGaussianGenerator_256Bits!"[:32]
    A1 = crypto_gaussian_matrix(key1, 32, 12)
    A2 = crypto_gaussian_matrix(key1, 32, 12)
    
    assert A1.shape == (32, 12)
    np.testing.assert_array_equal(A1, A2)
    
    # Different key produces decorrelated Gaussian matrix
    key2 = b"DifferentKeyForGaussian_256Bits!"[:32]
    A_diff = crypto_gaussian_matrix(key2, 32, 12)
    assert not np.array_equal(A1, A_diff)
    corr = np.corrcoef(A1.flatten(), A_diff.flatten())[0, 1]
    assert abs(corr) < 0.2
