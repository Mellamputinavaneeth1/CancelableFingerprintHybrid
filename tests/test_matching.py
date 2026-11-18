import numpy as np
import pytest
from src.matching.bipartite_matcher import (
    match_unprotected_descriptors,
    match_protected_templates,
    match_protected_greedy,
    build_augmented_cost
)

def test_unprotected_matching():
    rng = np.random.default_rng(42)
    desc1 = rng.uniform(0.1, 0.9, size=(10, 12)).astype(np.float32)
    weights1 = np.ones(10, dtype=np.float32) * 0.9
    
    score_self = match_unprotected_descriptors(desc1, desc1, weights1, weights1, distance_threshold=0.3)
    assert score_self > 0.8
    
    desc2 = rng.uniform(0.1, 0.9, size=(10, 12)).astype(np.float32)
    weights2 = np.ones(10, dtype=np.float32) * 0.9
    score_diff = match_unprotected_descriptors(desc1, desc2, weights1, weights2, distance_threshold=0.2)
    assert score_diff < score_self

def test_coverage_aware_asymmetric_template_matching():
    """
    Tests that the coverage-aware normalization properly penalizes unmatched triangles,
    preventing a small partial match from achieving a false 1.0 similarity.
    """
    rng = np.random.default_rng(100)
    bits_enrolled = rng.integers(0, 2, size=(100, 32), dtype=np.uint8)
    weights_enrolled = np.ones(100, dtype=np.float32)
    
    bits_query = bits_enrolled[:10].copy()
    weights_query = np.ones(10, dtype=np.float32)
    
    # 10 matches out of 100 enrolled: 2 * 10 / (10 + 100) ~ 0.18
    score_asymmetric = match_protected_templates(
        bits_query, bits_enrolled, weights_query, weights_enrolled, hamming_threshold=0.22
    )
    assert score_asymmetric < 0.25, f"Expected coverage penalty, but got score {score_asymmetric}"
    
    # Matching 10 against 10 identical triangles should yield a high score (> 0.85)
    score_symmetric = match_protected_templates(
        bits_query, bits_query, weights_query, weights_query, hamming_threshold=0.22
    )
    assert score_symmetric > 0.85

def test_protected_matching_impostor_rejection():
    rng = np.random.default_rng(200)
    bits_user1 = rng.integers(0, 2, size=(25, 32), dtype=np.uint8)
    bits_user2 = rng.integers(0, 2, size=(25, 32), dtype=np.uint8)
    weights = np.ones(25, dtype=np.float32)
    
    score_imp = match_protected_templates(
        bits_user1, bits_user2, weights, weights, hamming_threshold=0.22
    )
    assert score_imp < 0.08
    score_imp_g1 = match_protected_greedy(
        bits_user1, bits_user2, weights, weights, hamming_threshold=0.22, mode="greedy_1to1"
    )
    assert score_imp_g1 < 0.08
    score_imp_gnn = match_protected_greedy(
        bits_user1, bits_user2, weights, weights, hamming_threshold=0.22, mode="greedy_nn"
    )
    assert score_imp_gnn < 0.15

def test_augmented_cost_no_free_dummy_bypass():
    """
    THE CRITICAL BUG TEST: With 2 query vs 2 enrolled descriptors, the old
    identity-based slack allowed total cost = 0 by routing through off-diagonal
    zeros. The corrected formulation must prevent this.
    """
    rng = np.random.default_rng(42)
    bits_q = rng.integers(0, 2, size=(2, 32), dtype=np.uint8)
    bits_e = np.ones((2, 32), dtype=np.uint8) - bits_q  # maximally different
    weights = np.ones(2, dtype=np.float32)
    
    score = match_protected_templates(bits_q, bits_e, weights, weights, hamming_threshold=0.22)
    # All distances ~1.0, well above tau=0.22, so NO valid real matches -> score must be 0.0
    assert score == 0.0, f"Expected 0.0 for maximally different templates, got {score}"

def test_self_match_high_score():
    """Identical templates must produce a high matching score."""
    rng = np.random.default_rng(99)
    bits = rng.integers(0, 2, size=(15, 32), dtype=np.uint8)
    weights = np.ones(15, dtype=np.float32) * 0.9
    
    score = match_protected_templates(bits, bits, weights, weights, hamming_threshold=0.22)
    assert score > 0.85, f"Self-match score too low: {score}"
    score_g1 = match_protected_greedy(bits, bits, weights, weights, hamming_threshold=0.22, mode="greedy_1to1")
    assert score_g1 > 0.85, f"Greedy 1-to-1 self-match score too low: {score_g1}"
    score_gnn = match_protected_greedy(bits, bits, weights, weights, hamming_threshold=0.22, mode="greedy_nn")
    assert score_gnn > 0.85, f"Greedy NN self-match score too low: {score_gnn}"

def test_asymmetric_10_vs_100_coverage_penalty():
    """10 matching descriptors out of 100 enrolled should yield low score due to coverage penalty."""
    rng = np.random.default_rng(100)
    bits_enrolled = rng.integers(0, 2, size=(100, 32), dtype=np.uint8)
    weights_enrolled = np.ones(100, dtype=np.float32)
    
    bits_query = bits_enrolled[:10].copy()
    weights_query = np.ones(10, dtype=np.float32)
    
    score = match_protected_templates(bits_query, bits_enrolled, weights_query, weights_enrolled, hamming_threshold=0.22)
    # 10 perfect matches out of 110 total descriptors: 2*10/(10+100) ≈ 0.18
    assert score < 0.25, f"Expected coverage penalty, got {score}"

def test_no_valid_matches_zero_score():
    """When ALL pairwise distances >= tau, score must be exactly 0.0."""
    bits_q = np.zeros((5, 32), dtype=np.uint8)
    bits_e = np.ones((5, 32), dtype=np.uint8)
    weights = np.ones(5, dtype=np.float32)
    
    score = match_protected_templates(bits_q, bits_e, weights, weights, hamming_threshold=0.22)
    assert score == 0.0, f"Expected 0.0 when all distances >= tau, got {score}"
    score_g1 = match_protected_greedy(bits_q, bits_e, weights, weights, hamming_threshold=0.22, mode="greedy_1to1")
    assert score_g1 == 0.0
    score_gnn = match_protected_greedy(bits_q, bits_e, weights, weights, hamming_threshold=0.22, mode="greedy_nn")
    assert score_gnn == 0.0

def test_boundary_distance_at_threshold():
    """Descriptors exactly at or above threshold tau should NOT contribute to the match score."""
    # Create bits that differ in exactly 7 positions (7/32 = 0.21875 < 0.22) -> should match
    bits_q = np.zeros((1, 32), dtype=np.uint8)
    bits_e = np.zeros((1, 32), dtype=np.uint8)
    bits_e[0, :7] = 1
    weights = np.ones(1, dtype=np.float32)
    
    score_below = match_protected_templates(bits_q, bits_e, weights, weights, hamming_threshold=0.22)
    assert score_below > 0.0, f"Distance below threshold should produce positive score, got {score_below}"
    score_below_g1 = match_protected_greedy(bits_q, bits_e, weights, weights, hamming_threshold=0.22, mode="greedy_1to1")
    assert score_below_g1 > 0.0
    
    # 8 positions: 8/32 = 0.25 >= 0.22 -> should NOT match
    bits_e2 = np.zeros((1, 32), dtype=np.uint8)
    bits_e2[0, :8] = 1
    score_above = match_protected_templates(bits_q, bits_e2, weights, weights, hamming_threshold=0.22)
    assert score_above == 0.0, f"Distance above threshold should yield 0.0, got {score_above}"
    score_above_g1 = match_protected_greedy(bits_q, bits_e2, weights, weights, hamming_threshold=0.22, mode="greedy_1to1")
    assert score_above_g1 == 0.0

    # Test error handling on invalid greedy mode and 1D support
    with pytest.raises(ValueError, match="Unknown greedy mode"):
        match_protected_greedy(bits_q, bits_e, weights, weights, mode="invalid_mode")
    s_1d_templ = match_protected_templates(bits_q[0], bits_e[0], 1.0, 1.0, hamming_threshold=0.22)
    s_1d_greedy = match_protected_greedy(bits_q[0], bits_e[0], 1.0, 1.0, hamming_threshold=0.22)
    assert s_1d_templ > 0.0 and s_1d_greedy > 0.0

def test_query_fewer_than_enrolled():
    """3 query descriptors vs 8 enrolled: only up to 3 can match."""
    rng = np.random.default_rng(400)
    bits_e = rng.integers(0, 2, size=(8, 32), dtype=np.uint8)
    bits_q = bits_e[:3].copy()  # First 3 match perfectly
    w_q = np.ones(3, dtype=np.float32)
    w_e = np.ones(8, dtype=np.float32)
    
    score = match_protected_templates(bits_q, bits_e, w_q, w_e, hamming_threshold=0.22)
    assert 0.2 < score < 0.7, f"Score out of expected range: {score}"

def test_query_more_than_enrolled():
    """12 query descriptors vs 5 enrolled: only up to 5 can match."""
    rng = np.random.default_rng(500)
    bits_e = rng.integers(0, 2, size=(5, 32), dtype=np.uint8)
    bits_q = np.concatenate([bits_e, rng.integers(0, 2, size=(7, 32), dtype=np.uint8)], axis=0)
    w_q = np.ones(12, dtype=np.float32)
    w_e = np.ones(5, dtype=np.float32)
    
    score = match_protected_templates(bits_q, bits_e, w_q, w_e, hamming_threshold=0.22)
    assert 0.2 < score < 0.7, f"Score out of expected range: {score}"

def test_unprotected_self_match():
    """Unprotected matching of identical descriptors should produce a high score."""
    rng = np.random.default_rng(600)
    desc = rng.uniform(0.1, 0.9, size=(10, 12)).astype(np.float32)
    weights = np.ones(10, dtype=np.float32) * 0.9
    
    score = match_unprotected_descriptors(desc, desc, weights, weights, distance_threshold=0.3)
    assert score > 0.85, f"Unprotected self-match score too low: {score}"

def test_unprotected_no_valid_matches():
    """Maximally different continuous descriptors with tight threshold should yield 0.0."""
    desc_q = np.zeros((5, 12), dtype=np.float32)
    desc_e = np.ones((5, 12), dtype=np.float32)
    weights = np.ones(5, dtype=np.float32)
    
    score = match_unprotected_descriptors(desc_q, desc_e, weights, weights, distance_threshold=0.1)
    assert score == 0.0, f"Expected 0.0, got {score}"

def test_augmented_cost_matrix_structure():
    """Validates the corrected augmented cost matrix has proper forbidden entries."""
    dist = np.array([[0.1, 0.5], [0.3, 0.05]], dtype=np.float32)
    tau = 0.22
    big = 1e6
    C = build_augmented_cost(dist, tau, big)
    
    assert C.shape == (4, 4)
    # Real block: dist < tau -> actual distance; dist >= tau -> big
    assert C[0, 0] == pytest.approx(0.1)
    assert C[0, 1] == pytest.approx(big)
    assert C[1, 0] == pytest.approx(big)
    assert C[1, 1] == pytest.approx(0.05)
    # Uq block: diagonal = tau / 2, off-diagonal = big
    assert C[0, 2] == pytest.approx(tau / 2.0)
    assert C[0, 3] == pytest.approx(big)
    assert C[1, 2] == pytest.approx(big)
    assert C[1, 3] == pytest.approx(tau / 2.0)
    # Ue block: diagonal = tau / 2, off-diagonal = big
    assert C[2, 0] == pytest.approx(tau / 2.0)
    assert C[2, 1] == pytest.approx(big)
    assert C[3, 0] == pytest.approx(big)
    assert C[3, 1] == pytest.approx(tau / 2.0)
    # DD block: all zeros
    assert C[2, 2] == 0.0
    assert C[2, 3] == 0.0
    assert C[3, 2] == 0.0
    assert C[3, 3] == 0.0

def test_unprotected_matching_with_zscore_standardization():
    """
    Tests that providing empirical mean and std standardizes continuous features,
    preventing high-variance features from skewing Euclidean distances.
    """
    rng = np.random.default_rng(777)
    # 9-D features where feature 2 has high mean (20.0) and high std (5.0),
    # while feature 0 has low std (0.1)
    base_desc = rng.uniform(0.1, 0.9, size=(10, 9)).astype(np.float32)
    base_desc[:, 2] = rng.normal(20.0, 5.0, size=10)
    
    mean = np.mean(base_desc, axis=0)
    std = np.std(base_desc, axis=0)
    weights = np.ones(10, dtype=np.float32)
    
    # Self-match with standardization
    score_std = match_unprotected_descriptors(base_desc, base_desc, weights, weights, 
                                             distance_threshold=0.6, mean=mean, std=std)
    assert score_std > 0.85
    
    # Perturb feature 2 slightly (0.5 mm in physical perimeter, which is only 0.1 std)
    perturbed_desc = base_desc.copy()
    perturbed_desc[:, 2] += 0.5
    
    # Under standardization, small 0.1 std perturbation should still match well
    score_pert = match_unprotected_descriptors(base_desc, perturbed_desc, weights, weights,
                                              distance_threshold=0.6, mean=mean, std=std)
    assert score_pert > 0.75

def test_cyclic_matching_support():
    """
    Tests that match_protected_templates and match_unprotected_descriptors accept
    (N_e, 3, M) / (N_e, 3, D) cyclic representations and correctly compute
    min_{pi in C3} distance over cyclic permutations.
    """
    rng = np.random.default_rng(888)
    # 5 query triangles
    q_bits = rng.integers(0, 2, size=(5, 32), dtype=np.uint8)
    
    # 5 enrolled triangles, each with 3 cyclic variations
    e_bits_cyclic = np.zeros((5, 3, 32), dtype=np.uint8)
    # Permutation 1 matches query exactly for all triangles, others have 50% Hamming distance
    for i in range(5):
        e_bits_cyclic[i, 0] = 1 - q_bits[i]  # 100% diff
        e_bits_cyclic[i, 1] = q_bits[i].copy()  # 0% diff (perfect match)
        e_bits_cyclic[i, 2] = rng.integers(0, 2, size=32, dtype=np.uint8)
        
    weights = np.ones(5, dtype=np.float32)
    score = match_protected_templates(q_bits, e_bits_cyclic, weights, weights, hamming_threshold=0.22)
    # Because cyclic shift 1 matches perfectly, score must be 1.0
    assert abs(score - 1.0) < 1e-5
    
    # Also test unprotected continuous cyclic matching
    q_desc = rng.uniform(0.2, 0.8, size=(5, 9)).astype(np.float32)
    e_desc_cyclic = np.zeros((5, 3, 9), dtype=np.float32)
    for i in range(5):
        e_desc_cyclic[i, 0] = q_desc[i] + 0.5
        e_desc_cyclic[i, 1] = q_desc[i] + 0.4
        e_desc_cyclic[i, 2] = q_desc[i].copy()  # perfect match
    score_unprot = match_unprotected_descriptors(q_desc, e_desc_cyclic, weights, weights, distance_threshold=0.25)
    assert abs(score_unprot - 1.0) < 1e-5

def test_graph_consensus_matching():
    """
    Tests topological graph consensus matching:
    1. Adjacency computation from triangle vertices.
    2. Shared-edge/shared-vertex consistency evaluation.
    3. Graph-augmented score bonus and violation handling without coordinate leakage.
    """
    from src.minutiae.extractor import Minutia
    from src.matching.graph_consensus import (
        compute_triangle_adjacency,
        evaluate_match_graph_consistency,
        match_protected_with_graph_consensus
    )
    
    m0 = Minutia(10.0, 10.0, 0.1, 0, 0.9)
    m1 = Minutia(30.0, 10.0, 0.2, 0, 0.9)
    m2 = Minutia(20.0, 30.0, 0.3, 1, 0.85)
    m3 = Minutia(40.0, 30.0, 0.4, 0, 0.8)
    
    # Triangle 0: (m0, m1, m2)
    # Triangle 1: (m1, m2, m3) -> shares edge (m1, m2) with tri 0 (2 shared vertices)
    tris = [(m0, m1, m2), (m1, m2, m3)]
    adj = compute_triangle_adjacency(tris)
    assert adj.shape == (2, 2)
    assert adj[0, 0] == 3 and adj[1, 1] == 3
    assert adj[0, 1] == 2 and adj[1, 0] == 2
    
    # Consistent matches: tri 0 -> tri 0, tri 1 -> tri 1
    eval_consistent = evaluate_match_graph_consistency([(0, 0), (1, 1)], adj, adj)
    assert eval_consistent["consistent_edges"] == 1
    assert eval_consistent["violations"] == 0
    assert eval_consistent["topological_ratio"] == 1.0
    
    # Inconsistent matches: query adjacent (0, 1) mapped to disjoint enrolled (adj=0)
    adj_disjoint = np.array([[3, 0], [0, 3]], dtype=np.uint8)
    eval_inconsistent = evaluate_match_graph_consistency([(0, 0), (1, 1)], adj, adj_disjoint)
    assert eval_inconsistent["consistent_edges"] == 0
    assert eval_inconsistent["violations"] == 1
    assert eval_inconsistent["topological_ratio"] == 0.0
    
    # Match with graph consensus
    rng = np.random.default_rng(999)
    q_bits = rng.integers(0, 2, size=(2, 32), dtype=np.uint8)
    e_bits = q_bits.copy() # identical
    wts = np.ones(2, dtype=np.float32)
    score, audit = match_protected_with_graph_consensus(q_bits, e_bits, wts, wts, adj, adj)
    assert score > 0.99
    assert audit["consistent_edges"] == 1


def test_matrix_packing_and_storage_accounting():
    """
    Tests lossless 2-bit symmetric packing/unpacking and template storage breakdown.
    """
    from src.matching.graph_consensus import (
        pack_adjacency_matrix,
        unpack_adjacency_matrix,
        compute_template_storage_breakdown
    )
    
    # 1. Edge cases: empty and 1x1
    adj0 = np.empty((0, 0), dtype=np.uint8)
    assert pack_adjacency_matrix(adj0) == b""
    assert unpack_adjacency_matrix(b"", 0).shape == (0, 0)
    
    adj1 = np.array([[3]], dtype=np.uint8)
    assert pack_adjacency_matrix(adj1) == b""
    assert np.array_equal(unpack_adjacency_matrix(b"", 1), adj1)
    
    # 2. Symmetric matrix roundtrip (N=10)
    rng = np.random.default_rng(123)
    n = 10
    raw_vals = rng.integers(0, 4, size=(n, n), dtype=np.uint8)
    adj_test = (raw_vals + raw_vals.T) % 4
    np.fill_diagonal(adj_test, 3)
    
    packed = pack_adjacency_matrix(adj_test)
    num_upper = n * (n - 1) // 2  # 45
    expected_bytes = (num_upper + 3) // 4  # 12 bytes
    assert len(packed) == expected_bytes
    
    unpacked = unpack_adjacency_matrix(packed, n)
    assert np.array_equal(unpacked, adj_test)
    
    # 3. Storage breakdown calculation (e.g. N=144 triangles from DB1_A)
    breakdown = compute_template_storage_breakdown(num_triangles=144, in_dim=9, out_dim=32)
    assert breakdown["descriptor_bytes"] == 144 * 4  # 576 bytes
    assert breakdown["weights_bytes"] == 144 * 2     # 288 bytes (float16)
    assert breakdown["matrix_uncompressed_bytes"] == 144 * 144  # 20736 bytes
    assert breakdown["matrix_symmetric_2bit_packed_bytes"] == (144 * 143 // 2 + 3) // 4  # 2574 bytes
    assert breakdown["total_credential_packed_bytes"] < breakdown["total_credential_uncompressed_bytes"]
    assert breakdown["true_compression_ratio_packed"] > 0.35  # >35% reduction vs continuous baseline

    # 4. Dataset-wide storage statistics calculation
    from src.matching.graph_consensus import calculate_dataset_storage_statistics
    mock_counts = [100, 144, 160, 200]
    ds_stats = calculate_dataset_storage_statistics(mock_counts, in_dim=9, out_dim=32)
    assert ds_stats["num_enrolled_templates"] == 4
    assert ds_stats["triangles"]["mean"] == 151.0
    assert ds_stats["packed_credential_bytes"]["min"] < ds_stats["packed_credential_bytes"]["max"]
    assert ds_stats["aggregate_serialized_compression"]["packed_vs_baseline"] > 0.30



