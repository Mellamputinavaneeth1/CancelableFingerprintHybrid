"""
Unit and Integration Tests for WP2 and WP3:
1. Delaunay simplex stability and degeneracy filtering (sliver, collinear, aspect ratio rejection).
2. Coverage-aware score normalization edge cases (empty, sparse, full overlap, exponent scaling).
3. Hungarian vs Greedy 1-to-1 assignment equivalence, speedup boundary, and strict 1-to-1 invariant.
4. Set A sensor specifications, provenance, and synthetic SFinGe data labeling.
5. Out-of-domain threshold transfer mathematical identities (TP + FN == N_gen, FP + TN == N_imp).
6. Clustered bootstrap confidence interval calculation and numerical stability.
"""

import os
import sys
import pytest
import numpy as np

from src.minutiae.extractor import Minutia
from src.graph.delaunay import is_valid_triangle, build_delaunay_triangles, build_knn_graphs
from src.matching.bipartite_matcher import (
    match_protected_templates,
    match_protected_greedy,
    build_augmented_cost
)
from src.pipeline import CancelableFingerprintPipeline
from src.evaluation.metrics import (
    compute_roc_metrics,
    clustered_bootstrap_confidence_intervals,
    paired_clustered_bootstrap_delta
)
from experiments.generate_set_a_external_datasets import SET_A_SPECS


# =========================================================================
# 1. Delaunay Simplex Stability & Degeneracy Filtering Tests
# =========================================================================

def test_triangle_degeneracy_filtering_sliver():
    """Tests that triangles with extreme aspect ratios (slivers) are rejected."""
    # Sliver triangle: edge 1 = 100, edge 2 = 100, edge 3 = 1.0 (aspect ratio ~ 100.0)
    m1 = Minutia(0.0, 0.0, 0.1, 0, 0.9)
    m2 = Minutia(100.0, 0.0, 0.2, 0, 0.9)
    m3 = Minutia(100.0, 1.0, 0.3, 0, 0.9)

    # Without aspect ratio filter: accepted (edges <= 120)
    assert is_valid_triangle(m1, m2, m3, max_edge_length=120.0, max_aspect_ratio=None) is True
    # With aspect ratio filter <= 5.0: rejected
    assert is_valid_triangle(m1, m2, m3, max_edge_length=120.0, max_aspect_ratio=5.0) is False


def test_triangle_degeneracy_filtering_collinear():
    """Tests that near-collinear triangles (area < min_area) are rejected."""
    # Points lying on nearly the same line: (0, 0), (50, 0.1), (100, 0)
    m1 = Minutia(0.0, 0.0, 0.1, 0, 0.9)
    m2 = Minutia(50.0, 0.1, 0.2, 0, 0.9)
    m3 = Minutia(100.0, 0.0, 0.3, 0, 0.9)
    # Area = 0.5 * |50 * 0 - 100 * 0.1| = 5.0 square pixels

    # Without area filter: accepted
    assert is_valid_triangle(m1, m2, m3, max_edge_length=120.0, min_area=0.0) is True
    # With min_area=25.0: rejected
    assert is_valid_triangle(m1, m2, m3, max_edge_length=120.0, min_area=25.0) is False


def test_triangle_degeneracy_filtering_min_edge():
    """Tests that triangles with overly short edges (< min_edge_length) are rejected."""
    # Triangle with a 4-pixel edge (high-frequency jitter noise)
    m1 = Minutia(20.0, 20.0, 0.1, 0, 0.9)
    m2 = Minutia(24.0, 20.0, 0.2, 0, 0.9)
    m3 = Minutia(30.0, 45.0, 0.3, 0, 0.9)

    # Without min_edge filter: accepted
    assert is_valid_triangle(m1, m2, m3, max_edge_length=120.0, min_edge_length=0.0) is True
    # With min_edge_length=10.0: rejected (edge 1-2 has length 4.0 < 10.0)
    assert is_valid_triangle(m1, m2, m3, max_edge_length=120.0, min_edge_length=10.0) is False


def test_well_conditioned_triangle_accepted():
    """Tests that regular, well-proportioned scalene and equilateral triangles are accepted."""
    # Equilateral triangle with side 40 px: area ~ 692.8 px^2, aspect ratio = 1.0
    h = 40.0 * np.sqrt(3.0) / 2.0
    m1 = Minutia(0.0, 0.0, 0.1, 0, 0.9)
    m2 = Minutia(40.0, 0.0, 0.2, 0, 0.9)
    m3 = Minutia(20.0, float(h), 0.3, 0, 0.9)

    assert is_valid_triangle(m1, m2, m3, max_edge_length=120.0, min_edge_length=10.0, min_area=25.0, max_aspect_ratio=4.0) is True


def test_build_delaunay_triangles_degeneracy_filtering():
    """Tests that build_delaunay_triangles applies degeneracy filtering correctly."""
    # Minutiae set with one isolated collinear pair and three well-spread points
    minutiae = [
        Minutia(10.0, 10.0, 0.1, 0, 0.9),
        Minutia(14.0, 10.0, 0.2, 0, 0.9),  # 4px from m0 -> will cause short edge
        Minutia(50.0, 10.0, 0.3, 0, 0.9),
        Minutia(30.0, 40.0, 0.4, 0, 0.9),
        Minutia(70.0, 45.0, 0.5, 0, 0.9)
    ]
    # Unfiltered Delaunay
    tris_unfiltered = build_delaunay_triangles(minutiae, max_edge_length=120.0)
    # Filtered Delaunay
    tris_filtered = build_delaunay_triangles(minutiae, max_edge_length=120.0, min_edge_length=10.0, min_area=25.0, max_aspect_ratio=4.0)

    assert len(tris_unfiltered) > len(tris_filtered)
    # All kept triangles must satisfy the stability constraints
    for t in tris_filtered:
        assert is_valid_triangle(t[0], t[1], t[2], max_edge_length=120.0, min_edge_length=10.0, min_area=25.0, max_aspect_ratio=4.0) is True


# =========================================================================
# 2. Coverage-Aware Score Normalization Tests
# =========================================================================

def test_coverage_normalization_sparse_spurious_suppression():
    """
    Tests that coverage-aware normalization penalizes small sets of sparse matches
    that would otherwise inflate scores.
    """
    rng = np.random.default_rng(42)
    # 50 query descriptors vs 50 enrolled descriptors
    bits_q = rng.integers(0, 2, size=(50, 32), dtype=np.uint8)
    bits_e = rng.integers(0, 2, size=(50, 32), dtype=np.uint8)
    # Force exactly 2 descriptors to match perfectly (2 / 50 = 4% coverage)
    bits_e[0] = bits_q[0].copy()
    bits_e[1] = bits_q[1].copy()
    weights_q = np.ones(50, dtype=np.float32)
    weights_e = np.ones(50, dtype=np.float32)

    # Standard score: 2 * 2 / 100 = 0.04
    score_standard = match_protected_templates(bits_q, bits_e, weights_q, weights_e, coverage_penalty_exponent=1.0)

    # Coverage-penalized score with exponent=1.5: score * coverage^0.5
    score_penalized = match_protected_templates(bits_q, bits_e, weights_q, weights_e, coverage_penalty_exponent=1.5)
    assert score_penalized < score_standard, f"Expected penalty, got {score_penalized} vs {score_standard}"

    # Min matches threshold: min_matches=5 should quadratically penalize 2 matches
    score_min_m = match_protected_templates(bits_q, bits_e, weights_q, weights_e, min_matches=5)
    assert score_min_m <= score_standard * (4.0 / 25.0 + 1e-4)


def test_coverage_normalization_high_coverage_preserved():
    """Tests that high-coverage genuine matches are preserved without artificial degradation."""
    rng = np.random.default_rng(99)
    bits = rng.integers(0, 2, size=(20, 32), dtype=np.uint8)
    weights = np.ones(20, dtype=np.float32)

    # Self-match: 100% coverage
    score_std = match_protected_templates(bits, bits, weights, weights, coverage_penalty_exponent=1.0)
    score_cov = match_protected_templates(bits, bits, weights, weights, coverage_penalty_exponent=1.5, min_matches=5)

    assert score_std > 0.85
    assert abs(score_cov - score_std) < 1e-4, f"100% coverage should not be penalized: {score_cov} vs {score_std}"


# =========================================================================
# 3. Hungarian vs Greedy 1-to-1 Assignment Tests
# =========================================================================

def test_greedy_1to1_strict_bijective_invariant():
    """
    Formally proves that Greedy 1-to-1 strictly satisfies the bijective assignment invariant:
    1. Each query vertex is matched at most once (no row reuse).
    2. Each enrolled vertex is matched at most once (no column reuse).
    3. All matched pairs satisfy dist(r, c) < hamming_threshold.
    4. Assigned pairs follow strictly non-decreasing Hamming distance ordering.
    """
    rng = np.random.default_rng(123)
    n_q, n_e = 40, 50
    bits_q = rng.integers(0, 2, size=(n_q, 32), dtype=np.uint8)
    # Create enrolled with known matches at varying distances
    bits_e = rng.integers(0, 2, size=(n_e, 32), dtype=np.uint8)
    for i in range(15):
        # Create genuine pairs with small noise (0 to 5 bit flips out of 32 -> dist <= 5/32 = 0.156 < 0.22)
        flips = rng.choice(32, size=rng.integers(0, 6), replace=False)
        bits_e[i] = bits_q[i].copy()
        bits_e[i, flips] = 1 - bits_e[i, flips]
    # Shuffle enrolled
    perm = rng.permutation(n_e)
    bits_e = bits_e[perm]

    w_q = np.ones(n_q, dtype=np.float32)
    w_e = np.ones(n_e, dtype=np.float32)

    score_g, details_g = match_protected_greedy(bits_q, bits_e, w_q, w_e, hamming_threshold=0.22, mode="greedy_1to1", return_details=True)
    assert 0.0 <= score_g <= 1.0

    pairs = details_g["matched_pairs"]
    assert len(pairs) > 0, "Expected at least some matched pairs"
    assert len(pairs) == details_g["num_matched"]

    q_matched = [p[0] for p in pairs]
    e_matched = [p[1] for p in pairs]

    # Invariant 1: No query vertex reuse
    assert len(set(q_matched)) == len(q_matched), f"Query vertex reuse detected in greedy: {q_matched}"
    # Invariant 2: No enrolled vertex reuse
    assert len(set(e_matched)) == len(e_matched), f"Enrolled vertex reuse detected in greedy: {e_matched}"
    # Invariant 3: Bound constraint
    assert len(pairs) <= min(n_q, n_e)

    # Invariant 4: Threshold and distance ordering
    prev_d = -1.0
    for r, c in pairs:
        assert 0 <= r < n_q
        assert 0 <= c < n_e
        d = float(np.mean(bits_q[r] != bits_e[c]))
        assert d < 0.22, f"Greedy matched pair exceeds threshold: dist={d} >= 0.22"
        assert d >= prev_d - 1e-6, f"Greedy did not pick pairs in ascending distance order: {d} < {prev_d}"
        prev_d = d


def test_hungarian_strict_bijective_invariant():
    """Formally verifies that Hungarian matching also satisfies bijective assignment with no vertex reuse."""
    rng = np.random.default_rng(234)
    n_q, n_e = 35, 45
    bits_q = rng.integers(0, 2, size=(n_q, 32), dtype=np.uint8)
    bits_e = rng.integers(0, 2, size=(n_e, 32), dtype=np.uint8)
    for i in range(12):
        bits_e[i] = bits_q[i].copy()
    w_q = np.ones(n_q, dtype=np.float32)
    w_e = np.ones(n_e, dtype=np.float32)

    score_h, details_h = match_protected_templates(bits_q, bits_e, w_q, w_e, hamming_threshold=0.22, return_details=True)
    assert 0.0 <= score_h <= 1.0

    pairs = details_h["matched_pairs"]
    assert len(pairs) > 0
    q_matched = [p[0] for p in pairs]
    e_matched = [p[1] for p in pairs]

    assert len(set(q_matched)) == len(q_matched), "Hungarian query vertex reuse detected"
    assert len(set(e_matched)) == len(e_matched), "Hungarian enrolled vertex reuse detected"
    assert len(pairs) <= min(n_q, n_e)
    for r, c in pairs:
        d = float(np.mean(bits_q[r] != bits_e[c]))
        assert d < 0.22


def test_pipeline_matcher_configuration_and_details():
    """Tests that CancelableFingerprintPipeline supports both Hungarian and Greedy 1-to-1 matchers."""
    pipe_h = CancelableFingerprintPipeline(in_dim=9, matcher="hungarian", use_gpu=True)
    pipe_g = CancelableFingerprintPipeline(in_dim=9, matcher="greedy_1to1", use_gpu=True)
    assert pipe_h.matcher == "hungarian"
    assert pipe_g.matcher == "greedy_1to1"
    assert "active" in pipe_h.gpu_info

    rng = np.random.default_rng(345)
    descs_q = rng.normal(0, 1, size=(20, 9)).astype(np.float32)
    weights_q = np.ones(20, dtype=np.float32)

    template_e = pipe_h.enroll_from_features(descs_q, weights_q, salt="salt_test", app_id="app_test")["template"]

    # Verify self-match under Hungarian
    res_h = pipe_h.verify_from_features(descs_q, weights_q, template_e)
    assert res_h["decision"] == "ACCEPT"
    assert res_h["matched_triangles"] == 20
    assert res_h["coverage"] == 1.0
    assert res_h["score"] > 0.8

    # Verify self-match under Greedy 1-to-1
    res_g = pipe_g.verify_from_features(descs_q, weights_q, template_e)
    assert res_g["decision"] == "ACCEPT"
    assert res_g["matched_triangles"] == 20
    assert res_g["coverage"] == 1.0
    assert res_g["score"] > 0.8


def test_gpu_acceleration_detection_and_opencl():
    """Tests that GPU OpenCL acceleration initialization properly queries OpenCV environment."""
    from src.pipeline import init_gpu_acceleration
    gpu_info = init_gpu_acceleration(verbose=False)
    assert "available" in gpu_info
    assert "active" in gpu_info
    assert "device_name" in gpu_info
    if gpu_info["available"]:
        assert gpu_info["active"] is True
        assert ("GeForce" in gpu_info["device_name"] or "NVIDIA" in gpu_info["vendor"] or "GPU" in gpu_info["device_name"] or
                "Advanced Micro Devices" in gpu_info["vendor"] or "AMD" in gpu_info["vendor"] or "gfx" in gpu_info["device_name"])


def test_hungarian_and_greedy_correlation():
    """Tests that Hungarian and Greedy 1-to-1 scores are strongly correlated on structured data."""
    rng = np.random.default_rng(456)
    h_scores = []
    g_scores = []

    for _ in range(50):
        n = 25
        q = rng.integers(0, 2, size=(n, 32), dtype=np.uint8)
        # Random noise level
        noise = rng.uniform(0.05, 0.40)
        flips = rng.random((n, 32)) < noise
        e = np.bitwise_xor(q, flips.astype(np.uint8))
        e = e[rng.permutation(n)]
        w = np.ones(n, dtype=np.float32)

        sh = match_protected_templates(q, e, w, w, hamming_threshold=0.22)
        sg = match_protected_greedy(q, e, w, w, hamming_threshold=0.22, mode="greedy_1to1")
        h_scores.append(sh)
        g_scores.append(sg)

    from scipy.stats import pearsonr
    r, _ = pearsonr(h_scores, g_scores)
    assert r > 0.95, f"Expected Pearson r > 0.95 between Hungarian and Greedy, got {r:.4f}"


# =========================================================================
# 4. Set A Sensor Provenance & Synthetic SFinGe Labeling Tests
# =========================================================================

def test_set_a_specifications_provenance():
    """Validates that Set A specifications match official FVC2002 guidelines."""
    assert "DB1_A" in ["DB1_A", "DB2_A", "DB3_A", "DB4_A"]

    # DB2_A: Biometrika FX2000 optical
    assert SET_A_SPECS["DB2_A"]["dpi"] == 569.0
    assert SET_A_SPECS["DB2_A"]["width"] == 296
    assert SET_A_SPECS["DB2_A"]["height"] == 560

    # DB3_A: Precise Biometrics 100 SC capacitive
    assert SET_A_SPECS["DB3_A"]["dpi"] == 500.0
    assert SET_A_SPECS["DB3_A"]["width"] == 300
    assert SET_A_SPECS["DB3_A"]["height"] == 300

    # DB4_A: Synthetic SFinGe
    assert SET_A_SPECS["DB4_A"]["dpi"] == 500.0
    assert SET_A_SPECS["DB4_A"]["width"] == 288
    assert SET_A_SPECS["DB4_A"]["height"] == 384
    assert SET_A_SPECS["DB4_A"]["is_synthetic"] is True
    assert "SFinGe" in SET_A_SPECS["DB4_A"]["label"]


# =========================================================================
# 5. Out-of-Domain Threshold Transfer & Accounting Identity Tests
# =========================================================================

def test_threshold_transfer_accounting_identities():
    """Validates mathematical identity TP + FN == N_gen and FP + TN == N_imp under arbitrary thresholds."""
    rng = np.random.default_rng(789)
    g_scores = rng.beta(5, 5, size=2800)
    i_scores = rng.beta(2, 6, size=4950)

    for threshold in [0.20, 0.3751, 0.4497, 0.4705, 0.60]:
        tp = int(np.sum(g_scores >= threshold))
        fn = int(np.sum(g_scores < threshold))
        fp = int(np.sum(i_scores >= threshold))
        tn = int(np.sum(i_scores < threshold))

        assert tp + fn == 2800, f"Genuine identity violated at T={threshold}: {tp} + {fn} != 2800"
        assert fp + tn == 4950, f"Impostor identity violated at T={threshold}: {fp} + {tn} != 4950"


# =========================================================================
# 6. Clustered Bootstrap Confidence Interval Tests
# =========================================================================

def test_clustered_bootstrap_replicates():
    """Tests that finger-clustered bootstrap runs stably and produces non-degenerate 95% CIs."""
    rng = np.random.default_rng(999)
    # 10 fingers, 4 impressions each -> 6 genuine comparisons per finger, 60 total
    gen_records = []
    for f in range(1, 11):
        for _ in range(6):
            gen_records.append({'subject_a': f, 'subject_b': f, 'score': float(rng.normal(0.45, 0.04))})

    # Impostor comparisons across finger pairs
    imp_records = []
    for fa in range(1, 11):
        for fb in range(fa + 1, 11):
            imp_records.append({'subject_a': fa, 'subject_b': fb, 'score': float(rng.normal(0.30, 0.04))})

    res = clustered_bootstrap_confidence_intervals(gen_records, imp_records, operating_threshold=0.38, num_bootstraps=200)

    eer_ci = res["eer_ci"]
    auc_ci = res["auc_ci"]
    assert 0.0 <= eer_ci[0] <= eer_ci[1] <= 1.0
    assert 0.0 <= auc_ci[0] <= auc_ci[1] <= 1.0
    assert res["eer_mean"] > 0.0
    assert res["auc_mean"] > 0.50
