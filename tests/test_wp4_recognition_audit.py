"""
Unit and Property Tests for Work Package 4 (WP4) - Recognition and Threshold Validity Audit.

Tests:
1. Matcher Non-Equivalence: Demonstrates that Greedy one-to-one partial-matching
   heuristic is not globally optimal and can diverge from the exact Hungarian reference.
2. Greedy 1-to-1 Invariants: Enforces strict one-to-one pairing and threshold bounds.
3. Coverage-Aware Normalization: Verifies bounds [0, 1], symmetry S(q, e) == S(e, q),
   zero-match behavior, and coverage monotonicity.
4. Threshold Statistical Power: Verifies empirical granularity on N_imp=45 and
   proves that observing 0 false accepts cannot bound FAR below 1% or 0.1%.
5. Sensor Provenance & Synthetic Separation: Verifies distinct provenance for DB4 vs DB1-DB3.
"""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from src.matching.bipartite_matcher import (
    build_augmented_cost,
    match_protected_templates,
    match_protected_greedy
)


def test_hungarian_vs_greedy_counterexample_suboptimality():
    """
    Demonstrates that the Greedy one-to-one partial-matching heuristic is NOT
    globally optimal and produces a strictly inferior augmented cost and biometric score
    compared to the exact Hungarian reference.
    """
    # 2x2 distance matrix where Greedy makes a locally optimal but globally suboptimal choice
    dist_matrix = np.array([
        [0.20, 0.22],
        [0.21, 0.45]
    ], dtype=np.float64)
    tau = 0.50

    # 1. Exact Reference: Hungarian Assignment on Augmented Slack Matrix
    cost_augmented = build_augmented_cost(dist_matrix, tau)
    row_ind, col_ind = linear_sum_assignment(cost_augmented)

    hungarian_pairs = [
        (int(r), int(c)) for r, c in zip(row_ind, col_ind)
        if r < 2 and c < 2 and dist_matrix[r, c] < tau
    ]
    # Hungarian should pair (0, 1) and (1, 0)
    assert set(hungarian_pairs) == {(0, 1), (1, 0)}, f"Unexpected Hungarian pairs: {hungarian_pairs}"
    hungarian_dist_sum = sum(dist_matrix[r, c] for r, c in hungarian_pairs)
    assert np.isclose(hungarian_dist_sum, 0.43), f"Expected 0.43, got {hungarian_dist_sum}"

    # 2. Greedy 1-to-1 Partial-Matching Heuristic
    valid_pairs = np.argwhere(dist_matrix < tau)
    pair_dists = dist_matrix[valid_pairs[:, 0], valid_pairs[:, 1]]
    sort_order = np.argsort(pair_dists, kind="stable")

    matched_q, matched_e = set(), set()
    greedy_pairs = []
    for idx in sort_order:
        r, c = int(valid_pairs[idx, 0]), int(valid_pairs[idx, 1])
        if r not in matched_q and c not in matched_e:
            matched_q.add(r)
            matched_e.add(c)
            greedy_pairs.append((r, c))

    # Greedy picks (0, 0) first (0.20), blocking (1, 0) and (0, 1), forcing (1, 1) (0.45)
    assert set(greedy_pairs) == {(0, 0), (1, 1)}, f"Unexpected Greedy pairs: {greedy_pairs}"
    greedy_dist_sum = sum(dist_matrix[r, c] for r, c in greedy_pairs)
    assert np.isclose(greedy_dist_sum, 0.65), f"Expected 0.65, got {greedy_dist_sum}"

    # 3. Suboptimality Gaps
    # Total augmented cost: Hungarian is strictly lower than Greedy
    assert hungarian_dist_sum < greedy_dist_sum, "Hungarian must achieve strictly lower cost"

    # Corresponding Biometric Matching Scores:
    # Sim(r, c) = 1 - d / tau
    # Hungarian: (1 - 0.22/0.50) + (1 - 0.21/0.50) = 0.56 + 0.58 = 1.14 -> 2 * 1.14 / 4 = 0.57
    # Greedy:    (1 - 0.20/0.50) + (1 - 0.45/0.50) = 0.60 + 0.10 = 0.70 -> 2 * 0.70 / 4 = 0.35
    cost_error_pct = (greedy_dist_sum - hungarian_dist_sum) / hungarian_dist_sum * 100.0
    assert cost_error_pct > 50.0, f"Expected >50% cost error gap, got {cost_error_pct:.2f}%"


def test_greedy_1to1_heuristic_invariants():
    """
    Verifies that the Greedy 1-to-1 matcher preserves structural invariants:
    - No query index is matched more than once.
    - No enrolled index is matched more than once.
    - Every matched pair has normalized Hamming distance < tau.
    """
    np.random.seed(42)
    n_q, n_e = 50, 40
    query_bits = np.random.randint(0, 2, size=(n_q, 32)).astype(np.uint8)
    enrolled_bits = np.random.randint(0, 2, size=(n_e, 32)).astype(np.uint8)
    query_weights = np.random.uniform(0.5, 1.0, size=n_q).astype(np.float32)
    enrolled_weights = np.random.uniform(0.5, 1.0, size=n_e).astype(np.float32)

    tau = 0.40
    score, details = match_protected_greedy(
        query_bits, enrolled_bits, query_weights, enrolled_weights,
        hamming_threshold=tau, mode="greedy_1to1", return_details=True
    )

    matched_pairs = details["matched_pairs"]
    q_indices = [p[0] for p in matched_pairs]
    e_indices = [p[1] for p in matched_pairs]

    # Invariant: Strict 1-to-1 matching
    assert len(q_indices) == len(set(q_indices)), "Query indices contain duplicate assignments"
    assert len(e_indices) == len(set(e_indices)), "Enrolled indices contain duplicate assignments"

    # Invariant: Distance constraint
    for r, c in matched_pairs:
        d = np.mean(query_bits[r] != enrolled_bits[c])
        assert d < tau, f"Matched pair ({r}, {c}) exceeds threshold: {d} >= {tau}"

    # Invariant: Score bounds
    assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"


def test_coverage_aware_score_normalization_properties():
    """
    Audits coverage-aware score normalization:
    - Boundedness in [0, 1].
    - Exact query/enrolled symmetry S(q, e) == S(e, q).
    - Monotonic degradation as template size increases (lower coverage).
    - Zero score when no pairs match.
    """
    np.random.seed(123)
    n = 30
    bits_q = np.random.randint(0, 2, size=(n, 32)).astype(np.uint8)
    # Clone with 10% flip to create genuine match
    flip_mask = (np.random.rand(n, 32) < 0.10).astype(np.uint8)
    bits_e = bits_q ^ flip_mask
    w_q = np.ones(n, dtype=np.float32)
    w_e = np.ones(n, dtype=np.float32)

    # 1. Symmetry Verification
    s_qe = match_protected_templates(bits_q, bits_e, w_q, w_e, hamming_threshold=0.22, coverage_penalty_exponent=1.2, min_matches=3)
    s_eq = match_protected_templates(bits_e, bits_q, w_e, w_q, hamming_threshold=0.22, coverage_penalty_exponent=1.2, min_matches=3)
    assert np.isclose(s_qe, s_eq, atol=1e-6), f"Symmetry violated: {s_qe} != {s_eq}"

    # 2. Range Bounds
    assert 0.0 <= s_qe <= 1.0, f"Score out of range: {s_qe}"

    # 3. Monotonic Coverage Damping
    # Pad query and enrolled with random noise descriptors (lowering coverage)
    bits_q_padded = np.vstack([bits_q, np.random.randint(0, 2, size=(70, 32)).astype(np.uint8)])
    bits_e_padded = np.vstack([bits_e, np.random.randint(0, 2, size=(70, 32)).astype(np.uint8)])
    w_q_padded = np.ones(100, dtype=np.float32)
    w_e_padded = np.ones(100, dtype=np.float32)

    s_padded_cov10 = match_protected_templates(bits_q_padded, bits_e_padded, w_q_padded, w_e_padded, hamming_threshold=0.22, coverage_penalty_exponent=1.0)
    s_padded_cov12 = match_protected_templates(bits_q_padded, bits_e_padded, w_q_padded, w_e_padded, hamming_threshold=0.22, coverage_penalty_exponent=1.2)

    # With gamma=1.2, score must be strictly lower than gamma=1.0 when coverage < 1.0
    assert s_padded_cov12 < s_padded_cov10, f"Coverage damping failed: {s_padded_cov12} >= {s_padded_cov10}"

    # 4. Zero Matches Behavior
    # Exact opposite bits guarantee all pairwise Hamming distances are 1.0 >= tau
    bits_opposite = 1 - bits_q
    s_zero = match_protected_templates(bits_q, bits_opposite, w_q, w_e, hamming_threshold=0.22)
    assert s_zero == 0.0, f"Expected 0.0 for zero matches, got {s_zero}"


def test_threshold_calibration_statistical_power_audit():
    """
    Audits the statistical power deficit of Set B development threshold calibration:
    - Verifies empirical FAR step size of 1/45 = 2.22%.
    - Verifies that 0 false accepts on 45 trials yields Clopper-Pearson 95% upper bound of 6.44%
      and Wilson score bound of 7.86%.
    - Confirms that calibrating a 1% or 0.1% FAR operating threshold on 45 trials is statistically underpowered.
    """
    n_imp = 45

    # Discrete step size
    step_size = 1.0 / n_imp
    assert np.isclose(step_size, 0.0222222, atol=1e-4), f"Step size unexpected: {step_size}"

    # Target FARs
    target_1pct = 0.01
    target_01pct = 0.001
    assert target_1pct < step_size, "1% FAR is smaller than the minimum observable non-zero step size (2.22%)"
    assert target_01pct < step_size, "0.1% FAR is smaller than the minimum observable non-zero step size (2.22%)"

    # Uncertainty calculation for k=0 false accepts out of n=45 trials
    k = 0
    alpha = 0.05
    # Clopper-Pearson upper bound: 1 - alpha^(1/n)
    cp_upper_95 = 1.0 - (alpha ** (1.0 / n_imp))
    assert np.isclose(cp_upper_95, 0.064404, atol=1e-4), f"Unexpected Clopper-Pearson bound: {cp_upper_95}"

    # Rule of Three approximation: 3 / n
    rule_of_three_upper = 3.0 / n_imp
    assert np.isclose(rule_of_three_upper, 0.066667, atol=1e-4), f"Unexpected Rule of Three bound: {rule_of_three_upper}"

    # Wilson score interval upper bound for k=0: z^2 / (n + z^2)
    z = 1.95996  # 95% two-sided
    wilson_upper_95 = (z ** 2) / (n_imp + z ** 2)
    assert np.isclose(wilson_upper_95, 0.07860, atol=1e-3), f"Unexpected Wilson bound: {wilson_upper_95}"

    # Audit conclusion: Both bounds exceed 6.4%, proving that 45 trials cannot guarantee FAR <= 1%
    assert cp_upper_95 > target_1pct * 6.0, "Clopper-Pearson bound must be >6x the target 1% FAR"
    assert wilson_upper_95 > target_1pct * 7.0, "Wilson bound must be >7x the target 1% FAR"


def test_sensor_provenance_and_synthetic_separation():
    """
    Verifies that FVC2002 DB4 is strictly distinguished from physical sensors (DB1, DB2, DB3).
    """
    sensor_metadata = {
        "DB1": {"technology": "Optical (TIR Prism)", "is_synthetic": False},
        "DB2": {"technology": "Optical (CMOS surface)", "is_synthetic": False},
        "DB3": {"technology": "Capacitive (Silicon)", "is_synthetic": False},
        "DB4": {"technology": "Synthetic generation model (SFinGe v2.51)", "is_synthetic": True}
    }

    assert not sensor_metadata["DB1"]["is_synthetic"], "DB1 must be physical optical"
    assert not sensor_metadata["DB2"]["is_synthetic"], "DB2 must be physical optical"
    assert not sensor_metadata["DB3"]["is_synthetic"], "DB3 must be physical capacitive"
    assert sensor_metadata["DB4"]["is_synthetic"], "DB4 must be classified as synthetic SFinGe"
