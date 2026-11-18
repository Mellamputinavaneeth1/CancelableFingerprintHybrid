"""
Unit and Property Tests for P0 Remediation & Hardened Mathematical Invariants.

Covers:
1. P0.1 Augmented Hungarian matrix slack penalties, dummy costs (tau/2), and strict boundary (< tau).
2. P0.2 Canonical Gómez-Barrero global unlinkability integral D_sys = integral p(s|H_m) D_omega(s) ds.
3. P0.3 & P0.4 SOCOFing 590 clean trials, 18 genuine per fold, and severity accounting.
4. P0.5 Affine coordinate transformation, standardized Chebyshev center LP, and bounded convex relaxation.
5. P0.6 Quantizer 1D/2D shapes, signed zeros (+0.0 vs -0.0), and canonical zero tie-breaking rule.
"""

import os
import csv
import pytest
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.matching.augmented_cost import build_augmented_cost
from src.matching.bipartite_matcher import match_protected_templates, match_unprotected_descriptors
from src.protection.quantizer import quantize_descriptors
from src.protection.inversion_defense import (
    physical_to_standardized,
    standardized_to_physical,
    get_standardized_relaxation_constraints,
    get_standardized_triangle_inequality,
    solve_chebyshev_center_standardized
)
from src.security.unlinkability import compute_system_unlinkability, compute_local_unlinkability
from src.evaluation.metrics import compute_roc_metrics


class TestP01AugmentedHungarianCost:
    """P0.1: Reconcile augmented cost matrix blocks, dummy penalty tau/2, and boundary semantics."""

    def test_augmented_cost_structure_and_penalties(self):
        tau = 0.40
        big = 1e6
        # 2 query, 3 enrolled
        # D_00 = 0.20 (< tau, eligible)
        # D_01 = 0.40 (== tau, ineligible under strict inequality)
        # D_02 = 0.50 (> tau, ineligible)
        # D_10 = 0.35 (< tau, eligible)
        # D_11 = 0.10 (< tau, eligible)
        # D_12 = 0.45 (> tau, ineligible)
        dist = np.array([
            [0.20, 0.40, 0.50],
            [0.35, 0.10, 0.45]
        ], dtype=np.float64)
        nq, ne = dist.shape

        c_aug = build_augmented_cost(dist, tau=tau, big=big)
        assert c_aug.shape == (nq + ne, nq + ne)  # (5, 5)

        # Real block [0:nq, 0:ne]
        real_block = c_aug[:nq, :ne]
        assert real_block[0, 0] == pytest.approx(0.20)
        assert real_block[0, 1] == pytest.approx(big)  # strict inequality: 0.40 is not < 0.40
        assert real_block[0, 2] == pytest.approx(big)
        assert real_block[1, 0] == pytest.approx(0.35)
        assert real_block[1, 1] == pytest.approx(0.10)
        assert real_block[1, 2] == pytest.approx(big)

        # Dummy block top-right [0:nq, ne:ne+nq]
        uq_block = c_aug[:nq, ne:]
        assert uq_block.shape == (nq, nq)
        for i in range(nq):
            for j in range(nq):
                if i == j:
                    assert uq_block[i, j] == pytest.approx(tau / 2.0)
                else:
                    assert uq_block[i, j] == pytest.approx(big)

        # Dummy block bottom-left [nq:nq+ne, 0:ne]
        ue_block = c_aug[nq:, :ne]
        assert ue_block.shape == (ne, ne)
        for i in range(ne):
            for j in range(ne):
                if i == j:
                    assert ue_block[i, j] == pytest.approx(tau / 2.0)
                else:
                    assert ue_block[i, j] == pytest.approx(big)

        # Bottom-right dummy-to-dummy block [nq:, ne:]
        dd_block = c_aug[nq:, ne:]
        assert dd_block.shape == (ne, nq)
        assert np.all(dd_block == 0.0)

    def test_objective_slack_reconciliation(self):
        """Verifies that Hungarian assignment cost on C_aug equals (Nq + Ne)*tau/2 - sum(tau - D_ij)."""
        tau = 0.50
        dist = np.array([
            [0.15, 0.60],
            [0.70, 0.25]
        ], dtype=np.float64)
        nq, ne = dist.shape

        c_aug = build_augmented_cost(dist, tau=tau)
        row_ind, col_ind = linear_sum_assignment(c_aug)
        hungarian_cost = float(c_aug[row_ind, col_ind].sum())

        # Matched real pairs: (0, 0) with D=0.15, (1, 1) with D=0.25
        # Gain for pair (0, 0): tau - 0.15 = 0.35
        # Gain for pair (1, 1): tau - 0.25 = 0.25
        # J(pi) = (2 + 2)*(0.50/2) - (0.35 + 0.25) = 1.0 - 0.60 = 0.40
        expected_cost = (nq + ne) * (tau / 2.0) - ((tau - 0.15) + (tau - 0.25))
        assert hungarian_cost == pytest.approx(expected_cost, abs=1e-6)

    def test_empty_or_zero_cost(self):
        tau = 0.50
        # 1. 0 query, 5 enrolled -> all 5 enrolled unmatched, cost = 5 * (tau/2) = 1.25
        c_q0 = build_augmented_cost(np.empty((0, 5)), tau=tau)
        assert c_q0.shape == (5, 5)
        r, col = linear_sum_assignment(c_q0)
        assert c_q0[r, col].sum() == pytest.approx(5 * (tau / 2.0))

        # 2. 5 query, 0 enrolled -> all 5 queries unmatched, cost = 5 * (tau/2) = 1.25
        c_e0 = build_augmented_cost(np.empty((5, 0)), tau=tau)
        assert c_e0.shape == (5, 5)
        r, col = linear_sum_assignment(c_e0)
        assert c_e0[r, col].sum() == pytest.approx(5 * (tau / 2.0))

        # 3. 0 query, 0 enrolled -> empty shape (0, 0), cost = 0.0
        c_none = build_augmented_cost(np.empty((0, 0)), tau=tau)
        assert c_none.shape == (0, 0)

        # 4. Dimension validation
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            build_augmented_cost(np.array([0.1, 0.2]), tau=tau)


class TestP02GomezBarreroUnlinkability:
    """P0.2: Canonical global unlinkability D_sys = integral p(s|H_m) D_omega(s) ds."""

    def test_ideal_unlinkability_identical_distributions(self):
        rng = np.random.default_rng(42)
        # Identical mated and non-mated scores
        scores = rng.beta(5.0, 5.0, size=500)
        d_sys, eval_points, d_omega, lr = compute_system_unlinkability(scores, scores, num_bins=100)
        # When distributions are identical, LR(s) = 1.0 for all s.
        # Under omega = 1.0, LR <= omega, so D_omega(s) = 0.0 everywhere.
        assert d_sys == pytest.approx(0.0, abs=0.02)
        assert np.max(d_omega) == pytest.approx(0.0, abs=0.02)

    def test_completely_separated_distributions(self):
        # Mated scores near 0.9, non-mated scores near 0.1
        mated = np.random.default_rng(101).normal(0.9, 0.02, size=500)
        non_mated = np.random.default_rng(102).normal(0.1, 0.02, size=500)
        d_sys, eval_points, d_omega, lr = compute_system_unlinkability(mated, non_mated, num_bins=100)
        # Fully separated distributions yield D_sys close to 1.0
        assert d_sys == pytest.approx(1.0, abs=0.05)

    def test_compute_local_unlinkability_reexport(self):
        m = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        nm = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        eval_pts, d_omega, lr = compute_local_unlinkability(m, nm, num_bins=50)
        assert len(eval_pts) == 50
        assert len(d_omega) == 50
        assert len(lr) == 50


class TestP05AffinePreImageLP:
    """P0.5: Explicit affine coordinate transformation between raw physical and standardized LP space."""

    def test_affine_forward_inverse_roundtrip(self):
        rng = np.random.default_rng(77)
        # 10 random 9-D physical descriptors within valid physical bounds
        f_raw = np.zeros((10, 9), dtype=np.float64)
        f_raw[:, 0] = rng.uniform(0.5, 0.9, size=10)   # r1
        f_raw[:, 1] = rng.uniform(0.2, 0.5, size=10)   # r2 <= r1
        f_raw[:, 2] = rng.uniform(4.0, 15.0, size=10)  # P_mm
        f_raw[:, 3:] = rng.uniform(0.0, 1.0, size=(10, 6))

        z_tilde = physical_to_standardized(f_raw)
        f_recovered = standardized_to_physical(z_tilde)
        np.testing.assert_allclose(f_raw, f_recovered, atol=1e-6)

    def test_standardized_relaxation_constraints_and_ratio(self):
        A_ratio, b_ratio, z_min, z_max = get_standardized_relaxation_constraints(d=9)
        assert A_ratio.shape == (1, 9)
        assert b_ratio.shape == (1,)
        assert len(z_min) == 9
        assert len(z_max) == 9

        # A point with r1 = 0.8, r2 = 0.4 has r2 <= r1 (valid).
        f_valid = np.array([0.8, 0.4, 7.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        z_valid = physical_to_standardized(f_valid)
        assert float((A_ratio @ z_valid).item()) <= float(b_ratio.item()) + 1e-6

        # A point with r1 = 0.3, r2 = 0.9 violates r2 <= r1 (invalid).
        f_invalid = np.array([0.3, 0.9, 7.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        z_invalid = physical_to_standardized(f_invalid)
        assert float((A_ratio @ z_invalid).item()) > float(b_ratio.item())

    def test_standardized_triangle_inequality_constraint(self):
        """Verifies standardized mapping of the linear triangle inequality r1 + r2 >= 1.0."""
        A_tri, b_tri = get_standardized_triangle_inequality(d=9, eps=0.0)
        assert A_tri.shape == (1, 9)
        assert b_tri.shape == (1,)

        # Valid triangle: r1 = 0.78, r2 = 0.50 (r1 + r2 = 1.28 >= 1.0)
        f_valid = np.array([0.78, 0.50, 7.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        z_valid = physical_to_standardized(f_valid)
        assert float((A_tri @ z_valid).item()) <= float(b_tri.item()) + 1e-6

        # Invalid degenerate geometry: r1 = 0.45, r2 = 0.40 (r1 + r2 = 0.85 < 1.0)
        f_invalid = np.array([0.45, 0.40, 7.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        z_invalid = physical_to_standardized(f_invalid)
        assert float((A_tri @ z_invalid).item()) > float(b_tri.item())

    def test_standardized_chebyshev_center_solver(self):
        rng = np.random.default_rng(88)
        # Generate random orthonormal matrix W in R^{32 x 9}
        H = rng.normal(0.0, 1.0, size=(32, 9))
        Q, _ = np.linalg.qr(H)
        W = Q.astype(np.float64)

        # Valid physical descriptor
        f_true = np.array([0.75, 0.45, 8.0, 0.3, 0.6, 0.4, 0.2, 0.8, 0.5], dtype=np.float64)
        z_true = physical_to_standardized(f_true)
        # Quantize to target bits b in {0, 1}^32
        target_bits = (W @ z_true >= 0.0).astype(np.uint8)

        feas, cand_desc, radius = solve_chebyshev_center_standardized(target_bits, W, delta=0.005)
        assert feas is True
        assert radius >= 0.0
        assert len(cand_desc) == 9
        # Check that candidate satisfies physical box bounds
        assert 0.1 <= cand_desc[0] <= 1.0
        assert 0.1 <= cand_desc[1] <= 1.0
        assert 2.0 <= cand_desc[2] <= 30.0
        assert np.all(cand_desc[3:9] >= 0.0) and np.all(cand_desc[3:9] <= 1.0)


class TestP06QuantizerHardening:
    """P0.6: Explicit quantizer tie rule at exact zero and signed zero sanitization."""

    def test_canonical_tie_breaking_at_zero(self):
        # Tie rule: b_j = 1 if projected >= 0.0 else 0
        vals = np.array([1.5, 0.0, -0.0, -1e-9, 1e-9, -2.5], dtype=np.float64)
        bits = quantize_descriptors(vals)
        assert bits.shape == (6,)
        # 1.5 -> 1
        # +0.0 -> 1
        # -0.0 -> 1
        # -1e-9 -> 0
        # 1e-9 -> 1
        # -2.5 -> 0
        expected = np.array([1, 1, 1, 0, 1, 0], dtype=np.uint8)
        np.testing.assert_array_equal(bits, expected)

    def test_quantizer_1d_vector_support(self):
        vec_1d = np.array([0.5, -0.2, 0.0, -0.8], dtype=np.float32)
        bits_1d = quantize_descriptors(vec_1d)
        assert bits_1d.ndim == 1
        assert bits_1d.shape == (4,)
        np.testing.assert_array_equal(bits_1d, np.array([1, 0, 1, 0], dtype=np.uint8))

    def test_quantizer_2d_batch_support(self):
        mat_2d = np.array([
            [0.5, -0.2, 0.0],
            [-1.0, 2.0, -0.0]
        ], dtype=np.float32)
        bits_2d = quantize_descriptors(mat_2d)
        assert bits_2d.ndim == 2
        assert bits_2d.shape == (2, 3)
        expected = np.array([
            [1, 0, 1],
            [0, 1, 1]
        ], dtype=np.uint8)
        np.testing.assert_array_equal(bits_2d, expected)

    def test_quantizer_empty_array(self):
        empty = np.empty((0, 32), dtype=np.float32)
        bits = quantize_descriptors(empty)
        assert bits.shape == (0, 32)
        assert bits.dtype == np.uint8

    def test_quantizer_nan_and_inf_rejection(self):
        """Explicitly rejects NaN and positive/negative infinity with ValueError."""
        with pytest.raises(ValueError, match="Cannot quantize non-finite"):
            quantize_descriptors(np.array([1.0, np.nan, -0.5]))

        with pytest.raises(ValueError, match="Cannot quantize non-finite"):
            quantize_descriptors(np.array([1.0, np.inf, -0.5]))

        with pytest.raises(ValueError, match="Cannot quantize non-finite"):
            quantize_descriptors(np.array([1.0, -np.inf, -0.5]))


class TestP03AndP04SOCOFingCleanTrials:
    """P0.3 & P0.4: SOCOFing 590-trial clean ledger and per-severity accounting."""

    def test_canonical_590_trial_ledger_invariants(self):
        ledger_path = os.path.join(
            os.path.dirname(__file__), "..",
            "experiments", "results", "socofing_full_canonical", "results", "canonical_clean_trials_590.csv"
        )
        if not os.path.exists(ledger_path):
            pytest.skip("canonical_clean_trials_590.csv not found")

        with open(ledger_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 590, f"Expected exactly 590 rows, got {len(rows)}"

        # Strict exclusion of same-image self-matches
        for r in rows:
            assert r["query_relative_path"] != r["gallery_relative_path"], "Exact self-match found in clean trials!"

        gen_rows = [r for r in rows if r["is_genuine"] == "1"]
        imp_rows = [r for r in rows if r["is_genuine"] == "0"]
        assert len(gen_rows) == 90, f"Expected 90 genuine, got {len(gen_rows)}"
        assert len(imp_rows) == 500, f"Expected 500 impostors, got {len(imp_rows)}"

        # Verify exactly 18 genuine trials per outer fold
        from collections import Counter
        fold_gen_counts = Counter(r["outer_fold"] for r in gen_rows)
        fold_imp_counts = Counter(r["outer_fold"] for r in imp_rows)
        assert len(fold_gen_counts) == 5
        for fold_name in [f"fold_{i:02d}" for i in range(1, 6)]:
            assert fold_gen_counts[fold_name] == 18, f"{fold_name} genuine count != 18"
            assert fold_imp_counts[fold_name] == 100, f"{fold_name} impostor count != 100"

    def test_severity_accounting_disclosed(self):
        ledger_path = os.path.join(
            os.path.dirname(__file__), "..",
            "experiments", "results", "socofing_full_canonical", "results", "canonical_clean_trials_590.csv"
        )
        if not os.path.exists(ledger_path):
            pytest.skip("canonical_clean_trials_590.csv not found")

        with open(ledger_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        from collections import Counter
        sev_gen = Counter(r["query_severity"] for r in rows if r["is_genuine"] == "1")
        assert sev_gen["Easy"] == 30
        assert sev_gen["Medium"] == 30
        assert sev_gen["Hard"] == 30
        assert sum(sev_gen.values()) == 90

        # Impostor trials paired with altered probes
        sev_imp = Counter(r["query_severity"] for r in rows if r["is_genuine"] == "0")
        assert sev_imp["Easy"] == 150
        assert sev_imp["Medium"] == 150
        assert sev_imp["Hard"] == 150
        assert sev_imp["Real"] == 50
        assert sum(sev_imp.values()) == 500
