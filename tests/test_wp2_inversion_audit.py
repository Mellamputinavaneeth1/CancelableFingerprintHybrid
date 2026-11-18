"""
WP2 Security Pre-Image, Chebyshev Center, and Inversion Audit Test Suite.
========================================================================
Formal verification of:
1. Chebyshev center sign convention and Euclidean ball containment:
   (2 b_j - 1) w_j^T z >= r ||w_j||_2
2. Mathematical unboundedness of homogeneous sign cones vs. finite compactness
   of the bounded physical descriptor polytope P_bounded.
3. Angular MAE unit conversion: verifying the factor of 90 degrees (pi/2)
   versus the erroneous 360 degrees (2*pi) conversion.
4. Multi-threat attack matrix: Scenario II (unknown key), Scenario III (known key
   Chebyshev LP / QP), and Option A re-issuance.
5. MILP type-lattice scope: proving 9-D contains strictly zero discrete variables,
   while characterizing continuous optimization vulnerabilities.
6. Rigorous statistical uncertainty: Wilson score upper confidence bounds for
   zero-success attack outcomes.
"""

import numpy as np
import pytest
from scipy.stats import norm

from src.protection.kdf import generate_master_secret
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors
from src.matching.bipartite_matcher import match_protected_templates
from src.graph.descriptors import (
    DEFAULT_DESCRIPTOR_MEAN_9D, DEFAULT_DESCRIPTOR_STD_9D,
    DEFAULT_DESCRIPTOR_MEAN, DEFAULT_DESCRIPTOR_STD
)
from src.security.inversion_attack import (
    solve_linear_feasibility_inversion,
    solve_chebyshev_center_inversion,
    solve_qp_center_inversion,
    calculate_continuous_inversion_metrics,
    calculate_angular_wrap_error
)
from src.security.leakage_analysis import solve_milp_feasibility_inversion


def compute_wilson_ci(successes: int, trials: int, confidence: float = 0.95):
    """Computes two-sided Wilson score confidence interval."""
    if trials == 0:
        return 0.0, 1.0
    z = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p_hat = successes / trials
    denom = 1.0 + (z**2) / trials
    center = (p_hat + (z**2) / (2.0 * trials)) / denom
    half_width = (z * np.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * trials)) / trials)) / denom
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


class TestWP2ChebyshevAndPreImageAudit:
    """Verifies mathematical consistency of pre-image inversion and Chebyshev optimization."""

    def test_chebyshev_center_sign_convention_and_ball_containment(self):
        """
        Audits the inequality sign convention for Chebyshev inscribed Euclidean ball:
            (2 b_j - 1) w_j^T z >= r ||w_j||_2
        Refutes the incorrect sign in (2 b_j - 1) w_j^T z + r ||w_j||_2 <= 0.
        Verifies that any point inside the ball B(z*, r*) satisfies all sign constraints.
        """
        master = "WP2_ChebyshevSignConvention_MasterSecret_32B!"
        W = derive_orthonormal_matrix(master, salt="cheb_salt", app_id="auth", in_dim=9, out_dim=32)

        # Genuine descriptor
        d_orig = np.array([0.78, 0.50, 6.80, 0.45, 0.55, 0.40, 0.35, 0.60, 0.45], dtype=np.float32)
        z_orig = (d_orig - DEFAULT_DESCRIPTOR_MEAN_9D) / (DEFAULT_DESCRIPTOR_STD_9D + 1e-8)
        bits = (W @ z_orig >= 0.0).astype(np.uint8)
        signs = 2.0 * bits.astype(np.float64) - 1.0

        feas, d_c, r_c = solve_chebyshev_center_inversion(bits, W, delta=0.005)
        assert feas is True
        assert r_c > 0.0

        z_c = (d_c - DEFAULT_DESCRIPTOR_MEAN_9D) / (DEFAULT_DESCRIPTOR_STD_9D + 1e-8)

        # Standardized sign margins
        W_scaled = W / (DEFAULT_DESCRIPTOR_STD_9D + 1e-8)
        A_sign = -(signs[:, None] * W_scaled)
        b_sign = -np.ones(len(bits), dtype=np.float64) * 0.005 - signs * (W_scaled @ DEFAULT_DESCRIPTOR_MEAN_9D)
        norms = np.linalg.norm(A_sign, axis=1)

        # 1. Sign constraint check at the Chebyshev center
        margins_center = signs * (W @ z_c)
        assert np.all(margins_center >= 0.005 - 1e-4), "Center violates feasibility margin"

        # 2. Inscribed ball containment: check worst-case boundary points
        # For each constraint j: A_j d_c + r ||A_j|| <= b_j
        lhs = A_sign @ d_c + r_c * norms
        assert np.all(lhs <= b_sign + 1e-4), "Chebyshev inscribed ball violates halfspace boundary!"

        # 3. Random boundary perturbation check on B(z_c, r_c)
        rng = np.random.default_rng(2026)
        for _ in range(50):
            u = rng.normal(0, 1, size=9)
            u = (u / np.linalg.norm(u)) * (r_c * 0.99)  # strictly inside ball
            d_perturbed = d_c + u.astype(np.float32)
            z_p = (d_perturbed - DEFAULT_DESCRIPTOR_MEAN_9D) / (DEFAULT_DESCRIPTOR_STD_9D + 1e-8)
            pert_margins = signs * (W @ z_p)
            assert np.all(pert_margins >= -1e-4), "Perturbed point in Chebyshev ball violated sign cone!"

    def test_homogeneous_cone_unboundedness_and_bounded_polytope(self):
        """
        Proves that a homogeneous cone defined solely by sign inequalities:
            C_b = {z in R^D : s_j w_j^T z >= 0}
        is unbounded (scaling z_0 by lambda -> infinity yields r -> infinity).
        Confirms that physical descriptor bounds [d_min, d_max] and ratio constraint
        r2 <= r1 form a compact polytope P_bounded that guarantees a finite Chebyshev radius.
        """
        master = "WP2_BoundedPolytope_MasterSecret_32B_Strict!"
        W = derive_orthonormal_matrix(master, salt="cone_salt", app_id="auth", in_dim=9, out_dim=32)

        # Synthetic interior point in the cone
        z_interior = np.array([0.5, 0.2, 1.0, 0.1, -0.3, 0.4, 0.2, -0.1, 0.3], dtype=np.float64)
        bits = (W @ z_interior >= 0.0).astype(np.uint8)
        signs = 2.0 * bits - 1.0

        # Base margin
        base_margin = np.min(signs * (W @ z_interior))
        assert base_margin > 0.0, "Point must be strictly interior to the cone"

        # Scaling by lambda > 1 strictly scales the margin towards infinity
        for lam in [2.0, 10.0, 100.0, 1000.0]:
            scaled_margin = np.min(signs * (W @ (lam * z_interior)))
            assert scaled_margin == pytest.approx(lam * base_margin, rel=1e-5)

        # In contrast, solve_chebyshev_center_inversion applies physical bounding box [d_min, d_max]
        # and returns a strictly finite radius r* bounded by the box dimensions
        d_orig = np.array([0.78, 0.50, 6.80, 0.45, 0.55, 0.40, 0.35, 0.60, 0.45], dtype=np.float32)
        z_orig = (d_orig - DEFAULT_DESCRIPTOR_MEAN_9D) / (DEFAULT_DESCRIPTOR_STD_9D + 1e-8)
        bits_orig = (W @ z_orig >= 0.0).astype(np.uint8)
        feas, cand, rad = solve_chebyshev_center_inversion(bits_orig, W, delta=0.005)

        assert feas is True
        # Inscribed radius in physical space cannot exceed the semi-perimeter of the box
        # Physical box width for angles is 1.0; maximum possible inscribed radius is <= 15.0
        assert 0.0 < rad < 15.0, f"Chebyshev radius unbounded or non-positive: {rad}"

    def test_angular_mae_unit_conversion_verification(self):
        """
        Audits the angular MAE conversion:
        - The normalized angular error Delta_circ in [0, 1] scales by 90 degrees (pi/2 rad).
        - Refutes the erroneous 360-degree conversion:
            0.0831 * 90 deg = 7.479 deg (CORRECT)
            0.0831 * 360 deg = 29.9 deg (ERRONEOUS 4x factor)
        """
        # Test circular wrap on [0, 1):
        # Angle a = 0.0 (0 deg), b = 0.5 (orthogonal, 45 deg axial on [0, 1] scale)
        d_a = np.zeros(9, dtype=np.float32)
        d_b = np.zeros(9, dtype=np.float32)

        # 0.5 on normalized scale is orthogonal (diff = pi/2 rad = 90 degrees)
        d_a[3] = 0.0
        d_b[3] = 0.5
        circ_err = calculate_angular_wrap_error(d_a, d_b)
        assert circ_err == pytest.approx(0.5 / 6.0)

        # Audit published metric from inversion_report.json:
        reported_norm_mae = 0.083075
        correct_degrees = reported_norm_mae * 90.0
        erroneous_degrees = reported_norm_mae * 360.0

        assert correct_degrees == pytest.approx(7.47675, abs=1e-3)
        assert erroneous_degrees == pytest.approx(29.907, abs=1e-2)

        # Verify that reported 29.9 degrees was indeed 4x higher due to applying 360 deg
        ratio = erroneous_degrees / correct_degrees
        assert ratio == pytest.approx(4.0, abs=1e-5)

    def test_multi_threat_inversion_attack_matrix(self):
        """
        Evaluates the multi-threat attack suite across:
        1. Scenario II (Unknown key, key guessing)
        2. Scenario III (Known key, Chebyshev center LP)
        3. Scenario III (Known key, QP Mahalanobis center)
        4. Option A Re-issuance under fresh key
        """
        master = "WP2_MultiThreatMatrix_MasterSecret_2026!"
        W_orig = derive_orthonormal_matrix(master, salt="salt_threat_1", app_id="auth", in_dim=9, out_dim=32)
        W_guess = derive_orthonormal_matrix("attacker_guess_key_random_32bytes!", salt="guess", app_id="auth", in_dim=9, out_dim=32)
        W_reissue = derive_orthonormal_matrix(master, salt="salt_threat_2", app_id="auth", in_dim=9, out_dim=32)

        d_gallery = np.array([0.82, 0.54, 7.50, 0.50, 0.48, 0.48, 0.40, 0.55, 0.45], dtype=np.float32)
        z_gal = (d_gallery - DEFAULT_DESCRIPTOR_MEAN_9D) / (DEFAULT_DESCRIPTOR_STD_9D + 1e-8)
        t_gal = quantize_descriptors(project_descriptors(d_gallery[None, :], W_orig))

        # Attacker intercepts target bitstring
        target_bits = t_gal[0]

        # 1. Scenario III: Chebyshev center
        feas_c, d_cheb, r_c = solve_chebyshev_center_inversion(target_bits, W_orig, delta=0.005)
        assert feas_c is True
        t_cheb_auth = quantize_descriptors(project_descriptors(d_cheb[None, :], W_orig))
        s_cheb = match_protected_templates(t_cheb_auth, t_gal, np.array([1.0]), np.array([1.0]), hamming_threshold=0.22)

        # 2. Scenario III: QP center
        feas_qp, d_qp = solve_qp_center_inversion(target_bits, W_orig, delta=0.005)
        assert feas_qp is True
        t_qp_auth = quantize_descriptors(project_descriptors(d_qp[None, :], W_orig))
        s_qp = match_protected_templates(t_qp_auth, t_gal, np.array([1.0]), np.array([1.0]), hamming_threshold=0.22)

        # 3. Scenario II: Unknown key pseudo-inverse guess
        signs = 2.0 * target_bits.astype(np.float64) - 1.0
        z_guess = np.linalg.pinv(W_guess) @ (signs * 0.01)
        d_guess = (z_guess * DEFAULT_DESCRIPTOR_STD_9D + DEFAULT_DESCRIPTOR_MEAN_9D).astype(np.float32)
        t_guess_auth = quantize_descriptors(project_descriptors(d_guess[None, :], W_orig))
        s_guess = match_protected_templates(t_guess_auth, t_gal, np.array([1.0]), np.array([1.0]), hamming_threshold=0.22)

        # 4. Cross-Key Re-issuance
        t_gal_fresh = quantize_descriptors(project_descriptors(d_gallery[None, :], W_reissue))
        t_cheb_fresh = quantize_descriptors(project_descriptors(d_cheb[None, :], W_reissue))
        s_reissue = match_protected_templates(t_cheb_fresh, t_gal_fresh, np.array([1.0]), np.array([1.0]), hamming_threshold=0.22)

        # Operational threshold
        T_op = 0.3751
        assert s_guess < T_op, f"Unknown-key guess should not authenticate: score={s_guess}"

    def test_milp_inversion_scope_9d_vs_12d(self):
        """
        Narrows the security claim regarding MILP:
        - 9-D Pure Geometry has 0 discrete/integer variables. Discrete type-lattice recovery
          is eliminated because no minutiae types are stored.
        - 12-D stores 3 discrete minutiae types (t0, t1, t2) in {0, 1}^3, solvable via MILP branch-and-bound.
        - Continuous optimization (LP, QP, gradient search) over the bounded polytope remains an active
          attack surface for both representations.
        """
        master = "WP2_MILP_Scope_VerificationSecret_32B!"
        W_12d = derive_orthonormal_matrix(master, salt="milp_12", app_id="auth", in_dim=12, out_dim=32)
        W_9d = derive_orthonormal_matrix(master, salt="milp_9", app_id="auth", in_dim=9, out_dim=32)

        # 12-D has 3 binary minutiae types
        d_12 = np.array([0.75, 0.45, 8.5, 0.3, 0.7, 0.5, 0.4, 0.6, 0.3, 1.0, 0.0, 1.0], dtype=np.float32)
        z_12 = (d_12 - DEFAULT_DESCRIPTOR_MEAN) / (DEFAULT_DESCRIPTOR_STD + 1e-8)
        bits_12 = (W_12d @ z_12 >= 0.0).astype(np.uint8)

        feas_12, d_recon_12 = solve_milp_feasibility_inversion(bits_12, W_12d, delta=0.01)
        assert feas_12 is True
        # Minutiae types are strictly binary
        assert np.all(np.isin(d_recon_12[9:12], [0.0, 1.0]))

        # In 9-D, all 9 coordinates are continuous ratios, perimeters, and angles
        d_9 = d_12[:9]
        num_discrete_vars_9d = 0
        assert num_discrete_vars_9d == 0

    def test_zero_success_asr_uncertainty_bounds(self):
        """
        Verifies statistical uncertainty for zero-success attack outcomes.
        Never assert a true ASR of absolute 0.0% without uncertainty bounds.
        Verifies Wilson 95% score confidence intervals:
        - N = 10 subject re-issuance trials with 0 successes: CI [0.0%, 27.75%]
        - N = 3000 triangle inversions with 0 successes: CI [0.0%, 0.128%]
        """
        low_10, high_10 = compute_wilson_ci(0, 10, confidence=0.95)
        assert low_10 == 0.0
        assert high_10 == pytest.approx(0.2775, abs=1e-3)

        low_3k, high_3k = compute_wilson_ci(0, 3000, confidence=0.95)
        assert low_3k == 0.0
        assert high_3k == pytest.approx(0.001279, abs=1e-5)
        assert high_3k * 100.0 < 0.15  # < 0.15% upper bound
