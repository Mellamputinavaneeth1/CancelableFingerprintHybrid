"""
WP1 Formal Mathematical and Descriptor Property Verification Suite.
===================================================================
Rigorous property-based tests verifying:
1. Minutiae orientation convention: axial ridge orientations modulo pi in [0, pi).
2. Deterministic canonical vertex ordering & tie-breaking:
   - Scalene triangles (all sides distinct).
   - Isosceles triangles (two sides equal).
   - Equilateral triangles (all three sides equal).
   - Invariance across all 6 input permutations: S_3 symmetry.
3. Planar cyclic permutation equivariance under C_3:
   - Cyclic shift of vertices produces cyclic shift of descriptor features.
4. Reflection chirality (O(2) \\ SO(2)):
   - Mirror reflections invert planar chirality and alter descriptors.
   - Preserving mirror asymmetry is essential for fingerprint biometric security.
5. Reduced 5-D geometric descriptor:
   - Scale, rotation, translation invariance.
   - Correspondence with 9-D features.
"""

import math
import itertools
import pytest
import numpy as np

from src.minutiae.extractor import Minutia
from src.graph.descriptors import (
    _axial_diff,
    extract_triangle_descriptor,
    extract_triangle_descriptor_5d,
    extract_triangle_descriptor_with_ordered_minutiae,
    extract_all_descriptors,
    extract_all_descriptors_5d,
    extract_all_cyclic_descriptors,
    DEFAULT_DESCRIPTOR_MEAN_9D,
    DEFAULT_DESCRIPTOR_STD_9D,
    DEFAULT_DESCRIPTOR_MEAN_5D,
    DEFAULT_DESCRIPTOR_STD_5D
)


class TestMinutiaeOrientationConvention:
    """Verifies axial ridge orientation modulo pi convention."""

    def test_axial_orientation_periodicity(self):
        """Orientation theta is invariant under theta + k * pi."""
        angles = [0.0, 0.1, 0.5, 1.0, 1.5, np.pi - 0.01]
        for th in angles:
            # Difference with itself shifted by pi should be 0.0
            diff = _axial_diff(th, th + np.pi)
            assert diff == pytest.approx(0.0, abs=1e-6), f"Failed for angle {th}"
            diff_2pi = _axial_diff(th, th + 2 * np.pi)
            assert diff_2pi == pytest.approx(0.0, abs=1e-6)

    def test_axial_diff_maximum_orthogonal(self):
        """Orthogonal orientations (difference = pi / 2) yield normalized distance 1.0."""
        for base in [0.0, 0.25, 0.5, 1.0, 2.0]:
            ortho = base + np.pi / 2.0
            diff = _axial_diff(base, ortho)
            assert diff == pytest.approx(1.0, abs=1e-6), f"Orthogonal diff must be 1.0, got {diff}"

    def test_axial_diff_symmetry_and_bounds(self):
        """_axial_diff is symmetric and strictly bounded in [0.0, 1.0]."""
        test_angles = np.linspace(-2 * np.pi, 2 * np.pi, 25)
        for a in test_angles:
            for b in test_angles:
                d_ab = _axial_diff(a, b)
                d_ba = _axial_diff(b, a)
                assert 0.0 <= d_ab <= 1.0
                assert d_ab == pytest.approx(d_ba, abs=1e-6)


class TestCanonicalVertexOrderingAndTieBreaking:
    """Tests deterministic canonical vertex ordering under all permutations."""

    def test_scalene_all_six_permutations(self):
        """All 6 permutations of a scalene triangle produce identical 9-D descriptors."""
        m1 = Minutia(12.0, 15.0, theta=0.3, type=0, quality=0.85)
        m2 = Minutia(58.0, 22.0, theta=1.1, type=1, quality=0.90)
        m3 = Minutia(35.0, 70.0, theta=2.4, type=0, quality=0.75)

        base_desc, base_q = extract_triangle_descriptor(m1, m2, m3, include_types=False)
        assert base_desc.shape == (9,)

        for perm in itertools.permutations([m1, m2, m3]):
            desc_p, q_p = extract_triangle_descriptor(perm[0], perm[1], perm[2], include_types=False)
            np.testing.assert_allclose(base_desc, desc_p, atol=1e-5,
                                       err_msg=f"Permutation failed: {perm}")
            assert base_q == pytest.approx(q_p, abs=1e-6)

    def test_isosceles_tie_breaking_all_permutations(self):
        """Isosceles triangle (two sides equal) resolves tie-breaking deterministically."""
        # Vertex A at (0, 0), B at (40, 0), C at (20, 30) -> AC = BC = sqrt(20^2 + 30^2) = sqrt(1300)
        mA = Minutia(0.0, 0.0, theta=0.4, type=0, quality=0.80)
        mB = Minutia(40.0, 0.0, theta=1.3, type=1, quality=0.88)
        mC = Minutia(20.0, 30.0, theta=2.2, type=0, quality=0.92)

        base_desc, base_q = extract_triangle_descriptor(mA, mB, mC, include_types=False)
        assert base_desc.shape == (9,)

        for perm in itertools.permutations([mA, mB, mC]):
            desc_p, q_p = extract_triangle_descriptor(perm[0], perm[1], perm[2], include_types=False)
            np.testing.assert_allclose(base_desc, desc_p, atol=1e-5,
                                       err_msg=f"Isosceles permutation tie-break failed: {perm}")
            assert base_q == pytest.approx(q_p, abs=1e-6)

    def test_equilateral_tie_breaking_all_permutations(self):
        """Equilateral triangle (all three sides equal) resolves tie-breaking deterministically."""
        s = 60.0
        h = s * np.sqrt(3.0) / 2.0
        m1 = Minutia(0.0, 0.0, theta=0.2, type=0, quality=0.85)
        m2 = Minutia(s, 0.0, theta=1.4, type=1, quality=0.90)
        m3 = Minutia(s / 2.0, h, theta=2.7, type=0, quality=0.80)

        base_desc, base_q = extract_triangle_descriptor(m1, m2, m3, include_types=False)
        assert base_desc.shape == (9,)

        for perm in itertools.permutations([m1, m2, m3]):
            desc_p, q_p = extract_triangle_descriptor(perm[0], perm[1], perm[2], include_types=False)
            np.testing.assert_allclose(base_desc, desc_p, atol=1e-5,
                                       err_msg=f"Equilateral permutation tie-break failed: {perm}")
            assert base_q == pytest.approx(q_p, abs=1e-6)

    def test_equilateral_identical_orientations_tie_breaking(self):
        """Equilateral triangle with identical minutiae orientations produces invariant descriptor."""
        s = 50.0
        h = s * np.sqrt(3.0) / 2.0
        # Exactly identical orientation on all vertices
        m1 = Minutia(0.0, 0.0, theta=1.0, type=0, quality=0.8)
        m2 = Minutia(s, 0.0, theta=1.0, type=0, quality=0.8)
        m3 = Minutia(s / 2.0, h, theta=1.0, type=0, quality=0.8)

        base_desc, _ = extract_triangle_descriptor(m1, m2, m3, include_types=False)
        for perm in itertools.permutations([m1, m2, m3]):
            desc_p, _ = extract_triangle_descriptor(perm[0], perm[1], perm[2], include_types=False)
            np.testing.assert_allclose(base_desc, desc_p, atol=1e-5)

    def test_symmetric_isosceles_ordered_minutiae_permutation_invariance(self):
        """
        Symmetric isosceles triangle with mirror-symmetric minutiae orientations:
        Verifies that planar chirality tie-breaking resolves the canonical vertex ordering
        strictly deterministically across all 6 input permutations, producing identical
        ordered minutiae coordinates and identical 9-D descriptors.
        """
        mA = Minutia(0.0, 0.0, theta=np.pi / 4.0, type=0, quality=0.80)
        mB = Minutia(40.0, 0.0, theta=3.0 * np.pi / 4.0, type=0, quality=0.80)
        mC = Minutia(20.0, 30.0, theta=np.pi / 2.0, type=0, quality=0.80)

        base_desc, base_q, base_ord = extract_triangle_descriptor_with_ordered_minutiae(
            mA, mB, mC, include_types=False
        )
        base_coords = [(m.x, m.y) for m in base_ord]

        for perm in itertools.permutations([mA, mB, mC]):
            d_p, q_p, ord_p = extract_triangle_descriptor_with_ordered_minutiae(
                perm[0], perm[1], perm[2], include_types=False
            )
            np.testing.assert_allclose(base_desc, d_p, atol=1e-5)
            assert base_q == pytest.approx(q_p, abs=1e-6)
            perm_coords = [(m.x, m.y) for m in ord_p]
            assert base_coords == perm_coords, f"Canonical vertex ordering differed for perm {perm}: {perm_coords} != {base_coords}"

    def test_degenerate_collinear_triangles_all_permutations(self):
        """
        Collinear minutiae triplets (area = 0, flat simplex):
        Delaunay simplices near fingerprint borders or along straight ridge lines
        must not produce NaNs or infinities and must resolve canonically across all 6 permutations.
        """
        mA = Minutia(0.0, 0.0, theta=0.2, type=0, quality=0.80)
        mB = Minutia(15.0, 0.0, theta=0.4, type=1, quality=0.85)
        mC = Minutia(30.0, 0.0, theta=0.6, type=0, quality=0.75)

        base_9d, q_9d = extract_triangle_descriptor(mA, mB, mC, include_types=False)
        base_5d, q_5d = extract_triangle_descriptor_5d(mA, mB, mC)

        assert not np.isnan(base_9d).any()
        assert not np.isnan(base_5d).any()
        assert not np.isinf(base_9d).any()
        assert not np.isinf(base_5d).any()

        for perm in itertools.permutations([mA, mB, mC]):
            d9, _ = extract_triangle_descriptor(perm[0], perm[1], perm[2], include_types=False)
            d5, _ = extract_triangle_descriptor_5d(perm[0], perm[1], perm[2])
            np.testing.assert_allclose(base_9d, d9, atol=1e-4)
            np.testing.assert_allclose(base_5d, d5, atol=1e-4)

    def test_coincident_minutiae_vertices_no_nan(self):
        """Coincident minutiae (distance = 0) do not cause division-by-zero or NaNs."""
        m_dup1 = Minutia(10.0, 10.0, theta=0.3, type=0, quality=0.8)
        m_dup2 = Minutia(10.0, 10.0, theta=0.5, type=1, quality=0.8)
        m_other = Minutia(30.0, 40.0, theta=1.1, type=0, quality=0.9)

        d9, q9 = extract_triangle_descriptor(m_dup1, m_dup2, m_other, include_types=False)
        d5, q5 = extract_triangle_descriptor_5d(m_dup1, m_dup2, m_other)

        assert not np.isnan(d9).any()
        assert not np.isnan(d5).any()
        assert d9.shape == (9,)
        assert d5.shape == (5,)

    def test_9d_rigid_rotation_and_translation_invariance(self):
        """
        Verifies 9-D descriptor invariance under arbitrary SE(2) rigid transformations
        (rotation angles in [0, pi] and spatial translations).
        """
        m1 = Minutia(20.0, 30.0, theta=0.5, type=0, quality=0.8)
        m2 = Minutia(70.0, 35.0, theta=1.2, type=1, quality=0.9)
        m3 = Minutia(45.0, 80.0, theta=2.4, type=0, quality=0.85)

        d_base, q_base = extract_triangle_descriptor(m1, m2, m3, include_types=False)

        for ang in [0.3, np.pi / 4, 1.0, np.pi / 2, 2.5]:
            c, s = np.cos(ang), np.sin(ang)
            tx, ty = 85.0, -42.0

            def transform(m):
                rx = m.x * c - m.y * s + tx
                ry = m.x * s + m.y * c + ty
                r_th = (m.theta + ang) % np.pi
                return Minutia(float(rx), float(ry), float(r_th), m.type, m.quality)

            m1_t, m2_t, m3_t = transform(m1), transform(m2), transform(m3)
            d_trans, q_trans = extract_triangle_descriptor(m1_t, m2_t, m3_t, include_types=False)
            np.testing.assert_allclose(d_base, d_trans, atol=1e-4)
            assert q_base == pytest.approx(q_trans, abs=1e-6)

    def test_9d_uniform_scaling_behavior(self):
        """
        Verifies behavior under uniform scaling s > 0:
        - Dimensionless ratios r1, r2 and angular features (phi0..2, dth01..20) are invariant.
        - Physical perimeter P_mm scales strictly linearly with s.
        """
        m1 = Minutia(20.0, 30.0, theta=0.5, type=0, quality=0.8)
        m2 = Minutia(70.0, 35.0, theta=1.2, type=1, quality=0.9)
        m3 = Minutia(45.0, 80.0, theta=2.4, type=0, quality=0.85)

        d_base, _ = extract_triangle_descriptor(m1, m2, m3, include_types=False)

        for scale in [0.5, 1.5, 2.0]:
            m1_s = Minutia(m1.x * scale, m1.y * scale, m1.theta, m1.type, m1.quality)
            m2_s = Minutia(m2.x * scale, m2.y * scale, m2.theta, m2.type, m2.quality)
            m3_s = Minutia(m3.x * scale, m3.y * scale, m3.theta, m3.type, m3.quality)

            d_s, _ = extract_triangle_descriptor(m1_s, m2_s, m3_s, include_types=False)
            # Ratios r1, r2 are scale-invariant
            np.testing.assert_allclose(d_base[0:2], d_s[0:2], atol=1e-5)
            # P_mm scales proportionally
            np.testing.assert_allclose(d_base[2] * scale, d_s[2], rtol=1e-4)
            # Angular features phi0..phi2 and dth01..dth20 are scale-invariant
            np.testing.assert_allclose(d_base[3:6], d_s[3:6], atol=1e-5)
            np.testing.assert_allclose(d_base[6:9], d_s[6:9], atol=1e-5)

    def test_near_tie_perturbation_stability(self):
        """
        Verifies numerical stability when side lengths are nearly equal (near-tie epsilon).
        Descriptor extraction must remain finite and well-conditioned without divergence.
        """
        for eps in [1e-7, 1e-5, 1e-3, 1e-2]:
            mA = Minutia(0.0, 0.0, theta=0.4, type=0, quality=0.80)
            mB = Minutia(40.0 + eps, 0.0, theta=1.3, type=1, quality=0.88)
            mC = Minutia(20.0, 30.0, theta=2.2, type=0, quality=0.92)

            d, q = extract_triangle_descriptor(mA, mB, mC, include_types=False)
            assert not np.isnan(d).any()
            assert not np.isinf(d).any()
            assert d.shape == (9,)
            assert 0.0 < q <= 1.0



class TestCyclicEquivarianceAndChirality:
    """Verifies C_3 cyclic equivariance and O(2) \\ SO(2) chirality inversion."""

    def test_c3_cyclic_permutation_representation(self):
        """
        Under cyclic permutation subgroup C_3 = {(0,1,2), (1,2,0), (2,0,1)},
        extract_all_cyclic_descriptors produces 3 circularly shifted descriptors.
        """
        mA = Minutia(15.0, 20.0, theta=0.4, type=0, quality=0.9)
        mB = Minutia(55.0, 25.0, theta=1.1, type=1, quality=0.85)
        mC = Minutia(30.0, 65.0, theta=2.3, type=0, quality=0.8)

        tris = [(mA, mB, mC)]
        cyclic_descs, _ = extract_all_cyclic_descriptors(tris, include_types=False)
        assert cyclic_descs.shape == (1, 3, 9)

        # Shift 0, 1, 2 must all have the same physical perimeter (feature 2)
        p0 = cyclic_descs[0, 0, 2]
        p1 = cyclic_descs[0, 1, 2]
        p2 = cyclic_descs[0, 2, 2]
        assert p0 == pytest.approx(p1, abs=1e-5)
        assert p1 == pytest.approx(p2, abs=1e-5)

        # Mutual differences (features 6, 7, 8) must be cyclic shifts of each other
        dth_shift0 = cyclic_descs[0, 0, 6:9]
        dth_shift1 = cyclic_descs[0, 1, 6:9]
        dth_shift2 = cyclic_descs[0, 2, 6:9]

        np.testing.assert_allclose(dth_shift1, np.roll(dth_shift0, -1), atol=1e-5)
        np.testing.assert_allclose(dth_shift2, np.roll(dth_shift0, -2), atol=1e-5)

    def test_reflection_chirality_inversion(self):
        """
        Mathematical Chirality Audit:
        1. In cyclic representation (extract_all_cyclic_descriptors), the planar counter-clockwise
           orientation (signed cross product >= 0) is enforced. Under mirror reflection across an axis,
           chirality is inverted (O(2) \\ SO(2)) and the cyclic descriptors are distinguished (dist > 0.01).
        2. In canonical representation (extract_triangle_descriptor), evaluating all 6 permutations in S_3
           (which includes odd reflection permutations) causes the canonical descriptor to be
           invariant under O(2) (including reflections).
        This test explicitly documents and verifies this mathematical property across both representations.
        """
        mA = Minutia(10.0, 15.0, theta=0.35, type=0, quality=0.85)
        mB = Minutia(60.0, 20.0, theta=1.20, type=1, quality=0.90)
        mC = Minutia(35.0, 65.0, theta=2.10, type=0, quality=0.80)

        # 1. Cyclic representation under planar orientation: Chirality inversion detected
        base_cyclic, _ = extract_all_cyclic_descriptors([(mA, mB, mC)], include_types=False)
        refl_cyclic, _ = extract_all_cyclic_descriptors([(
            Minutia(mA.x, -mA.y, (-mA.theta) % np.pi, mA.type, mA.quality),
            Minutia(mB.x, -mB.y, (-mB.theta) % np.pi, mB.type, mB.quality),
            Minutia(mC.x, -mC.y, (-mC.theta) % np.pi, mC.type, mC.quality)
        )], include_types=False)

        cyclic_min_diff = np.min([np.mean(np.abs(refl_cyclic[0, 0] - base_cyclic[0, k])) for k in range(3)])
        assert cyclic_min_diff > 0.01, f"Chirality inversion not detected in cyclic representation: {cyclic_min_diff}"

        # 2. Canonical representation: S_3 symmetry covers O(2) isometries
        desc_orig, _ = extract_triangle_descriptor(mA, mB, mC, include_types=False)
        desc_refl, _ = extract_triangle_descriptor(
            Minutia(mA.x, -mA.y, (-mA.theta) % np.pi, mA.type, mA.quality),
            Minutia(mB.x, -mB.y, (-mB.theta) % np.pi, mB.type, mB.quality),
            Minutia(mC.x, -mC.y, (-mC.theta) % np.pi, mC.type, mC.quality),
            include_types=False
        )
        # S_3 optimization evaluates both orientations, finding equivalent canonical ordering
        np.testing.assert_allclose(desc_orig, desc_refl, atol=1e-5)


class TestReduced5DGeometricDescriptor:
    """Verifies the reduced 5-D geometric descriptor: [L1, L2, dth01, dth12, dth20]."""

    def test_5d_shape_and_range(self):
        """5-D descriptor has shape (5,) with all components in valid mathematical ranges."""
        m1 = Minutia(15.0, 20.0, theta=0.4, type=0, quality=0.9)
        m2 = Minutia(55.0, 25.0, theta=1.1, type=1, quality=0.85)
        m3 = Minutia(30.0, 65.0, theta=2.3, type=0, quality=0.8)

        d5, q = extract_triangle_descriptor_5d(m1, m2, m3)
        assert d5.shape == (5,)
        assert 0.0 < q <= 1.0

        # L1, L2 are normalized side length ratios: 0 < L1, L2 < 1.0, and L1 + L2 < 1.0
        assert 0.0 < d5[0] < 1.0
        assert 0.0 < d5[1] < 1.0
        assert d5[0] + d5[1] < 1.0

        # dth01, dth12, dth20 are axial differences in [0.0, 1.0]
        for i in range(2, 5):
            assert 0.0 <= d5[i] <= 1.0

    def test_5d_scale_invariance(self):
        """5-D descriptor is strictly scale-invariant (unlike 9-D which has P_mm)."""
        m1 = Minutia(15.0, 20.0, theta=0.4, type=0, quality=0.9)
        m2 = Minutia(55.0, 25.0, theta=1.1, type=1, quality=0.85)
        m3 = Minutia(30.0, 65.0, theta=2.3, type=0, quality=0.8)

        d_orig, _ = extract_triangle_descriptor_5d(m1, m2, m3)

        for scale in [0.5, 1.5, 2.0, 3.0]:
            m1_scaled = Minutia(m1.x * scale, m1.y * scale, m1.theta, m1.type, m1.quality)
            m2_scaled = Minutia(m2.x * scale, m2.y * scale, m2.theta, m2.type, m2.quality)
            m3_scaled = Minutia(m3.x * scale, m3.y * scale, m3.theta, m3.type, m3.quality)

            d_scaled, _ = extract_triangle_descriptor_5d(m1_scaled, m2_scaled, m3_scaled)
            np.testing.assert_allclose(d_orig, d_scaled, atol=1e-5,
                                       err_msg=f"5-D descriptor failed scale invariance at scale={scale}")

    def test_5d_rigid_rotation_and_translation_invariance(self):
        """5-D descriptor is strictly invariant to 2D rotation and translation."""
        m1 = Minutia(20.0, 30.0, theta=0.5, type=0, quality=0.8)
        m2 = Minutia(70.0, 35.0, theta=1.2, type=1, quality=0.9)
        m3 = Minutia(45.0, 80.0, theta=2.4, type=0, quality=0.85)

        d_base, _ = extract_triangle_descriptor_5d(m1, m2, m3)

        for ang in [0.3, np.pi / 4, 1.0, np.pi / 2, 2.5]:
            c, s = np.cos(ang), np.sin(ang)
            tx, ty = 85.0, -42.0

            def transform(m):
                rx = m.x * c - m.y * s + tx
                ry = m.x * s + m.y * c + ty
                r_th = (m.theta + ang) % np.pi
                return Minutia(float(rx), float(ry), float(r_th), m.type, m.quality)

            m1_t, m2_t, m3_t = transform(m1), transform(m2), transform(m3)
            d_trans, _ = extract_triangle_descriptor_5d(m1_t, m2_t, m3_t)
            np.testing.assert_allclose(d_base, d_trans, atol=1e-4)

    def test_5d_extract_all_descriptors(self):
        """extract_all_descriptors_5d returns correct shapes on triangle list."""
        m1 = Minutia(10.0, 10.0, 0.1, 0, 0.9)
        m2 = Minutia(30.0, 15.0, 0.4, 1, 0.8)
        m3 = Minutia(20.0, 40.0, 0.8, 0, 0.7)
        m4 = Minutia(50.0, 45.0, 1.2, 1, 0.85)

        triangles = [(m1, m2, m3), (m2, m3, m4)]
        descs, weights = extract_all_descriptors_5d(triangles)
        assert descs.shape == (2, 5)
        assert weights.shape == (2,)

        empty_descs, empty_weights = extract_all_descriptors_5d([])
        assert empty_descs.shape == (0, 5)
        assert empty_weights.shape == (0,)


class TestMathematicalDependenciesAndRedundancy:
    """
    Mathematical Proof and Invariant Verification of Feature Dependencies (WP1):
    1. The 3 Euclidean side lengths (l0, l1, l2) and interior angles (gamma0, gamma1, gamma2)
       are completely determined by (r1, r2, P_mm) via the Law of Cosines.
    2. Directed mutual orientation differences sum to 0 mod pi, proving the 3 mutual orientations
       lie on a 2-D submanifold, mathematically demonstrating the 9-D descriptor's redundancy.
    3. The minimal complete non-redundant parameterization of an oriented planar triangle
       under SE(2) rigid isometries has intrinsic dimension dim = 9 - 3 = 6.
    """

    def test_law_of_cosines_interior_angle_determination(self):
        """r1 and r2 mathematically determine interior angles gamma_i with zero residual degrees of freedom."""
        rng = np.random.RandomState(42)
        for _ in range(20):
            # Generate random non-degenerate triangle
            pts = rng.uniform(10.0, 100.0, size=(3, 2))
            l0 = float(np.hypot(pts[1, 0] - pts[2, 0], pts[1, 1] - pts[2, 1]))
            l1 = float(np.hypot(pts[2, 0] - pts[0, 0], pts[2, 1] - pts[0, 1]))
            l2 = float(np.hypot(pts[0, 0] - pts[1, 0], pts[0, 1] - pts[1, 1]))

            # Skip degenerate
            if l0 + l1 <= l2 or l1 + l2 <= l0 or l2 + l0 <= l1:
                continue

            # Sort descending
            edges = sorted([l0, l1, l2], reverse=True)
            e_max, e_mid, e_min = edges[0], edges[1], edges[2]
            r1 = e_mid / e_max
            r2 = e_min / e_max

            # Compute interior angles using side lengths
            cos_g0 = (e_mid**2 + e_min**2 - e_max**2) / (2.0 * e_mid * e_min)
            cos_g1 = (e_max**2 + e_min**2 - e_mid**2) / (2.0 * e_max * e_min)
            cos_g2 = (e_max**2 + e_mid**2 - e_min**2) / (2.0 * e_max * e_mid)

            # Compute interior angles using purely normalized ratios (r1, r2)
            cos_g0_r = (r1**2 + r2**2 - 1.0) / (2.0 * r1 * r2)
            cos_g1_r = (1.0 + r2**2 - r1**2) / (2.0 * r2)
            cos_g2_r = (1.0 + r1**2 - r2**2) / (2.0 * r1)

            np.testing.assert_allclose(cos_g0, cos_g0_r, atol=1e-6)
            np.testing.assert_allclose(cos_g1, cos_g1_r, atol=1e-6)
            np.testing.assert_allclose(cos_g2, cos_g2_r, atol=1e-6)

    def test_mutual_orientation_circular_sum_constraint(self):
        """Directed minutiae orientation differences sum to 0 mod pi."""
        rng = np.random.RandomState(123)
        for _ in range(20):
            th0, th1, th2 = rng.uniform(0.0, np.pi, size=3)
            # Directed differences mod pi
            d01 = (th0 - th1) % np.pi
            d12 = (th1 - th2) % np.pi
            d20 = (th2 - th0) % np.pi
            total = (d01 + d12 + d20) % np.pi
            assert total == pytest.approx(0.0, abs=1e-6) or total == pytest.approx(np.pi, abs=1e-6)


class TestMinimalManifold6DAnd8D:
    """Verifies property invariants of 6-D and 8-D candidate representations."""

    def test_6d_descriptor_invariance_and_permutations(self):
        """6-D descriptor [r1, r2, P_mm, phi0, phi1, phi2] is invariant to SE(2) and all 6 vertex permutations."""
        from src.graph.descriptors import extract_triangle_descriptor_6d, extract_all_descriptors_6d

        m1 = Minutia(15.0, 25.0, theta=0.6, type=0, quality=0.85)
        m2 = Minutia(65.0, 30.0, theta=1.4, type=1, quality=0.90)
        m3 = Minutia(40.0, 75.0, theta=2.2, type=0, quality=0.80)

        base_6d, base_q = extract_triangle_descriptor_6d(m1, m2, m3)
        assert base_6d.shape == (6,)

        # All 6 permutations produce identical 6-D descriptor
        for perm in itertools.permutations([m1, m2, m3]):
            d_p, q_p = extract_triangle_descriptor_6d(perm[0], perm[1], perm[2])
            np.testing.assert_allclose(base_6d, d_p, atol=1e-5)

        # SE(2) rigid rotation and translation
        c, s = np.cos(0.9), np.sin(0.9)
        tx, ty = 120.0, -80.0
        def tf(m):
            return Minutia(m.x * c - m.y * s + tx, m.x * s + m.y * c + ty, (m.theta + 0.9) % np.pi, m.type, m.quality)

        d_tf, _ = extract_triangle_descriptor_6d(tf(m1), tf(m2), tf(m3))
        np.testing.assert_allclose(base_6d, d_tf, atol=1e-4)

    def test_8d_descriptor_invariance_and_permutations(self):
        """8-D descriptor [r1, r2, P_mm, phi0, phi1, phi2, dth01, dth12] is invariant to SE(2) and all 6 permutations."""
        from src.graph.descriptors import extract_triangle_descriptor_8d

        m1 = Minutia(15.0, 25.0, theta=0.6, type=0, quality=0.85)
        m2 = Minutia(65.0, 30.0, theta=1.4, type=1, quality=0.90)
        m3 = Minutia(40.0, 75.0, theta=2.2, type=0, quality=0.80)

        base_8d, _ = extract_triangle_descriptor_8d(m1, m2, m3)
        assert base_8d.shape == (8,)

        for perm in itertools.permutations([m1, m2, m3]):
            d_p, _ = extract_triangle_descriptor_8d(perm[0], perm[1], perm[2])
            np.testing.assert_allclose(base_8d, d_p, atol=1e-5)

    def test_candidate_dimensionality_spectrum(self):
        """Verifies pipeline feature extraction across all 5 candidates: 5-D, 6-D, 8-D, 9-D, 12-D."""
        from src.graph.descriptors import extract_all_descriptors

        m1 = Minutia(10.0, 10.0, 0.2, 0, 0.9)
        m2 = Minutia(40.0, 20.0, 0.8, 1, 0.85)
        m3 = Minutia(25.0, 50.0, 1.9, 0, 0.8)
        triangles = [(m1, m2, m3)]

        for d in [5, 6, 8, 9, 12]:
            include_types = (d == 12)
            descs, weights = extract_all_descriptors(triangles, include_types=include_types, desc_dim=d)
            assert descs.shape == (1, d), f"Expected shape (1, {d}), got {descs.shape}"
            assert weights.shape == (1,)

    def test_8d_absolute_distance_constraint_counterexample(self):
        """
        Refutes the claim that dth20 is a deterministic scalar function of (dth01, dth12).
        Because distances on RP^1 = R / (pi Z) are undirected absolute values in [0, 1],
        the circular sum can resolve in multiple branches depending on orientation topology.

        Counterexample:
          Case A: thetas = [0, pi/4, pi/2] -> dth01=0.5, dth12=0.5, dth20=1.0
          Case B: thetas = [pi/4, 0, pi/4] -> dth01=0.5, dth12=0.5, dth20=0.0
        Both cases have identical (dth01, dth12) = (0.5, 0.5), yet dth20 differs by 1.0.
        Thus dth20 provides independent metric information constrained by triangle inequalities,
        not exact algebraic redundancy.
        """
        dth01_A = _axial_diff(0.0, np.pi / 4.0)
        dth12_A = _axial_diff(np.pi / 4.0, np.pi / 2.0)
        dth20_A = _axial_diff(np.pi / 2.0, 0.0)

        dth01_B = _axial_diff(np.pi / 4.0, 0.0)
        dth12_B = _axial_diff(0.0, np.pi / 4.0)
        dth20_B = _axial_diff(np.pi / 4.0, np.pi / 4.0)

        assert dth01_A == pytest.approx(0.5, abs=1e-6)
        assert dth12_A == pytest.approx(0.5, abs=1e-6)
        assert dth20_A == pytest.approx(1.0, abs=1e-6)

        assert dth01_B == pytest.approx(0.5, abs=1e-6)
        assert dth12_B == pytest.approx(0.5, abs=1e-6)
        assert dth20_B == pytest.approx(0.0, abs=1e-6)

        # Identical first two differences
        assert dth01_A == pytest.approx(dth01_B, abs=1e-6)
        assert dth12_A == pytest.approx(dth12_B, abs=1e-6)
        # 3rd difference differs: proves non-determinism
        assert abs(dth20_A - dth20_B) == pytest.approx(1.0, abs=1e-6)

    def test_6d_sign_ambiguity_eightfold_collision(self):
        """
        Constructive proof that discarding orientation signs in 6-D produces 8-to-1 collisions.
        For an oriented triangle, each phi_i = min(|theta_i - alpha_i|, pi - |theta_i - alpha_i|)
        collapses sign information.
        There exist 2^3 = 8 distinct orientation triplets on the same geometric triangle that yield
        identical 6-D descriptors, yet produce 8 distinct mutual orientation triplets in 9-D.
        """
        from src.graph.descriptors import extract_triangle_descriptor_6d

        p0 = np.array([20.0, 30.0])
        p1 = np.array([80.0, 35.0])
        p2 = np.array([50.0, 85.0])

        ang01 = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        ang12 = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        ang20 = np.arctan2(p0[1] - p2[1], p0[0] - p2[0])

        delta0 = 0.3 * (0.5 * np.pi)
        delta1 = 0.5 * (0.5 * np.pi)
        delta2 = 0.7 * (0.5 * np.pi)

        configs = []
        d6_list = []
        d9_list = []

        for s0, s1, s2 in itertools.product([1, -1], repeat=3):
            th0 = (ang01 + s0 * delta0) % np.pi
            th1 = (ang12 + s1 * delta1) % np.pi
            th2 = (ang20 + s2 * delta2) % np.pi

            m0 = Minutia(p0[0], p0[1], theta=th0, type=0, quality=0.9)
            m1 = Minutia(p1[0], p1[1], theta=th1, type=0, quality=0.9)
            m2 = Minutia(p2[0], p2[1], theta=th2, type=0, quality=0.9)

            d6, _ = extract_triangle_descriptor_6d(m0, m1, m2)
            d9, _ = extract_triangle_descriptor(m0, m1, m2, include_types=False)

            configs.append((th0, th1, th2))
            d6_list.append(d6)
            d9_list.append(d9)

        assert len(configs) == 8

        # All 8 produce numerically identical 6-D descriptors
        for d6 in d6_list[1:]:
            np.testing.assert_allclose(d6_list[0], d6, atol=1e-4)

        # But all 8 produce distinct 9-D mutual orientation differences
        unique_mutuals = {tuple(np.round(d[6:9], 4)) for d in d9_list}
        assert len(unique_mutuals) == 8

