import numpy as np
import pytest
from src.minutiae.extractor import Minutia
from src.graph.delaunay import build_delaunay_triangles, build_knn_graphs
from src.graph.descriptors import extract_triangle_descriptor, extract_all_descriptors, _axial_diff

def test_axial_diff_bounds_and_symmetry():
    """
    Tests that _axial_diff handles angles from atan2 across [-pi, pi]
    and minutiae orientations across [0, pi), strictly returning values in [0.0, 1.0].
    """
    angles = [-np.pi, -2.5, -np.pi/2, -0.5, 0.0, 0.5, np.pi/2, 2.5, np.pi]
    for a in angles:
        for b in angles:
            diff = _axial_diff(a, b)
            assert 0.0 <= diff <= 1.0, f"Expected [0, 1] but got {diff} for a={a}, b={b}"
            diff_rev = _axial_diff(b, a)
            assert abs(diff - diff_rev) < 1e-6, f"Expected symmetry but got {diff} vs {diff_rev}"

def test_scalene_triangle_descriptor_invariance():
    """Tests that scalene triangles produce identical 12-D descriptors under permutations and translations."""
    m1 = Minutia(10.0, 10.0, theta=0.5, type=0, quality=0.8)
    m2 = Minutia(40.0, 10.0, theta=1.2, type=1, quality=0.9)
    m3 = Minutia(25.0, 35.0, theta=2.1, type=0, quality=0.85)
    
    desc_orig, q_orig = extract_triangle_descriptor(m1, m2, m3)
    assert desc_orig.shape == (12,)
    assert 0.0 < q_orig <= 1.0
    
    # Test all permutations of inputs produce identical descriptors
    for p in [(m1, m2, m3), (m1, m3, m2), (m2, m1, m3), (m2, m3, m1), (m3, m1, m2), (m3, m2, m1)]:
        desc_perm, q_perm = extract_triangle_descriptor(p[0], p[1], p[2])
        np.testing.assert_allclose(desc_orig, desc_perm, atol=1e-5)
        assert abs(q_orig - q_perm) < 1e-6
        
    # Translation invariance: shift all coordinates by (60, 80)
    m1_shift = Minutia(70.0, 90.0, theta=0.5, type=0, quality=0.8)
    m2_shift = Minutia(100.0, 90.0, theta=1.2, type=1, quality=0.9)
    m3_shift = Minutia(85.0, 115.0, theta=2.1, type=0, quality=0.85)
    desc_shift, q_shift = extract_triangle_descriptor(m1_shift, m2_shift, m3_shift)
    np.testing.assert_allclose(desc_orig, desc_shift, atol=1e-5)

def test_isosceles_triangle_tie_breaking():
    """
    Tests that isosceles triangles (where two opposite edge lengths are exactly equal)
    resolve tie-breaking deterministically and yield invariant descriptors regardless
    of the input vertex permutation.
    """
    mA = Minutia(0.0, 0.0, theta=0.3, type=0, quality=0.8)
    mB = Minutia(40.0, 0.0, theta=1.0, type=1, quality=0.9)
    mC = Minutia(20.0, 30.0, theta=2.2, type=0, quality=0.85)
    
    desc_base, _ = extract_triangle_descriptor(mA, mB, mC)
    assert desc_base.shape == (12,)
    
    for p in [(mA, mB, mC), (mA, mC, mB), (mB, mA, mC), (mB, mC, mA), (mC, mA, mB), (mC, mB, mA)]:
        desc_perm, _ = extract_triangle_descriptor(p[0], p[1], p[2])
        np.testing.assert_allclose(desc_base, desc_perm, atol=1e-5)

def test_equilateral_triangle_tie_breaking():
    """
    Tests that equilateral triangles (all three edge lengths equal) resolve tie-breaking
    deterministically via lexicographical canonical selection.
    """
    s = 50.0
    h = s * np.sqrt(3.0) / 2.0
    m1 = Minutia(0.0, 0.0, theta=0.2, type=0, quality=0.8)
    m2 = Minutia(s, 0.0, theta=1.5, type=1, quality=0.9)
    m3 = Minutia(s / 2.0, h, theta=2.8, type=0, quality=0.85)
    
    desc_base, _ = extract_triangle_descriptor(m1, m2, m3)
    assert desc_base.shape == (12,)
    
    for p in [(m1, m2, m3), (m1, m3, m2), (m2, m1, m3), (m2, m3, m1), (m3, m1, m2), (m3, m2, m1)]:
        desc_perm, _ = extract_triangle_descriptor(p[0], p[1], p[2])
        np.testing.assert_allclose(desc_base, desc_perm, atol=1e-5)

def test_delaunay_and_knn_graphs():
    minutiae = [
        Minutia(10.0, 10.0, 0.1, 0, 0.9),
        Minutia(30.0, 15.0, 0.4, 1, 0.8),
        Minutia(20.0, 40.0, 0.8, 0, 0.7),
        Minutia(50.0, 45.0, 1.2, 1, 0.85),
        Minutia(40.0, 80.0, 1.5, 0, 0.9)
    ]
    
    delaunay_tris = build_delaunay_triangles(minutiae, max_edge_length=150.0)
    assert len(delaunay_tris) >= 1
    
    knn_tris = build_knn_graphs(minutiae, k=3, max_edge_length=150.0)
    assert len(knn_tris) >= 1
    
    descs, weights = extract_all_descriptors(delaunay_tris)
    assert descs.shape[0] == len(delaunay_tris)
    assert descs.shape[1] == 12
    assert len(weights) == len(delaunay_tris)

def test_nearly_isosceles_triangle_stability():
    """
    Tests that a triangle with a 1-pixel vertex jitter from perfect isosceles
    still produces a stable, permutation-invariant descriptor.
    """
    mA = Minutia(0.0, 0.0, theta=0.3, type=0, quality=0.8)
    mB = Minutia(40.0, 0.0, theta=1.0, type=1, quality=0.9)
    mC = Minutia(20.0, 30.0, theta=2.2, type=0, quality=0.85)
    
    desc_base, _ = extract_triangle_descriptor(mA, mB, mC)
    
    # Perturbed version: shift one vertex by ~1px
    mC_perturbed = Minutia(20.5, 30.3, theta=2.2, type=0, quality=0.85)
    desc_perturbed, _ = extract_triangle_descriptor(mA, mB, mC_perturbed)
    
    norm_diff = np.linalg.norm(desc_base - desc_perturbed) / np.sqrt(12.0)
    assert norm_diff < 0.05, f"Near-isosceles jitter produced large normalized change: {norm_diff}"
    
    # Permutation invariance of the perturbed triangle
    for p in [(mA, mB, mC_perturbed), (mA, mC_perturbed, mB), (mB, mA, mC_perturbed),
              (mB, mC_perturbed, mA), (mC_perturbed, mA, mB), (mC_perturbed, mB, mA)]:
        desc_perm, _ = extract_triangle_descriptor(p[0], p[1], p[2])
        np.testing.assert_allclose(desc_perturbed, desc_perm, atol=1e-5,
                                   err_msg="Near-isosceles perturbed triangle not permutation-invariant")

def test_nearly_equilateral_triangle_stability():
    """
    Tests that a triangle with a 1-2 pixel vertex jitter from perfect equilateral
    still produces a stable, permutation-invariant descriptor.
    """
    s = 50.0
    h = s * np.sqrt(3.0) / 2.0
    m1 = Minutia(0.0, 0.0, theta=0.2, type=0, quality=0.8)
    m2 = Minutia(s, 0.0, theta=1.5, type=1, quality=0.9)
    m3 = Minutia(s / 2.0, h, theta=2.8, type=0, quality=0.85)
    
    desc_base, _ = extract_triangle_descriptor(m1, m2, m3)
    
    # Perturbed equilateral: jitter by ~1.5px
    m3_perturbed = Minutia(s / 2.0 + 1.0, h - 0.8, theta=2.8, type=0, quality=0.85)
    desc_perturbed, _ = extract_triangle_descriptor(m1, m2, m3_perturbed)
    
    # Measure normalized descriptor distance delta_d = ||d1 - d2|| / sqrt(12)
    norm_diff = np.linalg.norm(desc_base - desc_perturbed) / np.sqrt(12.0)
    assert norm_diff < 0.05, f"Near-equilateral jitter produced large normalized change: {norm_diff}"
    
    # Permutation invariance
    for p in [(m1, m2, m3_perturbed), (m1, m3_perturbed, m2), (m2, m1, m3_perturbed),
              (m2, m3_perturbed, m1), (m3_perturbed, m1, m2), (m3_perturbed, m2, m1)]:
        desc_perm, _ = extract_triangle_descriptor(p[0], p[1], p[2])
        np.testing.assert_allclose(desc_perturbed, desc_perm, atol=1e-5,
                                   err_msg="Near-equilateral perturbed triangle not permutation-invariant")

def test_rotation_invariance_on_near_symmetric_triangles():
    """
    Tests that rotating a near-symmetric triangle by arbitrary angles
    produces identical 12-D descriptors, verifying that tie-breaking
    relies solely on rotation-invariant relative quantities.
    """
    mA = Minutia(0.0, 0.0, theta=0.3, type=0, quality=0.8)
    mB = Minutia(40.0, 0.0, theta=1.0, type=1, quality=0.9)
    mC = Minutia(20.5, 30.3, theta=2.2, type=0, quality=0.85)
    
    desc_orig, _ = extract_triangle_descriptor(mA, mB, mC)
    
    for angle in [0.35, np.pi / 4, 1.25, np.pi / 2, 2.4]:
        c, s = np.cos(angle), np.sin(angle)
        def rot_m(m):
            xr = m.x * c - m.y * s + 75.0
            yr = m.x * s + m.y * c - 45.0
            thr = (m.theta + angle) % np.pi
            return Minutia(float(xr), float(yr), float(thr), m.type, m.quality)
            
        mAr, mBr, mCr = rot_m(mA), rot_m(mB), rot_m(mC)
        desc_rot, _ = extract_triangle_descriptor(mAr, mBr, mCr)
        np.testing.assert_allclose(desc_orig, desc_rot, atol=1e-4,
                                   err_msg=f"Rotation by {angle} altered canonical descriptor!")

def test_geometry_only_tie_breaking_mode():
    """
    Tests that geometry-only canonicalization breaks ties without minutiae types,
    ensuring permutation invariance for both 12-D and 9-D descriptors.
    """
    mA = Minutia(0.0, 0.0, theta=0.4, type=0, quality=0.85)
    mB = Minutia(50.0, 0.0, theta=1.1, type=1, quality=0.90)
    mC = Minutia(25.0, 40.0, theta=2.3, type=0, quality=0.80)
    
    desc_12d, q_12d = extract_triangle_descriptor(mA, mB, mC, include_types=True, canonical_tie_breaker="geometry_only")
    assert desc_12d.shape == (12,)
    
    for p in [(mA, mB, mC), (mA, mC, mB), (mB, mA, mC), (mB, mC, mA), (mC, mA, mB), (mC, mB, mA)]:
        dp, qp = extract_triangle_descriptor(p[0], p[1], p[2], include_types=True, canonical_tie_breaker="geometry_only")
        np.testing.assert_allclose(desc_12d, dp, atol=1e-5)
        assert abs(q_12d - qp) < 1e-6

def test_pure_9d_descriptor_extraction():
    """
    Tests pure 9-D geometric descriptor extraction (excluding minutiae types),
    verifying dimension (9,) and exact correspondence with continuous features.
    """
    mA = Minutia(10.0, 10.0, theta=0.5, type=0, quality=0.8)
    mB = Minutia(40.0, 10.0, theta=1.2, type=1, quality=0.9)
    mC = Minutia(25.0, 35.0, theta=2.1, type=0, quality=0.85)
    
    d12, q12 = extract_triangle_descriptor(mA, mB, mC, include_types=True, canonical_tie_breaker="geometry_only")
    d9, q9 = extract_triangle_descriptor(mA, mB, mC, include_types=False, canonical_tie_breaker="geometry_only")
    
    assert d9.shape == (9,)
    assert d12.shape == (12,)
    np.testing.assert_allclose(d9, d12[:9], atol=1e-6)
    assert abs(q9 - q12) < 1e-6

def test_ordered_minutiae_tracking():
    """
    Tests extract_all_descriptors_with_minutiae for tracking unique minutiae references.
    """
    from src.graph.descriptors import extract_all_descriptors_with_minutiae
    from src.graph.delaunay import build_delaunay_triangles
    
    minutiae = [
        Minutia(10.0, 10.0, 0.1, 0, 0.9),
        Minutia(30.0, 15.0, 0.4, 1, 0.8),
        Minutia(20.0, 40.0, 0.8, 0, 0.7),
        Minutia(50.0, 45.0, 1.2, 1, 0.85)
    ]
    tris = build_delaunay_triangles(minutiae)
    descs, weights, ordered_m = extract_all_descriptors_with_minutiae(tris, include_types=True)
    
    assert len(descs) == len(tris)
    assert len(ordered_m) == len(tris)
    for i in range(len(tris)):
        assert len(ordered_m[i]) == 3
        # Minutiae types in descriptor must match the ordered minutiae types
        assert descs[i, 9] == float(ordered_m[i][0].type)
        assert descs[i, 10] == float(ordered_m[i][1].type)
        assert descs[i, 11] == float(ordered_m[i][2].type)

def test_cyclic_triangle_descriptors():
    """
    Tests extract_all_cyclic_descriptors for producing (N, 3, D) representations
    with planar counter-clockwise orientation consistency.
    """
    from src.graph.descriptors import extract_all_cyclic_descriptors
    
    mA = Minutia(10.0, 10.0, theta=0.5, type=0, quality=0.8)
    mB = Minutia(40.0, 10.0, theta=1.2, type=1, quality=0.9)
    mC = Minutia(25.0, 35.0, theta=2.1, type=0, quality=0.85)
    
    # Counter-clockwise input
    tris_ccw = [(mA, mB, mC)]
    descs_9d, w_9d = extract_all_cyclic_descriptors(tris_ccw, include_types=False)
    descs_12d, w_12d = extract_all_cyclic_descriptors(tris_ccw, include_types=True)
    
    assert descs_9d.shape == (1, 3, 9)
    assert descs_12d.shape == (1, 3, 12)
    assert w_9d.shape == (1,)
    assert abs(w_9d[0] - w_12d[0]) < 1e-6
    
    # Clockwise input (swapped mB and mC)
    tris_cw = [(mA, mC, mB)]
    descs_cw, w_cw = extract_all_cyclic_descriptors(tris_cw, include_types=False)
    
    # After CCW normalization, the set of 3 cyclic descriptors must match
    # (one permutation in descs_cw matches one in descs_9d)
    matched = 0
    for i in range(3):
        for j in range(3):
            if np.allclose(descs_9d[0, i], descs_cw[0, j], atol=1e-5):
                matched += 1
                break
    assert matched == 3, f"Expected all 3 cyclic shifts to match across orientations, matched {matched}"

def test_cyclic_matching_rigid_and_non_rigid_transformations():
    """
    Formally validates cyclic matching under:
    1. Rigid 2D Planar Rotations (30 deg, 45 deg, 90 deg, 180 deg) -> distance = 0.0.
    2. 2D Translations -> distance = 0.0.
    3. Coordinate perturbations (noise) -> bounded continuous distance response.
    4. Mirror reflections (chirality inversion) -> distinguished from planar rotations.
    """
    from src.graph.descriptors import extract_all_cyclic_descriptors
    from src.protection.kdf import generate_master_secret
    from src.protection.projection import derive_orthonormal_matrix, project_descriptors
    from src.protection.quantizer import quantize_descriptors
    from src.matching.bipartite_matcher import match_protected_templates
    
    # Base triangle
    mA = Minutia(15.0, 20.0, theta=0.4, type=0, quality=0.9)
    mB = Minutia(55.0, 25.0, theta=1.1, type=1, quality=0.85)
    mC = Minutia(30.0, 65.0, theta=2.3, type=0, quality=0.8)
    base_tri = [(mA, mB, mC)]
    
    # Master secret and projection matrix for 9-D
    master = "master_secret_fvc2002_secure_32b_!"
    W = derive_orthonormal_matrix(master, "cyclic_val_salt", "cyclic_app", in_dim=9, out_dim=32)
    
    base_cyclic_descs, base_wts = extract_all_cyclic_descriptors(base_tri, include_types=False)
    # Shape: (1, 3, 9) -> project all 3 shifts
    base_proj = project_descriptors(base_cyclic_descs.reshape(-1, 9), W)
    base_bits_cyclic = quantize_descriptors(base_proj).reshape(1, 3, 32)
    
    # 1. Rigid 2D Planar Rotations: 30°, 45°, 90°, 180°
    rotation_angles = [np.pi / 6.0, np.pi / 4.0, np.pi / 2.0, np.pi]
    for ang in rotation_angles:
        cos_a = np.cos(ang)
        sin_a = np.sin(ang)
        
        def rotate_minutia(m):
            # Center of rotation at (30, 30)
            cx, cy = 30.0, 30.0
            rx = cos_a * (m.x - cx) - sin_a * (m.y - cy) + cx
            ry = sin_a * (m.x - cx) + cos_a * (m.y - cy) + cy
            r_theta = (m.theta + ang) % np.pi
            return Minutia(float(rx), float(ry), float(r_theta), m.type, m.quality)
            
        rot_tri = [(rotate_minutia(mA), rotate_minutia(mB), rotate_minutia(mC))]
        rot_cyclic_descs, rot_wts = extract_all_cyclic_descriptors(rot_tri, include_types=False)
        rot_proj = project_descriptors(rot_cyclic_descs[:, 0, :], W)  # query uses shift 0
        rot_bits_query = quantize_descriptors(rot_proj)
        
        # Match rot_bits_query against enrolled base_bits_cyclic
        s = match_protected_templates(rot_bits_query, base_bits_cyclic, rot_wts, base_wts, hamming_threshold=0.22)
        assert s == pytest.approx(1.0, abs=1e-4), f"Rotation by {np.degrees(ang):.0f} deg failed: score={s}"
        
    # 2. 2D Translations: shift by (120.0, -85.0)
    tx, ty = 120.0, -85.0
    trans_tri = [(
        Minutia(mA.x + tx, mA.y + ty, mA.theta, mA.type, mA.quality),
        Minutia(mB.x + tx, mB.y + ty, mB.theta, mB.type, mB.quality),
        Minutia(mC.x + tx, mC.y + ty, mC.theta, mC.type, mC.quality)
    )]
    trans_cyclic_descs, trans_wts = extract_all_cyclic_descriptors(trans_tri, include_types=False)
    trans_bits_query = quantize_descriptors(project_descriptors(trans_cyclic_descs[:, 0, :], W))
    s_trans = match_protected_templates(trans_bits_query, base_bits_cyclic, trans_wts, base_wts, hamming_threshold=0.22)
    assert s_trans == pytest.approx(1.0, abs=1e-4)
    
    # 3. Continuous Coordinate Perturbation (1.5 px noise)
    noise_tri = [(
        Minutia(mA.x + 1.2, mA.y - 0.8, (mA.theta + 0.05) % np.pi, mA.type, mA.quality),
        Minutia(mB.x - 1.0, mB.y + 0.5, (mB.theta - 0.04) % np.pi, mB.type, mB.quality),
        Minutia(mC.x + 0.5, mC.y + 1.1, (mC.theta + 0.03) % np.pi, mC.type, mC.quality)
    )]
    noise_cyclic_descs, noise_wts = extract_all_cyclic_descriptors(noise_tri, include_types=False)
    noise_bits_query = quantize_descriptors(project_descriptors(noise_cyclic_descs[:, 0, :], W))
    s_noise = match_protected_templates(noise_bits_query, base_bits_cyclic, noise_wts, base_wts, hamming_threshold=0.22)
    # Small noise must still match with significant similarity (> 0.50)
    assert s_noise > 0.50
    
    # 4. Mirror Reflection across X-axis: (x, -y), theta -> (-theta % pi)
    refl_tri = [(
        Minutia(mA.x, -mA.y, (-mA.theta) % np.pi, mA.type, mA.quality),
        Minutia(mB.x, -mB.y, (-mB.theta) % np.pi, mB.type, mB.quality),
        Minutia(mC.x, -mC.y, (-mC.theta) % np.pi, mC.type, mC.quality)
    )]
    refl_cyclic_descs, refl_wts = extract_all_cyclic_descriptors(refl_tri, include_types=False)
    # Reflection flips triangle chirality (orientation sign).
    # Planar internal CCW angles differ from reflected triangle.
    diffs = np.min([np.mean(np.abs(refl_cyclic_descs[0, 0] - base_cyclic_descs[0, k])) for k in range(3)])
    assert diffs > 0.01  # Chirality inversion detected


