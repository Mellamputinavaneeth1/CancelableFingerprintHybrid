import itertools
from typing import List, Tuple, Optional
import numpy as np
from src.minutiae.extractor import Minutia

# Empirical reference mean and standard deviation for 12-D invariant Delaunay descriptor.
# Features:
# 0: r1 = L2 / L1 (shape ratio 1)
# 1: r2 = L3 / L1 (shape ratio 2)
# 2: P_mm = (L1 + L2 + L3) * 25.4 / dpi (physical sensor-calibrated perimeter in mm)
# 3: phi0 (axial angle minutia 0 to directed edge 0-1)
# 4: phi1 (axial angle minutia 1 to directed edge 1-2)
# 5: phi2 (axial angle minutia 2 to directed edge 2-0)
# 6: dth01 (axial mutual orientation difference 0-1)
# 7: dth12 (axial mutual orientation difference 1-2)
# 8: dth20 (axial mutual orientation difference 2-0)
# 9: t0 (minutia type 0: ending=0.0, bifurcation=1.0)
# 10: t1 (minutia type 1)
# 11: t2 (minutia type 2)
# Note: Quality Q_Delta is strictly separated from the identity descriptor and used as a matching weight.
# Note: Circularity C_Delta is removed as it is mathematically redundant with (r1, r2).

DEFAULT_DESCRIPTOR_MEAN = np.array([
    0.720,  # r1 = L2 / L1
    0.480,  # r2 = L3 / L1
    7.500,  # P_mm (physical perimeter in mm)
    0.500,  # phi0
    0.500,  # phi1
    0.500,  # phi2
    0.500,  # dth01
    0.500,  # dth12
    0.500,  # dth20
    0.500,  # t0
    0.500,  # t1
    0.500   # t2
], dtype=np.float32)

DEFAULT_DESCRIPTOR_STD = np.array([
    0.180,  # r1 std
    0.200,  # r2 std
    2.500,  # P_mm std
    0.280,  # phi0 std
    0.280,  # phi1 std
    0.280,  # phi2 std
    0.280,  # dth01 std
    0.280,  # dth12 std
    0.280,  # dth20 std
    0.500,  # t0 std
    0.500,  # t1 std
    0.500   # t2 std
], dtype=np.float32)

# Pure 9-D geometric reference vectors (excluding discrete minutiae types)
DEFAULT_DESCRIPTOR_MEAN_9D = DEFAULT_DESCRIPTOR_MEAN[:9].copy()
DEFAULT_DESCRIPTOR_STD_9D = DEFAULT_DESCRIPTOR_STD[:9].copy()

# Non-redundant 8-D geometric reference vectors (dropping redundant mutual orientation dth20)
DEFAULT_DESCRIPTOR_MEAN_8D = DEFAULT_DESCRIPTOR_MEAN[:8].copy()
DEFAULT_DESCRIPTOR_STD_8D = DEFAULT_DESCRIPTOR_STD[:8].copy()

# Minimal non-redundant 6-D geometric manifold vectors: (r1, r2, P_mm, phi0, phi1, phi2)
DEFAULT_DESCRIPTOR_MEAN_6D = DEFAULT_DESCRIPTOR_MEAN[:6].copy()
DEFAULT_DESCRIPTOR_STD_6D = DEFAULT_DESCRIPTOR_STD[:6].copy()

# Reduced 5-D geometric reference vectors: (L1, L2, dth01, dth12, dth20)
# L1 = l1 / (l0 + l1 + l2), L2 = l2 / (l0 + l1 + l2)
DEFAULT_DESCRIPTOR_MEAN_5D = np.array([0.350, 0.280, 0.500, 0.500, 0.500], dtype=np.float32)
DEFAULT_DESCRIPTOR_STD_5D = np.array([0.080, 0.080, 0.280, 0.280, 0.280], dtype=np.float32)

def _axial_diff(th1: float, th2: float) -> float:
    """
    Computes axial angular difference between two orientations modulo pi,
    normalized strictly to [0, 1].
    
    Robust to atan2 angles in [-pi, pi] and orientations in [0, pi):
        d = abs((th1 - th2) % pi)
        d_axial = min(d, pi - d)  # in [0, pi/2]
        return d_axial / (pi / 2)  # in [0.0, 1.0]
    """
    d = abs((th1 - th2) % np.pi)
    d_axial = min(d, np.pi - d)
    return float(d_axial / (0.5 * np.pi))

def _compute_ordered_descriptor(m_ordered: List[Minutia], pts_ordered: np.ndarray, 
                                dpi: float = 500.0,
                                include_types: bool = True) -> np.ndarray:
    """
    Computes an invariant identity descriptor for a specific vertex ordering (v0, v1, v2).
    Returns 12-D if include_types=True, or 9-D (pure geometric) if include_types=False.

    Edge Orientation Convention:
        phi_0: axial angle between minutia 0's ridge orientation and directed edge v0->v1
        phi_1: axial angle between minutia 1's ridge orientation and directed edge v1->v2
        phi_2: axial angle between minutia 2's ridge orientation and directed edge v2->v0
    
        dth01: axial mutual orientation difference between minutiae 0 and 1
        dth12: axial mutual orientation difference between minutiae 1 and 2
        dth20: axial mutual orientation difference between minutiae 2 and 0
    """
    p0, p1, p2 = pts_ordered[0], pts_ordered[1], pts_ordered[2]
    
    # Opposite edges:
    # L0 opposite v0 (between v1 and v2)
    # L1 opposite v1 (between v2 and v0)
    # L2 opposite v2 (between v0 and v1)
    l0 = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
    l1 = float(np.hypot(p2[0] - p0[0], p2[1] - p0[1]))
    l2 = float(np.hypot(p0[0] - p1[0], p0[1] - p1[1]))
    
    l_max = max(l0, l1, l2) + 1e-8
    r1 = l1 / l_max
    r2 = l2 / l_max
    
    # Physical sensor-calibrated perimeter in millimeters
    # 1 pixel = 25.4 / dpi mm
    perimeter_px = l0 + l1 + l2
    perimeter_mm = perimeter_px * (25.4 / dpi)
    
    # Directed edge orientations from atan2 in [-pi, pi]
    ang01 = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
    ang12 = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    ang20 = np.arctan2(p0[1] - p2[1], p0[0] - p2[0])
    
    # Axial minutiae relative orientations to directed edges
    phi0 = _axial_diff(m_ordered[0].theta, ang01)
    phi1 = _axial_diff(m_ordered[1].theta, ang12)
    phi2 = _axial_diff(m_ordered[2].theta, ang20)
    
    # Mutual minutiae orientation differences (axial)
    dth01 = _axial_diff(m_ordered[0].theta, m_ordered[1].theta)
    dth12 = _axial_diff(m_ordered[1].theta, m_ordered[2].theta)
    dth20 = _axial_diff(m_ordered[2].theta, m_ordered[0].theta)
    
    geom_features = [
        r1, r2, perimeter_mm,
        phi0, phi1, phi2,
        dth01, dth12, dth20
    ]
    
    if not include_types:
        return np.array(geom_features, dtype=np.float32)
        
    # Minutiae types (0.0: Ending, 1.0: Bifurcation)
    t0 = float(m_ordered[0].type)
    t1 = float(m_ordered[1].type)
    t2 = float(m_ordered[2].type)
    
    return np.array(geom_features + [t0, t1, t2], dtype=np.float32)

def extract_triangle_descriptor(m1: Minutia, m2: Minutia, m3: Minutia, 
                                dpi: float = 500.0,
                                eps_rel: float = 0.03,
                                tol: Optional[float] = None,
                                include_types: bool = True,
                                canonical_tie_breaker: str = "geometry_only",
                                desc_dim: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """
    Extracts a non-redundant, sensor-calibrated Delaunay triangle descriptor.
    
    Canonical Ordering & Robust Tie-Breaking:
    -----------------------------------------
    1. Evaluates all 6 vertex permutations of (v0, v1, v2).
    2. Determines descending edge lengths (E_max >= E_mid >= E_min) and filters permutations
       whose opposite edges match the descending order within relative tolerance tol = eps_rel * E_max.
    3. Any remaining ties within `tol` are resolved using the specified tie-breaker hierarchy:
       - 'geometry_only' (Default): Tie-breaking strictly relies on relative orientations (dth),
         edge differences (phi), side length ratios (r), and full geometric features. Minutiae types
         are NEVER used in sorting, completely decoupling canonicalization from minutiae type distributions.
       - 'types_first': Legacy ordering placing minutiae types (t0, t1, t2) at the head of the tie key.
       
    Args:
        m1, m2, m3: Minutia objects forming the triangle.
        dpi: Sensor resolution in dots per inch for physical scale normalization (default: 500.0).
        eps_rel: Relative tolerance for edge length tie detection (default: 0.03, ~3%).
        tol: Optional explicit absolute tolerance override.
        include_types: Whether to include 3 minutiae type features (12-D) or exclude them (9-D).
        canonical_tie_breaker: Tie-breaking mode ('geometry_only' or 'types_first').
        desc_dim: Feature dimension (9 for standard pure geometry, 5 for reduced, 12 for with types).
        
    Returns:
        descriptor: np.ndarray of shape (12,), (9,), or (5,), float32.
        quality: float composite quality weight Q_Delta in [0, 1].
    """
    desc, q, _ = extract_triangle_descriptor_with_ordered_minutiae(
        m1, m2, m3, dpi=dpi, eps_rel=eps_rel, tol=tol,
        include_types=include_types, canonical_tie_breaker=canonical_tie_breaker,
        desc_dim=desc_dim
    )
    return desc, q

def extract_triangle_descriptor_with_ordered_minutiae(m1: Minutia, m2: Minutia, m3: Minutia, 
                                                      dpi: float = 500.0,
                                                      eps_rel: float = 0.03,
                                                      tol: Optional[float] = None,
                                                      include_types: bool = True,
                                                      canonical_tie_breaker: str = "geometry_only",
                                                      desc_dim: Optional[int] = None) -> Tuple[np.ndarray, float, List[Minutia]]:
    """
    Extracts canonical descriptor, quality, and the ordered Minutia references (v0, v1, v2).
    Facilitates unique-minutia level recovery tracking across shared Delaunay vertices.
    Supports desc_dim=5 for reduced geometric parameterization.
    """
    minutiae = [m1, m2, m3]
    pts = np.array([[m.x, m.y] for m in minutiae], dtype=np.float32)
    
    # Calculate sorted edge lengths
    all_edges = [
        float(np.hypot(pts[1, 0] - pts[2, 0], pts[1, 1] - pts[2, 1])),
        float(np.hypot(pts[2, 0] - pts[0, 0], pts[2, 1] - pts[0, 1])),
        float(np.hypot(pts[0, 0] - pts[1, 0], pts[0, 1] - pts[1, 1]))
    ]
    all_edges.sort(reverse=True)
    e_max, e_mid, e_min = all_edges[0], all_edges[1], all_edges[2]
    
    effective_tol = tol if tol is not None else (eps_rel * max(e_max, 1.0))
    
    perm_candidates = []
    for perm in itertools.permutations([0, 1, 2]):
        p_perm = pts[list(perm)]
        m_perm = [minutiae[i] for i in perm]
        
        l0 = float(np.hypot(p_perm[1, 0] - p_perm[2, 0], p_perm[1, 1] - p_perm[2, 1]))
        l1 = float(np.hypot(p_perm[2, 0] - p_perm[0, 0], p_perm[2, 1] - p_perm[0, 1]))
        l2 = float(np.hypot(p_perm[0, 0] - p_perm[1, 0], p_perm[0, 1] - p_perm[1, 1]))
        
        perm_candidates.append({
            "perm": perm,
            "edges": (l0, l1, l2),
            "m_ordered": m_perm,
            "pts_ordered": p_perm
        })
        
    valid_candidates = []
    for c in perm_candidates:
        l0, l1, l2 = c["edges"]
        if (abs(l0 - e_max) <= effective_tol and 
            abs(l1 - e_mid) <= effective_tol and 
            abs(l2 - e_min) <= effective_tol):
            valid_candidates.append(c)
            
    if not valid_candidates:
        valid_candidates = perm_candidates
        
    candidates_with_keys = []
    for c in valid_candidates:
        d_full = _compute_ordered_descriptor(c["m_ordered"], c["pts_ordered"], dpi=dpi, include_types=True)
        mutual_diff_key = tuple(np.round(d_full[6:9], decimals=3))
        edge_diff_key = tuple(np.round(d_full[3:6], decimals=3))
        ratios_key = tuple(np.round(d_full[0:2], decimals=3))
        full_geom_key = tuple(np.round(d_full[:9], decimals=4))
        
        # Planar chirality: 2D signed cross-product (invariant under rigid SO(2) rotation and translation)
        # Prefers counter-clockwise orientation (cross >= 0) to deterministically break reflection/orientation ties
        cross = float((c["pts_ordered"][1, 0] - c["pts_ordered"][0, 0]) * (c["pts_ordered"][2, 1] - c["pts_ordered"][0, 1]) -
                      (c["pts_ordered"][1, 1] - c["pts_ordered"][0, 1]) * (c["pts_ordered"][2, 0] - c["pts_ordered"][0, 0]))
        chirality_key = 0 if cross >= 0 else 1
        quality_key = tuple(-round(float(m.quality), 4) for m in c["m_ordered"])
        type_key = tuple(int(m.type) for m in c["m_ordered"])
        unrounded_geom_key = tuple(float(x) for x in d_full[:9])

        if canonical_tie_breaker == "types_first":
            types_key = (float(d_full[9]), float(d_full[10]), float(d_full[11]))
            tie_key = (types_key, mutual_diff_key, edge_diff_key, ratios_key, full_geom_key, chirality_key, unrounded_geom_key, quality_key)
        else:
            # geometry_only (Default): Strictly geometric keys, with chirality, unrounded geometry, quality, and type as hierarchical tie-breakers
            tie_key = (mutual_diff_key, edge_diff_key, ratios_key, full_geom_key, chirality_key, unrounded_geom_key, quality_key, type_key)
            
        target_dim = desc_dim if desc_dim is not None else (12 if include_types else 9)
        if target_dim == 5:
            # Option A: 2 normalized side length ratios (L1, L2) + 3 mutual orientation differences
            p_sum = c["edges"][0] + c["edges"][1] + c["edges"][2] + 1e-8
            L1 = float(c["edges"][1] / p_sum)
            L2 = float(c["edges"][2] / p_sum)
            final_desc = np.array([L1, L2, d_full[6], d_full[7], d_full[8]], dtype=np.float32)
        elif target_dim == 6:
            # 6-D candidate: Minimal non-redundant manifold [r1, r2, P_mm, phi0, phi1, phi2]
            final_desc = d_full[:6]
        elif target_dim == 8:
            # 8-D candidate: Non-redundant mutual angles [r1, r2, P_mm, phi0, phi1, phi2, dth01, dth12]
            final_desc = d_full[:8]
        elif target_dim == 9:
            final_desc = d_full[:9]
        else:
            final_desc = d_full
            
        candidates_with_keys.append((tie_key, final_desc, c["m_ordered"]))
        
    candidates_with_keys.sort(key=lambda item: item[0])
    canonical_descriptor = candidates_with_keys[0][1]
    canonical_ordered_minutiae = candidates_with_keys[0][2]
    
    # Composite quality weight: geometric mean of the three minutia confidences
    q_comp = float((m1.quality * m2.quality * m3.quality) ** (1.0 / 3.0))
    return canonical_descriptor, q_comp, canonical_ordered_minutiae

def extract_triangle_descriptor_5d(m1: Minutia, m2: Minutia, m3: Minutia, 
                                   dpi: float = 500.0,
                                   eps_rel: float = 0.03,
                                   tol: Optional[float] = None,
                                   canonical_tie_breaker: str = "geometry_only") -> Tuple[np.ndarray, float]:
    """
    Extracts a reduced 5-D geometric descriptor: [L1, L2, dth01, dth12, dth20].
    """
    desc, q, _ = extract_triangle_descriptor_with_ordered_minutiae(
        m1, m2, m3, dpi=dpi, eps_rel=eps_rel, tol=tol,
        include_types=False, canonical_tie_breaker=canonical_tie_breaker, desc_dim=5
    )
    return desc, q

def extract_all_descriptors_5d(triangles: List[Tuple[Minutia, Minutia, Minutia]], 
                               dpi: float = 500.0,
                               canonical_tie_breaker: str = "geometry_only") -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts reduced 5-D geometric descriptors for all triangles. Returns (N, 5).
    """
    if not triangles:
        return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=np.float32)
        
    desc_list = []
    qual_list = []
    for tri in triangles:
        d, q = extract_triangle_descriptor_5d(
            tri[0], tri[1], tri[2], dpi=dpi, canonical_tie_breaker=canonical_tie_breaker
        )
        desc_list.append(d)
        qual_list.append(q)
        
    return np.array(desc_list, dtype=np.float32), np.array(qual_list, dtype=np.float32)

def extract_triangle_descriptor_6d(m1: Minutia, m2: Minutia, m3: Minutia, 
                                   dpi: float = 500.0,
                                   eps_rel: float = 0.03,
                                   tol: Optional[float] = None,
                                   canonical_tie_breaker: str = "geometry_only") -> Tuple[np.ndarray, float]:
    """
    Extracts minimal non-redundant 6-D geometric manifold descriptor:
    [r1, r2, P_mm, phi0, phi1, phi2].
    """
    desc, q, _ = extract_triangle_descriptor_with_ordered_minutiae(
        m1, m2, m3, dpi=dpi, eps_rel=eps_rel, tol=tol,
        include_types=False, canonical_tie_breaker=canonical_tie_breaker, desc_dim=6
    )
    return desc, q

def extract_all_descriptors_6d(triangles: List[Tuple[Minutia, Minutia, Minutia]], 
                               dpi: float = 500.0,
                               canonical_tie_breaker: str = "geometry_only") -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts minimal non-redundant 6-D geometric descriptors for all triangles. Returns (N, 6).
    """
    if not triangles:
        return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.float32)
        
    desc_list = []
    qual_list = []
    for tri in triangles:
        d, q = extract_triangle_descriptor_6d(
            tri[0], tri[1], tri[2], dpi=dpi, canonical_tie_breaker=canonical_tie_breaker
        )
        desc_list.append(d)
        qual_list.append(q)
        
    return np.array(desc_list, dtype=np.float32), np.array(qual_list, dtype=np.float32)

def extract_triangle_descriptor_8d(m1: Minutia, m2: Minutia, m3: Minutia, 
                                   dpi: float = 500.0,
                                   eps_rel: float = 0.03,
                                   tol: Optional[float] = None,
                                   canonical_tie_breaker: str = "geometry_only") -> Tuple[np.ndarray, float]:
    """
    Extracts non-redundant 8-D geometric descriptor (dropping redundant cyclic angle dth20):
    [r1, r2, P_mm, phi0, phi1, phi2, dth01, dth12].
    """
    desc, q, _ = extract_triangle_descriptor_with_ordered_minutiae(
        m1, m2, m3, dpi=dpi, eps_rel=eps_rel, tol=tol,
        include_types=False, canonical_tie_breaker=canonical_tie_breaker, desc_dim=8
    )
    return desc, q

def extract_all_descriptors_8d(triangles: List[Tuple[Minutia, Minutia, Minutia]], 
                               dpi: float = 500.0,
                               canonical_tie_breaker: str = "geometry_only") -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts non-redundant 8-D geometric descriptors for all triangles. Returns (N, 8).
    """
    if not triangles:
        return np.empty((0, 8), dtype=np.float32), np.empty((0,), dtype=np.float32)
        
    desc_list = []
    qual_list = []
    for tri in triangles:
        d, q = extract_triangle_descriptor_8d(
            tri[0], tri[1], tri[2], dpi=dpi, canonical_tie_breaker=canonical_tie_breaker
        )
        desc_list.append(d)
        qual_list.append(q)
        
    return np.array(desc_list, dtype=np.float32), np.array(qual_list, dtype=np.float32)

def extract_all_descriptors(triangles: List[Tuple[Minutia, Minutia, Minutia]], 
                            dpi: float = 500.0,
                            include_types: bool = True,
                            canonical_tie_breaker: str = "geometry_only",
                            desc_dim: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts canonical descriptors and composite quality weights for all triangles.
    Returns array of shape (N, 12), (N, 9), (N, 8), (N, 6), or (N, 5).
    """
    target_dim = desc_dim if desc_dim is not None else (12 if include_types else 9)
    out_dim = target_dim

    if not triangles:
        return np.empty((0, out_dim), dtype=np.float32), np.empty((0,), dtype=np.float32)
        
    desc_list = []
    qual_list = []
    for tri in triangles:
        d, q = extract_triangle_descriptor(
            tri[0], tri[1], tri[2], dpi=dpi,
            include_types=include_types, canonical_tie_breaker=canonical_tie_breaker,
            desc_dim=desc_dim
        )
        desc_list.append(d)
        qual_list.append(q)
        
    return np.array(desc_list, dtype=np.float32), np.array(qual_list, dtype=np.float32)

def extract_all_descriptors_with_minutiae(triangles: List[Tuple[Minutia, Minutia, Minutia]], 
                                         dpi: float = 500.0,
                                         include_types: bool = True,
                                         canonical_tie_breaker: str = "geometry_only") -> Tuple[np.ndarray, np.ndarray, List[List[Minutia]]]:
    """
    Extracts descriptors, weights, and lists of ordered Minutia references for each triangle.
    """
    out_dim = 12 if include_types else 9
    if not triangles:
        return np.empty((0, out_dim), dtype=np.float32), np.empty((0,), dtype=np.float32), []
        
    desc_list = []
    qual_list = []
    ordered_minutiae_list = []
    for tri in triangles:
        d, q, o_m = extract_triangle_descriptor_with_ordered_minutiae(
            tri[0], tri[1], tri[2], dpi=dpi,
            include_types=include_types, canonical_tie_breaker=canonical_tie_breaker
        )
        desc_list.append(d)
        qual_list.append(q)
        ordered_minutiae_list.append(o_m)
        
    return np.array(desc_list, dtype=np.float32), np.array(qual_list, dtype=np.float32), ordered_minutiae_list

def extract_all_cyclic_descriptors(triangles: List[Tuple[Minutia, Minutia, Minutia]], 
                                   dpi: float = 500.0,
                                   include_types: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts planar counter-clockwise cyclic descriptors for all triangles.
    For each triangle, evaluates the cyclic subgroup C_3 = {(0,1,2), (1,2,0), (2,0,1)}
    after enforcing counter-clockwise planar orientation (cross product >= 0).
    
    Returns:
        cyclic_descriptors: np.ndarray of shape (N, 3, D) where D is 9 (if include_types=False)
                            or 12 (if include_types=True).
        weights: np.ndarray of shape (N,) containing composite quality weights Q_Delta.
    """
    out_dim = 12 if include_types else 9
    if not triangles:
        return np.empty((0, 3, out_dim), dtype=np.float32), np.empty((0,), dtype=np.float32)
        
    cyclic_descs = []
    qual_list = []
    
    for tri in triangles:
        m0, m1, m2 = tri[0], tri[1], tri[2]
        pts = np.array([[m0.x, m0.y], [m1.x, m1.y], [m2.x, m2.y]], dtype=np.float32)
        minutiae = [m0, m1, m2]
        
        # 2D cross product: (p1 - p0) x (p2 - p0)
        cross_prod = (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1]) - (pts[1, 1] - pts[0, 1]) * (pts[2, 0] - pts[0, 0])
        if cross_prod < 0:
            # Clockwise: swap vertices 1 and 2 to enforce counter-clockwise orientation
            pts = pts[[0, 2, 1]]
            minutiae = [minutiae[0], minutiae[2], minutiae[1]]
            
        # The 3 counter-clockwise cyclic permutations C_3:
        # (0, 1, 2), (1, 2, 0), (2, 0, 1)
        shifts = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        tri_shifts = []
        for s in shifts:
            m_shift = [minutiae[s[0]], minutiae[s[1]], minutiae[s[2]]]
            p_shift = pts[list(s)]
            d = _compute_ordered_descriptor(m_shift, p_shift, dpi=dpi, include_types=include_types)
            tri_shifts.append(d)
            
        cyclic_descs.append(np.stack(tri_shifts, axis=0))  # shape: (3, D)
        q_comp = float((m0.quality * m1.quality * m2.quality) ** (1.0 / 3.0))
        qual_list.append(q_comp)
        
    return np.array(cyclic_descs, dtype=np.float32), np.array(qual_list, dtype=np.float32)

