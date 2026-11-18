"""
Pre-image Inversion Defense & Affine Coordinate Geometry (P0.5, P1.1).

Defines the exact affine transformation mapping raw physical descriptor space R^D
to standardized projection space R^D, formulates the bounded convex relaxation of
the descriptor manifold, and solves the Chebyshev center LP in standardized coordinates.

Affine Mapping:
    f_tilde = S^{-1} (f - mu_dev),   where S = diag(sigma_dev)
    f       = S f_tilde + mu_dev

Bounded Convex Relaxation Semantics:
    The domain defined by physical box bounds f in [f_min, f_max] combined with
    descending side-ratio ordering f_1 <= f_0 (r_2 <= r_1) forms a bounded convex
    polytope relaxation. It captures coordinate ranges and side length sorting,
    but does not enforce non-linear triangle Euclidean metric embeddability
    (e.g., triangle inequalities on side lengths L1, L2, L3 or exact angle closure).
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy.optimize import linprog

from src.graph.descriptors import (
    DEFAULT_DESCRIPTOR_MEAN_9D, DEFAULT_DESCRIPTOR_STD_9D,
    DEFAULT_DESCRIPTOR_MEAN, DEFAULT_DESCRIPTOR_STD
)

# Physical bounds for 9-D continuous Delaunay triangle descriptors
RAW_PHYSICAL_MIN_9D = np.array([0.1, 0.1, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
RAW_PHYSICAL_MAX_9D = np.array([1.0, 1.0, 30.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

RAW_PHYSICAL_MIN_12D = np.array([0.1, 0.1, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
RAW_PHYSICAL_MAX_12D = np.array([1.0, 1.0, 30.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)


def physical_to_standardized(f: np.ndarray,
                             mean: Optional[np.ndarray] = None,
                             std: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Applies affine mapping z_tilde = S^{-1} (f - mu).
    """
    f_arr = np.asarray(f, dtype=np.float64)
    d = f_arr.shape[-1]
    ref_mean = mean if mean is not None else (DEFAULT_DESCRIPTOR_MEAN_9D if d == 9 else DEFAULT_DESCRIPTOR_MEAN)
    ref_std = std if std is not None else (DEFAULT_DESCRIPTOR_STD_9D if d == 9 else DEFAULT_DESCRIPTOR_STD)
    return (f_arr - ref_mean) / (ref_std + 1e-8)


def standardized_to_physical(z_tilde: np.ndarray,
                             mean: Optional[np.ndarray] = None,
                             std: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Applies inverse affine mapping f = S z_tilde + mu.
    """
    z_arr = np.asarray(z_tilde, dtype=np.float64)
    d = z_arr.shape[-1]
    ref_mean = mean if mean is not None else (DEFAULT_DESCRIPTOR_MEAN_9D if d == 9 else DEFAULT_DESCRIPTOR_MEAN)
    ref_std = std if std is not None else (DEFAULT_DESCRIPTOR_STD_9D if d == 9 else DEFAULT_DESCRIPTOR_STD)
    return z_arr * (ref_std + 1e-8) + ref_mean


def get_standardized_relaxation_constraints(d: int = 9,
                                            mean: Optional[np.ndarray] = None,
                                            std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms raw physical bounds and descending ratio constraint r2 <= r1
    into standardized coordinate space z_tilde in R^d.
    
    Raw constraints:
        f_min <= f <= f_max
        f_1 - f_0 <= 0   (i.e. r_2 <= r_1)
        
    Standardized constraints:
        z_min <= z_tilde <= z_max
        sigma_1 * z_tilde_1 - sigma_0 * z_tilde_0 <= mu_0 - mu_1
        
    Returns:
        A_ratio: (1, d) ratio inequality row
        b_ratio: (1,) ratio RHS scalar
        z_min: (d,) lower coordinate bounds in standardized space
        z_max: (d,) upper coordinate bounds in standardized space
    """
    ref_mean = np.asarray(mean if mean is not None else (DEFAULT_DESCRIPTOR_MEAN_9D if d == 9 else DEFAULT_DESCRIPTOR_MEAN), dtype=np.float64)
    ref_std = np.asarray(std if std is not None else (DEFAULT_DESCRIPTOR_STD_9D if d == 9 else DEFAULT_DESCRIPTOR_STD), dtype=np.float64)
    raw_min = RAW_PHYSICAL_MIN_9D if d == 9 else RAW_PHYSICAL_MIN_12D
    raw_max = RAW_PHYSICAL_MAX_9D if d == 9 else RAW_PHYSICAL_MAX_12D
    
    z_min = (raw_min - ref_mean) / (ref_std + 1e-8)
    z_max = (raw_max - ref_mean) / (ref_std + 1e-8)
    
    A_ratio = np.zeros((1, d), dtype=np.float64)
    A_ratio[0, 0] = -ref_std[0]
    A_ratio[0, 1] = ref_std[1]
    b_ratio = np.array([ref_mean[0] - ref_mean[1]], dtype=np.float64)
    
    return A_ratio, b_ratio, z_min, z_max


def get_standardized_triangle_inequality(d: int = 9,
                                         mean: Optional[np.ndarray] = None,
                                         std: Optional[np.ndarray] = None,
                                         eps: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms the linear triangle inequality r1 + r2 >= 1 + eps (L_mid + L_min >= L_max)
    into standardized coordinates:
        -sigma_0 * z_tilde_0 - sigma_1 * z_tilde_1 <= mu_0 + mu_1 - (1.0 + eps)
        
    Args:
        d: Feature dimension (9 or 12).
        mean: Optional reference mean vector.
        std: Optional reference std vector.
        eps: Positive margin (default 0.0).
        
    Returns:
        A_tri: (1, d) row for inequality A_tri @ z_tilde <= b_tri
        b_tri: (1,) RHS scalar
    """
    ref_mean = np.asarray(mean if mean is not None else (DEFAULT_DESCRIPTOR_MEAN_9D if d == 9 else DEFAULT_DESCRIPTOR_MEAN), dtype=np.float64)
    ref_std = np.asarray(std if std is not None else (DEFAULT_DESCRIPTOR_STD_9D if d == 9 else DEFAULT_DESCRIPTOR_STD), dtype=np.float64)
    A_tri = np.zeros((1, d), dtype=np.float64)
    A_tri[0, 0] = -ref_std[0]
    A_tri[0, 1] = -ref_std[1]
    b_tri = np.array([ref_mean[0] + ref_mean[1] - (1.0 + eps)], dtype=np.float64)
    return A_tri, b_tri


def solve_chebyshev_center_standardized(target_bits: np.ndarray,
                                        W_key: np.ndarray,
                                        delta: float = 0.005,
                                        mean: Optional[np.ndarray] = None,
                                        std: Optional[np.ndarray] = None,
                                        enforce_triangle_inequality: bool = True) -> Tuple[bool, np.ndarray, float]:
    """
    Solves polyhedral Chebyshev center inversion directly in standardized coordinate space z_tilde:
        max r
        subject to:
            -(2 b_j - 1) W_j z_tilde_c + r ||W_j||_2 <= -delta
            A_ratio z_tilde_c + r ||A_ratio||_2 <= b_ratio
            A_tri z_tilde_c + r ||A_tri||_2 <= b_tri   (if enforce_triangle_inequality)
            z_tilde_c + r <= z_max
            -z_tilde_c + r <= -z_min
            r >= 0
            
    The resulting center z_tilde_c is mapped back to physical descriptor space via f = S z_tilde_c + mu.
    
    Returns:
        (is_feasible, candidate_physical_descriptor, chebyshev_radius)
    """
    m, d = W_key.shape
    signs = 2.0 * target_bits.astype(np.float64) - 1.0
    
    ref_mean = np.asarray(mean if mean is not None else (DEFAULT_DESCRIPTOR_MEAN_9D if d == 9 else DEFAULT_DESCRIPTOR_MEAN), dtype=np.float64)
    ref_std = np.asarray(std if std is not None else (DEFAULT_DESCRIPTOR_STD_9D if d == 9 else DEFAULT_DESCRIPTOR_STD), dtype=np.float64)
    
    A_ratio, b_ratio, z_min, z_max = get_standardized_relaxation_constraints(d=d, mean=ref_mean, std=ref_std)
    
    # 1. Sign half-space constraints in standardized space
    # -(signs_j) * (W_j . z_tilde) <= -delta
    A_sign = -(signs[:, None] * W_key)
    b_sign = -np.ones(m, dtype=np.float64) * delta
    
    # 2. Standardized box constraints
    I_d = np.eye(d, dtype=np.float64)
    A_box_u = I_d
    b_box_u = z_max
    A_box_l = -I_d
    b_box_l = -z_min
    
    # 3. Stack linear inequalities A_all z_tilde <= b_all
    if enforce_triangle_inequality:
        A_tri, b_tri = get_standardized_triangle_inequality(d=d, mean=ref_mean, std=ref_std, eps=0.0)
        A_all = np.vstack([A_sign, A_ratio, A_tri, A_box_u, A_box_l])
        b_all = np.concatenate([b_sign, b_ratio, b_tri, b_box_u, b_box_l])
    else:
        A_all = np.vstack([A_sign, A_ratio, A_box_u, A_box_l])
        b_all = np.concatenate([b_sign, b_ratio, b_box_u, b_box_l])
    
    # Compute L2 row norms of constraint normals
    row_norms = np.linalg.norm(A_all, axis=1, keepdims=True)
    
    # Chebyshev formulation: A_all z_tilde_c + r * ||A_all_i||_2 <= b_all
    A_cheb = np.hstack([A_all, row_norms])
    b_cheb = b_all
    
    # Objective: maximize r  ==>  minimize -r
    c = np.zeros(d + 1, dtype=np.float64)
    c[-1] = -1.0
    
    bounds = [(float(z_min[i]), float(z_max[i])) for i in range(d)] + [(0.0, None)]
    
    res = linprog(c, A_ub=A_cheb, b_ub=b_cheb, bounds=bounds, method="highs")
    
    if res.success and -res.fun > 0.0:
        z_c = res.x[:d]
        radius = float(-res.fun)
        f_phys = standardized_to_physical(z_c, mean=ref_mean, std=ref_std).astype(np.float32)
        # Bounded relaxation clamping to physical box
        f_phys[0:2] = np.clip(f_phys[0:2], 0.1, 1.0)
        f_phys[2] = np.clip(f_phys[2], 2.0, 30.0)
        f_phys[3:9] = np.clip(f_phys[3:9], 0.0, 1.0)
        return True, f_phys, radius
    else:
        # Fallback to pseudo-inverse standardized solution
        z_pinv = np.linalg.pinv(W_key) @ (signs * delta)
        f_phys = standardized_to_physical(z_pinv, mean=ref_mean, std=ref_std).astype(np.float32)
        f_phys[0:2] = np.clip(f_phys[0:2], 0.1, 1.0)
        f_phys[2] = np.clip(f_phys[2], 2.0, 30.0)
        f_phys[3:9] = np.clip(f_phys[3:9], 0.0, 1.0)
        return False, f_phys, 0.0
