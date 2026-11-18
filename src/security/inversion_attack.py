from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from scipy.optimize import linprog, minimize
from src.graph.descriptors import (
    DEFAULT_DESCRIPTOR_MEAN, DEFAULT_DESCRIPTOR_STD,
    DEFAULT_DESCRIPTOR_MEAN_9D, DEFAULT_DESCRIPTOR_STD_9D
)
from src.protection.inversion_defense import (
    physical_to_standardized,
    standardized_to_physical,
    get_standardized_relaxation_constraints,
    solve_chebyshev_center_standardized
)

def solve_linear_feasibility_inversion(target_bits: np.ndarray, W_key: np.ndarray,
                                       delta: float = 0.01,
                                       enforce_physical_constraints: bool = True) -> Tuple[bool, np.ndarray]:
    """
    Solves the known-key (Scenario III) linear feasibility pre-image attack using Linear Programming
    with physical descriptor bounds and ordering constraints.
    
    Mathematical Formulation:
    -------------------------
    Under known orthonormal embedding W in R^{M x D} and zero-threshold sign quantization,
    the target bitstring b in {0, 1}^M defines a polyhedral cone:
        b_j = 1  ==>  W_j . z >= 0
        b_j = 0  ==>  W_j . z < 0
        
    Defining signs s_j = 2 * b_j - 1 in {-1, +1}, the system of sign inequalities is:
        -s_j * (W_j . z) <= -delta   for all j = 1, ..., M
        
    Physical Constraints (when enforce_physical_constraints=True):
        1. Side ratio bounds: 0.1 <= r_2 <= r_1 <= 1.0
        2. Side ordering inequality: -sigma_0 * z_0 + sigma_1 * z_1 <= mu_0 - mu_1
        3. Physical perimeter: 2.0 mm <= P_mm <= 30.0 mm
        4. Relative axial angles: 0.0 <= phi_i <= 1.0
        5. Relative mutual orientations: 0.0 <= dth_ij <= 1.0
        6. Minutiae types: 0.0 <= t_i <= 1.0, with MIP rounding to {0, 1} (if D=12)
        
    Args:
        target_bits: 1D array of M binary bits {0, 1}.
        W_key: (M, D) known projection matrix.
        delta: Strictly positive feasibility margin.
        enforce_physical_constraints: Whether to apply physical descriptor bounds.
        
    Returns:
        (is_feasible, candidate_descriptor_physical)
    """
    m, d = W_key.shape
    signs = 2.0 * target_bits.astype(np.float64) - 1.0
    
    ref_mean = DEFAULT_DESCRIPTOR_MEAN if d == 12 else DEFAULT_DESCRIPTOR_MEAN_9D
    ref_std = DEFAULT_DESCRIPTOR_STD if d == 12 else DEFAULT_DESCRIPTOR_STD_9D
    
    A_sign = -(signs[:, None] * W_key)
    b_sign = -np.ones(m, dtype=np.float64) * delta
    
    if enforce_physical_constraints:
        # Enforce descending ratio inequality: r2 <= r1  ==>  -sigma0*z0 + sigma1*z1 <= mu0 - mu1
        A_ratio = np.zeros((1, d), dtype=np.float64)
        A_ratio[0, 0] = -float(ref_std[0])
        A_ratio[0, 1] = float(ref_std[1])
        b_ratio = np.array([float(ref_mean[0] - ref_mean[1])], dtype=np.float64)
        
        A_ub = np.vstack([A_sign, A_ratio])
        b_ub = np.concatenate([b_sign, b_ratio])
        
        if d == 12:
            d_min = np.array([0.1, 0.1, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            d_max = np.array([1.0, 1.0, 30.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        else:
            d_min = np.array([0.1, 0.1, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            d_max = np.array([1.0, 1.0, 30.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
            
        z_min = (d_min - ref_mean) / (ref_std + 1e-8)
        z_max = (d_max - ref_mean) / (ref_std + 1e-8)
        bounds = [(float(z_min[i]), float(z_max[i])) for i in range(d)]
    else:
        A_ub = A_sign
        b_ub = b_sign
        bounds = [(-5.0, 5.0) for _ in range(d)]
        
    c = np.zeros(d, dtype=np.float64)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        z_norm = res.x.astype(np.float32)
        d_physical = z_norm * (ref_std + 1e-8) + ref_mean
        if d == 12:
            # Discrete MIP rounding for binary minutiae types (ending=0.0 vs bifurcation=1.0)
            d_physical[9:12] = np.round(np.clip(d_physical[9:12], 0.0, 1.0))
        # Clip physical ranges
        d_physical[0:2] = np.clip(d_physical[0:2], 0.1, 1.0)
        d_physical[2] = np.clip(d_physical[2], 2.0, 30.0)
        d_physical[3:9] = np.clip(d_physical[3:9], 0.0, 1.0)
        return True, d_physical
    else:
        z_approx = np.linalg.pinv(W_key) @ (signs * delta)
        d_physical = z_approx.astype(np.float32) * (ref_std + 1e-8) + ref_mean
        if d == 12:
            d_physical[9:12] = np.round(np.clip(d_physical[9:12], 0.0, 1.0))
        d_physical[0:2] = np.clip(d_physical[0:2], 0.1, 1.0)
        d_physical[2] = np.clip(d_physical[2], 2.0, 30.0)
        d_physical[3:9] = np.clip(d_physical[3:9], 0.0, 1.0)
        return False, d_physical

def calculate_descriptor_error(d_orig: np.ndarray, d_recon: np.ndarray) -> float:
    """
    Computes normalized Euclidean distance between original and reconstructed descriptors:
        Error = ||d_orig - d_recon||_2 / sqrt(D)
    """
    diff = d_orig - d_recon
    return float(np.linalg.norm(diff) / np.sqrt(len(d_orig)))

def simulate_hill_climbing_inversion(target_template: np.ndarray,
                                     W_key: Optional[np.ndarray],
                                     matcher_fn,
                                     max_iterations: int = 250,
                                     num_triangles: int = 30,
                                     in_dim: int = 12,
                                     out_dim: int = 32) -> Tuple[float, np.ndarray]:
    """
    Simulates a black-box / gradient-free hill-climbing synthetic inversion attack.
    The attacker perturbs synthetic candidate descriptors to maximize match score.
    
    Args:
        target_template: Stolen (N, M) binary template.
        W_key: Projection matrix if known (Scenario III), or None if unknown (Scenario II).
        matcher_fn: Evaluation matching function.
        max_iterations: Iteration search budget (default: 250).
        num_triangles: Number of candidate triangles to synthesize.
        in_dim: Input descriptor dimension (default: 12).
        out_dim: Projection dimension (default: 32).
        
    Returns:
        best_score: Peak similarity score achieved by the attacker.
        best_synthetic_desc: Synthetic descriptor array found.
    """
    rng = np.random.default_rng(42)
    candidate_desc = rng.normal(0.0, 1.0, size=(num_triangles, in_dim)).astype(np.float32)
    dummy_weights = np.ones(num_triangles, dtype=np.float32)
    target_weights = np.ones(len(target_template), dtype=np.float32)
    
    if W_key is None:
        from src.protection.projection import crypto_gaussian_matrix
        A = crypto_gaussian_matrix(b"HillClimbingAttackerGuessKey_32B!", out_dim, in_dim)
        if out_dim >= in_dim:
            Q, _ = np.linalg.qr(A)
            W_guess = Q[:, :in_dim]
        else:
            Q, _ = np.linalg.qr(A.T)
            W_guess = Q[:, :out_dim].T
    else:
        W_guess = W_key
        
    best_candidate = candidate_desc.copy()
    candidate_bits = (candidate_desc @ W_guess.T >= 0.0).astype(np.uint8)
    best_score = matcher_fn(candidate_bits, target_template, dummy_weights, target_weights)
    
    for _ in range(max_iterations):
        mutated = best_candidate.copy()
        row_idx = rng.integers(0, num_triangles)
        col_idx = rng.integers(0, in_dim)
        mutated[row_idx, col_idx] += rng.normal(0.0, 0.15)
        
        mut_bits = (mutated @ W_guess.T >= 0.0).astype(np.uint8)
        score = matcher_fn(mut_bits, target_template, dummy_weights, target_weights)
        
        if score > best_score:
            best_score = score
            best_candidate = mutated
            
    return best_score, best_candidate

def evaluate_reconstruction_resistance(templates: List[np.ndarray],
                                       W_keys: List[np.ndarray],
                                       matcher_fn,
                                       operating_threshold: float = 0.20,
                                       W_reissued_keys: Optional[List[np.ndarray]] = None,
                                       original_descriptors: Optional[List[np.ndarray]] = None) -> Dict:
    """
    Evaluates empirical inversion resistance across Scenarios II and III.
    
    Includes:
    1. Black-box hill-climbing under unknown key (Scenario II).
    2. Black-box hill-climbing under known key (Scenario III HC).
    3. Analytical Linear Programming feasibility inversion under physical constraints (Scenario III LP).
    4. Reconstruction error and minutiae type recovery accuracy.
    5. Cross-key authentication evaluation of LP-synthesized descriptors under reissued keys (Option A).
    
    Returns:
        dict with success rates, mean scores, reconstruction errors, and reissuance resistance.
    """
    scenario2_scores = []
    scenario3_hc_scores = []
    scenario3_lp_success = []
    reconstruction_errors = []
    type_recovery_accuracies = []
    cross_key_lp_scores = []
    
    for i in range(min(5, len(templates))):
        t = templates[i]
        W = W_keys[i]
        d_orig_list = original_descriptors[i] if (original_descriptors is not None and i < len(original_descriptors)) else None
        
        # Scenario II (Attacker does not possess key)
        s2_score, _ = simulate_hill_climbing_inversion(t, W_key=None, matcher_fn=matcher_fn, max_iterations=150, in_dim=12)
        scenario2_scores.append(s2_score)
        
        # Scenario III (Attacker possesses key W - Hill Climbing)
        s3_score, _ = simulate_hill_climbing_inversion(t, W_key=W, matcher_fn=matcher_fn, max_iterations=150, in_dim=12)
        scenario3_hc_scores.append(s3_score)
        
        # Scenario III (Linear Programming Feasibility under physical constraints)
        lp_feasible_descriptors = []
        for j, row in enumerate(t):
            feas, cand_desc = solve_linear_feasibility_inversion(row, W, enforce_physical_constraints=True)
            if feas:
                lp_feasible_descriptors.append(cand_desc)
                if d_orig_list is not None and j < len(d_orig_list):
                    err = calculate_descriptor_error(d_orig_list[j], cand_desc)
                    reconstruction_errors.append(err)
                    # Minutiae type recovery accuracy on categorical features 9, 10, 11
                    type_acc = np.mean(d_orig_list[j][9:12] == cand_desc[9:12])
                    type_recovery_accuracies.append(type_acc)
                    
        feas_ratio = len(lp_feasible_descriptors) / len(t) if len(t) > 0 else 0.0
        scenario3_lp_success.append(feas_ratio)
        
        # Cross-key evaluation under fresh key (Option A Revocation Check)
        if W_reissued_keys is not None and i < len(W_reissued_keys) and lp_feasible_descriptors:
            W_new = W_reissued_keys[i]
            synth_arr = np.array(lp_feasible_descriptors, dtype=np.float32)
            z_synth = (synth_arr - DEFAULT_DESCRIPTOR_MEAN) / (DEFAULT_DESCRIPTOR_STD + 1e-8)
            synth_bits = (z_synth @ W_new.T >= 0.0).astype(np.uint8)
            
            score_cross = matcher_fn(synth_bits, t, np.ones(len(synth_bits)), np.ones(len(t)))
            cross_key_lp_scores.append(score_cross)
            
    sr_s2 = float(np.mean([s >= operating_threshold for s in scenario2_scores]))
    sr_s3 = float(np.mean([s >= operating_threshold for s in scenario3_hc_scores]))
    
    res = {
        "scenario_2_mean_score": float(np.mean(scenario2_scores)),
        "scenario_2_success_rate": sr_s2,
        "scenario_3_hc_mean_score": float(np.mean(scenario3_hc_scores)),
        "scenario_3_hc_success_rate": sr_s3,
        "scenario_3_lp_polytope_feasibility": float(np.mean(scenario3_lp_success))
    }
    if reconstruction_errors:
        res["mean_reconstruction_error"] = float(np.mean(reconstruction_errors))
    if type_recovery_accuracies:
        res["mean_type_recovery_accuracy"] = float(np.mean(type_recovery_accuracies))
    if cross_key_lp_scores:
        res["lp_cross_key_mean_score"] = float(np.mean(cross_key_lp_scores))
    return res


def calculate_angular_wrap_error(d_orig: np.ndarray, d_recon: np.ndarray) -> float:
    """
    Calculates mean circular difference on angular coordinates (indices 3..8: axial angles & mutual orientations)
    accounting for periodic wrap-around on the circle [0, 1):
        Delta_circ(a, b) = min(|a - b| mod 1, 1 - (|a - b| mod 1))
    """
    diff = np.abs(d_orig[3:9] - d_recon[3:9]) % 1.0
    circ_diff = np.minimum(diff, 1.0 - diff)
    return float(np.mean(circ_diff))


def calculate_continuous_inversion_metrics(d_orig: np.ndarray, d_recon: np.ndarray) -> Dict[str, float]:
    """
    Calculates comprehensive continuous feature reconstruction metrics:
    - MAE: Mean Absolute Error over continuous coordinates (with circular wrap on angles)
    - RMSE: Root Mean Squared Error over continuous coordinates (with circular wrap on angles)
    - Normalized Euclidean Error: ||d_orig - d_recon||_2 / sqrt(D_cont)
    - Angular Circular MAE
    - Minutiae-type recovery accuracy (if D=12)
    """
    d_len = 9 if len(d_orig) in (9, 12) else len(d_orig)
    # Geometric coordinates 0..2 (ratios & perimeter)
    geom_diff = np.abs(d_orig[:3] - d_recon[:3])
    # Angular coordinates 3..8 (axial angles & mutual orientations)
    ang_diff = np.abs(d_orig[3:9] - d_recon[3:9]) % 1.0
    ang_circ = np.minimum(ang_diff, 1.0 - ang_diff)
    
    all_cont_diff = np.concatenate([geom_diff, ang_circ])
    mae = float(np.mean(all_cont_diff))
    rmse = float(np.sqrt(np.mean(all_cont_diff**2)))
    norm_euclid = float(np.linalg.norm(all_cont_diff) / np.sqrt(d_len))
    ang_mae = float(np.mean(ang_circ))
    
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "normalized_euclidean": norm_euclid,
        "angular_mae": ang_mae
    }
    if len(d_orig) == 12 and len(d_recon) >= 12:
        type_acc = float(np.mean(np.round(d_orig[9:12]) == np.round(d_recon[9:12])))
        metrics["type_accuracy"] = type_acc
    return metrics


def solve_chebyshev_center_inversion(target_bits: np.ndarray, W_key: np.ndarray,
                                     delta: float = 0.005,
                                     enforce_physical_constraints: bool = True,
                                     mean: Optional[np.ndarray] = None,
                                     std: Optional[np.ndarray] = None) -> Tuple[bool, np.ndarray, float]:
    """
    Solves the known-key polyhedral Chebyshev center inversion attack.
    Finds the center of the largest Euclidean ball inscribed in the bounded physical polyhedron:
        max r
        subject to A_i d_c + r ||A_i||_2 <= b_i
        
    For 12-D descriptors:
        First determines the discrete integer tuple (t0, t1, t2) in {0, 1}^3 via MILP (HiGHS).
        Then solves the continuous Chebyshev center LP over the 9 continuous coordinates d_0..d_8
        with the binary minutiae types fixed to (t0, t1, t2).
        
    For 9-D descriptors:
        All 9 coordinates are continuous; Chebyshev center is solved directly over R^9.
        
    Returns:
        (is_feasible, candidate_descriptor, chebyshev_radius)
    """
    from src.security.leakage_analysis import solve_milp_feasibility_inversion
    
    m, d = W_key.shape
    signs = 2.0 * target_bits.astype(np.float64) - 1.0
    
    if mean is not None and std is not None:
        ref_mean = np.asarray(mean, dtype=np.float64)
        ref_std = np.asarray(std, dtype=np.float64)
    else:
        ref_mean = DEFAULT_DESCRIPTOR_MEAN if d == 12 else DEFAULT_DESCRIPTOR_MEAN_9D
        ref_std = DEFAULT_DESCRIPTOR_STD if d == 12 else DEFAULT_DESCRIPTOR_STD_9D
    
    d_min_cont = np.array([0.1, 0.1, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    d_max_cont = np.array([1.0, 1.0, 30.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    
    if d == 12:
        # Step 1: Solve MILP for discrete minutiae types
        milp_feas, cand_milp = solve_milp_feasibility_inversion(target_bits, W_key, delta=delta)
        if not milp_feas:
            # Fallback to pseudo-inverse
            z_approx = np.linalg.pinv(W_key) @ (signs * delta)
            d_approx = z_approx.astype(np.float32) * (ref_std + 1e-8) + ref_mean
            d_approx[9:12] = np.round(np.clip(d_approx[9:12], 0.0, 1.0))
            d_approx[0:2] = np.clip(d_approx[0:2], 0.1, 1.0)
            d_approx[2] = np.clip(d_approx[2], 2.0, 30.0)
            d_approx[3:9] = np.clip(d_approx[3:9], 0.0, 1.0)
            return False, d_approx, 0.0
            
        t_fixed = np.round(np.clip(cand_milp[9:12], 0.0, 1.0))
        W_scaled = W_key / (ref_std + 1e-8)
        
        A_sign_cont = -(signs[:, None] * W_scaled[:, :9])
        A_sign_disc = -(signs[:, None] * W_scaled[:, 9:12])
        b_sign_cont = -np.ones(m, dtype=np.float64) * delta - signs * (W_scaled @ ref_mean) - (A_sign_disc @ t_fixed)
        
        A_ratio = np.zeros((1, 9), dtype=np.float64)
        A_ratio[0, 0] = -1.0
        A_ratio[0, 1] = 1.0
        b_ratio = np.array([0.0], dtype=np.float64)
        
        I_9 = np.eye(9, dtype=np.float64)
        A_box_u = I_9
        b_box_u = d_max_cont
        A_box_l = -I_9
        b_box_l = -d_min_cont
        
        A_all = np.vstack([A_sign_cont, A_ratio, A_box_u, A_box_l])
        b_all = np.concatenate([b_sign_cont, b_ratio, b_box_u, b_box_l])
        
        norms = np.linalg.norm(A_all, axis=1, keepdims=True)
        A_cheb = np.hstack([A_all, norms])
        b_cheb = b_all
        
        c = np.zeros(10, dtype=np.float64)
        c[-1] = -1.0
        bounds = [(float(d_min_cont[i]), float(d_max_cont[i])) for i in range(9)] + [(0.0, None)]
        
        res = linprog(c, A_ub=A_cheb, b_ub=b_cheb, bounds=bounds, method='highs')
        if res.success and -res.fun > 0.0:
            d_cont = res.x[:9].astype(np.float32)
            d_full = np.concatenate([d_cont, t_fixed.astype(np.float32)])
            return True, d_full, float(-res.fun)
        else:
            return True, cand_milp.astype(np.float32), 0.0
    else:
        # D = 9: Pure geometric descriptor
        W_scaled = W_key / (ref_std + 1e-8)
        A_sign = -(signs[:, None] * W_scaled)
        b_sign = -np.ones(m, dtype=np.float64) * delta - signs * (W_scaled @ ref_mean)
        
        A_ratio = np.zeros((1, 9), dtype=np.float64)
        A_ratio[0, 0] = -1.0
        A_ratio[0, 1] = 1.0
        b_ratio = np.array([0.0], dtype=np.float64)
        
        I_9 = np.eye(9, dtype=np.float64)
        A_box_u = I_9
        b_box_u = d_max_cont
        A_box_l = -I_9
        b_box_l = -d_min_cont
        
        A_all = np.vstack([A_sign, A_ratio, A_box_u, A_box_l])
        b_all = np.concatenate([b_sign, b_ratio, b_box_u, b_box_l])
        
        norms = np.linalg.norm(A_all, axis=1, keepdims=True)
        A_cheb = np.hstack([A_all, norms])
        b_cheb = b_all
        
        c = np.zeros(10, dtype=np.float64)
        c[-1] = -1.0
        bounds = [(float(d_min_cont[i]), float(d_max_cont[i])) for i in range(9)] + [(0.0, None)]
        
        res = linprog(c, A_ub=A_cheb, b_ub=b_cheb, bounds=bounds, method='highs')
        if res.success and -res.fun > 0.0:
            d_cand = res.x[:9].astype(np.float32)
            d_cand[0:2] = np.clip(d_cand[0:2], 0.1, 1.0)
            d_cand[2] = np.clip(d_cand[2], 2.0, 30.0)
            d_cand[3:9] = np.clip(d_cand[3:9], 0.0, 1.0)
            return True, d_cand, float(-res.fun)
        else:
            # Fallback to standard feasibility
            feas, cand_lin = solve_linear_feasibility_inversion(target_bits, W_key, delta=delta, enforce_physical_constraints=True)
            return feas, cand_lin.astype(np.float32), 0.0


def solve_qp_center_inversion(target_bits: np.ndarray, W_key: np.ndarray,
                              delta: float = 0.005,
                              enforce_physical_constraints: bool = True,
                              mean: Optional[np.ndarray] = None,
                              std: Optional[np.ndarray] = None) -> Tuple[bool, np.ndarray]:
    """
    Solves Quadratic Programming (QP) Euclidean center inversion attack.
    Minimizes Euclidean Mahalanobis distance from prior mean ||(d - mu) / sigma||_2^2
    subject to polyhedral cone half-spaces and physical ordering constraints.
    """
    m, d = W_key.shape
    signs = 2.0 * target_bits.astype(np.float64) - 1.0
    
    if mean is not None and std is not None:
        ref_mean = np.asarray(mean, dtype=np.float64)
        ref_std = np.asarray(std, dtype=np.float64)
    else:
        ref_mean = DEFAULT_DESCRIPTOR_MEAN if d == 12 else DEFAULT_DESCRIPTOR_MEAN_9D
        ref_std = DEFAULT_DESCRIPTOR_STD if d == 12 else DEFAULT_DESCRIPTOR_STD_9D
    
    if d == 12:
        from src.security.leakage_analysis import solve_milp_feasibility_inversion
        milp_feas, cand_milp = solve_milp_feasibility_inversion(target_bits, W_key, delta=delta)
        if not milp_feas:
            return False, cand_milp
        t_fixed = np.round(np.clip(cand_milp[9:12], 0.0, 1.0))
        
        W_scaled = W_key / (ref_std + 1e-8)
        A_sign_cont = -(signs[:, None] * W_scaled[:, :9])
        A_sign_disc = -(signs[:, None] * W_scaled[:, 9:12])
        b_sign_cont = -np.ones(m, dtype=np.float64) * delta - signs * (W_scaled @ ref_mean) - (A_sign_disc @ t_fixed)
        
        A_ratio = np.zeros((1, 9), dtype=np.float64)
        A_ratio[0, 0] = -1.0
        A_ratio[0, 1] = 1.0
        b_ratio = np.array([0.0], dtype=np.float64)
        
        A_ineq = np.vstack([A_sign_cont, A_ratio])
        b_ineq = np.concatenate([b_sign_cont, b_ratio])
        
        mu_cont = ref_mean[:9]
        sig_cont = ref_std[:9]
        
        def obj(x):
            z = (x - mu_cont) / sig_cont
            return 0.5 * np.sum(z**2)
            
        def grad(x):
            return (x - mu_cont) / (sig_cont**2)
            
        d_min = np.array([0.1, 0.1, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        d_max = np.array([1.0, 1.0, 30.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        bounds = [(float(d_min[i]), float(d_max[i])) for i in range(9)]
        cons = {'type': 'ineq', 'fun': lambda x: b_ineq - A_ineq @ x, 'jac': lambda x: -A_ineq}
        
        res = minimize(obj, cand_milp[:9], jac=grad, bounds=bounds, constraints=cons, method='SLSQP')
        if res.success:
            d_full = np.concatenate([res.x.astype(np.float32), t_fixed.astype(np.float32)])
            return True, d_full
        else:
            return True, cand_milp.astype(np.float32)
    else:
        # D = 9
        W_scaled = W_key / (ref_std + 1e-8)
        A_sign = -(signs[:, None] * W_scaled)
        b_sign = -np.ones(m, dtype=np.float64) * delta - signs * (W_scaled @ ref_mean)
        
        A_ratio = np.zeros((1, 9), dtype=np.float64)
        A_ratio[0, 0] = -1.0
        A_ratio[0, 1] = 1.0
        b_ratio = np.array([0.0], dtype=np.float64)
        
        A_ineq = np.vstack([A_sign, A_ratio])
        b_ineq = np.concatenate([b_sign, b_ratio])
        
        def obj(x):
            z = (x - ref_mean) / ref_std
            return 0.5 * np.sum(z**2)
            
        def grad(x):
            return (x - ref_mean) / (ref_std**2)
            
        d_min = np.array([0.1, 0.1, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        d_max = np.array([1.0, 1.0, 30.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        bounds = [(float(d_min[i]), float(d_max[i])) for i in range(9)]
        cons = {'type': 'ineq', 'fun': lambda x: b_ineq - A_ineq @ x, 'jac': lambda x: -A_ineq}
        
        # Initial point from standard LP
        feas, x0 = solve_linear_feasibility_inversion(target_bits, W_key, delta=delta, enforce_physical_constraints=True)
        res = minimize(obj, x0, jac=grad, bounds=bounds, constraints=cons, method='SLSQP')
        if res.success:
            return True, res.x.astype(np.float32)
        else:
            return feas, x0.astype(np.float32)


def solve_mcc_chebyshev_inversion(
    target_bits: np.ndarray,
    W_key: np.ndarray,
    delta: float = 0.005,
    box_bound: float = 4.0,
) -> Tuple[bool, np.ndarray, float]:
    """
    Solves the known-key (Scenario III) Chebyshev-center pre-image LP for the
    256-D -> 192-D unary MCC embedding in standardized z in [-box_bound, box_bound]^D space:
        max r
        s.t. -(2b_j - 1) * (W_j . z) + r * ||W_j||_2 <= -delta   for j = 1..M
             -box_bound + r <= z_i <= box_bound - r            for i = 1..D

    When D > M (e.g. D=256, M=192), the 64-D orthogonal nullspace ker(W_key) permits
    continuous variation without altering any bit b_j, yielding r* > 0.
    """
    b = np.asarray(target_bits, dtype=np.uint8).ravel()
    W = np.asarray(W_key, dtype=np.float64)
    m, d = W.shape
    if len(b) != m:
        raise ValueError(f"target_bits length {len(b)} must match W_key rows {m}")

    signs = 2.0 * b.astype(np.float64) - 1.0
    A_sign = -(signs[:, None] * W)
    b_sign = -np.full(m, float(delta), dtype=np.float64)

    I_d = np.eye(d, dtype=np.float64)
    A_all = np.vstack([A_sign, I_d, -I_d])
    b_all = np.concatenate([b_sign, np.full(2 * d, float(box_bound), dtype=np.float64)])

    norms = np.linalg.norm(A_all, axis=1, keepdims=True)
    A_cheb = np.hstack([A_all, norms])
    c = np.zeros(d + 1, dtype=np.float64)
    c[-1] = -1.0
    bounds = [(-float(box_bound), float(box_bound)) for _ in range(d)] + [(0.0, None)]

    res = linprog(c, A_ub=A_cheb, b_ub=b_all, bounds=bounds, method="highs")
    if res.success:
        z_cand = res.x[:d].astype(np.float64)
        radius = float(res.x[-1])
        return True, z_cand, radius

    z_pinv = np.linalg.pinv(W) @ (signs * float(delta))
    return False, np.clip(z_pinv, -box_bound, box_bound), 0.0


def _procrustes_rmse(src: np.ndarray, dst: np.ndarray) -> float:
    """Best-fit 2D rigid alignment RMSE between ordered point sets src and dst."""
    if len(src) < 2 or len(dst) < 2:
        return float("inf")
    s_c = src - src.mean(axis=0, keepdims=True)
    d_c = dst - dst.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(s_c.T @ d_c)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    aligned = s_c @ r.T
    return float(np.sqrt(np.mean(np.sum((aligned - d_c) ** 2, axis=1))))


def evaluate_joint_rff_mcc_inversion(
    target_bits: np.ndarray,
    target_edge_tokens: np.ndarray,
    W_key: np.ndarray,
    W_rff: np.ndarray,
    b_rff: np.ndarray,
    true_mcc: Optional[np.ndarray] = None,
    true_coords: Optional[np.ndarray] = None,
    true_thetas: Optional[np.ndarray] = None,
    dpi: float = 500.0,
    sigma_dist: float = 18.0,
    sigma_angle: float = 0.38,
    sigma_edge_angle: float = 0.34,
    cross_key_W: Optional[np.ndarray] = None,
    cross_key_W_rff: Optional[np.ndarray] = None,
    cross_key_b_rff: Optional[np.ndarray] = None,
    cross_key_bits: Optional[np.ndarray] = None,
    cross_key_edge_tokens: Optional[np.ndarray] = None,
    tau: float = 0.36,
    max_nodes_opt: int = 12,
    n_starts: int = 2,
    maxiter: int = 35,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates Scenario-III joint compromised-key pre-image inversion resistance
    when an adversary possesses BOTH enrolled protected templates `(B, E)` AND
    derived keys `(W_key, W_rff, b_rff)`:
      1. Solves the 256-D -> 192-D Chebyshev LP on unary MCC bitstrings, measuring
         the 64-D orthogonal nullspace ambiguity (`r*`, continuous cosine error,
         and bit-invariance under nullspace perturbations).
      2. Performs multi-start gradient optimization against keyed RFF edge constraints
         `cos(W_rff . g(u,v) + b_rff)` to attempt pairwise SE(2) coordinate recovery.
      3. Evaluates whether the recovered pre-image breaches an independent/re-issued
         credential protected under `(cross_key_W, cross_key_W_rff, cross_key_b_rff)`.
    """
    from src.matching.spectral_consensus import (
        SpectralGraphMatcher,
        _build_edge_feature_tensor,
        encode_edge_rff,
    )
    from src.matching.bipartite_matcher import compute_pairwise_hamming_distance

    rng = np.random.default_rng(seed)
    bits_arr = np.asarray(target_bits, dtype=np.uint8)
    edges_u64 = np.asarray(target_edge_tokens, dtype=np.uint64)
    W = np.asarray(W_key, dtype=np.float64)
    Wr = np.asarray(W_rff, dtype=np.float64)
    br = np.asarray(b_rff, dtype=np.float64)

    n = len(bits_arr)
    m_sub = min(n, max(3, int(max_nodes_opt)))
    w64 = edges_u64.shape[-1]
    rff_bits = int(Wr.shape[0])

    # 1. Unary MCC Chebyshev-center inversion & nullspace witness
    radii: List[float] = []
    bit_sats: List[float] = []
    cos_sims: List[float] = []
    nullspace_preserved: List[float] = []
    z_reconstructed = np.zeros((n, W.shape[1]), dtype=np.float64)

    # Compute an explicit basis vector in ker(W) when D > M
    _, _, vh = np.linalg.svd(W, full_matrices=True)
    null_basis = vh[W.shape[0]:, :]  # Shape: (D - M, D)

    for idx in range(min(n, 16)):
        feas, z_hat, rad = solve_mcc_chebyshev_inversion(bits_arr[idx], W)
        z_reconstructed[idx] = z_hat
        radii.append(float(rad) if feas else 0.0)
        b_hat = (W @ z_hat >= 0.0).astype(np.uint8)
        bit_sats.append(float(np.mean(b_hat == bits_arr[idx])))
        if len(null_basis) > 0:
            # Move 2.0 sigma along nullspace without changing W @ z
            z_pert = z_hat + 2.0 * null_basis[0]
            b_pert = (W @ z_pert >= 0.0).astype(np.uint8)
            nullspace_preserved.append(float(np.mean(b_pert == bits_arr[idx])))
        if true_mcc is not None and np.linalg.norm(true_mcc[idx]) > 1e-8 and np.linalg.norm(z_hat) > 1e-8:
            cos_sims.append(
                float(np.dot(true_mcc[idx], z_hat) / (np.linalg.norm(true_mcc[idx]) * np.linalg.norm(z_hat)))
            )

    # 2. Joint RFF edge-token surrogate optimization on m_sub nodes
    edge_sub_u64 = edges_u64[:m_sub, :m_sub, :]
    edge_sub_bits = np.unpackbits(edge_sub_u64.view(np.uint8), axis=-1)[..., :rff_bits]
    signs_edge = 2.0 * edge_sub_bits.astype(np.float64) - 1.0  # (m_sub, m_sub, rff_bits)
    offdiag_mask = ~np.eye(m_sub, dtype=bool)

    scale = max(float(dpi), 100.0) / 500.0
    box_px = 360.0 * scale

    best_loss = float("inf")
    best_coords = np.zeros((m_sub, 2), dtype=np.float64)
    best_thetas = np.zeros(m_sub, dtype=np.float64)

    def _unpack_state(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c = vec[: 2 * m_sub].reshape((m_sub, 2))
        t = np.mod(vec[2 * m_sub :], 2.0 * np.pi)
        return c, t

    def _edge_loss(vec: np.ndarray) -> float:
        coords_c, thetas_c = _unpack_state(vec)
        g = _build_edge_feature_tensor(
            coords_c,
            thetas_c,
            sigma_dist=sigma_dist,
            sigma_angle=sigma_angle,
            sigma_edge_angle=sigma_edge_angle,
            dpi=dpi,
        )
        proj = np.einsum("ijd,kd->ijk", g, Wr) + br[None, None, :]
        margins = signs_edge * np.cos(proj)
        # Penalize violated RFF sign half-spaces smoothly
        hinge = np.maximum(0.0, 0.15 - margins)
        return float(np.mean((hinge[offdiag_mask]) ** 2))

    opt_bounds = [(0.0, box_px)] * (2 * m_sub) + [(-2.0 * np.pi, 4.0 * np.pi)] * m_sub
    for _ in range(max(1, int(n_starts))):
        x0 = np.concatenate([
            rng.uniform(20.0 * scale, box_px - 20.0 * scale, size=2 * m_sub),
            rng.uniform(0.0, 2.0 * np.pi, size=m_sub),
        ])
        res = minimize(_edge_loss, x0, method="L-BFGS-B", bounds=opt_bounds, options={"maxiter": int(maxiter)})
        if float(res.fun) < best_loss:
            best_loss = float(res.fun)
            best_coords, best_thetas = _unpack_state(res.x)

    rec_edge_u64 = encode_edge_rff(
        best_coords,
        best_thetas,
        Wr,
        br,
        sigma_dist=sigma_dist,
        sigma_angle=sigma_angle,
        sigma_edge_angle=sigma_edge_angle,
        dpi=dpi,
    )
    edge_xor = np.bitwise_xor(rec_edge_u64, edge_sub_u64)
    edge_bit_agreement = 1.0 - float(np.sum(np.bitwise_count(edge_xor)) / max(1, m_sub * m_sub * rff_bits))

    out: Dict[str, Any] = {
        "in_dim": int(W.shape[1]),
        "out_dim": int(W.shape[0]),
        "nullspace_dim": int(max(0, W.shape[1] - W.shape[0])),
        "mean_chebyshev_radius": float(np.mean(radii)) if radii else 0.0,
        "mcc_bit_satisfaction": float(np.mean(bit_sats)) if bit_sats else 0.0,
        "nullspace_bit_invariance": float(np.mean(nullspace_preserved)) if nullspace_preserved else 0.0,
        "mcc_cosine_similarity": float(np.mean(cos_sims)) if cos_sims else None,
        "edge_opt_loss": float(best_loss),
        "edge_bit_agreement": float(edge_bit_agreement),
        "procrustes_rmse_px": None,
        "pairwise_dist_mae_px": None,
        "cross_key_spectral_score": None,
    }

    if true_coords is not None and len(true_coords) >= m_sub:
        tc = np.asarray(true_coords[:m_sub], dtype=np.float64)
        out["procrustes_rmse_px"] = _procrustes_rmse(best_coords, tc)
        d_rec = np.linalg.norm(best_coords[:, None, :] - best_coords[None, :, :], axis=-1)
        d_true = np.linalg.norm(tc[:, None, :] - tc[None, :, :], axis=-1)
        out["pairwise_dist_mae_px"] = float(np.mean(np.abs(d_rec[offdiag_mask] - d_true[offdiag_mask])))

    # 3. Cross-key / post-revocation replay check
    if (
        cross_key_W is not None
        and cross_key_W_rff is not None
        and cross_key_b_rff is not None
        and cross_key_bits is not None
        and cross_key_edge_tokens is not None
    ):
        att_bits_cross = (z_reconstructed[:m_sub] @ np.asarray(cross_key_W, dtype=np.float64).T >= 0.0).astype(np.uint8)
        att_edge_cross = encode_edge_rff(
            best_coords,
            best_thetas,
            np.asarray(cross_key_W_rff, dtype=np.float64),
            np.asarray(cross_key_b_rff, dtype=np.float64),
            sigma_dist=sigma_dist,
            sigma_angle=sigma_angle,
            sigma_edge_angle=sigma_edge_angle,
            dpi=dpi,
        )
        dmat = compute_pairwise_hamming_distance(att_bits_cross, np.asarray(cross_key_bits, dtype=np.uint8))
        matcher = SpectralGraphMatcher(
            sigma_dist=sigma_dist,
            sigma_angle=sigma_angle,
            sigma_edge_angle=sigma_edge_angle,
        )
        score, _ = matcher.match_protected(
            query_edge_tokens=att_edge_cross,
            enrolled_edge_tokens=np.asarray(cross_key_edge_tokens, dtype=np.uint64),
            pairwise_dists=dmat,
            tau=tau,
        )
        out["cross_key_spectral_score"] = float(score)

    return out

