"""
System Unlinkability Evaluation according to Gómez-Barrero et al. (IEEE TIFS 2018).
Evaluates local linkability D_omega(s) and global system linkability D_sys across
mated (same-subject, different-key) and non-mated (different-subject, different-key) score distributions.

Supports:
- Kernel Density Estimation (Gaussian KDE) over the system comparator domain [0, 1].
- Regularized histogram binning with formal score domains.
- Multi-attacker evaluation: Descriptor-only, Topology-only, and Combined (Descriptor + Topology).
- Subject-level clustered bootstrap confidence intervals (B >= 500).
- Balanced comparison designs across all K(K-1)/2 application key pairs.
"""

from typing import Tuple, List, Dict, Optional, Any
import numpy as np
from scipy.stats import gaussian_kde


def compute_system_unlinkability(mated_scores: np.ndarray, 
                                 non_mated_scores: np.ndarray, 
                                 num_bins: int = 100,
                                 score_domain: Optional[Tuple[float, float]] = (0.0, 1.0),
                                 method: str = "kde",
                                 omega: float = 1.0,
                                 bw_factor: float = 1.0) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes system unlinkability metric D_sys according to the Gómez-Barrero et al. framework:
    'General Framework to Evaluate Unlinkability in Biometric Template Protection Systems' (IEEE TIFS 2018).
    
    Likelihood Ratio and Local Measure with Prior Ratio:
        LR(s) = p(s | H_m) / p(s | H_nm)
        omega = P(H_nm) / P(H_m)
        D_omega(s) = 0 if LR(s) <= omega, else 2 * (LR(s) / (omega + LR(s))) - 1
        D_sys = integral_{S} D_omega(s) * p(s | H_m) ds
        
    Prior Ratio Assumption & Convention:
        In ISO/IEC 30136:2018 and Gómez-Barrero et al. (IEEE TIFS 2018), local linkability is:
            D_omega(s) = 0 if LR(s) <= omega, else 2 * (LR(s) / (omega + LR(s))) - 1
        where omega = P(H_nm) / P(H_m).
        
        Inverse Ratio Convention Note:
        Some publications define the prior ratio inversely as omega_inv = P(H_m) / P(H_nm) = 1 / omega,
        testing LR(s) * omega_inv > 1. Under non-informative or unknown operational priors,
        P(H_m) = P(H_nm) = 0.5, so both conventions coincide at omega = omega_inv = 1.0.
        Setting omega = 1.0 represents the non-informative, worst-case attacker assumption.
        
    Boundary-Corrected KDE:
        Standard Gaussian KDE over a compact domain [s_min, s_max] exhibits boundary distortion
        due to mass leaking outside the interval. Boundary bias is removed via Silverman (1986)
        3-term mirror reflection:
        f_corr(s) = f(s) + f(2*s_min - s) + f(2*s_max - s)
        and normalized such that integral_{s_min}^{s_max} p(s) ds == 1.0.
        The derivative at s_min is:
            f_corr'(s_min) = f'(s_min) - f'(s_min) - f'(2*s_max - s_min) = -f'(2*s_max - s_min)
        which vanishes exponentially as |s_max - s_min| / h >> 1.
        Similarly, f_corr'(s_max) = -f'(2*s_min - s_max) ≈ 0.
        Trapezoidal discrete quadrature strictly ensures sum(p_mated) * delta_s == 1.0.
        
    Metric Interpretation:
        D_sys = 0: Ideal unlinkability. Mated cross-key and non-mated distributions coincide.
        0 < D_sys < 0.20: Low residual linkability (substantial privacy preservation).
        0.20 <= D_sys < 0.50: Semi-unlinkable / moderate linkability.
        D_sys >= 0.50: High linkability / linkable.
        D_sys = 1.0: Complete linkability (distributions fully separated).
        
    Args:
        mated_scores: Comparison scores between templates from the SAME subject under DIFFERENT keys.
        non_mated_scores: Comparison scores between templates from DIFFERENT subjects under DIFFERENT keys.
        num_bins: Evaluation sample points across the score domain (default: 100).
        score_domain: Theoretical/operational score domain of the comparator (default: (0.0, 1.0)).
        method: Density estimation method ('kde' with regularized fallback or 'histogram').
        omega: Prior ratio P(H_nm) / P(H_m) (default: 1.0, worst-case / non-informative prior).
        bw_factor: Multiplier for Scott's rule KDE bandwidth (default: 1.0).
        
    Returns:
        D_sys: Global system unlinkability scalar in [0, 1].
        eval_points: Sample points along the score domain.
        d_omega: Local unlinkability measure D_omega(s) at each point.
        lr: Likelihood ratio LR(s) at each point.
    """
    if mated_scores is None or len(mated_scores) == 0:
        raise ValueError("Missing mated (same-subject cross-key) score distribution. Both mated and non-mated distributions are required.")
    if non_mated_scores is None or len(non_mated_scores) == 0:
        raise ValueError("Missing non-mated (different-subject cross-key) score distribution. Both mated and non-mated distributions are required.")
        
    m_arr = np.asarray(mated_scores, dtype=np.float64)
    nm_arr = np.asarray(non_mated_scores, dtype=np.float64)
    
    if score_domain is not None:
        s_min, s_max = float(score_domain[0]), float(score_domain[1])
    else:
        all_s = np.concatenate([m_arr, nm_arr])
        s_min, s_max = float(np.min(all_s)), float(np.max(all_s))
        
    if s_min >= s_max:
        return 0.0, np.array([s_min]), np.array([0.0]), np.array([1.0])
        
    eval_points = np.linspace(s_min, s_max, num_bins)
    delta_s = eval_points[1] - eval_points[0]
    
    # Density estimation
    use_kde = (method == "kde") and (len(m_arr) >= 5) and (len(nm_arr) >= 5)
    
    p_mated = None
    p_non_mated = None
    
    if use_kde:
        try:
            # Add micro-jitter if variance is near-zero to prevent LinAlgError
            std_m = np.std(m_arr)
            std_nm = np.std(nm_arr)
            jit_m = m_arr if std_m > 1e-7 else m_arr + 1e-5 * np.random.default_rng(42).normal(size=len(m_arr))
            jit_nm = nm_arr if std_nm > 1e-7 else nm_arr + 1e-5 * np.random.default_rng(43).normal(size=len(nm_arr))
            
            kde_m = gaussian_kde(jit_m)
            kde_nm = gaussian_kde(jit_nm)
            
            if bw_factor != 1.0:
                kde_m.set_bandwidth(bw_method=kde_m.factor * bw_factor)
                kde_nm.set_bandwidth(bw_method=kde_nm.factor * bw_factor)
                
            # Boundary-corrected KDE via mirror reflection at s_min and s_max
            pm = kde_m(eval_points) + kde_m(2.0 * s_min - eval_points) + kde_m(2.0 * s_max - eval_points)
            pnm = kde_nm(eval_points) + kde_nm(2.0 * s_min - eval_points) + kde_nm(2.0 * s_max - eval_points)
            
            sum_pm = np.sum(pm) * delta_s
            sum_pnm = np.sum(pnm) * delta_s
            if sum_pm > 1e-12 and sum_pnm > 1e-12:
                p_mated = pm / sum_pm
                p_non_mated = pnm / sum_pnm
                assert abs(np.sum(p_mated) * delta_s - 1.0) < 1e-3, "Mated density must integrate to 1.0"
                assert abs(np.sum(p_non_mated) * delta_s - 1.0) < 1e-3, "Non-mated density must integrate to 1.0"
        except Exception:
            use_kde = False
            
    if not use_kde or p_mated is None:
        # Regularized histogram fallback
        bins = np.linspace(s_min, s_max, num_bins + 1)
        eval_points = 0.5 * (bins[:-1] + bins[1:])
        delta_s = bins[1] - bins[0]
        
        c_m, _ = np.histogram(m_arr, bins=bins)
        c_nm, _ = np.histogram(nm_arr, bins=bins)
        
        # Laplace smoothing (add-1 epsilon)
        eps = 1e-4
        p_mated = (c_m + eps) / (np.sum(c_m) + eps * num_bins) / delta_s
        p_non_mated = (c_nm + eps) / (np.sum(c_nm) + eps * num_bins) / delta_s
        
    eps_reg = 1e-12
    lr = (p_mated + eps_reg) / (p_non_mated + eps_reg)
    
    # Local linkability D_omega(s) with prior ratio omega = P(H_nm) / P(H_m)
    # D_omega(s) = 0 if LR <= omega, else 2 * (LR / (omega + LR)) - 1
    d_omega = np.zeros_like(lr)
    mask = (lr > omega)
    d_omega[mask] = 2.0 * (lr[mask] / (omega + lr[mask])) - 1.0
    
    # Global system unlinkability D_sys: integral D_omega(s) * p(s | H_m) ds
    prob_mated = p_mated * delta_s
    d_sys = float(np.sum(d_omega * prob_mated))
    d_sys = float(np.clip(d_sys, 0.0, 1.0))
    
    return d_sys, eval_points, d_omega, lr


def compute_local_unlinkability(mated_scores: np.ndarray, 
                                non_mated_scores: np.ndarray, 
                                num_bins: int = 100,
                                score_domain: Optional[Tuple[float, float]] = (0.0, 1.0),
                                method: str = "kde",
                                omega: float = 1.0,
                                bw_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes local unlinkability measure D_omega(s) and likelihood ratio LR(s)
    according to Gómez-Barrero et al. (IEEE TIFS 2018).
    
    Returns:
        eval_points: Sample points along the score domain.
        d_omega: Local unlinkability measure D_omega(s) at each point.
        lr: Likelihood ratio LR(s) at each point.
    """
    _, eval_points, d_omega, lr = compute_system_unlinkability(
        mated_scores, non_mated_scores, num_bins=num_bins,
        score_domain=score_domain, method=method, omega=omega, bw_factor=bw_factor
    )
    return eval_points, d_omega, lr


def compute_system_unlinkability_with_ci(mated_scores: np.ndarray, 
                                         non_mated_scores: np.ndarray,
                                         num_bins: int = 100, 
                                         score_domain: Optional[Tuple[float, float]] = (0.0, 1.0),
                                         num_bootstraps: int = 2000, 
                                         alpha: float = 0.05,
                                         method: str = "kde",
                                         omega: float = 1.0,
                                         bw_factor: float = 1.0) -> Dict[str, Any]:
    """
    Computes system unlinkability D_sys with 95% bootstrap confidence intervals.
    """
    rng = np.random.default_rng(42)
    n_m = len(mated_scores)
    n_nm = len(non_mated_scores)
    
    d_sys_point, eval_points, d_omega, lr = compute_system_unlinkability(
        mated_scores, non_mated_scores, num_bins=num_bins, score_domain=score_domain, 
        method=method, omega=omega, bw_factor=bw_factor
    )
    
    d_sys_samples = []
    for _ in range(num_bootstraps):
        m_sample = rng.choice(mated_scores, size=n_m, replace=True)
        nm_sample = rng.choice(non_mated_scores, size=n_nm, replace=True)
        ds, _, _, _ = compute_system_unlinkability(
            m_sample, nm_sample, num_bins=num_bins, score_domain=score_domain, 
            method=method, omega=omega, bw_factor=bw_factor
        )
        d_sys_samples.append(ds)
        
    d_sys_samples = np.sort(d_sys_samples)
    low_idx = int((alpha / 2.0) * num_bootstraps)
    high_idx = int((1.0 - alpha / 2.0) * num_bootstraps)
    high_idx = min(high_idx, len(d_sys_samples) - 1)
    
    return {
        "d_sys": d_sys_point,
        "ci_lower": float(d_sys_samples[low_idx]),
        "ci_upper": float(d_sys_samples[high_idx]),
        "bin_centers": eval_points,
        "d_omega": d_omega,
        "omega": omega,
        "bw_factor": bw_factor
    }


def compute_system_unlinkability_clustered_ci(mated_records: List[Dict[str, Any]], 
                                              non_mated_records: List[Dict[str, Any]],
                                              num_bins: int = 100, 
                                              score_domain: Optional[Tuple[float, float]] = (0.0, 1.0),
                                              num_bootstraps: int = 2000, 
                                              alpha: float = 0.05,
                                              method: str = "kde",
                                              omega: float = 1.0,
                                              bw_factor: float = 1.0,
                                              seed: int = 42) -> Dict[str, Any]:
    """
    Computes system unlinkability D_sys with subject-level clustered bootstrap confidence intervals.
    Resampling occurs strictly at the independent subject cluster level.
    """
    if not mated_records or not non_mated_records:
        raise ValueError("Both mated_records and non_mated_records must be non-empty for clustered unlinkability bootstrap.")
        
    mated_scores = np.array([r["score"] for r in mated_records], dtype=np.float64)
    non_mated_scores = np.array([r["score"] for r in non_mated_records], dtype=np.float64)
    
    d_sys_point, eval_points, d_omega, lr = compute_system_unlinkability(
        mated_scores, non_mated_scores, num_bins=num_bins, score_domain=score_domain, 
        method=method, omega=omega, bw_factor=bw_factor
    )
    
    subjects = sorted(list(set(
        [r["subject"] for r in mated_records] + 
        [r["subject_a"] for r in non_mated_records] + 
        [r["subject_b"] for r in non_mated_records]
    )))
    n_subj = len(subjects)
    if n_subj < 2:
        return {
            "d_sys": d_sys_point,
            "clustered_ci_lower": d_sys_point,
            "clustered_ci_upper": d_sys_point,
            "bin_centers": eval_points,
            "d_omega": d_omega,
            "omega": omega,
            "bw_factor": bw_factor
        }
        
    mated_by_subj: Dict[Any, List[float]] = {}
    for r in mated_records:
        mated_by_subj.setdefault(r["subject"], []).append(float(r["score"]))
        
    non_mated_by_pair: Dict[Tuple[Any, Any], List[float]] = {}
    for r in non_mated_records:
        pair = (r["subject_a"], r["subject_b"])
        non_mated_by_pair.setdefault(pair, []).append(float(r["score"]))
        
    rng = np.random.default_rng(seed)
    d_sys_samples = []
    
    for _ in range(num_bootstraps):
        sampled_subjs = rng.choice(subjects, size=n_subj, replace=True)
        sampled_set = set(sampled_subjs)
        
        boot_mated = []
        for s in sampled_subjs:
            if s in mated_by_subj:
                boot_mated.extend(mated_by_subj[s])
                
        boot_non_mated = []
        for (sa, sb), scores in non_mated_by_pair.items():
            if sa in sampled_set and sb in sampled_set:
                boot_non_mated.extend(scores)
                
        if len(boot_mated) >= 5 and len(boot_non_mated) >= 5:
            try:
                ds, _, _, _ = compute_system_unlinkability(
                    np.array(boot_mated), np.array(boot_non_mated),
                    num_bins=num_bins, score_domain=score_domain, 
                    method=method, omega=omega, bw_factor=bw_factor
                )
                d_sys_samples.append(ds)
            except Exception:
                pass
                
    if not d_sys_samples:
        return {
            "d_sys": d_sys_point,
            "clustered_ci_lower": d_sys_point,
            "clustered_ci_upper": d_sys_point,
            "bin_centers": eval_points,
            "d_omega": d_omega,
            "omega": omega,
            "bw_factor": bw_factor
        }
        
    d_sys_samples = np.sort(d_sys_samples)
    low_idx = int((alpha / 2.0) * len(d_sys_samples))
    high_idx = int((1.0 - alpha / 2.0) * len(d_sys_samples))
    high_idx = min(high_idx, len(d_sys_samples) - 1)
    
    return {
        "d_sys": d_sys_point,
        "clustered_ci_lower": float(d_sys_samples[low_idx]),
        "clustered_ci_upper": float(d_sys_samples[high_idx]),
        "bin_centers": eval_points,
        "d_omega": d_omega,
        "omega": omega,
        "bw_factor": bw_factor
    }


def compute_multi_threat_unlinkability_bootstrap(threat_records: Dict[str, Dict[str, List[Dict[str, Any]]]],
                                                 num_bins: int = 100,
                                                 score_domain: Optional[Tuple[float, float]] = (0.0, 1.0),
                                                 num_bootstraps: int = 2000,
                                                 alpha: float = 0.05,
                                                 method: str = "kde",
                                                 omega: float = 1.0,
                                                 bw_factor: float = 1.0,
                                                 seed: int = 42) -> Dict[str, Any]:
    """
    Performs joint multi-threat-model unlinkability evaluation with two-way cluster bootstrap
    across both subjects and application key pairs.
    
    Computes point estimates and 95% confidence intervals for:
    1. Descriptor-only attacker D_sys^desc
    2. Topology-only attacker D_sys^topo
    3. Combined attacker D_sys^comb
    4. Maximum attacker across threat models:
           D_sys^max(b) = max(D_sys^desc(b), D_sys^topo(b), D_sys^comb(b))
       evaluated per bootstrap replicate to yield the true joint confidence interval
       for maximum system linkage.
    """
    model_names = list(threat_records.keys())
    first_model = model_names[0]
    first_mated = threat_records[first_model]["mated"]
    first_non_mated = threat_records[first_model]["non_mated"]
    
    # 1. Point estimates for all models
    point_estimates: Dict[str, float] = {}
    for m_name in model_names:
        m_sc = np.array([r["score"] for r in threat_records[m_name]["mated"]], dtype=np.float64)
        nm_sc = np.array([r["score"] for r in threat_records[m_name]["non_mated"]], dtype=np.float64)
        ds, _, _, _ = compute_system_unlinkability(
            m_sc, nm_sc, num_bins=num_bins, score_domain=score_domain, 
            method=method, omega=omega, bw_factor=bw_factor
        )
        point_estimates[m_name] = ds
        
    d_sys_max_point = float(max(point_estimates.values()))
    
    # 2. Extract cluster keys: subjects and key pairs
    subjects = sorted(list(set(
        [r["subject"] for r in first_mated] + 
        [r["subject_a"] for r in first_non_mated] + 
        [r["subject_b"] for r in first_non_mated]
    )))
    n_subj = len(subjects)
    
    key_pairs = sorted(list(set(
        [(r["key1"], r["key2"]) for r in first_mated]
    )))
    n_kp = len(key_pairs)
    
    # Pre-index records by (subject, (key1, key2)) for mated and ((sa, sb), (key1, key2)) for non-mated
    mated_by_sk: Dict[str, Dict[Tuple[Any, Tuple[Any, Any]], float]] = {m: {} for m in model_names}
    for m_name in model_names:
        for r in threat_records[m_name]["mated"]:
            kp = (r["key1"], r["key2"])
            mated_by_sk[m_name][(r["subject"], kp)] = float(r["score"])
            
    non_mated_by_sk: Dict[str, Dict[Tuple[Tuple[Any, Any], Tuple[Any, Any]], float]] = {m: {} for m in model_names}
    for m_name in model_names:
        for r in threat_records[m_name]["non_mated"]:
            sa, sb = r["subject_a"], r["subject_b"]
            sp = (sa, sb) if sa <= sb else (sb, sa)
            kp = (r["key1"], r["key2"])
            non_mated_by_sk[m_name][(sp, kp)] = float(r["score"])
            
    # 3. Two-way cluster bootstrap over subjects and key pairs
    rng = np.random.default_rng(seed)
    model_bootstrap_samples: Dict[str, List[float]] = {m: [] for m in model_names}
    max_bootstrap_samples: List[float] = []
    
    for _ in range(num_bootstraps):
        sampled_subjs = set(rng.choice(subjects, size=n_subj, replace=True))
        sampled_kps = rng.choice(key_pairs, size=n_kp, replace=True)
        
        rep_estimates: Dict[str, float] = {}
        valid_rep = True
        
        for m_name in model_names:
            boot_mated = []
            for s in sampled_subjs:
                for kp in sampled_kps:
                    kp_tuple = (kp[0], kp[1])
                    val = mated_by_sk[m_name].get((s, kp_tuple))
                    if val is not None:
                        boot_mated.append(val)
                        
            boot_non_mated = []
            u_subjs = list(sampled_subjs)
            for idx_a in range(len(u_subjs)):
                sa = u_subjs[idx_a]
                for idx_b in range(idx_a + 1, len(u_subjs)):
                    sb = u_subjs[idx_b]
                    sp = (sa, sb) if sa <= sb else (sb, sa)
                    for kp in sampled_kps:
                        kp_tuple = (kp[0], kp[1])
                        val = non_mated_by_sk[m_name].get((sp, kp_tuple))
                        if val is not None:
                            boot_non_mated.append(val)
                            
            if len(boot_mated) >= 5 and len(boot_non_mated) >= 5:
                try:
                    ds, _, _, _ = compute_system_unlinkability(
                        np.array(boot_mated), np.array(boot_non_mated),
                        num_bins=num_bins, score_domain=score_domain, 
                        method=method, omega=omega, bw_factor=bw_factor
                    )
                    rep_estimates[m_name] = ds
                except Exception:
                    valid_rep = False
                    break
            else:
                valid_rep = False
                break
                
        if valid_rep and len(rep_estimates) == len(model_names):
            for m_name in model_names:
                model_bootstrap_samples[m_name].append(rep_estimates[m_name])
            max_bootstrap_samples.append(float(max(rep_estimates.values())))
            
    # 4. Compute confidence intervals
    results = {
        "point_estimates": point_estimates,
        "d_sys_max_point": d_sys_max_point,
        "models": {},
        "omega": omega,
        "bw_factor": bw_factor,
        "num_bootstraps_completed": len(max_bootstrap_samples)
    }
    
    for m_name in model_names:
        samples = np.sort(model_bootstrap_samples[m_name])
        if len(samples) > 0:
            l_idx = int((alpha / 2.0) * len(samples))
            h_idx = min(int((1.0 - alpha / 2.0) * len(samples)), len(samples) - 1)
            ci = [float(samples[l_idx]), float(samples[h_idx])]
        else:
            ci = [point_estimates[m_name], point_estimates[m_name]]
        results["models"][m_name] = {
            "d_sys": point_estimates[m_name],
            "ci_95": ci
        }
        
    max_samples = np.sort(max_bootstrap_samples)
    if len(max_samples) > 0:
        l_idx = int((alpha / 2.0) * len(max_samples))
        h_idx = min(int((1.0 - alpha / 2.0) * len(max_samples)), len(max_samples) - 1)
        max_ci = [float(max_samples[l_idx]), float(max_samples[h_idx])]
    else:
        max_ci = [d_sys_max_point, d_sys_max_point]
        
    results["d_sys_max_ci_95"] = max_ci
    return results


def match_topology_only(adj1: np.ndarray, adj2: np.ndarray) -> float:
    """
    Computes graph-only cross-matching similarity between two dual incidence matrices A1 and A2.
    Evaluates degree histogram and mesh size similarity.
    
    Returns:
        similarity: float in [0.0, 1.0].
    """
    n1 = len(adj1)
    n2 = len(adj2)
    if n1 == 0 or n2 == 0:
        return 0.0
        
    # Degrees (number of shared edges, A[i, j] == 2)
    deg1 = np.sum(adj1 == 2, axis=1) if n1 > 1 else np.zeros(n1)
    deg2 = np.sum(adj2 == 2, axis=1) if n2 > 1 else np.zeros(n2)
    
    # 4-bin degree histogram (0, 1, 2, 3 neighbors)
    h1, _ = np.histogram(deg1, bins=[-0.5, 0.5, 1.5, 2.5, 3.5])
    h2, _ = np.histogram(deg2, bins=[-0.5, 0.5, 1.5, 2.5, 3.5])
    
    p1 = h1 / max(np.sum(h1), 1)
    p2 = h2 / max(np.sum(h2), 1)
    
    l1_dist = float(np.sum(np.abs(p1 - p2)))  # in [0, 2]
    degree_sim = float(np.exp(-l1_dist))
    
    # Size penalty
    size_sim = 2.0 * min(n1, n2) / (n1 + n2)
    return float(np.clip(degree_sim * size_sim, 0.0, 1.0))


def evaluate_unlinkability(fingerprint_descriptors: List[np.ndarray],
                           fingerprint_weights: List[np.ndarray],
                           derive_matrix_fn,
                           matcher_fn,
                           num_keys: int = 5) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluates mated vs non-mated cross-key score distributions under a balanced design.
    """
    n_subj = len(fingerprint_descriptors)
    mated_scores = []
    non_mated_scores = []
    
    keys = [f"key_{i}" for i in range(num_keys)]
    
    # Evaluate across all K(K-1)/2 key pairs in balanced fashion
    key_matrices = [derive_matrix_fn(k) for k in keys]
    
    for k1 in range(num_keys):
        for k2 in range(k1 + 1, num_keys):
            W1 = key_matrices[k1]
            W2 = key_matrices[k2]
            
            # Mated: same subject across key pair (k1, k2)
            for i in range(n_subj):
                desc = fingerprint_descriptors[i]
                w = fingerprint_weights[i]
                if len(desc) == 0:
                    continue
                t1 = (desc @ W1.T >= 0.0).astype(np.uint8)
                t2 = (desc @ W2.T >= 0.0).astype(np.uint8)
                s = matcher_fn(t1, t2, w, w)
                mated_scores.append(s)
                
            # Non-mated: different subjects across key pair (k1, k2)
            for i in range(n_subj):
                desc_i = fingerprint_descriptors[i]
                w_i = fingerprint_weights[i]
                if len(desc_i) == 0:
                    continue
                for j in range(i + 1, n_subj):
                    desc_j = fingerprint_descriptors[j]
                    w_j = fingerprint_weights[j]
                    if len(desc_j) == 0:
                        continue
                    t1 = (desc_i @ W1.T >= 0.0).astype(np.uint8)
                    t2 = (desc_j @ W2.T >= 0.0).astype(np.uint8)
                    s = matcher_fn(t1, t2, w_i, w_j)
                    non_mated_scores.append(s)
                    
    mated_arr = np.array(mated_scores, dtype=np.float64)
    non_mated_arr = np.array(non_mated_scores, dtype=np.float64)
    
    d_sys, _, _, _ = compute_system_unlinkability(mated_arr, non_mated_arr, score_domain=(0.0, 1.0))
    return d_sys, mated_arr, non_mated_arr
