import warnings
from typing import Tuple, Dict, Any, List
import numpy as np

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Computes exact two-sided Clopper-Pearson binomial confidence interval for k observed
    errors/events in n independent Bernoulli trials.
    
    NOTE: Biometric verification across all pairwise combinations violates the independent 
    Bernoulli assumption due to subject/finger reuse. For pairwise evaluations, use
    clustered_bootstrap_confidence_intervals instead.
    
    Args:
        k: Observed event count (e.g. false accepts or false rejections).
        n: Total number of trials.
        alpha: Significance level (default: 0.05 for 95% confidence).
        
    Returns:
        (lower_bound, upper_bound) as floats in [0.0, 1.0].
    """
    from scipy.stats import beta
    if n == 0:
        return (0.0, 1.0)
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lower, upper)

def compute_roc_metrics(genuine_scores: np.ndarray, impostor_scores: np.ndarray, 
                        num_thresholds: int = 1000,
                        warn_on_inversion: bool = True) -> Dict:
    """
    Computes complete biometric verification curves: FAR, FRR, TAR, ROC, exact AUC, and EER.
    Standardized to higher-is-more-genuine similarity scores in [0, 1].
    
    Includes an automatic score directionality diagnostic warning if mean(genuine) <= mean(impostor).
    
    Args:
        genuine_scores: 1D array of genuine comparison similarity scores.
        impostor_scores: 1D array of impostor comparison similarity scores.
        num_thresholds: Number of evaluation thresholds.
        warn_on_inversion: Whether to issue a UserWarning if genuine mean <= impostor mean.
        
    Returns:
        dict containing 'thresholds', 'far', 'frr', 'tar', 'eer', 'auc', 'tar_at_1pct_far', 'tar_at_01pct_far'.
    """
    if warn_on_inversion and len(genuine_scores) > 0 and len(impostor_scores) > 0:
        mu_gen = float(np.mean(genuine_scores))
        mu_imp = float(np.mean(impostor_scores))
        if mu_gen <= mu_imp:
            warnings.warn(
                f"Score inversion diagnostic triggered: genuine mean ({mu_gen:.4f}) "
                f"<= impostor mean ({mu_imp:.4f}). Matching similarity scores must be higher for genuine comparisons. "
                f"Check descriptor standardization and sign.",
                UserWarning,
                stacklevel=2
            )
            
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    s_min, s_max = float(np.min(all_scores)), float(np.max(all_scores))
    thresholds = np.linspace(s_min - 1e-4, s_max + 1e-4, num_thresholds)
    
    n_gen = len(genuine_scores)
    n_imp = len(impostor_scores)
    
    far = np.zeros(num_thresholds, dtype=np.float64)
    frr = np.zeros(num_thresholds, dtype=np.float64)
    
    for i, t in enumerate(thresholds):
        # Score >= t means ACCEPT
        fp = np.sum(impostor_scores >= t)
        fn = np.sum(genuine_scores < t)
        far[i] = fp / n_imp
        frr[i] = fn / n_gen
        
    tar = 1.0 - frr
    
    # Calculate EER (point where FAR == FRR)
    diff = np.abs(far - frr)
    min_idx = np.argmin(diff)
    
    # Linear interpolation around min_idx
    if min_idx > 0 and min_idx < num_thresholds - 1:
        x0, x1 = thresholds[min_idx - 1], thresholds[min_idx + 1]
        far0, far1 = far[min_idx - 1], far[min_idx + 1]
        frr0, frr1 = frr[min_idx - 1], frr[min_idx + 1]
        denom = (far1 - far0) - (frr1 - frr0)
        if abs(denom) > 1e-9:
            alpha = (frr0 - far0) / denom
            eer = float(np.clip(far0 + alpha * (far1 - far0), 0.0, 1.0))
        else:
            eer = float((far[min_idx] + frr[min_idx]) / 2.0)
    else:
        eer = float((far[min_idx] + frr[min_idx]) / 2.0)
        
    # Exact non-parametric Area Under ROC curve via Mann-Whitney U statistic:
    # AUC = P(Score_genuine > Score_impostor) + 0.5 * P(Score_genuine == Score_impostor)
    # Eliminates discrete trapezoidal integration approximation errors.
    # Uses rankdata for exact O(N log N) computational scaling across large bootstrap sets.
    from scipy.stats import rankdata
    ranks = rankdata(np.concatenate([genuine_scores, impostor_scores]))
    u_stat = float(np.sum(ranks[:n_gen]) - n_gen * (n_gen + 1.0) / 2.0)
    exact_auc = float(u_stat / (n_gen * n_imp))
    auc = float(np.clip(exact_auc, 0.0, 1.0))
    
    # Calculate TAR at fixed FAR values
    tar_1pct = calculate_tar_at_far(far, tar, target_far=0.01)
    tar_01pct = calculate_tar_at_far(far, tar, target_far=0.001)
    
    return {
        "thresholds": thresholds,
        "far": far,
        "frr": frr,
        "tar": tar,
        "eer": eer,
        "auc": auc,
        "tar_at_1pct_far": tar_1pct,
        "tar_at_01pct_far": tar_01pct,
        "eer_threshold": float(thresholds[min_idx])
    }

def compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
    """Convenience function returning just the EER float."""
    metrics = compute_roc_metrics(genuine_scores, impostor_scores)
    return metrics["eer"]

def calculate_tar_at_far(far: np.ndarray, tar: np.ndarray, target_far: float) -> float:
    """Finds the maximum TAR achievable such that FAR <= target_far."""
    valid_indices = np.where(far <= target_far)[0]
    if len(valid_indices) == 0:
        return 0.0
    return float(np.max(tar[valid_indices]))

def bootstrap_confidence_intervals(genuine_scores: np.ndarray, impostor_scores: np.ndarray, 
                                   num_bootstraps: int = 500, alpha: float = 0.05) -> Dict:
    """
    Computes 95% bootstrap confidence intervals for EER and AUC.
    """
    rng = np.random.default_rng(42)
    n_gen = len(genuine_scores)
    n_imp = len(impostor_scores)
    
    eers = []
    aucs = []
    
    for _ in range(num_bootstraps):
        g_sample = rng.choice(genuine_scores, size=n_gen, replace=True)
        i_sample = rng.choice(impostor_scores, size=n_imp, replace=True)
        m = compute_roc_metrics(g_sample, i_sample, num_thresholds=200, warn_on_inversion=False)
        eers.append(m["eer"])
        aucs.append(m["auc"])
        
    eers = np.sort(eers)
    aucs = np.sort(aucs)
    
    low_idx = int((alpha / 2.0) * num_bootstraps)
    high_idx = int((1.0 - alpha / 2.0) * num_bootstraps)
    
    return {
        "eer_mean": float(np.mean(eers)),
        "eer_ci": (float(eers[low_idx]), float(eers[high_idx])),
        "auc_mean": float(np.mean(aucs)),
        "auc_ci": (float(aucs[low_idx]), float(aucs[high_idx]))
    }

def clustered_bootstrap_confidence_intervals(genuine_records: list, impostor_records: list,
                                             operating_threshold: float = 0.20,
                                             num_bootstraps: int = 500,
                                             alpha: float = 0.05,
                                             seed: int = 42) -> Dict:
    """
    Computes subject/finger-level clustered bootstrap confidence intervals
    to account for within-subject correlations across repeated impressions.
    Resamples entire finger clusters with replacement preserving intra-subject dependencies.
    
    Args:
        genuine_records: List of dicts with 'subject_a', 'subject_b', 'score'.
        impostor_records: List of dicts with 'subject_a', 'subject_b', 'score'.
        operating_threshold: Operating point for FAR/FRR.
        num_bootstraps: Number of cluster bootstrap iterations.
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed.
        
    Returns:
        dict with clustered confidence intervals for FAR, FRR, EER, and AUC.
    """
    from collections import Counter
    all_subjects = np.unique(
        [r['subject_a'] for r in genuine_records] + 
        [r['subject_a'] for r in impostor_records] + 
        [r['subject_b'] for r in impostor_records]
    )
    n_subj = len(all_subjects)
    rng = np.random.default_rng(seed)
    
    gen_by_subj: Dict[Any, List[float]] = {}
    for r in genuine_records:
        gen_by_subj.setdefault(r['subject_a'], []).append(float(r['score']))
        
    imp_by_pair: Dict[Tuple[Any, Any], List[float]] = {}
    for r in impostor_records:
        sa, sb = r['subject_a'], r['subject_b']
        key = (sa, sb) if sa <= sb else (sb, sa)
        imp_by_pair.setdefault(key, []).append(float(r['score']))
    
    boot_fars, boot_frrs = [], []
    boot_eers, boot_aucs = [], []
    
    for _ in range(num_bootstraps):
        sampled_subjs = rng.choice(all_subjects, size=n_subj, replace=True)
        counts = Counter(sampled_subjs)
        
        g_scores = []
        for s, cnt in counts.items():
            if s in gen_by_subj:
                scores = gen_by_subj[s]
                for _ in range(cnt):
                    g_scores.extend(scores)
                    
        i_scores = []
        u_subjs = list(counts.keys())
        for idx_a in range(len(u_subjs)):
            sa = u_subjs[idx_a]
            ca = counts[sa]
            for idx_b in range(idx_a + 1, len(u_subjs)):
                sb = u_subjs[idx_b]
                cb = counts[sb]
                key = (sa, sb) if sa <= sb else (sb, sa)
                if key in imp_by_pair:
                    scores = imp_by_pair[key]
                    reps = ca * cb
                    for _ in range(reps):
                        i_scores.extend(scores)
        
        if len(g_scores) >= 5 and len(i_scores) >= 5:
            g_arr = np.array(g_scores, dtype=np.float64)
            i_arr = np.array(i_scores, dtype=np.float64)
            boot_fars.append(float(np.mean(i_arr >= operating_threshold)))
            boot_frrs.append(float(np.mean(g_arr < operating_threshold)))
            
            roc_m = compute_roc_metrics(g_arr, i_arr, num_thresholds=100, warn_on_inversion=False)
            boot_eers.append(roc_m["eer"])
            boot_aucs.append(roc_m["auc"])
            
    if not boot_fars:
        return {
            "clustered_far_ci": (0.0, 0.0), 
            "clustered_frr_ci": (0.0, 0.0),
            "eer_ci": (0.0, 0.0),
            "auc_ci": (0.0, 0.0),
            "eer_mean": 0.0,
            "auc_mean": 0.0
        }
        
    low_idx = int((alpha / 2.0) * len(boot_fars))
    high_idx = int((1.0 - alpha / 2.0) * len(boot_fars))
    high_idx = min(high_idx, len(boot_fars) - 1)
    
    boot_fars.sort()
    boot_frrs.sort()
    boot_eers.sort()
    boot_aucs.sort()
    
    return {
        "clustered_far_ci": (float(boot_fars[low_idx]), float(boot_fars[high_idx])),
        "clustered_frr_ci": (float(boot_frrs[low_idx]), float(boot_frrs[high_idx])),
        "eer_ci": (float(boot_eers[low_idx]), float(boot_eers[high_idx])),
        "auc_ci": (float(boot_aucs[low_idx]), float(boot_aucs[high_idx])),
        "eer_mean": float(np.mean(boot_eers)),
        "auc_mean": float(np.mean(boot_aucs))
    }


def paired_clustered_bootstrap_delta(genuine_pairs_prot: List[Dict[str, Any]],
                                     impostor_pairs_prot: List[Dict[str, Any]],
                                     genuine_pairs_unprot: List[Dict[str, Any]],
                                     impostor_pairs_unprot: List[Dict[str, Any]],
                                     num_bootstraps: int = 500,
                                     alpha: float = 0.05,
                                     seed: int = 42) -> Dict[str, Any]:
    """
    Computes paired finger-clustered bootstrap 95% confidence intervals for the
    differences between protected and unprotected biometric systems:
        Delta_EER = EER_prot - EER_unprot
        Delta_AUC = AUC_prot - AUC_unprot
        
    Resampling occurs strictly at the independent finger cluster level,
    preserving intra-subject dependencies and exact pairwise comparison alignments.
    """
    from collections import Counter
    all_subjects = np.unique(
        [r['subject_a'] for r in genuine_pairs_prot] + 
        [r['subject_a'] for r in impostor_pairs_prot] + 
        [r['subject_b'] for r in impostor_pairs_prot]
    )
    n_subj = len(all_subjects)
    rng = np.random.default_rng(seed)
    
    # Pre-index paired scores by cluster
    gen_by_subj: Dict[Any, List[Tuple[float, float]]] = {}
    for rp, ru in zip(genuine_pairs_prot, genuine_pairs_unprot):
        sa = rp['subject_a']
        gen_by_subj.setdefault(sa, []).append((float(rp['score']), float(ru['score'])))
        
    imp_by_pair: Dict[Tuple[Any, Any], List[Tuple[float, float]]] = {}
    for rp, ru in zip(impostor_pairs_prot, impostor_pairs_unprot):
        sa, sb = rp['subject_a'], rp['subject_b']
        key = (sa, sb) if sa <= sb else (sb, sa)
        imp_by_pair.setdefault(key, []).append((float(rp['score']), float(ru['score'])))
        
    # Point estimates on full sample
    gp_full = np.array([r['score'] for r in genuine_pairs_prot], dtype=np.float64)
    ip_full = np.array([r['score'] for r in impostor_pairs_prot], dtype=np.float64)
    gu_full = np.array([r['score'] for r in genuine_pairs_unprot], dtype=np.float64)
    iu_full = np.array([r['score'] for r in impostor_pairs_unprot], dtype=np.float64)
    
    roc_p_full = compute_roc_metrics(gp_full, ip_full, warn_on_inversion=False)
    roc_u_full = compute_roc_metrics(gu_full, iu_full, warn_on_inversion=False)
    delta_eer_point = float(roc_p_full["eer"] - roc_u_full["eer"])
    delta_auc_point = float(roc_p_full["auc"] - roc_u_full["auc"])
    
    delta_eers = []
    delta_aucs = []
    
    for _ in range(num_bootstraps):
        sampled_subjs = rng.choice(all_subjects, size=n_subj, replace=True)
        counts = Counter(sampled_subjs)
        
        gp_list, gu_list = [], []
        for s, cnt in counts.items():
            if s in gen_by_subj:
                pairs = gen_by_subj[s]
                for _ in range(cnt):
                    for sp, su in pairs:
                        gp_list.append(sp)
                        gu_list.append(su)
                        
        ip_list, iu_list = [], []
        u_subjs = list(counts.keys())
        for idx_a in range(len(u_subjs)):
            sa = u_subjs[idx_a]
            ca = counts[sa]
            for idx_b in range(idx_a + 1, len(u_subjs)):
                sb = u_subjs[idx_b]
                cb = counts[sb]
                key = (sa, sb) if sa <= sb else (sb, sa)
                if key in imp_by_pair:
                    pairs = imp_by_pair[key]
                    reps = ca * cb
                    for _ in range(reps):
                        for sp, su in pairs:
                            ip_list.append(sp)
                            iu_list.append(su)
                            
        if len(gp_list) >= 5 and len(ip_list) >= 5:
            m_prot = compute_roc_metrics(np.array(gp_list), np.array(ip_list), num_thresholds=100, warn_on_inversion=False)
            m_unprot = compute_roc_metrics(np.array(gu_list), np.array(iu_list), num_thresholds=100, warn_on_inversion=False)
            delta_eers.append(float(m_prot["eer"] - m_unprot["eer"]))
            delta_aucs.append(float(m_prot["auc"] - m_unprot["auc"]))
            
    if not delta_eers:
        return {
            "delta_eer_point": delta_eer_point,
            "delta_eer_95ci": [delta_eer_point, delta_eer_point],
            "delta_auc_point": delta_auc_point,
            "delta_auc_95ci": [delta_auc_point, delta_auc_point]
        }
        
    delta_eers.sort()
    delta_aucs.sort()
    low_idx = int((alpha / 2.0) * len(delta_eers))
    high_idx = int((1.0 - alpha / 2.0) * len(delta_eers))
    high_idx = min(high_idx, len(delta_eers) - 1)
    
    return {
        "delta_eer_point": delta_eer_point,
        "delta_eer_95ci": [float(delta_eers[low_idx]), float(delta_eers[high_idx])],
        "delta_auc_point": delta_auc_point,
        "delta_auc_95ci": [float(delta_aucs[low_idx]), float(delta_aucs[high_idx])],
        "delta_eer_mean": float(np.mean(delta_eers)),
        "delta_auc_mean": float(np.mean(delta_aucs))
    }

def compute_confusion_at_threshold(genuine_scores: np.ndarray, impostor_scores: np.ndarray, 
                                   threshold: float, alpha: float = 0.05) -> Dict:
    """
    Computes exact confusion matrix and biometric verification rates at a fixed decision threshold T_accept.
    Decision Rule: Accept genuine if score >= threshold, Reject impostor if score < threshold.
    """
    tp = int(np.sum(genuine_scores >= threshold))
    fn = int(np.sum(genuine_scores < threshold))
    fp = int(np.sum(impostor_scores >= threshold))
    tn = int(np.sum(impostor_scores < threshold))
    
    n_gen = len(genuine_scores)
    n_imp = len(impostor_scores)
    
    tar = float(tp / max(n_gen, 1))
    frr = float(fn / max(n_gen, 1))
    far = float(fp / max(n_imp, 1))
    trr = float(tn / max(n_imp, 1))
    
    tar_ci = clopper_pearson_ci(tp, n_gen, alpha=alpha)
    frr_ci = clopper_pearson_ci(fn, n_gen, alpha=alpha)
    far_ci = clopper_pearson_ci(fp, n_imp, alpha=alpha)
    trr_ci = clopper_pearson_ci(tn, n_imp, alpha=alpha)
    
    return {
        "threshold": float(threshold),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "n_genuine": n_gen, "n_impostor": n_imp,
        "tar": tar, "frr": frr, "far": far, "trr": trr,
        "tar_95ci": [float(tar_ci[0]), float(tar_ci[1])],
        "frr_95ci": [float(frr_ci[0]), float(frr_ci[1])],
        "far_95ci": [float(far_ci[0]), float(far_ci[1])],
        "trr_95ci": [float(trr_ci[0]), float(trr_ci[1])]
    }


# Re-export Gómez-Barrero unlinkability metrics
from src.security.unlinkability import compute_system_unlinkability, compute_local_unlinkability
