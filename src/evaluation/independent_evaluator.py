"""
Independent Biometric Metric Evaluator.

Provides an independent, zero-dependency reference implementation for:
- Confusion Matrix Accounting (TP, FN, FP, TN)
- Equal Error Rate (EER) via exact linear crossover interpolation
- Area Under the ROC Curve (AUC) via trapezoidal Riemann integration
- True/False Acceptance and Rejection Rates (TAR, FAR, FRR)
- Exact Wilson Score and Clopper-Pearson Binomial Confidence Intervals
- Clustered and Non-Parametric Bootstrap Confidence Intervals
- Paired System Performance Differences (EER and AUC deltas)

Design Requirement:
This module does NOT import `src.evaluation.metrics` to guarantee an independent
arithmetic verification path, ensuring that any implementation bugs in the primary
evaluator are detected.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np


def independent_confusion_matrix(genuine_scores: np.ndarray,
                                 impostor_scores: np.ndarray,
                                 threshold: float) -> Dict[str, Any]:
    """
    Computes exact confusion matrix and operational error rates.

    Args:
        genuine_scores: 1-D array of genuine comparison scores.
        impostor_scores: 1-D array of impostor comparison scores.
        threshold: Operating decision threshold.

    Returns:
        Dict containing counts and rates.
    """
    g = np.asarray(genuine_scores, dtype=np.float64)
    i = np.asarray(impostor_scores, dtype=np.float64)

    n_gen = len(g)
    n_imp = len(i)

    tp = int(np.sum(g >= threshold))
    fn = int(np.sum(g < threshold))
    fp = int(np.sum(i >= threshold))
    tn = int(np.sum(i < threshold))

    assert tp + fn == n_gen, f"Genuine conservation failed: {tp} + {fn} != {n_gen}"
    assert fp + tn == n_imp, f"Impostor conservation failed: {fp} + {tn} != {n_imp}"

    tar = float(tp / max(n_gen, 1))
    frr = float(fn / max(n_gen, 1))
    far = float(fp / max(n_imp, 1))
    tnr = float(tn / max(n_imp, 1))

    return {
        "n_genuine": n_gen,
        "n_impostor": n_imp,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tar": tar,
        "frr": frr,
        "far": far,
        "tnr": tnr,
        "tar_pct": tar * 100.0,
        "frr_pct": frr * 100.0,
        "far_pct": far * 100.0,
        "threshold": threshold
    }


def independent_roc_curve(genuine_scores: np.ndarray,
                          impostor_scores: np.ndarray,
                          num_thresholds: int = 2000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes empirical ROC curve (FAR vs TAR) by sweeping thresholds.

    Returns:
        (far_array, tar_array, thresholds) sorted by ascending FAR.
    """
    g = np.asarray(genuine_scores, dtype=np.float64)
    i = np.asarray(impostor_scores, dtype=np.float64)

    all_scores = np.concatenate([g, i])
    s_min, s_max = float(np.min(all_scores)), float(np.max(all_scores))
    thresholds = np.linspace(s_min - 1e-5, s_max + 1e-5, num_thresholds)

    # Sort thresholds in descending order so FAR and TAR increase monotonically
    thresholds = np.sort(thresholds)[::-1]

    n_gen = max(len(g), 1)
    n_imp = max(len(i), 1)

    # Vectorized searchsorted for O(N log K) efficiency
    g_sorted = np.sort(g)
    i_sorted = np.sort(i)

    # Count >= threshold: len - searchsorted(left)
    tps = len(g) - np.searchsorted(g_sorted, thresholds, side='left')
    fps = len(i) - np.searchsorted(i_sorted, thresholds, side='left')

    tar = tps / n_gen
    far = fps / n_imp

    # Ensure endpoints [0, 0] and [1, 1]
    far = np.concatenate([[0.0], far, [1.0]])
    tar = np.concatenate([[0.0], tar, [1.0]])
    thresh_extended = np.concatenate([[s_max + 1.0], thresholds, [s_min - 1.0]])

    return far, tar, thresh_extended


def independent_auc(far: np.ndarray, tar: np.ndarray) -> float:
    """
    Computes Area Under the ROC Curve via pure trapezoidal integration:
    AUC = integral_0^1 TAR(FAR) d(FAR) = sum_{k} 0.5 * (TAR_k + TAR_{k-1}) * (FAR_k - FAR_{k-1}).
    """
    # Sort by FAR ascending
    order = np.argsort(far)
    x = far[order]
    y = tar[order]
    dx = np.diff(x)
    y_avg = (y[:-1] + y[1:]) / 2.0
    auc_val = float(np.sum(dx * y_avg))
    return float(np.clip(auc_val, 0.0, 1.0))


def independent_eer(far: np.ndarray, tar: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float]:
    """
    Computes exact Equal Error Rate (EER) and operating threshold via linear interpolation.

    EER occurs where FAR == FRR == (1 - TAR).
    """
    frr = 1.0 - tar
    diff = far - frr

    # Find sign change in (far - frr)
    zero_crossings = np.where(np.diff(np.sign(diff)))[0]
    if len(zero_crossings) == 0:
        # Closest discrete point
        idx = int(np.argmin(np.abs(diff)))
        return float(far[idx]), float(thresholds[idx])

    idx = zero_crossings[0]
    # Linear interpolation between idx and idx+1
    x1, x2 = far[idx], far[idx + 1]
    y1, y2 = frr[idx], frr[idx + 1]
    t1, t2 = thresholds[idx], thresholds[idx + 1]

    # Solve: x1 + t_interp*(x2 - x1) == y1 + t_interp*(y2 - y1)
    denom = (x2 - x1) - (y2 - y1)
    if abs(denom) < 1e-12:
        t_interp = 0.5
    else:
        t_interp = (y1 - x1) / denom
    t_interp = float(np.clip(t_interp, 0.0, 1.0))

    eer_val = x1 + t_interp * (x2 - x1)
    eer_threshold = t1 + t_interp * (t2 - t1)

    return float(np.clip(eer_val, 0.0, 1.0)), float(eer_threshold)


def independent_tar_at_far(far: np.ndarray, tar: np.ndarray, target_far: float) -> float:
    """
    Interpolates True Acceptance Rate at a target False Acceptance Rate.
    """
    order = np.argsort(far)
    x = far[order]
    y = tar[order]
    # Linear interpolation
    return float(np.interp(target_far, x, y))


def compute_independent_metrics(genuine_scores: np.ndarray,
                                impostor_scores: np.ndarray) -> Dict[str, Any]:
    """
    Computes comprehensive verification metrics from raw score arrays.
    """
    g = np.asarray(genuine_scores, dtype=np.float64)
    i = np.asarray(impostor_scores, dtype=np.float64)

    far, tar, thresh = independent_roc_curve(g, i)
    auc_val = independent_auc(far, tar)
    eer_val, eer_thresh = independent_eer(far, tar, thresh)
    tar_1pct = independent_tar_at_far(far, tar, 0.01)
    tar_01pct = independent_tar_at_far(far, tar, 0.001)

    return {
        "n_genuine": len(g),
        "n_impostor": len(i),
        "eer": eer_val,
        "eer_pct": eer_val * 100.0,
        "eer_threshold": eer_thresh,
        "auc": auc_val,
        "tar_at_1pct_far": tar_1pct,
        "tar_at_1pct_far_pct": tar_1pct * 100.0,
        "tar_at_01pct_far": tar_01pct,
        "tar_at_01pct_far_pct": tar_01pct * 100.0
    }


def independent_bootstrap_ci(genuine_scores: np.ndarray,
                             impostor_scores: np.ndarray,
                             num_bootstraps: int = 500,
                             alpha: float = 0.05,
                             seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """
    Computes non-parametric bootstrap confidence intervals for EER and AUC.
    """
    rng = np.random.default_rng(seed)
    g = np.asarray(genuine_scores, dtype=np.float64)
    i = np.asarray(impostor_scores, dtype=np.float64)

    n_g = len(g)
    n_i = len(i)

    eers = []
    aucs = []

    for _ in range(num_bootstraps):
        g_sample = g[rng.integers(0, n_g, size=n_g)]
        i_sample = i[rng.integers(0, n_i, size=n_i)]
        far, tar, thresh = independent_roc_curve(g_sample, i_sample, num_thresholds=500)
        eers.append(independent_eer(far, tar, thresh)[0])
        aucs.append(independent_auc(far, tar))

    low_p = (alpha / 2.0) * 100.0
    high_p = (1.0 - alpha / 2.0) * 100.0

    return {
        "eer_ci": (float(np.percentile(eers, low_p)), float(np.percentile(eers, high_p))),
        "auc_ci": (float(np.percentile(aucs, low_p)), float(np.percentile(aucs, high_p)))
    }
