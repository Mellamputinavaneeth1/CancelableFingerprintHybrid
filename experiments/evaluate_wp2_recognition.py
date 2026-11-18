"""
WP2: Recognition and Matching Improvement Evaluation Runner.
Evaluates Delaunay simplex stability enhancements, triangle degeneracy filtering,
coverage-aware score normalization, and Hungarian vs Greedy 1-to-1 matching on
FVC2002 DB1_A (and DB1_B).

Reports:
1. Genuine and impostor score distribution statistics (mean, std, median, IQR, min, max).
2. Descriptive ROC metrics: EER, AUC, TAR @ 1% FAR, TAR @ 0.1% FAR.
3. Operational metrics under frozen Set B thresholds (T_EER=0.3751, T_1%=0.4497, T_0.1%=0.4705):
   TAR, FAR, FRR, and confusion matrix (TP, FN, FP, TN).
4. Comparative analysis across baseline, degeneracy filtered, coverage normalized, and greedy configurations.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import CancelableFingerprintPipeline, init_gpu_acceleration
from src.graph.delaunay import build_delaunay_triangles
from src.graph.descriptors import extract_all_descriptors
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors
from src.matching.bipartite_matcher import match_protected_templates, match_protected_greedy
from src.evaluation.metrics import (
    compute_roc_metrics,
    bootstrap_confidence_intervals,
    clustered_bootstrap_confidence_intervals
)
from experiments.datasets.fvc_loader import load_fvc_dataset
from experiments.protocols.fvc2002_protocol import generate_fvc_protocol_pairs

FROZEN_SET_B_THRESHOLDS = {
    "T_EER": 0.3751,
    "T_1pct_FAR": 0.4497,
    "T_01pct_FAR": 0.4705
}

DEV_CALIBRATED_SET_B_THRESHOLDS = {
    "T_EER": 0.3620,
    "T_1pct_FAR": 0.4545,
    "T_01pct_FAR": 0.4545
}


def compute_distribution_stats(scores: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive descriptive distribution statistics."""
    if len(scores) == 0:
        return {}
    q25, q75 = np.percentile(scores, [25, 75])
    return {
        "count": len(scores),
        "mean": round(float(np.mean(scores)), 6),
        "std": round(float(np.std(scores)), 6),
        "median": round(float(np.median(scores)), 6),
        "iqr": round(float(q75 - q25), 6),
        "p25": round(float(q25), 6),
        "p75": round(float(q75), 6),
        "min": round(float(np.min(scores)), 6),
        "max": round(float(np.max(scores)), 6)
    }


def evaluate_operational_point(
    g_scores: np.ndarray,
    i_scores: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """Computes operational verification metrics at a fixed decision threshold."""
    n_gen = len(g_scores)
    n_imp = len(i_scores)
    tp = int(np.sum(g_scores >= threshold))
    fn = int(np.sum(g_scores < threshold))
    fp = int(np.sum(i_scores >= threshold))
    tn = int(np.sum(i_scores < threshold))

    assert tp + fn == n_gen, "TP + FN != N_gen"
    assert fp + tn == n_imp, "FP + TN != N_imp"

    tar = float(tp / max(n_gen, 1))
    far = float(fp / max(n_imp, 1))
    frr = float(fn / max(n_gen, 1))

    return {
        "threshold": threshold,
        "tar_pct": round(tar * 100.0, 2),
        "far_pct": round(far * 100.0, 2),
        "frr_pct": round(frr * 100.0, 2),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn}
    }


def evaluate_wp2_subset(
    subset: str = "DB1_A",
    output_dir: str = "experiments/results/wp2"
) -> Dict[str, Any]:
    """
    Evaluates DB1_A across WP2 candidate configurations:
    1. Baseline (standard Delaunay, Hungarian, no coverage penalty)
    2. Enhanced Delaunay Simplex Stability (degeneracy filtering)
    3. Coverage-Aware Score Normalization
    4. Greedy 1-to-1 Assignment (throughput-optimized)
    5. Enhanced Simplex Stability + Greedy 1-to-1
    6. Fully Combined: Simplex Stability + Coverage Normalization + Greedy 1-to-1
    7. Fully Combined: Simplex Stability + Coverage Normalization + Hungarian
    """
    os.makedirs(output_dir, exist_ok=True)
    gpu_info = init_gpu_acceleration(verbose=True)
    print("\n" + "=" * 80)
    print(f"  WP2: Recognition and Matching Improvement Evaluation on {subset}")
    print("=" * 80)

    t0 = time.time()
    ds = load_fvc_dataset(subset, auto_download=False)
    gen_pairs, imp_pairs = generate_fvc_protocol_pairs(ds)
    all_pairs = gen_pairs + imp_pairs
    n_gen = len(gen_pairs)
    n_imp = len(imp_pairs)
    print(f"[{subset}] Loaded {len(ds['all_images'])} images, {n_gen} genuine, {n_imp} impostor comparisons (Total: {len(all_pairs)}).")

    # Configurations to evaluate
    configurations = [
        {
            "id": "cfg1_baseline",
            "name": "Baseline (Standard Delaunay + Hungarian)",
            "min_edge": 0.0,
            "min_area": 0.0,
            "max_aspect": None,
            "cov_exp": 1.0,
            "min_matches": 0,
            "matcher": "hungarian"
        },
        {
            "id": "cfg2_simplex_stability",
            "name": "Enhanced Simplex Stability (Degeneracy Filtered)",
            "min_edge": 10.0,
            "min_area": 25.0,
            "max_aspect": 4.0,
            "cov_exp": 1.0,
            "min_matches": 0,
            "matcher": "hungarian"
        },
        {
            "id": "cfg3_coverage_normalized",
            "name": "Coverage-Aware Score Normalization (Exponent=1.2)",
            "min_edge": 0.0,
            "min_area": 0.0,
            "max_aspect": None,
            "cov_exp": 1.2,
            "min_matches": 3,
            "matcher": "hungarian"
        },
        {
            "id": "cfg4_greedy_1to1",
            "name": "Greedy 1-to-1 Assignment (Identical Cost Matrix)",
            "min_edge": 0.0,
            "min_area": 0.0,
            "max_aspect": None,
            "cov_exp": 1.0,
            "min_matches": 0,
            "matcher": "greedy_1to1"
        },
        {
            "id": "cfg5_stability_plus_greedy",
            "name": "Simplex Stability + Greedy 1-to-1",
            "min_edge": 10.0,
            "min_area": 25.0,
            "max_aspect": 4.0,
            "cov_exp": 1.0,
            "min_matches": 0,
            "matcher": "greedy_1to1"
        },
        {
            "id": "cfg6_fully_combined_greedy",
            "name": "Fully Combined (Stability + Coverage Norm + Greedy 1-to-1)",
            "min_edge": 10.0,
            "min_area": 25.0,
            "max_aspect": 4.0,
            "cov_exp": 1.2,
            "min_matches": 3,
            "matcher": "greedy_1to1"
        },
        {
            "id": "cfg7_fully_combined_hungarian",
            "name": "Fully Combined (Stability + Coverage Norm + Hungarian)",
            "min_edge": 10.0,
            "min_area": 25.0,
            "max_aspect": 4.0,
            "cov_exp": 1.2,
            "min_matches": 3,
            "matcher": "hungarian"
        }
    ]

    pipe = CancelableFingerprintPipeline(in_dim=9, dpi=ds["dpi"], use_gpu=True)

    # Extract minutiae for all images once
    print(f"[{subset}] Extracting base minutiae for all {len(ds['all_images'])} images...")
    t_feat_start = time.time()
    minutiae_cache = {}
    for item in ds["all_images"]:
        p = item["path"]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        feat = pipe.extract_features(img)
        minutiae_cache[p] = feat["minutiae"]
    print(f"[{subset}] Minutiae caching completed in {time.time() - t_feat_start:.2f}s.")

    W = derive_orthonormal_matrix(pipe.master_secret, salt=f"salt_{subset}", app_id=f"fvc2002_{subset}", in_dim=9, out_dim=32)

    config_results = {}

    for cfg in configurations:
        cfg_id = cfg["id"]
        cfg_name = cfg["name"]
        print(f"\n--- Evaluating Configuration: {cfg_name} ---")

        # Build templates with this config's Delaunay parameters
        templates = {}
        all_d = []
        triangle_counts = []

        for p, mins in minutiae_cache.items():
            tris = build_delaunay_triangles(
                mins,
                max_edge_length=120.0,
                min_edge_length=cfg["min_edge"],
                min_area=cfg["min_area"],
                max_aspect_ratio=cfg["max_aspect"]
            )
            d, w = extract_all_descriptors(tris, dpi=ds["dpi"], include_types=False)
            templates[p] = {"descs": d, "weights": w}
            if len(d) > 0:
                all_d.append(d)
                triangle_counts.append(len(d))

        stacked = np.vstack(all_d) if all_d else np.empty((0, 9))
        emp_m = np.mean(stacked, axis=0) if len(stacked) > 0 else np.zeros(9)
        emp_s = np.std(stacked, axis=0) if len(stacked) > 0 else np.ones(9)

        for p in templates:
            d = templates[p]["descs"]
            if len(d) > 0:
                proj = project_descriptors(d, W, mean=emp_m, std=emp_s)
                bits = quantize_descriptors(proj)
            else:
                bits = np.empty((0, 32), dtype=np.uint8)
            templates[p]["bits"] = bits

        # Matching loop
        t_match_start = time.time()
        g_scores = []
        i_scores = []

        # Genuine comparisons
        for p in gen_pairs:
            t_q = templates[p["probe_path"]]
            t_e = templates[p["gallery_path"]]
            if cfg["matcher"] == "greedy_1to1":
                s = match_protected_greedy(
                    t_q["bits"], t_e["bits"], t_q["weights"], t_e["weights"],
                    hamming_threshold=0.22,
                    mode="greedy_1to1",
                    coverage_penalty_exponent=cfg["cov_exp"],
                    min_matches=cfg["min_matches"]
                )
            else:
                s = match_protected_templates(
                    t_q["bits"], t_e["bits"], t_q["weights"], t_e["weights"],
                    hamming_threshold=0.22,
                    coverage_penalty_exponent=cfg["cov_exp"],
                    min_matches=cfg["min_matches"]
                )
            g_scores.append(s)

        # Impostor comparisons
        for p in imp_pairs:
            t_q = templates[p["probe_path"]]
            t_e = templates[p["gallery_path"]]
            if cfg["matcher"] == "greedy_1to1":
                s = match_protected_greedy(
                    t_q["bits"], t_e["bits"], t_q["weights"], t_e["weights"],
                    hamming_threshold=0.22,
                    mode="greedy_1to1",
                    coverage_penalty_exponent=cfg["cov_exp"],
                    min_matches=cfg["min_matches"]
                )
            else:
                s = match_protected_templates(
                    t_q["bits"], t_e["bits"], t_q["weights"], t_e["weights"],
                    hamming_threshold=0.22,
                    coverage_penalty_exponent=cfg["cov_exp"],
                    min_matches=cfg["min_matches"]
                )
            i_scores.append(s)

        t_match_end = time.time()
        match_time = max(t_match_end - t_match_start, 1e-6)
        throughput = float(len(all_pairs) / match_time)

        g_arr = np.array(g_scores, dtype=np.float32)
        i_arr = np.array(i_scores, dtype=np.float32)

        roc = compute_roc_metrics(g_arr, i_arr, warn_on_inversion=False)
        boot = bootstrap_confidence_intervals(g_arr, i_arr, num_bootstraps=500)

        # Clustered bootstrap over fingers
        gen_rec = [{'subject_a': p['probe_finger_id'], 'subject_b': p['gallery_finger_id'], 'score': float(s)} 
                   for p, s in zip(gen_pairs, g_arr)]
        imp_rec = [{'subject_a': p['probe_finger_id'], 'subject_b': p['gallery_finger_id'], 'score': float(s)} 
                   for p, s in zip(imp_pairs, i_arr)]
        clust = clustered_bootstrap_confidence_intervals(gen_rec, imp_rec, operating_threshold=float(roc["eer_threshold"]), num_bootstraps=500)

        # Evaluations at frozen Set B developmental thresholds
        frozen_ops = {}
        for t_name, t_val in FROZEN_SET_B_THRESHOLDS.items():
            frozen_ops[t_name] = evaluate_operational_point(g_arr, i_arr, t_val)

        # Evaluations at freshly calibrated DB1_B operating thresholds
        dev_calibrated_ops = {}
        for t_name, t_val in DEV_CALIBRATED_SET_B_THRESHOLDS.items():
            dev_calibrated_ops[t_name] = evaluate_operational_point(g_arr, i_arr, t_val)

        # Distribution stats
        g_dist = compute_distribution_stats(g_arr)
        i_dist = compute_distribution_stats(i_arr)

        cfg_summary = {
            "config_id": cfg_id,
            "config_name": cfg_name,
            "parameters": cfg,
            "avg_triangles": float(np.mean(triangle_counts)) if triangle_counts else 0.0,
            "matching_throughput_comp_per_sec": round(throughput, 1),
            "match_time_seconds": round(match_time, 3),
            "score_distributions": {
                "genuine": g_dist,
                "impostor": i_dist
            },
            "roc_metrics": {
                "eer_pct": round(float(roc["eer"] * 100.0), 2),
                "eer_threshold": round(float(roc["eer_threshold"]), 6),
                "auc": round(float(roc["auc"]), 4),
                "tar_at_1pct_far": round(float(roc["tar_at_1pct_far"] * 100.0), 2),
                "tar_at_01pct_far": round(float(roc["tar_at_01pct_far"] * 100.0), 2),
                "bootstrap_eer_95ci": [round(float(boot["eer_ci"][0] * 100.0), 2), round(float(boot["eer_ci"][1] * 100.0), 2)],
                "bootstrap_auc_95ci": [round(float(boot["auc_ci"][0]), 4), round(float(boot["auc_ci"][1]), 4)],
                "clustered_eer_95ci": [round(float(clust["eer_ci"][0] * 100.0), 2), round(float(clust["eer_ci"][1] * 100.0), 2)],
                "clustered_auc_95ci": [round(float(clust["auc_ci"][0]), 4), round(float(clust["auc_ci"][1]), 4)]
            },
            "frozen_set_b_evaluations": frozen_ops,
            "dev_calibrated_set_b_evaluations": dev_calibrated_ops
        }

        config_results[cfg_id] = cfg_summary

        print(f"  EER: {cfg_summary['roc_metrics']['eer_pct']}% (Clustered 95% CI: [{cfg_summary['roc_metrics']['clustered_eer_95ci'][0]}%, {cfg_summary['roc_metrics']['clustered_eer_95ci'][1]}%]), AUC: {cfg_summary['roc_metrics']['auc']}")
        print(f"  TAR@1%FAR: {cfg_summary['roc_metrics']['tar_at_1pct_far']}%, TAR@0.1%FAR: {cfg_summary['roc_metrics']['tar_at_01pct_far']}%")
        print(f"  At Frozen T_EER={FROZEN_SET_B_THRESHOLDS['T_EER']}: TAR={frozen_ops['T_EER']['tar_pct']}%, FAR={frozen_ops['T_EER']['far_pct']}%")
        print(f"  At Calibrated T_EER={DEV_CALIBRATED_SET_B_THRESHOLDS['T_EER']}: TAR={dev_calibrated_ops['T_EER']['tar_pct']}%, FAR={dev_calibrated_ops['T_EER']['far_pct']}%")
        print(f"  Throughput: {throughput:.1f} comp/s (Total time: {match_time:.2f}s)")

    report_payload = {
        "title": f"WP2: Recognition and Matching Improvement Evaluation ({subset})",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "subset": subset,
        "total_comparisons": len(all_pairs),
        "num_genuine": n_gen,
        "num_impostor": n_imp,
        "hardware_acceleration": gpu_info,
        "frozen_set_b_thresholds": FROZEN_SET_B_THRESHOLDS,
        "dev_calibrated_set_b_thresholds": DEV_CALIBRATED_SET_B_THRESHOLDS,
        "configurations": config_results
    }

    out_file = os.path.join(output_dir, "wp2_recognition_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\n[OK] WP2 Recognition report written to: {out_file}\n")
    return report_payload


if __name__ == "__main__":
    evaluate_wp2_subset(subset="DB1_A")
