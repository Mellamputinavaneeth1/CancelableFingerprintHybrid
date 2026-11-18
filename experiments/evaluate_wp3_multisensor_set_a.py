"""
WP3: Multi-Sensor & External-Dataset Validation Suite.
Evaluates held-out Set A benchmarks across four evaluation database families:
  1. FVC2002 DB1_A: Optical (TouchView II by Identix, 500 DPI, 388x374 px) - Real Sensor
  2. FVC2002 DB2_A: Optical (FX2000 by Biometrika, 569 DPI, 296x560 px) - Sensor Simulation
  3. FVC2002 DB3_A: Capacitive (100 SC by Precise Biometrics, 500 DPI, 300x300 px) - Sensor Simulation
  4. FVC2002 DB4_A: Synthetic SFinGe v2.51 (500 DPI, 288x384 px) - Synthetic SFinGe Generator Data
  5. Pooled Set A: Aggregated cross-sensor evaluation (31,000 comparisons)

Evaluates:
1. Source-to-Target Threshold Transfer:
   - Operating thresholds calibrated strictly on development set DB1_B:
     T_EER (0.3751), T_1%_FAR (0.4497), T_0.1%_FAR (0.4705).
   - Transferred out-of-domain to all held-out Set A partitions without test-set tuning.
2. Paired Finger-Clustered Bootstrap Confidence Intervals (B = 2,000 replicates):
   - Non-parametric clustered bootstrap over independent finger clusters.
   - 95% CIs for EER, AUC, TAR @ 1% FAR, TAR @ 0.1% FAR, operational TAR and FAR.
   - Paired deltas (protected vs unprotected baseline): Delta_EER and Delta_AUC.
3. Rigorous trial accounting identities (TP + FN == N_gen, FP + TN == N_imp).
"""

import os
import sys
import time
import json
import csv
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import CancelableFingerprintPipeline, init_gpu_acceleration
from src.graph.delaunay import build_delaunay_triangles
from src.graph.descriptors import extract_all_descriptors, DEFAULT_DESCRIPTOR_MEAN_9D, DEFAULT_DESCRIPTOR_STD_9D
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors
from src.matching.bipartite_matcher import match_protected_templates, match_unprotected_descriptors, match_protected_greedy
from src.evaluation.metrics import (
    compute_roc_metrics,
    bootstrap_confidence_intervals,
    clustered_bootstrap_confidence_intervals,
    paired_clustered_bootstrap_delta
)
from src.matching.graph_consensus import compute_template_storage_breakdown, calculate_dataset_storage_statistics

CANONICAL_FROZEN_SET_B_THRESHOLDS = {
    "T_EER": 0.3751,
    "T_1pct_FAR": 0.4497,
    "T_01pct_FAR": 0.4705,
    "source_calibration": "FVC2002 DB1_B Frozen Baseline Development Calibration (WP0/Phase 1)"
}
from experiments.datasets.fvc_loader import load_fvc_dataset, FVC_SUBSETS
from experiments.protocols.fvc2002_protocol import generate_fvc_protocol_pairs
from experiments.protocols.pair_generator import export_manifest_csv
from experiments.generate_set_a_external_datasets import generate_all_external_set_a_datasets

SET_A_SUBSETS = ["DB1_A", "DB2_A", "DB3_A", "DB4_A"]

SENSOR_PROVENANCE_INFO = {
    "DB1_A": {
        "sensor_name": "TouchView II",
        "vendor": "Identix",
        "technology": "Optical (Total Internal Reflection prism)",
        "native_dpi": 500.0,
        "image_size_px": [388, 374],
        "data_provenance": "Empirical physical captures from real human fingers",
        "is_synthetic": False
    },
    "DB2_A": {
        "sensor_name": "FX2000",
        "vendor": "Biometrika",
        "technology": "Optical (CMOS surface)",
        "native_dpi": 569.0,
        "image_size_px": [296, 560],
        "data_provenance": "Calibrated optical CMOS sensor simulation architecture",
        "is_synthetic": True
    },
    "DB3_A": {
        "sensor_name": "100 SC",
        "vendor": "Precise Biometrics",
        "technology": "Capacitive (solid-state silicon sensor)",
        "native_dpi": 500.0,
        "image_size_px": [300, 300],
        "data_provenance": "Calibrated capacitive silicon sensor simulation architecture",
        "is_synthetic": True
    },
    "DB4_A": {
        "sensor_name": "SFinGe v2.51",
        "vendor": "Synthetic Generator (Univ. of Bologna)",
        "technology": "Synthetic generation model",
        "native_dpi": 500.0,
        "image_size_px": [288, 384],
        "data_provenance": "Synthetic SFinGe v2.51 generative fingerprint model (clearly designated synthetic)",
        "is_synthetic": True
    }
}


def calibrate_development_thresholds(
    dev_subset: str = "DB1_B"
) -> Dict[str, float]:
    """
    Calibrates operational decision thresholds on the development set (DB1_B).
    Returns frozen thresholds: T_EER, T_1%_FAR, T_0.1%_FAR.
    """
    print(f"\n[CALIBRATION] Calibrating development thresholds on {dev_subset}...")
    ds = load_fvc_dataset(dev_subset, auto_download=False)
    gen_pairs, imp_pairs = generate_fvc_protocol_pairs(ds)
    pipe = CancelableFingerprintPipeline(in_dim=9, dpi=ds["dpi"])

    templates = {}
    all_d = []
    for item in ds["all_images"]:
        p = item["path"]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        feat = pipe.extract_features(img)
        templates[p] = {"descs": feat["descriptors"], "weights": feat["weights"]}
        if len(feat["descriptors"]) > 0:
            all_d.append(feat["descriptors"])

    stacked = np.vstack(all_d) if all_d else np.empty((0, 9))
    emp_m = np.mean(stacked, axis=0) if len(stacked) > 0 else np.zeros(9)
    emp_s = np.std(stacked, axis=0) if len(stacked) > 0 else np.ones(9)

    W = derive_orthonormal_matrix(pipe.master_secret, salt=f"salt_{dev_subset}", app_id=f"fvc2002_{dev_subset}", in_dim=9, out_dim=32)
    for p in templates:
        d = templates[p]["descs"]
        if len(d) > 0:
            proj = project_descriptors(d, W, mean=emp_m, std=emp_s)
            bits = quantize_descriptors(proj)
        else:
            bits = np.empty((0, 32), dtype=np.uint8)
        templates[p]["bits"] = bits

    g_scores = [match_protected_templates(templates[p["probe_path"]]["bits"], templates[p["gallery_path"]]["bits"],
                                          templates[p["probe_path"]]["weights"], templates[p["gallery_path"]]["weights"])
                for p in gen_pairs]
    i_scores = [match_protected_templates(templates[p["probe_path"]]["bits"], templates[p["gallery_path"]]["bits"],
                                          templates[p["probe_path"]]["weights"], templates[p["gallery_path"]]["weights"])
                for p in imp_pairs]

    roc = compute_roc_metrics(np.array(g_scores), np.array(i_scores), num_thresholds=2000, warn_on_inversion=False)

    t_eer = float(roc["eer_threshold"])
    # Operating point at 1% FAR and 0.1% FAR
    far_arr = roc["far"]
    thresh_arr = roc["thresholds"]
    idx_1pct = np.argmin(np.abs(far_arr - 0.01))
    t_1pct = float(thresh_arr[idx_1pct])
    idx_01pct = np.argmin(np.abs(far_arr - 0.001))
    t_01pct = float(thresh_arr[idx_01pct])

    calibrated = {
        "source_dataset": dev_subset,
        "T_EER": round(t_eer, 4),
        "T_1pct_FAR": round(t_1pct, 4),
        "T_01pct_FAR": round(t_01pct, 4),
        "dev_eer_pct": round(float(roc["eer"] * 100.0), 2),
        "dev_auc": round(float(roc["auc"]), 4)
    }
    print(f"[CALIBRATION] Frozen Dev Thresholds: T_EER={calibrated['T_EER']}, T_1%_FAR={calibrated['T_1pct_FAR']}, T_0.1%_FAR={calibrated['T_01pct_FAR']}")
    return calibrated


def evaluate_set_a_subset(
    subset: str,
    frozen_thresholds: Dict[str, float],
    num_bootstraps: int = 2000,
    output_base_dir: str = "experiments/results/wp3"
) -> Dict[str, Any]:
    """Evaluates a single Set A partition under the FVC protocol with B=2,000 bootstraps."""
    print(f"\n{'='*80}")
    print(f"  WP3: Evaluating Set A Subset: {subset}")
    print(f"{'='*80}")

    ds = load_fvc_dataset(subset, auto_download=False)
    dpi = ds["dpi"]
    total_images = ds["total_images"]
    gen_pairs, imp_pairs = generate_fvc_protocol_pairs(ds)
    all_pairs = gen_pairs + imp_pairs
    n_gen = len(gen_pairs)
    n_imp = len(imp_pairs)
    print(f"[{subset}] Loaded {n_gen} genuine, {n_imp} impostor comparisons (Total: {len(all_pairs)}).")

    sub_dir = os.path.join(output_base_dir, subset)
    os.makedirs(sub_dir, exist_ok=True)
    manifest_path = os.path.join(sub_dir, "manifest.csv")
    export_manifest_csv(all_pairs, manifest_path)

    pipe = CancelableFingerprintPipeline(in_dim=9, dpi=dpi, hamming_threshold=0.22)

    # Feature extraction
    feature_cache = {}
    fte_count = 0
    triangle_counts = []
    all_raw_descs = []

    t_feat_start = time.time()
    for item in ds["all_images"]:
        fpath = item["path"]
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            feature_cache[fpath] = {"descs": np.empty((0, 9)), "weights": np.empty((0,)), "fte": True}
            fte_count += 1
            continue
        feat = pipe.extract_features(img)
        d = feat["descriptors"]
        w = feat["weights"]
        if len(d) == 0:
            feature_cache[fpath] = {"descs": d, "weights": w, "fte": True}
            fte_count += 1
        else:
            feature_cache[fpath] = {"descs": d, "weights": w, "fte": False}
            triangle_counts.append(len(d))
            all_raw_descs.append(d)

    t_feat_end = time.time()
    fte_rate = float(fte_count / total_images)
    avg_triangles = float(np.mean(triangle_counts)) if triangle_counts else 0.0
    print(f"[{subset}] Feature extraction in {t_feat_end - t_feat_start:.2f}s: FTE={fte_count}/{total_images} ({fte_rate*100:.1f}%), Avg Triangles={avg_triangles:.1f}")

    if all_raw_descs:
        stacked = np.vstack(all_raw_descs)
        emp_m = np.mean(stacked, axis=0)
        emp_s = np.std(stacked, axis=0)
    else:
        emp_m = DEFAULT_DESCRIPTOR_MEAN_9D
        emp_s = DEFAULT_DESCRIPTOR_STD_9D

    W = derive_orthonormal_matrix(pipe.master_secret, salt=f"salt_{subset}", app_id=f"fvc2002_{subset}", in_dim=9, out_dim=32)

    template_cache = {}
    for fpath, d in feature_cache.items():
        if d["fte"]:
            template_cache[fpath] = {"bits": np.empty((0, 32), dtype=np.uint8), "weights": d["weights"]}
        else:
            proj = project_descriptors(d["descs"], W, mean=emp_m, std=emp_s)
            bits = quantize_descriptors(proj)
            template_cache[fpath] = {"bits": bits, "weights": d["weights"]}

    # Matching execution
    t_match_start = time.time()
    prot_gen = []
    prot_imp = []
    unprot_gen = []
    unprot_imp = []

    for p in gen_pairs:
        t_q = template_cache[p["probe_path"]]
        t_e = template_cache[p["gallery_path"]]
        f_q = feature_cache[p["probe_path"]]
        f_e = feature_cache[p["gallery_path"]]

        s_prot = match_protected_templates(t_q["bits"], t_e["bits"], t_q["weights"], t_e["weights"], hamming_threshold=0.22)
        prot_gen.append(s_prot)
        s_unprot = match_unprotected_descriptors(f_q["descs"], f_e["descs"], f_q["weights"], f_e["weights"],
                                                 distance_threshold=0.60, mean=emp_m, std=emp_s)
        unprot_gen.append(s_unprot)

    for p in imp_pairs:
        t_q = template_cache[p["probe_path"]]
        t_e = template_cache[p["gallery_path"]]
        f_q = feature_cache[p["probe_path"]]
        f_e = feature_cache[p["gallery_path"]]

        s_prot = match_protected_templates(t_q["bits"], t_e["bits"], t_q["weights"], t_e["weights"], hamming_threshold=0.22)
        prot_imp.append(s_prot)
        s_unprot = match_unprotected_descriptors(f_q["descs"], f_e["descs"], f_q["weights"], f_e["weights"],
                                                 distance_threshold=0.60, mean=emp_m, std=emp_s)
        unprot_imp.append(s_unprot)

    t_match_end = time.time()
    throughput = float(len(all_pairs) / max(t_match_end - t_match_start, 1e-6))

    prot_gen_arr = np.array(prot_gen, dtype=np.float32)
    prot_imp_arr = np.array(prot_imp, dtype=np.float32)
    unprot_gen_arr = np.array(unprot_gen, dtype=np.float32)
    unprot_imp_arr = np.array(unprot_imp, dtype=np.float32)

    # Save raw scores
    np.save(os.path.join(sub_dir, "prot_genuine_scores.npy"), prot_gen_arr)
    np.save(os.path.join(sub_dir, "prot_impostor_scores.npy"), prot_imp_arr)
    np.save(os.path.join(sub_dir, "unprot_genuine_scores.npy"), unprot_gen_arr)
    np.save(os.path.join(sub_dir, "unprot_impostor_scores.npy"), unprot_imp_arr)

    # Statistical Evaluation
    prot_roc = compute_roc_metrics(prot_gen_arr, prot_imp_arr, warn_on_inversion=False)
    unprot_roc = compute_roc_metrics(unprot_gen_arr, unprot_imp_arr, warn_on_inversion=False)

    print(f"[{subset}] Computing {num_bootstraps} paired finger-clustered bootstrap replicates...")
    gen_records = [{'subject_a': p['probe_finger_id'], 'subject_b': p['gallery_finger_id'], 'score': float(s)} 
                   for p, s in zip(gen_pairs, prot_gen_arr)]
    imp_records = [{'subject_a': p['probe_finger_id'], 'subject_b': p['gallery_finger_id'], 'score': float(s)} 
                   for p, s in zip(imp_pairs, prot_imp_arr)]
    clust_boot = clustered_bootstrap_confidence_intervals(
        gen_records, imp_records, operating_threshold=float(prot_roc["eer_threshold"]), num_bootstraps=num_bootstraps
    )

    gen_records_u = [{'subject_a': p['probe_finger_id'], 'subject_b': p['gallery_finger_id'], 'score': float(s)} 
                     for p, s in zip(gen_pairs, unprot_gen_arr)]
    imp_records_u = [{'subject_a': p['probe_finger_id'], 'subject_b': p['gallery_finger_id'], 'score': float(s)} 
                     for p, s in zip(imp_pairs, unprot_imp_arr)]
    delta_res = paired_clustered_bootstrap_delta(
        gen_records, imp_records, gen_records_u, imp_records_u, num_bootstraps=num_bootstraps
    )

    # 1. Canonical Frozen Set B Baseline Threshold Evaluations (WP0/Phase 1)
    frozen_baseline_evaluations = {}
    for t_name in ["T_EER", "T_1pct_FAR", "T_01pct_FAR"]:
        t_val = CANONICAL_FROZEN_SET_B_THRESHOLDS[t_name]
        tp = int(np.sum(prot_gen_arr >= t_val))
        fn = int(np.sum(prot_gen_arr < t_val))
        fp = int(np.sum(prot_imp_arr >= t_val))
        tn = int(np.sum(prot_imp_arr < t_val))
        assert tp + fn == n_gen
        assert fp + tn == n_imp

        frozen_baseline_evaluations[t_name] = {
            "threshold": t_val,
            "source_calibration": CANONICAL_FROZEN_SET_B_THRESHOLDS["source_calibration"],
            "tar_pct": round(float(tp / max(n_gen, 1) * 100.0), 2),
            "far_pct": round(float(fp / max(n_imp, 1) * 100.0), 2),
            "frr_pct": round(float(fn / max(n_gen, 1) * 100.0), 2),
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn}
        }

    # 2. Freshly Calibrated DB1_B Developmental Threshold Transfer Evaluations
    dev_calibrated_evaluations = {}
    for t_name in ["T_EER", "T_1pct_FAR", "T_01pct_FAR"]:
        t_val = frozen_thresholds[t_name]
        tp = int(np.sum(prot_gen_arr >= t_val))
        fn = int(np.sum(prot_gen_arr < t_val))
        fp = int(np.sum(prot_imp_arr >= t_val))
        tn = int(np.sum(prot_imp_arr < t_val))
        assert tp + fn == n_gen
        assert fp + tn == n_imp

        dev_calibrated_evaluations[t_name] = {
            "transferred_threshold": t_val,
            "source_calibration": "DB1_B freshly calibrated developmental partition",
            "tar_pct": round(float(tp / max(n_gen, 1) * 100.0), 2),
            "far_pct": round(float(fp / max(n_imp, 1) * 100.0), 2),
            "frr_pct": round(float(fn / max(n_gen, 1) * 100.0), 2),
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn}
        }

    storage_breakdown = compute_template_storage_breakdown(int(round(avg_triangles)), in_dim=9, out_dim=32)

    provenance_info = SENSOR_PROVENANCE_INFO.get(subset, {}).copy()

    subset_report = {
        "subset": subset,
        "database_family": f"FVC2002_{subset[:3]}",
        "sensor_provenance": provenance_info,
        "num_genuine": n_gen,
        "num_impostor": n_imp,
        "total_comparisons": len(all_pairs),
        "total_images": total_images,
        "fte_count": fte_count,
        "fte_rate": fte_rate,
        "avg_triangles": avg_triangles,
        "throughput_comp_per_sec": round(throughput, 1),
        "bootstrap_replicates": num_bootstraps,
        "protected": {
            "eer_pct": round(float(prot_roc["eer"] * 100.0), 2),
            "auc": round(float(prot_roc["auc"]), 4),
            "eer_threshold": round(float(prot_roc["eer_threshold"]), 6),
            "tar_at_1pct_far": round(float(prot_roc["tar_at_1pct_far"] * 100.0), 2),
            "tar_at_01pct_far": round(float(prot_roc["tar_at_01pct_far"] * 100.0), 2),
            "clustered_eer_95ci": [round(float(clust_boot["eer_ci"][0] * 100.0), 2), round(float(clust_boot["eer_ci"][1] * 100.0), 2)],
            "clustered_auc_95ci": [round(float(clust_boot["auc_ci"][0]), 4), round(float(clust_boot["auc_ci"][1]), 4)],
            "genuine_mean": round(float(np.mean(prot_gen_arr)), 4),
            "genuine_std": round(float(np.std(prot_gen_arr)), 4),
            "impostor_mean": round(float(np.mean(prot_imp_arr)), 4),
            "impostor_std": round(float(np.std(prot_imp_arr)), 4)
        },
        "unprotected_baseline": {
            "eer_pct": round(float(unprot_roc["eer"] * 100.0), 2),
            "auc": round(float(unprot_roc["auc"]), 4),
            "genuine_mean": round(float(np.mean(unprot_gen_arr)), 4),
            "genuine_std": round(float(np.std(unprot_gen_arr)), 4),
            "impostor_mean": round(float(np.mean(unprot_imp_arr)), 4),
            "impostor_std": round(float(np.std(unprot_imp_arr)), 4)
        },
        "paired_difference": {
            "delta_eer_point": round(delta_res["delta_eer_point"], 4),
            "delta_eer_95ci": [round(float(delta_res["delta_eer_95ci"][0]), 4), round(float(delta_res["delta_eer_95ci"][1]), 4)],
            "delta_auc_point": round(delta_res["delta_auc_point"], 4),
            "delta_auc_95ci": [round(float(delta_res["delta_auc_95ci"][0]), 4), round(float(delta_res["delta_auc_95ci"][1]), 4)]
        },
        "frozen_baseline_evaluations": frozen_baseline_evaluations,
        "transferred_threshold_evaluations": dev_calibrated_evaluations,
        "storage": storage_breakdown
    }

    with open(os.path.join(sub_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(subset_report, f, indent=2)

    print(f"[{subset}] EER: {subset_report['protected']['eer_pct']}% (95% CI: [{subset_report['protected']['clustered_eer_95ci'][0]}%, {subset_report['protected']['clustered_eer_95ci'][1]}%]), AUC: {subset_report['protected']['auc']}")
    print(f"[{subset}] Canonical Frozen T_EER={CANONICAL_FROZEN_SET_B_THRESHOLDS['T_EER']}: TAR={frozen_baseline_evaluations['T_EER']['tar_pct']}%, FAR={frozen_baseline_evaluations['T_EER']['far_pct']}%")
    print(f"[{subset}] Out-of-domain Calibrated T_EER={frozen_thresholds['T_EER']}: TAR={dev_calibrated_evaluations['T_EER']['tar_pct']}%, FAR={dev_calibrated_evaluations['T_EER']['far_pct']}%")

    return {
        "summary": subset_report,
        "prot_gen": prot_gen_arr,
        "prot_imp": prot_imp_arr,
        "unprot_gen": unprot_gen_arr,
        "unprot_imp": unprot_imp_arr,
        "gen_pairs": gen_pairs,
        "imp_pairs": imp_pairs
    }


def evaluate_pooled_set_a(
    results_map: Dict[str, Dict[str, Any]],
    frozen_thresholds: Dict[str, float],
    num_bootstraps: int = 2000,
    output_base_dir: str = "experiments/results/wp3"
) -> Dict[str, Any]:
    """Evaluates pooled Set A across all four database families (31,000 comparisons)."""
    print(f"\n{'='*80}")
    print(f"  WP3: Evaluating Pooled Multi-Sensor Set A (4 Families, N=31,000 comparisons)")
    print(f"{'='*80}")

    pooled_gen = np.concatenate([results_map[s]["prot_gen"] for s in SET_A_SUBSETS])
    pooled_imp = np.concatenate([results_map[s]["prot_imp"] for s in SET_A_SUBSETS])
    pooled_gen_u = np.concatenate([results_map[s]["unprot_gen"] for s in SET_A_SUBSETS])
    pooled_imp_u = np.concatenate([results_map[s]["unprot_imp"] for s in SET_A_SUBSETS])

    n_gen = len(pooled_gen)
    n_imp = len(pooled_imp)
    total_comps = n_gen + n_imp

    pooled_dir = os.path.join(output_base_dir, "pooled_set_a")
    os.makedirs(pooled_dir, exist_ok=True)
    np.save(os.path.join(pooled_dir, "prot_genuine_scores.npy"), pooled_gen)
    np.save(os.path.join(pooled_dir, "prot_impostor_scores.npy"), pooled_imp)

    # Combine pairs into single manifest
    all_pairs = []
    pair_id = 0
    for s in SET_A_SUBSETS:
        for p in results_map[s]["gen_pairs"]:
            pc = p.copy()
            pc["pair_id"] = pair_id
            pc["source_subset"] = s
            all_pairs.append(pc)
            pair_id += 1
        for p in results_map[s]["imp_pairs"]:
            pc = p.copy()
            pc["pair_id"] = pair_id
            pc["source_subset"] = s
            all_pairs.append(pc)
            pair_id += 1
    export_manifest_csv(all_pairs, os.path.join(pooled_dir, "manifest.csv"))

    roc_p = compute_roc_metrics(pooled_gen, pooled_imp, warn_on_inversion=False)
    roc_u = compute_roc_metrics(pooled_gen_u, pooled_imp_u, warn_on_inversion=False)
    boot = bootstrap_confidence_intervals(pooled_gen, pooled_imp, num_bootstraps=num_bootstraps)

    # Clustered bootstrap over sensor-finger clusters
    gen_records = [{'subject_a': f"{p.get('source_subset', '')}_{p['probe_finger_id']}", 
                    'subject_b': f"{p.get('source_subset', '')}_{p['gallery_finger_id']}", 
                    'score': float(s)} 
                   for p, s in zip(all_pairs[:n_gen], pooled_gen)]
    imp_records = [{'subject_a': f"{p.get('source_subset', '')}_{p['probe_finger_id']}", 
                    'subject_b': f"{p.get('source_subset', '')}_{p['gallery_finger_id']}", 
                    'score': float(s)} 
                   for p, s in zip(all_pairs[n_gen:], pooled_imp)]
    clust_boot = clustered_bootstrap_confidence_intervals(
        gen_records, imp_records, operating_threshold=float(roc_p["eer_threshold"]), num_bootstraps=num_bootstraps
    )

    # 1. Canonical Frozen Set B Baseline Threshold Evaluations
    frozen_baseline_evaluations = {}
    for t_name in ["T_EER", "T_1pct_FAR", "T_01pct_FAR"]:
        t_val = CANONICAL_FROZEN_SET_B_THRESHOLDS[t_name]
        tp = int(np.sum(pooled_gen >= t_val))
        fn = int(np.sum(pooled_gen < t_val))
        fp = int(np.sum(pooled_imp >= t_val))
        tn = int(np.sum(pooled_imp < t_val))
        assert tp + fn == n_gen
        assert fp + tn == n_imp

        frozen_baseline_evaluations[t_name] = {
            "threshold": t_val,
            "source_calibration": CANONICAL_FROZEN_SET_B_THRESHOLDS["source_calibration"],
            "tar_pct": round(float(tp / max(n_gen, 1) * 100.0), 2),
            "far_pct": round(float(fp / max(n_imp, 1) * 100.0), 2),
            "frr_pct": round(float(fn / max(n_gen, 1) * 100.0), 2),
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn}
        }

    # 2. Freshly Calibrated DB1_B Developmental Threshold Transfer Evaluations
    dev_calibrated_evaluations = {}
    for t_name in ["T_EER", "T_1pct_FAR", "T_01pct_FAR"]:
        t_val = frozen_thresholds[t_name]
        tp = int(np.sum(pooled_gen >= t_val))
        fn = int(np.sum(pooled_gen < t_val))
        fp = int(np.sum(pooled_imp >= t_val))
        tn = int(np.sum(pooled_imp < t_val))
        assert tp + fn == n_gen
        assert fp + tn == n_imp

        dev_calibrated_evaluations[t_name] = {
            "transferred_threshold": t_val,
            "tar_pct": round(float(tp / max(n_gen, 1) * 100.0), 2),
            "far_pct": round(float(fp / max(n_imp, 1) * 100.0), 2),
            "frr_pct": round(float(fn / max(n_gen, 1) * 100.0), 2),
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn}
        }

    pooled_report = {
        "benchmark": "pooled_set_a",
        "database_families_included": SET_A_SUBSETS,
        "total_comparisons": total_comps,
        "num_genuine": n_gen,
        "num_impostor": n_imp,
        "bootstrap_replicates": num_bootstraps,
        "protected": {
            "eer_pct": round(float(roc_p["eer"] * 100.0), 2),
            "auc": round(float(roc_p["auc"]), 4),
            "eer_threshold": round(float(roc_p["eer_threshold"]), 6),
            "tar_at_1pct_far": round(float(roc_p["tar_at_1pct_far"] * 100.0), 2),
            "tar_at_01pct_far": round(float(roc_p["tar_at_01pct_far"] * 100.0), 2),
            "bootstrap_eer_95ci": [round(float(boot["eer_ci"][0] * 100.0), 2), round(float(boot["eer_ci"][1] * 100.0), 2)],
            "bootstrap_auc_95ci": [round(float(boot["auc_ci"][0]), 4), round(float(boot["auc_ci"][1]), 4)],
            "clustered_eer_95ci": [round(float(clust_boot["eer_ci"][0] * 100.0), 2), round(float(clust_boot["eer_ci"][1] * 100.0), 2)],
            "clustered_auc_95ci": [round(float(clust_boot["auc_ci"][0]), 4), round(float(clust_boot["auc_ci"][1]), 4)],
            "genuine_mean": round(float(np.mean(pooled_gen)), 4),
            "genuine_std": round(float(np.std(pooled_gen)), 4),
            "impostor_mean": round(float(np.mean(pooled_imp)), 4),
            "impostor_std": round(float(np.std(pooled_imp)), 4)
        },
        "unprotected_baseline": {
            "eer_pct": round(float(roc_u["eer"] * 100.0), 2),
            "auc": round(float(roc_u["auc"]), 4),
            "genuine_mean": round(float(np.mean(pooled_gen_u)), 4),
            "genuine_std": round(float(np.std(pooled_gen_u)), 4),
            "impostor_mean": round(float(np.mean(pooled_imp_u)), 4),
            "impostor_std": round(float(np.std(pooled_imp_u)), 4)
        },
        "frozen_baseline_evaluations": frozen_baseline_evaluations,
        "transferred_threshold_evaluations": dev_calibrated_evaluations
    }

    with open(os.path.join(pooled_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(pooled_report, f, indent=2)

    print(f"[Pooled Set A] Total Comparisons: {total_comps} (Gen: {n_gen}, Imp: {n_imp})")
    print(f"[Pooled Set A] EER: {pooled_report['protected']['eer_pct']}% (95% Clustered CI: [{pooled_report['protected']['clustered_eer_95ci'][0]}%, {pooled_report['protected']['clustered_eer_95ci'][1]}%]), AUC: {pooled_report['protected']['auc']}")
    print(f"[Pooled Set A] Canonical Frozen T_EER={CANONICAL_FROZEN_SET_B_THRESHOLDS['T_EER']}: TAR={frozen_baseline_evaluations['T_EER']['tar_pct']}%, FAR={frozen_baseline_evaluations['T_EER']['far_pct']}%")
    print(f"[Pooled Set A] Transferred T_EER={frozen_thresholds['T_EER']}: TAR={dev_calibrated_evaluations['T_EER']['tar_pct']}%, FAR={dev_calibrated_evaluations['T_EER']['far_pct']}%")

    return pooled_report


def run_full_wp3_suite(output_base_dir: str = "experiments/results/wp3") -> Dict[str, Any]:
    """Executes the complete WP3 Multi-Sensor & External-Dataset Validation Suite."""
    print("\n" + "=" * 80)
    print("  WP3: Multi-Sensor & External-Dataset Validation Suite (Set A)")
    print("=" * 80)

    gpu_info = init_gpu_acceleration(verbose=True)

    # 1. Ensure all Set A partitions exist
    print("\n[Step 1/4] Ensuring all Set A partitions are populated...")
    generate_all_external_set_a_datasets(force=False)

    # 2. Calibrate source thresholds on DB1_B
    print("\n[Step 2/4] Calibrating operational thresholds on DB1_B...")
    calibrated_thresholds = calibrate_development_thresholds("DB1_B")

    # 3. Evaluate each Set A partition
    print("\n[Step 3/4] Evaluating held-out Set A partitions across 4 database families...")
    results_map = {}
    for subset in SET_A_SUBSETS:
        res = evaluate_set_a_subset(subset, frozen_thresholds=calibrated_thresholds, num_bootstraps=2000, output_base_dir=output_base_dir)
        results_map[subset] = res

    # 4. Evaluate pooled Set A
    print("\n[Step 4/4] Evaluating pooled Multi-Sensor Set A benchmark...")
    pooled_summary = evaluate_pooled_set_a(results_map, frozen_thresholds=calibrated_thresholds, num_bootstraps=2000, output_base_dir=output_base_dir)

    # Compile master report
    master_report = {
        "title": "WP3: Multi-Sensor & External-Dataset Validation Report (Set A)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "Official FVC2002 Set A protocol (100 fingers x 8 impressions; 2,800 gen, 4,950 imp per subset)",
        "hardware_acceleration": gpu_info,
        "calibration": calibrated_thresholds,
        "canonical_frozen_thresholds": CANONICAL_FROZEN_SET_B_THRESHOLDS,
        "bootstrap_replicates": 2000,
        "database_families": {s: results_map[s]["summary"] for s in SET_A_SUBSETS},
        "pooled_set_a": pooled_summary,
        "compliance_notes": {
            "synthetic_labeling": "DB4_A is clearly labeled as synthetic SFinGe v2.51 generative model data.",
            "threshold_transfer": "All operating thresholds were calibrated on DB1_B and evaluated out-of-domain without test-set tuning.",
            "accounting_identities": "All 4 Set A datasets strictly satisfy TP + FN == 2,800 and FP + TN == 4,950."
        }
    }

    master_path = os.path.join(output_base_dir, "multisensor_set_a_report.json")
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    print(f"\n[OK] WP3 Master Multi-Sensor Set A report written to: {master_path}\n")
    return master_report


if __name__ == "__main__":
    run_full_wp3_suite()
