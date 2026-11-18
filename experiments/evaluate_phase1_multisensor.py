"""
Phase 1: Multi-Sensor Recognition Performance Benchmark Runner.
Evaluates the 9-D Pure Geometric Cancelable Fingerprint System across:
  - FVC2002 DB1_A (Identix optical TIR prism, N=7,750 comparisons)
  - FVC2002 DB1_B (Identix optical TIR prism, N=325 comparisons)
  - FVC2002 DB2_B (Biometrika optical FX2000, N=325 comparisons)
  - FVC2002 DB3_B (Precise Biometrics capacitive 100 SC, N=325 comparisons)
  - FVC2002 DB4_B (SFinGe synthetic generator v2.51, N=325 comparisons)
  - Pooled Set B (4 sensors, N=1,300 comparisons)
  - Pooled All-Available (5 benchmarks, N=9,050 comparisons)

Reports: EER, AUC, TAR @ 1% FAR, FAR @ frozen dev threshold (T=0.3751),
score means, and 95% bootstrap CIs for each sensor separately and pooled.
"""

import os
import sys
import time
import json
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import CancelableFingerprintPipeline
from src.graph.descriptors import extract_all_descriptors, DEFAULT_DESCRIPTOR_MEAN_9D, DEFAULT_DESCRIPTOR_STD_9D
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors
from src.matching.bipartite_matcher import match_protected_templates, match_unprotected_descriptors
from src.evaluation.metrics import (
    compute_roc_metrics, 
    clopper_pearson_ci, 
    bootstrap_confidence_intervals,
    clustered_bootstrap_confidence_intervals,
    paired_clustered_bootstrap_delta
)
from src.matching.graph_consensus import compute_template_storage_breakdown, calculate_dataset_storage_statistics
from experiments.datasets.fvc_loader import load_fvc_dataset, FVC_SUBSETS
from experiments.protocols.fvc2002_protocol import generate_fvc_protocol_pairs
from experiments.protocols.pair_generator import export_manifest_csv

SENSOR_PROVENANCE = {
    "DB1": {
        "sensor_name": "TouchView II",
        "vendor": "Identix",
        "technology": "Optical (Total Internal Reflection prism)",
        "native_dpi": 500.0,
        "image_size_px": [388, 374]
    },
    "DB2": {
        "sensor_name": "FX2000",
        "vendor": "Biometrika",
        "technology": "Optical (CMOS surface)",
        "native_dpi": 569.0,
        "image_size_px": [296, 560]
    },
    "DB3": {
        "sensor_name": "100 SC",
        "vendor": "Precise Biometrics",
        "technology": "Capacitive (solid-state silicon sensor)",
        "native_dpi": 500.0,
        "image_size_px": [300, 300]
    },
    "DB4": {
        "sensor_name": "SFinGe v2.51",
        "vendor": "Synthetic Generator (Univ. of Bologna)",
        "technology": "Synthetic generation model",
        "native_dpi": 500.0,
        "image_size_px": [288, 384]
    }
}

FROZEN_DEV_THRESHOLD = 0.3751


def evaluate_sensor_subset(
    subset: str,
    output_base_dir: str = "experiments/results/phase1/multi_sensor",
    save_scores: bool = True
) -> Dict[str, Any]:
    """
    Evaluates one FVC2002 dataset subset under the standard FVC protocol.
    """
    print(f"\n{'='*75}")
    print(f"  Evaluating Multi-Sensor Subset: {subset}")
    print(f"{'='*75}")
    
    t0 = time.time()
    dataset = load_fvc_dataset(subset, auto_download=False)
    dpi = dataset["dpi"]
    total_images = dataset["total_images"]
    
    gen_pairs, imp_pairs = generate_fvc_protocol_pairs(dataset)
    num_gen = len(gen_pairs)
    num_imp = len(imp_pairs)
    all_pairs = gen_pairs + imp_pairs
    print(f"[{subset}] Loaded {num_gen} genuine, {num_imp} impostor comparisons (Total: {len(all_pairs)}).")
    
    sub_dir = os.path.join(output_base_dir, subset)
    os.makedirs(sub_dir, exist_ok=True)
    manifest_path = os.path.join(sub_dir, "manifest.csv")
    export_manifest_csv(all_pairs, manifest_path)
    
    pipe = CancelableFingerprintPipeline(in_dim=9, dpi=dpi, hamming_threshold=0.22)
    
    # Feature extraction & caching
    feature_cache: Dict[str, Dict[str, Any]] = {}
    fte_count = 0
    triangle_counts = []
    all_raw_descs = []
    
    t_feat_start = time.time()
    for item in dataset["all_images"]:
        fpath = item["path"]
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            feature_cache[fpath] = {
                "descriptors": np.empty((0, 9), dtype=np.float32),
                "weights": np.empty((0,), dtype=np.float32),
                "fte": True
            }
            fte_count += 1
            continue
            
        feat = pipe.extract_features(img)
        descs = feat["descriptors"]
        weights = feat["weights"]
        
        if len(descs) == 0:
            fte_count += 1
            feature_cache[fpath] = {"descriptors": descs, "weights": weights, "fte": True}
        else:
            feature_cache[fpath] = {"descriptors": descs, "weights": weights, "fte": False}
            triangle_counts.append(len(descs))
            all_raw_descs.append(descs)
            
    t_feat_end = time.time()
    fte_rate = float(fte_count / total_images)
    avg_triangles = float(np.mean(triangle_counts)) if triangle_counts else 0.0
    print(f"[{subset}] Extraction done in {t_feat_end - t_feat_start:.2f}s: FTE={fte_count}/{total_images} ({fte_rate*100:.1f}%), Avg Triangles={avg_triangles:.1f}")
    
    # Normalization parameters
    if all_raw_descs:
        stacked = np.vstack(all_raw_descs)
        emp_mean = np.mean(stacked, axis=0)
        emp_std = np.std(stacked, axis=0)
    else:
        emp_mean = DEFAULT_DESCRIPTOR_MEAN_9D
        emp_std = DEFAULT_DESCRIPTOR_STD_9D
        
    # Protected projection (M=32 bits)
    W = derive_orthonormal_matrix(pipe.master_secret, salt=f"salt_{subset}", app_id=f"fvc2002_{subset}", in_dim=9, out_dim=32)
    template_cache: Dict[str, Dict[str, Any]] = {}
    for fpath, data in feature_cache.items():
        if data["fte"]:
            template_cache[fpath] = {"bitstrings": np.empty((0, 32), dtype=np.uint8), "weights": data["weights"]}
        else:
            proj = project_descriptors(data["descriptors"], W, mean=emp_mean, std=emp_std)
            bits = quantize_descriptors(proj)
            template_cache[fpath] = {"bitstrings": bits, "weights": data["weights"]}
            
    # Matching
    t_match_start = time.time()
    prot_genuine_scores = []
    prot_impostor_scores = []
    unprot_genuine_scores = []
    unprot_impostor_scores = []
    
    for p in gen_pairs:
        t_q = template_cache[p["probe_path"]]
        t_e = template_cache[p["gallery_path"]]
        f_q = feature_cache[p["probe_path"]]
        f_e = feature_cache[p["gallery_path"]]
        
        s_prot = match_protected_templates(t_q["bitstrings"], t_e["bitstrings"], t_q["weights"], t_e["weights"], hamming_threshold=0.22)
        prot_genuine_scores.append(s_prot)
        
        s_unprot = match_unprotected_descriptors(f_q["descriptors"], f_e["descriptors"], f_q["weights"], f_e["weights"],
                                                distance_threshold=0.60, mean=emp_mean, std=emp_std)
        unprot_genuine_scores.append(s_unprot)
        
    for p in imp_pairs:
        t_q = template_cache[p["probe_path"]]
        t_e = template_cache[p["gallery_path"]]
        f_q = feature_cache[p["probe_path"]]
        f_e = feature_cache[p["gallery_path"]]
        
        s_prot = match_protected_templates(t_q["bitstrings"], t_e["bitstrings"], t_q["weights"], t_e["weights"], hamming_threshold=0.22)
        prot_impostor_scores.append(s_prot)
        
        s_unprot = match_unprotected_descriptors(f_q["descriptors"], f_e["descriptors"], f_q["weights"], f_e["weights"],
                                                distance_threshold=0.60, mean=emp_mean, std=emp_std)
        unprot_impostor_scores.append(s_unprot)
        
    t_match_end = time.time()
    total_comps = len(all_pairs)
    throughput = float(total_comps / max(t_match_end - t_match_start, 1e-6))
    
    prot_genuine_scores = np.array(prot_genuine_scores, dtype=np.float32)
    prot_impostor_scores = np.array(prot_impostor_scores, dtype=np.float32)
    unprot_genuine_scores = np.array(unprot_genuine_scores, dtype=np.float32)
    unprot_impostor_scores = np.array(unprot_impostor_scores, dtype=np.float32)
    
    # Save raw score arrays
    if save_scores:
        np.save(os.path.join(sub_dir, "prot_genuine_scores.npy"), prot_genuine_scores)
        np.save(os.path.join(sub_dir, "prot_impostor_scores.npy"), prot_impostor_scores)
        np.save(os.path.join(sub_dir, "unprot_genuine_scores.npy"), unprot_genuine_scores)
        np.save(os.path.join(sub_dir, "unprot_impostor_scores.npy"), unprot_impostor_scores)
        
    # Statistical analysis
    prot_metrics = compute_roc_metrics(prot_genuine_scores, prot_impostor_scores)
    unprot_metrics = compute_roc_metrics(unprot_genuine_scores, unprot_impostor_scores)
    
    boot_res = bootstrap_confidence_intervals(prot_genuine_scores, prot_impostor_scores, num_bootstraps=500)
    
    # Clustered bootstrap over fingers
    gen_records = [{'subject_a': p['probe_finger_id'], 'subject_b': p['gallery_finger_id'], 'score': float(s)} 
                   for p, s in zip(gen_pairs, prot_genuine_scores)]
    imp_records = [{'subject_a': p['probe_finger_id'], 'subject_b': p['gallery_finger_id'], 'score': float(s)} 
                   for p, s in zip(imp_pairs, prot_impostor_scores)]
    clust_res = clustered_bootstrap_confidence_intervals(
        gen_records, imp_records, operating_threshold=float(prot_metrics['eer_threshold']), num_bootstraps=500
    )
    
    # Paired delta
    gen_records_unprot = [{'subject_a': p['probe_finger_id'], 'subject_b': p['gallery_finger_id'], 'score': float(s)} 
                          for p, s in zip(gen_pairs, unprot_genuine_scores)]
    imp_records_unprot = [{'subject_a': p['probe_finger_id'], 'subject_b': p['gallery_finger_id'], 'score': float(s)} 
                          for p, s in zip(imp_pairs, unprot_impostor_scores)]
    delta_res = paired_clustered_bootstrap_delta(
        gen_records, imp_records, gen_records_unprot, imp_records_unprot, num_bootstraps=500
    )
    
    # Operating point at frozen dev threshold
    t_frozen = FROZEN_DEV_THRESHOLD
    tp_frozen = int(np.sum(prot_genuine_scores >= t_frozen))
    fn_frozen = int(np.sum(prot_genuine_scores < t_frozen))
    fp_frozen = int(np.sum(prot_impostor_scores >= t_frozen))
    tn_frozen = int(np.sum(prot_impostor_scores < t_frozen))
    assert tp_frozen + fn_frozen == num_gen
    assert fp_frozen + tn_frozen == num_imp
    
    tar_frozen = float(tp_frozen / num_gen)
    far_frozen = float(fp_frozen / num_imp)
    frr_frozen = float(fn_frozen / num_gen)
    
    # Storage accounting
    storage_breakdown = compute_template_storage_breakdown(
        int(round(avg_triangles)), in_dim=9, out_dim=32, weight_bytes=2, header_bytes=32
    )
    
    db_prefix = subset[:3]
    sensor_info = SENSOR_PROVENANCE.get(db_prefix, {}).copy()
    
    res = {
        "subset": subset,
        "sensor_provenance": sensor_info,
        "descriptor_dim": 9,
        "topology": "delaunay",
        "projection_dim": 32,
        "dpi": dpi,
        "total_images": total_images,
        "fta": "N/A (offline database)",
        "fte_count": fte_count,
        "fte_rate": fte_rate,
        "num_genuine": num_gen,
        "num_impostor": num_imp,
        "avg_triangles": avg_triangles,
        "matching_throughput_comp_per_sec": throughput,
        "protected": {
            "eer": float(prot_metrics["eer"]),
            "auc": float(prot_metrics["auc"]),
            "eer_threshold": float(prot_metrics["eer_threshold"]),
            "tar_at_1pct_far": float(prot_metrics["tar_at_1pct_far"]),
            "tar_at_01pct_far": float(prot_metrics["tar_at_01pct_far"]),
            "bootstrap_eer_95ci": [float(boot_res["eer_ci"][0]), float(boot_res["eer_ci"][1])],
            "bootstrap_auc_95ci": [float(boot_res["auc_ci"][0]), float(boot_res["auc_ci"][1])],
            "clustered_eer_95ci": [float(clust_res["eer_ci"][0]), float(clust_res["eer_ci"][1])],
            "clustered_auc_95ci": [float(clust_res["auc_ci"][0]), float(clust_res["auc_ci"][1])],
            "genuine_mean": float(np.mean(prot_genuine_scores)),
            "genuine_std": float(np.std(prot_genuine_scores)),
            "impostor_mean": float(np.mean(prot_impostor_scores)),
            "impostor_std": float(np.std(prot_impostor_scores)),
            "frozen_dev_threshold_eval": {
                "threshold": t_frozen,
                "tar": tar_frozen,
                "far": far_frozen,
                "frr": frr_frozen,
                "confusion": {"tp": tp_frozen, "fn": fn_frozen, "fp": fp_frozen, "tn": tn_frozen}
            }
        },
        "unprotected_baseline": {
            "eer": float(unprot_metrics["eer"]),
            "auc": float(unprot_metrics["auc"]),
            "genuine_mean": float(np.mean(unprot_genuine_scores)),
            "genuine_std": float(np.std(unprot_genuine_scores)),
            "impostor_mean": float(np.mean(unprot_impostor_scores)),
            "impostor_std": float(np.std(unprot_impostor_scores))
        },
        "paired_difference": delta_res,
        "storage": storage_breakdown
    }
    
    # Save individual report
    with open(os.path.join(sub_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
        
    print(f"[{subset}] EER: {prot_metrics['eer']*100:.2f}% (Clustered CI: [{clust_res['eer_ci'][0]*100:.2f}%, {clust_res['eer_ci'][1]*100:.2f}%]), AUC: {prot_metrics['auc']:.4f}")
    print(f"[{subset}] TAR@1%FAR: {prot_metrics['tar_at_1pct_far']*100:.2f}%, At T={t_frozen}: TAR={tar_frozen*100:.2f}%, FAR={far_frozen*100:.2f}%")
    print(f"[{subset}] Genuine Mean: {np.mean(prot_genuine_scores):.4f} ± {np.std(prot_genuine_scores):.4f}, Impostor Mean: {np.mean(prot_impostor_scores):.4f} ± {np.std(prot_impostor_scores):.4f}")
    
    return {
        "summary": res,
        "prot_gen": prot_genuine_scores,
        "prot_imp": prot_impostor_scores,
        "unprot_gen": unprot_genuine_scores,
        "unprot_imp": unprot_impostor_scores,
        "gen_pairs": gen_pairs,
        "imp_pairs": imp_pairs
    }


def evaluate_pooled_performance(
    name: str,
    results_list: List[Dict[str, Any]],
    output_base_dir: str = "experiments/results/phase1/multi_sensor"
) -> Dict[str, Any]:
    """
    Evaluates pooled genuine and impostor score arrays across multiple subsets.
    """
    print(f"\n{'='*75}")
    print(f"  Evaluating Pooled Multi-Sensor Benchmark: {name}")
    print(f"{'='*75}")
    
    pooled_gen = np.concatenate([r["prot_gen"] for r in results_list])
    pooled_imp = np.concatenate([r["prot_imp"] for r in results_list])
    
    num_gen = len(pooled_gen)
    num_imp = len(pooled_imp)
    total_comps = num_gen + num_imp
    
    pooled_dir = os.path.join(output_base_dir, name)
    os.makedirs(pooled_dir, exist_ok=True)
    
    np.save(os.path.join(pooled_dir, "prot_genuine_scores.npy"), pooled_gen)
    np.save(os.path.join(pooled_dir, "prot_impostor_scores.npy"), pooled_imp)
    
    # Export combined manifest
    all_pairs = []
    pair_counter = 0
    for r in results_list:
        for p in r["gen_pairs"]:
            p_copy = p.copy()
            p_copy["pair_id"] = pair_counter
            all_pairs.append(p_copy)
            pair_counter += 1
        for p in r["imp_pairs"]:
            p_copy = p.copy()
            p_copy["pair_id"] = pair_counter
            all_pairs.append(p_copy)
            pair_counter += 1
    export_manifest_csv(all_pairs, os.path.join(pooled_dir, "manifest.csv"))
    
    roc_res = compute_roc_metrics(pooled_gen, pooled_imp)
    boot_res = bootstrap_confidence_intervals(pooled_gen, pooled_imp, num_bootstraps=500)
    
    # Operating point at frozen dev threshold
    t_frozen = FROZEN_DEV_THRESHOLD
    tp_frozen = int(np.sum(pooled_gen >= t_frozen))
    fn_frozen = int(np.sum(pooled_gen < t_frozen))
    fp_frozen = int(np.sum(pooled_imp >= t_frozen))
    tn_frozen = int(np.sum(pooled_imp < t_frozen))
    assert tp_frozen + fn_frozen == num_gen
    assert fp_frozen + tn_frozen == num_imp
    
    tar_frozen = float(tp_frozen / num_gen)
    far_frozen = float(fp_frozen / num_imp)
    frr_frozen = float(fn_frozen / num_gen)
    
    summary = {
        "pooled_benchmark": name,
        "subsets_included": [r["summary"]["subset"] for r in results_list],
        "total_comparisons": total_comps,
        "num_genuine": num_gen,
        "num_impostor": num_imp,
        "eer": float(roc_res["eer"]),
        "auc": float(roc_res["auc"]),
        "eer_threshold": float(roc_res["eer_threshold"]),
        "tar_at_1pct_far": float(roc_res["tar_at_1pct_far"]),
        "tar_at_01pct_far": float(roc_res["tar_at_01pct_far"]),
        "bootstrap_eer_95ci": [float(boot_res["eer_ci"][0]), float(boot_res["eer_ci"][1])],
        "bootstrap_auc_95ci": [float(boot_res["auc_ci"][0]), float(boot_res["auc_ci"][1])],
        "genuine_mean": float(np.mean(pooled_gen)),
        "genuine_std": float(np.std(pooled_gen)),
        "impostor_mean": float(np.mean(pooled_imp)),
        "impostor_std": float(np.std(pooled_imp)),
        "frozen_dev_threshold_eval": {
            "threshold": t_frozen,
            "tar": tar_frozen,
            "far": far_frozen,
            "frr": frr_frozen,
            "confusion": {"tp": tp_frozen, "fn": fn_frozen, "fp": fp_frozen, "tn": tn_frozen}
        }
    }
    
    with open(os.path.join(pooled_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"[{name}] Total: {total_comps} (Gen: {num_gen}, Imp: {num_imp})")
    print(f"[{name}] EER: {roc_res['eer']*100:.2f}% (95% CI: [{boot_res['eer_ci'][0]*100:.2f}%, {boot_res['eer_ci'][1]*100:.2f}%]), AUC: {roc_res['auc']:.4f} (95% CI: [{boot_res['auc_ci'][0]:.4f}, {boot_res['auc_ci'][1]:.4f}])")
    print(f"[{name}] TAR@1%FAR: {roc_res['tar_at_1pct_far']*100:.2f}%, At T={t_frozen}: TAR={tar_frozen*100:.2f}%, FAR={far_frozen*100:.2f}%")
    print(f"[{name}] Genuine: {np.mean(pooled_gen):.4f} ± {np.std(pooled_gen):.4f}, Impostor: {np.mean(pooled_imp):.4f} ± {np.std(pooled_imp):.4f}")
    
    return summary


def run_all_multi_sensor_benchmarks(output_base_dir: str = "experiments/results/phase1/multi_sensor") -> Dict[str, Any]:
    """
    Executes the full multi-sensor benchmark suite across DB1_A, DB1_B, DB2_B, DB3_B, DB4_B,
    and calculates both pooled Set B and pooled all-available statistics.
    """
    subsets = ["DB1_A", "DB1_B", "DB2_B", "DB3_B", "DB4_B"]
    results_map = {}
    
    for s in subsets:
        res = evaluate_sensor_subset(s, output_base_dir=output_base_dir)
        results_map[s] = res
        
    # Pooled Set B (4 sensors: DB1_B, DB2_B, DB3_B, DB4_B)
    set_b_list = [results_map[s] for s in ["DB1_B", "DB2_B", "DB3_B", "DB4_B"]]
    pooled_set_b_summary = evaluate_pooled_performance("pooled_set_b", set_b_list, output_base_dir=output_base_dir)
    
    # Pooled All Available (5 benchmarks: DB1_A + Set B)
    all_list = [results_map[s] for s in subsets]
    pooled_all_summary = evaluate_pooled_performance("pooled_all", all_list, output_base_dir=output_base_dir)
    
    # Master multi-sensor summary report
    master_report = {
        "title": "Phase 1: Multi-Sensor Recognition Performance Benchmark Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "Official FVC2002 standard matching protocol (Genuine: all i<j; Impostor: imp 1 across distinct fingers)",
        "individual_subsets": {s: results_map[s]["summary"] for s in subsets},
        "pooled_benchmarks": {
            "pooled_set_b": pooled_set_b_summary,
            "pooled_all": pooled_all_summary
        },
        "external_acquisition_notice": (
            "FVC2002 DB2_A, DB3_A, and DB4_A require external manual download and remain "
            "documented as pending external acquisition. All available local datasets (DB1_A, "
            "DB1_B, DB2_B, DB3_B, DB4_B) are fully evaluated and reported herein."
        )
    }
    
    master_report_path = os.path.join(output_base_dir, "multi_sensor_report.json")
    with open(master_report_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)
        
    print(f"\n[OK] Master multi-sensor report written to: {master_report_path}")
    return master_report


if __name__ == "__main__":
    run_all_multi_sensor_benchmarks()
