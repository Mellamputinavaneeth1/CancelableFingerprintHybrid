"""
Phase 1: Controlled Recognition Ablations Runner.
Evaluates the 6 required recognition ablation axes:
  1. Descriptor: 9-D pure geometry vs 12-D minutiae-type ablation
  2. Triangle generation: Delaunay triangulation vs k-NN (k=3, 4, 5)
  3. Projection length: M in {32, 64, 128} bits
  4. Representation: Continuous standardized descriptor vs Keyed binary descriptor (quantization loss)
  5. Matching: Block-augmented Hungarian bipartite matching vs Greedy nearest-neighbor (1-to-1 and NN)
  6. Quality weighting: Minutiae coherence reliability weights vs Uniform weighting

Evaluated on FVC2002 DB1_B (development benchmark, N=325) and validated on DB1_A (N=7,750).
Saves all raw .npy score arrays, manifests, and JSON reports in experiments/results/phase1/ablations/.
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
from src.minutiae.extractor import Minutia
from src.graph.delaunay import build_delaunay_triangles, build_knn_graphs
from src.graph.descriptors import extract_all_descriptors, DEFAULT_DESCRIPTOR_MEAN, DEFAULT_DESCRIPTOR_STD, DEFAULT_DESCRIPTOR_MEAN_9D, DEFAULT_DESCRIPTOR_STD_9D
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors
from src.matching.bipartite_matcher import (
    match_protected_templates, 
    match_unprotected_descriptors,
    match_protected_greedy
)
from src.evaluation.metrics import (
    compute_roc_metrics, 
    bootstrap_confidence_intervals,
    paired_clustered_bootstrap_delta
)
from experiments.datasets.fvc_loader import load_fvc_dataset
from experiments.protocols.fvc2002_protocol import generate_fvc_protocol_pairs
from experiments.protocols.pair_generator import export_manifest_csv


def cache_dataset_minutiae(dataset: Dict[str, Any], pipe: CancelableFingerprintPipeline) -> Dict[str, Any]:
    """
    Extracts and caches enhanced image minutiae once to avoid redundant filtering.
    """
    cache = {}
    print(f"Caching minutiae across {dataset['total_images']} images in {dataset['subset']}...")
    for item in dataset["all_images"]:
        fpath = item["path"]
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            cache[fpath] = {"minutiae": [], "fte": True}
            continue
        feat = pipe.extract_features(img)
        cache[fpath] = {
            "minutiae": feat["minutiae"],
            "fte": len(feat["minutiae"]) < 3
        }
    return cache



def run_ablation_condition(
    condition_name: str,
    dataset: Dict[str, Any],
    gen_pairs: List[Dict[str, Any]],
    imp_pairs: List[Dict[str, Any]],
    minutiae_cache: Dict[str, Any],
    master_secret: str,
    in_dim: int = 9,
    topology: str = "delaunay",
    k_nn: int = 5,
    projection_dim: int = 32,
    matcher_algo: str = "hungarian",  # "hungarian", "greedy_1to1", "greedy_nn", "continuous"
    use_quality_weights: bool = True,
    hamming_threshold: float = 0.22,
    continuous_threshold: float = 0.60,
    output_dir: str = "experiments/results/phase1/ablations"
) -> Dict[str, Any]:
    """
    Executes a single ablation condition and returns full metrics and score arrays.
    """
    print(f"\n--- Running Ablation Condition: {condition_name} ---")
    print(f"    Params: Dim={in_dim}D, Topo={topology} (k={k_nn}), M={projection_dim}b, Matcher={matcher_algo}, QualityWeights={use_quality_weights}")
    
    dpi = dataset["dpi"]
    all_pairs = gen_pairs + imp_pairs
    num_gen = len(gen_pairs)
    num_imp = len(imp_pairs)
    
    cond_dir = os.path.join(output_dir, condition_name)
    os.makedirs(cond_dir, exist_ok=True)
    export_manifest_csv(all_pairs, os.path.join(cond_dir, "manifest.csv"))
    
    # Extract triangles & descriptors
    feature_cache = {}
    all_raw_descs = []
    include_types = (in_dim == 12)
    
    for fpath, min_data in minutiae_cache.items():
        if min_data["fte"]:
            feature_cache[fpath] = {
                "descriptors": np.empty((0, in_dim), dtype=np.float32),
                "weights": np.empty((0,), dtype=np.float32),
                "fte": True
            }
            continue
            
        minutiae = min_data["minutiae"]
        if topology == "knn":
            triangles = build_knn_graphs(minutiae, k=k_nn, max_edge_length=120.0)
        else:
            triangles = build_delaunay_triangles(minutiae, max_edge_length=120.0)
            
        descs, wts = extract_all_descriptors(triangles, dpi=dpi, include_types=include_types)
        
        if len(descs) == 0:
            feature_cache[fpath] = {
                "descriptors": descs,
                "weights": wts,
                "fte": True
            }
        else:
            feature_cache[fpath] = {
                "descriptors": descs,
                "weights": wts if use_quality_weights else np.ones_like(wts),
                "fte": False
            }
            all_raw_descs.append(descs)
            
    # Empirical normalization
    if all_raw_descs:
        stacked = np.vstack(all_raw_descs)
        emp_mean = np.mean(stacked, axis=0)
        emp_std = np.std(stacked, axis=0)
    else:
        emp_mean = DEFAULT_DESCRIPTOR_MEAN_9D if in_dim == 9 else DEFAULT_DESCRIPTOR_MEAN
        emp_std = DEFAULT_DESCRIPTOR_STD_9D if in_dim == 9 else DEFAULT_DESCRIPTOR_STD
        
    # Template projection & quantization if binary matcher
    template_cache = {}
    if matcher_algo != "continuous":
        W = derive_orthonormal_matrix(
            master_secret, salt=f"salt_ablation_{condition_name}", app_id=f"fvc2002_{condition_name}",
            in_dim=in_dim, out_dim=projection_dim
        )
        for fpath, data in feature_cache.items():
            if data["fte"]:
                template_cache[fpath] = {
                    "bitstrings": np.empty((0, projection_dim), dtype=np.uint8),
                    "weights": data["weights"]
                }
            else:
                proj = project_descriptors(data["descriptors"], W, mean=emp_mean, std=emp_std)
                bits = quantize_descriptors(proj)
                template_cache[fpath] = {
                    "bitstrings": bits,
                    "weights": data["weights"]
                }
                
    # Run matching
    t_start = time.time()
    gen_scores = []
    imp_scores = []
    
    # Genuine
    for p in gen_pairs:
        p_path, g_path = p["probe_path"], p["gallery_path"]
        f_q, f_e = feature_cache[p_path], feature_cache[g_path]
        
        if matcher_algo == "continuous":
            s = match_unprotected_descriptors(
                f_q["descriptors"], f_e["descriptors"], f_q["weights"], f_e["weights"],
                distance_threshold=continuous_threshold, mean=emp_mean, std=emp_std
            )
        elif matcher_algo == "hungarian":
            t_q, t_e = template_cache[p_path], template_cache[g_path]
            s = match_protected_templates(
                t_q["bitstrings"], t_e["bitstrings"], t_q["weights"], t_e["weights"],
                hamming_threshold=hamming_threshold
            )
        elif matcher_algo in ["greedy_1to1", "greedy_nn"]:
            t_q, t_e = template_cache[p_path], template_cache[g_path]
            s = match_protected_greedy(
                t_q["bitstrings"], t_e["bitstrings"], t_q["weights"], t_e["weights"],
                hamming_threshold=hamming_threshold, mode=matcher_algo
            )
        else:
            raise ValueError(f"Unknown matcher algorithm: {matcher_algo}")
            
        gen_scores.append(s)
        
    # Impostor
    for p in imp_pairs:
        p_path, g_path = p["probe_path"], p["gallery_path"]
        f_q, f_e = feature_cache[p_path], feature_cache[g_path]
        
        if matcher_algo == "continuous":
            s = match_unprotected_descriptors(
                f_q["descriptors"], f_e["descriptors"], f_q["weights"], f_e["weights"],
                distance_threshold=continuous_threshold, mean=emp_mean, std=emp_std
            )
        elif matcher_algo == "hungarian":
            t_q, t_e = template_cache[p_path], template_cache[g_path]
            s = match_protected_templates(
                t_q["bitstrings"], t_e["bitstrings"], t_q["weights"], t_e["weights"],
                hamming_threshold=hamming_threshold
            )
        elif matcher_algo in ["greedy_1to1", "greedy_nn"]:
            t_q, t_e = template_cache[p_path], template_cache[g_path]
            s = match_protected_greedy(
                t_q["bitstrings"], t_e["bitstrings"], t_q["weights"], t_e["weights"],
                hamming_threshold=hamming_threshold, mode=matcher_algo
            )
            
        imp_scores.append(s)
        
    t_end = time.time()
    throughput = float(len(all_pairs) / max(t_end - t_start, 1e-6))
    
    gen_scores = np.array(gen_scores, dtype=np.float32)
    imp_scores = np.array(imp_scores, dtype=np.float32)
    
    # Save raw scores
    np.save(os.path.join(cond_dir, "prot_genuine_scores.npy"), gen_scores)
    np.save(os.path.join(cond_dir, "prot_impostor_scores.npy"), imp_scores)
    
    # Metrics
    metrics = compute_roc_metrics(gen_scores, imp_scores)
    boot = bootstrap_confidence_intervals(gen_scores, imp_scores, num_bootstraps=500)
    
    result = {
        "condition_name": condition_name,
        "dataset": dataset["subset"],
        "in_dim": in_dim,
        "topology": topology,
        "k_nn": k_nn if topology == "knn" else None,
        "projection_dim": projection_dim,
        "matcher_algo": matcher_algo,
        "use_quality_weights": use_quality_weights,
        "num_genuine": num_gen,
        "num_impostor": num_imp,
        "eer": float(metrics["eer"]),
        "auc": float(metrics["auc"]),
        "eer_threshold": float(metrics["eer_threshold"]),
        "tar_at_1pct_far": float(metrics["tar_at_1pct_far"]),
        "tar_at_01pct_far": float(metrics["tar_at_01pct_far"]),
        "bootstrap_eer_95ci": [float(boot["eer_ci"][0]), float(boot["eer_ci"][1])],
        "bootstrap_auc_95ci": [float(boot["auc_ci"][0]), float(boot["auc_ci"][1])],
        "genuine_mean": float(np.mean(gen_scores)),
        "genuine_std": float(np.std(gen_scores)),
        "impostor_mean": float(np.mean(imp_scores)),
        "impostor_std": float(np.std(imp_scores)),
        "throughput_comp_per_sec": throughput
    }
    
    with open(os.path.join(cond_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        
    print(f"    --> EER: {metrics['eer']*100:.2f}% [{boot['eer_ci'][0]*100:.2f}%, {boot['eer_ci'][1]*100:.2f}%], AUC: {metrics['auc']:.4f} [{boot['auc_ci'][0]:.4f}, {boot['auc_ci'][1]:.4f}]")
    print(f"    --> TAR@1%FAR: {metrics['tar_at_1pct_far']*100:.2f}%, Gen: {np.mean(gen_scores):.4f} ± {np.std(gen_scores):.4f}, Imp: {np.mean(imp_scores):.4f} ± {np.std(imp_scores):.4f}")
    
    return {
        "summary": result,
        "gen_scores": gen_scores,
        "imp_scores": imp_scores,
        "gen_pairs": gen_pairs,
        "imp_pairs": imp_pairs
    }


def run_full_phase1_ablations(output_base_dir: str = "experiments/results/phase1/ablations") -> Dict[str, Any]:
    """
    Executes all 6 controlled recognition ablations on DB1_B (development) and validates on DB1_A.
    """
    os.makedirs(output_base_dir, exist_ok=True)
    master_secret = "CancelableBioMasterSecret_2026_256Bits_KMS!"
    pipe = CancelableFingerprintPipeline(in_dim=9, dpi=500.0)
    
    # Load DB1_B
    ds_b = load_fvc_dataset("DB1_B", auto_download=False)
    gen_pairs_b, imp_pairs_b = generate_fvc_protocol_pairs(ds_b)
    min_cache_b = cache_dataset_minutiae(ds_b, pipe)
    
    ablation_results = {}
    
    # -------------------------------------------------------------
    # 0. Reference Candidate: 9D, Delaunay, M=32, Hungarian, Weights
    # -------------------------------------------------------------
    ref = run_ablation_condition(
        "ref_9D_delaunay_32b_hungarian_weights",
        ds_b, gen_pairs_b, imp_pairs_b, min_cache_b, master_secret,
        in_dim=9, topology="delaunay", projection_dim=32, matcher_algo="hungarian", use_quality_weights=True,
        output_dir=output_base_dir
    )
    ablation_results["reference"] = ref["summary"]
    
    # -------------------------------------------------------------
    # 1. Descriptor Ablation: 9-D pure geometry vs 12-D minutiae-type
    # -------------------------------------------------------------
    desc_12d = run_ablation_condition(
        "ablation_desc_12D_minutiae_type",
        ds_b, gen_pairs_b, imp_pairs_b, min_cache_b, master_secret,
        in_dim=12, topology="delaunay", projection_dim=32, matcher_algo="hungarian", use_quality_weights=True,
        output_dir=output_base_dir
    )
    ablation_results["descriptor_12D"] = desc_12d["summary"]
    
    # -------------------------------------------------------------
    # 2. Triangle Generation: Delaunay vs k-NN (k=3, 4, 5)
    # -------------------------------------------------------------
    knn_k3 = run_ablation_condition(
        "ablation_topology_knn_k3",
        ds_b, gen_pairs_b, imp_pairs_b, min_cache_b, master_secret,
        in_dim=9, topology="knn", k_nn=3, projection_dim=32, matcher_algo="hungarian", use_quality_weights=True,
        output_dir=output_base_dir
    )
    ablation_results["topology_knn_k3"] = knn_k3["summary"]
    
    knn_k4 = run_ablation_condition(
        "ablation_topology_knn_k4",
        ds_b, gen_pairs_b, imp_pairs_b, min_cache_b, master_secret,
        in_dim=9, topology="knn", k_nn=4, projection_dim=32, matcher_algo="hungarian", use_quality_weights=True,
        output_dir=output_base_dir
    )
    ablation_results["topology_knn_k4"] = knn_k4["summary"]
    
    knn_k5 = run_ablation_condition(
        "ablation_topology_knn_k5",
        ds_b, gen_pairs_b, imp_pairs_b, min_cache_b, master_secret,
        in_dim=9, topology="knn", k_nn=5, projection_dim=32, matcher_algo="hungarian", use_quality_weights=True,
        output_dir=output_base_dir
    )
    ablation_results["topology_knn_k5"] = knn_k5["summary"]
    
    # -------------------------------------------------------------
    # 3. Projection Length: M in {32, 64, 128} bits
    # -------------------------------------------------------------
    proj_64 = run_ablation_condition(
        "ablation_projection_64b",
        ds_b, gen_pairs_b, imp_pairs_b, min_cache_b, master_secret,
        in_dim=9, topology="delaunay", projection_dim=64, matcher_algo="hungarian", use_quality_weights=True,
        output_dir=output_base_dir
    )
    ablation_results["projection_64b"] = proj_64["summary"]
    
    proj_128 = run_ablation_condition(
        "ablation_projection_128b",
        ds_b, gen_pairs_b, imp_pairs_b, min_cache_b, master_secret,
        in_dim=9, topology="delaunay", projection_dim=128, matcher_algo="hungarian", use_quality_weights=True,
        output_dir=output_base_dir
    )
    ablation_results["projection_128b"] = proj_128["summary"]
    
    # -------------------------------------------------------------
    # 4. Representation: Continuous standardized vs Keyed binary (quantization loss)
    # -------------------------------------------------------------
    rep_cont = run_ablation_condition(
        "ablation_rep_continuous_standardized",
        ds_b, gen_pairs_b, imp_pairs_b, min_cache_b, master_secret,
        in_dim=9, topology="delaunay", projection_dim=32, matcher_algo="continuous", use_quality_weights=True,
        output_dir=output_base_dir
    )
    ablation_results["representation_continuous"] = rep_cont["summary"]
    
    # -------------------------------------------------------------
    # 5. Matching Algorithm: Block-augmented Hungarian vs Greedy (1-to-1 and NN)
    # -------------------------------------------------------------
    match_greedy_1to1 = run_ablation_condition(
        "ablation_matching_greedy_1to1",
        ds_b, gen_pairs_b, imp_pairs_b, min_cache_b, master_secret,
        in_dim=9, topology="delaunay", projection_dim=32, matcher_algo="greedy_1to1", use_quality_weights=True,
        output_dir=output_base_dir
    )
    ablation_results["matching_greedy_1to1"] = match_greedy_1to1["summary"]
    
    match_greedy_nn = run_ablation_condition(
        "ablation_matching_greedy_nn",
        ds_b, gen_pairs_b, imp_pairs_b, min_cache_b, master_secret,
        in_dim=9, topology="delaunay", projection_dim=32, matcher_algo="greedy_nn", use_quality_weights=True,
        output_dir=output_base_dir
    )
    ablation_results["matching_greedy_nn"] = match_greedy_nn["summary"]
    
    # -------------------------------------------------------------
    # 6. Quality Weighting: Coherence weights vs Uniform weighting
    # -------------------------------------------------------------
    weight_uniform = run_ablation_condition(
        "ablation_weighting_uniform",
        ds_b, gen_pairs_b, imp_pairs_b, min_cache_b, master_secret,
        in_dim=9, topology="delaunay", projection_dim=32, matcher_algo="hungarian", use_quality_weights=False,
        output_dir=output_base_dir
    )
    ablation_results["weighting_uniform"] = weight_uniform["summary"]
    
    # -------------------------------------------------------------
    # Validation Sweep on DB1_A (Large Benchmark, N=7,750)
    # -------------------------------------------------------------
    print(f"\n{'='*75}")
    print("  Executing Key Ablation Cross-Validation Sweep on DB1_A (N=7,750)")
    print(f"{'='*75}")
    ds_a = load_fvc_dataset("DB1_A", auto_download=False)
    gen_pairs_a, imp_pairs_a = generate_fvc_protocol_pairs(ds_a)
    min_cache_a = cache_dataset_minutiae(ds_a, pipe)
    
    # DB1_A Reference (9D, Hungarian, Coherence weights, M=32)
    ref_a = run_ablation_condition(
        "db1_a_ref_9D_delaunay_32b_hungarian_weights",
        ds_a, gen_pairs_a, imp_pairs_a, min_cache_a, master_secret,
        in_dim=9, topology="delaunay", projection_dim=32, matcher_algo="hungarian", use_quality_weights=True,
        output_dir=output_base_dir
    )
    ablation_results["db1_a_reference"] = ref_a["summary"]
    
    # DB1_A 12-D Descriptor
    desc_12d_a = run_ablation_condition(
        "db1_a_ablation_desc_12D",
        ds_a, gen_pairs_a, imp_pairs_a, min_cache_a, master_secret,
        in_dim=12, topology="delaunay", projection_dim=32, matcher_algo="hungarian", use_quality_weights=True,
        output_dir=output_base_dir
    )
    ablation_results["db1_a_descriptor_12D"] = desc_12d_a["summary"]
    
    # DB1_A Greedy Matcher
    greedy_a = run_ablation_condition(
        "db1_a_ablation_matching_greedy_1to1",
        ds_a, gen_pairs_a, imp_pairs_a, min_cache_a, master_secret,
        in_dim=9, topology="delaunay", projection_dim=32, matcher_algo="greedy_1to1", use_quality_weights=True,
        output_dir=output_base_dir
    )
    ablation_results["db1_a_matching_greedy_1to1"] = greedy_a["summary"]
    
    # DB1_A Uniform Weighting
    uni_a = run_ablation_condition(
        "db1_a_ablation_weighting_uniform",
        ds_a, gen_pairs_a, imp_pairs_a, min_cache_a, master_secret,
        in_dim=9, topology="delaunay", projection_dim=32, matcher_algo="hungarian", use_quality_weights=False,
        output_dir=output_base_dir
    )
    ablation_results["db1_a_weighting_uniform"] = uni_a["summary"]
    
    # DB1_A Continuous Unprotected Baseline
    cont_a = run_ablation_condition(
        "db1_a_ablation_rep_continuous",
        ds_a, gen_pairs_a, imp_pairs_a, min_cache_a, master_secret,
        in_dim=9, topology="delaunay", projection_dim=32, matcher_algo="continuous", use_quality_weights=True,
        output_dir=output_base_dir
    )
    ablation_results["db1_a_representation_continuous"] = cont_a["summary"]
    
    # Master ablation report
    report = {
        "title": "Phase 1: Controlled Recognition Factorial Ablation Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_configuration": {
            "descriptor": "9-D Pure Geometry",
            "topology": "Delaunay Triangulation (max_edge=120px)",
            "projection_length": "M = 32 bits",
            "representation": "Keyed 1-Bit Binary Indicator Quantization",
            "matching": "Block-Augmented Hungarian Bipartite Matching",
            "weighting": "Minutiae Directional Coherence Reliability Weights"
        },
        "ablations": ablation_results
    }
    
    report_path = os.path.join(output_base_dir, "ablations_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\n[OK] Master ablations report written to: {report_path}")
    return report


if __name__ == "__main__":
    run_full_phase1_ablations()
