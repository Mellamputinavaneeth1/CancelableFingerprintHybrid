"""
System Unlinkability Evaluation on Real Fingerprint Dataset (FVC2002 DB1_B).
Evaluates the Gómez-Barrero et al. D_sys framework (IEEE TIFS 2018):
- Balanced comparison design across all K(K-1)/2 = 10 application key pairs.
- Mated distribution (S_cross): same subject, different keys (10 subjects x 10 pairs = 100 trials).
- Non-mated distribution (S_diff): different subjects, different keys (45 pairs x 10 = 450 trials).
- Evaluates 3 distinct threat models:
    1. Descriptor-only attacker (b_i vs b_j).
    2. Topology-only attacker (A_i vs A_j using degree/mesh similarity).
    3. Combined attacker (Descriptor matching with topological graph consensus).
- Subject-level clustered bootstrap confidence intervals (B >= 500).
- Pure geometric 9-D primary candidate.
- Saves raw distribution .npy arrays and comprehensive JSON audit report.
"""

import os
import sys
import json
from typing import Dict, Any, List
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.datasets.fvc_loader import load_fvc_dataset
from src.pipeline import CancelableFingerprintPipeline
from src.graph.delaunay import build_delaunay_triangles
from src.graph.descriptors import extract_all_descriptors
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors
from src.matching.bipartite_matcher import match_protected_templates
from src.matching.graph_consensus import compute_triangle_adjacency, match_protected_with_graph_consensus
from src.security.unlinkability import (
    compute_system_unlinkability,
    compute_system_unlinkability_clustered_ci,
    compute_multi_threat_unlinkability_bootstrap,
    match_topology_only
)


def run_system_unlinkability_eval(subset: str = "DB1_B", 
                                  num_keys: int = 5,
                                  output_dir: str = "experiments/results/unlinkability") -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"  System Unlinkability Evaluation on {subset} ({num_keys} Application Keys)")
    print(f"  Threat Models: Descriptor-Only | Topology-Only | Combined")
    print(f"=======================================================\n")
    
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_fvc_dataset(subset, auto_download=False)
    dpi = dataset["dpi"]
    
    master_secret = "master_secret_fvc2002_secure_32b_!"
    app_keys = [f"app_banking_{i}" for i in range(num_keys)]
    
    # 1. Derive orthonormal projection matrices for each application domain
    W_matrices = [
        derive_orthonormal_matrix(master_secret, salt=f"salt_{subset}", app_id=key_id, in_dim=9, out_dim=32)
        for key_id in app_keys
    ]
    
    pipe = CancelableFingerprintPipeline(in_dim=9, dpi=dpi)
    fingers = dataset["fingers"]
    subject_ids = sorted(list(fingers.keys()))
    n_subj = len(subject_ids)
    
    # 2. Extract features across impressions for each subject to compute empirical standardization
    all_raw_descs = []
    subject_features = {}  # {fid: {imp_idx: (descs, weights, adj, tris)}}
    
    print(f"Extracting 9-D geometric features across {n_subj} subjects (5 impressions each)...")
    for fid in subject_ids:
        subject_features[fid] = {}
        for k in range(num_keys):
            imp_idx = k + 1  # Impressions 1 through num_keys
            path = fingers[fid][imp_idx]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            feat = pipe.extract_features(img)
            tris = build_delaunay_triangles(feat["minutiae"], max_edge_length=120.0)
            descs, wts = extract_all_descriptors(tris, dpi=dpi, include_types=False)
            adj = compute_triangle_adjacency(tris)
            
            subject_features[fid][k] = {
                "descriptors": descs,
                "weights": wts,
                "adj": adj
            }
            if len(descs) > 0:
                all_raw_descs.append(descs)
                
    if all_raw_descs:
        stacked = np.vstack(all_raw_descs)
        emp_mean = np.mean(stacked, axis=0)
        emp_std = np.std(stacked, axis=0)
    else:
        from src.graph.descriptors import DEFAULT_DESCRIPTOR_MEAN, DEFAULT_DESCRIPTOR_STD
        emp_mean = DEFAULT_DESCRIPTOR_MEAN[:9]
        emp_std = DEFAULT_DESCRIPTOR_STD[:9]
        
    # 3. Generate enrolled credentials: Key k applied to Impression (k + 1)
    # Credential = (bitstrings_k, weights_k, adj_k)
    enrolled_data = {}  # {fid: {k: {"bitstrings": ..., "weights": ..., "adj": ...}}}
    for fid in subject_ids:
        enrolled_data[fid] = {}
        for k in range(num_keys):
            raw = subject_features[fid].get(k, {})
            descs = raw.get("descriptors", np.empty((0, 9), dtype=np.float32))
            wts = raw.get("weights", np.empty((0,), dtype=np.float32))
            adj = raw.get("adj", np.empty((0, 0), dtype=np.uint8))
            
            if len(descs) > 0:
                W = W_matrices[k]
                proj = project_descriptors(descs, W, mean=emp_mean, std=emp_std)
                bits = quantize_descriptors(proj)
            else:
                bits = np.empty((0, 32), dtype=np.uint8)
                
            enrolled_data[fid][k] = {
                "bitstrings": bits,
                "weights": wts,
                "adj": adj
            }
            
    # 4. Generate all K(K-1)/2 key pairs (Balanced Design)
    key_pairs = []
    for k1 in range(num_keys):
        for k2 in range(k1 + 1, num_keys):
            key_pairs.append((k1, k2))
    num_key_pairs = len(key_pairs)
    print(f"Generated {num_key_pairs} cross-application key pairs for balanced evaluation.")
    
    # 5. Evaluate all three threat models
    threat_models = ["descriptor_only", "topology_only", "combined"]
    model_records = {m: {"mated": [], "non_mated": []} for m in threat_models}
    
    print("\nExecuting balanced comparisons across all key pairs...")
    
    # Mated comparisons: same subject, different keys
    for fid in subject_ids:
        for (k1, k2) in key_pairs:
            d1 = enrolled_data[fid][k1]
            d2 = enrolled_data[fid][k2]
            
            # Model 1: Descriptor-only
            s_desc = match_protected_templates(
                d1["bitstrings"], d2["bitstrings"], d1["weights"], d2["weights"], hamming_threshold=0.22
            )
            model_records["descriptor_only"]["mated"].append({
                "subject": fid, "score": s_desc, "key1": k1, "key2": k2
            })
            
            # Model 2: Topology-only
            s_topo = match_topology_only(d1["adj"], d2["adj"])
            model_records["topology_only"]["mated"].append({
                "subject": fid, "score": s_topo, "key1": k1, "key2": k2
            })
            
            # Model 3: Combined
            s_comb, _ = match_protected_with_graph_consensus(
                d1["bitstrings"], d2["bitstrings"], d1["weights"], d2["weights"],
                d1["adj"], d2["adj"], hamming_threshold=0.22
            )
            model_records["combined"]["mated"].append({
                "subject": fid, "score": s_comb, "key1": k1, "key2": k2
            })
            
    # Non-mated comparisons: different subjects, different keys
    for i in range(n_subj):
        fid_a = subject_ids[i]
        for j in range(i + 1, n_subj):
            fid_b = subject_ids[j]
            for (k1, k2) in key_pairs:
                d1 = enrolled_data[fid_a][k1]
                d2 = enrolled_data[fid_b][k2]
                
                # Model 1: Descriptor-only
                s_desc = match_protected_templates(
                    d1["bitstrings"], d2["bitstrings"], d1["weights"], d2["weights"], hamming_threshold=0.22
                )
                model_records["descriptor_only"]["non_mated"].append({
                    "subject_a": fid_a, "subject_b": fid_b, "score": s_desc, "key1": k1, "key2": k2
                })
                
                # Model 2: Topology-only
                s_topo = match_topology_only(d1["adj"], d2["adj"])
                model_records["topology_only"]["non_mated"].append({
                    "subject_a": fid_a, "subject_b": fid_b, "score": s_topo, "key1": k1, "key2": k2
                })
                
                # Model 3: Combined
                s_comb, _ = match_protected_with_graph_consensus(
                    d1["bitstrings"], d2["bitstrings"], d1["weights"], d2["weights"],
                    d1["adj"], d2["adj"], hamming_threshold=0.22
                )
                model_records["combined"]["non_mated"].append({
                    "subject_a": fid_a, "subject_b": fid_b, "score": s_comb, "key1": k1, "key2": k2
                })
                
    # 6. Analyze and export results for each threat model
    report = {
        "dataset": subset,
        "num_subjects": n_subj,
        "num_keys": num_keys,
        "num_key_pairs": num_key_pairs,
        "models": {}
    }
    
    print("\n-------------------------------------------------------")
    print(f"{'Threat Model':<20} | {'Mated N':<7} | {'Non-Mated N':<11} | {'D_sys':<7} | {'95% Clustered CI':<18} | {'Status'}")
    print("-------------------------------------------------------------------------------------------------------")
    
    # Run multi-threat joint two-way clustered bootstrap (B=2000)
    print("Running joint multi-threat two-way cluster bootstrap (B=2000 replicates across subjects & keys)...")
    multi_boot_res = compute_multi_threat_unlinkability_bootstrap(
        model_records, num_bins=100, score_domain=(0.0, 1.0),
        num_bootstraps=2000, omega=1.0, bw_factor=1.0, seed=42
    )
    
    bandwidth_results = {}
    
    for m_name in threat_models:
        mated_recs = model_records[m_name]["mated"]
        non_mated_recs = model_records[m_name]["non_mated"]
        
        m_scores = np.array([r["score"] for r in mated_recs], dtype=np.float64)
        nm_scores = np.array([r["score"] for r in non_mated_recs], dtype=np.float64)
        
        # Save raw .npy arrays
        np.save(os.path.join(output_dir, f"mated_scores_{m_name}.npy"), m_scores)
        np.save(os.path.join(output_dir, f"non_mated_scores_{m_name}.npy"), nm_scores)
        
        ds = float(multi_boot_res["models"][m_name]["d_sys"])
        ci_l = float(multi_boot_res["models"][m_name]["ci_95"][0])
        ci_u = float(multi_boot_res["models"][m_name]["ci_95"][1])
        
        # Bandwidth sensitivity sweep
        bw_sweep = {}
        for factor in [0.5, 0.75, 1.0, 1.5, 2.0]:
            ds_bw, _, _, _ = compute_system_unlinkability(
                m_scores, nm_scores, num_bins=100, score_domain=(0.0, 1.0), 
                omega=1.0, bw_factor=factor
            )
            bw_sweep[f"bw_{factor:.2f}"] = float(ds_bw)
        bandwidth_results[m_name] = bw_sweep
        
        if ds <= 0.05 and ci_u <= 0.05:
            status = "Fully Unlinkable (D_sys <= 0.05, Upper CI <= 0.05)"
        elif ds <= 0.05:
            status = "Low Residual Linkability (D_sys <= 0.05, Upper CI > 0.05)"
        elif ds < 0.20:
            status = "Low Residual Linkability (D_sys < 0.20)"
        elif ds < 0.50:
            status = "Semi-Unlinkable (0.20 <= D_sys < 0.50)"
        else:
            status = "Linkable (D_sys >= 0.50)"
            
        report["models"][m_name] = {
            "num_mated": len(m_scores),
            "num_non_mated": len(nm_scores),
            "mated_mean": float(np.mean(m_scores)),
            "mated_std": float(np.std(m_scores)),
            "non_mated_mean": float(np.mean(nm_scores)),
            "non_mated_std": float(np.std(nm_scores)),
            "d_sys": ds,
            "clustered_ci_95": [ci_l, ci_u],
            "bandwidth_sensitivity": bw_sweep,
            "status": status
        }
        
        print(f"{m_name:<20} | {len(m_scores):<7} | {len(nm_scores):<11} | {ds:<7.4f} | [{ci_l:.4f}, {ci_u:.4f}]   | {status}")
        
    print("-------------------------------------------------------------------------------------------------------\n")
    
    # Save standalone bandwidth sensitivity report
    bw_path = os.path.join(output_dir, "bandwidth_sensitivity.json")
    with open(bw_path, "w", encoding="utf-8") as f:
        json.dump(bandwidth_results, f, indent=2)
    print(f"Saved bandwidth sensitivity report to: {bw_path}")
    
    # Calculate maximum linkage observed across all threat models with joint bootstrap CI
    d_sys_max = float(multi_boot_res["d_sys_max_point"])
    ci_max_l = float(multi_boot_res["d_sys_max_ci_95"][0])
    ci_max_u = float(multi_boot_res["d_sys_max_ci_95"][1])
    
    if d_sys_max <= 0.05 and ci_max_u <= 0.05:
        system_status = "Fully Unlinkable across Evaluated Threat Models (D_sys_max <= 0.05)"
    elif d_sys_max < 0.20:
        system_status = f"Low Residual Linkability across Evaluated Threat Models (D_sys_max = {d_sys_max:.4f}, Upper CI = {ci_max_u:.4f})"
    elif d_sys_max < 0.50:
        system_status = f"Semi-Unlinkable (D_sys_max = {d_sys_max:.4f})"
    else:
        system_status = f"Linkable (D_sys_max = {d_sys_max:.4f})"
        
    print(f"Maximum Linkage D_sys_max across threat models: {d_sys_max:.4f} (95% CI: [{ci_max_l:.4f}, {ci_max_u:.4f}])")
    print(f"Overall System Privacy Status: {system_status}\n")
    
    # Save primary aliases for backward compatibility
    np.save(os.path.join(output_dir, "mated_cross_key_scores.npy"), 
            np.array([r["score"] for r in model_records["descriptor_only"]["mated"]]))
    np.save(os.path.join(output_dir, "non_mated_cross_key_scores.npy"), 
            np.array([r["score"] for r in model_records["descriptor_only"]["non_mated"]]))
    
    report["d_sys"] = report["models"]["descriptor_only"]["d_sys"]
    report["clustered_bootstrap_95ci"] = report["models"]["descriptor_only"]["clustered_ci_95"]
    report["d_sys_max"] = d_sys_max
    report["d_sys_max_ci_95"] = [ci_max_l, ci_max_u]
    report["status"] = report["models"]["descriptor_only"]["status"]
    report["system_privacy_status"] = system_status
    report["system_privacy_status"] = system_status
    
    report_path = os.path.join(output_dir, "unlinkability_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved comprehensive unlinkability report to: {report_path}\n")
    return report


if __name__ == "__main__":
    run_system_unlinkability_eval()
