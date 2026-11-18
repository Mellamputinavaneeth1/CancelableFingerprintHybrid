"""
Ablation Study: Canonical vs Cyclic-Aware Triangle Matching and Graph Consensus.
Evaluates:
- Experiment A: Delaunay + Canonical
- Experiment B: Delaunay + Cyclic
- Experiment C: k-NN + Canonical
- Experiment D: k-NN + Cyclic
- Experiment E: Delaunay + Cyclic + Graph Consensus

Runs on identical frozen pair lists on FVC2002 DB1_B (280 genuine, 45 impostor = 325 comparisons).
Saves raw .npy score arrays, manifests, and JSON reports.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.datasets.fvc_loader import load_fvc_dataset
from experiments.protocols.fvc2002_protocol import generate_fvc_protocol_pairs
from experiments.protocols.pair_generator import export_manifest_csv
from src.pipeline import CancelableFingerprintPipeline
from src.graph.delaunay import build_delaunay_triangles, build_knn_graphs
from src.graph.descriptors import extract_all_descriptors, extract_all_cyclic_descriptors
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors
from src.matching.bipartite_matcher import match_protected_templates
from src.matching.graph_consensus import compute_triangle_adjacency, match_protected_with_graph_consensus
from src.evaluation.metrics import (
    compute_roc_metrics, 
    clopper_pearson_ci, 
    bootstrap_confidence_intervals,
    clustered_bootstrap_confidence_intervals
)


def run_cyclic_ablation(subset: str = "DB1_B", 
                         output_dir: str = "experiments/results/cyclic_ablation") -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"  Cyclic-Aware Triangle Matching & Graph Consensus Ablation on {subset}")
    print(f"=======================================================\n")
    
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_fvc_dataset(subset, auto_download=False)
    gen_pairs, imp_pairs = generate_fvc_protocol_pairs(dataset)
    all_pairs = gen_pairs + imp_pairs
    
    # Save frozen manifest
    manifest_path = os.path.join(output_dir, "manifest.csv")
    export_manifest_csv(all_pairs, manifest_path)
    print(f"Saved frozen manifest ({len(all_pairs)} pairs) to: {manifest_path}")
    
    # Pre-extract minutiae for all images
    print("Pre-extracting minutiae across all images...")
    pipe = CancelableFingerprintPipeline(in_dim=9, dpi=dataset["dpi"])
    minutiae_cache = {}
    all_raw_descs = []
    for item in dataset["all_images"]:
        p = item["path"]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            feat = pipe.extract_features(img)
            minutiae_cache[p] = feat["minutiae"]
            if len(feat["minutiae"]) >= 3:
                tris = build_delaunay_triangles(feat["minutiae"], max_edge_length=120.0)
                descs, _ = extract_all_descriptors(tris, dpi=dataset["dpi"], include_types=False)
                if len(descs) > 0:
                    all_raw_descs.append(descs)
        else:
            minutiae_cache[p] = []
            
    if all_raw_descs:
        stacked_all = np.vstack(all_raw_descs)
        emp_mean = np.mean(stacked_all, axis=0)
        emp_std = np.std(stacked_all, axis=0)
    else:
        from src.graph.descriptors import DEFAULT_DESCRIPTOR_MEAN, DEFAULT_DESCRIPTOR_STD
        emp_mean = DEFAULT_DESCRIPTOR_MEAN[:9]
        emp_std = DEFAULT_DESCRIPTOR_STD[:9]
        
    print(f"Computed empirical feature normalization (P_mm mean={emp_mean[2]:.2f}, std={emp_std[2]:.2f}).")
        
    # Derive 9-D projection matrix W
    master_secret = "master_secret_fvc2002_secure_32b_!"  # 34 bytes (>= 32 bytes CSPRNG requirement)
    W9 = derive_orthonormal_matrix(master_secret, salt=f"salt_{subset}", app_id="fvc_ablation", in_dim=9, out_dim=32)
    
    configs = [
        {"name": "Exp_A_Delaunay_Canonical", "topology": "delaunay", "mode": "canonical", "consensus": False},
        {"name": "Exp_B_Delaunay_Cyclic", "topology": "delaunay", "mode": "cyclic", "consensus": False},
        {"name": "Exp_C_kNN_Canonical", "topology": "knn", "mode": "canonical", "consensus": False},
        {"name": "Exp_D_kNN_Cyclic", "topology": "knn", "mode": "cyclic", "consensus": False},
        {"name": "Exp_E_Delaunay_Cyclic_Consensus", "topology": "delaunay", "mode": "cyclic", "consensus": True},
    ]
    
    ablation_results = {}
    
    for cfg in configs:
        cname = cfg["name"]
        topo = cfg["topology"]
        mode = cfg["mode"]
        use_consensus = cfg["consensus"]
        
        print(f"Evaluating {cname} (Topo: {topo}, Mode: {mode}, Consensus: {use_consensus})...")
        exp_dir = os.path.join(output_dir, cname)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Build templates
        templates = {}
        for p, mins in minutiae_cache.items():
            if len(mins) < 3:
                templates[p] = {"bits": np.empty((0, 32), dtype=np.uint8), "wts": np.empty((0,), dtype=np.float32), "adj": np.empty((0, 0), dtype=np.uint8)}
                continue
                
            if topo == "knn":
                tris = build_knn_graphs(mins, k=5, max_edge_length=120.0)
            else:
                tris = build_delaunay_triangles(mins, max_edge_length=120.0)
                
            adj = compute_triangle_adjacency(tris) if use_consensus else np.empty((0, 0), dtype=np.uint8)
            
            if mode == "canonical":
                descs, wts = extract_all_descriptors(tris, dpi=dataset["dpi"], include_types=False)
                if len(descs) > 0:
                    proj = project_descriptors(descs, W9, mean=emp_mean, std=emp_std)
                    bits = quantize_descriptors(proj)
                else:
                    bits = np.empty((0, 32), dtype=np.uint8)
            else:
                # Cyclic
                c_descs, wts = extract_all_cyclic_descriptors(tris, dpi=dataset["dpi"], include_types=False)
                if len(c_descs) > 0:
                    stacked = c_descs.reshape(-1, 9)
                    proj = project_descriptors(stacked, W9, mean=emp_mean, std=emp_std)
                    bits = quantize_descriptors(proj).reshape(len(c_descs), 3, 32)
                else:
                    bits = np.empty((0, 3, 32), dtype=np.uint8)
                    
            templates[p] = {"bits": bits, "wts": wts, "adj": adj}
            
        # Match genuine and impostor
        t_start = time.time()
        gen_scores = []
        for pair in gen_pairs:
            t_q = templates[pair["probe_path"]]
            t_e = templates[pair["gallery_path"]]
            if use_consensus:
                q_bits = t_q["bits"][:, 0, :] if t_q["bits"].ndim == 3 else t_q["bits"]
                s, _ = match_protected_with_graph_consensus(
                    q_bits, t_e["bits"], t_q["wts"], t_e["wts"],
                    t_q["adj"], t_e["adj"], hamming_threshold=0.22
                )
            else:
                if mode == "canonical":
                    s = match_protected_templates(t_q["bits"], t_e["bits"], t_q["wts"], t_e["wts"], hamming_threshold=0.22)
                else:
                    q_bits = t_q["bits"][:, 0, :] if t_q["bits"].ndim == 3 else t_q["bits"]
                    s = match_protected_templates(q_bits, t_e["bits"], t_q["wts"], t_e["wts"], hamming_threshold=0.22)
            gen_scores.append(s)
            
        imp_scores = []
        for pair in imp_pairs:
            t_q = templates[pair["probe_path"]]
            t_e = templates[pair["gallery_path"]]
            if use_consensus:
                q_bits = t_q["bits"][:, 0, :] if t_q["bits"].ndim == 3 else t_q["bits"]
                s, _ = match_protected_with_graph_consensus(
                    q_bits, t_e["bits"], t_q["wts"], t_e["wts"],
                    t_q["adj"], t_e["adj"], hamming_threshold=0.22
                )
            else:
                if mode == "canonical":
                    s = match_protected_templates(t_q["bits"], t_e["bits"], t_q["wts"], t_e["wts"], hamming_threshold=0.22)
                else:
                    q_bits = t_q["bits"][:, 0, :] if t_q["bits"].ndim == 3 else t_q["bits"]
                    s = match_protected_templates(q_bits, t_e["bits"], t_q["wts"], t_e["wts"], hamming_threshold=0.22)
            imp_scores.append(s)
            
        t_elapsed = time.time() - t_start
        throughput = float(len(all_pairs) / max(t_elapsed, 1e-6))
        
        gen_scores = np.array(gen_scores, dtype=np.float32)
        imp_scores = np.array(imp_scores, dtype=np.float32)
        
        # Save raw scores
        np.save(os.path.join(exp_dir, "genuine_scores.npy"), gen_scores)
        np.save(os.path.join(exp_dir, "impostor_scores.npy"), imp_scores)
        
        # Compute metrics
        roc = compute_roc_metrics(gen_scores, imp_scores)
        cp_ci = clopper_pearson_ci(int(round(roc["eer"] * len(gen_scores))), len(gen_scores))
        boot_ci = bootstrap_confidence_intervals(gen_scores, imp_scores, num_bootstraps=500)
        
        # Clustered bootstrap over subjects (B=500)
        gen_recs = [{"subject_a": p["probe_finger_id"], "subject_b": p["gallery_finger_id"], "score": float(s)} 
                    for p, s in zip(gen_pairs, gen_scores)]
        imp_recs = [{"subject_a": p["probe_finger_id"], "subject_b": p["gallery_finger_id"], "score": float(s)} 
                    for p, s in zip(imp_pairs, imp_scores)]
        clust_res = clustered_bootstrap_confidence_intervals(gen_recs, imp_recs, 
                                                             operating_threshold=float(roc["eer_threshold"]), 
                                                             num_bootstraps=500)
        
        exp_summary = {
            "name": cname,
            "topology": topo,
            "mode": mode,
            "consensus": use_consensus,
            "num_genuine": len(gen_scores),
            "num_impostor": len(imp_scores),
            "eer": float(roc["eer"]),
            "auc": float(roc["auc"]),
            "eer_threshold": float(roc["eer_threshold"]),
            "tar_at_1pct_far": float(roc["tar_at_1pct_far"]),
            "tar_at_01pct_far": float(roc["tar_at_01pct_far"]),
            "clopper_pearson_95ci": [float(cp_ci[0]), float(cp_ci[1])],
            "bootstrap_eer_95ci": [float(boot_ci["eer_ci"][0]), float(boot_ci["eer_ci"][1])],
            "bootstrap_auc_95ci": [float(boot_ci["auc_ci"][0]), float(boot_ci["auc_ci"][1])],
            "clustered_eer_95ci": [float(clust_res["eer_ci"][0]), float(clust_res["eer_ci"][1])],
            "clustered_auc_95ci": [float(clust_res["auc_ci"][0]), float(clust_res["auc_ci"][1])],
            "genuine_mean": float(np.mean(gen_scores)),
            "genuine_std": float(np.std(gen_scores)),
            "impostor_mean": float(np.mean(imp_scores)),
            "impostor_std": float(np.std(imp_scores)),
            "throughput_comp_per_sec": throughput
        }
        
        with open(os.path.join(exp_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(exp_summary, f, indent=2)
            
        ablation_results[cname] = exp_summary
        print(f"  -> EER: {roc['eer']*100:.2f}% | AUC: {roc['auc']:.4f} | GenMean: {np.mean(gen_scores):.4f} | ImpMean: {np.mean(imp_scores):.4f}")
        
    # Aggregate report
    summary_path = os.path.join(output_dir, "ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
        
    print(f"\n=======================================================")
    print(f"  Summary table saved to: {summary_path}")
    print(f"=======================================================\n")
    return ablation_results


if __name__ == "__main__":
    run_cyclic_ablation(subset="DB1_B")
