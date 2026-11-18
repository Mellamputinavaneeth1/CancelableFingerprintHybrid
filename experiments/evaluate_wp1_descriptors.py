"""
WP1: Descriptor Mathematics & Candidate Representation Benchmark.

Evaluates the 5 candidate representations under identical, controlled conditions on DB1_B:
1. 12-D Minutiae-Type Leakage Ablation (9-D geometry + t0, t1, t2)
2. 9-D Primary Geometric Baseline (r1, r2, P_mm, phi0, phi1, phi2, dth01, dth12, dth20)
3. 8-D Non-Redundant Candidate (r1, r2, P_mm, phi0, phi1, phi2, dth01, dth12)
4. 6-D Minimal Manifold Candidate (r1, r2, P_mm, phi0, phi1, phi2)
5. 5-D Reduced Scale-Invariant Candidate (L1, L2, dth01, dth12, dth20)
"""

import os
import sys
import json
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import CancelableFingerprintPipeline
from src.matching.bipartite_matcher import match_protected_templates
from src.evaluation.metrics import compute_roc_metrics, compute_confusion_at_threshold
from src.graph.descriptors import extract_all_descriptors
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors
from experiments.datasets.fvc_loader import load_fvc_dataset
from experiments.protocols.fvc2002_protocol import generate_fvc_protocol_pairs

OUTPUT_DIR = "experiments/results/wp1"

CANDIDATES = [
    {
        "name": "12-D Minutiae-Type Leakage Ablation",
        "dim": 12,
        "include_types": True,
        "features": ["r1", "r2", "P_mm", "phi0", "phi1", "phi2", "dth01", "dth12", "dth20", "t0", "t1", "t2"],
        "description": "Stores discrete minutiae types (ending=0, bifurcation=1). Susceptible to MILP inversion."
    },
    {
        "name": "9-D Primary Geometric Baseline",
        "dim": 9,
        "include_types": False,
        "features": ["r1", "r2", "P_mm", "phi0", "phi1", "phi2", "dth01", "dth12", "dth20"],
        "description": "Complete rigid geometry with redundant mutual orientations. Primary production system."
    },
    {
        "name": "8-D Non-Redundant Mutual Angle Candidate",
        "dim": 8,
        "include_types": False,
        "features": ["r1", "r2", "P_mm", "phi0", "phi1", "phi2", "dth01", "dth12"],
        "description": "Eliminates dth20 redundancy (since directed angle sum mod pi = 0)."
    },
    {
        "name": "6-D Minimal Non-Redundant Manifold",
        "dim": 6,
        "include_types": False,
        "features": ["r1", "r2", "P_mm", "phi0", "phi1", "phi2"],
        "description": "Minimal intrinsic manifold dim(SE(2)) = 6. All mutual angles determined by geometry."
    },
    {
        "name": "5-D Reduced Scale-Invariant Candidate",
        "dim": 5,
        "include_types": False,
        "features": ["L1", "L2", "dth01", "dth12", "dth20"],
        "description": "Discards physical perimeter and edge relative angles; scale and rotation invariant."
    }
]


def evaluate_candidates():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fvc_data = load_fvc_dataset("DB1_B", root_dir="dataset/FVC2002")
    gen_pair_dicts, imp_pair_dicts = generate_fvc_protocol_pairs(fvc_data)
    print(f"Loaded {len(gen_pair_dicts)} genuine pairs and {len(imp_pair_dicts)} impostor pairs for DB1_B.")

    # 1. Preload and extract base minutiae & triangles for all 80 images
    print("Pre-extracting Delaunay triangulations for DB1_B (80 images)...")
    base_pipeline = CancelableFingerprintPipeline(in_dim=9, use_gpu=True)
    image_cache = {}
    for fid, imps in fvc_data["fingers"].items():
        for iid, img_path in imps.items():
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            feat = base_pipeline.extract_features(img)
            image_cache[img_path] = feat

    results = []

    for cand in CANDIDATES:
        c_name = cand["name"]
        dim = cand["dim"]
        inc_types = cand["include_types"]
        print(f"\nEvaluating: {c_name} (dim={dim})...")

        pipeline = CancelableFingerprintPipeline(
            in_dim=dim,
            master_secret="CancelableBioMasterSecret_2026_256Bits_KMS!",
            use_gpu=True
        )

        # Extract descriptors across all images and compute dev-set standardization statistics
        all_candidate_descs = []
        for img_path, feat in image_cache.items():
            triangles = feat["triangles"]
            descs, weights = extract_all_descriptors(
                triangles, dpi=500.0, include_types=inc_types, desc_dim=dim
            )
            if len(descs) > 0:
                all_candidate_descs.append(descs)

        if all_candidate_descs:
            stacked_descs = np.vstack(all_candidate_descs)
            emp_mean = np.mean(stacked_descs, axis=0)
            emp_std = np.std(stacked_descs, axis=0)
        else:
            emp_mean = None
            emp_std = None

        # Derive column-orthonormal matrix W (32 x dim)
        W = derive_orthonormal_matrix(
            pipeline.master_secret, salt="salt_DB1_B", app_id="fvc2002_DB1_B", in_dim=dim, out_dim=32
        )

        # Enroll all images into protected templates
        templates = {}
        for img_path, feat in image_cache.items():
            triangles = feat["triangles"]
            descs, weights = extract_all_descriptors(
                triangles, dpi=500.0, include_types=inc_types, desc_dim=dim
            )
            if len(descs) > 0:
                proj = project_descriptors(descs, W, mean=emp_mean, std=emp_std)
                bits = quantize_descriptors(proj)
            else:
                bits = np.empty((0, 32), dtype=np.uint8)
            templates[img_path] = {"bitstrings": bits, "weights": weights}

        # Score genuine pairs
        gen_scores = []
        start_time = time.time()
        for p in gen_pair_dicts:
            t1, t2 = templates[p["probe_path"]], templates[p["gallery_path"]]
            score = match_protected_templates(
                t1["bitstrings"], t2["bitstrings"],
                query_weights=t1["weights"], enrolled_weights=t2["weights"],
                hamming_threshold=0.22, coverage_penalty_exponent=1.2
            )
            gen_scores.append(float(score))

        # Score impostor pairs
        imp_scores = []
        for p in imp_pair_dicts:
            t1, t2 = templates[p["probe_path"]], templates[p["gallery_path"]]
            score = match_protected_templates(
                t1["bitstrings"], t2["bitstrings"],
                query_weights=t1["weights"], enrolled_weights=t2["weights"],
                hamming_threshold=0.22, coverage_penalty_exponent=1.2
            )
            imp_scores.append(float(score))

        match_time = time.time() - start_time
        throughput = len(gen_scores + imp_scores) / max(match_time, 0.001)

        gen_arr = np.array(gen_scores, dtype=np.float32)
        imp_arr = np.array(imp_scores, dtype=np.float32)

        roc = compute_roc_metrics(gen_arr, imp_arr)
        conf = compute_confusion_at_threshold(gen_arr, imp_arr, roc["eer_threshold"])

        record = {
            "candidate_name": c_name,
            "dimension": dim,
            "features": cand["features"],
            "description": cand["description"],
            "eer_pct": float(roc["eer"] * 100.0),
            "auc": float(roc["auc"]),
            "eer_threshold": float(roc["eer_threshold"]),
            "tar_at_1pct_far": float(roc["tar_at_1pct_far"] * 100.0),
            "tar_at_01pct_far": float(roc["tar_at_01pct_far"] * 100.0),
            "separation_delta": float(np.mean(gen_arr) - np.mean(imp_arr)),
            "genuine_mean": float(np.mean(gen_arr)),
            "genuine_std": float(np.std(gen_arr)),
            "impostor_mean": float(np.mean(imp_arr)),
            "impostor_std": float(np.std(imp_arr)),
            "throughput_comp_per_sec": round(throughput, 1),
            "confusion": {"tp": conf["tp"], "fn": conf["fn"], "fp": conf["fp"], "tn": conf["tn"]}
        }
        results.append(record)

        print(f"  EER: {record['eer_pct']:.2f}% | AUC: {record['auc']:.4f} | Sep: {record['separation_delta']:.4f} | Throughput: {throughput:.1f} comp/s")

        # Save raw scores
        np.save(os.path.join(OUTPUT_DIR, f"candidate_{dim}d_gen_scores.npy"), gen_arr)
        np.save(os.path.join(OUTPUT_DIR, f"candidate_{dim}d_imp_scores.npy"), imp_arr)

    # Save summary report
    summary_path = os.path.join(OUTPUT_DIR, "descriptor_candidates_report.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "title": "WP1 Descriptor Candidate Benchmark (DB1_B)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "database": "FVC2002 DB1_B",
            "trials": len(gen_pair_dicts) + len(imp_pair_dicts),
            "genuine": len(gen_pair_dicts),
            "impostor": len(imp_pair_dicts),
            "candidates": results
        }, f, indent=2)

    print(f"\nReport written to: {summary_path}")
    return results


if __name__ == "__main__":
    evaluate_candidates()
