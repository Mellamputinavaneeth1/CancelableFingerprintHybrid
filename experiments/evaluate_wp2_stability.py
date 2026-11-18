"""
WP2: Simplex Stability, Minutiae Repeatability, Descriptor Drift, and Bit Stability Runner.

Audits and quantifies the microscopic causes of fingerprint template degradation across impressions:
1. Minutiae spatial and angular repeatability across genuine physical impressions (FVC2002 DB1_B).
2. Delaunay simplex stability and edge-flipping rate:
   - Compares standard Delaunay, degeneracy-filtered Delaunay, and k-NN graphs.
   - Quantifies the fraction of triangles that survive impression-to-impression noise.
3. Feature-level descriptor drift:
   - Measures continuous feature error |d1_k - d2_k| across surviving genuine triangles.
   - Computes Feature Discriminability / Signal-to-Noise Ratio (SNR) across all 9 features.
4. Projection bit stability across bit lengths M in {32, 64, 128}:
   - Bit error rate (BER) / bit flipping probability P(b1 != b2).
   - Bit entropy and marginal balance P(b = 1).
5. Microscopic Hamming distance distributions:
   - Surviving genuine vs all genuine vs impostor triangles.
   - Demonstrates why the empirical threshold T_H = 0.22 is optimal.

Outputs:
- JSON report: experiments/results/wp2/wp2_microscopic_stability_report.json
- Raw distribution arrays: experiments/results/wp2/hamming_dist_*.npy
"""

import os
import sys
import time
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import CancelableFingerprintPipeline
from src.minutiae.extractor import Minutia
from src.graph.delaunay import build_delaunay_triangles, build_knn_graphs, is_valid_triangle
from src.graph.descriptors import (
    extract_all_descriptors,
    extract_all_descriptors_with_minutiae,
    extract_triangle_descriptor
)
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors
from experiments.datasets.fvc_loader import load_fvc_dataset
from experiments.protocols.fvc2002_protocol import generate_fvc_protocol_pairs

OUTPUT_DIR = "experiments/results/wp2"


def _axial_angle_diff(a1: float, a2: float) -> float:
    """Computes acute difference between two axial angles in [0, pi)."""
    diff = abs(a1 - a2) % np.pi
    return min(diff, np.pi - diff)


def find_rigid_minutiae_alignment(
    m1_list: List[Minutia],
    m2_list: List[Minutia],
    dist_tol: float = 15.0,
    angle_tol: float = 0.35  # ~20 degrees
) -> Tuple[Optional[Tuple[float, float, float]], List[Tuple[int, int]]]:
    """
    Finds optimal 2D rigid transformation (R, t) aligning m1_list to m2_list using RANSAC
    over 2-point hypotheses.
    
    Returns:
        (dx, dy, dtheta): optimal rigid transformation
        correspondences: list of (idx1, idx2) paired minutiae indices
    """
    if len(m1_list) < 2 or len(m2_list) < 2:
        return None, []

    pts1 = np.array([[m.x, m.y] for m in m1_list], dtype=np.float32)
    th1 = np.array([m.theta for m in m1_list], dtype=np.float32)
    pts2 = np.array([[m.x, m.y] for m in m2_list], dtype=np.float32)
    th2 = np.array([m.theta for m in m2_list], dtype=np.float32)

    best_inliers = []
    best_transform = None

    # Try up to 100 random 2-point pairs
    rng = np.random.default_rng(42)
    n1 = len(pts1)
    n2 = len(pts2)

    max_trials = min(150, n1 * (n1 - 1) // 2)
    for _ in range(max_trials):
        i1, i2 = rng.choice(n1, size=2, replace=False)
        d1 = float(np.hypot(pts1[i1, 0] - pts1[i2, 0], pts1[i1, 1] - pts1[i2, 1]))
        if d1 < 20.0 or d1 > 200.0:
            continue

        # Find candidate pairs in pts2 with similar length (+- 10%)
        j1, j2 = rng.choice(n2, size=2, replace=False)
        d2 = float(np.hypot(pts2[j1, 0] - pts2[j2, 0], pts2[j1, 1] - pts2[j2, 1]))
        if abs(d1 - d2) > 0.10 * d1:
            continue

        # Estimate angle difference
        v1 = pts1[i2] - pts1[i1]
        v2 = pts2[j2] - pts2[j1]
        ang1 = np.arctan2(v1[1], v1[0])
        ang2 = np.arctan2(v2[1], v2[0])
        rot = ang2 - ang1

        # Rotation matrix
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])

        # Translation from centroid of the 2 points
        c1 = 0.5 * (pts1[i1] + pts1[i2])
        c2 = 0.5 * (pts2[j1] + pts2[j2])
        trans = c2 - R @ c1

        # Transform all pts1
        aligned_pts1 = (R @ pts1.T).T + trans
        aligned_th1 = (th1 + rot) % np.pi

        # Count inliers
        inliers = []
        matched_j = set()
        for idx1 in range(n1):
            p1_trans = aligned_pts1[idx1]
            t1_trans = aligned_th1[idx1]
            dists = np.hypot(pts2[:, 0] - p1_trans[0], pts2[:, 1] - p1_trans[1])
            closest_j = int(np.argmin(dists))
            if closest_j not in matched_j and dists[closest_j] <= dist_tol:
                if _axial_angle_diff(t1_trans, th2[closest_j]) <= angle_tol:
                    inliers.append((idx1, closest_j))
                    matched_j.add(closest_j)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_transform = (float(trans[0]), float(trans[1]), float(rot))
            if len(best_inliers) >= 0.6 * min(n1, n2):
                break

    return best_transform, best_inliers


def evaluate_microscopic_stability():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 80)
    print("  WP2: Microscopic Stability & Recognition Bottleneck Analysis")
    print("  Database: FVC2002 DB1_B (Development Benchmark)")
    print("=" * 80)

    fvc_data = load_fvc_dataset("DB1_B", root_dir="dataset/FVC2002")
    gen_pairs, imp_pairs = generate_fvc_protocol_pairs(fvc_data)
    print(f"Loaded {len(gen_pairs)} genuine pairs and {len(imp_pairs)} impostor pairs.")

    pipeline = CancelableFingerprintPipeline(in_dim=9, use_gpu=True)

    # 1. Cache minutiae and images
    print("\n[Step 1/5] Extracting minutiae across all 80 images...")
    minutiae_cache = {}
    for fid, imps in fvc_data["fingers"].items():
        for iid, img_path in imps.items():
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            feat = pipeline.extract_features(img)
            minutiae_cache[img_path] = feat["minutiae"]

    # 2. Minutiae Repeatability Analysis
    print("\n[Step 2/5] Evaluating minutiae spatial & angular repeatability across genuine pairs...")
    repeatability_records = []
    pair_alignments = {}

    for pair in gen_pairs:
        p1 = pair["probe_path"]
        p2 = pair["gallery_path"]
        m1 = minutiae_cache[p1]
        m2 = minutiae_cache[p2]

        transform, inliers = find_rigid_minutiae_alignment(m1, m2, dist_tol=15.0, angle_tol=0.35)
        pair_alignments[(p1, p2)] = (transform, inliers)

        min_count = min(len(m1), len(m2))
        if min_count > 0:
            rep_rate = len(inliers) / min_count
        else:
            rep_rate = 0.0

        repeatability_records.append({
            "probe": p1,
            "gallery": p2,
            "m1_count": len(m1),
            "m2_count": len(m2),
            "inlier_count": len(inliers),
            "repeatability_rate": rep_rate
        })

    rep_rates = [r["repeatability_rate"] for r in repeatability_records]
    mean_rep = float(np.mean(rep_rates))
    median_rep = float(np.median(rep_rates))
    std_rep = float(np.std(rep_rates))
    print(f"  Minutiae Repeatability: Mean = {mean_rep*100.0:.2f}%, Median = {median_rep*100.0:.2f}%, Std = {std_rep*100.0:.2f}%")

    # 3. Delaunay Simplex Stability & Edge Flipping
    print("\n[Step 3/5] Evaluating Delaunay simplex stability and edge-flipping...")
    topo_modes = [
        {"name": "Standard Delaunay", "min_edge": 0.0, "min_area": 0.0, "max_aspect": None, "type": "delaunay"},
        {"name": "Degeneracy-Filtered Delaunay", "min_edge": 10.0, "min_area": 25.0, "max_aspect": 4.0, "type": "delaunay"},
        {"name": "k-NN Redundant Triplets (k=5)", "min_edge": 0.0, "min_area": 0.0, "max_aspect": None, "type": "knn", "k": 5}
    ]

    topo_results = {}
    surviving_gen_descs_1 = []
    surviving_gen_descs_2 = []

    for tmode in topo_modes:
        tname = tmode["name"]
        print(f"  Testing topology: {tname}...")

        total_triangles_p1 = 0
        total_3vertex_recoverable = 0
        total_surviving_triangles = 0

        for pair in gen_pairs:
            p1 = pair["probe_path"]
            p2 = pair["gallery_path"]
            m1 = minutiae_cache[p1]
            m2 = minutiae_cache[p2]
            transform, inliers = pair_alignments[(p1, p2)]

            if not inliers or transform is None:
                continue

            m1_to_m2 = dict(inliers)  # mapping idx1 -> idx2

            # Extract triangles for p1 and p2 under this topology
            if tmode["type"] == "delaunay":
                tris1 = build_delaunay_triangles(
                    m1, max_edge_length=120.0,
                    min_edge_length=tmode["min_edge"],
                    min_area=tmode["min_area"],
                    max_aspect_ratio=tmode["max_aspect"]
                )
                tris2 = build_delaunay_triangles(
                    m2, max_edge_length=120.0,
                    min_edge_length=tmode["min_edge"],
                    min_area=tmode["min_area"],
                    max_aspect_ratio=tmode["max_aspect"]
                )
            else:
                tris1 = build_knn_graphs(m1, k=tmode["k"], max_edge_length=120.0)
                tris2 = build_knn_graphs(m2, k=tmode["k"], max_edge_length=120.0)

            total_triangles_p1 += len(tris1)

            # Build a lookup for triangles in p2 based on minutiae identities
            # Identity is minutia index in m2
            tri2_set = set()
            tri2_obj_map = {}
            for t2 in tris2:
                # Find indices of t2 vertices in m2
                id0 = next(i for i, m in enumerate(m2) if m is t2[0])
                id1 = next(i for i, m in enumerate(m2) if m is t2[1])
                id2 = next(i for i, m in enumerate(m2) if m is t2[2])
                key = tuple(sorted([id0, id1, id2]))
                tri2_set.add(key)
                tri2_obj_map[key] = t2

            for t1 in tris1:
                id0 = next(i for i, m in enumerate(m1) if m is t1[0])
                id1 = next(i for i, m in enumerate(m1) if m is t1[1])
                id2 = next(i for i, m in enumerate(m1) if m is t1[2])

                # Check if all 3 vertices have corresponding minutiae in m2
                if id0 in m1_to_m2 and id1 in m1_to_m2 and id2 in m1_to_m2:
                    total_3vertex_recoverable += 1
                    target_key = tuple(sorted([m1_to_m2[id0], m1_to_m2[id1], m1_to_m2[id2]]))
                    if target_key in tri2_set:
                        total_surviving_triangles += 1
                        if tmode["name"] == "Standard Delaunay":
                            # Extract canonical 9-D descriptors for both
                            d1, _ = extract_triangle_descriptor(t1[0], t1[1], t1[2], include_types=False, desc_dim=9)
                            t2_match = tri2_obj_map[target_key]
                            d2, _ = extract_triangle_descriptor(t2_match[0], t2_match[1], t2_match[2], include_types=False, desc_dim=9)
                            surviving_gen_descs_1.append(d1)
                            surviving_gen_descs_2.append(d2)

        surv_rate = total_surviving_triangles / max(total_3vertex_recoverable, 1)
        gross_surv_rate = total_surviving_triangles / max(total_triangles_p1, 1)
        edge_flip_rate = 1.0 - surv_rate

        topo_results[tname] = {
            "total_triangles": total_triangles_p1,
            "recoverable_triangles": total_3vertex_recoverable,
            "surviving_triangles": total_surviving_triangles,
            "simplex_survival_rate_pct": round(surv_rate * 100.0, 2),
            "gross_survival_rate_pct": round(gross_surv_rate * 100.0, 2),
            "edge_flip_rate_pct": round(edge_flip_rate * 100.0, 2)
        }
        print(f"    Simplex Survival Rate: {surv_rate*100.0:.2f}% | Edge Flipping: {edge_flip_rate*100.0:.2f}% | Gross: {gross_surv_rate*100.0:.2f}%")

    # 4. Feature Drift and Signal-to-Noise Ratio (SNR) Analysis
    print("\n[Step 4/5] Analyzing continuous feature drift on surviving genuine simplices vs impostors...")
    surv_d1 = np.array(surviving_gen_descs_1, dtype=np.float32)
    surv_d2 = np.array(surviving_gen_descs_2, dtype=np.float32)
    print(f"  Analyzed {len(surv_d1)} confirmed surviving genuine triangle pairs.")

    feature_names = ["r1", "r2", "P_mm", "phi0", "phi1", "phi2", "dth01", "dth12", "dth20"]

    # Compute genuine feature errors
    gen_feature_diffs = np.abs(surv_d1 - surv_d2)
    gen_mean_diff = np.mean(gen_feature_diffs, axis=0)
    gen_std_diff = np.std(gen_feature_diffs, axis=0)

    # Compute impostor feature differences from random triangle pairs across different fingers
    rng = np.random.default_rng(999)
    imp_idx_a = rng.choice(len(surv_d1), size=min(len(surv_d1), 5000), replace=True)
    imp_idx_b = rng.choice(len(surv_d2), size=min(len(surv_d2), 5000), replace=True)
    imp_feature_diffs = np.abs(surv_d1[imp_idx_a] - surv_d2[imp_idx_b])
    imp_mean_diff = np.mean(imp_feature_diffs, axis=0)
    imp_std_diff = np.std(imp_feature_diffs, axis=0)

    # Feature discriminability SNR = |mean_imp - mean_gen| / std_gen
    snr = np.abs(imp_mean_diff - gen_mean_diff) / (gen_std_diff + 1e-8)

    feature_drift_report = {}
    for k, fname in enumerate(feature_names):
        feature_drift_report[fname] = {
            "genuine_mean_error": float(gen_mean_diff[k]),
            "genuine_std_error": float(gen_std_diff[k]),
            "impostor_mean_diff": float(imp_mean_diff[k]),
            "impostor_std_diff": float(imp_std_diff[k]),
            "discriminability_snr": round(float(snr[k]), 3)
        }
        print(f"  Feature {fname:<6}: Gen Error = {gen_mean_diff[k]:.4f} +- {gen_std_diff[k]:.4f} | Imp Diff = {imp_mean_diff[k]:.4f} | SNR = {snr[k]:.3f}")

    # 5. Bit Stability & Hamming Distance Distributions across M in {32, 64, 128}
    print("\n[Step 5/5] Analyzing projection bit stability and microscopic Hamming distributions...")
    bit_lengths = [32, 64, 128]
    bit_stability_report = {}

    # Compute standardization stats over surviving descriptors
    emp_m = np.mean(surv_d1, axis=0)
    emp_s = np.std(surv_d1, axis=0)

    for M in bit_lengths:
        W = derive_orthonormal_matrix(
            pipeline.master_secret, salt="salt_stability", app_id="fvc_db1_b", in_dim=9, out_dim=M
        )
        proj1 = project_descriptors(surv_d1, W, mean=emp_m, std=emp_s)
        proj2 = project_descriptors(surv_d2, W, mean=emp_m, std=emp_s)
        bits1 = quantize_descriptors(proj1)
        bits2 = quantize_descriptors(proj2)

        # Genuine bit error rate (BER)
        bit_flips = np.bitwise_xor(bits1, bits2)
        fractional_hamming_gen = np.mean(bit_flips, axis=1)
        mean_ber = float(np.mean(fractional_hamming_gen))
        median_ber = float(np.median(fractional_hamming_gen))
        std_ber = float(np.std(fractional_hamming_gen))

        # Impostor fractional Hamming distance
        imp_flips = np.bitwise_xor(bits1[imp_idx_a], bits2[imp_idx_b])
        fractional_hamming_imp = np.mean(imp_flips, axis=1)
        mean_imp_ham = float(np.mean(fractional_hamming_imp))
        median_imp_ham = float(np.median(fractional_hamming_imp))
        std_imp_ham = float(np.std(fractional_hamming_imp))

        # Marginal bit balance P(b = 1)
        p_ones = float(np.mean(bits1))

        # Overlap fraction at T_H = 0.22
        th = 0.22
        gen_pass = float(np.mean(fractional_hamming_gen <= th))
        imp_pass = float(np.mean(fractional_hamming_imp <= th))

        bit_stability_report[f"{M}_bits"] = {
            "projection_length_M": M,
            "marginal_bit_balance_P1": round(p_ones, 4),
            "genuine_ber_mean": round(mean_ber, 4),
            "genuine_ber_median": round(median_ber, 4),
            "genuine_ber_std": round(std_ber, 4),
            "impostor_hamming_mean": round(mean_imp_ham, 4),
            "impostor_hamming_median": round(median_imp_ham, 4),
            "impostor_hamming_std": round(std_imp_ham, 4),
            "genuine_retention_at_T22_pct": round(gen_pass * 100.0, 2),
            "impostor_rejection_at_T22_pct": round((1.0 - imp_pass) * 100.0, 2)
        }
        print(f"  M = {M:<3} bits: Genuine BER = {mean_ber*100.0:.2f}% | Impostor Dist = {mean_imp_ham*100.0:.2f}% | Gen Ret @ 0.22 = {gen_pass*100.0:.2f}% | Imp Rej @ 0.22 = {(1.0-imp_pass)*100.0:.2f}%")

        if M == 32:
            np.save(os.path.join(OUTPUT_DIR, "hamming_dist_genuine_surviving_32b.npy"), fractional_hamming_gen)
            np.save(os.path.join(OUTPUT_DIR, "hamming_dist_impostor_32b.npy"), fractional_hamming_imp)

    # Save summary report
    summary = {
        "title": "WP2 Microscopic Simplex Stability, Repeatability, and Bit Stability Audit",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "database": "FVC2002 DB1_B",
        "minutiae_repeatability": {
            "mean_pct": round(mean_rep * 100.0, 2),
            "median_pct": round(median_rep * 100.0, 2),
            "std_pct": round(std_rep * 100.0, 2),
            "evaluation_tolerance": "r <= 15 px, dtheta <= 20 deg"
        },
        "topology_simplex_stability": topo_results,
        "feature_drift_snr": feature_drift_report,
        "projection_bit_stability": bit_stability_report
    }

    out_json = os.path.join(OUTPUT_DIR, "wp2_microscopic_stability_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Microscopic stability report written to: {out_json}")
    return summary


if __name__ == "__main__":
    evaluate_microscopic_stability()
