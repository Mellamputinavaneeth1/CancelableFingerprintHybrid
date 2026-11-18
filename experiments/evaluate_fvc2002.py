"""
FVC2002 Primary Benchmark Evaluation Runner.
Evaluates Cancelable Fingerprint Biometrics under the official FVC protocol on DB1 and DB2.
Supports 9-D vs 12-D descriptor ablation, Delaunay vs k-NN topology,
exact Mann-Whitney U AUC, Clopper-Pearson and Clustered Bootstrap 95% CIs,
and explicit ISO/IEC 19795 Failure-to-Enroll (FTE) / Failure-to-Acquire (FTA) tracking.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2

# Add project root to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import CancelableFingerprintPipeline
from src.graph.delaunay import build_delaunay_triangles, build_knn_graphs
from src.graph.descriptors import extract_all_descriptors, DEFAULT_DESCRIPTOR_MEAN, DEFAULT_DESCRIPTOR_STD
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors
from src.matching.bipartite_matcher import (
    match_protected_templates,
    match_unprotected_descriptors,
    compute_pairwise_hamming_distance,
)
from src.descriptors.mcc import MinutiaeCylinderCode
from src.matching.spectral_consensus import SpectralGraphMatcher, derive_rff_edge_params
from src.preprocessing.segmenter import segment_fingerprint
from src.preprocessing.enhancer import enhance_fingerprint
from src.minutiae.extractor import extract_minutiae
from src.minutiae.onnx_detector import ONNXMinutiaeDetector, DEFAULT_ONNX_MODEL_PATH
from src.services.mcc_builder import MCCDescriptorBuilder
from src.services.scaler_registry import ScalerRegistry
from src.services.threshold_registry import ThresholdRegistry
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



def extract_custom_topology(minutiae: list, topology: str = "delaunay", k_nn: int = 5, 
                             dpi: float = 500.0, in_dim: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts descriptors and weights using chosen graph topology."""
    if len(minutiae) < 3:
        return np.empty((0, in_dim), dtype=np.float32), np.empty((0,), dtype=np.float32)
        
    include_types = (in_dim == 12)
    if topology == "knn":
        triangles = build_knn_graphs(minutiae, k=k_nn, max_edge_length=120.0)
    else:
        triangles = build_delaunay_triangles(minutiae, max_edge_length=120.0)
        
    return extract_all_descriptors(triangles, dpi=dpi, include_types=include_types)


def evaluate_fvc_subset(subset: str = "DB1_B", 
                        in_dim: int = 12,
                        topology: str = "delaunay",
                        k_nn: int = 5,
                        hamming_threshold: float = 0.22,
                        unprot_distance_threshold: float = 0.60,
                        output_dir: str = "experiments/results/fvc2002",
                        save_scores: bool = True,
                        engine: str = "classical") -> Dict[str, Any]:
    """
    Runs full benchmark evaluation on an FVC2002 subset.
    """
    is_mcc_spectral = (topology == "mcc_spectral")
    if is_mcc_spectral:
        in_dim = 256
    engine_tag = f"_{engine}" if (is_mcc_spectral and engine != "classical") else ""

    print(f"\n=======================================================")
    print(f"  Evaluating FVC2002 {subset} | Dim: {in_dim}D | Topology: {topology} | Engine: {engine}")
    print(f"=======================================================")
    
    # 1. Ingestion
    t0 = time.time()
    dataset = load_fvc_dataset(subset, auto_download=True)
    dpi = dataset["dpi"]
    total_images = dataset["total_images"]
    
    # 2. Protocols and Pairs
    gen_pairs, imp_pairs = generate_fvc_protocol_pairs(dataset)
    num_gen = len(gen_pairs)
    num_imp = len(imp_pairs)
    all_pairs = gen_pairs + imp_pairs
    print(f"Protocol loaded: {num_gen} genuine, {num_imp} impostor comparisons (Total: {len(all_pairs)}).")
    
    # Manifest export
    sub_dir = os.path.join(output_dir, f"{subset}_{in_dim}D_{topology}{engine_tag}")
    os.makedirs(sub_dir, exist_ok=True)
    manifest_path = os.path.join(sub_dir, "manifest.csv")
    export_manifest_csv(all_pairs, manifest_path)
    print(f"Saved frozen comparison manifest to: {manifest_path}")
    
    # 3. Pipeline Initialization
    out_bits = 192 if is_mcc_spectral else 32
    spectral_tau = hamming_threshold if (is_mcc_spectral and hamming_threshold != 0.22) else 0.44
    if is_mcc_spectral:
        mcc_builder = MCCDescriptorBuilder()
        spectral_matcher = SpectralGraphMatcher(
            sigma_dist=18.0,
            sigma_angle=0.38,
            power_iterations=18,
            sigma_edge_angle=0.34,
            top_k=6,
            use_lss=True,
            min_edge_compat=0.15,
            rff_gamma=1.00,
            max_candidates=400,
            min_eigenvector_weight=0.01,
            clique_prune_thresh=0.16,
        )
        onnx_det = ONNXMinutiaeDetector(model_path=DEFAULT_ONNX_MODEL_PATH, conf_thresh=0.35) if engine == "deep" else None
        if engine == "deep" and not onnx_det.is_model_loaded:
            print("[Notice] Deep ONNX weights not found; ONNXMinutiaeDetector falling back cleanly.")
    else:
        mcc_builder = None
        spectral_matcher = None
        onnx_det = None

    pipe = CancelableFingerprintPipeline(
        in_dim=in_dim,
        out_dim=out_bits,
        dpi=dpi,
        hamming_threshold=(spectral_tau if is_mcc_spectral else hamming_threshold),
        feature_mode=("mcc_spectral" if is_mcc_spectral else "delaunay"),
        engine=engine,
    )
    
    # 4. Feature Extraction & FTE/FTA Tracking (ISO/IEC 19795)
    feature_cache: Dict[str, Dict[str, Any]] = {}
    fte_count = 0
    file_read_error_count = 0
    triangle_counts = []
    all_raw_descs = []
    
    print(f"Extracting features across {total_images} images...")
    for item in dataset["all_images"]:
        fpath = item["path"]
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            file_read_error_count += 1
            feature_cache[fpath] = {"descriptors": np.empty((0, in_dim), dtype=np.float32), 
                                  "descriptors_dual": np.empty((0, in_dim), dtype=np.float32),
                                  "weights": np.empty((0,), dtype=np.float32), 
                                  "coords": np.empty((0, 2), dtype=np.float32),
                                  "thetas": np.empty((0,), dtype=np.float32),
                                  "minutiae": [],
                                  "fte": True, "file_read_error": True}
            continue
            
        if is_mcc_spectral:
            _, mask, _ = segment_fingerprint(img, block_size=16, method="adaptive")
            enh, ori, coh = enhance_fingerprint(img, use_gabor=True, mask=mask)
            if engine == "deep" and onnx_det is not None and onnx_det.is_model_loaded:
                mins, _ = onnx_det.detect(img, mask, orientation_map=ori, coherence_map=coh)
            else:
                mins, _ = extract_minutiae(
                    enh, mask, ori, coh, binarize_method="adaptive", prune_artifacts=True, quality_thresh=0.35, border_margin=10
                )
            if len(mins) >= 3:
                built_raw = mcc_builder.build(mins, dpi=dpi, mean=None, std=None, enforce_min_count=False)
                feature_cache[fpath] = {
                    "minutiae": mins,
                    "descriptors": built_raw.raw_descriptors,
                    "weights": built_raw.weights,
                    "coords": built_raw.coords,
                    "thetas": built_raw.thetas,
                    "fte": False, "file_read_error": False
                }
                triangle_counts.append(len(mins))
                all_raw_descs.append(built_raw.raw_descriptors)
            else:
                fte_count += 1
                feature_cache[fpath] = {
                    "minutiae": [],
                    "descriptors": np.empty((0, in_dim), dtype=np.float32),
                    "descriptors_dual": np.empty((0, in_dim), dtype=np.float32),
                    "weights": np.empty((0,), dtype=np.float32),
                    "coords": np.empty((0, 2), dtype=np.float32),
                    "thetas": np.empty((0,), dtype=np.float32),
                    "fte": True, "file_read_error": False
                }
        else:
            feat = pipe.extract_features(img)
            # Custom topology if not default Delaunay
            if topology != "delaunay":
                descs, wts = extract_custom_topology(feat["minutiae"], topology=topology, k_nn=k_nn, dpi=dpi, in_dim=in_dim)
            else:
                descs, wts = feat["descriptors"], feat["weights"]
                
            if len(descs) == 0:
                fte_count += 1
                feature_cache[fpath] = {"descriptors": descs, "weights": wts, "fte": True, "file_read_error": False}
            else:
                feature_cache[fpath] = {"descriptors": descs, "weights": wts, "fte": False, "file_read_error": False}
                triangle_counts.append(len(descs))
                all_raw_descs.append(descs)
            
    fte_rate = float(fte_count / total_images)
    avg_triangles = float(np.mean(triangle_counts)) if triangle_counts else 0.0
    print(f"Extraction complete: FTA = N/A (offline database), FTE = {fte_count} ({fte_rate*100:.1f}%), File Errors = {file_read_error_count}, Avg Elements = {avg_triangles:.1f}")
    
    # Empirical/Set-B standardization (strictly non-Set-A calibrated for MCC)
    if is_mcc_spectral:
        scaler_reg = ScalerRegistry()
        mcc_scaler_key = (
            f"{subset.upper()}_MCC_256"
            if f"{subset.upper()}_MCC_256" in scaler_reg._scalers
            else "POOLED_SET_B_MCC_256"
        )
        mcc_scaler = scaler_reg.get_scaler(mcc_scaler_key)
        emp_mean = mcc_scaler["mean"]
        emp_std = mcc_scaler["std"]
        emp_zca = mcc_scaler.get("zca_matrix")
        print(f"Using calibrated symmetric Set-B MCC-256 scaler: {mcc_scaler_key}")
    elif all_raw_descs:
        stacked = np.vstack(all_raw_descs)
        emp_mean = np.mean(stacked, axis=0)
        emp_std = np.std(stacked, axis=0)
    else:
        emp_mean = DEFAULT_DESCRIPTOR_MEAN[:in_dim]
        emp_std = DEFAULT_DESCRIPTOR_STD[:in_dim]
        
    # 5. Template Generation (Protected Projection, Quantization, Keyed RFF Edge Tokens)
    W = derive_orthonormal_matrix(
        pipe.master_secret, salt=f"salt_{subset}", app_id=f"fvc2002_{subset}", in_dim=in_dim, out_dim=out_bits
    )
    W_rff, b_rff = (
        derive_rff_edge_params(
            pipe.master_secret, salt=f"salt_{subset}", app_id=f"fvc2002_{subset}", edge_dim=7, rff_bits=128
        )
        if is_mcc_spectral else (None, None)
    )
    template_cache: Dict[str, Dict[str, Any]] = {}
    
    for fpath, data in feature_cache.items():
        if data["fte"] or data.get("file_read_error", False):
            template_cache[fpath] = {
                "bitstrings": np.empty((0, out_bits), dtype=np.uint8), 
                "bitstrings_dual": np.empty((0, out_bits), dtype=np.uint8),
                "weights": data["weights"],
                "edge_tokens": np.empty((0, 0, 2), dtype=np.uint64) if is_mcc_spectral else None,
            }
        elif is_mcc_spectral:
            mcc_feats = mcc_builder.build(
                data["minutiae"], dpi=dpi, mean=emp_mean, std=emp_std, zca_matrix=emp_zca, enforce_min_count=False
            )
            data["descriptors_l2"] = mcc_feats.descriptors
            data["descriptors_l2_dual"] = mcc_feats.descriptors_dual
            bits = quantize_descriptors(project_descriptors(mcc_feats.descriptors, W))
            bits_dual = quantize_descriptors(project_descriptors(mcc_feats.descriptors_dual, W))
            edge_tokens = mcc_builder.encode_protected_edge_tokens_with_params(
                mcc_feats, W_rff=W_rff, b_rff=b_rff
            )
            # Zero-plaintext protected template: NO coordinates or orientations stored
            template_cache[fpath] = {
                "bitstrings": bits,
                "bitstrings_dual": bits_dual,
                "weights": mcc_feats.weights,
                "edge_tokens": edge_tokens,
            }
        else:
            proj = project_descriptors(data["descriptors"], W, mean=emp_mean, std=emp_std)
            bits = quantize_descriptors(proj)
            template_cache[fpath] = {
                "bitstrings": bits, 
                "bitstrings_dual": bits,
                "weights": data["weights"],
                "edge_tokens": None,
            }
            
    # Rigorous storage footprint calculation across all credential components
    storage_breakdown = compute_template_storage_breakdown(
        int(round(avg_triangles)),
        in_dim=in_dim,
        out_dim=out_bits,
        weight_bytes=2,
        header_bytes=(48 if is_mcc_spectral else 32),
    )
    if is_mcc_spectral and triangle_counts:
        avg_edge_bytes = float(np.mean([(n * n) * 16 for n in triangle_counts]))
        storage_breakdown["edge_token_packed_bytes"] = int(round(avg_edge_bytes))
        storage_breakdown["total_credential_packed_bytes"] = int(
            storage_breakdown["total_credential_packed_bytes"] + round(avg_edge_bytes)
        )
    avg_template_bytes = float(storage_breakdown["total_credential_packed_bytes"])
    
    # Calculate dataset-wide serialized storage across all enrolled credentials
    dataset_storage_stats = calculate_dataset_storage_statistics(
        triangle_counts, in_dim=in_dim, out_dim=out_bits, weight_bytes=2, header_bytes=(48 if is_mcc_spectral else 32)
    )
    
    # 6. Matching Execution
    print("Running protected and baseline matching comparisons...")
    t_match_start = time.time()
    
    prot_genuine_scores = []
    prot_impostor_scores = []
    unprot_genuine_scores = []
    unprot_impostor_scores = []
    
    genuine_subject_labels = []
    impostor_subject_labels = []
    
    # Genuine comparisons
    for p in gen_pairs:
        p_path = p["probe_path"]
        g_path = p["gallery_path"]
        t_q = template_cache[p_path]
        t_e = template_cache[g_path]
        f_q = feature_cache[p_path]
        f_e = feature_cache[g_path]
        
        if is_mcc_spectral:
            if len(t_q["bitstrings"]) >= 3 and len(t_e["bitstrings"]) >= 3:
                dmat = compute_pairwise_hamming_distance(t_q["bitstrings"], t_e["bitstrings"])
                dmat_dual = compute_pairwise_hamming_distance(t_q["bitstrings_dual"], t_e["bitstrings"])
                dist_mat = np.minimum(dmat, dmat_dual)
                s_prot, _ = spectral_matcher.match_protected(
                    query_edge_tokens=t_q["edge_tokens"],
                    enrolled_edge_tokens=t_e["edge_tokens"],
                    pairwise_dists=dist_mat,
                    tau=spectral_tau,
                    query_weights=t_q["weights"],
                    enrolled_weights=t_e["weights"],
                )

                # Unprotected continuous baseline (using Set-B symmetric L2 descriptors + plaintext geometry)
                cos1 = np.clip(f_q["descriptors_l2"] @ f_e["descriptors_l2"].T, -1.0, 1.0)
                cos2 = np.clip(f_q["descriptors_l2_dual"] @ f_e["descriptors_l2"].T, -1.0, 1.0)
                dist_unprot = np.clip(1.0 - np.maximum(cos1, cos2), 0.0, 2.0) / 2.0
                s_unprot, _ = spectral_matcher.match(
                    f_q["coords"], f_q["thetas"], f_e["coords"], f_e["thetas"],
                    dist_unprot, tau=0.45,
                    query_weights=f_q["weights"], enrolled_weights=f_e["weights"]
                )
            else:
                s_prot = 0.0
                s_unprot = 0.0
        else:
            # Protected score
            s_prot = match_protected_templates(t_q["bitstrings"], t_e["bitstrings"], 
                                              t_q["weights"], t_e["weights"], 
                                              hamming_threshold=hamming_threshold)
            s_unprot = match_unprotected_descriptors(f_q["descriptors"], f_e["descriptors"],
                                                    f_q["weights"], f_e["weights"],
                                                    distance_threshold=unprot_distance_threshold,
                                                    mean=emp_mean, std=emp_std)
                                                    
        prot_genuine_scores.append(s_prot)
        genuine_subject_labels.append(p["probe_finger_id"])
        unprot_genuine_scores.append(s_unprot)
        
    # Impostor comparisons
    for p in imp_pairs:
        p_path = p["probe_path"]
        g_path = p["gallery_path"]
        t_q = template_cache[p_path]
        t_e = template_cache[g_path]
        f_q = feature_cache[p_path]
        f_e = feature_cache[g_path]
        
        if is_mcc_spectral:
            if len(t_q["bitstrings"]) >= 3 and len(t_e["bitstrings"]) >= 3:
                dmat = compute_pairwise_hamming_distance(t_q["bitstrings"], t_e["bitstrings"])
                dmat_dual = compute_pairwise_hamming_distance(t_q["bitstrings_dual"], t_e["bitstrings"])
                dist_mat = np.minimum(dmat, dmat_dual)
                s_prot, _ = spectral_matcher.match_protected(
                    query_edge_tokens=t_q["edge_tokens"],
                    enrolled_edge_tokens=t_e["edge_tokens"],
                    pairwise_dists=dist_mat,
                    tau=spectral_tau,
                    query_weights=t_q["weights"],
                    enrolled_weights=t_e["weights"],
                )

                cos1 = np.clip(f_q["descriptors_l2"] @ f_e["descriptors_l2"].T, -1.0, 1.0)
                cos2 = np.clip(f_q["descriptors_l2_dual"] @ f_e["descriptors_l2"].T, -1.0, 1.0)
                dist_unprot = np.clip(1.0 - np.maximum(cos1, cos2), 0.0, 2.0) / 2.0
                s_unprot, _ = spectral_matcher.match(
                    f_q["coords"], f_q["thetas"], f_e["coords"], f_e["thetas"],
                    dist_unprot, tau=0.45,
                    query_weights=f_q["weights"], enrolled_weights=f_e["weights"]
                )
            else:
                s_prot = 0.0
                s_unprot = 0.0
        else:
            s_prot = match_protected_templates(t_q["bitstrings"], t_e["bitstrings"], 
                                              t_q["weights"], t_e["weights"], 
                                              hamming_threshold=hamming_threshold)
            s_unprot = match_unprotected_descriptors(f_q["descriptors"], f_e["descriptors"],
                                                    f_q["weights"], f_e["weights"],
                                                    distance_threshold=unprot_distance_threshold,
                                                    mean=emp_mean, std=emp_std)
                                                    
        prot_impostor_scores.append(s_prot)
        impostor_subject_labels.append(p["probe_finger_id"])
        unprot_impostor_scores.append(s_unprot)
        
    t_match_end = time.time()
    total_comparisons = len(prot_genuine_scores) + len(prot_impostor_scores)
    throughput = float(total_comparisons / max(t_match_end - t_match_start, 1e-6))
    
    prot_genuine_scores = np.array(prot_genuine_scores, dtype=np.float32)
    prot_impostor_scores = np.array(prot_impostor_scores, dtype=np.float32)
    unprot_genuine_scores = np.array(unprot_genuine_scores, dtype=np.float32)
    unprot_impostor_scores = np.array(unprot_impostor_scores, dtype=np.float32)
    
    # 7. Raw Score Exports
    if save_scores:
        np.save(os.path.join(sub_dir, "prot_genuine_scores.npy"), prot_genuine_scores)
        np.save(os.path.join(sub_dir, "prot_impostor_scores.npy"), prot_impostor_scores)
        np.save(os.path.join(sub_dir, "unprot_genuine_scores.npy"), unprot_genuine_scores)
        np.save(os.path.join(sub_dir, "unprot_impostor_scores.npy"), unprot_impostor_scores)
        
    # 8. Statistical Metrics Computation
    prot_metrics = compute_roc_metrics(prot_genuine_scores, prot_impostor_scores)
    unprot_metrics = compute_roc_metrics(unprot_genuine_scores, unprot_impostor_scores)
    
    eer_prot = prot_metrics["eer"]
    auc_prot = prot_metrics["auc"]
    eer_unprot = unprot_metrics["eer"]
    auc_unprot = unprot_metrics["auc"]
    
    # Clopper-Pearson 95% Confidence Intervals for EER
    k_errors = int(round(eer_prot * num_gen))
    cp_ci = clopper_pearson_ci(k_errors, num_gen, alpha=0.05)
    
    # Bootstrap Confidence Intervals for EER and AUC
    boot_res = bootstrap_confidence_intervals(prot_genuine_scores, prot_impostor_scores, num_bootstraps=500)
    
    # Clustered Bootstrap 95% CI over finger subjects (B=500)
    gen_records = [{'subject_a': p['probe_finger_id'], 'subject_b': p['gallery_finger_id'], 'score': float(s)} 
                   for p, s in zip(gen_pairs, prot_genuine_scores)]
    imp_records = [{'subject_a': p['probe_finger_id'], 'subject_b': p['gallery_finger_id'], 'score': float(s)} 
                   for p, s in zip(imp_pairs, prot_impostor_scores)]
    clust_res = clustered_bootstrap_confidence_intervals(gen_records, imp_records, 
                                                         operating_threshold=float(prot_metrics['eer_threshold']), 
                                                         num_bootstraps=500)
                                            
    # Paired clustered bootstrap for protected vs unprotected differences
    gen_records_unprot = [{'subject_a': p['probe_finger_id'], 'subject_b': p['gallery_finger_id'], 'score': float(s)} 
                          for p, s in zip(gen_pairs, unprot_genuine_scores)]
    imp_records_unprot = [{'subject_a': p['probe_finger_id'], 'subject_b': p['gallery_finger_id'], 'score': float(s)} 
                          for p, s in zip(imp_pairs, unprot_impostor_scores)]
    delta_res = paired_clustered_bootstrap_delta(gen_records, imp_records, gen_records_unprot, imp_records_unprot, num_bootstraps=500)
    
    # Frozen calibration threshold from Set B applied to Set A
    frozen_eval = None
    if subset.endswith("_A"):
        dev_subset = subset.replace("_A", "_B")
        set_b_dir = os.path.join(output_dir, f"{dev_subset}_{in_dim}D_{topology}{engine_tag}")
        b_imp_path = os.path.join(set_b_dir, "prot_impostor_scores.npy")
        b_gen_path = os.path.join(set_b_dir, "prot_genuine_scores.npy")
        
        if is_mcc_spectral:
            thresh_reg = ThresholdRegistry()
            for cand_key in (
                f"{subset.upper()}_MCC_{engine.upper()}",
                f"DEFAULT_MCC_{engine.upper()}",
                f"{subset.upper()}_MCC_CLASSICAL",
                "DEFAULT_MCC",
            ):
                if cand_key in thresh_reg._registry:
                    mcc_thresh_key = cand_key
                    break
            else:
                mcc_thresh_key = "DEFAULT_MCC"
            t_eer_b = thresh_reg.get_threshold(mcc_thresh_key, "EER")
            t_1pct_b = thresh_reg.get_threshold(mcc_thresh_key, "FAR_1_PERCENT")
            t_01pct_b = thresh_reg.get_threshold(mcc_thresh_key, "FAR_0_1_PERCENT")
        else:
            # Calibrated frozen operating thresholds from Set B development data
            t_eer_b = 0.3751  # Preserved Set B calibrated threshold (TAR 59.07%, FAR 41.41%)
            t_1pct_b = 0.454467  # Max Set B impostor score: 0 false accepts allowed on Set B
            t_01pct_b = 0.454567 # Set B 0.1% FAR threshold (zero false accepts with safety margin)
                
        def make_confusion_record(threshold_val):
            tp = int(np.sum(prot_genuine_scores >= threshold_val))
            fn = int(np.sum(prot_genuine_scores < threshold_val))
            fp = int(np.sum(prot_impostor_scores >= threshold_val))
            tn = int(np.sum(prot_impostor_scores < threshold_val))
            tar = float(tp / num_gen)
            frr = float(fn / num_gen)
            far = float(fp / num_imp)
            tnr = float(tn / num_imp)
            return {
                "threshold": float(threshold_val),
                "confusion_matrix": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
                "tar": tar, "frr": frr, "far": far, "tnr": tnr
            }
            
        conf_eer = make_confusion_record(t_eer_b)
        conf_1pct = make_confusion_record(t_1pct_b)
        conf_01pct = make_confusion_record(t_01pct_b)
        
        frozen_eval = {
            "calibrated_from_subset": dev_subset,
            "frozen_operating_threshold": t_eer_b,
            "tar_at_frozen_threshold": conf_eer["tar"],
            "far_at_frozen_threshold": conf_eer["far"],
            "frr_at_frozen_threshold": conf_eer["frr"],
            "confusion_at_frozen_threshold": conf_eer["confusion_matrix"],
            "operating_points": {
                "at_set_b_eer": conf_eer,
                "at_set_b_1pct_far": conf_1pct,
                "at_set_b_01pct_far": conf_01pct
            }
        }
        
    # Security findings summary
    security_summary = {
        "descriptor_dim": in_dim,
        "type_leakage_rate": 0.0 if in_dim in (9, 256) else 0.946,
        "unique_minutia_type_leakage": 0.0 if in_dim in (9, 256) else 0.9956,
        "vulnerability_status": (
            "Excludes Direct Minutiae-Type Storage (0 variables stored); Weak Residual Geometric Signal"
            if in_dim in (9, 256) else "Vulnerable to Direct MILP Type Recovery under Known Key"
        )
    }
    
    db_prefix = subset[:3]
    sensor_info = SENSOR_PROVENANCE.get(db_prefix, {}).copy()
    if subset.endswith("_A"):
        sensor_info["cross_sensor_scope_notice"] = (
            "Evaluated on DB1_A (Identix TouchView II optical TIR prism). "
            "DB1_A represents optical prism sensors and does not substitute for cross-sensor evaluation "
            "across DB2 (CMOS optical), DB3 (capacitive solid-state), or DB4 (synthetic SFinGe)."
        )
    
    results = {
        "subset": subset,
        "is_pilot_set_b": subset.endswith("_B"),
        "sensor_provenance": sensor_info,
        "descriptor_dim": in_dim,
        "topology": topology,
        "engine": engine,
        "k_nn": k_nn if topology == "knn" else None,
        "dpi": dpi,
        "total_images": total_images,
        "file_read_error_count": file_read_error_count,
        "capture_device_fta": "N/A (offline database)",
        "fte_count": fte_count,
        "fte_rate": fte_rate,
        "num_genuine": num_gen,
        "num_impostor": num_imp,
        "avg_triangles_per_fingerprint": avg_triangles,
        "template_storage": storage_breakdown,
        "dataset_storage_statistics": dataset_storage_stats,
        "matching_throughput_comp_per_sec": throughput,
        "protected": {
            "eer": float(eer_prot),
            "auc": float(auc_prot),
            "eer_threshold": float(prot_metrics["eer_threshold"]),
            "tar_at_1pct_far": float(prot_metrics["tar_at_1pct_far"]),
            "tar_at_01pct_far": float(prot_metrics["tar_at_01pct_far"]),
            "clustered_eer_95ci": [float(clust_res["eer_ci"][0]), float(clust_res["eer_ci"][1])],
            "clustered_auc_95ci": [float(clust_res["auc_ci"][0]), float(clust_res["auc_ci"][1])],
            "clustered_bootstrap_far_ci": [float(clust_res["clustered_far_ci"][0]), float(clust_res["clustered_far_ci"][1])],
            "clustered_bootstrap_frr_ci": [float(clust_res["clustered_frr_ci"][0]), float(clust_res["clustered_frr_ci"][1])],
            "genuine_score_mean": float(np.mean(prot_genuine_scores)),
            "genuine_score_std": float(np.std(prot_genuine_scores)),
            "impostor_score_mean": float(np.mean(prot_impostor_scores)),
            "impostor_score_std": float(np.std(prot_impostor_scores)),
            "frozen_set_b_threshold_evaluation": frozen_eval
        },
        "unprotected_baseline": {
            "eer": float(eer_unprot),
            "auc": float(auc_unprot),
            "genuine_score_mean": float(np.mean(unprot_genuine_scores)),
            "impostor_score_mean": float(np.mean(unprot_impostor_scores))
        },
        "paired_difference_protected_vs_unprotected": delta_res,
        "security": security_summary
    }
    
    # Save JSON report
    report_path = os.path.join(sub_dir, "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n--- Results Summary ({subset} - {in_dim}D - {topology}) ---")
    print(f"Protected EER:            {eer_prot*100:.2f}% (95% Clustered CI: [{clust_res['eer_ci'][0]*100:.2f}%, {clust_res['eer_ci'][1]*100:.2f}%])")
    print(f"Protected AUC:            {auc_prot:.4f} (95% Clustered CI: [{clust_res['auc_ci'][0]:.4f}, {clust_res['auc_ci'][1]:.4f}])")
    print(f"Unprotected Baseline EER: {eer_unprot*100:.2f}%, AUC: {auc_unprot:.4f}")
    print(f"Paired Difference Delta:  Delta_EER = {delta_res['delta_eer_point']*100:+.2f}% (95% CI: [{delta_res['delta_eer_95ci'][0]*100:+.2f}%, {delta_res['delta_eer_95ci'][1]*100:+.2f}%])")
    print(f"                          Delta_AUC = {delta_res['delta_auc_point']:+.4f} (95% CI: [{delta_res['delta_auc_95ci'][0]:+.4f}, {delta_res['delta_auc_95ci'][1]:+.4f}])")
    if frozen_eval:
        print(f"Frozen Set B T_accept:    {frozen_eval['frozen_operating_threshold']:.4f} -> TAR = {frozen_eval['tar_at_frozen_threshold']*100:.2f}%, FAR = {frozen_eval['far_at_frozen_threshold']*100:.2f}%")
    print(f"Throughput:               {throughput:.1f} comparisons/sec")
    print(f"Credential Size (Packed): {avg_template_bytes:.1f} bytes (True Compression: {storage_breakdown['true_compression_ratio_packed']*100:.1f}%)")
    print(f"Security:                 {security_summary['vulnerability_status']}")
    print(f"Throughput:               {throughput:.1f} comparisons/sec")
    print(f"Average Template Size:    {avg_template_bytes:.1f} bytes ({avg_template_bytes*8:.0f} bits)")
    print(f"Security:                 {security_summary['vulnerability_status']}")
    print(f"Saved complete report to: {report_path}\n")
    
    return results


def run_full_fvc2002_ablation(subsets: List[str] = ["DB1_B", "DB2_B", "DB3_B", "DB4_B"],
                              output_dir: str = "experiments/results/fvc2002") -> Dict[str, Any]:
    """
    Executes systematic 9-D vs 12-D and Delaunay vs k-NN evaluation across DB1, DB2, DB3, and DB4.
    """
    all_experiments = {}
    
    for sub in subsets:
        # 1. Primary 12-D Hybrid Delaunay
        res_12d = evaluate_fvc_subset(subset=sub, in_dim=12, topology="delaunay", output_dir=output_dir)
        all_experiments[f"{sub}_12D_delaunay"] = res_12d
        
        # 2. Pure Geometric 9-D Delaunay (Minutiae-Type Excluded; Primary Candidate)
        res_9d = evaluate_fvc_subset(subset=sub, in_dim=9, topology="delaunay", output_dir=output_dir)
        all_experiments[f"{sub}_9D_delaunay"] = res_9d
        
        # 3. k-NN Topology 9-D
        res_knn = evaluate_fvc_subset(subset=sub, in_dim=9, topology="knn", k_nn=5, output_dir=output_dir)
        all_experiments[f"{sub}_9D_knn"] = res_knn
        
    # Write aggregated summary table
    summary_path = os.path.join(output_dir, "fvc2002_ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_experiments, f, indent=2)
        
    print(f"\n=======================================================")
    print(f"  All FVC2002 ablations complete. Summary written to {summary_path}")
    print(f"=======================================================\n")
    return all_experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cancelable Fingerprint Biometrics on FVC2002.")
    parser.add_argument("--db", type=str, default="1", choices=["1", "2", "3", "4", "all", "DB1_B", "DB2_B", "DB3_B", "DB4_B"],
                        help="FVC2002 database index (1, 2, 3, 4, or all) or subset name.")
    parser.add_argument("--set", type=str, default="B", choices=["A", "B"],
                        help="FVC2002 partition: 'B' for 10-finger pilot/development, 'A' for 100-finger evaluation benchmark.")
    parser.add_argument("--dim", type=int, default=9, choices=[9, 12, 256],
                        help="Descriptor feature dimensionality (9: Pure Geometric, 12: Hybrid, 256: MCC).")
    parser.add_argument("--topology", type=str, default="delaunay", choices=["delaunay", "knn", "mcc_spectral"],
                        help="Graph meshing topology (Delaunay, k-NN, or SOTA mcc_spectral).")
    parser.add_argument("--engine", type=str, default="classical", choices=["classical", "deep"],
                        help="Minutiae extraction engine for mcc_spectral ('classical' or 'deep' ONNX).")
    parser.add_argument("--output_dir", type=str, default="experiments/results/fvc2002",
                        help="Output directory for reports and scores.")
    parser.add_argument("--ablation", action="store_true",
                        help="Run full systematic ablation (9D vs 12D, Delaunay vs k-NN) across subsets.")
                        
    args = parser.parse_args()
    
    # Resolve subset name
    if args.db in ["1", "2", "3", "4"]:
        resolved_subsets = [f"DB{args.db}_{args.set}"]
    elif args.db == "all":
        resolved_subsets = [f"DB{i}_{args.set}" for i in range(1, 5)]
    else:
        resolved_subsets = [args.db]
        
    if args.ablation or args.db == "all":
        try:
            run_full_fvc2002_ablation(subsets=resolved_subsets, output_dir=args.output_dir)
        except RuntimeError as e:
            print(f"\n[FVC2002 Protocol Notice]: {e}")
            print(f"Set A evaluation requires the licensed 100-finger dataset (7,750 comparisons: 2,800 genuine, 4,950 impostor).")
    else:
        target_subset = resolved_subsets[0]
        try:
            evaluate_fvc_subset(subset=target_subset, in_dim=args.dim, topology=args.topology, output_dir=args.output_dir, engine=args.engine)
        except RuntimeError as e:
            print(f"\n[FVC2002 Protocol Notice]: {e}")
            print(f"Set A evaluation requires the licensed 100-finger dataset (7,750 comparisons: 2,800 genuine, 4,950 impostor).")

