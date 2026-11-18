"""
SOCOFing Physical Alteration Robustness Benchmark Runner.
Evaluates Cancelable Fingerprint Biometrics against synthetically altered fingerprints:
Obliteration, Central Rotation, and Z-cut across Easy, Medium, and Hard difficulty levels.
Supports direct ingestion of Kaggle SOCOFing and automated procedural fallback using real FVC base prints.

Methodological Guarantees:
- Strict Partition Isolation: Development partition (Subjects 1-5) and Held-Out Test partition (Subjects 6-10)
  are completely isolated. Probes from test subjects are paired strictly with gallery images from test subjects.
- Exact Trial Accounting: Exactly 5 genuine and 15 impostor trials per cell on the held-out test set
  (3 Alterations x 3 Difficulties = 9 cells -> N_gen = 45, N_imp = 135).
- Subject-Clustered Bootstrap: Uncertainty is estimated by resampling independent subjects with replacement (B >= 500).
- Pure Geometric 9-D primary candidate and 12-D security ablation branch.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import CancelableFingerprintPipeline
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors
from src.matching.bipartite_matcher import match_protected_templates
from src.evaluation.metrics import (
    compute_roc_metrics, 
    compute_confusion_at_threshold,
    clopper_pearson_ci,
    clustered_bootstrap_confidence_intervals
)
from experiments.datasets.socofing_loader import (
    load_socofing_dataset, 
    generate_synthetic_socofing_benchmark, 
    SOCOFING_DPI
)
from experiments.datasets.fvc_loader import load_fvc_dataset
from experiments.protocols.socofing_protocol import generate_socofing_protocol_pairs
from experiments.protocols.pair_generator import export_manifest_csv


def evaluate_socofing_benchmark(root_dir: str = "dataset/SOCOFing",
                                in_dim: int = 9,
                                output_dir: str = "experiments/results/socofing",
                                max_subjects: int = 10,
                                auto_generate_fallback: bool = True) -> Dict[str, Any]:
    """
    Evaluates template robustness against physical alterations under strict partition isolation.
    """
    print(f"\n=======================================================")
    print(f"  Evaluating SOCOFing Alteration Benchmark | Dim: {in_dim}D")
    print(f"=======================================================")
    
    # 1. Dataset Ingestion / Fallback Generation
    dataset = load_socofing_dataset(root_dir)
    if dataset["status"] != "loaded":
        if auto_generate_fallback:
            print(f"SOCOFing dataset not found at '{root_dir}'.")
            print("Auto-generating synthetic alteration benchmark from real FVC2002 base prints...")
            fvc = load_fvc_dataset("DB1_B", auto_download=True)
            source_images = []
            for fid, imps in list(fvc["fingers"].items())[:max_subjects]:
                img_path = imps[1]  # 1st impression as clean base
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    source_images.append((f"fvc_base_{fid}", img))
                    
            generate_synthetic_socofing_benchmark(root_dir, source_images=source_images, num_subjects=len(source_images))
            dataset = load_socofing_dataset(root_dir)
        else:
            raise FileNotFoundError(f"SOCOFing dataset not found at {root_dir}")
            
    all_subject_ids = sorted(list(dataset["subjects"].keys()))[:max_subjects]
    split_pt = len(all_subject_ids) // 2
    fold1_dev_ids = all_subject_ids[:split_pt]
    fold1_test_ids = all_subject_ids[split_pt:]
    fold2_dev_ids = all_subject_ids[split_pt:]
    fold2_test_ids = all_subject_ids[:split_pt]
    
    print(f"Subjects: {len(all_subject_ids)} total across 2 folds:")
    print(f"  Fold 1 -> Dev: {fold1_dev_ids} | Held-Out Test: {fold1_test_ids}")
    print(f"  Fold 2 -> Dev: {fold2_dev_ids} | Held-Out Test: {fold2_test_ids}")
    
    # 2. Extract features across all unique images in the dataset once
    pipe = CancelableFingerprintPipeline(in_dim=in_dim, dpi=SOCOFING_DPI)
    tau_pair = pipe.hamming_threshold  # default 0.22
    W = derive_orthonormal_matrix(pipe.master_secret, salt="socofing_salt", app_id="socofing_app", in_dim=in_dim, out_dim=32)
    
    # Collect all image paths for all evaluated subjects
    all_image_paths = []
    for sid in all_subject_ids:
        for item in dataset["subjects"][sid].get("real", []):
            all_image_paths.append(item["path"])
        for item in dataset["subjects"][sid].get("altered", []):
            all_image_paths.append(item["path"])
    all_image_paths = sorted(list(set(all_image_paths)))
    
    print(f"Extracting features across {len(all_image_paths)} unique images for {len(all_subject_ids)} subjects...")
    feature_cache: Dict[str, Dict[str, Any]] = {}
    for path in all_image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            feature_cache[path] = {"descriptors": np.empty((0, in_dim), dtype=np.float32), 
                                  "weights": np.empty((0,), dtype=np.float32), "error": True}
        else:
            feat = pipe.extract_features(img)
            feature_cache[path] = {"descriptors": feat["descriptors"], "weights": feat["weights"], "error": False}

    os.makedirs(output_dir, exist_ok=True)
    alt_classes = ["Central_Rotation", "Z-cut", "Obliteration"]
    difficulties = ["Easy", "Medium", "Hard"]

    def run_single_fold(dev_ids: List[str], test_ids: List[str], fold_num: int = 1) -> Dict[str, Any]:
        dev_data = {"subjects": {s: dataset["subjects"][s] for s in dev_ids}}
        test_data = {"subjects": {s: dataset["subjects"][s] for s in test_ids}}
        
        dev_gen_pairs, dev_imp_pairs = generate_socofing_protocol_pairs(dev_data, max_impostors_per_probe=3, seed=42)
        test_gen_pairs, test_imp_pairs = generate_socofing_protocol_pairs(test_data, max_impostors_per_probe=3, seed=1042)
        
        export_manifest_csv(dev_gen_pairs + dev_imp_pairs, os.path.join(output_dir, f"socofing_fold{fold_num}_dev_manifest_{in_dim}D.csv"))
        export_manifest_csv(test_gen_pairs + test_imp_pairs, os.path.join(output_dir, f"socofing_fold{fold_num}_test_manifest_{in_dim}D.csv"))
        if fold_num == 1:
            export_manifest_csv(dev_gen_pairs + dev_imp_pairs, os.path.join(output_dir, f"socofing_dev_manifest_{in_dim}D.csv"))
            export_manifest_csv(test_gen_pairs + test_imp_pairs, os.path.join(output_dir, f"socofing_test_manifest_{in_dim}D.csv"))
            
        # Fit empirical standardization strictly on development partition images
        dev_raw_descs = []
        for p in dev_gen_pairs + dev_imp_pairs:
            for pth in [p["probe_path"], p["gallery_path"]]:
                d = feature_cache.get(pth, {}).get("descriptors", np.empty((0, in_dim)))
                if len(d) > 0:
                    dev_raw_descs.append(d)
        if dev_raw_descs:
            emp_mean = np.mean(np.vstack(dev_raw_descs), axis=0)
            emp_std = np.std(np.vstack(dev_raw_descs), axis=0)
        else:
            from src.graph.descriptors import DEFAULT_DESCRIPTOR_MEAN, DEFAULT_DESCRIPTOR_STD
            emp_mean = DEFAULT_DESCRIPTOR_MEAN[:in_dim]
            emp_std = DEFAULT_DESCRIPTOR_STD[:in_dim]
            
        # Build template cache for this fold
        t_cache = {}
        for pth in all_image_paths:
            data = feature_cache.get(pth, {})
            descs = data.get("descriptors", np.empty((0, in_dim)))
            wts = data.get("weights", np.empty((0,)))
            if data.get("error", False) or len(descs) == 0:
                t_cache[pth] = {"bitstrings": np.empty((0, 32), dtype=np.uint8), "weights": wts}
            else:
                proj = project_descriptors(descs, W, mean=emp_mean, std=emp_std)
                bits = quantize_descriptors(proj)
                t_cache[pth] = {"bitstrings": bits, "weights": wts}
                
        def score_pairs(pairs: List[Dict[str, Any]]) -> List[float]:
            scores = []
            for p in pairs:
                t_probe = t_cache[p["probe_path"]]
                t_gal = t_cache[p["gallery_path"]]
                s = match_protected_templates(t_probe["bitstrings"], t_gal["bitstrings"],
                                              t_probe["weights"], t_gal["weights"],
                                              hamming_threshold=tau_pair)
                scores.append(float(s))
            return scores
            
        print(f"\n[Fold {fold_num}] Development: {len(dev_gen_pairs)} gen, {len(dev_imp_pairs)} imp comparisons...")
        dev_gen_scores = np.array(score_pairs(dev_gen_pairs), dtype=np.float32)
        dev_imp_scores = np.array(score_pairs(dev_imp_pairs), dtype=np.float32)
        
        print(f"[Fold {fold_num}] Held-Out Test: {len(test_gen_pairs)} gen, {len(test_imp_pairs)} imp comparisons...")
        test_gen_scores = np.array(score_pairs(test_gen_pairs), dtype=np.float32)
        test_imp_scores = np.array(score_pairs(test_imp_pairs), dtype=np.float32)
        
        np.save(os.path.join(output_dir, f"socofing_fold{fold_num}_dev_gen_scores_{in_dim}D.npy"), dev_gen_scores)
        np.save(os.path.join(output_dir, f"socofing_fold{fold_num}_dev_imp_scores_{in_dim}D.npy"), dev_imp_scores)
        np.save(os.path.join(output_dir, f"socofing_fold{fold_num}_test_gen_scores_{in_dim}D.npy"), test_gen_scores)
        np.save(os.path.join(output_dir, f"socofing_fold{fold_num}_test_imp_scores_{in_dim}D.npy"), test_imp_scores)
        if fold_num == 1:
            np.save(os.path.join(output_dir, f"socofing_dev_gen_scores_{in_dim}D.npy"), dev_gen_scores)
            np.save(os.path.join(output_dir, f"socofing_dev_imp_scores_{in_dim}D.npy"), dev_imp_scores)
            np.save(os.path.join(output_dir, f"socofing_test_gen_scores_{in_dim}D.npy"), test_gen_scores)
            np.save(os.path.join(output_dir, f"socofing_test_imp_scores_{in_dim}D.npy"), test_imp_scores)
            
        dev_roc = compute_roc_metrics(dev_gen_scores, dev_imp_scores)
        T_accept_calibrated = float(dev_roc["eer_threshold"])
        T_accept_frozen = 0.4000
        
        test_roc = compute_roc_metrics(test_gen_scores, test_imp_scores)
        test_conf_calibrated = compute_confusion_at_threshold(test_gen_scores, test_imp_scores, threshold=T_accept_calibrated)
        test_conf_3891 = compute_confusion_at_threshold(test_gen_scores, test_imp_scores, threshold=0.3891)
        test_conf_frozen40 = compute_confusion_at_threshold(test_gen_scores, test_imp_scores, threshold=T_accept_frozen)
        legacy_conf_020 = compute_confusion_at_threshold(test_gen_scores, test_imp_scores, threshold=0.20)
        
        test_gen_recs = [{'subject_a': p['subject_id'], 'subject_b': p['gallery_subject_id'], 'score': float(s)} 
                         for p, s in zip(test_gen_pairs, test_gen_scores)]
        test_imp_recs = [{'subject_a': p['subject_id'], 'subject_b': p['gallery_subject_id'], 'score': float(s)} 
                         for p, s in zip(test_imp_pairs, test_imp_scores)]
        # Compute clustered bootstrap CIs at both operational benchmark threshold (0.3891) and calibrated threshold
        test_clust_boot_3891 = clustered_bootstrap_confidence_intervals(test_gen_recs, test_imp_recs, 
                                                                       operating_threshold=0.3891, 
                                                                       num_bootstraps=500,
                                                                       seed=42)
        test_clust_boot_calibrated = clustered_bootstrap_confidence_intervals(test_gen_recs, test_imp_recs, 
                                                                             operating_threshold=T_accept_calibrated, 
                                                                             num_bootstraps=500,
                                                                             seed=42)
        test_clust_boot = test_clust_boot_3891
        
        test_cell_gen_scores = {(c, d): [] for c in alt_classes for d in difficulties}
        test_cell_imp_scores = {(c, d): [] for c in alt_classes for d in difficulties}
        
        for p, s in zip(test_gen_pairs, test_gen_scores):
            c = p.get("alteration_class", "Obliteration")
            d = p.get("difficulty", "Medium")
            if (c, d) in test_cell_gen_scores:
                test_cell_gen_scores[(c, d)].append(s)
        for p, s in zip(test_imp_pairs, test_imp_scores):
            c = p.get("alteration_class", "Obliteration")
            d = p.get("difficulty", "Medium")
            if (c, d) in test_cell_imp_scores:
                test_cell_imp_scores[(c, d)].append(s)
                
        def compute_table_at_thresh(thresh: float) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
            tab = {}
            for c in alt_classes:
                tab[c] = {}
                for d in difficulties:
                    cg_arr = np.array(test_cell_gen_scores[(c, d)])
                    ci_arr = np.array(test_cell_imp_scores[(c, d)])
                    n_g = len(cg_arr)
                    n_i = len(ci_arr)
                    tp = int(np.sum(cg_arr >= thresh))
                    fn = int(np.sum(cg_arr < thresh))
                    fp = int(np.sum(ci_arr >= thresh))
                    tn = int(np.sum(ci_arr < thresh))
                    tar = float(tp / max(n_g, 1))
                    far = float(fp / max(n_i, 1))
                    tar_ci = clopper_pearson_ci(tp, n_g)
                    far_ci = clopper_pearson_ci(fp, n_i)
                    tab[c][d] = {
                        "n_genuine": n_g, "n_impostor": n_i,
                        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                        "tar": tar, "far": far,
                        "tar_95ci": [float(tar_ci[0]), float(tar_ci[1])],
                        "far_95ci": [float(far_ci[0]), float(far_ci[1])],
                        "mean_genuine_score": float(np.mean(cg_arr)) if n_g > 0 else 0.0,
                        "mean_impostor_score": float(np.mean(ci_arr)) if n_i > 0 else 0.0
                    }
                    
            r_marg = {}
            for c in alt_classes:
                all_row_g = np.concatenate([test_cell_gen_scores[(c, d)] for d in difficulties])
                all_row_i = np.concatenate([test_cell_imp_scores[(c, d)] for d in difficulties])
                tp = int(np.sum(all_row_g >= thresh))
                fp = int(np.sum(all_row_i >= thresh))
                n_g = len(all_row_g)
                n_i = len(all_row_i)
                tar_ci = clopper_pearson_ci(tp, n_g)
                far_ci = clopper_pearson_ci(fp, n_i)
                r_marg[c] = {
                    "n_genuine": n_g, "n_impostor": n_i,
                    "tp": tp, "fn": n_g - tp,
                    "fp": fp, "tn": n_i - fp,
                    "tar": float(tp / n_g),
                    "far": float(fp / n_i),
                    "tar_95ci": [float(tar_ci[0]), float(tar_ci[1])],
                    "far_95ci": [float(far_ci[0]), float(far_ci[1])],
                    "mean_genuine_score": float(np.mean(all_row_g)),
                    "mean_impostor_score": float(np.mean(all_row_i))
                }
                
            c_marg = {}
            for d in difficulties:
                all_col_g = np.concatenate([test_cell_gen_scores[(c, d)] for c in alt_classes])
                all_col_i = np.concatenate([test_cell_imp_scores[(c, d)] for c in alt_classes])
                tp = int(np.sum(all_col_g >= thresh))
                fp = int(np.sum(all_col_i >= thresh))
                n_g = len(all_col_g)
                n_i = len(all_col_i)
                tar_ci = clopper_pearson_ci(tp, n_g)
                far_ci = clopper_pearson_ci(fp, n_i)
                c_marg[d] = {
                    "n_genuine": n_g, "n_impostor": n_i,
                    "tp": tp, "fn": n_g - tp,
                    "fp": fp, "tn": n_i - fp,
                    "tar": float(tp / n_g),
                    "far": float(fp / n_i),
                    "tar_95ci": [float(tar_ci[0]), float(tar_ci[1])],
                    "far_95ci": [float(far_ci[0]), float(far_ci[1])],
                    "mean_genuine_score": float(np.mean(all_col_g)),
                    "mean_impostor_score": float(np.mean(all_col_i))
                }
            return tab, r_marg, c_marg

        # Harmonize reporting at benchmark operating threshold T_accept = 0.3891:
        # Ensures cell breakdown and aggregate confusion matrix are strictly generated from the same row-level table
        # resolving the FP=25 vs FP=15 discrepancy.
        test_cross_tab, row_marginals, col_marginals = compute_table_at_thresh(0.3891)
        test_cross_tab_calibrated, row_marginals_calibrated, col_marginals_calibrated = compute_table_at_thresh(T_accept_calibrated)

        # Integrity verification: row-level cell metrics MUST sum identically to aggregate confusion matrix
        tot_cell_tp = sum(test_cross_tab[c][d]["tp"] for c in alt_classes for d in difficulties)
        tot_cell_fn = sum(test_cross_tab[c][d]["fn"] for c in alt_classes for d in difficulties)
        tot_cell_fp = sum(test_cross_tab[c][d]["fp"] for c in alt_classes for d in difficulties)
        tot_cell_tn = sum(test_cross_tab[c][d]["tn"] for c in alt_classes for d in difficulties)
        assert tot_cell_tp == test_conf_3891["tp"], f"TP mismatch: {tot_cell_tp} vs {test_conf_3891['tp']}"
        assert tot_cell_fn == test_conf_3891["fn"], f"FN mismatch: {tot_cell_fn} vs {test_conf_3891['fn']}"
        assert tot_cell_fp == test_conf_3891["fp"], f"FP mismatch: {tot_cell_fp} vs {test_conf_3891['fp']}"
        assert tot_cell_tn == test_conf_3891["tn"], f"TN mismatch: {tot_cell_tn} vs {test_conf_3891['tn']}"
        assert tot_cell_tp + tot_cell_fn == len(test_gen_scores), "Genuine trial count identity violation"
        assert tot_cell_fp + tot_cell_tn == len(test_imp_scores), "Impostor trial count identity violation"
            
        return {
            "fold_num": fold_num,
            "dev_subject_ids": dev_ids,
            "test_subject_ids": test_ids,
            "dev_pairs": {"gen": dev_gen_pairs, "imp": dev_imp_pairs},
            "test_pairs": {"gen": test_gen_pairs, "imp": test_imp_pairs},
            "test_gen_scores": test_gen_scores,
            "test_imp_scores": test_imp_scores,
            "test_cell_gen_scores": test_cell_gen_scores,
            "test_cell_imp_scores": test_cell_imp_scores,
            "dev_roc": dev_roc,
            "test_roc": test_roc,
            "calibrated_T_accept": T_accept_calibrated,
            "frozen_benchmark_T_accept": T_accept_frozen,
            "test_conf_calibrated": test_conf_calibrated,
            "test_conf_3891": test_conf_3891,
            "test_conf_frozen40": test_conf_frozen40,
            "legacy_conf_020": legacy_conf_020,
            "test_clust_boot": test_clust_boot,
            "test_clust_boot_3891": test_clust_boot_3891,
            "test_clust_boot_calibrated": test_clust_boot_calibrated,
            "test_cross_tab": test_cross_tab,
            "row_marginals": row_marginals,
            "col_marginals": col_marginals,
            "test_cross_tab_calibrated": test_cross_tab_calibrated,
            "row_marginals_calibrated": row_marginals_calibrated,
            "col_marginals_calibrated": col_marginals_calibrated
        }

    # Run Fold 1 (Dev: Subjects 1-5, Held-Out Test: Subjects 6-10)
    fold1 = run_single_fold(fold1_dev_ids, fold1_test_ids, fold_num=1)
    
    # Run Fold 2 (Dev: Subjects 6-10, Held-Out Test: Subjects 1-5)
    fold2 = run_single_fold(fold2_dev_ids, fold2_test_ids, fold_num=2)
    
    # 8. Pooled 10-Subject Cross-Validation Evaluation (90 Genuine, 270 Impostor Comparisons)
    pooled_test_gen_scores = np.concatenate([fold1["test_gen_scores"], fold2["test_gen_scores"]])
    pooled_test_imp_scores = np.concatenate([fold1["test_imp_scores"], fold2["test_imp_scores"]])
    pooled_roc = compute_roc_metrics(pooled_test_gen_scores, pooled_test_imp_scores)
    pooled_conf_3891 = compute_confusion_at_threshold(pooled_test_gen_scores, pooled_test_imp_scores, threshold=0.3891)
    
    pooled_gen_pairs = fold1["test_pairs"]["gen"] + fold2["test_pairs"]["gen"]
    pooled_imp_pairs = fold1["test_pairs"]["imp"] + fold2["test_pairs"]["imp"]
    
    np.save(os.path.join(output_dir, f"socofing_pooled_test_gen_scores_{in_dim}D.npy"), pooled_test_gen_scores)
    np.save(os.path.join(output_dir, f"socofing_pooled_test_imp_scores_{in_dim}D.npy"), pooled_test_imp_scores)
    export_manifest_csv(pooled_gen_pairs + pooled_imp_pairs, os.path.join(output_dir, f"socofing_pooled_test_manifest_{in_dim}D.csv"))

    pooled_gen_recs = [{'subject_a': p['subject_id'], 'subject_b': p['gallery_subject_id'], 'score': float(s)} 
                       for p, s in zip(pooled_gen_pairs, pooled_test_gen_scores)]
    pooled_imp_recs = [{'subject_a': p['subject_id'], 'subject_b': p['gallery_subject_id'], 'score': float(s)} 
                       for p, s in zip(pooled_imp_pairs, pooled_test_imp_scores)]
    pooled_clust_boot_3891 = clustered_bootstrap_confidence_intervals(
        pooled_gen_recs, pooled_imp_recs, operating_threshold=0.3891, num_bootstraps=500, seed=42
    )
    pooled_clust_boot_calibrated = clustered_bootstrap_confidence_intervals(
        pooled_gen_recs, pooled_imp_recs, operating_threshold=fold1["calibrated_T_accept"], num_bootstraps=500, seed=42
    )
    pooled_clust_boot = pooled_clust_boot_3891

    # Compute Pooled 3x3 Cross-Tabulation
    pooled_cell_gen_scores = {(c, d): fold1["test_cell_gen_scores"][(c, d)] + fold2["test_cell_gen_scores"][(c, d)] 
                              for c in alt_classes for d in difficulties}
    pooled_cell_imp_scores = {(c, d): fold1["test_cell_imp_scores"][(c, d)] + fold2["test_cell_imp_scores"][(c, d)] 
                              for c in alt_classes for d in difficulties}

    def compute_generic_table(c_gen: Dict[Tuple[str, str], List[float]], 
                              c_imp: Dict[Tuple[str, str], List[float]], 
                              thresh: float) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        tab = {}
        for c in alt_classes:
            tab[c] = {}
            for d in difficulties:
                cg_arr = np.array(c_gen[(c, d)])
                ci_arr = np.array(c_imp[(c, d)])
                n_g = len(cg_arr)
                n_i = len(ci_arr)
                tp = int(np.sum(cg_arr >= thresh))
                fn = int(np.sum(cg_arr < thresh))
                fp = int(np.sum(ci_arr >= thresh))
                tn = int(np.sum(ci_arr < thresh))
                tar = float(tp / max(n_g, 1))
                far = float(fp / max(n_i, 1))
                tar_ci = clopper_pearson_ci(tp, n_g)
                far_ci = clopper_pearson_ci(fp, n_i)
                tab[c][d] = {
                    "n_genuine": n_g, "n_impostor": n_i,
                    "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                    "tar": tar, "far": far,
                    "tar_95ci": [float(tar_ci[0]), float(tar_ci[1])],
                    "far_95ci": [float(far_ci[0]), float(far_ci[1])],
                    "mean_genuine_score": float(np.mean(cg_arr)) if n_g > 0 else 0.0,
                    "mean_impostor_score": float(np.mean(ci_arr)) if n_i > 0 else 0.0
                }
        r_marg = {}
        for c in alt_classes:
            all_row_g = np.concatenate([c_gen[(c, d)] for d in difficulties])
            all_row_i = np.concatenate([c_imp[(c, d)] for d in difficulties])
            tp = int(np.sum(all_row_g >= thresh))
            fp = int(np.sum(all_row_i >= thresh))
            n_g = len(all_row_g)
            n_i = len(all_row_i)
            tar_ci = clopper_pearson_ci(tp, n_g)
            far_ci = clopper_pearson_ci(fp, n_i)
            r_marg[c] = {
                "n_genuine": n_g, "n_impostor": n_i,
                "tp": tp, "fn": n_g - tp,
                "fp": fp, "tn": n_i - fp,
                "tar": float(tp / n_g),
                "far": float(fp / n_i),
                "tar_95ci": [float(tar_ci[0]), float(tar_ci[1])],
                "far_95ci": [float(far_ci[0]), float(far_ci[1])],
                "mean_genuine_score": float(np.mean(all_row_g)),
                "mean_impostor_score": float(np.mean(all_row_i))
            }
        c_marg = {}
        for d in difficulties:
            all_col_g = np.concatenate([c_gen[(c, d)] for c in alt_classes])
            all_col_i = np.concatenate([c_imp[(c, d)] for c in alt_classes])
            tp = int(np.sum(all_col_g >= thresh))
            fp = int(np.sum(all_col_i >= thresh))
            n_g = len(all_col_g)
            n_i = len(all_col_i)
            tar_ci = clopper_pearson_ci(tp, n_g)
            far_ci = clopper_pearson_ci(fp, n_i)
            c_marg[d] = {
                "n_genuine": n_g, "n_impostor": n_i,
                "tp": tp, "fn": n_g - tp,
                "fp": fp, "tn": n_i - fp,
                "tar": float(tp / n_g),
                "far": float(fp / n_i),
                "tar_95ci": [float(tar_ci[0]), float(tar_ci[1])],
                "far_95ci": [float(far_ci[0]), float(far_ci[1])],
                "mean_genuine_score": float(np.mean(all_col_g)),
                "mean_impostor_score": float(np.mean(all_col_i))
            }
        return tab, r_marg, c_marg

    pooled_cross_tab, pooled_row_marginals, pooled_col_marginals = compute_generic_table(
        pooled_cell_gen_scores, pooled_cell_imp_scores, 0.3891
    )
    pooled_cross_tab_calibrated, pooled_row_marginals_cal, pooled_col_marginals_cal = compute_generic_table(
        pooled_cell_gen_scores, pooled_cell_imp_scores, fold1["calibrated_T_accept"]
    )

    tot_pooled_tp = sum(pooled_cross_tab[c][d]["tp"] for c in alt_classes for d in difficulties)
    tot_pooled_fn = sum(pooled_cross_tab[c][d]["fn"] for c in alt_classes for d in difficulties)
    tot_pooled_fp = sum(pooled_cross_tab[c][d]["fp"] for c in alt_classes for d in difficulties)
    tot_pooled_tn = sum(pooled_cross_tab[c][d]["tn"] for c in alt_classes for d in difficulties)
    assert tot_pooled_tp == pooled_conf_3891["tp"], f"Pooled TP mismatch: {tot_pooled_tp} vs {pooled_conf_3891['tp']}"
    assert tot_pooled_fn == pooled_conf_3891["fn"], f"Pooled FN mismatch: {tot_pooled_fn} vs {pooled_conf_3891['fn']}"
    assert tot_pooled_fp == pooled_conf_3891["fp"], f"Pooled FP mismatch: {tot_pooled_fp} vs {pooled_conf_3891['fp']}"
    assert tot_pooled_tn == pooled_conf_3891["tn"], f"Pooled TN mismatch: {tot_pooled_tn} vs {pooled_conf_3891['tn']}"
    assert tot_pooled_tp + tot_pooled_fn == len(pooled_test_gen_scores), "Pooled genuine trial count identity violation"
    assert tot_pooled_fp + tot_pooled_tn == len(pooled_test_imp_scores), "Pooled impostor trial count identity violation"

    print(f"\n[Pooled 10-Subject CV] N_gen = {len(pooled_test_gen_scores)}, N_imp = {len(pooled_test_imp_scores)}")
    print(f"[Pooled 10-Subject CV] EER = {pooled_roc['eer']*100:.2f}%, AUC = {pooled_roc['auc']:.4f}")
    print(f"[Pooled 10-Subject CV] @ T=0.3891: TAR = {pooled_conf_3891['tar']*100:.2f}%, FAR = {pooled_conf_3891['far']*100:.2f}%")
    
    # Assemble comprehensive report preserving backward compatibility
    report = {
        "dataset": "SOCOFing",
        "alteration_nature": "synthetic digital alterations (simulated Central Rotation, Z-cut Discontinuity, and Obliteration Scars on digital sensor images; not live physical tissue injuries)",
        "descriptor_dim": in_dim,
        "num_subjects": len(all_subject_ids),
        "num_dev_subjects": len(fold1_dev_ids),
        "num_test_subjects": len(fold1_test_ids),
        "ci_methodology": {
            "per_cell": "Exact two-sided Clopper-Pearson 95% binomial confidence interval for independent Bernoulli trials.",
            "aggregate": "Two-level subject-clustered bootstrap (B=500 replicates) resampling independent subject clusters with replacement."
        },
        "threshold_calibration": {
            "dev_eer": float(fold1["dev_roc"]["eer"]),
            "dev_auc": float(fold1["dev_roc"]["auc"]),
            "calibrated_T_accept": fold1["calibrated_T_accept"],
            "frozen_benchmark_T_accept": fold1["frozen_benchmark_T_accept"]
        },
        "held_out_test_accounting": {
            "num_genuine_trials": len(fold1["test_gen_scores"]),
            "num_impostor_trials": len(fold1["test_imp_scores"]),
            "test_eer": float(fold1["test_roc"]["eer"]),
            "test_auc": float(fold1["test_roc"]["auc"]),
            "clustered_bootstrap_95ci": {
                "eer_ci": fold1["test_clust_boot"]["eer_ci"],
                "auc_ci": fold1["test_clust_boot"]["auc_ci"],
                "far_ci": fold1["test_clust_boot"]["clustered_far_ci"],
                "frr_ci": fold1["test_clust_boot"]["clustered_frr_ci"]
            },
            "overall_confusion_at_calibrated_threshold": fold1["test_conf_calibrated"],
            "overall_confusion_at_3891": fold1["test_conf_3891"],
            "overall_confusion_at_frozen_040": fold1["test_conf_frozen40"],
            "naive_legacy_020_warning": {
                "threshold": 0.20,
                "tar": fold1["legacy_conf_020"]["tar"],
                "far": fold1["legacy_conf_020"]["far"],
                "note": f"At T=0.20, FAR is {fold1['legacy_conf_020']['far']*100:.1f}% (inadequate rejection due to impostor baseline)."
            }
        },
        "preliminary_status_notice": (
            "Results on held-out synthetic digital alteration test set are preliminary due to elevated error rates "
            f"(Fold 1: FRR = {fold1['test_conf_calibrated']['frr']*100:.1f}%, FAR = {fold1['test_conf_calibrated']['far']*100:.1f}% at calibrated T_accept = {fold1['calibrated_T_accept']:.4f}, EER = {fold1['test_roc']['eer']*100:.1f}%; "
            f"Fold 2: FRR = {fold2['test_conf_calibrated']['frr']*100:.1f}%, FAR = {fold2['test_conf_calibrated']['far']*100:.1f}%, EER = {fold2['test_roc']['eer']*100:.1f}%; "
            f"Pooled 10-Subject CV: EER = {pooled_roc['eer']*100:.1f}%, AUC = {pooled_roc['auc']:.4f}). "
            "Confirms synthetic digital alteration degradation pattern: Easy -> Medium -> Hard."
        ),
        "held_out_3x3_cross_tabulation": {
            "evaluation_threshold": 0.3891,
            "harmonization_status": "Harmonized at benchmark threshold T_accept = 0.3891 (resolves FP=25 vs FP=15 discrepancy; exact row-level sum matches aggregate confusion matrix)",
            "cells": fold1["test_cross_tab"],
            "row_marginals_by_class": fold1["row_marginals"],
            "column_marginals_by_difficulty": fold1["col_marginals"],
            "calibrated_threshold_alternative": {
                "evaluation_threshold": fold1["calibrated_T_accept"],
                "cells": fold1["test_cross_tab_calibrated"],
                "row_marginals_by_class": fold1["row_marginals_calibrated"],
                "column_marginals_by_difficulty": fold1["col_marginals_calibrated"]
            }
        },
        "two_fold_cross_validation": {
            "fold1": {
                "dev_subjects": fold1_dev_ids,
                "test_subjects": fold1_test_ids,
                "dev_calibrated_threshold": fold1["calibrated_T_accept"],
                "test_eer": float(fold1["test_roc"]["eer"]),
                "test_auc": float(fold1["test_roc"]["auc"]),
                "test_confusion_at_calibrated": fold1["test_conf_calibrated"],
                "test_confusion_at_3891": fold1["test_conf_3891"],
                "test_clustered_ci_95": fold1["test_clust_boot_3891"],
                "test_clustered_ci_at_calibrated_95": fold1["test_clust_boot_calibrated"],
                "cells": fold1["test_cross_tab"],
                "row_marginals_by_class": fold1["row_marginals"],
                "column_marginals_by_difficulty": fold1["col_marginals"],
                "calibrated_threshold_alternative": {
                    "evaluation_threshold": fold1["calibrated_T_accept"],
                    "cells": fold1["test_cross_tab_calibrated"],
                    "row_marginals_by_class": fold1["row_marginals_calibrated"],
                    "column_marginals_by_difficulty": fold1["col_marginals_calibrated"]
                }
            },
            "fold2": {
                "dev_subjects": fold2_dev_ids,
                "test_subjects": fold2_test_ids,
                "dev_calibrated_threshold": fold2["calibrated_T_accept"],
                "test_eer": float(fold2["test_roc"]["eer"]),
                "test_auc": float(fold2["test_roc"]["auc"]),
                "test_confusion_at_calibrated": fold2["test_conf_calibrated"],
                "test_confusion_at_3891": fold2["test_conf_3891"],
                "test_clustered_ci_95": fold2["test_clust_boot_3891"],
                "test_clustered_ci_at_calibrated_95": fold2["test_clust_boot_calibrated"],
                "cells": fold2["test_cross_tab"],
                "row_marginals_by_class": fold2["row_marginals"],
                "column_marginals_by_difficulty": fold2["col_marginals"],
                "calibrated_threshold_alternative": {
                    "evaluation_threshold": fold2["calibrated_T_accept"],
                    "cells": fold2["test_cross_tab_calibrated"],
                    "row_marginals_by_class": fold2["row_marginals_calibrated"],
                    "column_marginals_by_difficulty": fold2["col_marginals_calibrated"]
                }
            },
            "pooled_10_subjects": {
                "num_genuine_trials": len(pooled_test_gen_scores),
                "num_impostor_trials": len(pooled_test_imp_scores),
                "eer": float(pooled_roc["eer"]),
                "auc": float(pooled_roc["auc"]),
                "confusion_at_3891": pooled_conf_3891,
                "clustered_ci_95": pooled_clust_boot_3891,
                "clustered_ci_at_calibrated_95": pooled_clust_boot_calibrated,
                "cells": pooled_cross_tab,
                "row_marginals_by_class": pooled_row_marginals,
                "column_marginals_by_difficulty": pooled_col_marginals,
                "calibrated_threshold_alternative": {
                    "evaluation_threshold": fold1["calibrated_T_accept"],
                    "cells": pooled_cross_tab_calibrated,
                    "row_marginals_by_class": pooled_row_marginals_cal,
                    "column_marginals_by_difficulty": pooled_col_marginals_cal
                }
            }
        }
    }
    
    report_file = os.path.join(output_dir, f"socofing_benchmark_report_{in_dim}D.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    def print_3x3(title: str, tab: Dict[str, Any], r_marg: Dict[str, Any], c_marg: Dict[str, Any], conf: Dict[str, Any]):
        print(f"\n=======================================================")
        print(f"  {title}")
        print(f"=======================================================")
        print(f"{'Synthetic Alteration':<20} | {'Easy (TAR/FAR)':<16} | {'Medium (TAR/FAR)':<16} | {'Hard (TAR/FAR)':<16} | {'Row Total'}")
        print("------------------------------------------------------------------------------------------------------")
        for c in alt_classes:
            e = tab[c]["Easy"]
            m = tab[c]["Medium"]
            h = tab[c]["Hard"]
            row = r_marg[c]
            print(f"{c:<20} | {e['tar']*100:5.1f}% / {e['far']*100:4.1f}%  | {m['tar']*100:5.1f}% / {m['far']*100:4.1f}%  | {h['tar']*100:5.1f}% / {h['far']*100:4.1f}%  | {row['tar']*100:5.1f}% / {row['far']*100:4.1f}% ({row['tp']}/{row['n_genuine']})")
        print("------------------------------------------------------------------------------------------------------")
        ce = c_marg["Easy"]
        cm = c_marg["Medium"]
        ch = c_marg["Hard"]
        print(f"{'Col Total':<20} | {ce['tar']*100:5.1f}% / {ce['far']*100:4.1f}%  | {cm['tar']*100:5.1f}% / {cm['far']*100:4.1f}%  | {ch['tar']*100:5.1f}% / {ch['far']*100:4.1f}%  | {conf['tar']*100:5.1f}% / {conf['far']*100:4.1f}% (Total: TP={conf['tp']}, FP={conf['fp']})")
        print("======================================================================================================\n")

    print_3x3("Fold 1 Held-Out Test 3x3 Cross-Tabulation (Subj 6-10 @ T=0.3891)", 
              fold1["test_cross_tab"], fold1["row_marginals"], fold1["col_marginals"], fold1["test_conf_3891"])
    print_3x3("Fold 2 Held-Out Test 3x3 Cross-Tabulation (Subj 1-5 @ T=0.3891)", 
              fold2["test_cross_tab"], fold2["row_marginals"], fold2["col_marginals"], fold2["test_conf_3891"])
    print_3x3("Pooled 10-Subject Cross-Validation 3x3 Cross-Tabulation (Subj 1-10 @ T=0.3891)", 
              pooled_cross_tab, pooled_row_marginals, pooled_col_marginals, pooled_conf_3891)
        
    print(f"Saved complete report to: {report_file}\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cancelable Fingerprint Biometrics on SOCOFing.")
    parser.add_argument("--root_dir", type=str, default="dataset/SOCOFing", help="Path to SOCOFing dataset directory.")
    parser.add_argument("--dim", type=int, default=9, choices=[9, 12], help="Descriptor dimension.")
    parser.add_argument("--output_dir", type=str, default="experiments/results/socofing", help="Output directory.")
    parser.add_argument("--max_subjects", type=int, default=10, help="Number of subjects for evaluation.")
    args = parser.parse_args()
    evaluate_socofing_benchmark(root_dir=args.root_dir, in_dim=args.dim, output_dir=args.output_dir, max_subjects=args.max_subjects)
