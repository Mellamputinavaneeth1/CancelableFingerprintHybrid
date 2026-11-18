"""
Phase 3: In-Silico Nuisance Stress Tests Runner.
Evaluates Cancelable Fingerprint Biometrics against controlled synthetic capture nuisances
on real fingerprint images from FVC2002 DB1_B (10 fingers, 8 impressions each):
  1. Planar Rotation: [-45°, -30°, -15°, 0°, +15°, +30°, +45°]
  2. Planar Translation: [(-30, -30), (-15, -15), (0, 0), (+15, +15), (+30, +30)] px
  3. Partial Print Cropping: [100%, 80%, 60%, 40%, 25%] area retention
  4. Non-Linear Elastic Deformation: [0.0, 5.0, 10.0, 15.0, 20.0] px peak displacement

For each condition and intensity level:
  - Extraction success rate, FTE rate, and FTA ("N/A (offline database)")
  - Average minutiae and Delaunay triangle counts (mean ± std)
  - Genuine score distribution (mean ± std) and score shift Delta_mu vs baseline
  - Impostor score distribution (mean ± std) and score shift Delta_mu vs baseline
  - TAR and FAR at frozen operational threshold T_accept = 0.3751 with exact Clopper-Pearson 95% CIs
  - Full raw score arrays (.npy), manifests (.csv), summary table (.csv), and canonical JSON report.
"""

import os
import sys
import time
import json
import csv
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import CancelableFingerprintPipeline
from src.preprocessing.segmenter import segment_fingerprint
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors
from src.matching.bipartite_matcher import match_protected_templates
from src.evaluation.metrics import clopper_pearson_ci, compute_confusion_at_threshold
from experiments.datasets.fvc_loader import load_fvc_dataset
from experiments.protocols.pair_generator import export_manifest_csv


FROZEN_DEV_THRESHOLD = 0.3751


# =====================================================================
# 1. Synthetic Nuisance Transformation Operators
# =====================================================================

def apply_planar_rotation(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotates image around its center by angle_deg.
    Fills border with 255 (standard optical sensor background).
    """
    if abs(angle_deg) < 1e-6:
        return image.copy()
    h, w = image.shape
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def apply_translation(image: np.ndarray, offset_px: Tuple[int, int]) -> np.ndarray:
    """
    Translates image by (dx, dy) pixels.
    Fills exposed margins with 255 (background).
    """
    dx, dy = offset_px
    if dx == 0 and dy == 0:
        return image.copy()
    h, w = image.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def apply_partial_crop(image: np.ndarray, area_ratio: float) -> np.ndarray:
    """
    Simulates partial fingerprint platen capture by retaining a bounding box
    retaining area_ratio of active print area centered on the active print center.
    Pixels outside the bounding box are masked to 255 (background).
    """
    if area_ratio >= 1.0 - 1e-6:
        return image.copy()
    h, w = image.shape
    _, mask, bbox = segment_fingerprint(image)
    if mask.sum() > 0:
        bx, by, bw, bh = bbox
        cx, cy = bx + bw / 2.0, by + bh / 2.0
    else:
        bw, bh = w, h
        cx, cy = w / 2.0, h / 2.0
        
    s = np.sqrt(max(0.01, min(1.0, area_ratio)))
    wc = int(round(bw * s))
    hc = int(round(bh * s))
    x1 = max(0, int(round(cx - wc / 2.0)))
    y1 = max(0, int(round(cy - hc / 2.0)))
    x2 = min(w, x1 + wc)
    y2 = min(h, y1 + hc)
    out = np.full_like(image, 255)
    out[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return out


def apply_elastic_deformation(image: np.ndarray, max_displacement_px: float) -> np.ndarray:
    """
    Applies non-linear radial and torsional elastic skin deformation.
    Displaces ridge coordinates continuously using a smooth Gaussian profile,
    modeling non-uniform contact finger pressure and skin twisting centered on the active print.
    """
    if max_displacement_px <= 1e-6:
        return image.copy()
    h, w = image.shape
    _, mask, bbox = segment_fingerprint(image)
    if mask.sum() > 0:
        bx, by, bw, bh = bbox
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        r0 = min(bw, bh) / 2.0
    else:
        cx, cy = w / 2.0, h / 2.0
        r0 = min(h, w) / 2.0
        
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u = x - cx
    v = y - cy
    r = np.sqrt(u**2 + v**2)
    phi = np.arctan2(v, u)
    
    sigma = r0 / 2.0
    # Peak displacement occurs near r = sigma (mid-contact annulus)
    radial_factor = max_displacement_px * np.exp(0.5) / sigma * r * np.exp(-r**2 / (2.0 * sigma**2))
    tang_twist = (max_displacement_px * 0.5 / r0) * np.exp(-r**2 / (2.0 * sigma**2))
    
    r_src = np.maximum(0.0, r - radial_factor)
    phi_src = phi - tang_twist
    
    map_x = (cx + r_src * np.cos(phi_src)).astype(np.float32)
    map_y = (cy + r_src * np.sin(phi_src)).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


# =====================================================================
# 2. Benchmark Definition
# =====================================================================

NUISANCE_CONDITIONS = {
    "planar_rotation": {
        "name": "Planar Rotation",
        "parameter_name": "angle_deg",
        "unit": "deg",
        "levels": [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0],
        "baseline_index": 3,
        "operator": lambda img, lvl: apply_planar_rotation(img, lvl)
    },
    "translation": {
        "name": "Translation",
        "parameter_name": "offset_px",
        "unit": "px",
        "levels": [(-30, -30), (-15, -15), (0, 0), (15, 15), (30, 30)],
        "baseline_index": 2,
        "operator": lambda img, lvl: apply_translation(img, lvl)
    },
    "partial_crop": {
        "name": "Partial Print Cropping",
        "parameter_name": "area_ratio",
        "unit": "ratio",
        "levels": [1.00, 0.80, 0.60, 0.40, 0.25],
        "baseline_index": 0,
        "operator": lambda img, lvl: apply_partial_crop(img, lvl)
    },
    "elastic_deformation": {
        "name": "Non-Linear Elastic Deformation",
        "parameter_name": "max_displacement_px",
        "unit": "px",
        "levels": [0.0, 5.0, 10.0, 15.0, 20.0],
        "baseline_index": 0,
        "operator": lambda img, lvl: apply_elastic_deformation(img, lvl)
    }
}


# =====================================================================
# 3. Benchmark Execution Runner
# =====================================================================

def evaluate_nuisance_stress_tests(
    subset: str = "DB1_B",
    output_dir: str = "experiments/results/phase3/nuisance",
    operating_threshold: float = FROZEN_DEV_THRESHOLD,
    save_raw_scores: bool = True
) -> Dict[str, Any]:
    """
    Executes the comprehensive Phase 3 capture-quality and nuisance stress test suite.
    """
    print(f"\n{'='*80}")
    print(f"  PHASE 3: IN-SILICO NUISANCE STRESS TESTS ({subset})")
    print(f"  Controlled Capture Variations: Rotation, Translation, Cropping, Deformation")
    print(f"{'='*80}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_fvc_dataset(subset, auto_download=False)
    dpi = dataset["dpi"]
    fingers = dataset["fingers"]
    finger_ids = sorted(list(fingers.keys()))
    num_fingers = len(finger_ids)
    
    pipe = CancelableFingerprintPipeline(in_dim=9, dpi=dpi, hamming_threshold=0.22)
    
    # 1. Fit empirical standardization parameters on clean DB1_B dataset
    print(f"Standardizing 9-D descriptor geometry across {dataset['total_images']} clean images...")
    clean_descs = []
    clean_gallery_images = {}
    clean_probe_images = {}  # Primary probe: Impression 2
    
    for fid in finger_ids:
        # Gallery: Impression 1
        g_img = cv2.imread(fingers[fid][1], cv2.IMREAD_GRAYSCALE)
        clean_gallery_images[fid] = g_img
        f_g = pipe.extract_features(g_img)
        if len(f_g["descriptors"]) > 0:
            clean_descs.append(f_g["descriptors"])
            
        # Primary Probe: Impression 2
        p_img = cv2.imread(fingers[fid][2], cv2.IMREAD_GRAYSCALE)
        clean_probe_images[fid] = p_img
        f_p = pipe.extract_features(p_img)
        if len(f_p["descriptors"]) > 0:
            clean_descs.append(f_p["descriptors"])
            
        # Impressions 3..8 for empirical descriptor standardization
        for imp_id in range(3, 9):
            if imp_id in fingers[fid]:
                imp_img = cv2.imread(fingers[fid][imp_id], cv2.IMREAD_GRAYSCALE)
                f_imp = pipe.extract_features(imp_img)
                if len(f_imp["descriptors"]) > 0:
                    clean_descs.append(f_imp["descriptors"])
                    
    emp_mean = np.mean(np.vstack(clean_descs), axis=0)
    emp_std = np.std(np.vstack(clean_descs), axis=0)
    
    # Derive canonical transform key W
    W = derive_orthonormal_matrix(
        pipe.master_secret, salt=f"salt_{subset}", app_id=f"fvc2002_{subset}", in_dim=9, out_dim=32
    )
    
    # Enroll clean Gallery (Impression 1)
    print(f"Enrolling clean Gallery credentials for {num_fingers} fingers...")
    gallery_templates: Dict[int, Dict[str, Any]] = {}
    for fid in finger_ids:
        f = pipe.extract_features(clean_gallery_images[fid])
        if len(f["descriptors"]) == 0:
            gallery_templates[fid] = {
                "bitstrings": np.empty((0, 32), dtype=np.uint8),
                "weights": np.empty((0,), dtype=np.float32),
                "fte": True
            }
        else:
            proj = project_descriptors(f["descriptors"], W, mean=emp_mean, std=emp_std)
            bits = quantize_descriptors(proj)
            gallery_templates[fid] = {
                "bitstrings": bits,
                "weights": f["weights"],
                "fte": False,
                "minutiae_count": len(f["minutiae"]),
                "triangle_count": len(f["triangles"])
            }
            
    # Structure for summary report
    results_summary: Dict[str, Any] = {
        "benchmark": "Phase 3 In-Silico Nuisance Stress Tests",
        "dataset": subset,
        "native_dpi": dpi,
        "num_fingers": num_fingers,
        "frozen_operational_threshold": operating_threshold,
        "ci_methodology": {
            "per_cell": "Exact two-sided Clopper-Pearson 95% binomial confidence interval for independent Bernoulli trials.",
            "aggregate": "Two-level subject-clustered bootstrap across independent finger clusters."
        },
        "fta_status": "N/A (offline database)",
        "conditions": {}
    }
    
    summary_csv_rows = []
    
    # 2. Iterate through each Nuisance Condition
    for cond_key, cond_meta in NUISANCE_CONDITIONS.items():
        cond_name = cond_meta["name"]
        param_name = cond_meta["parameter_name"]
        unit = cond_meta["unit"]
        levels = cond_meta["levels"]
        baseline_idx = cond_meta["baseline_index"]
        op_fn = cond_meta["operator"]
        
        print(f"\n--- Testing Capture Nuisance: {cond_name} ({len(levels)} levels) ---")
        cond_dir = os.path.join(output_dir, cond_key)
        os.makedirs(cond_dir, exist_ok=True)
        
        cond_results: List[Dict[str, Any]] = []
        baseline_gen_mean: Optional[float] = None
        baseline_imp_mean: Optional[float] = None
        
        for lvl_idx, lvl in enumerate(levels):
            lvl_str = f"{lvl[0]}_{lvl[1]}" if isinstance(lvl, tuple) else f"{lvl}"
            if unit == "deg":
                lvl_label = f"{int(lvl):+d}°"
            elif unit == "ratio":
                lvl_label = f"{lvl*100:.0f}%"
            elif isinstance(lvl, tuple):
                lvl_label = f"({lvl[0]:+d}, {lvl[1]:+d}) px"
            else:
                lvl_label = f"{lvl:.1f} px"
            
            probe_templates: Dict[int, Dict[str, Any]] = {}
            min_counts = []
            tri_counts = []
            fte_count = 0
            
            # Apply transformation to probe impressions (Impression 2)
            for fid in finger_ids:
                orig_probe = clean_probe_images[fid]
                warped_probe = op_fn(orig_probe, lvl)
                
                feat = pipe.extract_features(warped_probe)
                min_counts.append(len(feat["minutiae"]))
                tri_counts.append(len(feat["triangles"]))
                
                if len(feat["descriptors"]) == 0:
                    fte_count += 1
                    probe_templates[fid] = {
                        "bitstrings": np.empty((0, 32), dtype=np.uint8),
                        "weights": np.empty((0,), dtype=np.float32),
                        "fte": True
                    }
                else:
                    proj = project_descriptors(feat["descriptors"], W, mean=emp_mean, std=emp_std)
                    bits = quantize_descriptors(proj)
                    probe_templates[fid] = {
                        "bitstrings": bits,
                        "weights": feat["weights"],
                        "fte": False
                    }
                    
            fte_rate = float(fte_count / num_fingers)
            avg_minutiae = float(np.mean(min_counts))
            std_minutiae = float(np.std(min_counts))
            avg_triangles = float(np.mean(tri_counts))
            std_triangles = float(np.std(tri_counts))
            
            # Compare Probes against Gallery
            gen_scores = []
            imp_scores = []
            pair_records = []
            pair_counter = 0
            
            # Genuine comparisons: Probe fid vs Gallery fid (N=10)
            for fid in finger_ids:
                t_p = probe_templates[fid]
                t_g = gallery_templates[fid]
                if t_p["fte"] or t_g["fte"]:
                    s = 0.0
                else:
                    s = match_protected_templates(
                        t_p["bitstrings"], t_g["bitstrings"], t_p["weights"], t_g["weights"], hamming_threshold=0.22
                    )
                gen_scores.append(s)
                pair_records.append({
                    "pair_id": pair_counter,
                    "condition": cond_key,
                    "level": lvl_str,
                    "probe_finger_id": fid,
                    "gallery_finger_id": fid,
                    "genuine": 1,
                    "score": s
                })
                pair_counter += 1
                
            # Impostor comparisons: Probe fid1 vs Gallery fid2 (fid1 != fid2, N=90)
            for fid1 in finger_ids:
                t_p = probe_templates[fid1]
                for fid2 in finger_ids:
                    if fid1 == fid2:
                        continue
                    t_g = gallery_templates[fid2]
                    if t_p["fte"] or t_g["fte"]:
                        s = 0.0
                    else:
                        s = match_protected_templates(
                            t_p["bitstrings"], t_g["bitstrings"], t_p["weights"], t_g["weights"], hamming_threshold=0.22
                        )
                    imp_scores.append(s)
                    pair_records.append({
                        "pair_id": pair_counter,
                        "condition": cond_key,
                        "level": lvl_str,
                        "probe_finger_id": fid1,
                        "gallery_finger_id": fid2,
                        "genuine": 0,
                        "score": s
                    })
                    pair_counter += 1
                    
            gen_arr = np.array(gen_scores, dtype=np.float32)
            imp_arr = np.array(imp_scores, dtype=np.float32)
            
            n_gen = len(gen_arr)
            n_imp = len(imp_arr)
            tp = int(np.sum(gen_arr >= operating_threshold))
            fn = int(np.sum(gen_arr < operating_threshold))
            fp = int(np.sum(imp_arr >= operating_threshold))
            tn = int(np.sum(imp_arr < operating_threshold))
            
            assert tp + fn == n_gen, f"Genuine trial identity violated at {cond_key} lvl {lvl_str}"
            assert fp + tn == n_imp, f"Impostor trial identity violated at {cond_key} lvl {lvl_str}"
            
            tar = float(tp / max(n_gen, 1))
            far = float(fp / max(n_imp, 1))
            tar_ci = clopper_pearson_ci(tp, n_gen)
            far_ci = clopper_pearson_ci(fp, n_imp)
            
            mu_gen = float(np.mean(gen_arr))
            std_gen = float(np.std(gen_arr))
            mu_imp = float(np.mean(imp_arr))
            std_imp = float(np.std(imp_arr))
            
            # Save raw arrays and manifest
            lvl_dir = os.path.join(cond_dir, f"lvl_{lvl_str}")
            os.makedirs(lvl_dir, exist_ok=True)
            if save_raw_scores:
                np.save(os.path.join(lvl_dir, "genuine_scores.npy"), gen_arr)
                np.save(os.path.join(lvl_dir, "impostor_scores.npy"), imp_arr)
                export_manifest_csv(pair_records, os.path.join(lvl_dir, "manifest.csv"))
                
            level_record = {
                "intensity_value": lvl if not isinstance(lvl, tuple) else list(lvl),
                "intensity_label": lvl_label,
                "is_baseline": (lvl_idx == baseline_idx),
                "extraction_success_count": num_fingers - fte_count,
                "fte_count": fte_count,
                "fte_rate": fte_rate,
                "fta": "N/A (offline database)",
                "avg_minutiae": avg_minutiae,
                "std_minutiae": std_minutiae,
                "avg_triangles": avg_triangles,
                "std_triangles": std_triangles,
                "num_genuine": n_gen,
                "num_impostor": n_imp,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "tar": tar,
                "far": far,
                "tar_95ci": [float(tar_ci[0]), float(tar_ci[1])],
                "far_95ci": [float(far_ci[0]), float(far_ci[1])],
                "genuine_mean": mu_gen,
                "genuine_std": std_gen,
                "impostor_mean": mu_imp,
                "impostor_std": std_imp,
                "genuine_score_shift_delta": 0.0,
                "impostor_score_shift_delta": 0.0
            }
            cond_results.append(level_record)
            
        # Post-process deltas relative to baseline
        base_gen_mean = cond_results[baseline_idx]["genuine_mean"]
        base_imp_mean = cond_results[baseline_idx]["impostor_mean"]
        for rec in cond_results:
            rec["genuine_score_shift_delta"] = float(rec["genuine_mean"] - base_gen_mean)
            rec["impostor_score_shift_delta"] = float(rec["impostor_mean"] - base_imp_mean)
            
            delta_gen = rec["genuine_score_shift_delta"]
            delta_imp = rec["impostor_score_shift_delta"]
            lvl_label = rec["intensity_label"]
            avg_triangles = rec["avg_triangles"]
            mu_gen = rec["genuine_mean"]
            mu_imp = rec["impostor_mean"]
            tar = rec["tar"]
            far = rec["far"]
            fte_rate = rec["fte_rate"]
            tar_ci = rec["tar_95ci"]
            far_ci = rec["far_95ci"]
            
            summary_csv_rows.append({
                "condition": cond_name,
                "intensity": lvl_label,
                "is_baseline": rec["is_baseline"],
                "avg_minutiae": f"{rec['avg_minutiae']:.1f} ± {rec['std_minutiae']:.1f}",
                "avg_triangles": f"{avg_triangles:.1f} ± {rec['std_triangles']:.1f}",
                "fte_rate": f"{fte_rate*100:.1f}%",
                "genuine_mean_std": f"{mu_gen:.4f} ± {rec['genuine_std']:.4f}",
                "impostor_mean_std": f"{mu_imp:.4f} ± {rec['impostor_std']:.4f}",
                "delta_genuine_mu": f"{delta_gen:+.4f}",
                "delta_impostor_mu": f"{delta_imp:+.4f}",
                "tar_pct": f"{tar*100:.1f}%",
                "tar_95ci": f"[{tar_ci[0]*100:.1f}%, {tar_ci[1]*100:.1f}%]",
                "far_pct": f"{far*100:.1f}%",
                "far_95ci": f"[{far_ci[0]*100:.1f}%, {far_ci[1]*100:.1f}%]"
            })
            
            print(f"  Level {lvl_label:<18} | Tri: {avg_triangles:5.1f} | Gen: {mu_gen:.4f} ({delta_gen:+.4f}) | Imp: {mu_imp:.4f} ({delta_imp:+.4f}) | TAR: {tar*100:5.1f}% | FAR: {far*100:4.1f}% | FTE: {fte_rate*100:.1f}%")
            
        results_summary["conditions"][cond_key] = {
            "name": cond_name,
            "parameter_name": param_name,
            "unit": unit,
            "levels_evaluated": cond_results
        }
        
    # Write summary CSV
    csv_file = os.path.join(output_dir, "nuisance_summary.csv")
    if summary_csv_rows:
        fieldnames = list(summary_csv_rows[0].keys())
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_csv_rows)
            
    # Write canonical report JSON
    report_file = os.path.join(output_dir, "nuisance_benchmark_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
        
    print(f"\nSaved Nuisance summary CSV to: {csv_file}")
    print(f"Saved complete Nuisance report JSON to: {report_file}\n")
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 3 In-Silico Nuisance Stress Tests.")
    parser.add_argument("--subset", type=str, default="DB1_B", help="FVC subset to evaluate.")
    parser.add_argument("--output_dir", type=str, default="experiments/results/phase3/nuisance", help="Output directory.")
    parser.add_argument("--threshold", type=float, default=FROZEN_DEV_THRESHOLD, help="Frozen operational threshold.")
    args = parser.parse_args()
    evaluate_nuisance_stress_tests(subset=args.subset, output_dir=args.output_dir, operating_threshold=args.threshold)
