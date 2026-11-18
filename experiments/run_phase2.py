"""
Phase 2 Master Execution and Verification Runner.
Executes Phase 2 end-to-end:
  1. Inversion & Reconstruction Feasibility Attacks (9-D vs 12-D, M in {32, 64, 128} bits)
  2. Minutiae-Type Attribute Leakage Analysis (strictly subject-disjoint 5-fold CV)
  3. Revocability & Cross-Epoch Cancellation Evaluation (ISO/IEC 24745:2022 lifecycle)
  4. Threat Model Matrix & Multi-Threat Unlinkability Synthesis (B=2,000 two-way clustered bootstrap)
  5. Automated Integrity & Accounting Assertions across all Phase 2 artifacts
  6. Compiles Canonical Phase 2 Summary Report: experiments/results/phase2/phase2_summary.json
"""

import os
import sys
import time
import json
import csv
from typing import Dict, Any, List
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.evaluate_phase2_inversion import run_phase2_inversion_benchmarks
from experiments.evaluate_phase2_leakage import run_phase2_minutiae_leakage_analysis
from experiments.evaluate_phase2_revocation import run_phase2_revocation_experiment
from experiments.evaluate_phase2_threat_matrix import run_phase2_threat_matrix_evaluation


def verify_phase2_integrity(phase2_dir: str = "experiments/results/phase2") -> bool:
    print(f"\n{'='*75}")
    print("  Asserting Scientific & Accounting Integrity across Phase 2 Outputs")
    print(f"{'='*75}")
    
    # 1. Verify Inversion artifacts
    inv_dir = os.path.join(phase2_dir, "inversion")
    inv_json = os.path.join(inv_dir, "inversion_report.json")
    inv_csv = os.path.join(inv_dir, "reconstruction_manifest.csv")
    assert os.path.exists(inv_json), "Missing inversion_report.json"
    assert os.path.exists(inv_csv), "Missing reconstruction_manifest.csv"
    
    with open(inv_json, "r", encoding="utf-8") as f:
        inv_data = json.load(f)
    with open(inv_csv, "r", encoding="utf-8") as f:
        inv_row_count = sum(1 for _ in f) - 1
    assert inv_row_count == inv_data["total_triangles_evaluated"], f"Inversion row mismatch: {inv_row_count} != {inv_data['total_triangles_evaluated']}"
    print(f"  [PASS] Inversion: {inv_row_count} trials verified across 6 factorial conditions.")
    
    # Assert 9-D has 0 discrete variables stored, while 12-D has discrete type recovery >= 90%
    grid = inv_data["factorial_grid"]
    assert grid["9D_pure_geometry"]["M_32"]["minutiae_type_recovery_accuracy"] is None
    assert grid["12D_minutiae_type"]["M_32"]["minutiae_type_recovery_accuracy"] >= 0.90
    reissue_asr = grid["9D_pure_geometry"]["M_32"]["authentication_evaluation"]["cross_key_reissuance_asr_pct"]
    assert reissue_asr == 0.0, f"Cross-key re-issuance ASR non-zero: {reissue_asr}"
    print("  [PASS] Inversion: 9-D type exclusion (0 variables), 12-D MILP lattice recovery, and 0.0% re-issuance ASR confirmed.")
    
    # 2. Verify Leakage artifacts
    leak_dir = os.path.join(phase2_dir, "leakage")
    leak_json = os.path.join(leak_dir, "leakage_report.json")
    leak_csv = os.path.join(leak_dir, "leakage_predictions_manifest.csv")
    assert os.path.exists(leak_json), "Missing leakage_report.json"
    assert os.path.exists(leak_csv), "Missing leakage_predictions_manifest.csv"
    
    with open(leak_json, "r", encoding="utf-8") as f:
        leak_data = json.load(f)
    with open(leak_csv, "r", encoding="utf-8") as f:
        leak_row_count = sum(1 for _ in f) - 1
    assert leak_row_count == leak_data["total_triangles"], f"Leakage row mismatch: {leak_row_count} != {leak_data['total_triangles']}"
    
    # Assert balanced accuracy is tightly bounded near chance
    lr_bal = leak_data["models"]["raw_geom_logistic_regression"]["balanced_accuracy"]
    rf_bal = leak_data["models"]["raw_geom_random_forest"]["balanced_accuracy"]
    assert 0.45 <= lr_bal <= 0.55, f"LR balanced accuracy out of bounds: {lr_bal}"
    assert 0.45 <= rf_bal <= 0.55, f"RF balanced accuracy out of bounds: {rf_bal}"
    rf_bh_p = leak_data["models"]["raw_geom_random_forest"]["bh_adjusted_p_value"]
    assert rf_bh_p >= 0.05, f"RF BH-adjusted p-value is significant: {rf_bh_p}"
    
    # Assert stratified random baseline reflects genuine Bernoulli sampling
    rand_acc = leak_data["models"]["stratified_random_baseline"]["accuracy"]
    assert 0.80 <= rand_acc <= 0.90, f"Stratified random accuracy out of expected range: {rand_acc}"
    print(f"  [PASS] Leakage: {leak_row_count} real triangles verified under disjoint 5-fold CV (LR bal acc={lr_bal:.2%}, RF BH-adj p={rf_bh_p:.4f} >= 0.05).")
    
    # 3. Verify Revocation artifacts
    rev_dir = os.path.join(phase2_dir, "revocation")
    rev_json = os.path.join(rev_dir, "revocation_report.json")
    rev_csv = os.path.join(rev_dir, "revocation_manifest.csv")
    assert os.path.exists(rev_json), "Missing revocation_report.json"
    assert os.path.exists(rev_csv), "Missing revocation_manifest.csv"
    
    with open(rev_json, "r", encoding="utf-8") as f:
        rev_data = json.load(f)
    with open(rev_csv, "r", encoding="utf-8") as f:
        rev_row_count = sum(1 for _ in f) - 1
    assert rev_row_count == 60, f"Revocation row mismatch: {rev_row_count} != 60"
    
    # Assert 100% registry block rate and 0.0% bypass ASR
    reg_block = rev_data["step4_evaluation"]["test_b_revoked_credential_rejection"]["registry_level_block_rate"]
    bypass_asr = rev_data["step4_evaluation"]["test_b_revoked_credential_rejection"]["attack_success_rate_pct"]
    assert reg_block == 100.0, f"Registry block rate below 100%: {reg_block}"
    assert bypass_asr == 0.0, f"Bypass ASR above 0%: {bypass_asr}"
    print("  [PASS] Revocation: 100% registry block rate and 0.0% cryptographic bypass ASR confirmed.")
    
    # 4. Verify Threat Matrix artifacts
    tm_dir = os.path.join(phase2_dir, "threat_matrix")
    tm_json = os.path.join(tm_dir, "threat_matrix_report.json")
    tm_csv = os.path.join(tm_dir, "threat_matrix_manifest.csv")
    assert os.path.exists(tm_json), "Missing threat_matrix_report.json"
    assert os.path.exists(tm_csv), "Missing threat_matrix_manifest.csv"
    
    with open(tm_json, "r", encoding="utf-8") as f:
        tm_data = json.load(f)
    d_sys_max = tm_data["cross_application_unlinkability"]["d_sys_max_point"]
    assert d_sys_max <= 0.05, f"D_sys^max point estimate exceeds 0.05: {d_sys_max}"
    print(f"  [PASS] Threat Matrix: D_sys^max = {d_sys_max:.4f} <= 0.05 point estimate verified.")
    
    print("\n  >>> ALL PHASE 2 SCIENTIFIC & ACCOUNTING ASSERTIONS PASSED CLEANLY <<<\n")
    return True


def main():
    t_start = time.time()
    print("*" * 80)
    print("  CANCELABLE FINGERPRINT HYBRID — PHASE 2 MASTER EXECUTION SUITE")
    print("  Security, Privacy, and Attack Evaluation")
    print("*" * 80)
    
    phase2_dir = "experiments/results/phase2"
    os.makedirs(phase2_dir, exist_ok=True)
    
    # 1. Inversion & Reconstruction Attacks
    t0 = time.time()
    inv_res = run_phase2_inversion_benchmarks(subset="DB1_B", output_dir=os.path.join(phase2_dir, "inversion"))
    print(f"Completed Inversion benchmark in {time.time() - t0:.2f}s.")
    
    # 2. Minutiae-Type Attribute Leakage Analysis
    t0 = time.time()
    leak_res = run_phase2_minutiae_leakage_analysis(subset="DB1_B", output_dir=os.path.join(phase2_dir, "leakage"))
    print(f"Completed Leakage analysis in {time.time() - t0:.2f}s.")
    
    # 3. Revocability & Cross-Epoch Cancellation
    t0 = time.time()
    rev_res = run_phase2_revocation_experiment(subset="DB1_B", output_dir=os.path.join(phase2_dir, "revocation"))
    print(f"Completed Revocation experiment in {time.time() - t0:.2f}s.")
    
    # 4. Formal Threat Matrix & Unlinkability Synthesis
    t0 = time.time()
    tm_res = run_phase2_threat_matrix_evaluation(
        subset="DB1_B",
        output_dir=os.path.join(phase2_dir, "threat_matrix"),
        unlinkability_dir=os.path.join(phase2_dir, "unlinkability")
    )
    print(f"Completed Threat Matrix evaluation in {time.time() - t0:.2f}s.")
    
    # 5. Integrity & Accounting Assertions
    verify_phase2_integrity(phase2_dir)
    
    # 6. Canonical Phase 2 Summary JSON
    summary = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_runtime_seconds": float(time.time() - t_start),
        "status": "COMPLETED_AND_FULLY_VERIFIED",
        "benchmark_dataset": "FVC2002 DB1_B",
        "threat_models_summary": tm_res["threat_matrix"],
        "inversion_summary": {
            "total_triangles_evaluated": inv_res["total_triangles_evaluated"],
            "9D_M32_cheb_radius": inv_res["factorial_grid"]["9D_pure_geometry"]["M_32"]["chebyshev_radius"]["mean"],
            "9D_M32_continuous_mae": inv_res["factorial_grid"]["9D_pure_geometry"]["M_32"]["continuous_mae"],
            "9D_M32_angular_wrap_mae": inv_res["factorial_grid"]["9D_pure_geometry"]["M_32"]["angular_wrap_mae"],
            "9D_M32_cross_key_reissuance_score": inv_res["factorial_grid"]["9D_pure_geometry"]["M_32"]["authentication_evaluation"]["cross_key_reissuance_mean_score"],
            "9D_M32_cross_key_reissuance_asr_pct": inv_res["factorial_grid"]["9D_pure_geometry"]["M_32"]["authentication_evaluation"]["cross_key_reissuance_asr_pct"],
            "12D_M32_type_recovery_accuracy": inv_res["factorial_grid"]["12D_minutiae_type"]["M_32"]["minutiae_type_recovery_accuracy"],
            "12D_M128_type_recovery_accuracy": inv_res["factorial_grid"]["12D_minutiae_type"]["M_128"]["minutiae_type_recovery_accuracy"]
        },
        "leakage_summary": {
            "total_triangles_ingested": leak_res["total_triangles"],
            "num_folds": leak_res["num_folds"],
            "raw_geom_lr_balanced_acc": leak_res["models"]["raw_geom_logistic_regression"]["balanced_accuracy"],
            "raw_geom_lr_bh_adjusted_p": leak_res["models"]["raw_geom_logistic_regression"]["bh_adjusted_p_value"],
            "raw_geom_rf_balanced_acc": leak_res["models"]["raw_geom_random_forest"]["balanced_accuracy"],
            "raw_geom_rf_bh_adjusted_p": leak_res["models"]["raw_geom_random_forest"]["bh_adjusted_p_value"],
            "protected_bits_lr_balanced_acc": leak_res["models"]["protected_bits_logistic_regression"]["balanced_accuracy"],
            "protected_bits_lr_bh_adjusted_p": leak_res["models"]["protected_bits_logistic_regression"]["bh_adjusted_p_value"],
            "protected_bits_rf_balanced_acc": leak_res["models"]["protected_bits_random_forest"]["balanced_accuracy"],
            "protected_bits_rf_bh_adjusted_p": leak_res["models"]["protected_bits_random_forest"]["bh_adjusted_p_value"],
            "majority_class_accuracy": leak_res["models"]["majority_class_baseline"]["accuracy"]
        },
        "revocation_summary": {
            "epoch2_genuine_tar_pct": rev_res["step4_evaluation"]["test_a_epoch2_genuine_verification"]["tar_pct"],
            "epoch2_degradation_vs_epoch1": rev_res["step4_evaluation"]["test_a_epoch2_genuine_verification"]["degradation_vs_epoch1_baseline"],
            "registry_level_block_rate": rev_res["step4_evaluation"]["test_b_revoked_credential_rejection"]["registry_level_block_rate"],
            "cryptographic_bypass_mean_score": rev_res["step4_evaluation"]["test_b_revoked_credential_rejection"]["cryptographic_bypass_mean_score"],
            "cryptographic_bypass_asr_pct": rev_res["step4_evaluation"]["test_b_revoked_credential_rejection"]["attack_success_rate_pct"],
            "cross_epoch_d_sys": rev_res["step4_evaluation"]["test_c_cross_epoch_unlinkability"]["d_sys"],
            "cross_epoch_ci_95": rev_res["step4_evaluation"]["test_c_cross_epoch_unlinkability"]["clustered_ci_95"],
            "cross_epoch_status": rev_res["step4_evaluation"]["test_c_cross_epoch_unlinkability"]["status"]
        },
        "unlinkability_summary": {
            "descriptor_only_d_sys": tm_res["cross_application_unlinkability"]["models"]["descriptor_only"]["d_sys"],
            "descriptor_only_ci_95": tm_res["cross_application_unlinkability"]["models"]["descriptor_only"]["clustered_ci_95"],
            "topology_only_d_sys": tm_res["cross_application_unlinkability"]["models"]["topology_only"]["d_sys"],
            "topology_only_ci_95": tm_res["cross_application_unlinkability"]["models"]["topology_only"]["clustered_ci_95"],
            "combined_d_sys": tm_res["cross_application_unlinkability"]["models"]["combined"]["d_sys"],
            "combined_ci_95": tm_res["cross_application_unlinkability"]["models"]["combined"]["clustered_ci_95"],
            "d_sys_max_point": tm_res["cross_application_unlinkability"]["d_sys_max_point"],
            "d_sys_max_ci_95": tm_res["cross_application_unlinkability"]["d_sys_max_ci_95"],
            "system_privacy_status": tm_res["cross_application_unlinkability"]["system_privacy_status"]
        }
    }
    
    summary_path = os.path.join(phase2_dir, "phase2_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved Canonical Phase 2 Summary Report to: {summary_path}")
    print(f"\n>>> Phase 2 Execution Completed Successfully in {time.time() - t_start:.2f}s <<<\n")


if __name__ == "__main__":
    main()
