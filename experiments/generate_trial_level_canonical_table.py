"""
Canonical Trial-Level Table & Manifest Index Generator (WP0 — Reproducibility Lock).
===================================================================================
Constructs:
1. experiments/results/trial_scores.csv: Unified row-level manifest containing all
   14,755 experimental trials across all 23 benchmark manifests.
2. experiments/results/manifest_index.csv & manifest_index.json: Canonical 23-row
   manifest index documenting sensor provenance, partitions, counts, and checksums.
"""

import os
import sys
import csv
import json
import hashlib
from typing import Dict, Any, List, Tuple
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.metrics import compute_roc_metrics

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "results")


def sha256_file(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "NOT_FOUND"
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def generate_trial_level_canonical_table() -> Tuple[str, str, str]:
    trial_scores_csv = os.path.join(RESULTS_DIR, "trial_scores.csv")
    manifest_index_csv = os.path.join(RESULTS_DIR, "manifest_index.csv")
    manifest_index_json = os.path.join(RESULTS_DIR, "manifest_index.json")

    trials: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    trial_counter = 0
    manifest_id = 0

    def add_manifest_batch(
        m_name: str,
        db_family: str,
        sensor_tech: str,
        partition_type: str,
        representation: str,
        topology: str,
        pairs: List[Dict[str, Any]],
        g_scores: np.ndarray,
        i_scores: np.ndarray,
        threshold: float,
        threshold_version: str,
        manifest_path: str,
        score_paths: List[str]
    ):
        nonlocal trial_counter, manifest_id
        manifest_id += 1
        n_gen = len(g_scores)
        n_imp = len(i_scores)
        assert len(pairs) == n_gen + n_imp, f"Mismatch in {m_name}: {len(pairs)} != {n_gen}+{n_imp}"

        tp = int(np.sum(g_scores >= threshold))
        fn = int(np.sum(g_scores < threshold))
        fp = int(np.sum(i_scores >= threshold))
        tn = int(np.sum(i_scores < threshold))

        assert tp + fn == n_gen
        assert fp + tn == n_imp

        roc = compute_roc_metrics(g_scores, i_scores, num_thresholds=500, warn_on_inversion=False)

        manifest_digest = sha256_file(manifest_path)
        score_digests = [sha256_file(p) for p in score_paths]

        manifest_rows.append({
            "manifest_id": manifest_id,
            "manifest_name": m_name,
            "database_family": db_family,
            "sensor_technology": sensor_tech,
            "partition_type": partition_type,
            "representation": representation,
            "topology": topology,
            "n_genuine": n_gen,
            "n_impostor": n_imp,
            "total_trials": n_gen + n_imp,
            "eval_threshold": threshold,
            "threshold_version": threshold_version,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "tar_pct": round(tp / max(n_gen, 1) * 100.0, 2),
            "far_pct": round(fp / max(n_imp, 1) * 100.0, 2),
            "eer_pct": round(roc["eer"] * 100.0, 2),
            "auc": round(roc["auc"], 4),
            "manifest_sha256": manifest_digest,
            "scores_sha256": ";".join(score_digests)
        })

        # Genuine trials
        for idx in range(n_gen):
            trial_counter += 1
            p = pairs[idx]
            s = float(g_scores[idx])
            dec = "ACCEPT" if s >= threshold else "REJECT"
            outcome = "TP" if s >= threshold else "FN"
            trials.append({
                "trial_id": f"trial_{trial_counter:05d}",
                "manifest_id": manifest_id,
                "manifest_name": m_name,
                "database_family": db_family,
                "sensor_technology": sensor_tech,
                "partition_type": partition_type,
                "subject_id": p.get("probe_finger_id", p.get("subject_id")),
                "gallery_subject_id": p.get("gallery_finger_id", p.get("gallery_subject_id")),
                "probe_id": os.path.basename(p.get("probe_path", "")),
                "gallery_id": os.path.basename(p.get("gallery_path", "")),
                "is_genuine": 1,
                "score": round(s, 6),
                "threshold": threshold,
                "threshold_version": threshold_version,
                "decision": dec,
                "trial_outcome": outcome
            })

        # Impostor trials
        for idx in range(n_imp):
            trial_counter += 1
            p = pairs[n_gen + idx]
            s = float(i_scores[idx])
            dec = "ACCEPT" if s >= threshold else "REJECT"
            outcome = "FP" if s >= threshold else "TN"
            trials.append({
                "trial_id": f"trial_{trial_counter:05d}",
                "manifest_id": manifest_id,
                "manifest_name": m_name,
                "database_family": db_family,
                "sensor_technology": sensor_tech,
                "partition_type": partition_type,
                "subject_id": p.get("probe_finger_id", p.get("subject_id")),
                "gallery_subject_id": p.get("gallery_finger_id", p.get("gallery_subject_id")),
                "probe_id": os.path.basename(p.get("probe_path", "")),
                "gallery_id": os.path.basename(p.get("gallery_path", "")),
                "is_genuine": 0,
                "score": round(s, 6),
                "threshold": threshold,
                "threshold_version": threshold_version,
                "decision": dec,
                "trial_outcome": outcome
            })

    # =========================================================================
    # 1. DB1_A Primary Benchmark (Held-out Test Partition Set A)
    # =========================================================================
    db1_a_dir = os.path.join(RESULTS_DIR, "fvc2002", "DB1_A_9D_delaunay")
    db1_a_m = os.path.join(db1_a_dir, "manifest.csv")
    with open(db1_a_m, "r", encoding="utf-8") as f:
        db1_a_pairs = list(csv.DictReader(f))
    g_db1a = np.load(os.path.join(db1_a_dir, "prot_genuine_scores.npy"))
    i_db1a = np.load(os.path.join(db1_a_dir, "prot_impostor_scores.npy"))

    add_manifest_batch(
        m_name="fvc2002/DB1_A_9D_delaunay/prot",
        db_family="FVC2002_DB1",
        sensor_tech="Optical Sensor (Identix Secure Desktop)",
        partition_type="held_out_test_set_a",
        representation="9-D Pure Geometric Keyed Binary (32b)",
        topology="delaunay",
        pairs=db1_a_pairs,
        g_scores=g_db1a,
        i_scores=i_db1a,
        threshold=0.3751,
        threshold_version="frozen_dev_set_b_eer_0.3751",
        manifest_path=db1_a_m,
        score_paths=[
            os.path.join(db1_a_dir, "prot_genuine_scores.npy"),
            os.path.join(db1_a_dir, "prot_impostor_scores.npy")
        ]
    )

    # =========================================================================
    # 2. All FVC2002 Set B Subsets (Development Partitions Set B)
    # 16 manifests across 8 folders x 2 (prot & unprot)
    # =========================================================================
    set_b_dirs = [
        ("DB1_B_12D_delaunay", "FVC2002_DB1", "Optical Sensor (Identix Secure Desktop)", "12-D Minutiae-Type", "delaunay"),
        ("DB1_B_9D_delaunay",  "FVC2002_DB1", "Optical Sensor (Identix Secure Desktop)", "9-D Pure Geometric", "delaunay"),
        ("DB1_B_9D_knn",       "FVC2002_DB1", "Optical Sensor (Identix Secure Desktop)", "9-D Pure Geometric", "knn_k5"),
        ("DB2_B_12D_delaunay", "FVC2002_DB2", "Optical Sensor (Fingerprint Cards)",     "12-D Minutiae-Type", "delaunay"),
        ("DB2_B_9D_delaunay",  "FVC2002_DB2", "Optical Sensor (Fingerprint Cards)",     "9-D Pure Geometric", "delaunay"),
        ("DB2_B_9D_knn",       "FVC2002_DB2", "Optical Sensor (Fingerprint Cards)",     "9-D Pure Geometric", "knn_k5"),
        ("DB3_B_9D_delaunay",  "FVC2002_DB3", "Capacitive Sensor (100 SC)",            "9-D Pure Geometric", "delaunay"),
        ("DB4_B_9D_delaunay",  "FVC2002_DB4", "Synthetic Generator (SFinGe v2.51)",    "9-D Pure Geometric", "delaunay")
    ]

    for folder_name, db_family, sensor_tech, repr_name, topo_name in set_b_dirs:
        d = os.path.join(RESULTS_DIR, "fvc2002", folder_name)
        m_file = os.path.join(d, "manifest.csv")
        with open(m_file, "r", encoding="utf-8") as f:
            pairs = list(csv.DictReader(f))

        # (a) Protected
        g_p = np.load(os.path.join(d, "prot_genuine_scores.npy"))
        i_p = np.load(os.path.join(d, "prot_impostor_scores.npy"))
        add_manifest_batch(
            m_name=f"fvc2002/{folder_name}/prot",
            db_family=db_family,
            sensor_tech=sensor_tech,
            partition_type="development_set_b",
            representation=f"{repr_name} Keyed Binary (32b)",
            topology=topo_name,
            pairs=pairs,
            g_scores=g_p,
            i_scores=i_p,
            threshold=0.3751,
            threshold_version="dev_set_b_eer_0.3751",
            manifest_path=m_file,
            score_paths=[
                os.path.join(d, "prot_genuine_scores.npy"),
                os.path.join(d, "prot_impostor_scores.npy")
            ]
        )

        # (b) Unprotected continuous baseline
        g_u = np.load(os.path.join(d, "unprot_genuine_scores.npy"))
        i_u = np.load(os.path.join(d, "unprot_impostor_scores.npy"))
        add_manifest_batch(
            m_name=f"fvc2002/{folder_name}/unprot",
            db_family=db_family,
            sensor_tech=sensor_tech,
            partition_type="development_set_b",
            representation=f"{repr_name} Continuous Standardized",
            topology=topo_name,
            pairs=pairs,
            g_scores=g_u,
            i_scores=i_u,
            threshold=0.3800,
            threshold_version="continuous_baseline_0.38",
            manifest_path=m_file,
            score_paths=[
                os.path.join(d, "unprot_genuine_scores.npy"),
                os.path.join(d, "unprot_impostor_scores.npy")
            ]
        )

    # =========================================================================
    # 3. Cyclic Invariance Ablation Benchmarks (5 manifests)
    # =========================================================================
    cyclic_m_file = os.path.join(RESULTS_DIR, "cyclic_ablation", "manifest.csv")
    with open(cyclic_m_file, "r", encoding="utf-8") as f:
        cyc_pairs = list(csv.DictReader(f))

    cyclic_exps = [
        ("Exp_A_Delaunay_Canonical", "Delaunay Canonical", "delaunay"),
        ("Exp_B_Delaunay_Cyclic",    "Delaunay Cyclic C3", "delaunay_cyclic"),
        ("Exp_C_kNN_Canonical",      "kNN Canonical",      "knn_k5"),
        ("Exp_D_kNN_Cyclic",          "kNN Cyclic C3",      "knn_cyclic"),
        ("Exp_E_Delaunay_Cyclic_Consensus", "Delaunay Cyclic + Consensus", "delaunay_consensus")
    ]

    for exp_folder, repr_label, topo_label in cyclic_exps:
        d = os.path.join(RESULTS_DIR, "cyclic_ablation", exp_folder)
        g_c = np.load(os.path.join(d, "genuine_scores.npy"))
        i_c = np.load(os.path.join(d, "impostor_scores.npy"))
        add_manifest_batch(
            m_name=f"cyclic_ablation/{exp_folder}",
            db_family="FVC2002_DB1",
            sensor_tech="Optical Sensor (Identix Secure Desktop)",
            partition_type="development_set_b",
            representation=f"9-D {repr_label}",
            topology=topo_label,
            pairs=cyc_pairs,
            g_scores=g_c,
            i_scores=i_c,
            threshold=0.3751,
            threshold_version="dev_set_b_eer_0.3751",
            manifest_path=cyclic_m_file,
            score_paths=[
                os.path.join(d, "genuine_scores.npy"),
                os.path.join(d, "impostor_scores.npy")
            ]
        )

    # =========================================================================
    # 4. SOCOFing Alteration Held-Out Test Benchmark (1 manifest)
    # =========================================================================
    soco_m_file = os.path.join(RESULTS_DIR, "socofing", "socofing_test_manifest_9D.csv")
    with open(soco_m_file, "r", encoding="utf-8") as f:
        soco_pairs = list(csv.DictReader(f))
    g_soco = np.load(os.path.join(RESULTS_DIR, "socofing", "socofing_test_gen_scores_9D.npy"))
    i_soco = np.load(os.path.join(RESULTS_DIR, "socofing", "socofing_test_imp_scores_9D.npy"))

    add_manifest_batch(
        m_name="socofing/socofing_test_manifest_9D",
        db_family="SOCOFing_synthetic",
        sensor_tech="Synthetic Digital Alterations (Obliteration, Z-Cut, Rotation)",
        partition_type="held_out_test_fold1",
        representation="9-D Pure Geometric Keyed Binary (32b)",
        topology="delaunay",
        pairs=soco_pairs,
        g_scores=g_soco,
        i_scores=i_soco,
        threshold=0.3891,
        threshold_version="socofing_calibrated_t_accept_0.3891",
        manifest_path=soco_m_file,
        score_paths=[
            os.path.join(RESULTS_DIR, "socofing", "socofing_test_gen_scores_9D.npy"),
            os.path.join(RESULTS_DIR, "socofing", "socofing_test_imp_scores_9D.npy")
        ]
    )

    # Assertions
    assert manifest_id == 23, f"Expected 23 manifests, got {manifest_id}"
    assert trial_counter == 14755, f"Expected 14,755 trials, got {trial_counter}"

    total_gen = sum(m["n_genuine"] for m in manifest_rows)
    total_imp = sum(m["n_impostor"] for m in manifest_rows)
    total_tp = sum(m["tp"] for m in manifest_rows)
    total_fn = sum(m["fn"] for m in manifest_rows)
    total_fp = sum(m["fp"] for m in manifest_rows)
    total_tn = sum(m["tn"] for m in manifest_rows)

    assert total_gen == 8725, f"Expected 8,725 genuine trials, got {total_gen}"
    assert total_imp == 6030, f"Expected 6,030 impostor trials, got {total_imp}"
    assert total_tp + total_fn == total_gen, "Genuine identity violated"
    assert total_fp + total_tn == total_imp, "Impostor identity violated"

    # Write trial_scores.csv
    print(f"Writing {trial_scores_csv} ({len(trials)} rows)...")
    fieldnames = [
        "trial_id", "manifest_id", "manifest_name", "database_family", "sensor_technology",
        "partition_type", "subject_id", "gallery_subject_id", "probe_id", "gallery_id",
        "is_genuine", "score", "threshold", "threshold_version", "decision", "trial_outcome"
    ]
    with open(trial_scores_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)

    # Write manifest_index.csv
    print(f"Writing {manifest_index_csv} ({len(manifest_rows)} rows)...")
    m_fieldnames = list(manifest_rows[0].keys())
    with open(manifest_index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=m_fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Write manifest_index.json
    manifest_index_payload = {
        "schema_version": "2.0.0",
        "title": "Cancelable Fingerprint Hybrid: Canonical 23-Manifest Benchmark Index",
        "total_manifests": 23,
        "total_trials_accounted": trial_counter,
        "total_genuine_trials": total_gen,
        "total_impostor_trials": total_imp,
        "confusion_totals": {
            "tp": total_tp, "fn": total_fn, "fp": total_fp, "tn": total_tn
        },
        "identities_satisfied": True,
        "manifests": manifest_rows
    }
    with open(manifest_index_json, "w", encoding="utf-8") as f:
        json.dump(manifest_index_payload, f, indent=2)

    print(f"[OK] WP0 Canonical Table and Manifest Index generated successfully.")
    print(f"     Total Manifests: {manifest_id}")
    print(f"     Total Trials:    {trial_counter:,} (Genuine: {total_gen:,}, Impostor: {total_imp:,})")
    print(f"     Identities:      TP({total_tp}) + FN({total_fn}) == {total_gen} | FP({total_fp}) + TN({total_tn}) == {total_imp}")

    return trial_scores_csv, manifest_index_csv, manifest_index_json


if __name__ == "__main__":
    generate_trial_level_canonical_table()
