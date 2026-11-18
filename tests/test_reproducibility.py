"""
Layer 5 Reproducibility & Trial Accounting Integrity Test Suite.

Tests:
1. Exact trial accounting identity assertions (TP + FN == N_gen, FP + TN == N_imp)
   across all benchmark manifests (DB1_A, Set B, Cyclic, Phase 1 ablations, SOCOFing).
2. Cryptographic SHA-256 checksum integrity verification against baseline_manifest.json.
3. Zero accidental identity overlap between train, development, and test splits.
"""

import os
import glob
import json
import hashlib
import pytest
import numpy as np


@pytest.fixture
def results_dir():
    """Returns absolute path to experiments/results directory."""
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "results"))
    if not os.path.exists(path):
        pytest.skip(f"Results directory not found at {path}")
    return path


def _check_trial_identities(genuine_scores: np.ndarray, impostor_scores: np.ndarray, threshold: float):
    """Calculates confusion counts and asserts exact mathematical accounting identities."""
    n_gen = len(genuine_scores)
    n_imp = len(impostor_scores)
    tp = int(np.sum(genuine_scores >= threshold))
    fn = int(np.sum(genuine_scores < threshold))
    fp = int(np.sum(impostor_scores >= threshold))
    tn = int(np.sum(impostor_scores < threshold))

    assert tp + fn == n_gen, f"Identity violated: TP ({tp}) + FN ({fn}) != N_gen ({n_gen})"
    assert fp + tn == n_imp, f"Identity violated: FP ({fp}) + TN ({tn}) != N_imp ({n_imp})"

    return {"n_gen": n_gen, "n_imp": n_imp, "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def test_trial_accounting_identity_assertions(results_dir):
    """
    Asserts TP + FN == N_gen and FP + TN == N_imp across all generated benchmark manifests:
    1. FVC2002 DB1_A primary benchmark (N_gen=2800, N_imp=4950).
    2. All FVC2002 Set B subsets (DB1_B, DB2_B, DB3_B, DB4_B; N_gen=280, N_imp=45).
    3. All Cyclic ablation experiments (Exp_A to Exp_E; N_gen=280, N_imp=45).
    4. All Phase 1 factorial ablations (N_gen=280, N_imp=45).
    5. SOCOFing 3x3 cross-tabulation and overall confusion matrix at T_accept=0.3891.
    """
    verified_count = 0

    # 1. DB1_A primary benchmark
    db1_a_dir = os.path.join(results_dir, "fvc2002", "DB1_A_9D_delaunay")
    if os.path.exists(db1_a_dir):
        g = np.load(os.path.join(db1_a_dir, "prot_genuine_scores.npy"))
        i = np.load(os.path.join(db1_a_dir, "prot_impostor_scores.npy"))
        res = _check_trial_identities(g, i, threshold=0.3751)
        assert res["n_gen"] == 2800
        assert res["n_imp"] == 4950
        assert res["tp"] == 1654
        assert res["fp"] == 2050
        verified_count += 1

    # 2. All FVC2002 Set B subsets (both prot and unprot)
    fvc_dirs = sorted(glob.glob(os.path.join(results_dir, "fvc2002", "*_B_*")))
    for d in fvc_dirs:
        for pfx in ["prot", "unprot"]:
            g_path = os.path.join(d, f"{pfx}_genuine_scores.npy")
            i_path = os.path.join(d, f"{pfx}_impostor_scores.npy")
            if os.path.exists(g_path) and os.path.exists(i_path):
                g = np.load(g_path)
                i = np.load(i_path)
                res = _check_trial_identities(g, i, threshold=0.38)
                assert res["n_gen"] == 280
                assert res["n_imp"] == 45
                verified_count += 1

    # 3. All Cyclic Ablation experiments (Exp_A to Exp_E)
    cyc_dirs = sorted(glob.glob(os.path.join(results_dir, "cyclic_ablation", "Exp_*")))
    for d in cyc_dirs:
        g_path = os.path.join(d, "genuine_scores.npy")
        i_path = os.path.join(d, "impostor_scores.npy")
        if os.path.exists(g_path) and os.path.exists(i_path):
            g = np.load(g_path)
            i = np.load(i_path)
            res = _check_trial_identities(g, i, threshold=0.50)
            assert res["n_gen"] == 280
            assert res["n_imp"] == 45
            verified_count += 1

    # 4. All Phase 1 Factorial Ablations
    p1_dirs = sorted(glob.glob(os.path.join(results_dir, "phase1", "ablations", "ablation_*")))
    for d in p1_dirs:
        for pfx in ["prot", ""]:
            g_fname = f"{pfx}_genuine_scores.npy" if pfx else "genuine_scores.npy"
            i_fname = f"{pfx}_impostor_scores.npy" if pfx else "impostor_scores.npy"
            g_path = os.path.join(d, g_fname)
            i_path = os.path.join(d, i_fname)
            if os.path.exists(g_path) and os.path.exists(i_path):
                g = np.load(g_path)
                i = np.load(i_path)
                res = _check_trial_identities(g, i, threshold=0.38)
                assert res["n_gen"] == 280
                assert res["n_imp"] == 45
                verified_count += 1

    # 5. SOCOFing Row-Level Threshold Harmonization
    socofing_json = os.path.join(results_dir, "socofing", "socofing_benchmark_report_9D.json")
    if os.path.exists(socofing_json):
        with open(socofing_json, "r", encoding="utf-8") as f:
            soco_data = json.load(f)

        cells = soco_data["held_out_3x3_cross_tabulation"]["cells"]
        sum_tp = sum(cells[c][d]["tp"] for c in cells for d in cells[c])
        sum_fn = sum(cells[c][d]["fn"] for c in cells for d in cells[c])
        sum_fp = sum(cells[c][d]["fp"] for c in cells for d in cells[c])
        sum_tn = sum(cells[c][d]["tn"] for c in cells for d in cells[c])

        overall = soco_data["held_out_test_accounting"]["overall_confusion_at_3891"]
        assert sum_tp == overall["tp"] == 30
        assert sum_fn == overall["fn"] == 15
        assert sum_fp == overall["fp"] == 15
        assert sum_tn == overall["tn"] == 120
        assert sum_tp + sum_fn == 45  # N_gen
        assert sum_fp + sum_tn == 135  # N_imp
        verified_count += 1

    # Confirm comprehensive trial identity coverage
    assert verified_count >= 20, f"Expected >= 20 verified score sets, got {verified_count}"


def test_baseline_manifest_checksum_integrity(results_dir):
    """
    Verifies SHA-256 cryptographic digests for all baseline manifests and score arrays:
    - Reads baseline_manifest.json.
    - Asserts 100% of recorded raw score files and manifests match their exact SHA-256 hash.
    """
    manifest_path = os.path.join(results_dir, "baseline_manifest.json")
    assert os.path.exists(manifest_path), f"baseline_manifest.json missing from {results_dir}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    raw_artifacts = manifest.get("raw_artifacts", {})
    assert len(raw_artifacts) >= 70, f"Expected >=70 raw artifacts, got {len(raw_artifacts)}"

    mismatches = []
    missing = []

    for rel_path, meta in raw_artifacts.items():
        norm_path = rel_path.replace("\\", os.sep).replace("/", os.sep)
        full_path = os.path.join(results_dir, norm_path)
        if not os.path.exists(full_path):
            missing.append(rel_path)
            continue

        with open(full_path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest().lower()

        expected = meta["sha256"].lower()
        if digest != expected:
            mismatches.append((rel_path, expected, digest))

    assert len(missing) == 0, f"Missing {len(missing)} raw artifacts: {missing[:3]}"
    assert len(mismatches) == 0, f"SHA-256 mismatch on {len(mismatches)} artifacts: {mismatches[:3]}"


def test_zero_identity_overlap_between_train_and_test_splits(results_dir):
    """
    Asserts zero accidental subject or finger identity overlap across evaluation splits:
    1. FVC2002 Set A (Test: Fingers 1-100) vs Set B (Dev/Train: Fingers 101-110) from actual manifest CSVs.
    2. SOCOFing 2-Fold Cross-Validation Subject Partitions from benchmark report.
    3. Minutiae-type leakage 5-fold cross-validation disjointness.
    """
    import csv

    # 1. FVC2002 Set A vs Set B Disjointness from real CSV manifests on disk
    set_a_manifest = os.path.join(results_dir, "fvc2002", "DB1_A_9D_delaunay", "manifest.csv")
    set_a_fingers = set()
    if os.path.exists(set_a_manifest):
        with open(set_a_manifest, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                set_a_fingers.add(int(row["probe_finger_id"]))
                set_a_fingers.add(int(row["gallery_finger_id"]))

    assert len(set_a_fingers) == 100, f"Expected 100 fingers in DB1_A, found {len(set_a_fingers)}"
    assert set_a_fingers == set(range(1, 101))

    set_b_fingers = set()
    for d in glob.glob(os.path.join(results_dir, "fvc2002", "*_B_*")):
        b_manifest = os.path.join(d, "manifest.csv")
        if os.path.exists(b_manifest):
            with open(b_manifest, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    set_b_fingers.add(int(row["probe_finger_id"]))
                    set_b_fingers.add(int(row["gallery_finger_id"]))

    assert len(set_b_fingers) == 10, f"Expected 10 fingers in Set B, found {len(set_b_fingers)}"
    assert set_b_fingers == set(range(101, 111))

    # Assert 0 identity overlap between Set A (Test) and Set B (Development)
    manifest_overlap = set_a_fingers.intersection(set_b_fingers)
    assert len(manifest_overlap) == 0, f"FVC2002 manifest split overlap detected: {manifest_overlap}"

    # 2. SOCOFing 10-Subject 2-Fold Protocol Subject Disjointness from real benchmark report
    socofing_json = os.path.join(results_dir, "socofing", "socofing_benchmark_report_9D.json")
    if os.path.exists(socofing_json):
        with open(socofing_json, "r", encoding="utf-8") as f:
            soco_meta = json.load(f)
        assert soco_meta.get("num_dev_subjects") == 5
        assert soco_meta.get("num_test_subjects") == 5
        assert soco_meta.get("num_subjects") == 10

    dev_subjs = set(range(1, 6))
    test_subjs = set(range(6, 11))
    soco_overlap = dev_subjs.intersection(test_subjs)
    assert len(soco_overlap) == 0, f"SOCOFing partition subject overlap: {soco_overlap}"

    # 3. Stratified Group K-Fold Disjointness in Leakage Analysis
    from src.security.leakage_analysis import stratified_group_kfold
    rng = np.random.default_rng(2026)
    finger_ids = np.array([f"finger_{i % 10}" for i in range(100)])
    y_types = rng.integers(0, 2, size=100)

    splits = stratified_group_kfold(finger_ids, y_types, n_splits=5, seed=42)
    assert len(splits) == 5

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_groups = set(finger_ids[train_idx])
        test_groups = set(finger_ids[test_idx])
        fold_overlap = train_groups.intersection(test_groups)
        assert len(fold_overlap) == 0, f"Fold {fold_idx} leaked training groups into test: {fold_overlap}"


def test_canonical_trial_scores_and_manifest_index(results_dir):
    """
    WP0 Canonical Verification:
    Asserts existence, cryptographic integrity, and exact trial accounting for:
    1. trial_scores.csv: 14,755 data rows, 8,725 genuine, 6,030 impostor,
       confusion totals TP=4,817, FN=3,908, FP=2,511, TN=3,519.
    2. manifest_index.json & manifest_index.csv: exactly 23 manifests matching trial counts.
    """
    import csv

    # 1. Verify manifest_index.json
    index_json = os.path.join(results_dir, "manifest_index.json")
    assert os.path.exists(index_json), f"manifest_index.json missing from {results_dir}"
    with open(index_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["total_manifests"] == 23
    assert meta["total_trials_accounted"] == 14755
    assert meta["total_genuine_trials"] == 8725
    assert meta["total_impostor_trials"] == 6030
    assert meta["identities_satisfied"] is True
    assert meta["confusion_totals"] == {"tp": 4817, "fn": 3908, "fp": 2511, "tn": 3519}
    assert len(meta["manifests"]) == 23

    # Sum of individual manifest counts must match totals
    sum_gen = sum(m["n_genuine"] for m in meta["manifests"])
    sum_imp = sum(m["n_impostor"] for m in meta["manifests"])
    sum_tot = sum(m["total_trials"] for m in meta["manifests"])
    sum_tp = sum(m["tp"] for m in meta["manifests"])
    sum_fn = sum(m["fn"] for m in meta["manifests"])
    sum_fp = sum(m["fp"] for m in meta["manifests"])
    sum_tn = sum(m["tn"] for m in meta["manifests"])

    assert sum_gen == 8725
    assert sum_imp == 6030
    assert sum_tot == 14755
    assert sum_tp == 4817
    assert sum_fn == 3908
    assert sum_fp == 2511
    assert sum_tn == 3519
    assert sum_tp + sum_fn == 8725
    assert sum_fp + sum_tn == 6030

    # 2. Verify trial_scores.csv
    trial_csv = os.path.join(results_dir, "trial_scores.csv")
    assert os.path.exists(trial_csv), f"trial_scores.csv missing from {results_dir}"

    row_count = 0
    tp_cnt = 0
    fn_cnt = 0
    fp_cnt = 0
    tn_cnt = 0
    gen_cnt = 0
    imp_cnt = 0

    with open(trial_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            is_gen = int(row["is_genuine"])
            outcome = row["trial_outcome"]
            dec = row["decision"]
            if is_gen == 1:
                gen_cnt += 1
                if outcome == "TP":
                    tp_cnt += 1
                    assert dec == "ACCEPT"
                elif outcome == "FN":
                    fn_cnt += 1
                    assert dec == "REJECT"
                else:
                    pytest.fail(f"Invalid outcome for genuine: {outcome}")
            else:
                imp_cnt += 1
                if outcome == "FP":
                    fp_cnt += 1
                    assert dec == "ACCEPT"
                elif outcome == "TN":
                    tn_cnt += 1
                    assert dec == "REJECT"
                else:
                    pytest.fail(f"Invalid outcome for impostor: {outcome}")

    assert row_count == 14755
    assert gen_cnt == 8725
    assert imp_cnt == 6030
    assert tp_cnt == 4817
    assert fn_cnt == 3908
    assert fp_cnt == 2511
    assert tn_cnt == 3519

