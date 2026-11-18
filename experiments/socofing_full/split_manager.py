"""
SOCOFing Subject-Disjoint Partitioning Service (Phase 3).

Constructs 5 outer subject-disjoint folds:
- Ensures all fingers, hands, and altered impressions of a subject remain in the same split.
- Separates development subjects into disjoint Development-Fit (80%) and Development-Calibration (20%) groups.
- Enforces strict zero-leakage assertions (Gate 3).
- Emits fold_01.json .. fold_05.json and split_audit.json.
"""

import os
import json
from typing import Dict, Any, List, Set
import numpy as np


def create_subject_disjoint_splits(manifest_records: List[Dict[str, Any]],
                                    output_dir: str = "artifacts/socofing/splits",
                                    num_folds: int = 5,
                                    seed: int = 20261109) -> Dict[str, Any]:
    """
    Generates 5 deterministic subject-disjoint outer folds with inner dev-fit/dev-cal separation.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    # 1. Identify all unique subjects
    all_subject_ids = sorted(list({int(r["subject_id"]) for r in manifest_records}))
    total_subjects = len(all_subject_ids)

    if total_subjects < num_folds:
        raise ValueError(f"Total subjects ({total_subjects}) must be >= num_folds ({num_folds})")

    # Shuffle subjects deterministically
    shuffled_subjects = np.array(all_subject_ids)
    rng.shuffle(shuffled_subjects)

    # Partition subjects into num_folds chunks
    fold_splits = np.array_split(shuffled_subjects, num_folds)

    folds_metadata: Dict[str, Any] = {}
    audit_records: List[Dict[str, Any]] = []

    for fold_idx in range(num_folds):
        fold_num = fold_idx + 1
        fold_name = f"fold_{fold_num:02d}"

        # Test subjects for this fold
        test_subjects = set(fold_splits[fold_idx].tolist())

        # Development subjects are all other subjects
        dev_subjects = set(all_subject_ids) - test_subjects

        # Inner split of dev_subjects: 80% fit, 20% calibration
        sorted_dev = sorted(list(dev_subjects))
        dev_rng = np.random.RandomState(seed + fold_num)
        dev_shuffled = np.array(sorted_dev)
        dev_rng.shuffle(dev_shuffled)

        n_dev = len(dev_shuffled)
        n_cal = max(1, int(round(n_dev * 0.20)))
        cal_subjects = set(dev_shuffled[:n_cal].tolist())
        fit_subjects = set(dev_shuffled[n_cal:].tolist())

        # Verify zero leakage (Gate 3 Assertions)
        assert dev_subjects.isdisjoint(test_subjects), f"Leakage: dev and test overlap in {fold_name}"
        assert fit_subjects.isdisjoint(cal_subjects), f"Leakage: fit and cal overlap in {fold_name}"
        assert cal_subjects.isdisjoint(test_subjects), f"Leakage: cal and test overlap in {fold_name}"
        assert fit_subjects.isdisjoint(test_subjects), f"Leakage: fit and test overlap in {fold_name}"
        assert fit_subjects.union(cal_subjects) == dev_subjects, f"Partition error in {fold_name}"

        # Partition manifest files by subject
        test_files = [r["file_id"] for r in manifest_records if int(r["subject_id"]) in test_subjects]
        fit_files = [r["file_id"] for r in manifest_records if int(r["subject_id"]) in fit_subjects]
        cal_files = [r["file_id"] for r in manifest_records if int(r["subject_id"]) in cal_subjects]

        fold_payload = {
            "fold_name": fold_name,
            "seed": seed,
            "subject_counts": {
                "total_subjects": total_subjects,
                "test_subjects_count": len(test_subjects),
                "dev_subjects_count": len(dev_subjects),
                "dev_fit_subjects_count": len(fit_subjects),
                "dev_cal_subjects_count": len(cal_subjects)
            },
            "subjects": {
                "test_subjects": sorted(list(test_subjects)),
                "dev_fit_subjects": sorted(list(fit_subjects)),
                "dev_cal_subjects": sorted(list(cal_subjects))
            },
            "file_counts": {
                "test_files": len(test_files),
                "dev_fit_files": len(fit_files),
                "dev_cal_files": len(cal_files)
            },
            "file_ids": {
                "test_files": test_files,
                "dev_fit_files": fit_files,
                "dev_cal_files": cal_files
            }
        }

        # Save fold json
        fold_file_path = os.path.join(output_dir, f"{fold_name}.json")
        with open(fold_file_path, "w", encoding="utf-8") as f:
            json.dump(fold_payload, f, indent=2)

        folds_metadata[fold_name] = fold_payload
        audit_records.append({
            "fold": fold_name,
            "test_subjects": len(test_subjects),
            "dev_fit_subjects": len(fit_subjects),
            "dev_cal_subjects": len(cal_subjects),
            "test_files": len(test_files),
            "dev_files": len(fit_files) + len(cal_files),
            "leakage_assertion_passed": True
        })

    audit_summary = {
        "num_folds": num_folds,
        "seed": seed,
        "total_subjects": total_subjects,
        "folds": audit_records,
        "zero_leakage_verified": True,
        "gate_3_splits_passed": True
    }

    audit_path = os.path.join(os.path.dirname(output_dir), "split_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    return audit_summary
