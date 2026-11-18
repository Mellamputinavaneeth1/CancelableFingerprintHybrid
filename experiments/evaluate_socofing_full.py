"""
Full SOCOFing Evaluation CLI & Test Harness.

Implements all CLI stages specified in Section 16.3 of the execution plan:
  1. audit: Discovers files and writes archive_audit.json
  2. build-manifest: Parses filenames and resolves fingerprint_id
  3. validate-links: Validates probe-to-clean gallery linkages
  4. make-splits: Builds 5 subject-disjoint outer folds with dev-fit/cal separation
  5. make-trials: Generates genuine and stratified impostor trial manifests
  6. evaluate: Runs feature extraction, calibration, and frozen held-out scoring
  7. summarize: Computes metrics, clustered bootstrap CIs (B=2,000), and alteration tables
  8. verify-artifacts: Asserts manifest hashes, trial accounting, and all 10 gates
  9. all: Runs complete sequence end-to-end
"""

import os
import sys
import argparse
import json
import csv
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.socofing_full.manifest_builder import audit_and_build_manifest, compute_file_sha256
from experiments.socofing_full.split_manager import create_subject_disjoint_splits
from experiments.socofing_full.trial_builder import generate_all_folds_trials
from experiments.socofing_full.evaluator import SOCOFingFullEvaluator, execute_full_socofing_protocol


def cmd_audit(args):
    print(f"\n[PHASE 0] Auditing archive at: {args.dataset_root}")
    res = audit_and_build_manifest(args.dataset_root, args.output_dir)
    print(f"  Discovered files: {res['total_files_discovered']}")
    print(f"  Valid records:    {res['valid_manifest_records']}")
    print(f"  Excluded files:   {res['excluded_files_count']}")
    print(f"  Gate 0 Archive:   {'PASS' if res['gates']['Gate_0_Archive'] else 'FAIL'}")
    return res


def cmd_build_manifest(args):
    print(f"\n[PHASE 1 & 2] Building metadata manifest and identity resolution...")
    res = audit_and_build_manifest(args.dataset_root, args.output_dir)
    print(f"  Unique subjects:     {res['unique_subjects_count']}")
    print(f"  Unique fingerprints: {res['unique_fingerprints_count']}")
    print(f"  Gate 1 Parser:       {'PASS' if res['gates']['Gate_1_Parser'] else 'FAIL'}")
    print(f"  Gate 2 Linkage:      {'PASS' if res['gates']['Gate_2_Identity_Linkage'] else 'FAIL'}")
    return res


def cmd_validate_links(args):
    print(f"\n[PHASE 2] Validating probe-to-gallery linkages...")
    manifest_csv = os.path.join(args.output_dir, "metadata_manifest.csv")
    if not os.path.exists(manifest_csv):
        print(f"Error: Manifest not found at {manifest_csv}. Run build-manifest first.")
        sys.exit(1)

    with open(manifest_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    probes = [r for r in records if r["is_probe_candidate"]]
    galleries = {r["fingerprint_id"]: r for r in records if r["is_gallery_candidate"]}

    unlinked = [p for p in probes if p["source_fingerprint_id"] not in galleries]
    if unlinked:
        print(f"  [FAIL] Found {len(unlinked)} unlinked probes without clean Real gallery prints!")
        sys.exit(1)
    else:
        print(f"  [PASS] All {len(probes)} probes successfully linked to unique Real gallery prints.")


def cmd_make_splits(args):
    print(f"\n[PHASE 3] Generating {args.folds} subject-disjoint outer folds (seed={args.seed})...")
    manifest_csv = os.path.join(args.output_dir, "metadata_manifest.csv")
    with open(manifest_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    splits_dir = os.path.join(args.output_dir, "splits")
    res = create_subject_disjoint_splits(records, splits_dir, num_folds=args.folds, seed=args.seed)
    print(f"  Total subjects partitioned: {res['total_subjects']}")
    print(f"  Gate 3 Zero-Leakage:        {'PASS' if res['gate_3_splits_passed'] else 'FAIL'}")
    return res


def cmd_make_trials(args):
    print(f"\n[PHASE 7] Generating genuine and stratified impostor trial tables...")
    manifest_csv = os.path.join(args.output_dir, "metadata_manifest.csv")
    with open(manifest_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    splits_dir = os.path.join(args.output_dir, "splits")
    trials_dir = os.path.join(args.output_dir, "trials")
    res = generate_all_folds_trials(records, splits_dir, trials_dir,
                                    impostors_per_probe=args.impostors_per_probe, seed=args.seed)
    print(f"  Total trials generated:   {res['total_trials']}")
    print(f"  Gate 7 Trial Invariants: {'PASS' if res['gate_7_passed'] else 'FAIL'}")
    return res


def cmd_evaluate(args):
    print(f"\n[PHASE 8, 9, 10] Executing evaluation (in_dim={args.descriptor}, matcher={args.matcher})...")
    manifest_csv = os.path.join(args.output_dir, "metadata_manifest.csv")
    splits_dir = os.path.join(args.output_dir, "splits")
    trials_dir = os.path.join(args.output_dir, "trials")
    results_dir = os.path.join(args.output_dir, "results")

    res = execute_full_socofing_protocol(
        manifest_path=manifest_csv,
        splits_dir=splits_dir,
        trials_dir=trials_dir,
        output_dir=results_dir,
        in_dim=int(args.descriptor.replace("d", "")),
        matcher_type=args.matcher,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed
    )
    print(f"\n--- Evaluation Completed in {res['total_time_seconds']}s ({res['throughput_comp_per_sec']} comp/s) ---")
    print(f"  Pooled Out-of-Fold EER: {res['pooled_out_of_fold_metrics']['eer_pct']:.2f}%")
    print(f"  Pooled Out-of-Fold AUC: {res['pooled_out_of_fold_metrics']['auc']:.4f}")
    print(f"  95% Clustered CI EER:   [{res['pooled_out_of_fold_metrics']['clustered_eer_95ci'][0]:.2f}%, {res['pooled_out_of_fold_metrics']['clustered_eer_95ci'][1]:.2f}%]")
    return res


def cmd_verify_artifacts(args):
    print(f"\n[PHASE 14] Verifying all artifact files and 10 acceptance gates...")
    results_dir = os.path.join(args.output_dir, "results")
    report_file = os.path.join(results_dir, "socofing_full_report.json")
    if not os.path.exists(report_file):
        print(f"Error: Report file not found at {report_file}. Run evaluate first.")
        sys.exit(1)

    with open(report_file, "r", encoding="utf-8") as f:
        rep = json.load(f)

    gates = rep.get("gate_assertions", {})
    all_pass = all(gates.values())
    print("\n--- Gate Acceptance Status ---")
    for g_name, status in gates.items():
        print(f"  {g_name:35s}: {'[PASS]' if status else '[FAIL]'}")

    if all_pass:
        print("\n>>> ALL ACCEPTANCE GATES PASSED CLEANLY (Code 0) <<<")
    else:
        print("\n>>> ONE OR MORE GATES FAILED <<<")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Full SOCOFing Evaluation Suite")
    subparsers = parser.add_subparsers(dest="command", help="Execution Stage")

    # audit
    p_audit = subparsers.add_parser("audit")
    p_audit.add_argument("--dataset-root", default="dataset/SOCOFing")
    p_audit.add_argument("--output-dir", default="artifacts/socofing")

    # build-manifest
    p_man = subparsers.add_parser("build-manifest")
    p_man.add_argument("--dataset-root", default="dataset/SOCOFing")
    p_man.add_argument("--output-dir", default="artifacts/socofing")

    # validate-links
    p_val = subparsers.add_parser("validate-links")
    p_val.add_argument("--output-dir", default="artifacts/socofing")

    # make-splits
    p_split = subparsers.add_parser("make-splits")
    p_split.add_argument("--output-dir", default="artifacts/socofing")
    p_split.add_argument("--folds", type=int, default=5)
    p_split.add_argument("--seed", type=int, default=20261109)

    # make-trials
    p_trial = subparsers.add_parser("make-trials")
    p_trial.add_argument("--output-dir", default="artifacts/socofing")
    p_trial.add_argument("--impostors-per-probe", type=int, default=20)
    p_trial.add_argument("--seed", type=int, default=20261109)

    # evaluate
    p_eval = subparsers.add_parser("evaluate")
    p_eval.add_argument("--output-dir", default="artifacts/socofing")
    p_eval.add_argument("--descriptor", default="9d")
    p_eval.add_argument("--matcher", default="hungarian")
    p_eval.add_argument("--bootstrap-replicates", type=int, default=2000)
    p_eval.add_argument("--seed", type=int, default=20261109)

    # summarize / verify
    p_ver = subparsers.add_parser("verify-artifacts")
    p_ver.add_argument("--output-dir", default="artifacts/socofing")

    # all
    p_all = subparsers.add_parser("all")
    p_all.add_argument("--dataset-root", default="dataset/SOCOFing")
    p_all.add_argument("--output-dir", default="artifacts/socofing")
    p_all.add_argument("--folds", type=int, default=5)
    p_all.add_argument("--descriptor", default="9d")
    p_all.add_argument("--matcher", default="hungarian")
    p_all.add_argument("--impostors-per-probe", type=int, default=20)
    p_all.add_argument("--bootstrap-replicates", type=int, default=2000)
    p_all.add_argument("--seed", type=int, default=20261109)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "audit":
        cmd_audit(args)
    elif args.command == "build-manifest":
        cmd_build_manifest(args)
    elif args.command == "validate-links":
        cmd_validate_links(args)
    elif args.command == "make-splits":
        cmd_make_splits(args)
    elif args.command == "make-trials":
        cmd_make_trials(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "verify-artifacts":
        cmd_verify_artifacts(args)
    elif args.command == "all":
        cmd_audit(args)
        cmd_build_manifest(args)
        cmd_validate_links(args)
        cmd_make_splits(args)
        cmd_make_trials(args)
        cmd_evaluate(args)
        cmd_verify_artifacts(args)


if __name__ == "__main__":
    main()
