"""
Release Packaging Script for CancelableFingerprintHybrid Platform Version 2.2-Hardened (P0.7).

Executes the dynamic test collection, test suite execution, seven-gate reproduction harness,
generates the release archive CancelableFingerprintHybrid-v2.2-Hardened.zip, computes its SHA-256,
and outputs experiments/results/v2_2_release_manifest.json.
"""

import os
import sys
import json
import zipfile
import hashlib
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def run_command_capture(cmd: list) -> tuple:
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def main():
    print("=" * 80)
    print("  PACKAGING CANCELABLE FINGERPRINT HYBRID v2.2-HARDENED RELEASE ARCHIVE")
    print("=" * 80)

    version_id = "v2.2-Hardened-20261110-p0-complete"
    timestamp = datetime.utcnow().isoformat() + "Z"

    # 1. Dynamic Pytest Collection
    print("\n[1/5] Collecting dynamic pytest tests...")
    code, out_coll, err_coll = run_command_capture([sys.executable, "-m", "pytest", "--collect-only", "-q"])
    test_items = [line.strip() for line in out_coll.splitlines() if line.strip() and not line.startswith("no tests") and "test" in line]
    print(f"  Collected {len(test_items)} test nodes.")

    # 2. Complete Pytest Execution
    print("\n[2/5] Running complete automated pytest suite...")
    code, out_test, err_test = run_command_capture([sys.executable, "-m", "pytest", "-v"])
    assert code == 0, f"Pytest failed with exit code {code}:\n{err_test}\n{out_test}"
    test_lines = out_test.strip().splitlines()
    summary_line = test_lines[-1] if test_lines else "Unknown"
    print(f"  Pytest Result: {summary_line}")

    # 3. Seven-Gate Execution Log
    print("\n[3/5] Executing 7-Gate Master Reproduction Harness...")
    code, out_gates, err_gates = run_command_capture([sys.executable, "experiments/reproduce_all_gates.py"])
    assert code == 0, f"7-Gate harness failed with exit code {code}:\n{err_gates}\n{out_gates}"
    print("  7-Gate Harness passed with exit code 0.")

    # 4. Create Release Archive
    print("\n[4/5] Building release zip archive...")
    zip_name = "CancelableFingerprintHybrid-v2.2-Hardened.zip"
    zip_root_path = os.path.join(PROJECT_ROOT, zip_name)
    zip_results_path = os.path.join(PROJECT_ROOT, "experiments", "results", zip_name)

    # Directories and files to include
    include_dirs = ["src", "tests", "experiments", "configs", "docs", "demo", "dataset"]
    include_files = [
        "final_research_report.md",
        "master_project_architecture_and_completion_report.md",
        "README.md",
        "requirements.txt",
        "walkthrough.md",
        "FINAL_V2_2_HARDENED_INDEPENDENT_VALIDATION_REPORT.md"
    ]

    # Exclusions
    exclude_extensions = {".pyc", ".pyo", ".zip"}
    exclude_dirs = {"__pycache__", ".pytest_cache", ".git", ".idea", ".vscode"}

    archive_count = 0
    with zipfile.ZipFile(zip_root_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add individual files
        for rel_file in include_files:
            file_path = os.path.join(PROJECT_ROOT, rel_file)
            if os.path.exists(file_path):
                zf.write(file_path, arcname=rel_file)
                archive_count += 1

        # Add directories
        for root_dir in include_dirs:
            full_dir = os.path.join(PROJECT_ROOT, root_dir)
            if not os.path.exists(full_dir):
                continue
            for dirpath, dirnames, filenames in os.walk(full_dir):
                # Filter out excluded directory names in-place
                dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in exclude_extensions:
                        continue
                    abs_f = os.path.join(dirpath, filename)
                    arc_f = os.path.relpath(abs_f, PROJECT_ROOT)
                    zf.write(abs_f, arcname=arc_f)
                    archive_count += 1

    # Also copy to experiments/results/
    import shutil
    shutil.copy2(zip_root_path, zip_results_path)

    archive_sha256 = sha256_file(zip_root_path)
    archive_size = os.path.getsize(zip_root_path)
    print(f"  Archive created: {zip_name}")
    print(f"  Files packaged: {archive_count}")
    print(f"  Archive size: {archive_size:,} bytes")
    print(f"  Archive SHA-256: {archive_sha256}")

    # 5. Generate v2_2_release_manifest.json
    print("\n[5/5] Generating release manifest...")
    report_sha256 = sha256_file(os.path.join(PROJECT_ROOT, "final_research_report.md"))
    master_sha256 = sha256_file(os.path.join(PROJECT_ROOT, "master_project_architecture_and_completion_report.md"))

    release_manifest = {
        "release_version": "v2.2-Hardened",
        "commit_identifier": version_id,
        "release_timestamp": timestamp,
        "archive_package": {
            "filename": zip_name,
            "sha256": archive_sha256,
            "size_bytes": archive_size,
            "packaged_file_count": archive_count
        },
        "authoritative_report_hashes": {
            "final_research_report_md": report_sha256,
            "master_project_architecture_and_completion_report_md": master_sha256
        },
        "dynamic_pytest_verification": {
            "collected_test_count": len(test_items),
            "collection_sample": test_items[:10],
            "execution_summary": summary_line,
            "exit_code": 0
        },
        "seven_gate_verification": {
            "all_gates_passed": True,
            "exit_code": 0,
            "execution_log": out_gates.strip()
        }
    }

    manifest_path = os.path.join(PROJECT_ROOT, "experiments", "results", "v2_2_release_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(release_manifest, f, indent=2)
    print(f"  Release manifest written to: {manifest_path}")

    print("\n" + "=" * 80)
    print("  RELEASE PACKAGING COMPLETE (SUCCESS)")
    print("=" * 80)


if __name__ == "__main__":
    main()
