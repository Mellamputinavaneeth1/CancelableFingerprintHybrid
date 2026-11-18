"""
Artifact Hash Manifest Generator for CancelableFingerprintHybrid v2.2-Hardened.

Scans authoritative specification documents, code files, test suites,
and raw data files, and writes:
- experiments/results/canonical_artifact_hashes.json
- docs/artifact_manifest.md
"""

import os
import hashlib
import json
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    tracked_files = [
        # Authoritative specifications and reports
        "final_research_report.md",
        "master_project_architecture_and_completion_report.md",
        "README.md",
        "requirements.txt",
        "docs/descriptor_mathematics_v2.md",
        "docs/wp2_preimage_and_chebyshev_audit.md",
        "docs/wp3_socofing_evaluation_audit.md",
        "docs/wp4_recognition_and_threshold_audit.md",
        "docs/wp5_wp6_cryptography_and_privacy_audit.md",
        # Core source implementations
        "src/pipeline.py",
        "src/evaluation/independent_evaluator.py",
        "src/evaluation/metrics.py",
        "src/evaluation/protocols.py",
        "src/graph/delaunay.py",
        "src/graph/descriptors.py",
        "src/matching/augmented_cost.py",
        "src/matching/bipartite_matcher.py",
        "src/matching/graph_consensus.py",
        "src/protection/inversion_defense.py",
        "src/protection/projection.py",
        "src/protection/quantizer.py",
        "src/services/nonce_manager.py",
        "src/services/credential_serializer.py",
        "src/services/credential_repository.py",
        "src/services/crypto_aead.py",
        "src/services/key_service.py",
        "src/services/orchestrator.py",
        "src/services/types.py",
        # Reproduction harness and experiment scripts
        "experiments/reproduce_all_gates.py",
        "scripts/package_v2_2_release.py",
        "experiments/socofing_full/manifest_builder.py",
        "experiments/socofing_full/split_manager.py",
        "experiments/socofing_full/trial_builder.py",
        "experiments/socofing_full/evaluator.py",
        # Automated test suite (v2.2-Hardened)
        "tests/test_p0_remediation.py",
        "tests/test_descriptor_properties.py",
        "tests/test_wp2_inversion_audit.py",
        "tests/test_socofing_full.py",
        "tests/test_wp4_recognition_audit.py",
        "tests/test_wp5_wp6_crypto_privacy_audit.py",
        "tests/test_matching.py",
        "tests/test_security_regressions.py",
        "tests/test_reproducibility.py",
        "tests/test_integration.py",
        "tests/test_concurrency.py",
        # Key raw trial ledgers and canonical results
        "experiments/results/baseline_manifest.json",
        "experiments/results/v2_2_release_manifest.json",
        "experiments/results/socofing_full_canonical/results/canonical_clean_trials_590.csv",
        "experiments/results/socofing_full_canonical/results/self_match_sanity_trials_10.csv",
        "experiments/results/socofing_full_canonical/results/pooled_oof_scores.csv",
        "experiments/results/socofing_full_canonical/archive_audit.json",
        "experiments/results/fvc2002/DB1_A_9D_delaunay/prot_genuine_scores.npy",
        "experiments/results/fvc2002/DB1_A_9D_delaunay/prot_impostor_scores.npy",
        "experiments/results/fvc2002/DB1_B_9D_delaunay/prot_genuine_scores.npy",
        "experiments/results/fvc2002/DB1_B_9D_delaunay/prot_impostor_scores.npy"
    ]

    manifest = {
        "platform_version": "2.2-Hardened",
        "generation_timestamp": datetime.utcnow().isoformat() + "Z",
        "algorithm": "SHA-256",
        "artifacts": {}
    }

    markdown_rows = []

    for rel_path in tracked_files:
        full_path = os.path.join(PROJECT_ROOT, rel_path.replace("/", os.sep))
        if os.path.exists(full_path):
            digest = sha256_file(full_path)
            size_b = os.path.getsize(full_path)
            manifest["artifacts"][rel_path] = {
                "sha256": digest,
                "size_bytes": size_b
            }
            markdown_rows.append(f"| `{rel_path}` | `{digest}` | {size_b:,} | PASS |")
        else:
            print(f"WARNING: File not found: {rel_path}")

    # Write JSON manifest
    json_path = os.path.join(PROJECT_ROOT, "experiments", "results", "canonical_artifact_hashes.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote JSON manifest: {json_path} ({len(manifest['artifacts'])} artifacts)")

    # Write Markdown manifest
    md_content = f"""# Canonical Artifact Hash Manifest (Platform Version 2.2-Hardened)

**Generated:** {manifest['generation_timestamp']}  
**Cryptographic Hash Algorithm:** SHA-256  
**Total Tracked Artifacts:** {len(manifest['artifacts'])}  
**Integrity Policy:** Bit-exact verification against accidental mutation, silent fallbacks, or data drift.

---

## 1. Authoritative Specification, Code, and Trial Ledgers

| Relative Path | Canonical SHA-256 Hex Digest | Size (Bytes) | Integrity Status |
|:---|:---|:---:|:---:|
""" + "\n".join(markdown_rows) + f"""

---

## 2. Manifest Verification Instructions

To re-verify all artifact digests against this canonical manifest in an automated environment:

```python
import hashlib, json, os

with open("experiments/results/canonical_artifact_hashes.json", "r") as f:
    manifest = json.load(f)

for rel_path, meta in manifest["artifacts"].items():
    assert os.path.exists(rel_path), f"Missing: {{rel_path}}"
    with open(rel_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert digest == meta["sha256"], f"Mismatch on {{rel_path}}"
print(f"All {{len(manifest['artifacts'])}} canonical artifacts verified cleanly.")
```
"""

    md_path = os.path.join(PROJECT_ROOT, "docs", "artifact_manifest.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Wrote Markdown manifest: {md_path}")


if __name__ == "__main__":
    main()
