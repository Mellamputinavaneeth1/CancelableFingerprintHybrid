"""
Automated Test Suite for Full SOCOFing Evaluation Framework.

Covers Gates 0-10:
- Parser against canonical filename patterns, alteration codes, and corruptions.
- Biometric identity resolution (fingerprint_id = subject_id|hand|finger).
- Probe-to-clean-gallery linkage invariants.
- Gate 3 zero-leakage subject-disjoint outer fold partitioning (dev-fit 80% / dev-cal 20% / test).
- Gate 7 genuine/impostor trial generation invariants and stratified impostor distribution.
- Gate 6 RFC 8439 AEAD ChaCha20-Poly1305 credential serialization and 1-bit tamper rejection.
"""

import os
import json
import tempfile
import pytest
import numpy as np

from experiments.socofing_full.manifest_builder import (
    parse_socofing_filename,
    compute_file_sha256,
    VALID_FINGERS,
    VALID_HANDS
)
from experiments.socofing_full.split_manager import (
    create_subject_disjoint_splits
)
from experiments.socofing_full.trial_builder import (
    generate_fold_trials
)
from experiments.socofing_full.evaluator import (
    SOCOFingFullEvaluator
)
from src.services.types import ProtectedCredentialDTO
from src.services.credential_serializer import CredentialSerializer
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors


class TestSOCOFingParser:
    """Tests filename parsing, normalization, and unmapped rejection (Gate 1)."""

    def test_parse_real_fingerprints(self):
        # Format: {id}__{gender}_{hand}_{finger}_finger.BMP
        res1 = parse_socofing_filename("1__M_Left_index_finger.BMP")
        assert res1 is not None
        assert res1["subject_id"] == 1
        assert res1["sex_metadata"] == "M"
        assert res1["hand"] == "Left"
        assert res1["finger"] == "index"
        assert res1["fingerprint_id"] == "001|Left|index"
        assert res1["is_altered"] is False
        assert res1["severity"] == "Real"
        assert res1["parse_status"] == "VALID_REAL"
        assert res1["is_gallery_candidate"] is True
        assert res1["is_probe_candidate"] is False

        res2 = parse_socofing_filename("250__F_Right_thumb_finger.BMP")
        assert res2 is not None
        assert res2["subject_id"] == 250
        assert res2["sex_metadata"] == "F"
        assert res2["hand"] == "Right"
        assert res2["finger"] == "thumb"
        assert res2["fingerprint_id"] == "250|Right|thumb"
        assert res2["is_altered"] is False

    def test_parse_altered_fingerprints(self):
        # Central rotation
        res_cr = parse_socofing_filename("10__F_Right_thumb_finger_CR.BMP", parent_dir="Altered/Altered-Easy")
        assert res_cr is not None
        assert res_cr["subject_id"] == 10
        assert res_cr["sex_metadata"] == "F"
        assert res_cr["hand"] == "Right"
        assert res_cr["finger"] == "thumb"
        assert res_cr["fingerprint_id"] == "010|Right|thumb"
        assert res_cr["is_altered"] is True
        assert res_cr["alteration_type"] == "Central_Rotation"
        assert res_cr["severity"] == "Easy"
        assert res_cr["is_gallery_candidate"] is False
        assert res_cr["is_probe_candidate"] is True

        # Z-cut
        res_z = parse_socofing_filename("005__M_Right_little_finger_Zcut.BMP", parent_dir="Altered/Altered-Medium")
        assert res_z is not None
        assert res_z["subject_id"] == 5
        assert res_z["alteration_type"] == "Z-cut"
        assert res_z["severity"] == "Medium"
        assert res_z["fingerprint_id"] == "005|Right|little"

        # Obliteration
        res_obl = parse_socofing_filename("7__F_Left_middle_finger_Obl.BMP", parent_dir="Altered/Altered-Hard")
        assert res_obl is not None
        assert res_obl["subject_id"] == 7
        assert res_obl["alteration_type"] == "Obliteration"
        assert res_obl["severity"] == "Hard"
        assert res_obl["fingerprint_id"] == "007|Left|middle"

    def test_all_five_fingers_and_both_hands(self):
        for hand in ["Left", "Right"]:
            for finger in ["thumb", "index", "middle", "ring", "little"]:
                fname = f"42__M_{hand}_{finger}_finger.BMP"
                parsed = parse_socofing_filename(fname)
                assert parsed is not None
                assert parsed["hand"] == hand
                assert parsed["finger"] == finger
                assert parsed["fingerprint_id"] == f"042|{hand}|{finger}"

    def test_unmapped_and_invalid_filenames(self):
        # Non-image extensions
        assert parse_socofing_filename("1__M_Left_index_finger.txt") is None
        assert parse_socofing_filename("1__M_Left_index_finger.csv") is None

        # Invalid hand name
        assert parse_socofing_filename("1__M_Top_index_finger.BMP") is None

        # Invalid finger name
        assert parse_socofing_filename("1__M_Left_palm_finger.BMP") is None

        # Malformed pattern
        assert parse_socofing_filename("random_scan_image.BMP") is None
        assert parse_socofing_filename("subj1_left_index.BMP") is None


class TestSOCOFingSplitManager:
    """Tests Gate 3 Subject-Disjoint Partitioning Invariants."""

    def _generate_synthetic_manifest(self, num_subjects: int = 15) -> list:
        records = []
        fid = 0
        for s in range(1, num_subjects + 1):
            for hand in ["Left", "Right"]:
                for finger in ["thumb", "index"]:
                    # Real
                    fid += 1
                    fp_id = f"{s:03d}|{hand}|{finger}"
                    records.append({
                        "file_id": f"file_{fid:06d}",
                        "subject_id": s,
                        "sex_metadata": "M",
                        "hand": hand,
                        "finger": finger,
                        "fingerprint_id": fp_id,
                        "source_fingerprint_id": fp_id,
                        "is_altered": False,
                        "severity": "Real",
                        "partition": "Real",
                        "is_gallery_candidate": True,
                        "is_probe_candidate": False,
                        "relative_path": f"Real/{s:03d}__{hand}_{finger}.BMP"
                    })
                    # Altered Easy
                    fid += 1
                    records.append({
                        "file_id": f"file_{fid:06d}",
                        "subject_id": s,
                        "sex_metadata": "M",
                        "hand": hand,
                        "finger": finger,
                        "fingerprint_id": fp_id,
                        "source_fingerprint_id": fp_id,
                        "is_altered": True,
                        "alteration_type": "Central_Rotation",
                        "severity": "Easy",
                        "partition": "Altered-Easy",
                        "is_gallery_candidate": False,
                        "is_probe_candidate": True,
                        "relative_path": f"Altered/Altered-Easy/{s:03d}__{hand}_{finger}_CR.BMP"
                    })
        return records

    def test_subject_disjoint_zero_leakage(self):
        records = self._generate_synthetic_manifest(num_subjects=15)
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = create_subject_disjoint_splits(records, output_dir=tmpdir, num_folds=5, seed=42)
            assert summary["zero_leakage_verified"] is True
            assert summary["num_folds"] == 5

            all_test_subjects = []
            for fold_audit in summary["folds"]:
                fold_name = fold_audit["fold"]
                fold_json_path = os.path.join(tmpdir, f"{fold_name}.json")
                with open(fold_json_path, "r", encoding="utf-8") as f:
                    fold_payload = json.load(f)

                test_subj = set(fold_payload["subjects"]["test_subjects"])
                fit_subj = set(fold_payload["subjects"]["dev_fit_subjects"])
                cal_subj = set(fold_payload["subjects"]["dev_cal_subjects"])
                dev_subj = fit_subj.union(cal_subj)

                # Strict zero leakage assertions
                assert test_subj.isdisjoint(dev_subj)
                assert fit_subj.isdisjoint(cal_subj)
                assert fit_subj.isdisjoint(test_subj)
                assert cal_subj.isdisjoint(test_subj)

                all_test_subjects.extend(list(test_subj))

            # Every subject must appear in test exactly once across all 5 folds
            assert sorted(all_test_subjects) == list(range(1, 16))


class TestSOCOFingTrialBuilder:
    """Tests Gate 7 Stratified Genuine & Impostor Trial Generation."""

    def test_trial_generation_invariants(self):
        records = []
        fid = 0
        # Create 4 test subjects with multiple fingers and alterations
        for s in [1, 2, 3, 4]:
            for hand in ["Left", "Right"]:
                for finger in ["thumb", "index"]:
                    fp_id = f"{s:03d}|{hand}|{finger}"
                    # Real gallery
                    fid += 1
                    records.append({
                        "file_id": f"file_{fid:06d}",
                        "subject_id": s,
                        "hand": hand,
                        "finger": finger,
                        "fingerprint_id": fp_id,
                        "is_gallery_candidate": True,
                        "is_probe_candidate": False,
                        "relative_path": f"Real/{fp_id}.BMP",
                        "severity": "Real"
                    })
                    # Altered probe
                    fid += 1
                    records.append({
                        "file_id": f"file_{fid:06d}",
                        "subject_id": s,
                        "hand": hand,
                        "finger": finger,
                        "fingerprint_id": fp_id,
                        "is_gallery_candidate": False,
                        "is_probe_candidate": True,
                        "relative_path": f"Altered/{fp_id}_CR.BMP",
                        "severity": "Easy",
                        "alteration_type": "Central_Rotation"
                    })

        fold_data = {
            "fold_name": "fold_01",
            "subjects": {
                "test_subjects": [1, 2, 3, 4],
                "dev_fit_subjects": [],
                "dev_cal_subjects": []
            }
        }

        trials = generate_fold_trials(records, fold_data, split_type="test", max_impostors_per_probe=5, seed=123)

        assert len(trials) > 0
        num_gen = sum(1 for t in trials if t["is_genuine"] == 1)
        num_imp = sum(1 for t in trials if t["is_genuine"] == 0)
        assert num_gen + num_imp == len(trials)

        for t in trials:
            # Genuine pairs must have identical fingerprint_id
            if t["is_genuine"] == 1:
                assert t["query_fingerprint_id"] == t["gallery_fingerprint_id"]
                assert t["query_subject_id"] == t["gallery_subject_id"]
                assert t["impostor_category"] == "none_genuine"
            else:
                # Impostor pairs must have different fingerprint_id
                assert t["query_fingerprint_id"] != t["gallery_fingerprint_id"]
                assert t["impostor_category"] in {
                    "within_subject_diff_finger",
                    "diff_subject_same_finger",
                    "diff_subject_diff_finger",
                    "diff_subject_general"
                }
                if t["impostor_category"] == "within_subject_diff_finger":
                    assert t["query_subject_id"] == t["gallery_subject_id"]
                else:
                    assert t["query_subject_id"] != t["gallery_subject_id"]


class TestSOCOFingGate6AEADProtection:
    """Tests Gate 6 Orthonormal Projection, Quantization, and AEAD Serialization."""

    def test_aead_roundtrip_and_tamper_detection(self):
        serializer = CredentialSerializer()
        rng = np.random.RandomState(42)

        # 10 random 9-D descriptors
        descriptors = rng.randn(10, 9).astype(np.float32)
        weights = rng.uniform(0.5, 1.5, size=(10,)).astype(np.float32)

        master_secret = "SOCOFing_Full_Master_Secret_2026_HKDF!"  # >= 32 bytes
        W = derive_orthonormal_matrix(master_secret, salt="test_salt", app_id="test_app", in_dim=9, out_dim=32)
        assert W.shape == (32, 9)
        # Check column orthonormality: W^T @ W = I_9
        WtW = np.dot(W.T, W)
        np.testing.assert_allclose(WtW, np.eye(9, dtype=np.float32), atol=1e-5)

        # Project and quantize
        projected = project_descriptors(descriptors, W)
        quantized = quantize_descriptors(projected)
        assert quantized.shape == (10, 32)
        assert quantized.dtype == np.uint8

        # Create credential DTO
        dto = ProtectedCredentialDTO(
            credential_id="cred_0000000000000000000000000001",
            subject_id="001",
            salt="test_salt",
            app_id="test_app",
            bitstrings=quantized,
            weights=weights,
            num_triangles=len(quantized),
            version="1.0"
        )

        storage_key = b"01234567890123456789012345678901"  # 32 bytes

        # Serialize & Encrypt with ChaCha20-Poly1305
        env = serializer.serialize_protected(dto, storage_key=storage_key, encrypt=True)
        raw_bytes = env.raw_bytes
        assert len(raw_bytes) > 0

        # Successful round-trip deserialization
        restored_dto = serializer.deserialize_protected(raw_bytes, storage_key=storage_key)
        assert restored_dto.num_triangles == 10
        np.testing.assert_array_equal(restored_dto.bitstrings, quantized)
        # Weights converted to float16 and back to float32
        np.testing.assert_allclose(restored_dto.weights, weights.astype(np.float16).astype(np.float32), atol=1e-3)

        # 1-bit tampering detection test: flip single bit in ciphertext
        tampered_bytes = bytearray(raw_bytes)
        tampered_bytes[-1] ^= 0x01  # Flip least significant bit of Poly1305 tag or ciphertext
        tampered_bytes = bytes(tampered_bytes)

        # Must reject with decryption/verification exception
        with pytest.raises(Exception):
            serializer.deserialize_protected(tampered_bytes, storage_key=storage_key)


class TestSOCOFingSelfMatchAndReconciliation:
    """
    WP3 Formal Verification of Self-Match Exclusion and Trial Reconciliation.
    
    Verifies:
    1. Identification and exclusion of 10 exact same-image self-matches (score 1.0).
    2. Recomputed distinct-impression genuine trials (N_gen = 90) and impostor trials (N_imp = 500).
    3. Exact metric shift from EER 24.55% / AUC 0.82338 to EER 26.14% / AUC 0.80376.
    4. Confusion matrix accounting identities: TP + FN = N_gen, FP + TN = N_imp.
    5. Monotonic degradation across alteration severities (Easy < Medium < Hard).
    6. Local dataset pilot scope boundary (10 subjects, 100 images, digital alteration).
    """

    @pytest.fixture
    def canonical_scores(self):
        import csv
        path = os.path.join(
            os.path.dirname(__file__), "..",
            "experiments", "results", "socofing_full_canonical", "results", "pooled_oof_scores.csv"
        )
        if not os.path.exists(path):
            pytest.skip("Canonical pooled_oof_scores.csv not found")
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_self_match_audit_and_metric_reconciliation(self, canonical_scores):
        from src.evaluation.metrics import compute_roc_metrics

        assert len(canonical_scores) == 600

        # Identify exact same-image self matches
        self_matches = [r for r in canonical_scores if r["query_relative_path"] == r["gallery_relative_path"]]
        assert len(self_matches) == 10
        for r in self_matches:
            assert float(r["score"]) == pytest.approx(1.0, abs=1e-5)
            assert r["is_genuine"] == "1"

        # Baseline pooled metrics WITH self-matches
        gen_600 = np.array([float(r["score"]) for r in canonical_scores if r["is_genuine"] == "1"])
        imp_600 = np.array([float(r["score"]) for r in canonical_scores if r["is_genuine"] == "0"])
        m_600 = compute_roc_metrics(gen_600, imp_600)
        assert m_600["eer"] * 100 == pytest.approx(24.55, abs=0.1)
        assert m_600["auc"] == pytest.approx(0.82338, abs=0.001)

        # Clean pooled metrics EXCLUDING self-matches
        clean = [r for r in canonical_scores if r["query_relative_path"] != r["gallery_relative_path"]]
        assert len(clean) == 590
        gen_clean = np.array([float(r["score"]) for r in clean if r["is_genuine"] == "1"])
        imp_clean = np.array([float(r["score"]) for r in clean if r["is_genuine"] == "0"])
        assert len(gen_clean) == 90
        assert len(imp_clean) == 500

        m_clean = compute_roc_metrics(gen_clean, imp_clean)
        assert m_clean["eer"] * 100 == pytest.approx(26.14, abs=0.1)
        assert m_clean["auc"] == pytest.approx(0.80376, abs=0.001)

    def test_trial_confusion_accounting_identities(self, canonical_scores):
        clean = [r for r in canonical_scores if r["query_relative_path"] != r["gallery_relative_path"]]
        gen_scores = [float(r["score"]) for r in clean if r["is_genuine"] == "1"]
        imp_scores = [float(r["score"]) for r in clean if r["is_genuine"] == "0"]

        n_gen = len(gen_scores)
        n_imp = len(imp_scores)
        assert n_gen == 90
        assert n_imp == 500

        for tau in [0.40, 0.45, 0.495, 0.50, 0.52, 0.55]:
            tp = sum(1 for s in gen_scores if s >= tau)
            fn = sum(1 for s in gen_scores if s < tau)
            fp = sum(1 for s in imp_scores if s >= tau)
            tn = sum(1 for s in imp_scores if s < tau)

            assert tp + fn == n_gen, f"TP + FN != N_gen at tau={tau}"
            assert fp + tn == n_imp, f"FP + TN != N_imp at tau={tau}"

    def test_severity_monotonicity_on_clean_trials(self, canonical_scores):
        from src.evaluation.metrics import compute_roc_metrics

        clean = [r for r in canonical_scores if r["query_relative_path"] != r["gallery_relative_path"]]

        metrics_by_sev = {}
        for sev in ["Easy", "Medium", "Hard"]:
            gen = np.array([float(r["score"]) for r in clean if r["query_severity"] == sev and r["is_genuine"] == "1"])
            imp = np.array([float(r["score"]) for r in clean if r["query_severity"] == sev and r["is_genuine"] == "0"])
            assert len(gen) == 30
            assert len(imp) == 150
            metrics_by_sev[sev] = compute_roc_metrics(gen, imp)

        # Monotonic EER increase with alteration severity
        assert metrics_by_sev["Easy"]["eer"] < metrics_by_sev["Medium"]["eer"] < metrics_by_sev["Hard"]["eer"]
        # Monotonic AUC decrease with alteration severity
        assert metrics_by_sev["Easy"]["auc"] > metrics_by_sev["Medium"]["auc"] > metrics_by_sev["Hard"]["auc"]

    def test_socofing_scope_pilot_boundary(self):
        import csv
        path = os.path.join(
            os.path.dirname(__file__), "..",
            "experiments", "results", "socofing_full_canonical", "metadata_manifest.csv"
        )
        if not os.path.exists(path):
            pytest.skip("metadata_manifest.csv not found")

        with open(path, "r", encoding="utf-8") as f:
            manifest = list(csv.DictReader(f))

        assert len(manifest) == 100
        real = [r for r in manifest if r["is_altered"] == "False"]
        altered = [r for r in manifest if r["is_altered"] == "True"]
        assert len(real) == 10
        assert len(altered) == 90

        unique_subjects = {int(r["subject_id"]) for r in manifest}
        assert unique_subjects == set(range(1, 11))

        # Every altered probe maps to a clean gallery
        for r in altered:
            assert r["has_clean_gallery"] == "True"
            assert r["gallery_file_id"].startswith("file_")

    def test_canonical_clean_ledger_and_folds(self):
        import csv
        from collections import defaultdict
        clean_path = os.path.join(
            os.path.dirname(__file__), "..",
            "experiments", "results", "socofing_full_canonical", "results", "canonical_clean_trials_590.csv"
        )
        if not os.path.exists(clean_path):
            pytest.skip("canonical_clean_trials_590.csv not found")

        with open(clean_path, "r", encoding="utf-8") as f:
            clean_trials = list(csv.DictReader(f))

        assert len(clean_trials) == 590
        # Assert strictly zero self matches
        for r in clean_trials:
            assert r["query_relative_path"] != r["gallery_relative_path"], "Found self-match in clean ledger!"

        # Group by outer fold
        by_fold = defaultdict(list)
        for r in clean_trials:
            by_fold[r["outer_fold"]].append(r)

        assert len(by_fold) == 5
        for fold_name, f_rows in by_fold.items():
            f_gen = [r for r in f_rows if r["is_genuine"] == "1"]
            f_imp = [r for r in f_rows if r["is_genuine"] == "0"]
            assert len(f_gen) == 18, f"{fold_name} has {len(f_gen)} genuine trials, expected exactly 18"
            assert len(f_imp) == 100, f"{fold_name} has {len(f_imp)} impostor trials, expected exactly 100"
            assert len(f_rows) == 118, f"{fold_name} has {len(f_rows)} total trials, expected exactly 118"
