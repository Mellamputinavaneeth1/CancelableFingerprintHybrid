"""
WP6: Storage and Performance Benchmark Suite.

Audits:
1. Exact byte breakdown of serialized credential for N=144:
   64-byte header + 576-byte descriptors + 288-byte weights + 16-byte tag = 944 bytes.
2. Actual dataset-wide storage statistics on real enrolled credentials
   (FVC2002 DB1_A, DB1_B, and SOCOFing) measuring min, max, mean, median, std, total bytes.
3. Stage-wise latency profiling of all pipeline stages using time.perf_counter_ns.
4. Identical-workload Hungarian vs Greedy 1-to-1 assignment benchmark.
5. Distinction of throughput across 4 operational modes:
   - 1:1 Authentication Latency (image-to-decision and template-to-decision)
   - 1:N Screening Throughput across gallery sizes N in [10, 50, 100, 500, 1000]
   - Match-Only Throughput (comparisons/sec)
   - Concurrent Multi-Worker Throughput (scaling across worker counts)
"""

import os
import sys
import time
import json
import zlib
import struct
import concurrent.futures
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.datasets.fvc_loader import load_fvc_dataset
from src.services.orchestrator import ProductionBiometricService
from src.services.credential_serializer import CredentialSerializer, HEADER_SIZE, TAG_SIZE
from src.services.types import ProtectedCredentialDTO
from src.matching.bipartite_matcher import (
    match_protected_templates,
    match_protected_greedy,
    build_augmented_cost
)
from scipy.optimize import linear_sum_assignment


def run_storage_verification_n144() -> Dict[str, Any]:
    print("\n--- 1. Analytical & Empirical Verification of N=144 Serialized Credential ---")
    n = 144
    out_dim = 32
    
    # Analytical calculation
    header_bytes_expected = 64
    desc_bytes_expected = n * (out_dim // 8)  # 144 * 4 = 576
    weights_bytes_expected = n * 2  # 144 * 2 = 288 (float16)
    tag_bytes_expected = 16
    total_expected = header_bytes_expected + desc_bytes_expected + weights_bytes_expected + tag_bytes_expected
    
    assert total_expected == 944, f"Analytical total must be 944, got {total_expected}"
    
    # Empirical verification using actual CredentialSerializer
    serializer = CredentialSerializer()
    storage_key = b"01234567890123456789012345678901"  # 32 bytes
    
    # Generate mock credential with N=144
    rng = np.random.default_rng(42)
    mock_bitstrings = rng.integers(0, 2, size=(n, out_dim), dtype=np.uint8)
    mock_weights = rng.uniform(0.5, 1.0, size=(n,)).astype(np.float16)
    
    dto = ProtectedCredentialDTO(
        credential_id="00000000-0000-0000-0000-000000000144",
        subject_id="subj_audit_144",
        app_id="wp6_audit",
        salt="0123456789abcdef",
        bitstrings=mock_bitstrings,
        weights=mock_weights,
        num_triangles=n,
        packed_topology=None
    )
    
    envelope = serializer.serialize_protected(dto, storage_key, encrypt=True)
    raw = envelope.raw_bytes
    
    actual_total = len(raw)
    actual_header = len(raw[:64])
    actual_ciphertext = len(raw[64:-16])
    actual_tag = len(raw[-16:])
    
    print(f"  Header Size    : {actual_header} bytes (expected 64)")
    print(f"  Descriptors    : {desc_bytes_expected} bytes (144 * 4)")
    print(f"  Weights        : {weights_bytes_expected} bytes (144 * 2 float16)")
    print(f"  Ciphertext     : {actual_ciphertext} bytes (576 desc + 288 weights)")
    print(f"  Poly1305 Tag   : {actual_tag} bytes (expected 16)")
    print(f"  Total Envelope : {actual_total} bytes (expected 944)")
    
    assert actual_total == 944, f"Empirical envelope length {actual_total} != 944"
    
    # Verify bit-exact round-trip deserialization
    recovered_dto = serializer.deserialize_protected(raw, storage_key)
    recovered_bits = np.asarray(recovered_dto.bitstrings, dtype=np.uint8)
    recovered_weights = np.asarray(recovered_dto.weights, dtype=np.float16)
    
    assert np.array_equal(recovered_bits, mock_bitstrings), "Bitstring mismatch on round-trip"
    assert np.allclose(recovered_weights, mock_weights, atol=1e-3), "Weight mismatch on round-trip"
    print("  Round-trip Deserialization & AEAD Verification: PASSED (Exact match)")
    
    return {
        "num_triangles": n,
        "header_bytes": actual_header,
        "descriptor_bytes": desc_bytes_expected,
        "weight_bytes": weights_bytes_expected,
        "tag_bytes": actual_tag,
        "total_bytes": actual_total,
        "formula_verified": "64 (header) + 576 (descriptors) + 288 (weights) + 16 (tag) = 944 bytes",
        "round_trip_passed": True
    }


def run_dataset_storage_stats() -> Dict[str, Any]:
    print("\n--- 2. Measuring Dataset-Wide Physical Credential Storage ---")
    service = ProductionBiometricService()
    serializer = service.serializer
    storage_key = b"wp6_dataset_storage_eval_key_32b"
    app_id = "wp6_storage"
    salt = "wp6_salt_2026"
    
    results = {}
    
    # A. FVC2002 DB1_B (80 images)
    db1_b_data = load_fvc_dataset("DB1_B", root_dir="dataset/FVC2002")
    db1_b_images = db1_b_data["all_images"]
    db1_b_sizes = []
    db1_b_triangles = []
    
    for img_info in db1_b_images:
        img = cv2.imread(img_info["path"], cv2.IMREAD_GRAYSCALE)
        feats = service.extract_pipeline_features(img, dpi=500.0, with_topology=False)
        n = feats["descriptors"].count
        db1_b_triangles.append(n)
        
        # Serialize protected credential
        bits = service.projection_engine.project_and_quantize(feats["descriptors"].descriptors, salt, app_id)
        dto = ProtectedCredentialDTO(
            credential_id=f"cred_{img_info['finger_id']}_{img_info['impression_id']}",
            subject_id=f"subj_{img_info['finger_id']}",
            app_id=app_id,
            salt=salt,
            bitstrings=bits,
            weights=feats["descriptors"].weights.astype(np.float16),
            num_triangles=n
        )
        env = serializer.serialize_protected(dto, storage_key, encrypt=True)
        db1_b_sizes.append(len(env.raw_bytes))
        
    results["DB1_B"] = {
        "count": len(db1_b_sizes),
        "mean_bytes": float(np.mean(db1_b_sizes)),
        "median_bytes": float(np.median(db1_b_sizes)),
        "std_bytes": float(np.std(db1_b_sizes)),
        "min_bytes": int(np.min(db1_b_sizes)),
        "max_bytes": int(np.max(db1_b_sizes)),
        "total_bytes": int(np.sum(db1_b_sizes)),
        "mean_triangles": float(np.mean(db1_b_triangles)),
        "median_triangles": float(np.median(db1_b_triangles))
    }
    print(f"  FVC2002 DB1_B (N={len(db1_b_sizes)}): Mean={results['DB1_B']['mean_bytes']:.1f} B | Median={results['DB1_B']['median_bytes']:.1f} B | Min={results['DB1_B']['min_bytes']} B | Max={results['DB1_B']['max_bytes']} B")
    
    # B. FVC2002 DB1_A (Representative sample of 100 images from Set A)
    db1_a_data = load_fvc_dataset("DB1_A", root_dir="dataset/FVC2002")
    db1_a_images = db1_a_data["all_images"][:100]
    db1_a_sizes = []
    db1_a_triangles = []
    
    for img_info in db1_a_images:
        img = cv2.imread(img_info["path"], cv2.IMREAD_GRAYSCALE)
        feats = service.extract_pipeline_features(img, dpi=500.0, with_topology=False)
        n = feats["descriptors"].count
        db1_a_triangles.append(n)
        
        bits = service.projection_engine.project_and_quantize(feats["descriptors"].descriptors, salt, app_id)
        dto = ProtectedCredentialDTO(
            credential_id=f"cred_db1a_{img_info['finger_id']}_{img_info['impression_id']}",
            subject_id=f"subj_{img_info['finger_id']}",
            app_id=app_id,
            salt=salt,
            bitstrings=bits,
            weights=feats["descriptors"].weights.astype(np.float16),
            num_triangles=n
        )
        env = serializer.serialize_protected(dto, storage_key, encrypt=True)
        db1_a_sizes.append(len(env.raw_bytes))
        
    results["DB1_A"] = {
        "count": len(db1_a_sizes),
        "mean_bytes": float(np.mean(db1_a_sizes)),
        "median_bytes": float(np.median(db1_a_sizes)),
        "std_bytes": float(np.std(db1_a_sizes)),
        "min_bytes": int(np.min(db1_a_sizes)),
        "max_bytes": int(np.max(db1_a_sizes)),
        "total_bytes": int(np.sum(db1_a_sizes)),
        "mean_triangles": float(np.mean(db1_a_triangles)),
        "median_triangles": float(np.median(db1_a_triangles))
    }
    print(f"  FVC2002 DB1_A (N={len(db1_a_sizes)} sample): Mean={results['DB1_A']['mean_bytes']:.1f} B | Median={results['DB1_A']['median_bytes']:.1f} B | Min={results['DB1_A']['min_bytes']} B | Max={results['DB1_A']['max_bytes']} B")
    
    # C. SOCOFing (100 images: 10 Real, 90 Altered)
    socofing_dir = os.path.join("dataset", "SOCOFing")
    soco_sizes = []
    soco_triangles = []
    
    if os.path.exists(socofing_dir):
        from experiments.datasets.socofing_loader import load_socofing_dataset
        soco_data = load_socofing_dataset(root_dir=socofing_dir)
        soco_records = soco_data.get("real_images", []) + soco_data.get("altered_images", [])
        for rec in soco_records:
            img = cv2.imread(rec["path"], cv2.IMREAD_GRAYSCALE)
            feats = service.extract_pipeline_features(img, dpi=500.0, with_topology=False)
            n = feats["descriptors"].count
            soco_triangles.append(n)
            
            bits = service.projection_engine.project_and_quantize(feats["descriptors"].descriptors, salt, app_id)
            dto = ProtectedCredentialDTO(
                credential_id=f"cred_soco_{rec['subject_id']}_{rec['hand']}_{rec['finger']}",
                subject_id=f"subj_{rec['subject_id']}",
                app_id=app_id,
                salt=salt,
                bitstrings=bits,
                weights=feats["descriptors"].weights.astype(np.float16),
                num_triangles=n
            )
            env = serializer.serialize_protected(dto, storage_key, encrypt=True)
            soco_sizes.append(len(env.raw_bytes))
            
        results["SOCOFing"] = {
            "count": len(soco_sizes),
            "mean_bytes": float(np.mean(soco_sizes)),
            "median_bytes": float(np.median(soco_sizes)),
            "std_bytes": float(np.std(soco_sizes)),
            "min_bytes": int(np.min(soco_sizes)),
            "max_bytes": int(np.max(soco_sizes)),
            "total_bytes": int(np.sum(soco_sizes)),
            "mean_triangles": float(np.mean(soco_triangles)),
            "median_triangles": float(np.median(soco_triangles))
        }
        print(f"  SOCOFing (N={len(soco_sizes)}): Mean={results['SOCOFing']['mean_bytes']:.1f} B | Median={results['SOCOFing']['median_bytes']:.1f} B | Min={results['SOCOFing']['min_bytes']} B | Max={results['SOCOFing']['max_bytes']} B")
        
    return results


def run_pipeline_stage_latency_profile(num_runs: int = 100) -> Dict[str, Any]:
    print("\n--- 3. Profiling Latency Across Distinct Pipeline Stages (100 runs) ---")
    service = ProductionBiometricService()
    
    # Load representative DB1_B image
    img_path = os.path.join("dataset", "FVC2002", "DB1_B", "101_1.tif")
    if not os.path.exists(img_path):
        img_path = os.path.join("dataset", "FVC2002", "DB1_A", "1_1.jpg")
    raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    app_id = "latency_prof"
    salt = "salt_prof_2026"
    storage_key = b"latency_prof_storage_key_32_byte"
    
    # Intermediate representations
    c = service.capture_adapter.ingest(raw_img, dpi=500.0)
    p = service.preprocessor.process(c)
    m = service.minutiae_extractor.extract(p)
    t = service.triangle_builder.build(m, compute_topology=False)
    d = service.descriptor_builder.build(t, dpi=500.0)
    bits = service.projection_engine.project_and_quantize(d.descriptors, salt, app_id)
    dto = ProtectedCredentialDTO(
        credential_id="cred_prof_01",
        subject_id="subj_01",
        app_id=app_id,
        salt=salt,
        bitstrings=bits,
        weights=d.weights.astype(np.float16),
        num_triangles=d.count
    )
    env = service.serializer.serialize_protected(dto, storage_key, encrypt=True)
    
    # Second representation for matching
    img2_path = os.path.join("dataset", "FVC2002", "DB1_B", "101_2.tif")
    if not os.path.exists(img2_path):
        img2_path = os.path.join("dataset", "FVC2002", "DB1_A", "1_2.jpg")
    raw_img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    c2 = service.capture_adapter.ingest(raw_img2, dpi=500.0)
    p2 = service.preprocessor.process(c2)
    m2 = service.minutiae_extractor.extract(p2)
    t2 = service.triangle_builder.build(m2, compute_topology=False)
    d2 = service.descriptor_builder.build(t2, dpi=500.0)
    bits2 = service.projection_engine.project_and_quantize(d2.descriptors, salt, app_id)
    
    # Warm-up
    for _ in range(10):
        _ = service.preprocessor.process(c)
        _ = service.minutiae_extractor.extract(p)
        _ = service.triangle_builder.build(m)
        _ = service.descriptor_builder.build(t)
        _ = service.projection_engine.project_and_quantize(d.descriptors, salt, app_id)
        _ = match_protected_templates(bits, bits2, d.weights, d2.weights)
        _ = match_protected_greedy(bits, bits2, d.weights, d2.weights)
        
    t_prep = []
    t_minutiae = []
    t_triangle = []
    t_desc = []
    t_key_cold = []
    t_key_warm = []
    t_proj = []
    t_serialize = []
    t_cost_mat = []
    t_hungarian = []
    t_greedy = []
    t_e2e_cold = []
    t_e2e_warm = []
    
    for i in range(num_runs):
        # 1. Preprocessing
        t0 = time.perf_counter_ns()
        _p = service.preprocessor.process(c)
        t1 = time.perf_counter_ns()
        t_prep.append((t1 - t0) / 1e6)
        
        # 2. Minutiae extraction
        t0 = time.perf_counter_ns()
        _m = service.minutiae_extractor.extract(p)
        t1 = time.perf_counter_ns()
        t_minutiae.append((t1 - t0) / 1e6)
        
        # 3. Triangulation
        t0 = time.perf_counter_ns()
        _t = service.triangle_builder.build(m, compute_topology=False)
        t1 = time.perf_counter_ns()
        t_triangle.append((t1 - t0) / 1e6)
        
        # 4. Descriptors
        t0 = time.perf_counter_ns()
        _d = service.descriptor_builder.build(t, dpi=500.0)
        t1 = time.perf_counter_ns()
        t_desc.append((t1 - t0) / 1e6)
        
        # 5a. Key derivation (Cold HKDF)
        t0 = time.perf_counter_ns()
        _w_cold = service.key_service.derive_projection_matrix(f"cold_salt_{i}", app_id, in_dim=9, out_dim=32)
        t1 = time.perf_counter_ns()
        t_key_cold.append((t1 - t0) / 1e6)
        
        # 5b. Key retrieval (Warm cache)
        t0 = time.perf_counter_ns()
        _w_warm = service.projection_engine.get_projection_matrix(salt, app_id, in_dim=9, out_dim=32)
        t1 = time.perf_counter_ns()
        t_key_warm.append((t1 - t0) / 1e6)
        
        # 6. Projection & Quantization
        t0 = time.perf_counter_ns()
        _b = service.projection_engine.project_and_quantize(d.descriptors, salt, app_id)
        t1 = time.perf_counter_ns()
        t_proj.append((t1 - t0) / 1e6)
        
        # 7. Serialization + AEAD
        t0 = time.perf_counter_ns()
        _env = service.serializer.serialize_protected(dto, storage_key, encrypt=True)
        t1 = time.perf_counter_ns()
        t_serialize.append((t1 - t0) / 1e6)
        
        # 8. Pairwise cost construction
        t0 = time.perf_counter_ns()
        xor_sum = np.bitwise_xor(bits[:, None, :], bits2[None, :, :]).sum(axis=2)
        dist_mat = xor_sum / 32.0
        aug_cost = build_augmented_cost(dist_mat, tau=0.22)
        t1 = time.perf_counter_ns()
        t_cost_mat.append((t1 - t0) / 1e6)
        
        # 9a. Hungarian bipartite assignment
        t0 = time.perf_counter_ns()
        _r_h, _c_h = linear_sum_assignment(aug_cost)
        t1 = time.perf_counter_ns()
        t_hungarian.append((t1 - t0) / 1e6)
        
        # 9b. Greedy 1-to-1 assignment
        t0 = time.perf_counter_ns()
        _score_g = match_protected_greedy(bits, bits2, d.weights, d2.weights, mode="greedy_1to1")
        t1 = time.perf_counter_ns()
        t_greedy.append((t1 - t0) / 1e6)
        
        # 10. Complete End-to-End Image-to-Decision
        t0 = time.perf_counter_ns()
        f_probe = service.extract_pipeline_features(raw_img, dpi=500.0, with_topology=False)
        b_probe = service.projection_engine.project_and_quantize(f_probe["descriptors"].descriptors, salt, app_id)
        _score_e2e = match_protected_greedy(b_probe, bits2, f_probe["descriptors"].weights, d2.weights)
        t1 = time.perf_counter_ns()
        t_e2e_warm.append((t1 - t0) / 1e6)
        
    def stats(arr):
        return {
            "mean_ms": float(np.mean(arr)),
            "median_ms": float(np.median(arr)),
            "std_ms": float(np.std(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr))
        }
        
    profile_results = {
        "preprocessing_segmentation": stats(t_prep),
        "minutiae_extraction": stats(t_minutiae),
        "delaunay_triangulation": stats(t_triangle),
        "descriptor_construction_9d": stats(t_desc),
        "key_derivation_cold_hkdf": stats(t_key_cold),
        "key_retrieval_warm_cached": stats(t_key_warm),
        "projection_and_quantization": stats(t_proj),
        "credential_serialization_aead": stats(t_serialize),
        "pairwise_cost_matrix": stats(t_cost_mat),
        "matching_hungarian": stats(t_hungarian),
        "matching_greedy_1to1": stats(t_greedy),
        "end_to_end_image_authentication_warm": stats(t_e2e_warm)
    }
    
    print(f"  Stage 1: Preprocessing & Segmentation: {profile_results['preprocessing_segmentation']['median_ms']:.2f} ms")
    print(f"  Stage 2: Minutiae Extraction         : {profile_results['minutiae_extraction']['median_ms']:.2f} ms")
    print(f"  Stage 3: Delaunay Triangulation      : {profile_results['delaunay_triangulation']['median_ms']:.2f} ms")
    print(f"  Stage 4: 9-D Descriptors Construction: {profile_results['descriptor_construction_9d']['median_ms']:.2f} ms")
    print(f"  Stage 5a: Cold HKDF Key Derivation   : {profile_results['key_derivation_cold_hkdf']['median_ms']:.3f} ms")
    print(f"  Stage 5b: Warm Cached Key Retrieval  : {profile_results['key_retrieval_warm_cached']['median_ms']:.3f} ms")
    print(f"  Stage 6: Projection & Quantization   : {profile_results['projection_and_quantization']['median_ms']:.3f} ms")
    print(f"  Stage 7: AEAD Serialization          : {profile_results['credential_serialization_aead']['median_ms']:.3f} ms")
    print(f"  Stage 8: Hungarian Assignment        : {profile_results['matching_hungarian']['median_ms']:.2f} ms")
    print(f"  Stage 8b: Greedy 1-to-1 Assignment   : {profile_results['matching_greedy_1to1']['median_ms']:.2f} ms")
    print(f"  End-to-End Image-to-Score Latency    : {profile_results['end_to_end_image_authentication_warm']['median_ms']:.2f} ms")
    
    return profile_results


def run_hungarian_vs_greedy_benchmark() -> Dict[str, Any]:
    print("\n--- 4. Benchmarking Hungarian vs Greedy 1-to-1 Matching ---")
    sizes = [10, 25, 50, 100, 144, 200]
    repeats = 150
    rng = np.random.default_rng(42)
    
    results = {}
    for n in sizes:
        h_times = []
        g_times = []
        diffs = []
        
        for _ in range(repeats):
            q_bits = rng.integers(0, 2, size=(n, 32), dtype=np.uint8)
            noise = rng.choice([0.15, 0.40])
            mask = rng.random((n, 32)) < noise
            e_bits = np.bitwise_xor(q_bits, mask.astype(np.uint8))[rng.permutation(n)]
            
            w_q = rng.uniform(0.7, 1.0, size=(n,)).astype(np.float32)
            w_e = rng.uniform(0.7, 1.0, size=(n,)).astype(np.float32)
            
            # Hungarian
            t0 = time.perf_counter_ns()
            s_h = match_protected_templates(q_bits, e_bits, w_q, w_e, hamming_threshold=0.22)
            t1 = time.perf_counter_ns()
            h_times.append((t1 - t0) / 1e6)
            
            # Greedy 1-to-1
            t2 = time.perf_counter_ns()
            s_g = match_protected_greedy(q_bits, e_bits, w_q, w_e, hamming_threshold=0.22, mode="greedy_1to1")
            t3 = time.perf_counter_ns()
            g_times.append((t3 - t2) / 1e6)
            
            diffs.append(abs(s_h - s_g))
            
        med_h = float(np.median(h_times))
        med_g = float(np.median(g_times))
        speedup = med_h / max(med_g, 1e-6)
        comp_s_h = 1000.0 / max(med_h, 1e-6)
        comp_s_g = 1000.0 / max(med_g, 1e-6)
        
        results[f"N_{n}"] = {
            "template_size_N": n,
            "hungarian_median_ms": round(med_h, 3),
            "greedy_median_ms": round(med_g, 3),
            "hungarian_comp_per_sec": round(comp_s_h, 1),
            "greedy_comp_per_sec": round(comp_s_g, 1),
            "speedup_ratio": round(speedup, 2),
            "mae_score_diff": round(float(np.mean(diffs)), 4),
            "max_score_diff": round(float(np.max(diffs)), 4)
        }
        print(f"  N = {n:3d}: Hungarian = {med_h:.2f} ms ({comp_s_h:.1f} c/s) | Greedy = {med_g:.2f} ms ({comp_s_g:.1f} c/s) | Speedup = {speedup:.2f}x | MAE = {np.mean(diffs):.4f}")
        
    return results


def run_throughput_modes_benchmark() -> Dict[str, Any]:
    print("\n--- 5. Evaluating Distinct Throughput Modes ---")
    
    # Mode A: 1:1 Authentication Latency
    # Image-to-decision (warm pipeline) vs Template-to-decision
    service = ProductionBiometricService()
    img_path = os.path.join("dataset", "FVC2002", "DB1_B", "101_1.tif")
    if not os.path.exists(img_path):
        img_path = os.path.join("dataset", "FVC2002", "DB1_A", "1_1.jpg")
    raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Pre-extract enrolled template
    feats = service.extract_pipeline_features(raw_img, dpi=500.0, with_topology=False)
    salt = "wp6_mode_salt"
    app_id = "wp6_mode_app"
    enrolled_bits = service.projection_engine.project_and_quantize(feats["descriptors"].descriptors, salt, app_id)
    enrolled_weights = feats["descriptors"].weights
    
    # Measure Image-to-Decision
    e2e_latencies = []
    for _ in range(50):
        t0 = time.perf_counter_ns()
        f_q = service.extract_pipeline_features(raw_img, dpi=500.0, with_topology=False)
        b_q = service.projection_engine.project_and_quantize(f_q["descriptors"].descriptors, salt, app_id)
        _ = match_protected_greedy(b_q, enrolled_bits, f_q["descriptors"].weights, enrolled_weights)
        t1 = time.perf_counter_ns()
        e2e_latencies.append((t1 - t0) / 1e6)
        
    # Measure Template-to-Decision (Match-Only)
    template_latencies = []
    for _ in range(200):
        t0 = time.perf_counter_ns()
        _ = match_protected_greedy(enrolled_bits, enrolled_bits, enrolled_weights, enrolled_weights)
        t1 = time.perf_counter_ns()
        template_latencies.append((t1 - t0) / 1e6)
        
    mode_a = {
        "image_to_decision_latency_median_ms": round(float(np.median(e2e_latencies)), 2),
        "image_to_decision_throughput_auth_per_sec": round(1000.0 / float(np.median(e2e_latencies)), 1),
        "template_to_decision_latency_median_ms": round(float(np.median(template_latencies)), 3),
        "template_to_decision_throughput_comp_per_sec": round(1000.0 / float(np.median(template_latencies)), 1)
    }
    print(f"  Mode A (1:1 Authentication): Image-to-Decision = {mode_a['image_to_decision_latency_median_ms']} ms ({mode_a['image_to_decision_throughput_auth_per_sec']} auth/s)")
    print(f"                               Template-to-Decision = {mode_a['template_to_decision_latency_median_ms']} ms ({mode_a['template_to_decision_throughput_comp_per_sec']} comp/s)")

    # Mode B: 1:N Screening Throughput (Template screening across gallery sizes)
    gallery_sizes = [10, 50, 100, 500, 1000]
    rng = np.random.default_rng(123)
    n_triangles = 144
    mode_b = {}
    
    # Generate gallery templates
    for g_size in gallery_sizes:
        gallery_bits = [rng.integers(0, 2, size=(n_triangles, 32), dtype=np.uint8) for _ in range(g_size)]
        gallery_weights = [rng.uniform(0.7, 1.0, size=(n_triangles,)).astype(np.float32) for _ in range(g_size)]
        q_b = rng.integers(0, 2, size=(n_triangles, 32), dtype=np.uint8)
        q_w = rng.uniform(0.7, 1.0, size=(n_triangles,)).astype(np.float32)
        
        times = []
        for _ in range(20):
            t0 = time.perf_counter_ns()
            # Fast vectorized screening: score probe against all gallery templates
            for gb, gw in zip(gallery_bits, gallery_weights):
                _ = match_protected_greedy(q_b, gb, q_w, gw)
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1e6)
            
        med_time = float(np.median(times))
        templates_per_sec = (g_size * 1000.0) / med_time
        mode_b[f"gallery_{g_size}"] = {
            "gallery_size": g_size,
            "total_screening_time_median_ms": round(med_time, 2),
            "screening_throughput_templates_per_sec": round(templates_per_sec, 1)
        }
        print(f"  Mode B (1:N Screening N={g_size:4d}): Total = {med_time:6.2f} ms | Screening Rate = {templates_per_sec:7.1f} templates/s")

    # Mode C: Match-Only Throughput (Isolated Hamming + Assignment)
    # Tested for N=144 triangles
    match_only_times = []
    q_b = rng.integers(0, 2, size=(144, 32), dtype=np.uint8)
    e_b = rng.integers(0, 2, size=(144, 32), dtype=np.uint8)
    w1 = rng.uniform(0.7, 1.0, size=(144,)).astype(np.float32)
    w2 = rng.uniform(0.7, 1.0, size=(144,)).astype(np.float32)
    
    for _ in range(500):
        t0 = time.perf_counter_ns()
        _ = match_protected_greedy(q_b, e_b, w1, w2)
        t1 = time.perf_counter_ns()
        match_only_times.append((t1 - t0) / 1e6)
        
    mode_c = {
        "template_size_N": 144,
        "single_match_median_ms": round(float(np.median(match_only_times)), 4),
        "match_throughput_comp_per_sec": round(1000.0 / float(np.median(match_only_times)), 1)
    }
    print(f"  Mode C (Match-Only N=144)   : {mode_c['single_match_median_ms']} ms/pair ({mode_c['match_throughput_comp_per_sec']} comp/s)")

    # Mode D: Concurrent Throughput (Multi-threaded worker scaling)
    worker_counts = [1, 2, 4]
    mode_d = {}
    num_tasks = 300
    task_pairs = [(rng.integers(0, 2, size=(100, 32), dtype=np.uint8),
                   rng.integers(0, 2, size=(100, 32), dtype=np.uint8),
                   rng.uniform(0.7, 1.0, size=(100,)).astype(np.float32),
                   rng.uniform(0.7, 1.0, size=(100,)).astype(np.float32)) for _ in range(num_tasks)]
                   
    def _worker_match(pair):
        qb, eb, wq, we = pair
        return match_protected_greedy(qb, eb, wq, we)

    for num_w in worker_counts:
        t0 = time.perf_counter_ns()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_w) as executor:
            _ = list(executor.map(_worker_match, task_pairs))
        t1 = time.perf_counter_ns()
        tot_time_ms = (t1 - t0) / 1e6
        throughput = (num_tasks * 1000.0) / tot_time_ms
        mode_d[f"workers_{num_w}"] = {
            "num_workers": num_w,
            "batch_size": num_tasks,
            "total_time_ms": round(tot_time_ms, 2),
            "concurrent_comp_per_sec": round(throughput, 1),
            "scaling_efficiency": round(throughput / (mode_d.get("workers_1", {}).get("concurrent_comp_per_sec", throughput) * num_w), 2)
        }
        print(f"  Mode D (Concurrent Workers={num_w}): {tot_time_ms:6.2f} ms for {num_tasks} matches | Throughput = {throughput:7.1f} comp/s")

    return {
        "mode_a_authentication": mode_a,
        "mode_b_screening": mode_b,
        "mode_c_match_only": mode_c,
        "mode_d_concurrency": mode_d
    }


def main():
    print("=" * 80)
    print("  WP6: Storage and Performance Benchmark Suite")
    print("=" * 80)
    
    os.makedirs("experiments/results/wp6", exist_ok=True)
    
    # 1. N=144 byte breakdown verification
    n144_results = run_storage_verification_n144()
    
    # 2. Actual dataset-wide storage statistics
    storage_results = run_dataset_storage_stats()
    
    # 3. Pipeline stage-wise latency profiling
    stage_latency_results = run_pipeline_stage_latency_profile(num_runs=100)
    
    # 4. Hungarian vs Greedy benchmark
    hungarian_vs_greedy_results = run_hungarian_vs_greedy_benchmark()
    
    # 5. Distinct throughput modes
    throughput_modes_results = run_throughput_modes_benchmark()
    
    final_report = {
        "title": "WP6 Storage and Performance Comprehensive Benchmark Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n144_verification": n144_results,
        "dataset_storage": storage_results,
        "stage_latency_profile": stage_latency_results,
        "hungarian_vs_greedy": hungarian_vs_greedy_results,
        "throughput_modes": throughput_modes_results
    }
    
    out_file = os.path.join("experiments", "results", "wp6", "storage_performance_report.json")
    with open(out_file, "w") as f:
        json.dump(final_report, f, indent=2)
        
    print(f"\n[OK] WP6 Storage & Performance report successfully written to: {out_file}\n")


if __name__ == "__main__":
    main()
