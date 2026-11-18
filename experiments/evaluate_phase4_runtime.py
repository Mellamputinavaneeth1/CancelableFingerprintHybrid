"""
Phase 4: Runtime & Latency Profiling Benchmark.

Profiles the 10 distinct pipeline stages using high-resolution monotonic timers
(time.perf_counter_ns) with warm-up iterations and repetitions:
1. Image validation & preprocessing
2. Minutiae extraction
3. Triangle generation
4. 9-D descriptor construction
5. Key derivation & projection generation (cold HKDF vs warm cached)
6. Bitstring generation (matrix projection + 1-bit sign quantization)
7. Pairwise cost construction (Hamming distance + slack augmentation)
8. Assignment: Hungarian bipartite matching vs Greedy 1-to-1 matching
9. Graph consensus (optional ablation)
10. Final score & threshold decision

Separates interactive single-query latency from offline batch throughput.
Generates latency reports and visualization artifacts.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.orchestrator import ProductionBiometricService
from src.matching.bipartite_matcher import (
    build_augmented_cost,
    match_protected_templates,
    match_protected_greedy
)
from src.matching.graph_consensus import (
    compute_triangle_adjacency,
    match_protected_with_graph_consensus
)
from scipy.optimize import linear_sum_assignment


def calc_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    arr = np.array(latencies_ms, dtype=np.float64)
    med = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    return {
        "mean_ms": round(mean, 4),
        "std_ms": round(std, 4),
        "median_ms": round(med, 4),
        "p95_ms": round(p95, 4),
        "min_ms": round(min_v, 4),
        "max_ms": round(max_v, 4)
    }


def run_runtime_profile(num_warmup: int = 10,
                        num_repetitions: int = 100,
                        output_dir: str = "experiments/results/phase4") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    service = ProductionBiometricService()

    # Load representative real fingerprint images from DB1_B or DB1_A
    img1_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "FVC2002", "DB1_B", "101_1.tif")
    img2_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "FVC2002", "DB1_B", "101_2.tif")

    if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
        # Try DB1_A
        img1_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "FVC2002", "DB1_A", "1_1.jpg")
        img2_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "FVC2002", "DB1_A", "1_2.jpg")

    raw_img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    raw_img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    app_id = "runtime_benchmark_app"
    salt = "runtime_benchmark_salt_2026"
    subj_id = "subj_profile_01"

    print(f"\n================================================================================")
    print(f"  PHASE 4 RUNTIME & LATENCY PROFILING (10 Distinct Pipeline Stages)")
    print(f"  Warm-up: {num_warmup} runs | Repetitions: {num_repetitions} runs")
    print(f"================================================================================")

    # Pre-extract intermediate representations to isolate individual stages
    cap1 = service.capture_adapter.ingest(raw_img1, dpi=500.0)
    cap2 = service.capture_adapter.ingest(raw_img2, dpi=500.0)
    prep1 = service.preprocessor.process(cap1)
    prep2 = service.preprocessor.process(cap2)
    min1 = service.minutiae_extractor.extract(prep1)
    min2 = service.minutiae_extractor.extract(prep2)
    tri1 = service.triangle_builder.build(min1, compute_topology=True)
    tri2 = service.triangle_builder.build(min2, compute_topology=True)
    desc1 = service.descriptor_builder.build(tri1, dpi=500.0)
    desc2 = service.descriptor_builder.build(tri2, dpi=500.0)

    W = service.projection_engine.get_projection_matrix(salt, app_id, in_dim=9, out_dim=32)
    bits1 = service.projection_engine.project_and_quantize(desc1.descriptors, salt, app_id)
    bits2 = service.projection_engine.project_and_quantize(desc2.descriptors, salt, app_id)

    # -------------------------------------------------------------
    # Warm-up phase
    # -------------------------------------------------------------
    print("Executing warm-up iterations...")
    for _ in range(num_warmup):
        _ = service.capture_adapter.ingest(raw_img1)
        _ = service.preprocessor.process(cap1)
        _ = service.minutiae_extractor.extract(prep1)
        _ = service.triangle_builder.build(min1)
        _ = service.descriptor_builder.build(tri1)
        _ = service.key_service.derive_projection_matrix(salt, app_id)
        _ = service.projection_engine.project_and_quantize(desc1.descriptors, salt, app_id)
        _ = match_protected_templates(bits1, bits2, desc1.weights, desc2.weights)
        _ = match_protected_greedy(bits1, bits2, desc1.weights, desc2.weights, mode="greedy_1to1")

    # -------------------------------------------------------------
    # Benchmark the 10 distinct stages
    # -------------------------------------------------------------
    times_stage1 = []  # Image validation & preprocessing
    times_stage2 = []  # Minutiae extraction
    times_stage3 = []  # Triangle generation
    times_stage4 = []  # 9-D descriptor construction
    times_stage5_cold = []  # Key derivation & projection generation (cold HKDF)
    times_stage5_warm = []  # Key retrieval (warm cache)
    times_stage6 = []  # Bitstring generation
    times_stage7 = []  # Pairwise cost construction
    times_stage8_hungarian = []  # Hungarian assignment
    times_stage8_greedy = []  # Greedy 1-to-1 assignment
    times_stage9_consensus = []  # Graph consensus
    times_stage10_decision = []  # Final score & decision
    times_e2e_cold = []  # End-to-end interactive single query (cold)
    times_e2e_warm = []  # End-to-end interactive single query (warm)

    for i in range(num_repetitions):
        # Stage 1: Image validation & preprocessing
        t0 = time.perf_counter_ns()
        c = service.capture_adapter.ingest(raw_img1)
        p = service.preprocessor.process(c)
        t1 = time.perf_counter_ns()
        times_stage1.append((t1 - t0) / 1e6)

        # Stage 2: Minutiae extraction
        t0 = time.perf_counter_ns()
        m = service.minutiae_extractor.extract(prep1)
        t1 = time.perf_counter_ns()
        times_stage2.append((t1 - t0) / 1e6)

        # Stage 3: Triangle generation
        t0 = time.perf_counter_ns()
        t = service.triangle_builder.build(min1, compute_topology=False)
        t1 = time.perf_counter_ns()
        times_stage3.append((t1 - t0) / 1e6)

        # Stage 4: 9-D descriptor construction
        t0 = time.perf_counter_ns()
        d = service.descriptor_builder.build(tri1, dpi=500.0)
        t1 = time.perf_counter_ns()
        times_stage4.append((t1 - t0) / 1e6)

        # Stage 5a: Cold key derivation
        t0 = time.perf_counter_ns()
        w_cold = service.key_service.derive_projection_matrix(f"salt_{i}", app_id, in_dim=9, out_dim=32)
        t1 = time.perf_counter_ns()
        times_stage5_cold.append((t1 - t0) / 1e6)

        # Stage 5b: Warm cached key retrieval
        t0 = time.perf_counter_ns()
        w_warm = service.projection_engine.get_projection_matrix(salt, app_id, in_dim=9, out_dim=32)
        t1 = time.perf_counter_ns()
        times_stage5_warm.append((t1 - t0) / 1e6)

        # Stage 6: Bitstring generation
        t0 = time.perf_counter_ns()
        b_gen = service.projection_engine.project_and_quantize(desc1.descriptors, salt, app_id)
        t1 = time.perf_counter_ns()
        times_stage6.append((t1 - t0) / 1e6)

        # Stage 7: Pairwise cost construction
        t0 = time.perf_counter_ns()
        xor_sum = np.bitwise_xor(bits1[:, None, :], bits2[None, :, :]).sum(axis=2)
        dist_mat = xor_sum / 32.0
        aug_cost = build_augmented_cost(dist_mat, tau=0.22)
        t1 = time.perf_counter_ns()
        times_stage7.append((t1 - t0) / 1e6)

        # Stage 8a: Hungarian assignment
        t0 = time.perf_counter_ns()
        row_ind, col_ind = linear_sum_assignment(aug_cost)
        t1 = time.perf_counter_ns()
        times_stage8_hungarian.append((t1 - t0) / 1e6)

        # Stage 8b: Greedy 1-to-1 assignment (assignment step on pairwise distance matrix)
        t0 = time.perf_counter_ns()
        valid_pairs = np.argwhere(dist_mat < 0.22)
        if len(valid_pairs) > 0:
            pair_dists = dist_mat[valid_pairs[:, 0], valid_pairs[:, 1]]
            sort_order = np.argsort(pair_dists, kind="stable")
            matched_q, matched_e = set(), set()
            for idx in sort_order:
                r, c = int(valid_pairs[idx, 0]), int(valid_pairs[idx, 1])
                if r not in matched_q and c not in matched_e:
                    matched_q.add(r)
                    matched_e.add(c)
        t1 = time.perf_counter_ns()
        times_stage8_greedy.append((t1 - t0) / 1e6)

        # Stage 9: Graph consensus (optional ablation)
        t0 = time.perf_counter_ns()
        _ = service.matcher.match_protected_with_topology(
            bits1, bits2, desc1.weights, desc2.weights,
            e_adj=tri2.adjacency_matrix, q_adj=tri1.adjacency_matrix, alpha=0.25
        )
        t1 = time.perf_counter_ns()
        times_stage9_consensus.append((t1 - t0) / 1e6)

        # Stage 10: Final score & decision
        score_val = 0.3850
        t0 = time.perf_counter_ns()
        tau = service.threshold_registry.get_threshold("DB1_A", "EER")
        dec = "ACCEPT" if score_val >= tau else "REJECT"
        t1 = time.perf_counter_ns()
        times_stage10_decision.append((t1 - t0) / 1e6)

    # -------------------------------------------------------------
    # Measure Offline Batch Throughput (Pure Matcher Loops)
    # -------------------------------------------------------------
    batch_n = 500
    print(f"Measuring offline batch matcher throughput ({batch_n} comparisons)...")
    
    # 1. Genuine workload (dense matches)
    t0 = time.perf_counter()
    for _ in range(batch_n):
        _ = match_protected_templates(bits1, bits2, desc1.weights, desc2.weights, hamming_threshold=0.22)
    t1 = time.perf_counter()
    hungarian_gen_comp_sec = batch_n / max(t1 - t0, 1e-6)

    t0 = time.perf_counter()
    for _ in range(batch_n):
        _ = match_protected_greedy(bits1, bits2, desc1.weights, desc2.weights, hamming_threshold=0.22, mode="greedy_1to1")
    t1 = time.perf_counter()
    greedy_gen_comp_sec = batch_n / max(t1 - t0, 1e-6)

    # 2. Impostor screening workload (sparse/zero matches, 1:N operational workload)
    rng = np.random.default_rng(2026)
    bits_imp = rng.integers(0, 2, size=(len(bits1), 32), dtype=np.uint8)
    weights_imp = rng.uniform(0.5, 1.0, size=(len(bits1),)).astype(np.float32)

    t0 = time.perf_counter()
    for _ in range(batch_n):
        _ = match_protected_templates(bits1, bits_imp, desc1.weights, weights_imp, hamming_threshold=0.22)
    t1 = time.perf_counter()
    hungarian_imp_comp_sec = batch_n / max(t1 - t0, 1e-6)

    t0 = time.perf_counter()
    for _ in range(batch_n):
        _ = match_protected_greedy(bits1, bits_imp, desc1.weights, weights_imp, hamming_threshold=0.22, mode="greedy_1to1")
    t1 = time.perf_counter()
    greedy_imp_comp_sec = batch_n / max(t1 - t0, 1e-6)

    # Interactive single-query latency (full image to decision)
    # First enroll image 2
    service.enroll(raw_img2, subj_id, salt, app_id)
    
    for _ in range(20):
        t0 = time.perf_counter()
        _ = service.verify(raw_img1, subj_id, app_id, dataset="DB1_A")
        t1 = time.perf_counter()
        times_e2e_warm.append((t1 - t0) * 1000.0)

    # Assemble report
    report = {
        "benchmark": "Phase 4 Runtime & Latency Profiling",
        "num_warmup": num_warmup,
        "num_repetitions": num_repetitions,
        "stages": {
            "stage_1_preprocessing": calc_latency_stats(times_stage1),
            "stage_2_minutiae_extraction": calc_latency_stats(times_stage2),
            "stage_3_triangle_generation": calc_latency_stats(times_stage3),
            "stage_4_descriptor_construction": calc_latency_stats(times_stage4),
            "stage_5_key_derivation_cold": calc_latency_stats(times_stage5_cold),
            "stage_5_key_retrieval_warm": calc_latency_stats(times_stage5_warm),
            "stage_6_bitstring_generation": calc_latency_stats(times_stage6),
            "stage_7_cost_construction": calc_latency_stats(times_stage7),
            "stage_8_hungarian_assignment": calc_latency_stats(times_stage8_hungarian),
            "stage_8_greedy_assignment": calc_latency_stats(times_stage8_greedy),
            "stage_9_graph_consensus": calc_latency_stats(times_stage9_consensus),
            "stage_10_decision": calc_latency_stats(times_stage10_decision)
        },
        "interactive_single_query_latency": {
            "full_pipeline_warm_cache": calc_latency_stats(times_e2e_warm),
            "interactive_queries_per_second": round(1000.0 / max(np.median(times_e2e_warm), 1e-6), 2)
        },
        "offline_batch_throughput": {
            "hungarian_comparisons_per_second": round(hungarian_gen_comp_sec, 2),
            "greedy_1to1_comparisons_per_second": round(greedy_gen_comp_sec, 2),
            "speedup_factor_greedy_vs_hungarian": round(greedy_gen_comp_sec / max(hungarian_gen_comp_sec, 1e-6), 2),
            "genuine_workload": {
                "hungarian_comp_per_second": round(hungarian_gen_comp_sec, 2),
                "greedy_1to1_comp_per_second": round(greedy_gen_comp_sec, 2)
            },
            "impostor_screening_workload": {
                "hungarian_comp_per_second": round(hungarian_imp_comp_sec, 2),
                "greedy_1to1_comp_per_second": round(greedy_imp_comp_sec, 2),
                "speedup_factor_greedy_vs_hungarian": round(greedy_imp_comp_sec / max(hungarian_imp_comp_sec, 1e-6), 2)
            }
        }
    }

    report_path = os.path.join(output_dir, "runtime_benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved complete runtime benchmark report to: {report_path}")

    # Generate Runtime Profile Visualization
    generate_runtime_visualizations(report, output_dir)

    print("\n================================================================================")
    print("  PHASE 4 RUNTIME BENCHMARK RESULTS")
    print("================================================================================")
    for stage_name, s_stats in report["stages"].items():
        print(f"  {stage_name:35s}: Median = {s_stats['median_ms']:8.4f} ms | P95 = {s_stats['p95_ms']:8.4f} ms")
    print("--------------------------------------------------------------------------------")
    print(f"  Interactive Single-Query Latency: Median = {report['interactive_single_query_latency']['full_pipeline_warm_cache']['median_ms']:.2f} ms ({report['interactive_single_query_latency']['interactive_queries_per_second']:.1f} queries/s)")
    print(f"  Offline Batch Throughput Genuine (Hungarian): {report['offline_batch_throughput']['genuine_workload']['hungarian_comp_per_second']:.1f} comp/s")
    print(f"  Offline Batch Throughput Genuine (Greedy 1-to-1): {report['offline_batch_throughput']['genuine_workload']['greedy_1to1_comp_per_second']:.1f} comp/s ({report['offline_batch_throughput']['speedup_factor_greedy_vs_hungarian']:.1f}x speedup)")
    print(f"  Offline Batch Throughput Impostor 1:N (Greedy 1-to-1): {report['offline_batch_throughput']['impostor_screening_workload']['greedy_1to1_comp_per_second']:.1f} comp/s ({report['offline_batch_throughput']['impostor_screening_workload']['speedup_factor_greedy_vs_hungarian']:.1f}x speedup)")
    print("================================================================================\n")

    return report


def generate_runtime_visualizations(report: Dict[str, Any], output_dir: str):
    """Generates runtime stage-wise latency profile bar chart."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    stages = [
        ("1. Preprocess", report["stages"]["stage_1_preprocessing"]["median_ms"]),
        ("2. Minutiae", report["stages"]["stage_2_minutiae_extraction"]["median_ms"]),
        ("3. Triangles", report["stages"]["stage_3_triangle_generation"]["median_ms"]),
        ("4. 9-D Descriptors", report["stages"]["stage_4_descriptor_construction"]["median_ms"]),
        ("5. Key Derivation (Cold)", report["stages"]["stage_5_key_derivation_cold"]["median_ms"]),
        ("5. Key Cache (Warm)", report["stages"]["stage_5_key_retrieval_warm"]["median_ms"]),
        ("6. Bitstrings", report["stages"]["stage_6_bitstring_generation"]["median_ms"]),
        ("7. Pairwise Cost", report["stages"]["stage_7_cost_construction"]["median_ms"]),
        ("8. Hungarian Assign", report["stages"]["stage_8_hungarian_assignment"]["median_ms"]),
        ("8. Greedy 1-to-1", report["stages"]["stage_8_greedy_assignment"]["median_ms"]),
        ("9. Topology Consensus", report["stages"]["stage_9_graph_consensus"]["median_ms"]),
        ("10. Decision", report["stages"]["stage_10_decision"]["median_ms"])
    ]

    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#4e79a7" if "8." not in l and "9." not in l and "5." not in l else
              ("#e15759" if "Hungarian" in l else ("#59a14f" if "Greedy" in l else "#f28e2b")) for l in labels]
    
    bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.invert_yaxis()  # Top-down pipeline order
    ax.set_xlabel("Median Latency (ms)", fontsize=11, fontweight="bold")
    ax.set_title("Phase 4: Monotonic Stage-Wise Latency Profile Across 10 Pipeline Stages", 
                 fontsize=13, fontweight="bold", pad=15)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + (0.1 if w > 0.5 else 0.02), bar.get_y() + bar.get_height() / 2.0, 
                f"{w:.3f} ms", va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_xlim(0, max(values) * 1.18)
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "runtime_profile_bar.png")
    fig.savefig(chart_path, dpi=300)
    plt.close(fig)
    print(f"Saved runtime profile chart to: {chart_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Phase 4 Runtime & Latency Profile.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--repetitions", type=int, default=100, help="Timing repetitions")
    parser.add_argument("--output_dir", type=str, default="experiments/results/phase4", help="Output directory")
    args = parser.parse_args()
    run_runtime_profile(args.warmup, args.repetitions, args.output_dir)
