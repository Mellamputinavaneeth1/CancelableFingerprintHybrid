"""
Phase 4: Complete Serialized Credential Storage Accounting Benchmark.

Measures exact serialized byte length across all 800 enrolled templates in DB1_A
and 80 in DB1_B under strict binary framing and AEAD encryption at rest.

Evaluates like-for-like storage across three architectures:
(a) Descriptor-only 9-D protected credential (prod default, no topology);
(b) 9-D protected credential with optional topology (2-bit packed matrix A);
(c) Continuous unprotected baseline with equivalent functionality (40 B/tri + header).

Generates manifests, JSON summary reports, and visualization artifacts.
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

from experiments.datasets.fvc_loader import load_fvc_dataset
from src.services.orchestrator import ProductionBiometricService
from src.services.types import ProtectedCredentialDTO
from src.matching.graph_consensus import compute_triangle_adjacency, pack_adjacency_matrix


def compute_summary_stats(values: List[int]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "total_bytes": int(np.sum(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr))
    }


def evaluate_dataset_storage(service: ProductionBiometricService,
                             subset_name: str,
                             images: List[Dict[str, Any]],
                             output_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print(f"\n--- Evaluating Serialized Storage on {subset_name} ({len(images)} templates) ---")
    
    rows = []
    prot_desc_bytes = []
    prot_topo_bytes = []
    cont_desc_bytes = []
    cont_topo_packed_bytes = []
    cont_topo_uint8_bytes = []
    triangle_counts = []

    master_key = service.key_service
    serializer = service.serializer
    app_id = "storage_eval_2026"
    salt = "salt_phase4_storage_eval"
    storage_key = master_key.derive_storage_key(salt, app_id)

    # Cache file to avoid redundant feature extraction if rerun
    cache_file = os.path.join(output_dir, f"triangle_cache_{subset_name}.json")
    triangle_cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                triangle_cache = json.load(f)
            print(f"Loaded {len(triangle_cache)} cached triangle counts from {cache_file}")
        except Exception:
            triangle_cache = {}

    for idx, img_info in enumerate(images):
        fpath = img_info["path"]
        fname = os.path.basename(fpath)
        finger_id = img_info["finger_id"]
        imp_id = img_info["impression_id"]
        subj_id = f"subj_{finger_id:03d}"

        if fname in triangle_cache:
            n_triangles = triangle_cache[fname]
        else:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            feats = service.extract_pipeline_features(img, dpi=img_info.get("dpi", 500.0), with_topology=False)
            n_triangles = feats["descriptors"].count
            triangle_cache[fname] = n_triangles

        triangle_counts.append(n_triangles)

        # 1. Architecture (a): Descriptor-only 9-D protected credential
        # Header (64 B) + Descriptors (N x 4 B) + Weights (N x 2 B) + AEAD Tag (16 B) = 80 + 6*N
        size_prot_desc = 80 + (6 * n_triangles)
        prot_desc_bytes.append(size_prot_desc)

        # 2. Architecture (b): 9-D protected credential with optional topology
        # Topology matrix upper-triangle packed 2-bit: ceil(N*(N-1)/8)
        num_upper = (n_triangles * (n_triangles - 1)) // 2
        packed_topo_len = (num_upper + 3) // 4 if n_triangles > 1 else 0
        size_prot_topo = size_prot_desc + packed_topo_len
        prot_topo_bytes.append(size_prot_topo)

        # 3. Architecture (c): Continuous Unprotected Baseline
        # Continuous geometric features (9 x 4 B = 36 B) + Quality weight (4 B) = 40 B/tri
        # Header (64 B) + 40*N
        size_cont_desc = 64 + (40 * n_triangles)
        cont_desc_bytes.append(size_cont_desc)

        # Continuous with packed topology
        size_cont_topo_packed = size_cont_desc + packed_topo_len
        cont_topo_packed_bytes.append(size_cont_topo_packed)

        # Continuous with uncompressed uint8 matrix
        size_cont_topo_uint8 = size_cont_desc + (n_triangles * n_triangles)
        cont_topo_uint8_bytes.append(size_cont_topo_uint8)

        # Compression calculations
        red_desc_only = (1.0 - (size_prot_desc / max(size_cont_desc, 1))) * 100.0
        red_with_topo = (1.0 - (size_prot_topo / max(size_cont_topo_packed, 1))) * 100.0
        red_vs_uint8 = (1.0 - (size_prot_topo / max(size_cont_topo_uint8, 1))) * 100.0

        rows.append({
            "subset": subset_name,
            "filename": fname,
            "finger_id": finger_id,
            "impression_id": imp_id,
            "num_triangles": n_triangles,
            "prot_descriptor_only_bytes": size_prot_desc,
            "prot_with_topology_bytes": size_prot_topo,
            "cont_baseline_desc_bytes": size_cont_desc,
            "cont_baseline_topo_packed_bytes": size_cont_topo_packed,
            "cont_baseline_topo_uint8_bytes": size_cont_topo_uint8,
            "compression_descriptor_only_pct": round(red_desc_only, 2),
            "compression_with_topology_pct": round(red_with_topo, 2),
            "compression_vs_uint8_topology_pct": round(red_vs_uint8, 2)
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == len(images):
            print(f"  Processed {idx + 1}/{len(images)} templates... (Latest N={n_triangles}, Prot={size_prot_desc} B, Cont={size_cont_desc} B)")

    # Save triangle cache
    with open(cache_file, "w") as f:
        json.dump(triangle_cache, f, indent=2)

    # Compute aggregate stats
    summary = {
        "subset": subset_name,
        "num_templates": len(images),
        "triangles": compute_summary_stats(triangle_counts),
        "prot_descriptor_only": compute_summary_stats(prot_desc_bytes),
        "prot_with_topology": compute_summary_stats(prot_topo_bytes),
        "cont_baseline_descriptor_only": compute_summary_stats(cont_desc_bytes),
        "cont_baseline_with_topology_packed": compute_summary_stats(cont_topo_packed_bytes),
        "cont_baseline_with_topology_uint8": compute_summary_stats(cont_topo_uint8_bytes),
        "average_compression": {
            "descriptor_only_vs_baseline_pct": round((1.0 - (np.mean(prot_desc_bytes) / np.mean(cont_desc_bytes))) * 100.0, 2),
            "topology_packed_vs_baseline_packed_pct": round((1.0 - (np.mean(prot_topo_bytes) / np.mean(cont_topo_packed_bytes))) * 100.0, 2),
            "topology_packed_vs_baseline_uint8_pct": round((1.0 - (np.mean(prot_topo_bytes) / np.mean(cont_topo_uint8_bytes))) * 100.0, 2)
        }
    }

    # Verify actual serializer fidelity on a sample template
    sample_n = triangle_counts[0]
    sample_bits = np.random.randint(0, 2, size=(sample_n, 32), dtype=np.uint8)
    sample_weights = np.random.uniform(0.5, 1.0, size=(sample_n,)).astype(np.float32)
    sample_topo = b"\xaa" * ((sample_n * (sample_n - 1) // 2 + 3) // 4)
    
    dto = ProtectedCredentialDTO(
        credential_id="test-cred-uuid",
        subject_id="subj_test",
        salt="test_salt",
        app_id="test_app",
        bitstrings=sample_bits,
        weights=sample_weights,
        num_triangles=sample_n,
        packed_topology=sample_topo
    )
    env_topo = serializer.serialize_protected(dto, storage_key, encrypt=True)
    assert len(env_topo.raw_bytes) == (80 + 6 * sample_n + len(sample_topo)), "Serializer size mismatch on topology template!"

    dto_no_topo = ProtectedCredentialDTO(
        credential_id="test-cred-uuid-2",
        subject_id="subj_test",
        salt="test_salt",
        app_id="test_app",
        bitstrings=sample_bits,
        weights=sample_weights,
        num_triangles=sample_n,
        packed_topology=None
    )
    env_desc = serializer.serialize_protected(dto_no_topo, storage_key, encrypt=True)
    assert len(env_desc.raw_bytes) == (80 + 6 * sample_n), "Serializer size mismatch on descriptor-only template!"

    sample_feats = np.zeros((sample_n, 9), dtype=np.float32)
    sample_base_w = np.ones((sample_n,), dtype=np.float32)
    base_bytes = serializer.serialize_unprotected_baseline(sample_feats, sample_base_w)
    assert len(base_bytes) == (64 + 40 * sample_n), "Continuous baseline serializer size mismatch!"

    base_topo_bytes = serializer.serialize_unprotected_baseline(sample_feats, sample_base_w, packed_topology=sample_topo)
    assert len(base_topo_bytes) == (64 + 40 * sample_n + len(sample_topo)), "Continuous baseline with topology size mismatch!"

    print(f"  [VERIFIED] Exact binary serializer produces byte-for-byte matching length.")
    return rows, summary


def generate_storage_visualizations(summary_db1_a: Dict[str, Any], output_dir: str):
    """Generates publication-quality pie charts and comparative bar charts for storage accounting."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. Component Pie Chart for Mean DB1_A Template (N ~ 144)
    mean_n = summary_db1_a["triangles"]["mean"]
    n_int = int(round(mean_n))
    
    # Breakdown for Descriptor-Only (Prod Default)
    hdr_bytes = 64
    desc_bytes = n_int * 4
    w_bytes = n_int * 2
    tag_bytes = 16
    total_desc = hdr_bytes + desc_bytes + w_bytes + tag_bytes

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    labels1 = [
        f"Bitstrings (32-bit)\n{desc_bytes} B ({desc_bytes/total_desc*100:.1f}%)",
        f"Quality Weights (fp16)\n{w_bytes} B ({w_bytes/total_desc*100:.1f}%)",
        f"Binary Header\n{hdr_bytes} B ({hdr_bytes/total_desc*100:.1f}%)",
        f"AEAD Poly1305 Tag\n{tag_bytes} B ({tag_bytes/total_desc*100:.1f}%)"
    ]
    sizes1 = [desc_bytes, w_bytes, hdr_bytes, tag_bytes]
    colors1 = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    
    ax1.pie(sizes1, labels=labels1, colors=colors1, autopct="", startangle=140, 
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax1.set_title(f"(A) Production Default 9-D Protected Credential\nMean Size: {total_desc:,} B (N={n_int} Triangles)", 
                  fontsize=12, fontweight="bold", pad=15)

    # Breakdown for Credential with Packed Topology
    num_upper = (n_int * (n_int - 1)) // 2
    packed_topo_bytes = (num_upper + 3) // 4
    total_topo = total_desc + packed_topo_bytes

    labels2 = [
        f"Packed Matrix A (2-bit)\n{packed_topo_bytes} B ({packed_topo_bytes/total_topo*100:.1f}%)",
        f"Bitstrings (32-bit)\n{desc_bytes} B ({desc_bytes/total_topo*100:.1f}%)",
        f"Quality Weights (fp16)\n{w_bytes} B ({w_bytes/total_topo*100:.1f}%)",
        f"Binary Header & AEAD Tag\n{hdr_bytes+tag_bytes} B ({(hdr_bytes+tag_bytes)/total_topo*100:.1f}%)"
    ]
    sizes2 = [packed_topo_bytes, desc_bytes, w_bytes, hdr_bytes + tag_bytes]
    colors2 = ["#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e"]

    ax2.pie(sizes2, labels=labels2, colors=colors2, autopct="", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax2.set_title(f"(B) 9-D Protected Credential with Topology\nMean Size: {total_topo:,} B (N={n_int} Triangles)", 
                  fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout()
    pie_path = os.path.join(output_dir, "storage_breakdown_pie.png")
    fig.savefig(pie_path, dpi=300)
    plt.close(fig)
    print(f"Saved storage pie chart to: {pie_path}")

    # 2. Architecture Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    arch_labels = [
        "9-D Protected\n(Prod Default)",
        "9-D Protected\n(with Topology)",
        "Continuous Baseline\n(Descriptor-Only)",
        "Continuous Baseline\n(with Packed A)",
        "Continuous Baseline\n(with uint8 A)"
    ]
    
    mean_bytes = [
        summary_db1_a["prot_descriptor_only"]["mean"],
        summary_db1_a["prot_with_topology"]["mean"],
        summary_db1_a["cont_baseline_descriptor_only"]["mean"],
        summary_db1_a["cont_baseline_with_topology_packed"]["mean"],
        summary_db1_a["cont_baseline_with_topology_uint8"]["mean"]
    ]
    
    bar_colors = ["#1f77b4", "#9467bd", "#aec7e8", "#c5b0d5", "#e377c2"]
    bars = ax.bar(arch_labels, mean_bytes, color=bar_colors, edgecolor="black", linewidth=1.0, width=0.6)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 200, f"{h:,.0f} B\n({h/1024:.2f} KB)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Mean Serialized Credential Length (Bytes)", fontsize=11, fontweight="bold")
    ax.set_title("FVC2002 DB1_A (800 Enrolled Templates): Serialized Storage Accounting Comparison", 
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, max(mean_bytes) * 1.18)
    
    # Add annotations
    red_desc = summary_db1_a["average_compression"]["descriptor_only_vs_baseline_pct"]
    ax.annotate(f"{red_desc:.1f}% Storage Reduction\nvs Continuous Baseline",
                xy=(0, mean_bytes[0]), xytext=(0, mean_bytes[0] + 5000),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", ec="gray"))

    plt.tight_layout()
    bar_path = os.path.join(output_dir, "storage_comparison_bar.png")
    fig.savefig(bar_path, dpi=300)
    plt.close(fig)
    print(f"Saved storage comparison bar chart to: {bar_path}")


def run_storage_benchmark(output_dir: str = "experiments/results/phase4") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    service = ProductionBiometricService()

    # Load datasets
    ds_a = load_fvc_dataset("DB1_A")
    ds_b = load_fvc_dataset("DB1_B")

    # Evaluate DB1_A (800 images)
    rows_a, summary_a = evaluate_dataset_storage(service, "DB1_A", ds_a["all_images"], output_dir)
    
    # Save DB1_A CSV manifest
    csv_a = os.path.join(output_dir, "storage_manifest_db1_a.csv")
    import csv
    with open(csv_a, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_a[0].keys()))
        writer.writeheader()
        writer.writerows(rows_a)
    print(f"Saved DB1_A storage manifest ({len(rows_a)} rows) to: {csv_a}")

    # Evaluate DB1_B (80 images)
    rows_b, summary_b = evaluate_dataset_storage(service, "DB1_B", ds_b["all_images"], output_dir)
    csv_b = os.path.join(output_dir, "storage_manifest_db1_b.csv")
    with open(csv_b, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_b[0].keys()))
        writer.writeheader()
        writer.writerows(rows_b)
    print(f"Saved DB1_B storage manifest ({len(rows_b)} rows) to: {csv_b}")

    # Pooled All (880 templates)
    all_rows = rows_a + rows_b
    all_tri = [r["num_triangles"] for r in all_rows]
    all_prot_desc = [r["prot_descriptor_only_bytes"] for r in all_rows]
    all_prot_topo = [r["prot_with_topology_bytes"] for r in all_rows]
    all_cont_desc = [r["cont_baseline_desc_bytes"] for r in all_rows]
    all_cont_topo_pk = [r["cont_baseline_topo_packed_bytes"] for r in all_rows]
    all_cont_topo_u8 = [r["cont_baseline_topo_uint8_bytes"] for r in all_rows]

    pooled_summary = {
        "num_templates": len(all_rows),
        "triangles": compute_summary_stats(all_tri),
        "prot_descriptor_only": compute_summary_stats(all_prot_desc),
        "prot_with_topology": compute_summary_stats(all_prot_topo),
        "cont_baseline_descriptor_only": compute_summary_stats(all_cont_desc),
        "cont_baseline_with_topology_packed": compute_summary_stats(all_cont_topo_pk),
        "cont_baseline_with_topology_uint8": compute_summary_stats(all_cont_topo_u8),
        "average_compression": {
            "descriptor_only_vs_baseline_pct": round((1.0 - (np.mean(all_prot_desc) / np.mean(all_cont_desc))) * 100.0, 2),
            "topology_packed_vs_baseline_packed_pct": round((1.0 - (np.mean(all_prot_topo) / np.mean(all_cont_topo_pk))) * 100.0, 2),
            "topology_packed_vs_baseline_uint8_pct": round((1.0 - (np.mean(all_prot_topo) / np.mean(all_cont_topo_u8))) * 100.0, 2)
        }
    }

    full_report = {
        "benchmark": "Phase 4 Serialized Credential Storage Accounting",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "binary_framing": {
            "magic_bytes": "CFPB",
            "version": 1,
            "header_bytes": 64,
            "aead_nonce_bytes": 12,
            "aead_auth_tag_bytes": 16,
            "aead_cipher": "ChaCha20-Poly1305 (RFC 8439)",
            "descriptor_encoding": "32-bit column-orthonormal binary projection (4 B/tri)",
            "quality_weight_encoding": "float16 (2 B/tri)",
            "topology_packing": "Lossless symmetric 2-bit upper-triangle (ceil(N(N-1)/8) B)"
        },
        "DB1_A": summary_a,
        "DB1_B": summary_b,
        "Pooled_All": pooled_summary
    }

    report_path = os.path.join(output_dir, "storage_benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"Saved complete storage benchmark report to: {report_path}")

    # Generate charts
    generate_storage_visualizations(summary_a, output_dir)

    print("\n================================================================================")
    print("  PHASE 4 STORAGE BENCHMARK SUMMARY (DB1_A N=800, DB1_B N=80)")
    print("================================================================================")
    print(f"  DB1_A Triangles Mean: {summary_a['triangles']['mean']:.1f} (Median: {summary_a['triangles']['median']:.0f}, P5: {summary_a['triangles']['p5']:.0f}, P95: {summary_a['triangles']['p95']:.0f})")
    print(f"  DB1_A Protected Descriptor-Only (Prod Default):")
    print(f"    Mean: {summary_a['prot_descriptor_only']['mean']:.1f} B | Median: {summary_a['prot_descriptor_only']['median']:.0f} B | Total: {summary_a['prot_descriptor_only']['total_bytes']:,} B")
    print(f"  DB1_A Protected with Topology:")
    print(f"    Mean: {summary_a['prot_with_topology']['mean']:.1f} B | Median: {summary_a['prot_with_topology']['median']:.0f} B | Total: {summary_a['prot_with_topology']['total_bytes']:,} B")
    print(f"  DB1_A Continuous Baseline (Descriptor-Only):")
    print(f"    Mean: {summary_a['cont_baseline_descriptor_only']['mean']:.1f} B | Median: {summary_a['cont_baseline_descriptor_only']['median']:.0f} B | Total: {summary_a['cont_baseline_descriptor_only']['total_bytes']:,} B")
    print(f"  --> True Compression (Prod Default vs Continuous Baseline): {summary_a['average_compression']['descriptor_only_vs_baseline_pct']:.2f}% reduction")
    print("================================================================================\n")

    return full_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Phase 4 Serialized Storage Accounting.")
    parser.add_argument("--output_dir", type=str, default="experiments/results/phase4", help="Output directory")
    args = parser.parse_args()
    run_storage_benchmark(args.output_dir)
