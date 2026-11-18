"""
Set-B Calibration Script for MCC-256 + Keyed RFF SE(2) Spectral Consensus.

Strictly calibrated on FVC2002 Set B datasets (DB1_B, DB2_B, DB3_B, DB4_B).
NEVER touches Set A (DB1_A, DB2_A, DB3_A, DB4_A).

Computes:
1. Symmetric 256-D raw MCC channel (mean, std):
   - Leave-one-DB-out disjoint stats for unbiased Set-B evaluation:
     DB1_B_MCC_256 (from DB2_B+DB3_B+DB4_B),
     DB2_B_MCC_256 (from DB1_B+DB3_B+DB4_B),
     DB3_B_MCC_256 (from DB1_B+DB2_B+DB4_B),
     DB4_B_MCC_256 (from DB1_B+DB2_B+DB3_B).
   - Pooled Set-B stats (DB1_B+DB2_B+DB3_B+DB4_B) for DB*_A / DEFAULT_MCC_256.
2. Protected matching calibration (192-bit MCC + 128-bit RFF edge tokens)
   to freeze Classical (and optionally Deep ONNX) EER / 1% FAR / 0.1% FAR thresholds.
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.descriptors.mcc import MinutiaeCylinderCode, flip_mcc_dual
from src.evaluation.metrics import compute_roc_metrics
from src.services.mcc_builder import MCCBuiltFeatures, MCCDescriptorBuilder, symmetrize_mcc_stats
from src.services.orchestrator import ProductionBiometricService
from src.services.types import ExtractedMinutiae


DB_CONFIG = {
    "DB1_B": 500.0,
    "DB2_B": 569.0,
    "DB3_B": 500.0,
    "DB4_B": 500.0,
}


def extract_setb_minutiae(
    service: ProductionBiometricService,
    subsets: Tuple[str, ...] = ("DB1_B", "DB2_B", "DB3_B", "DB4_B"),
) -> Dict[str, Dict[str, Tuple[ExtractedMinutiae, np.ndarray, np.ndarray]]]:
    """
    Runs capture -> preprocess -> minutiae extraction -> raw 256-D MCC extraction
    once per Set B image.
    Returns {db: {fid: (ExtractedMinutiae, raw_descs, weights)}}.
    """
    results: Dict[str, Dict[str, Tuple[ExtractedMinutiae, np.ndarray, np.ndarray]]] = {}
    builder = service.mcc_builder

    for db in subsets:
        dpi = DB_CONFIG[db]
        paths = sorted(glob.glob(f"dataset/FVC2002/{db}/*.tif"))
        t0 = time.perf_counter()
        db_map: Dict[str, Tuple[ExtractedMinutiae, np.ndarray, np.ndarray]] = {}
        for p in paths:
            fid = os.path.basename(p).split(".")[0]
            capture = service.capture_adapter.ingest(p, dpi=dpi)
            pre = service.preprocessor.process(capture)
            mins = service.minutiae_extractor.extract(pre, raw_image=capture.image)
            built = builder.build(mins, dpi=dpi, mean=None, std=None, enforce_min_count=False)
            db_map[fid] = (mins, built.raw_descriptors, built.weights)
        elapsed = time.perf_counter() - t0
        print(f"Extracted {db}: {len(paths)} images in {elapsed:.1f}s ({1000.0*elapsed/max(1,len(paths)):.1f} ms/img)")
        results[db] = db_map
    return results


def compute_symmetric_stats(
    extracted: Dict[str, Dict[str, Tuple[ExtractedMinutiae, np.ndarray, np.ndarray]]],
    include_dbs: Tuple[str, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    chunks = []
    for db in include_dbs:
        for _, raw_d, _ in extracted[db].values():
            if len(raw_d) > 0:
                chunks.append(raw_d)
    stacked = np.vstack(chunks)
    mu = np.mean(stacked, axis=0)
    sd = np.maximum(np.std(stacked, axis=0), 1e-4)
    return symmetrize_mcc_stats(mu, sd, num_spatial_cells=8, num_directional_slices=4)


def compute_symmetric_zca(
    extracted: Dict[str, Dict[str, Tuple[ExtractedMinutiae, np.ndarray, np.ndarray]]],
    include_dbs: Tuple[str, ...],
    mean: np.ndarray,
    std: np.ndarray,
    zca_reg: float = 1.0,
) -> np.ndarray:
    """Computes symmetric ZCA whitening matrix W_zca that commutes with flip_mcc_dual."""
    chunks = []
    for db in include_dbs:
        for _, raw_d, _ in extracted[db].values():
            if len(raw_d) > 0:
                chunks.append(raw_d)
    stacked = np.vstack(chunks).astype(np.float64)
    Z = (stacked - np.asarray(mean, dtype=np.float64)) / np.asarray(std, dtype=np.float64)
    Zf = flip_mcc_dual(Z.astype(np.float32), num_spatial_cells=8, num_directional_slices=4).astype(np.float64)
    C = 0.5 * ((Z.T @ Z) + (Zf.T @ Zf)) / max(1, len(Z))
    evals, evecs = np.linalg.eigh(C)
    evals = np.maximum(evals, 0.0)
    W_zca = (evecs / np.sqrt(evals + zca_reg)) @ evecs.T
    return W_zca.astype(np.float32)


def prepare_protected_templates(
    service: ProductionBiometricService,
    db_map: Dict[str, Tuple[ExtractedMinutiae, np.ndarray, np.ndarray]],
    dpi: float,
    mean: np.ndarray,
    std: np.ndarray,
    zca_matrix: Optional[np.ndarray] = None,
    app_id: str = "calibrate_setb",
    salt: str = "setb_cal_salt_01",
    subject_key: str = "cal_user",
) -> Dict[str, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Standardizes + L2-normalizes + projects 256->192 bitstrings and encodes 128-bit
    RFF edge tokens using a shared application key for FVC pair scoring.
    (Valid cross-template comparison requires identical projection parameters
    under single-keyed benchmark protocol.)
    """
    W_rff, b_rff = service.projection_engine.get_rff_edge_params(
        salt=salt, app_id=app_id, subject_id=subject_key, edge_dim=7, rff_bits=service.rff_bits
    )
    templates: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}

    for fid, (mins, _, _) in db_map.items():
        if len(mins.minutiae) < 3:
            templates[fid] = None
            continue
        feats = service.mcc_builder.build(
            mins, dpi=dpi, mean=mean, std=std, zca_matrix=zca_matrix, enforce_min_count=False
        )
        bits = service.projection_engine.project_and_quantize(
            descriptors=feats.descriptors,
            salt=salt,
            app_id=app_id,
            subject_id=subject_key,
            in_dim=256,
            out_dim=service.out_dim,
            mean=None,
            std=None,
        )
        bits_dual = service.projection_engine.project_and_quantize(
            descriptors=feats.descriptors_dual,
            salt=salt,
            app_id=app_id,
            subject_id=subject_key,
            in_dim=256,
            out_dim=service.out_dim,
            mean=None,
            std=None,
        )
        edges = service.mcc_builder.encode_protected_edge_tokens_with_params(
            feats, W_rff=W_rff, b_rff=b_rff
        )
        templates[fid] = (bits, bits_dual, feats.weights, edges)
    return templates


def score_db_pairs(
    service: ProductionBiometricService,
    templates: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
    all_impostors: bool = True,
) -> Dict[str, object]:
    gen_scores = []
    imp_fvc_scores = []
    imp_all_scores = []

    # Genuine: 10 fingers (101..110), 8 impressions -> 280 pairs
    for f in range(101, 111):
        for i in range(1, 9):
            ki = f"{f}_{i}"
            ti = templates.get(ki)
            for j in range(i + 1, 9):
                kj = f"{f}_{j}"
                tj = templates.get(kj)
                if ti is None or tj is None:
                    gen_scores.append(0.0)
                else:
                    sc = service.matcher.match_protected_spectral(
                        q_bitstrings=ti[0],
                        e_bitstrings=tj[0],
                        q_weights=ti[2],
                        e_weights=tj[2],
                        q_edge_tokens=ti[3],
                        e_edge_tokens=tj[3],
                        q_bitstrings_dual=ti[1],
                    )
                    gen_scores.append(sc)

    # FVC standard impostors: impression 1 vs impression 1 (45 pairs)
    for fi in range(101, 111):
        ti = templates.get(f"{fi}_1")
        for fj in range(fi + 1, 111):
            tj = templates.get(f"{fj}_1")
            if ti is None or tj is None:
                imp_fvc_scores.append(0.0)
            else:
                sc = service.matcher.match_protected_spectral(
                    q_bitstrings=ti[0],
                    e_bitstrings=tj[0],
                    q_weights=ti[2],
                    e_weights=tj[2],
                    q_edge_tokens=ti[3],
                    e_edge_tokens=tj[3],
                    q_bitstrings_dual=ti[1],
                )
                imp_fvc_scores.append(sc)

    # All-pairs impostor for smooth FAR 1% / 0.1% estimation (2880 pairs)
    if all_impostors:
        for fi in range(101, 111):
            for ii in range(1, 9):
                ti = templates.get(f"{fi}_{ii}")
                for fj in range(fi + 1, 111):
                    for ij in range(1, 9):
                        tj = templates.get(f"{fj}_{ij}")
                        if ti is None or tj is None:
                            imp_all_scores.append(0.0)
                        else:
                            sc = service.matcher.match_protected_spectral(
                                q_bitstrings=ti[0],
                                e_bitstrings=tj[0],
                                q_weights=ti[2],
                                e_weights=tj[2],
                                q_edge_tokens=ti[3],
                                e_edge_tokens=tj[3],
                                q_bitstrings_dual=ti[1],
                            )
                            imp_all_scores.append(sc)

    gen = np.array(gen_scores, dtype=np.float64)
    imp_fvc = np.array(imp_fvc_scores, dtype=np.float64)
    imp_eval = np.array(imp_all_scores, dtype=np.float64) if all_impostors else imp_fvc
    roc_fvc = compute_roc_metrics(gen, imp_fvc)
    roc_smooth = compute_roc_metrics(gen, imp_eval)

    def _thresh_at_far(roc, target):
        valid = np.where(roc["far"] <= target)[0]
        return float(roc["thresholds"][valid[0]]) if len(valid) > 0 else float(np.max(imp_eval) + 1e-4)

    return {
        "gen": gen,
        "imp_fvc": imp_fvc,
        "imp_all": imp_eval,
        "eer_fvc": float(roc_fvc["eer"]),
        "auc_fvc": float(roc_fvc["auc"]),
        "eer_smooth": float(roc_smooth["eer"]),
        "auc_smooth": float(roc_smooth["auc"]),
        "eer_threshold": float(roc_smooth["eer_threshold"]),
        "far_1pct_threshold": _thresh_at_far(roc_smooth, 0.01),
        "far_01pct_threshold": _thresh_at_far(roc_smooth, 0.001),
        "tar_1pct": float(roc_smooth["tar_at_1pct_far"]),
        "tar_01pct": float(roc_smooth["tar_at_01pct_far"]),
        "gen_mean": float(np.mean(gen)),
        "imp_mean": float(np.mean(imp_eval)),
    }


def write_calibrated_stats_module(
    out_path: str,
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    thresholds: Dict[str, Dict[str, float]],
):
    lines = [
        '"""',
        "Frozen Set-B Calibrated Symmetric 256-D MCC Scalers & Protected Spectral Thresholds.",
        "",
        "Generated strictly on FVC2002 Set B (DB1_B, DB2_B, DB3_B, DB4_B) by",
        "scripts/calibrate_mcc_setb.py. Set A is strictly excluded.",
        '"""',
        "",
        "import numpy as np",
        "",
    ]

    def _fmt_arr(name: str, arr: np.ndarray):
        vals = ", ".join(f"{float(v):.8f}" for v in arr)
        lines.append(f"{name} = np.array([{vals}], dtype=np.float32)")

    for key, (mu, sd) in stats.items():
        _fmt_arr(f"{key}_MEAN", mu)
        _fmt_arr(f"{key}_STD", sd)
        lines.append("")

    lines.append("CALIBRATED_MCC_THRESHOLDS = {")
    for k, v in thresholds.items():
        lines.append(f'    "{k}": {{')
        for tk, tv in v.items():
            lines.append(f'        "{tk}": {float(tv):.6f},')
        lines.append("    },")
    lines.append("}")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote calibrated stats module to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["classical", "deep", "both"], default="classical")
    parser.add_argument("--zca-reg", type=float, default=1.0)
    parser.add_argument("--write-module", action="store_true")
    parser.add_argument("--json-out", default="experiments/results/setb_mcc_calibration.json")
    args = parser.parse_args()

    all_dbs = ("DB1_B", "DB2_B", "DB3_B", "DB4_B")
    engines = ["classical", "deep"] if args.engine == "both" else [args.engine]

    summary = {"engines": {}}
    existing_thresholds: Dict[str, Dict[str, float]] = {}

    if os.path.exists(args.json_out):
        try:
            with open(args.json_out, "r", encoding="utf-8") as f:
                prev = json.load(f)
                summary.update(prev)
        except Exception:
            pass

    stats_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    zca_arrays: Dict[str, np.ndarray] = {}

    for eng in engines:
        print(f"\n=== Calibrating MCC-256 + ZCA(reg={args.zca_reg}) + RFF128 Spectral ({eng.upper()}) on Set B ===")
        service = ProductionBiometricService(feature_mode="mcc_spectral", engine=eng)
        extracted = extract_setb_minutiae(service, all_dbs)

        # Compute symmetric 256-D stats and ZCA whitening from classical engine (shared across engines)
        if eng == "classical" or not stats_arrays:
            mu_p, sd_p = compute_symmetric_stats(extracted, all_dbs)
            stats_arrays["POOLED_SET_B_MCC_256"] = (mu_p, sd_p)
            zca_arrays["POOLED_SET_B_MCC_256"] = compute_symmetric_zca(
                extracted, all_dbs, mu_p, sd_p, zca_reg=args.zca_reg
            )
            for db in all_dbs:
                others = tuple(d for d in all_dbs if d != db)
                mu_lo, sd_lo = compute_symmetric_stats(extracted, others)
                stats_arrays[f"{db}_MCC_256"] = (mu_lo, sd_lo)
                zca_arrays[f"{db}_MCC_256"] = compute_symmetric_zca(
                    extracted, others, mu_lo, sd_lo, zca_reg=args.zca_reg
                )

        # Evaluate Set B partitions using STRICTLY leave-one-DB-out stats & ZCA
        per_db_metrics = {}
        pooled_gen, pooled_imp = [], []
        for eval_db in all_dbs:
            mu_lo, sd_lo = stats_arrays[f"{eval_db}_MCC_256"]
            zca_lo = zca_arrays[f"{eval_db}_MCC_256"]
            templates = prepare_protected_templates(
                service, extracted[eval_db], dpi=DB_CONFIG[eval_db], mean=mu_lo, std=sd_lo, zca_matrix=zca_lo
            )
            res = score_db_pairs(service, templates, all_impostors=True)
            pooled_gen.append(res["gen"])
            pooled_imp.append(res["imp_all"])
            per_db_metrics[eval_db] = {
                "eer_fvc": res["eer_fvc"],
                "auc_fvc": res["auc_fvc"],
                "eer_smooth": res["eer_smooth"],
                "auc_smooth": res["auc_smooth"],
                "eer_threshold": res["eer_threshold"],
                "far_1pct_threshold": res["far_1pct_threshold"],
                "far_01pct_threshold": res["far_01pct_threshold"],
                "tar_1pct": res["tar_1pct"],
                "tar_01pct": res["tar_01pct"],
                "gen_mean": res["gen_mean"],
                "imp_mean": res["imp_mean"],
            }
            print(
                f"  [{eng.upper()} {eval_db}] FVC(45imp) EER={100*res['eer_fvc']:.2f}% AUC={res['auc_fvc']:.4f} | "
                f"AllImp(2880) EER={100*res['eer_smooth']:.2f}% AUC={res['auc_smooth']:.4f} "
                f"T_eer={res['eer_threshold']:.4f} T_1%={res['far_1pct_threshold']:.4f} T_0.1%={res['far_01pct_threshold']:.4f}"
            )

        p_gen = np.concatenate(pooled_gen)
        p_imp = np.concatenate(pooled_imp)
        roc_p = compute_roc_metrics(p_gen, p_imp)
        valid_1 = np.where(roc_p["far"] <= 0.01)[0]
        valid_01 = np.where(roc_p["far"] <= 0.001)[0]
        t_eer = float(roc_p["eer_threshold"])
        t_1pct = float(roc_p["thresholds"][valid_1[0]]) if len(valid_1) > 0 else float(np.max(p_imp) + 1e-4)
        t_01pct = float(roc_p["thresholds"][valid_01[0]]) if len(valid_01) > 0 else float(np.max(p_imp) + 2e-4)

        print(
            f"  [{eng.upper()} POOLED DB1_B+DB2_B] EER={100*roc_p['eer']:.2f}% AUC={roc_p['auc']:.4f} "
            f"T_eer={t_eer:.4f} T_1%={t_1pct:.4f} T_0.1%={t_01pct:.4f}"
        )

        summary["engines"][eng] = {
            "per_db": per_db_metrics,
            "pooled_db1b_db2b": {
                "eer": float(roc_p["eer"]),
                "auc": float(roc_p["auc"]),
                "eer_threshold": t_eer,
                "far_1pct_threshold": t_1pct,
                "far_01pct_threshold": t_01pct,
                "tar_1pct": float(roc_p["tar_at_1pct_far"]),
                "tar_01pct": float(roc_p["tar_at_01pct_far"]),
            },
        }

        eng_upper = eng.upper()
        existing_thresholds[f"DEFAULT_MCC_{eng_upper}"] = {
            "EER": t_eer,
            "FAR_1_PERCENT": t_1pct,
            "FAR_0_1_PERCENT": t_01pct,
            "BENCHMARK_ACCEPT": t_eer,
        }
        for eval_db, met in per_db_metrics.items():
            existing_thresholds[f"{eval_db}_MCC_{eng_upper}"] = {
                "EER": met["eer_threshold"],
                "FAR_1_PERCENT": met["far_1pct_threshold"],
                "FAR_0_1_PERCENT": met["far_01pct_threshold"],
                "BENCHMARK_ACCEPT": met["eer_threshold"],
            }
        for db_num in (1, 2, 3, 4):
            b_key = f"DB{db_num}_B"
            if b_key in per_db_metrics:
                met = per_db_metrics[b_key]
                existing_thresholds[f"DB{db_num}_A_MCC_{eng_upper}"] = {
                    "EER": met["eer_threshold"],
                    "FAR_1_PERCENT": met["far_1pct_threshold"],
                    "FAR_0_1_PERCENT": met["far_01pct_threshold"],
                    "BENCHMARK_ACCEPT": met["eer_threshold"],
                }
            else:
                existing_thresholds[f"DB{db_num}_A_MCC_{eng_upper}"] = dict(existing_thresholds[f"DEFAULT_MCC_{eng_upper}"])
        if eng == "classical":
            existing_thresholds["DEFAULT_MCC"] = dict(existing_thresholds["DEFAULT_MCC_CLASSICAL"])

    # Preserve previously calibrated engine thresholds if running single-engine
    for prev_eng, prev_data in summary.get("engines", {}).items():
        if prev_eng not in engines:
            pu = prev_eng.upper()
            pb = prev_data["pooled_db1b_db2b"]
            existing_thresholds[f"DEFAULT_MCC_{pu}"] = {
                "EER": pb["eer_threshold"],
                "FAR_1_PERCENT": pb["far_1pct_threshold"],
                "FAR_0_1_PERCENT": pb["far_01pct_threshold"],
                "BENCHMARK_ACCEPT": pb["eer_threshold"],
            }
            for eval_db, met in prev_data["per_db"].items():
                existing_thresholds[f"{eval_db}_MCC_{pu}"] = {
                    "EER": met["eer_threshold"],
                    "FAR_1_PERCENT": met["far_1pct_threshold"],
                    "FAR_0_1_PERCENT": met["far_01pct_threshold"],
                    "BENCHMARK_ACCEPT": met["eer_threshold"],
                }
            for db_num in (1, 2, 3, 4):
                b_key = f"DB{db_num}_B"
                if b_key in prev_data["per_db"]:
                    met = prev_data["per_db"][b_key]
                    existing_thresholds[f"DB{db_num}_A_MCC_{pu}"] = {
                        "EER": met["eer_threshold"],
                        "FAR_1_PERCENT": met["far_1pct_threshold"],
                        "FAR_0_1_PERCENT": met["far_01pct_threshold"],
                        "BENCHMARK_ACCEPT": met["eer_threshold"],
                    }
                else:
                    existing_thresholds[f"DB{db_num}_A_MCC_{pu}"] = dict(existing_thresholds[f"DEFAULT_MCC_{pu}"])

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.write_module:
        # Load existing stats if deep-only run didn't recompute classical stats
        if "POOLED_SET_B_MCC_256" not in stats_arrays:
            raise RuntimeError("Missing MCC_256 stats arrays; run with --engine classical or both.")
        zca_out_path = "src/services/mcc_zca_setb.npz"
        np.savez_compressed(zca_out_path, **zca_arrays)
        print(f"Wrote Set-B ZCA whitening matrices to {zca_out_path}")
        write_calibrated_stats_module(
            "src/services/mcc_calibrated_stats.py", stats_arrays, existing_thresholds
        )


if __name__ == "__main__":
    main()
