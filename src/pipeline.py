import os
from typing import Dict, Any, Optional, List, Union
import numpy as np
import cv2

from src.preprocessing.segmenter import segment_fingerprint
from src.preprocessing.enhancer import enhance_fingerprint
from src.minutiae.extractor import extract_minutiae, Minutia
from src.minutiae.onnx_detector import ONNXMinutiaeDetector, DEFAULT_ONNX_MODEL_PATH
from src.graph.delaunay import build_delaunay_triangles
from src.graph.descriptors import extract_all_descriptors
from src.descriptors.mcc import MinutiaeCylinderCode
from src.matching.spectral_consensus import (
    SpectralGraphMatcher,
    derive_rff_edge_params,
    encode_edge_rff,
)
from src.services.mcc_builder import symmetrize_mcc_stats
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors
from src.matching.bipartite_matcher import (
    match_protected_templates,
    match_unprotected_descriptors,
    match_protected_greedy,
    compute_pairwise_hamming_distance,
)
from src.protection.kdf import validate_master_secret


def init_gpu_acceleration(verbose: bool = False) -> Dict[str, Any]:
    """
    Initializes and verifies GPU hardware acceleration via OpenCV Transparent API (OpenCL 3.0 CUDA).
    Leverages NVIDIA GeForce RTX 3050 Laptop GPU when available.
    """
    has_ocl = cv2.ocl.haveOpenCL()
    if has_ocl:
        cv2.ocl.setUseOpenCL(True)
        dev = cv2.ocl.Device.getDefault()
        info = {
            "available": True,
            "active": bool(cv2.ocl.useOpenCL()),
            "device_name": dev.name(),
            "opencl_version": dev.OpenCLVersion(),
            "vendor": dev.vendorName()
        }
    else:
        info = {
            "available": False,
            "active": False,
            "device_name": "CPU",
            "opencl_version": "N/A",
            "vendor": "N/A"
        }
    if verbose and info["active"]:
        print(f"[GPU ACCEL] Active device: {info['device_name']} ({info['opencl_version']})")
    return info


class CancelableFingerprintPipeline:
    """
    End-to-end Cancelable Fingerprint Biometrics Pipeline implementing
    Quality-Aware Keyed Minutiae-Graph & MCC Spectral Protection.
    """
    def __init__(self, master_secret: Union[bytes, str] = "CancelableBioMasterSecret_2026_256Bits_KMS!",
                 in_dim: int = 12, out_dim: int = 32, 
                 dpi: float = 500.0,
                 hamming_threshold: float = 0.22,
                 operating_threshold: float = 0.20,
                 min_edge_length: float = 0.0,
                 min_area: float = 0.0,
                 max_aspect_ratio: Optional[float] = None,
                 coverage_penalty_exponent: float = 1.0,
                 min_matches: int = 0,
                 min_coverage_ratio: float = 0.0,
                 matcher: str = "hungarian",
                 use_gpu: bool = True,
                 feature_mode: str = "delaunay",
                 engine: str = "classical",
                 onnx_model_path: Optional[str] = None,
                 rff_bits: int = 128,
                 scaler_mean: Optional[np.ndarray] = None,
                 scaler_std: Optional[np.ndarray] = None,
                 scaler_zca: Optional[np.ndarray] = None):
        # Validate that the master secret is at least 32 bytes (256 bits)
        self.master_secret = validate_master_secret(master_secret)
        if feature_mode not in ("delaunay", "mcc_spectral"):
            raise ValueError(f"Unknown feature_mode '{feature_mode}'. Must be 'delaunay' or 'mcc_spectral'.")
        if engine not in ("classical", "deep"):
            raise ValueError(f"Unknown engine '{engine}'. Must be 'classical' or 'deep'.")

        self.feature_mode = feature_mode
        self.engine = engine
        self.in_dim = 256 if (feature_mode == "mcc_spectral" and in_dim == 12) else in_dim
        self.out_dim = 192 if (feature_mode == "mcc_spectral" and out_dim == 32) else out_dim
        self.dpi = dpi
        self.hamming_threshold = 0.44 if (feature_mode == "mcc_spectral" and hamming_threshold == 0.22) else hamming_threshold
        self.operating_threshold = operating_threshold
        self.min_edge_length = min_edge_length
        self.min_area = min_area
        self.max_aspect_ratio = max_aspect_ratio
        self.coverage_penalty_exponent = coverage_penalty_exponent
        self.min_matches = min_matches
        self.min_coverage_ratio = min_coverage_ratio
        if matcher not in ("hungarian", "greedy_1to1", "spectral_consensus"):
            raise ValueError(f"Unknown matcher '{matcher}'. Must be 'hungarian', 'greedy_1to1', or 'spectral_consensus'")
        self.matcher = "spectral_consensus" if feature_mode == "mcc_spectral" else matcher
        self.use_gpu = use_gpu
        self.rff_bits = rff_bits
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.scaler_zca = scaler_zca
        self.gpu_info = init_gpu_acceleration(verbose=False) if use_gpu else {"available": False, "active": False}

        self.onnx_detector: Optional[ONNXMinutiaeDetector] = None
        if engine == "deep":
            resolved = onnx_model_path if onnx_model_path is not None else DEFAULT_ONNX_MODEL_PATH
            self.onnx_detector = ONNXMinutiaeDetector(model_path=resolved, conf_thresh=0.35)

        self.mcc_engine = MinutiaeCylinderCode(
            radius=70.0, num_spatial_cells=8, num_directional_slices=4,
            sigma_spatial=8.5, sigma_directional=np.pi / 5.0, sigmoid_gain=2.0, dpi=dpi,
        )
        self.spectral_matcher = SpectralGraphMatcher(
            sigma_dist=18.0, sigma_angle=0.38, power_iterations=18,
            sigma_edge_angle=0.34, top_k=6, use_lss=True,
            min_edge_compat=0.15, rff_gamma=1.00, max_candidates=400,
            min_eigenvector_weight=0.01, clique_prune_thresh=0.16,
        )

    def _standardize_mcc(self, descs: np.ndarray) -> np.ndarray:
        if len(descs) == 0:
            return descs
        if self.scaler_mean is not None and self.scaler_std is not None:
            mu_s, sd_s = symmetrize_mcc_stats(self.scaler_mean, self.scaler_std)
            ds = (descs - mu_s) / sd_s
        else:
            ds = descs.astype(np.float32, copy=True)
        if self.scaler_zca is not None:
            ds = (ds @ np.asarray(self.scaler_zca, dtype=np.float32)).astype(np.float32)
        norms = np.linalg.norm(ds, axis=1, keepdims=True)
        return (ds / np.maximum(norms, 1e-6)).astype(np.float32)

    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Runs preprocessing, minutiae detection, and Delaunay or MCC+RFF descriptors."""
        if self.feature_mode == "mcc_spectral":
            segmented, mask, bbox = segment_fingerprint(image, method="adaptive")
            enhanced, orientations, coherence = enhance_fingerprint(image, use_gabor=True, mask=mask)
            if self.engine == "deep" and self.onnx_detector is not None and self.onnx_detector.is_model_loaded:
                minutiae, skeleton = self.onnx_detector.detect(image, mask, orientations, coherence)
            else:
                minutiae, skeleton = extract_minutiae(
                    enhanced, mask, orientations, coherence,
                    binarize_method="adaptive", prune_artifacts=True, quality_thresh=0.35,
                    border_margin=10,
                )
            if len(minutiae) >= 3:
                raw_descs, weights = self.mcc_engine.extract_descriptors(minutiae)
                descriptors = self._standardize_mcc(raw_descs)
                descriptors_dual = self.mcc_engine.get_dual_descriptors(descriptors)
                coords = np.array([[m.x, m.y] for m in minutiae], dtype=np.float32)
                thetas = np.array([m.theta for m in minutiae], dtype=np.float32)
            else:
                descriptors = np.empty((0, 256), dtype=np.float32)
                descriptors_dual = np.empty((0, 256), dtype=np.float32)
                weights = np.empty((0,), dtype=np.float32)
                coords = np.empty((0, 2), dtype=np.float32)
                thetas = np.empty((0,), dtype=np.float32)
            return {
                "segmented": segmented,
                "mask": mask,
                "enhanced": enhanced,
                "skeleton": skeleton,
                "minutiae": minutiae,
                "triangles": [],
                "descriptors": descriptors,
                "descriptors_dual": descriptors_dual,
                "weights": weights,
                "coords": coords,
                "thetas": thetas,
            }

        segmented, mask, bbox = segment_fingerprint(image)
        enhanced, orientations, coherence = enhance_fingerprint(segmented)
        if self.engine == "deep" and self.onnx_detector is not None and self.onnx_detector.is_model_loaded:
            minutiae, skeleton = self.onnx_detector.detect(image, mask, orientations, coherence)
        else:
            minutiae, skeleton = extract_minutiae(enhanced, mask, orientations, coherence)
        triangles = build_delaunay_triangles(minutiae, 
                                             min_edge_length=self.min_edge_length,
                                             min_area=self.min_area,
                                             max_aspect_ratio=self.max_aspect_ratio)
        include_types = (self.in_dim == 12)
        descriptors, weights = extract_all_descriptors(
            triangles, dpi=self.dpi, include_types=include_types, desc_dim=self.in_dim
        )
        
        return {
            "segmented": segmented,
            "mask": mask,
            "enhanced": enhanced,
            "skeleton": skeleton,
            "minutiae": minutiae,
            "triangles": triangles,
            "descriptors": descriptors,
            "weights": weights
        }

    def process_minutiae_list(self, minutiae: List[Minutia]) -> Dict[str, Any]:
        """Extracts Delaunay triangles (or MCC descriptors) directly from a minutiae list."""
        if self.feature_mode == "mcc_spectral":
            if len(minutiae) < 3:
                return {
                    "minutiae": minutiae, "triangles": [],
                    "descriptors": np.empty((0, 256), dtype=np.float32),
                    "descriptors_dual": np.empty((0, 256), dtype=np.float32),
                    "weights": np.empty((0,), dtype=np.float32),
                    "coords": np.empty((0, 2), dtype=np.float32),
                    "thetas": np.empty((0,), dtype=np.float32),
                }
            raw_descs, weights = self.mcc_engine.extract_descriptors(minutiae)
            descriptors = self._standardize_mcc(raw_descs)
            descriptors_dual = self.mcc_engine.get_dual_descriptors(descriptors)
            coords = np.array([[m.x, m.y] for m in minutiae], dtype=np.float32)
            thetas = np.array([m.theta for m in minutiae], dtype=np.float32)
            return {
                "minutiae": minutiae,
                "triangles": [],
                "descriptors": descriptors,
                "descriptors_dual": descriptors_dual,
                "weights": weights,
                "coords": coords,
                "thetas": thetas,
            }

        triangles = build_delaunay_triangles(minutiae,
                                             min_edge_length=self.min_edge_length,
                                             min_area=self.min_area,
                                             max_aspect_ratio=self.max_aspect_ratio)
        include_types = (self.in_dim == 12)
        descriptors, weights = extract_all_descriptors(
            triangles, dpi=self.dpi, include_types=include_types, desc_dim=self.in_dim
        )
        return {
            "minutiae": minutiae,
            "triangles": triangles,
            "descriptors": descriptors,
            "weights": weights
        }

    def enroll_from_features(self, descriptors: np.ndarray, weights: np.ndarray, 
                             salt: str, app_id: str, subject_id: str = "",
                             coords: Optional[np.ndarray] = None,
                             thetas: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Enrolls pre-extracted descriptors and produces the protected template."""
        if len(descriptors) == 0:
            return {"status": "error", "message": "Insufficient minutiae detected", "template": None}
            
        W = derive_orthonormal_matrix(self.master_secret, salt, app_id, subject_id=subject_id,
                                      in_dim=self.in_dim, out_dim=self.out_dim)
        projected = project_descriptors(descriptors, W)
        bitstrings = quantize_descriptors(projected)

        edge_tokens = None
        if self.feature_mode == "mcc_spectral" and coords is not None and thetas is not None:
            W_rff, b_rff = derive_rff_edge_params(
                self.master_secret, salt, app_id, subject_id=subject_id,
                edge_dim=7, rff_bits=self.rff_bits,
            )
            edge_tokens = encode_edge_rff(coords, thetas, W_rff, b_rff, dpi=self.dpi)
        
        template = {
            "bitstrings": bitstrings,
            "weights": weights,
            "salt": salt,
            "app_id": app_id,
            "subject_id": subject_id,
            "version": "1.0",
            "num_descriptors": len(descriptors),
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
        }
        if edge_tokens is not None:
            template["edge_tokens"] = edge_tokens
        return {"status": "success", "template": template, "num_triangles": len(descriptors)}

    def enroll(self, image: np.ndarray, salt: str, app_id: str, subject_id: str = "") -> Dict[str, Any]:
        """Enrolls a fingerprint image and produces the protected template."""
        features = self.extract_features(image)
        return self.enroll_from_features(
            features["descriptors"],
            features["weights"],
            salt,
            app_id,
            subject_id=subject_id,
            coords=features.get("coords"),
            thetas=features.get("thetas"),
        )

    def verify_from_features(self, q_descriptors: np.ndarray, q_weights: np.ndarray, 
                             enrolled_template: Dict[str, Any],
                             q_descriptors_dual: Optional[np.ndarray] = None,
                             q_coords: Optional[np.ndarray] = None,
                             q_thetas: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Verifies query descriptors against an enrolled template."""
        salt = enrolled_template["salt"]
        app_id = enrolled_template["app_id"]
        subject_id = enrolled_template.get("subject_id", "")
        enrolled_bits = enrolled_template["bitstrings"]
        enrolled_weights = enrolled_template["weights"]
        
        if len(q_descriptors) == 0:
            return {"decision": "REJECT", "score": 0.0, "reason": "No features in query"}
            
        W = derive_orthonormal_matrix(self.master_secret, salt, app_id, subject_id=subject_id,
                                      in_dim=self.in_dim, out_dim=self.out_dim)
        q_projected = project_descriptors(q_descriptors, W)
        q_bits = quantize_descriptors(q_projected)

        e_edges = enrolled_template.get("edge_tokens")
        if self.feature_mode == "mcc_spectral" and e_edges is not None and q_coords is not None and q_thetas is not None:
            rff_bits = int(e_edges.shape[-1]) * 64
            W_rff, b_rff = derive_rff_edge_params(
                self.master_secret, salt, app_id, subject_id=subject_id,
                edge_dim=7, rff_bits=rff_bits,
            )
            q_edges = encode_edge_rff(q_coords, q_thetas, W_rff, b_rff, dpi=self.dpi)
            dmat = compute_pairwise_hamming_distance(q_bits, enrolled_bits)
            if q_descriptors_dual is not None and len(q_descriptors_dual) == len(q_descriptors):
                q_bits_dual = quantize_descriptors(project_descriptors(q_descriptors_dual, W))
                dmat = np.minimum(dmat, compute_pairwise_hamming_distance(q_bits_dual, enrolled_bits))
            score, details = self.spectral_matcher.match_protected(
                query_edge_tokens=q_edges,
                enrolled_edge_tokens=e_edges,
                pairwise_dists=dmat,
                tau=self.hamming_threshold,
                query_weights=q_weights,
                enrolled_weights=enrolled_weights,
            )
            cov = float(details["num_matched"] / max(1, min(len(q_bits), len(enrolled_bits))))
            details["coverage"] = cov
        elif self.matcher == "greedy_1to1":
            score, details = match_protected_greedy(
                q_bits, enrolled_bits, q_weights, enrolled_weights,
                hamming_threshold=self.hamming_threshold,
                mode="greedy_1to1",
                coverage_penalty_exponent=self.coverage_penalty_exponent,
                min_matches=self.min_matches,
                min_coverage_ratio=self.min_coverage_ratio,
                return_details=True
            )
        else:
            score, details = match_protected_templates(
                q_bits, enrolled_bits, q_weights, enrolled_weights,
                hamming_threshold=self.hamming_threshold,
                coverage_penalty_exponent=self.coverage_penalty_exponent,
                min_matches=self.min_matches,
                min_coverage_ratio=self.min_coverage_ratio,
                return_details=True
            )
        
        decision = "ACCEPT" if score >= self.operating_threshold else "REJECT"
        return {
            "decision": decision,
            "score": float(score),
            "threshold": self.operating_threshold,
            "matched_triangles": details["num_matched"],
            "coverage": details["coverage"],
            "num_query_triangles": len(q_descriptors),
            "num_enrolled_triangles": len(enrolled_bits)
        }

    def verify(self, query_image: np.ndarray, enrolled_template: Dict[str, Any]) -> Dict[str, Any]:
        """Verifies a query fingerprint image against an enrolled protected template."""
        features = self.extract_features(query_image)
        return self.verify_from_features(
            features["descriptors"],
            features["weights"],
            enrolled_template,
            q_descriptors_dual=features.get("descriptors_dual"),
            q_coords=features.get("coords"),
            q_thetas=features.get("thetas"),
        )

    def reissue_from_fresh_capture(self, fresh_descriptors: np.ndarray, fresh_weights: np.ndarray,
                                   new_salt: str, app_id: str, subject_id: str = "",
                                   coords: Optional[np.ndarray] = None,
                                   thetas: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Revocation & Reissuance Workflow (Option A - Fresh Presentation).
        """
        res = self.enroll_from_features(
            fresh_descriptors, fresh_weights, salt=new_salt, app_id=app_id,
            subject_id=subject_id, coords=coords, thetas=thetas
        )
        if res["status"] == "success":
            res["template"]["revoked_previous"] = True
            res["template"]["version"] = f"reissued_{new_salt[:8]}"
        return res
