"""
Production Biometric Orchestrator Service.

Assembles all 13 modular services into a high-assurance, end-to-end biometric
enrollment and verification lifecycle pipeline.
"""

import time
import uuid
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from src.services.types import (
    BiometricErrorCode,
    BiometricServiceException,
    MatchResult,
    ProtectedCredentialDTO,
    SerializedCredentialEnvelope,
    RawCapture
)
from src.services.capture_adapter import CaptureAdapter
from src.services.preprocessor import Preprocessor
from src.services.minutiae_extractor import MinutiaeExtractor
from src.services.triangle_builder import TriangleBuilder
from src.services.descriptor_builder import DescriptorBuilder
from src.services.mcc_builder import MCCDescriptorBuilder
from src.services.scaler_registry import ScalerRegistry
from src.services.key_service import KeyService
from src.services.projection_engine import ProjectionEngine
from src.services.credential_serializer import CredentialSerializer, normalize_salt
from src.services.credential_repository import CredentialRepository
from src.services.matcher import Matcher
from src.services.threshold_registry import ThresholdRegistry
from src.services.audit_logger import AuditLogger
from src.matching.graph_consensus import pack_adjacency_matrix, unpack_adjacency_matrix


class ProductionBiometricService:
    """
    End-to-End Modular Production Biometric Service Pipeline.
    Coordinates CaptureAdapter, Preprocessor, MinutiaeExtractor, TriangleBuilder/MCCBuilder,
    ScalerRegistry, KeyService, ProjectionEngine, CredentialSerializer,
    CredentialRepository, Matcher, ThresholdRegistry, and AuditLogger.
    """
    def __init__(self, 
                 master_secret: Union[bytes, str] = "CancelableBioMasterSecret_2026_256Bits_KMS!",
                 kms_key_id: str = "kms://vault-01/cancelable-bio-master-key",
                 default_dataset: str = "DB1_A",
                 default_algorithm: str = "hungarian",
                 audit_log_file: Optional[str] = None,
                 feature_mode: str = "delaunay",
                 engine: str = "classical",
                 onnx_model_path: Optional[str] = None,
                 out_dim: Optional[int] = None,
                 rff_bits: int = 128,
                 compact_edges: bool = False,
                 edge_top_k: int = 20):
        if feature_mode not in ("delaunay", "mcc_spectral"):
            raise ValueError(f"Unknown feature_mode '{feature_mode}'. Must be 'delaunay' or 'mcc_spectral'.")
        if engine not in ("classical", "deep"):
            raise ValueError(f"Unknown engine '{engine}'. Must be 'classical' or 'deep'.")

        self.feature_mode = feature_mode
        self.engine = engine
        self.out_dim = out_dim if out_dim is not None else (192 if feature_mode == "mcc_spectral" else 32)
        self.rff_bits = rff_bits
        self.compact_edges = compact_edges
        self.edge_top_k = edge_top_k

        # Initialize modular services
        self.capture_adapter = CaptureAdapter()
        if feature_mode == "mcc_spectral":
            self.preprocessor = Preprocessor(segmentation_method="adaptive", mask_enhancer=True)
            self.minutiae_extractor = MinutiaeExtractor(
                min_minutiae=3,
                engine=engine,
                binarize_method="adaptive",
                prune_artifacts=True,
                quality_thresh=0.35,
                model_path=onnx_model_path,
                conf_thresh=0.35,
                border_margin=10,
            )
        else:
            self.preprocessor = Preprocessor()
            self.minutiae_extractor = MinutiaeExtractor(min_minutiae=3)

        self.triangle_builder = TriangleBuilder(method="delaunay")
        self.descriptor_builder = DescriptorBuilder(in_dim=9)
        self.mcc_builder = MCCDescriptorBuilder()
        self.scaler_registry = ScalerRegistry()
        self.key_service = KeyService(master_secret=master_secret, kms_key_id=kms_key_id)
        self.projection_engine = ProjectionEngine(key_service=self.key_service, cache_matrices=True)
        self.serializer = CredentialSerializer()
        self.repository = CredentialRepository(serializer=self.serializer)
        self.matcher = Matcher(default_algorithm=default_algorithm)
        self.threshold_registry = ThresholdRegistry()
        self.audit_logger = AuditLogger(log_file=audit_log_file)
        self.default_dataset = default_dataset

    def _get_mcc_scaler(self, dataset: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        for key in (f"{dataset.upper()}_MCC_256", "POOLED_SET_B_MCC_256", "DEFAULT_MCC_256"):
            if key in self.scaler_registry._scalers:
                p = self.scaler_registry._scalers[key]
                return p.get("mean"), p.get("std")
        return None, None

    def _get_mcc_zca(self, dataset: str) -> Optional[np.ndarray]:
        for key in (f"{dataset.upper()}_MCC_256", "POOLED_SET_B_MCC_256", "DEFAULT_MCC_256"):
            if key in self.scaler_registry._scalers:
                return self.scaler_registry._scalers[key].get("zca_matrix")
        return None

    def _resolve_threshold_dataset(self, target_dataset: str, use_mcc: bool) -> str:
        if not use_mcc:
            return target_dataset
        for key in (
            f"{target_dataset.upper()}_MCC_{self.engine.upper()}",
            f"{target_dataset.upper()}_MCC",
            f"DEFAULT_MCC_{self.engine.upper()}",
            "DEFAULT_MCC",
        ):
            if key in self.threshold_registry._registry:
                return key
        return target_dataset

    def extract_pipeline_features(self, 
                                  source: Union[np.ndarray, bytes, str], 
                                  dpi: Optional[float] = None,
                                  with_topology: bool = False,
                                  feature_mode: Optional[str] = None,
                                  dataset: Optional[str] = None) -> Dict[str, Any]:
        """Runs validation, preprocessing, minutiae extraction, and Delaunay or MCC descriptors."""
        mode = feature_mode or self.feature_mode
        ds = dataset or self.default_dataset
        capture = self.capture_adapter.ingest(source, dpi=dpi)
        preprocessed = self.preprocessor.process(capture)
        minutiae = self.minutiae_extractor.extract(preprocessed, raw_image=capture.image)

        if mode == "mcc_spectral":
            mean, std = self._get_mcc_scaler(ds)
            zca_mat = self._get_mcc_zca(ds)
            mcc_feats = self.mcc_builder.build(
                minutiae, dpi=capture.dpi, mean=mean, std=std, zca_matrix=zca_mat
            )
            return {
                "capture": capture,
                "preprocessed": preprocessed,
                "minutiae": minutiae,
                "mcc_features": mcc_feats,
                "descriptors": mcc_feats,
            }

        triangles = self.triangle_builder.build(minutiae, compute_topology=with_topology)
        descriptors = self.descriptor_builder.build(triangles, dpi=capture.dpi)
        return {
            "capture": capture,
            "preprocessed": preprocessed,
            "minutiae": minutiae,
            "triangles": triangles,
            "descriptors": descriptors
        }

    def enroll(self, 
               source: Union[np.ndarray, bytes, str], 
               subject_id: str, 
               salt: str, 
               app_id: str, 
               with_topology: bool = False,
               dpi: Optional[float] = None) -> Dict[str, Any]:
        """
        Executes full enrollment lifecycle:
        Capture -> Preprocess -> Minutiae -> Descriptors ->
        Keyed Projection -> Bitstrings (+ RFF Edge Tokens) -> Binary Framing -> AEAD Encrypt -> Store.
        """
        t0 = time.perf_counter()
        cred_id = str(uuid.uuid4())
        try:
            feats = self.extract_pipeline_features(source, dpi=dpi, with_topology=with_topology)
            desc_dto = feats["descriptors"]
            salt_norm = normalize_salt(salt)

            if self.feature_mode == "mcc_spectral":
                bitstrings = self.projection_engine.project_and_quantize(
                    descriptors=desc_dto.descriptors,
                    salt=salt_norm,
                    app_id=app_id,
                    subject_id=subject_id,
                    in_dim=256,
                    out_dim=self.out_dim,
                    mean=None,
                    std=None
                )
                W_rff, b_rff = self.projection_engine.get_rff_edge_params(
                    salt=salt_norm,
                    app_id=app_id,
                    subject_id=subject_id,
                    edge_dim=7,
                    rff_bits=self.rff_bits
                )
                edge_mask = None
                if self.compact_edges:
                    edge_tokens, edge_mask = self.mcc_builder.encode_protected_sparse_edge_tokens_with_params(
                        desc_dto, W_rff=W_rff, b_rff=b_rff, top_k=self.edge_top_k
                    )
                else:
                    edge_tokens = self.mcc_builder.encode_protected_edge_tokens_with_params(
                        desc_dto, W_rff=W_rff, b_rff=b_rff
                    )
                cred_dto = ProtectedCredentialDTO(
                    credential_id=cred_id,
                    subject_id=subject_id,
                    salt=salt_norm,
                    app_id=app_id,
                    bitstrings=bitstrings,
                    weights=desc_dto.weights,
                    num_triangles=desc_dto.count,
                    packed_topology=None,
                    edge_tokens=edge_tokens,
                    in_dim=256,
                    edge_mask=edge_mask,
                )
            else:
                tri_dto = feats["triangles"]
                scaler = self.scaler_registry.get_scaler_params(self.default_dataset)
                mean = scaler.get("mean") if scaler else None
                std = scaler.get("std") if scaler else None

                # Keyed projection and quantization
                bitstrings = self.projection_engine.project_and_quantize(
                    descriptors=desc_dto.descriptors,
                    salt=salt_norm,
                    app_id=app_id,
                    subject_id=subject_id,
                    in_dim=9,
                    out_dim=32,
                    mean=mean,
                    std=std
                )

                # Optional topology packing
                packed_topo = None
                if with_topology and tri_dto.adjacency_matrix is not None:
                    packed_topo = pack_adjacency_matrix(tri_dto.adjacency_matrix)

                cred_dto = ProtectedCredentialDTO(
                    credential_id=cred_id,
                    subject_id=subject_id,
                    salt=salt_norm,
                    app_id=app_id,
                    bitstrings=bitstrings,
                    weights=desc_dto.weights,
                    num_triangles=desc_dto.count,
                    packed_topology=packed_topo
                )

            # Derive AEAD storage key via KeyService
            storage_key = self.key_service.derive_storage_key(salt=salt_norm, app_id=app_id)

            # Serialize and encrypt at rest
            envelope = self.serializer.serialize_protected(
                credential=cred_dto,
                storage_key=storage_key,
                encrypt=True
            )

            # Persist in repository
            stored_id = self.repository.store(
                subject_id=subject_id,
                envelope=envelope,
                metadata={"app_id": app_id, "num_triangles": desc_dto.count, "feature_mode": self.feature_mode}
            )

            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.audit_logger.log_event(
                operation="ENROLL",
                subject_id=subject_id,
                decision="SUCCESS",
                score=None,
                latency_ms=latency_ms,
                error_code=int(BiometricErrorCode.SUCCESS),
                metadata={
                    "num_triangles": desc_dto.count,
                    "has_topology": with_topology,
                    "has_edge_tokens": envelope.has_edge_tokens,
                    "bytes": envelope.total_bytes,
                }
            )

            return {
                "status": "success",
                "credential_id": stored_id,
                "subject_id": subject_id,
                "num_triangles": desc_dto.count,
                "serialized_bytes": envelope.total_bytes,
                "latency_ms": latency_ms
            }

        except BiometricServiceException as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.audit_logger.log_event(
                operation="ENROLL",
                subject_id=subject_id,
                decision="ERROR",
                score=None,
                latency_ms=latency_ms,
                error_code=int(e.code),
                metadata={"error": e.message}
            )
            return {"status": "error", "code": int(e.code), "message": e.message}

    def verify(self, 
               source: Union[np.ndarray, bytes, str], 
               subject_id: str, 
               app_id: str, 
               dataset: Optional[str] = None,
               operating_point: str = "EER",
               algorithm: Optional[str] = None,
               dpi: Optional[float] = None) -> MatchResult:
        """
        Executes full verification lifecycle:
        Fetches & AEAD decrypts enrolled credential -> Extracts query features ->
        Projects query under enrolled salt/key -> Matches templates -> Evaluates threshold decision.
        """
        t0 = time.perf_counter()
        target_dataset = dataset or self.default_dataset
        algo = algorithm or self.matcher.default_algorithm
        thresh_ds = self._resolve_threshold_dataset(target_dataset, self.feature_mode == "mcc_spectral")

        try:
            # 1. Check repository and retrieve active enrolled credential
            raw_bytes = self.repository.get_raw_bytes(subject_id)
            # Peek salt from header (bytes 28..44)
            salt_raw = raw_bytes[28:44]
            salt_str = salt_raw.rstrip(b"\x00").decode("utf-8", errors="replace")

            storage_key = self.key_service.derive_storage_key(salt=salt_str, app_id=app_id)
            enrolled_cred = self.serializer.deserialize_protected(
                raw_bytes=raw_bytes,
                storage_key=storage_key,
                subject_id=subject_id,
                app_id=app_id
            )

            use_mcc = (enrolled_cred.in_dim == 256) or (enrolled_cred.edge_tokens is not None)
            thresh_ds = self._resolve_threshold_dataset(target_dataset, use_mcc)

            # 2. Extract query features
            has_topology = enrolled_cred.packed_topology is not None
            feats = self.extract_pipeline_features(
                source,
                dpi=dpi,
                with_topology=has_topology,
                feature_mode="mcc_spectral" if use_mcc else "delaunay",
                dataset=target_dataset,
            )
            q_desc = feats["descriptors"]

            if use_mcc:
                enrolled_out_dim = int(enrolled_cred.bitstrings.shape[1]) if enrolled_cred.bitstrings.ndim == 2 else self.out_dim
                q_bitstrings = self.projection_engine.project_and_quantize(
                    descriptors=q_desc.descriptors,
                    salt=enrolled_cred.salt,
                    app_id=app_id,
                    subject_id=subject_id,
                    in_dim=256,
                    out_dim=enrolled_out_dim,
                    mean=None,
                    std=None
                )
                q_bitstrings_dual = self.projection_engine.project_and_quantize(
                    descriptors=q_desc.descriptors_dual,
                    salt=enrolled_cred.salt,
                    app_id=app_id,
                    subject_id=subject_id,
                    in_dim=256,
                    out_dim=enrolled_out_dim,
                    mean=None,
                    std=None
                )
                enrolled_rff_bits = (
                    int(enrolled_cred.edge_tokens.shape[-1]) * 64
                    if enrolled_cred.edge_tokens is not None
                    else self.rff_bits
                )
                W_rff, b_rff = self.projection_engine.get_rff_edge_params(
                    salt=enrolled_cred.salt,
                    app_id=app_id,
                    subject_id=subject_id,
                    edge_dim=7,
                    rff_bits=enrolled_rff_bits,
                )
                q_edge_tokens = self.mcc_builder.encode_protected_edge_tokens_with_params(
                    q_desc, W_rff=W_rff, b_rff=b_rff
                )
                score, det = self.matcher.match_protected_spectral(
                    q_bitstrings=q_bitstrings,
                    e_bitstrings=enrolled_cred.bitstrings,
                    q_weights=q_desc.weights,
                    e_weights=enrolled_cred.weights,
                    q_edge_tokens=q_edge_tokens,
                    e_edge_tokens=enrolled_cred.edge_tokens,
                    q_bitstrings_dual=q_bitstrings_dual,
                    e_edge_mask=enrolled_cred.edge_mask,
                    return_details=True,
                )
                matched_count = int(det.get("num_matched", 0))
                algo_used = algorithm or "mcc_spectral"
            else:
                # 3. Project query descriptors under enrolled salt and key
                scaler = self.scaler_registry.get_scaler_params(target_dataset)
                mean = scaler.get("mean") if scaler else None
                std = scaler.get("std") if scaler else None

                q_bitstrings = self.projection_engine.project_and_quantize(
                    descriptors=q_desc.descriptors,
                    salt=enrolled_cred.salt,
                    app_id=app_id,
                    subject_id=subject_id,
                    in_dim=9,
                    out_dim=32,
                    mean=mean,
                    std=std
                )

                # 4. Match
                if has_topology and algo == "graph_consensus":
                    e_adj = unpack_adjacency_matrix(enrolled_cred.packed_topology, enrolled_cred.num_triangles)
                    q_adj = feats["triangles"].adjacency_matrix
                    score = self.matcher.match_protected_with_topology(
                        q_bitstrings=q_bitstrings,
                        e_bitstrings=enrolled_cred.bitstrings,
                        q_weights=q_desc.weights,
                        e_weights=enrolled_cred.weights,
                        e_adj=e_adj,
                        q_adj=q_adj
                    )
                else:
                    score = self.matcher.match_protected(
                        q_bitstrings=q_bitstrings,
                        e_bitstrings=enrolled_cred.bitstrings,
                        q_weights=q_desc.weights,
                        e_weights=enrolled_cred.weights,
                        algorithm=algo
                    )
                matched_count = min(len(q_bitstrings), enrolled_cred.num_triangles)
                algo_used = algo

            # 5. Evaluate decision against calibrated threshold
            threshold = self.threshold_registry.get_threshold(thresh_ds, operating_point)
            decision = "ACCEPT" if score >= threshold else "REJECT"
            latency_ms = (time.perf_counter() - t0) * 1000.0

            result = MatchResult(
                decision=decision,
                score=float(score),
                threshold=float(threshold),
                operating_point=operating_point,
                algorithm=algo_used,
                matched_triangles=matched_count,
                total_query_triangles=len(q_bitstrings),
                total_enrolled_triangles=enrolled_cred.num_triangles,
                error_code=BiometricErrorCode.SUCCESS,
                latency_ms=latency_ms
            )

            self.audit_logger.log_event(
                operation="VERIFY",
                subject_id=subject_id,
                decision=decision,
                score=score,
                latency_ms=latency_ms,
                error_code=int(BiometricErrorCode.SUCCESS),
                metadata={"dataset": thresh_ds, "algorithm": algo_used}
            )

            return result

        except BiometricServiceException as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            threshold = (
                self.threshold_registry.get_threshold(thresh_ds, operating_point)
                if e.code not in (BiometricErrorCode.THRESHOLD_NOT_FOUND, BiometricErrorCode.INVALID_OPERATING_POINT)
                else 0.0
            )
            self.audit_logger.log_event(
                operation="VERIFY",
                subject_id=subject_id,
                decision="ERROR",
                score=0.0,
                latency_ms=latency_ms,
                error_code=int(e.code),
                metadata={"error": e.message}
            )
            return MatchResult(
                decision="REJECT",
                score=0.0,
                threshold=threshold,
                operating_point=operating_point,
                algorithm=algo,
                matched_triangles=0,
                total_query_triangles=0,
                total_enrolled_triangles=0,
                error_code=e.code,
                latency_ms=latency_ms
            )

    def revoke(self, subject_id: str, reason: str = "revocation_request") -> bool:
        """Revokes a subject's credential in the repository and logs an audit event."""
        t0 = time.perf_counter()
        ok = self.repository.revoke(subject_id, reason=reason)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.audit_logger.log_event(
            operation="REVOKE",
            subject_id=subject_id,
            decision="REVOKED" if ok else "NOT_FOUND",
            score=None,
            latency_ms=latency_ms,
            error_code=int(BiometricErrorCode.SUCCESS if ok else BiometricErrorCode.CREDENTIAL_NOT_FOUND),
            metadata={"reason": reason}
        )
        return ok

    def reissue(self, 
                source: Union[np.ndarray, bytes, str], 
                subject_id: str, 
                new_salt: str, 
                app_id: str, 
                with_topology: bool = False,
                dpi: Optional[float] = None) -> Dict[str, Any]:
        """
        Executes fresh presentation reissuance workflow:
        Revokes old credential -> Enrolls newly captured biometric under new_salt.
        """
        self.revoke(subject_id, reason="reissuance_fresh_presentation")
        res = self.enroll(
            source=source,
            subject_id=subject_id,
            salt=new_salt,
            app_id=app_id,
            with_topology=with_topology,
            dpi=dpi
        )
        if res.get("status") == "success":
            res["reissued"] = True
        return res

    def rotate_master_secret(self, new_master_secret: Union[bytes, str], reason: str = "scheduled_rotation"):
        """
        Rotates the underlying master secret in KeyService, invalidates the projection matrix cache,
        and records a privacy-minimised audit event.
        """
        self.key_service.rotate_master_secret(new_master_secret, reason=reason)
        self.projection_engine.clear_cache()
        self.audit_logger.log_event(
            operation="KEY_ROTATION",
            subject_id="SYSTEM_KMS",
            decision="SUCCESS",
            score=None,
            latency_ms=0.0,
            error_code=int(BiometricErrorCode.SUCCESS),
            metadata={"new_epoch": self.key_service.epoch, "reason": reason}
        )
