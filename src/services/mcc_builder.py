"""
Minutiae Cylinder-Code (MCC) Descriptor & Edge-Token Builder Service.

Provides modular extraction of fixed-length 256-D MCC local cylinder vectors,
cross-database symmetric channel standardization, dual 180-degree ridge-orientation
descriptors, and zero-plaintext keyed RFF binary edge tokens for SE(2) spectral matching.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.descriptors.mcc import MinutiaeCylinderCode, flip_mcc_dual
from src.matching.spectral_consensus import derive_rff_edge_params, encode_edge_rff
from src.services.types import (
    BiometricErrorCode,
    ExtractedMinutiae,
    InsufficientMinutiaeError,
)


def symmetrize_mcc_stats(
    mean: np.ndarray,
    std: np.ndarray,
    num_spatial_cells: int = 8,
    num_directional_slices: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetrizes empirical channel mean and std under 180-degree orientation flip
    so forward and dual (theta + pi) MCC descriptors inhabit the same calibrated space.
    """
    mu = np.asarray(mean, dtype=np.float32)
    sd = np.asarray(std, dtype=np.float32)
    mu_sym = 0.5 * (mu + flip_mcc_dual(mu, num_spatial_cells, num_directional_slices))
    sd_sym = 0.5 * (sd + flip_mcc_dual(sd, num_spatial_cells, num_directional_slices))
    return mu_sym.astype(np.float32), np.maximum(sd_sym, 1e-4).astype(np.float32)


@dataclass
class MCCBuiltFeatures:
    """Container for MCC descriptor matrices, quality weights, and transient geometry."""

    descriptors: np.ndarray       # (N, 256) float32 standardized + L2-normalized
    descriptors_dual: np.ndarray  # (N, 256) float32 dual standardized + L2-normalized
    raw_descriptors: np.ndarray   # (N, 256) float32 raw unstandardized
    weights: np.ndarray           # (N,) float32 quality weights
    coords: np.ndarray            # (N, 2) float32 transient pixel coordinates
    thetas: np.ndarray            # (N,) float32 transient ridge tangent orientations
    dpi: float = 500.0
    in_dim: int = 256
    count: int = 0

    def __post_init__(self):
        self.count = len(self.descriptors)


class MCCDescriptorBuilder:
    """
    Modular service building calibrated 256-D MCC descriptors and keyed RFF edge tokens.
    """

    def __init__(
        self,
        radius: float = 70.0,
        num_spatial_cells: int = 8,
        num_directional_slices: int = 4,
        sigma_spatial: float = 8.5,
        sigma_directional: float = float(np.pi / 5.0),
        sigmoid_gain: float = 2.0,
        min_minutiae: int = 3,
    ):
        self.radius = radius
        self.num_spatial_cells = num_spatial_cells
        self.num_directional_slices = num_directional_slices
        self.sigma_spatial = sigma_spatial
        self.sigma_directional = sigma_directional
        self.sigmoid_gain = sigmoid_gain
        self.min_minutiae = min_minutiae
        self.in_dim = num_spatial_cells * num_spatial_cells * num_directional_slices

    def build(
        self,
        extracted: Union[ExtractedMinutiae, List[Any]],
        dpi: float = 500.0,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        zca_matrix: Optional[np.ndarray] = None,
        enforce_min_count: bool = True,
    ) -> MCCBuiltFeatures:
        """
        Extracts 256-D MCC descriptors, applies symmetric Set-B z-score (+ optional
        symmetric ZCA whitening) and L2 normalization when stats are supplied, and
        constructs dual orientation copies.
        """
        minutiae = extracted.minutiae if isinstance(extracted, ExtractedMinutiae) else list(extracted)
        if enforce_min_count and len(minutiae) < self.min_minutiae:
            raise InsufficientMinutiaeError(
                BiometricErrorCode.INSUFFICIENT_MINUTIAE,
                f"MCC requires at least {self.min_minutiae} minutiae, got {len(minutiae)}",
            )

        if len(minutiae) == 0:
            return MCCBuiltFeatures(
                descriptors=np.empty((0, self.in_dim), dtype=np.float32),
                descriptors_dual=np.empty((0, self.in_dim), dtype=np.float32),
                raw_descriptors=np.empty((0, self.in_dim), dtype=np.float32),
                weights=np.empty((0,), dtype=np.float32),
                coords=np.empty((0, 2), dtype=np.float32),
                thetas=np.empty((0,), dtype=np.float32),
                dpi=dpi,
                in_dim=self.in_dim,
            )

        mcc = MinutiaeCylinderCode(
            radius=self.radius,
            num_spatial_cells=self.num_spatial_cells,
            num_directional_slices=self.num_directional_slices,
            sigma_spatial=self.sigma_spatial,
            sigma_directional=self.sigma_directional,
            sigmoid_gain=self.sigmoid_gain,
            dpi=dpi,
        )
        raw_descs, weights = mcc.extract_descriptors(minutiae)

        if mean is not None and std is not None:
            mu_s, sd_s = symmetrize_mcc_stats(
                mean, std, self.num_spatial_cells, self.num_directional_slices
            )
            ds = (raw_descs - mu_s) / sd_s
        elif mean is not None:
            mu_s = 0.5 * (
                np.asarray(mean, dtype=np.float32)
                + flip_mcc_dual(np.asarray(mean, dtype=np.float32), self.num_spatial_cells, self.num_directional_slices)
            )
            ds = raw_descs - mu_s
        else:
            ds = raw_descs.astype(np.float32, copy=True)

        if zca_matrix is not None:
            w_zca = np.asarray(zca_matrix, dtype=np.float32)
            if w_zca.shape == (self.in_dim, self.in_dim):
                ds = (ds @ w_zca).astype(np.float32)

        norms = np.linalg.norm(ds, axis=1, keepdims=True)
        ds = (ds / np.maximum(norms, 1e-6)).astype(np.float32)
        ds_dual = mcc.get_dual_descriptors(ds).astype(np.float32)

        coords = np.array([[m.x, m.y] for m in minutiae], dtype=np.float32)
        thetas = np.array([m.theta for m in minutiae], dtype=np.float32)

        return MCCBuiltFeatures(
            descriptors=ds,
            descriptors_dual=ds_dual,
            raw_descriptors=raw_descs.astype(np.float32),
            weights=weights.astype(np.float32),
            coords=coords,
            thetas=thetas,
            dpi=dpi,
            in_dim=self.in_dim,
        )

    @staticmethod
    def encode_protected_edge_tokens_with_params(
        features: MCCBuiltFeatures,
        W_rff: np.ndarray,
        b_rff: np.ndarray,
        sigma_dist: float = 18.0,
        sigma_angle: float = 0.38,
        sigma_edge_angle: float = 0.34,
    ) -> np.ndarray:
        """Encodes transient geometry into keyed RFF edge tokens using pre-derived KMS params."""
        return encode_edge_rff(
            coords=features.coords,
            thetas=features.thetas,
            W=W_rff,
            b=b_rff,
            sigma_dist=sigma_dist,
            sigma_angle=sigma_angle,
            sigma_edge_angle=sigma_edge_angle,
            dpi=features.dpi,
        )

    @staticmethod
    def encode_protected_sparse_edge_tokens_with_params(
        features: MCCBuiltFeatures,
        W_rff: np.ndarray,
        b_rff: np.ndarray,
        top_k: int = 20,
        mode: str = "topk",
        sigma_dist: float = 18.0,
        sigma_angle: float = 0.38,
        sigma_edge_angle: float = 0.34,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encodes transient geometry into masked sparse keyed RFF edge tokens
        `(edge_tokens_dense_zeroed, edge_mask)` for compact wire serialization.
        """
        from src.matching.spectral_consensus import decode_sparse_edge_rff, encode_sparse_edge_rff

        compact_tokens, edge_mask = encode_sparse_edge_rff(
            coords=features.coords,
            thetas=features.thetas,
            W=W_rff,
            b=b_rff,
            top_k=top_k,
            mode=mode,
            sigma_dist=sigma_dist,
            sigma_angle=sigma_angle,
            sigma_edge_angle=sigma_edge_angle,
            dpi=features.dpi,
        )
        return decode_sparse_edge_rff(compact_tokens, edge_mask), edge_mask

    @staticmethod
    def encode_protected_edge_tokens(
        features: MCCBuiltFeatures,
        master_secret: Union[bytes, str],
        salt: Union[bytes, str],
        app_id: str,
        subject_id: str = "",
        rff_bits: int = 128,
        sigma_dist: float = 18.0,
        sigma_angle: float = 0.38,
        sigma_edge_angle: float = 0.34,
    ) -> np.ndarray:
        """
        Transforms transient minutia geometry into zero-plaintext keyed RFF binary
        edge tokens `(N, N, rff_bits // 64)` suitable for spectral graph consensus.
        """
        W_rff, b_rff = derive_rff_edge_params(
            master_secret=master_secret,
            salt=salt,
            app_id=app_id,
            subject_id=subject_id,
            edge_dim=7,
            rff_bits=rff_bits,
        )
        return MCCDescriptorBuilder.encode_protected_edge_tokens_with_params(
            features=features,
            W_rff=W_rff,
            b_rff=b_rff,
            sigma_dist=sigma_dist,
            sigma_angle=sigma_angle,
            sigma_edge_angle=sigma_edge_angle,
        )
