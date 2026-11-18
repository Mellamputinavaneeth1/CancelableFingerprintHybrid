"""
Projection Engine Service.

Manages column-orthonormal projection matrices, matrix multiplication,
and 1-bit sign quantization into cancelable bitstrings.
Includes high-performance in-memory projection matrix caching.
"""

import threading
from typing import Dict, Tuple, Optional
import numpy as np

from src.protection.projection import project_descriptors
from src.protection.quantizer import quantize_descriptors
from src.services.key_service import KeyService


class ProjectionEngine:
    """Production service for keyed projection and binary quantization."""
    def __init__(self, key_service: KeyService, cache_matrices: bool = True):
        self.key_service = key_service
        self.cache_matrices = cache_matrices
        self._matrix_cache: Dict[Tuple[str, str, str, int, int], np.ndarray] = {}
        self._rff_cache: Dict[Tuple[str, str, str, int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._cache_generation: int = 0
        self._lock = threading.Lock()

    def get_projection_matrix(self, 
                              salt: str, 
                              app_id: str, 
                              subject_id: str = "",
                              in_dim: int = 9, 
                              out_dim: int = 32) -> np.ndarray:
        """
        Retrieves or derives column-orthonormal projection matrix W.
        Uses cached copy if caching is enabled.
        """
        cache_key = (salt, app_id, subject_id, in_dim, out_dim)
        gen_at_start = 0
        if self.cache_matrices:
            with self._lock:
                if cache_key in self._matrix_cache:
                    return self._matrix_cache[cache_key]
                gen_at_start = self._cache_generation

        W = self.key_service.derive_projection_matrix(
            salt=salt,
            app_id=app_id,
            subject_id=subject_id,
            in_dim=in_dim,
            out_dim=out_dim
        )

        if self.cache_matrices:
            with self._lock:
                # Only insert into cache if no cache invalidation / rotation occurred mid-flight
                if self._cache_generation == gen_at_start:
                    self._matrix_cache[cache_key] = W

        return W

    def get_rff_edge_params(
        self,
        salt: str,
        app_id: str,
        subject_id: str = "",
        edge_dim: int = 7,
        rff_bits: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieves or derives keyed RFF edge parameters (W_rff, b_rff)."""
        cache_key = (salt, app_id, subject_id, edge_dim, rff_bits)
        gen_at_start = 0
        if self.cache_matrices:
            with self._lock:
                if cache_key in self._rff_cache:
                    return self._rff_cache[cache_key]
                gen_at_start = self._cache_generation

        params = self.key_service.derive_rff_edge_params(
            salt=salt,
            app_id=app_id,
            subject_id=subject_id,
            edge_dim=edge_dim,
            rff_bits=rff_bits,
        )

        if self.cache_matrices:
            with self._lock:
                if self._cache_generation == gen_at_start:
                    self._rff_cache[cache_key] = params

        return params

    def clear_cache(self):
        """Clears cached projection matrices and RFF params and increments cache generation."""
        with self._lock:
            self._matrix_cache.clear()
            self._rff_cache.clear()
            self._cache_generation += 1

    def project_and_quantize(self, 
                             descriptors: np.ndarray, 
                             salt: str, 
                             app_id: str, 
                             subject_id: str = "",
                             in_dim: int = 9, 
                             out_dim: int = 32,
                             mean: Optional[np.ndarray] = None,
                             std: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Projects descriptors using keyed orthonormal matrix and quantizes via sign(z).

        Args:
            descriptors: (N, in_dim) raw geometric features.
            salt: Application/presentation salt.
            app_id: Application identifier.
            subject_id: Optional subject identifier.
            in_dim: Descriptor dimension.
            out_dim: Bitstring length (default 32).
            mean: Optional reference mean vector for z-score standardization.
            std: Optional reference std vector for z-score standardization.

        Returns:
            bitstrings: (N, out_dim) array of uint8 in {0, 1}.
        """
        if len(descriptors) == 0:
            return np.empty((0, out_dim), dtype=np.uint8)

        W = self.get_projection_matrix(
            salt=salt,
            app_id=app_id,
            subject_id=subject_id,
            in_dim=in_dim,
            out_dim=out_dim
        )
        projected = project_descriptors(descriptors, W, mean=mean, std=std)
        bitstrings = quantize_descriptors(projected)
        return bitstrings
