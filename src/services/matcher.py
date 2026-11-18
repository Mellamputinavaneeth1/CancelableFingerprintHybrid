"""
Matcher Service.

Provides modular matching algorithms across protected cancelable representations
and continuous unprotected baselines:
1. Block-augmented Hungarian bipartite matching (optimal injective assignment).
2. Greedy 1-to-1 matching (fast injective nearest neighbor).
3. Graph topological consensus matching (utility ablation).
4. Continuous unprotected descriptor-set matching.
"""

import time
from typing import Optional, Dict, Any
import numpy as np

from src.matching.bipartite_matcher import (
    match_protected_templates,
    match_protected_greedy,
    match_unprotected_descriptors,
    compute_pairwise_hamming_distance,
)
from src.matching.graph_consensus import match_protected_with_graph_consensus
from src.matching.spectral_consensus import SpectralGraphMatcher
from src.services.types import MatchResult, BiometricErrorCode, BiometricServiceException


class Matcher:
    """Production biometric matching service."""
    def __init__(self, 
                 default_algorithm: str = "hungarian",
                 hamming_threshold: float = 0.22,
                 distance_threshold: float = 0.25,
                 spectral_hamming_threshold: float = 0.44):
        self.default_algorithm = default_algorithm
        self.hamming_threshold = hamming_threshold
        self.distance_threshold = distance_threshold
        self.spectral_hamming_threshold = spectral_hamming_threshold
        self.spectral_matcher = SpectralGraphMatcher(
            sigma_dist=18.0,
            sigma_angle=0.38,
            power_iterations=18,
            sigma_edge_angle=0.34,
            top_k=6,
            use_lss=True,
            min_edge_compat=0.15,
            rff_gamma=1.00,
            max_candidates=400,
            min_eigenvector_weight=0.01,
            clique_prune_thresh=0.16,
        )

    def match_protected(self, 
                        q_bitstrings: np.ndarray, 
                        e_bitstrings: np.ndarray, 
                        q_weights: np.ndarray, 
                        e_weights: np.ndarray, 
                        algorithm: Optional[str] = None,
                        hamming_threshold: Optional[float] = None) -> float:
        """
        Executes cancelable binary template matching.

        Args:
            q_bitstrings: Query bitstrings (N_q, out_dim).
            e_bitstrings: Enrolled bitstrings (N_e, out_dim).
            q_weights: Query quality weights (N_q,).
            e_weights: Enrolled quality weights (N_e,).
            algorithm: "hungarian", "greedy_1to1", or "greedy_nn".
            hamming_threshold: Normalized distance threshold (default 0.22).

        Returns:
            Quality-weighted coverage-aware matching score in [0, 1].
        """
        algo = algorithm or self.default_algorithm
        tau = hamming_threshold if hamming_threshold is not None else self.hamming_threshold

        if len(q_bitstrings) == 0 or len(e_bitstrings) == 0:
            return 0.0

        if q_bitstrings.ndim not in (2, 3) or e_bitstrings.ndim not in (2, 3):
            raise BiometricServiceException(
                BiometricErrorCode.DIMENSION_MISMATCH,
                f"Bitstring arrays must be 2D or 3D, got query {q_bitstrings.shape}, enrolled {e_bitstrings.shape}"
            )
        q_dim = q_bitstrings.shape[-1]
        e_dim = e_bitstrings.shape[-1]
        if q_dim != e_dim:
            raise BiometricServiceException(
                BiometricErrorCode.DIMENSION_MISMATCH,
                f"Projection dimension mismatch: query out_dim={q_dim} vs enrolled out_dim={e_dim}"
            )
        if len(q_weights) != len(q_bitstrings) or len(e_weights) != len(e_bitstrings):
            raise BiometricServiceException(
                BiometricErrorCode.DIMENSION_MISMATCH,
                f"Weight length mismatch: query {len(q_weights)} vs {len(q_bitstrings)}, enrolled {len(e_weights)} vs {len(e_bitstrings)}"
            )

        try:
            if algo == "hungarian":
                return match_protected_templates(
                    q_bitstrings, e_bitstrings, q_weights, e_weights,
                    hamming_threshold=tau
                )
            elif algo == "greedy_1to1":
                return match_protected_greedy(
                    q_bitstrings, e_bitstrings, q_weights, e_weights,
                    hamming_threshold=tau, mode="greedy_1to1"
                )
            elif algo == "greedy_nn":
                return match_protected_greedy(
                    q_bitstrings, e_bitstrings, q_weights, e_weights,
                    hamming_threshold=tau, mode="greedy_nn"
                )
            else:
                raise BiometricServiceException(
                    BiometricErrorCode.MATCHING_ERROR,
                    f"Unknown protected matching algorithm: '{algo}'"
                )
        except BiometricServiceException:
            raise
        except ValueError as e:
            raise BiometricServiceException(
                BiometricErrorCode.DIMENSION_MISMATCH,
                f"Matcher dimension error: {str(e)}"
            )
        except Exception as e:
            raise BiometricServiceException(
                BiometricErrorCode.MATCHING_ERROR,
                f"Protected matching failed: {str(e)}"
            )

    def match_protected_with_topology(self,
                                      q_bitstrings: np.ndarray,
                                      e_bitstrings: np.ndarray,
                                      q_weights: np.ndarray,
                                      e_weights: np.ndarray,
                                      e_adj: np.ndarray,
                                      q_adj: Optional[np.ndarray] = None,
                                      alpha: float = 0.25,
                                      hamming_threshold: Optional[float] = None) -> float:
        """
        Executes protected matching with optional graph consensus re-ranking.
        """
        tau = hamming_threshold if hamming_threshold is not None else self.hamming_threshold
        if len(q_bitstrings) == 0 or len(e_bitstrings) == 0:
            return 0.0

        if q_bitstrings.ndim not in (2, 3) or e_bitstrings.ndim not in (2, 3):
            raise BiometricServiceException(
                BiometricErrorCode.DIMENSION_MISMATCH,
                f"Bitstring arrays must be 2D or 3D, got query {q_bitstrings.shape}, enrolled {e_bitstrings.shape}"
            )
        if q_bitstrings.shape[-1] != e_bitstrings.shape[-1]:
            raise BiometricServiceException(
                BiometricErrorCode.DIMENSION_MISMATCH,
                f"Projection dimension mismatch: query out_dim={q_bitstrings.shape[-1]} vs enrolled out_dim={e_bitstrings.shape[-1]}"
            )
        if len(q_weights) != len(q_bitstrings) or len(e_weights) != len(e_bitstrings):
            raise BiometricServiceException(
                BiometricErrorCode.DIMENSION_MISMATCH,
                f"Weight length mismatch: query {len(q_weights)} vs {len(q_bitstrings)}, enrolled {len(e_weights)} vs {len(e_bitstrings)}"
            )

        n_e = len(e_bitstrings)
        if e_adj is None or e_adj.shape != (n_e, n_e):
            raise BiometricServiceException(
                BiometricErrorCode.DIMENSION_MISMATCH,
                f"Enrolled adjacency matrix shape {getattr(e_adj, 'shape', None)} must match enrolled count ({n_e}, {n_e})"
            )

        n_q = len(q_bitstrings)
        if q_adj is None:
            q_adj = np.zeros((n_q, n_q), dtype=np.uint8)
            np.fill_diagonal(q_adj, 3)
        elif q_adj.shape != (n_q, n_q):
            raise BiometricServiceException(
                BiometricErrorCode.DIMENSION_MISMATCH,
                f"Query adjacency matrix shape {q_adj.shape} must match query count ({n_q}, {n_q})"
            )

        try:
            score, _ = match_protected_with_graph_consensus(
                query_bits=q_bitstrings,
                enrolled_bits=e_bitstrings,
                query_weights=q_weights,
                enrolled_weights=e_weights,
                query_adj=q_adj,
                enrolled_adj=e_adj,
                hamming_threshold=tau,
                affinity_bonus=alpha
            )
            return float(score)
        except Exception as e:
            raise BiometricServiceException(
                BiometricErrorCode.MATCHING_ERROR,
                f"Topology consensus matching failed: {str(e)}"
            )

    def match_unprotected(self, 
                          q_descriptors: np.ndarray, 
                          e_descriptors: np.ndarray, 
                          q_weights: np.ndarray, 
                          e_weights: np.ndarray, 
                          distance_threshold: Optional[float] = None,
                          mean: Optional[np.ndarray] = None,
                          std: Optional[np.ndarray] = None) -> float:
        """
        Executes continuous unprotected baseline matching.
        """
        tau = distance_threshold if distance_threshold is not None else self.distance_threshold
        if len(q_descriptors) == 0 or len(e_descriptors) == 0:
            return 0.0

        if q_descriptors.shape[-1] != e_descriptors.shape[-1]:
            raise BiometricServiceException(
                BiometricErrorCode.DIMENSION_MISMATCH,
                f"Continuous feature dimension mismatch: query {q_descriptors.shape[-1]} vs enrolled {e_descriptors.shape[-1]}"
            )
        if len(q_weights) != len(q_descriptors) or len(e_weights) != len(e_descriptors):
            raise BiometricServiceException(
                BiometricErrorCode.DIMENSION_MISMATCH,
                f"Continuous weight length mismatch: query {len(q_weights)} vs {len(q_descriptors)}, enrolled {len(e_weights)} vs {len(e_descriptors)}"
            )

        try:
            return match_unprotected_descriptors(
                q_descriptors, e_descriptors, q_weights, e_weights,
                distance_threshold=tau, mean=mean, std=std
            )
        except Exception as e:
            raise BiometricServiceException(
                BiometricErrorCode.MATCHING_ERROR,
                f"Unprotected matching failed: {str(e)}"
            )

    def match_protected_spectral(
        self,
        q_bitstrings: np.ndarray,
        e_bitstrings: np.ndarray,
        q_weights: np.ndarray,
        e_weights: np.ndarray,
        q_edge_tokens: np.ndarray,
        e_edge_tokens: np.ndarray,
        q_bitstrings_dual: Optional[np.ndarray] = None,
        q_edge_mask: Optional[np.ndarray] = None,
        e_edge_mask: Optional[np.ndarray] = None,
        hamming_threshold: Optional[float] = None,
        return_details: bool = False,
    ):
        """
        Executes privacy-preserving spectral graph consensus matching over protected
        MCC bitstrings (and optional 180-deg query dual) plus keyed RFF edge tokens.
        """
        tau = hamming_threshold if hamming_threshold is not None else self.spectral_hamming_threshold
        if len(q_bitstrings) < 3 or len(e_bitstrings) < 3:
            return (0.0, {"num_matched": 0}) if return_details else 0.0

        if q_bitstrings.ndim != 2 or e_bitstrings.ndim != 2:
            raise BiometricServiceException(
                BiometricErrorCode.DIMENSION_MISMATCH,
                f"Bitstring arrays must be 2D, got query {q_bitstrings.shape}, enrolled {e_bitstrings.shape}"
            )
        if q_bitstrings.shape[1] != e_bitstrings.shape[1]:
            raise BiometricServiceException(
                BiometricErrorCode.DIMENSION_MISMATCH,
                f"Bitstring dimension mismatch: query {q_bitstrings.shape[1]} vs enrolled {e_bitstrings.shape[1]}"
            )
        n_q, n_e = len(q_bitstrings), len(e_bitstrings)
        if len(q_weights) != n_q or len(e_weights) != n_e:
            raise BiometricServiceException(
                BiometricErrorCode.DIMENSION_MISMATCH,
                f"Weight length mismatch: query {len(q_weights)} vs {n_q}, enrolled {len(e_weights)} vs {n_e}"
            )
        if (
            q_edge_tokens is None
            or e_edge_tokens is None
            or q_edge_tokens.shape[:2] != (n_q, n_q)
            or e_edge_tokens.shape[:2] != (n_e, n_e)
            or q_edge_tokens.shape[-1] != e_edge_tokens.shape[-1]
        ):
            raise BiometricServiceException(
                BiometricErrorCode.DIMENSION_MISMATCH,
                "Edge token shapes must match query/enrolled node counts and RFF bit width"
            )

        try:
            dmat = compute_pairwise_hamming_distance(q_bitstrings, e_bitstrings)
            if q_bitstrings_dual is not None and len(q_bitstrings_dual) == n_q:
                dmat_dual = compute_pairwise_hamming_distance(q_bitstrings_dual, e_bitstrings)
                dmat = np.minimum(dmat, dmat_dual)
            score, details = self.spectral_matcher.match_protected(
                query_edge_tokens=q_edge_tokens,
                enrolled_edge_tokens=e_edge_tokens,
                pairwise_dists=dmat,
                tau=tau,
                query_weights=q_weights,
                enrolled_weights=e_weights,
                query_edge_mask=q_edge_mask,
                enrolled_edge_mask=e_edge_mask,
            )
            if return_details:
                return float(score), details
            return float(score)
        except BiometricServiceException:
            raise
        except Exception as e:
            raise BiometricServiceException(
                BiometricErrorCode.MATCHING_ERROR,
                f"Spectral protected matching failed: {str(e)}"
            )
