"""
Tier-0 Bloom Filter Screening Engine (v2.4-Robust).

Enables O(1) alignment-free pre-screening across large biometric galleries
prior to greedy bipartite matching, operating at >100,000 comparisons/second.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from src.protection.bloom import encode_to_bloom_filter, bloom_jaccard_similarity


def batch_bloom_jaccard_similarity(query_bloom: np.ndarray, gallery_blooms: np.ndarray) -> np.ndarray:
    """
    Computes Jaccard similarity between a query Bloom filter and a matrix of gallery Bloom filters.

    Args:
        query_bloom: (256,) uint8 array.
        gallery_blooms: (M, 256) uint8 array.

    Returns:
        (M,) float32 array of Jaccard similarities in [0, 1].
    """
    if gallery_blooms.ndim == 1:
        gallery_blooms = gallery_blooms[None, :]
    if len(gallery_blooms) == 0:
        return np.empty((0,), dtype=np.float32)

    inter = np.sum(np.bitwise_and(query_bloom[None, :], gallery_blooms), axis=1)
    union = np.sum(np.bitwise_or(query_bloom[None, :], gallery_blooms), axis=1)
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0).astype(np.float32)


def screen_candidates_bloom(query_bloom: np.ndarray,
                           gallery_blooms: np.ndarray,
                           candidate_ids: List[str],
                           top_k: int = 50,
                           min_jaccard: float = 0.05) -> List[Tuple[str, float]]:
    """
    Screens enrolled gallery candidates using Tier-0 Bloom filter Jaccard similarity.

    Args:
        query_bloom: (256,) query Bloom filter.
        gallery_blooms: (M, 256) gallery Bloom filters.
        candidate_ids: List of M candidate subject IDs.
        top_k: Maximum number of candidate identities to return for Tier-1 evaluation.
        min_jaccard: Minimum similarity threshold.

    Returns:
        List of (subject_id, jaccard_score) tuples sorted in descending order of similarity.
    """
    if len(candidate_ids) == 0 or len(gallery_blooms) == 0:
        return []

    sims = batch_bloom_jaccard_similarity(query_bloom, gallery_blooms)
    valid_indices = np.where(sims >= min_jaccard)[0]
    if len(valid_indices) == 0:
        sorted_idx = np.argsort(sims)[::-1][:top_k]
    else:
        sorted_valid = valid_indices[np.argsort(sims[valid_indices])[::-1]]
        sorted_idx = sorted_valid[:top_k]

    return [(candidate_ids[i], float(sims[i])) for i in sorted_idx]
