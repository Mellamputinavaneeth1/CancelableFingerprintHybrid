from typing import Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.matching.augmented_cost import build_augmented_cost


def compute_pairwise_hamming_distance(query_bits: np.ndarray, enrolled_bits: np.ndarray) -> np.ndarray:
    """
    Computes pairwise normalized Hamming distance in [0, 1].
    Optimized with native bitwise XOR and hardware popcount (np.bitwise_count)
    when inputs are bit matrices whose bit width is a multiple of 64, 32, or 8.
    """
    if enrolled_bits.ndim == 3 and query_bits.ndim == 2:
        diff_3 = (query_bits[:, None, None, :] != enrolled_bits[None, :, :, :])
        return np.min(np.mean(diff_3, axis=3), axis=2).astype(np.float32)
    elif enrolled_bits.ndim == 3 and query_bits.ndim == 3:
        q0 = query_bits[:, 0, :]
        diff_3 = (q0[:, None, None, :] != enrolled_bits[None, :, :, :])
        return np.min(np.mean(diff_3, axis=3), axis=2).astype(np.float32)
    elif enrolled_bits.ndim == 2 and query_bits.ndim == 3:
        diff_3 = (query_bits[:, None, :, :] != enrolled_bits[None, :, None, :])
        return np.min(np.mean(diff_3, axis=3), axis=2).astype(np.float32)
    elif (query_bits.ndim == 2 and enrolled_bits.ndim == 2 and
          query_bits.shape[1] == enrolled_bits.shape[1] and
          query_bits.shape[1] > 0 and hasattr(np, "bitwise_count")):
        if query_bits.shape[0] == 0 or enrolled_bits.shape[0] == 0:
            return np.empty((query_bits.shape[0], enrolled_bits.shape[0]), dtype=np.float32)
        n_bits = int(query_bits.shape[1])
        q_u8 = np.ascontiguousarray(query_bits > 0, dtype=np.uint8)
        e_u8 = np.ascontiguousarray(enrolled_bits > 0, dtype=np.uint8)
        if n_bits == 32:
            q_packed = np.packbits(q_u8, axis=1).view(np.uint32).squeeze(axis=1)
            e_packed = np.packbits(e_u8, axis=1).view(np.uint32).squeeze(axis=1)
            xor_mat = np.bitwise_xor(q_packed[:, None], e_packed[None, :])
            return (np.bitwise_count(xor_mat).astype(np.float32)) / 32.0
        elif n_bits % 64 == 0:
            q_packed = np.ascontiguousarray(np.packbits(q_u8, axis=1)).view(np.uint64)
            e_packed = np.ascontiguousarray(np.packbits(e_u8, axis=1)).view(np.uint64)
            xor_mat = np.bitwise_xor(q_packed[:, None, :], e_packed[None, :, :])
            return np.bitwise_count(xor_mat).sum(axis=2, dtype=np.float32) * np.float32(1.0 / n_bits)
        elif n_bits % 8 == 0:
            q_packed = np.packbits(q_u8, axis=1)
            e_packed = np.packbits(e_u8, axis=1)
            xor_mat = np.bitwise_xor(q_packed[:, None, :], e_packed[None, :, :])
            return np.bitwise_count(xor_mat).sum(axis=2, dtype=np.float32) * np.float32(1.0 / n_bits)
    return np.mean(query_bits[:, None, :] != enrolled_bits[None, :, :], axis=2).astype(np.float32)

def match_unprotected_descriptors(query_desc: np.ndarray, enrolled_desc: np.ndarray,
                                  query_weights: np.ndarray, enrolled_weights: np.ndarray,
                                  distance_threshold: float = 0.25,
                                  mean: Optional[np.ndarray] = None,
                                  std: Optional[np.ndarray] = None) -> float:
    """
    Computes baseline matching score between raw continuous descriptors using
    Delaunay triangle descriptor-set bipartite matching with a corrected augmented slack penalty matrix.
    
    Terminology & Scope:
        This is a coverage-aware descriptor-set matcher with Hungarian one-to-one assignment.
        It evaluates the set of local invariant descriptors with one-to-one pairing; it does not
        enforce full graph isomorphism or global topological compatibility between adjacent triangles.
        
    Augmented Cost Matrix Formulation:
        Size: (N_q + N_e) x (N_q + N_e)
        Block [0:N_q, 0:N_e]: Valid pairwise normalized Euclidean distances (forbidden if >= tau)
        Block [0:N_q, N_e:]: Real query to dummy enrolled slack with diagonal tau / 2, off-diagonal big
        Block [N_q:, 0:N_e]: Real enrolled to dummy query slack with diagonal tau / 2, off-diagonal big
        Block [N_q:, N_e:]: Dummy-to-dummy slack pairs with cost 0.0
        
    Coverage-Aware Normalization:
        S = 2 * sum(s_uv) / (sum(w_q) + sum(w_e))
        Penalizes unmatched descriptors on BOTH query and enrolled sides.
        
    Standardization:
        If mean and std are provided, descriptors are standardized (z-scored) prior to distance
        computation, preventing high-variance physical dimensions (such as perimeter P_mm)
        from artificially dominating the Euclidean metric.
    """
    if query_desc.ndim == 1:
        query_desc = query_desc[None, :]
    if enrolled_desc.ndim == 1:
        enrolled_desc = enrolled_desc[None, :]
        
    n_q = len(query_desc)
    n_e = len(enrolled_desc)
    if n_q == 0 or n_e == 0:
        return 0.0
        
    query_weights = np.atleast_1d(query_weights).astype(np.float32)
    enrolled_weights = np.atleast_1d(enrolled_weights).astype(np.float32)
    if len(query_weights) != n_q or len(enrolled_weights) != n_e:
        raise ValueError(f"Weight length mismatch: query {len(query_weights)} vs {n_q}, enrolled {len(enrolled_weights)} vs {n_e}")
        
    if enrolled_desc.ndim == 3 and query_desc.ndim == 2:
        d_dim = query_desc.shape[1]
        if mean is not None and std is not None:
            q_norm = (query_desc - mean) / np.maximum(std, 1e-8)
            e_norm = (enrolled_desc - mean) / np.maximum(std, 1e-8)
            diff = q_norm[:, None, None, :] - e_norm[None, :, :, :]
        else:
            diff = query_desc[:, None, None, :] - enrolled_desc[None, :, :, :]
        dist_3 = np.linalg.norm(diff, axis=3) / np.sqrt(max(d_dim, 1))
        dist_matrix = np.min(dist_3, axis=2)
    elif enrolled_desc.ndim == 3 and query_desc.ndim == 3:
        d_dim = query_desc.shape[2]
        if mean is not None and std is not None:
            q_norm = (query_desc[:, 0, :] - mean) / np.maximum(std, 1e-8)
            e_norm = (enrolled_desc - mean) / np.maximum(std, 1e-8)
            diff = q_norm[:, None, None, :] - e_norm[None, :, :, :]
        else:
            diff = query_desc[:, 0, None, None, :] - enrolled_desc[None, :, :, :]
        dist_3 = np.linalg.norm(diff, axis=3) / np.sqrt(max(d_dim, 1))
        dist_matrix = np.min(dist_3, axis=2)
    elif enrolled_desc.ndim == 2 and query_desc.ndim == 3:
        d_dim = enrolled_desc.shape[1]
        if mean is not None and std is not None:
            q_norm = (query_desc - mean) / np.maximum(std, 1e-8)
            e_norm = (enrolled_desc - mean) / np.maximum(std, 1e-8)
            diff = q_norm[:, None, :, :] - e_norm[None, :, None, :]
        else:
            diff = query_desc[:, None, :, :] - enrolled_desc[None, :, None, :]
        dist_3 = np.linalg.norm(diff, axis=3) / np.sqrt(max(d_dim, 1))
        dist_matrix = np.min(dist_3, axis=2)
    else:
        d_dim = query_desc.shape[1]
        if mean is not None and std is not None:
            q_norm = (query_desc - mean) / np.maximum(std, 1e-8)
            e_norm = (enrolled_desc - mean) / np.maximum(std, 1e-8)
            diff = q_norm[:, None, :] - e_norm[None, :, :]
        else:
            diff = query_desc[:, None, :] - enrolled_desc[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2) / np.sqrt(max(d_dim, 1))

    
    tau = distance_threshold
    cost_augmented = build_augmented_cost(dist_matrix, tau)
    
    row_ind, col_ind = linear_sum_assignment(cost_augmented)
    
    matched_scores = []
    for r, c in zip(row_ind, col_ind):
        if r < n_q and c < n_e:
            d = dist_matrix[r, c]
            if d < tau:
                sim = 1.0 - (d / tau)
                w = np.sqrt(query_weights[r] * enrolled_weights[c])
                matched_scores.append(sim * w)
                
    if not matched_scores:
        return 0.0
        
    sum_q_weights = float(np.sum(query_weights))
    sum_e_weights = float(np.sum(enrolled_weights))
    coverage_denom = sum_q_weights + sum_e_weights + 1e-8
    
    return float(np.clip(2.0 * np.sum(matched_scores) / coverage_denom, 0.0, 1.0))

def match_protected_templates(query_bits: np.ndarray, enrolled_bits: np.ndarray,
                              query_weights: np.ndarray, enrolled_weights: np.ndarray,
                              hamming_threshold: float = 0.22,
                              coverage_penalty_exponent: float = 1.0,
                              min_matches: int = 0,
                              min_coverage_ratio: float = 0.0,
                              return_details: bool = False):
    """
    Computes optimal one-to-one matching score between protected binary descriptor sets
    using Delaunay triangle descriptor-set bipartite matching with a corrected augmented slack penalty matrix.
    
    Terminology & Scope:
        This is a coverage-aware descriptor-set bipartite matcher. It compares sets of local
        protected descriptors rather than verifying global topological graph isomorphism.
        
    Augmented Cost Matrix Structure:
        Constructs a (N_q + N_e) x (N_q + N_e) matrix:
        - Real pairs (u, v) have cost H_{u,v} if H_{u,v} < tau, else large forbidden cost.
        - Unmatched real query u maps to dummy enrolled node with cost tau / 2 on diagonal.
        - Unmatched real enrolled v maps to dummy query node with cost tau / 2 on diagonal.
        - Off-diagonal dummy entries are forbidden (large cost).
        - Dummy-to-dummy matches have cost 0.
        
    Coverage-Aware Normalization:
        S = 2 * sum(s_uv) / (sum(w_q) + sum(w_e))
        where s_uv = (1 - H_uv / tau) * sqrt(w_u * w_v).
        Ensures that partial-overlap matches receive proportional coverage penalties.
        When coverage_penalty_exponent > 1.0, min_matches > 0, or min_coverage_ratio > 0.0,
        sparse spurious matches are penalized to prevent low-coverage impostors from dominating.

    Returns:
        float if return_details is False, else tuple (float, dict with num_matched, matched_pairs, coverage, base_score).
    """
    empty_details = {"num_matched": 0, "matched_pairs": [], "coverage": 0.0, "base_score": 0.0, "tau": hamming_threshold}
    if query_bits.ndim == 1:
        query_bits = query_bits[None, :]
    if enrolled_bits.ndim == 1:
        enrolled_bits = enrolled_bits[None, :]
        
    n_q = len(query_bits)
    n_e = len(enrolled_bits)
    if n_q == 0 or n_e == 0:
        return (0.0, empty_details) if return_details else 0.0
        
    query_weights = np.atleast_1d(query_weights).astype(np.float32)
    enrolled_weights = np.atleast_1d(enrolled_weights).astype(np.float32)
    if len(query_weights) != n_q or len(enrolled_weights) != n_e:
        raise ValueError(f"Weight length mismatch: query {len(query_weights)} vs {n_q}, enrolled {len(enrolled_weights)} vs {n_e}")
        
    # Pairwise normalized Hamming distance matrix in [0, 1]
    dist_matrix = compute_pairwise_hamming_distance(query_bits, enrolled_bits)
    
    tau = hamming_threshold
    cost_augmented = build_augmented_cost(dist_matrix, tau)
    
    row_ind, col_ind = linear_sum_assignment(cost_augmented)
    
    matched_scores = []
    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        if r < n_q and c < n_e:
            dist = dist_matrix[r, c]
            if dist < tau:
                sim = 1.0 - (dist / tau)
                w = np.sqrt(query_weights[r] * enrolled_weights[c])
                matched_scores.append(sim * w)
                matched_pairs.append((int(r), int(c)))
                
    if not matched_scores:
        return (0.0, empty_details) if return_details else 0.0
        
    sum_q_weights = float(np.sum(query_weights))
    sum_e_weights = float(np.sum(enrolled_weights))
    coverage_denom = sum_q_weights + sum_e_weights + 1e-8
    
    base_score = float(np.clip(2.0 * np.sum(matched_scores) / coverage_denom, 0.0, 1.0))
    if base_score == 0.0:
        details = {"num_matched": len(matched_scores), "matched_pairs": matched_pairs, "coverage": 0.0, "base_score": 0.0, "tau": tau}
        return (0.0, details) if return_details else 0.0
        
    num_matched = len(matched_scores)
    coverage = float((2.0 * num_matched) / (n_q + n_e))
    score = base_score
    
    if coverage_penalty_exponent != 1.0:
        score = score * (max(coverage, 1e-6) ** (coverage_penalty_exponent - 1.0))
    if min_matches > 0 and num_matched < min_matches:
        score = score * ((num_matched / float(min_matches)) ** 2)
    if min_coverage_ratio > 0.0 and coverage < min_coverage_ratio:
        score = score * ((coverage / float(min_coverage_ratio)) ** 2)
        
    final_score = float(np.clip(score, 0.0, 1.0))
    details = {
        "num_matched": num_matched,
        "matched_pairs": matched_pairs,
        "coverage": coverage,
        "base_score": base_score,
        "tau": tau
    }
    return (final_score, details) if return_details else final_score


def match_protected_greedy(query_bits: np.ndarray, enrolled_bits: np.ndarray,
                           query_weights: np.ndarray, enrolled_weights: np.ndarray,
                           hamming_threshold: float = 0.22,
                           mode: str = "greedy_1to1",
                           coverage_penalty_exponent: float = 1.0,
                           min_matches: int = 0,
                           min_coverage_ratio: float = 0.0,
                           return_details: bool = False):
    """
    Computes greedy matching score between protected binary descriptor sets
    (no-Hungarian baseline).
    
    Modes:
        - "greedy_1to1": Greedily matches candidate pairs (r, c) with distance < hamming_threshold
          in ascending order of Hamming distance, enforcing 1-to-1 matching without global Hungarian optimization.
        - "greedy_nn": For each query descriptor r, finds nearest enrolled descriptor c.
          If dist < hamming_threshold, matches r to c (unconstrained greedy nearest-neighbor).

    Returns:
        float if return_details is False, else tuple (float, dict with num_matched, matched_pairs, coverage, base_score).
    """
    if mode not in ("greedy_1to1", "greedy_nn"):
        raise ValueError(f"Unknown greedy mode: '{mode}'. Must be 'greedy_1to1' or 'greedy_nn'")

    empty_details = {"num_matched": 0, "matched_pairs": [], "coverage": 0.0, "base_score": 0.0, "mode": mode, "tau": hamming_threshold}
    if query_bits.ndim == 1:
        query_bits = query_bits[None, :]
    if enrolled_bits.ndim == 1:
        enrolled_bits = enrolled_bits[None, :]

    n_q = len(query_bits)
    n_e = len(enrolled_bits)
    if n_q == 0 or n_e == 0:
        return (0.0, empty_details) if return_details else 0.0

    query_weights = np.atleast_1d(query_weights).astype(np.float32)
    enrolled_weights = np.atleast_1d(enrolled_weights).astype(np.float32)
    if len(query_weights) != n_q or len(enrolled_weights) != n_e:
        raise ValueError(f"Weight length mismatch: query {len(query_weights)} vs {n_q}, enrolled {len(enrolled_weights)} vs {n_e}")

    # Pairwise normalized Hamming distance matrix in [0, 1]
    dist_matrix = compute_pairwise_hamming_distance(query_bits, enrolled_bits)

    tau = hamming_threshold
    matched_scores = []
    matched_pairs = []

    if mode == "greedy_1to1":
        valid_pairs = np.argwhere(dist_matrix < tau)
        if len(valid_pairs) == 0:
            return (0.0, empty_details) if return_details else 0.0
        pair_dists = dist_matrix[valid_pairs[:, 0], valid_pairs[:, 1]]
        sort_order = np.argsort(pair_dists, kind="stable")
        
        matched_q = set()
        matched_e = set()
        for idx in sort_order:
            r, c = int(valid_pairs[idx, 0]), int(valid_pairs[idx, 1])
            if r not in matched_q and c not in matched_e:
                matched_q.add(r)
                matched_e.add(c)
                d = float(dist_matrix[r, c])
                sim = 1.0 - (d / tau)
                w = float(np.sqrt(query_weights[r] * enrolled_weights[c]))
                matched_scores.append(sim * w)
                matched_pairs.append((r, c))
    elif mode == "greedy_nn":
        for r in range(n_q):
            c = int(np.argmin(dist_matrix[r]))
            d = float(dist_matrix[r, c])
            if d < tau:
                sim = 1.0 - (d / tau)
                w = float(np.sqrt(query_weights[r] * enrolled_weights[c]))
                matched_scores.append(sim * w)
                matched_pairs.append((r, c))

    if not matched_scores:
        return (0.0, empty_details) if return_details else 0.0

    sum_q_weights = float(np.sum(query_weights))
    sum_e_weights = float(np.sum(enrolled_weights))
    coverage_denom = sum_q_weights + sum_e_weights + 1e-8

    base_score = float(np.clip(2.0 * np.sum(matched_scores) / coverage_denom, 0.0, 1.0))
    if base_score == 0.0:
        details = {"num_matched": len(matched_scores), "matched_pairs": matched_pairs, "coverage": 0.0, "base_score": 0.0, "mode": mode, "tau": tau}
        return (0.0, details) if return_details else 0.0

    num_matched = len(matched_scores)
    coverage = float((2.0 * num_matched) / (n_q + n_e))
    score = base_score

    if coverage_penalty_exponent != 1.0:
        score = score * (max(coverage, 1e-6) ** (coverage_penalty_exponent - 1.0))
    if min_matches > 0 and num_matched < min_matches:
        score = score * ((num_matched / float(min_matches)) ** 2)
    if min_coverage_ratio > 0.0 and coverage < min_coverage_ratio:
        score = score * ((coverage / float(min_coverage_ratio)) ** 2)

    final_score = float(np.clip(score, 0.0, 1.0))
    details = {
        "num_matched": num_matched,
        "matched_pairs": matched_pairs,
        "coverage": coverage,
        "base_score": base_score,
        "mode": mode,
        "tau": tau
    }
    return (final_score, details) if return_details else final_score

