"""
Augmented bipartite slack cost matrix for Kuhn-Munkres Hungarian matching.
Standardizes dummy cost to tau / 2.0 and strict distance inequality D_ij < tau.
"""

from typing import Tuple
import numpy as np


def build_augmented_cost(dist_matrix: np.ndarray, tau: float, big: float = 1e6) -> np.ndarray:
    """
    Constructs the augmented cost matrix for Kuhn-Munkres Hungarian bipartite matching.
    
    Augmented Matrix Structure of size (N_q + N_e) x (N_q + N_e):
        ┌─────────────┬─────────────┐
        │  D_valid    │  U_q        │  ← N_q rows
        ├─────────────┼─────────────┤
        │  U_e        │  0          │  ← N_e rows
        └─────────────┴─────────────┘
        
    Where:
        D_valid[i, j] = dist[i, j] if dist[i, j] < tau else big (forbidden)
        U_q[i, j] = tau / 2.0 if i == j else big (forbidden off-diagonal slack)
        U_e[i, j] = tau / 2.0 if i == j else big (forbidden off-diagonal slack)
        Bottom-right (N_e x N_q) = 0.0 (dummy-to-dummy pairing)
        
    Objective Alignment:
        Matching k real pairs with distances D_ij < tau leaves (N_q - k) query nodes
        and (N_e - k) enrolled nodes unmatched:
        J(pi) = sum_{matched} D_ij + (N_q - k)*(tau/2) + (N_e - k)*(tau/2)
              = (N_q + N_e) * (tau / 2) - sum_{matched} (tau - D_ij)
              
    Boundary Semantics:
        A real match is eligible if and only if D_ij < tau (strict inequality).
        At D_ij == tau, matching cost D_ij equals the cost of leaving both nodes
        unmatched (tau/2 + tau/2 = tau), yielding zero objective reduction. Strict
        inequality prevents degenerate non-improving assignments.
        
    Args:
        dist_matrix: (N_q, N_e) pairwise distance matrix.
        tau: Distance threshold.
        big: Large forbidden cost (default 1e6).
        
    Returns:
        Augmented cost matrix of shape (N_q + N_e, N_q + N_e) as float64.
    """
    if dist_matrix.ndim != 2:
        raise ValueError(f"dist_matrix must be 2-dimensional, got {dist_matrix.ndim}D")
    nq, ne = dist_matrix.shape
    if nq == 0 and ne == 0:
        return np.zeros((0, 0), dtype=np.float64)
        
    real = np.where(dist_matrix < tau, dist_matrix, big).astype(np.float64)
    
    uq = np.full((nq, nq), big, dtype=np.float64)
    np.fill_diagonal(uq, tau / 2.0)
    
    ue = np.full((ne, ne), big, dtype=np.float64)
    np.fill_diagonal(ue, tau / 2.0)
    
    dd = np.zeros((ne, nq), dtype=np.float64)
    return np.block([[real, uq], [ue, dd]])
