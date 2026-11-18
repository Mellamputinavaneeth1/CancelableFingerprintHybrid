"""
Deformable SE(2) Spectral Graph Consensus & Privacy-Preserving Edge Matching.

Implements higher-order structural verification over local minutiae correspondences
using both:
1. Plaintext / continuous SE(2) edge compatibility (pairwise distance + relative ridge
   orientation + directed edge-to-minutia bearing angles) for unprotected baseline utility
   evaluation.
2. Keyed Random Fourier Feature (RFF) binary edge tokens (`match_protected`), enabling
   full spectral geometric consensus over encrypted/protected edge tokens WITHOUT storing
   or transmitting any plaintext `(x, y, theta)` minutia coordinates.
"""

import hashlib
import hmac
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.protection.kdf import derive_application_key
from src.protection.projection import crypto_gaussian_matrix


def _build_edge_feature_tensor(
    coords: np.ndarray,
    thetas: np.ndarray,
    sigma_dist: float,
    sigma_angle: float,
    sigma_edge_angle: float,
    dpi: float = 500.0,
) -> np.ndarray:
    """
    Constructs SE(2)-invariant 7-D pairwise relation vectors for all pairs (i, j):
        [d_ij / sigma_d,
         cos(2*dtheta) / (2*sigma_angle), sin(2*dtheta) / (2*sigma_angle),
         cos(2*alpha_i) / (2*sigma_edge), sin(2*alpha_i) / (2*sigma_edge),
         cos(2*alpha_j) / (2*sigma_edge), sin(2*alpha_j) / (2*sigma_edge)]
    where alpha_i = (phi_ij - theta_i) mod pi and phi_ij is the ridge-undirected
    bearing vector between minutia i and minutia j.
    """
    n = len(coords)
    if n == 0:
        return np.empty((0, 0, 7), dtype=np.float32)

    scale = dpi / 500.0
    sd_px = max(float(sigma_dist) * scale, 1e-4)
    sa2 = max(2.0 * float(sigma_angle), 1e-4)
    se2 = max(2.0 * float(sigma_edge_angle), 1e-4)

    coords_f = np.asarray(coords, dtype=np.float32)
    thetas_f = np.asarray(thetas, dtype=np.float32)

    dv = coords_f[None, :, :] - coords_f[:, None, :]  # (N, N, 2)
    dist = np.linalg.norm(dv, axis=2)
    dth = (thetas_f[None, :] - thetas_f[:, None]) % np.pi
    phi = np.arctan2(dv[:, :, 1], dv[:, :, 0])
    a_i = (phi - thetas_f[:, None]) % np.pi
    a_j = (phi - thetas_f[None, :]) % np.pi

    g = np.stack(
        [
            dist / sd_px,
            np.cos(2.0 * dth) / sa2,
            np.sin(2.0 * dth) / sa2,
            np.cos(2.0 * a_i) / se2,
            np.sin(2.0 * a_i) / se2,
            np.cos(2.0 * a_j) / se2,
            np.sin(2.0 * a_j) / se2,
        ],
        axis=-1,
    )
    return g.astype(np.float32, copy=False)


def derive_rff_edge_params(
    master_secret: Union[bytes, str],
    salt: Union[bytes, str],
    app_id: str,
    subject_id: str = "",
    edge_dim: int = 7,
    rff_bits: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derives application- and subject-bound Random Fourier Feature (RFF) projection
    parameters `(W, b)` from the KMS master secret via HKDF/HMAC domain separation.
    """
    if rff_bits <= 0 or rff_bits % 64 != 0:
        raise ValueError(f"rff_bits must be a positive multiple of 64, got {rff_bits}")

    key_bytes = derive_application_key(
        master_secret,
        salt,
        app_id,
        subject_id=subject_id,
        purpose=f"spectral_edge_rff_v1_{rff_bits}",
        version="1.0",
        length=32,
    )
    W = crypto_gaussian_matrix(key_bytes, rff_bits, edge_dim).astype(np.float32)

    # Derive uniform random phase shifts b ~ U[0, 2*pi) via HMAC-SHA256 CTR
    b_vals = []
    ctr = 0
    while len(b_vals) < rff_bits:
        blk = hmac.new(
            key_bytes,
            b"RFF_PHASE|v1.0|" + ctr.to_bytes(4, "big"),
            hashlib.sha256,
        ).digest()
        for off in range(0, len(blk), 4):
            v = int.from_bytes(blk[off : off + 4], "big")
            b_vals.append((2.0 * np.pi) * ((v + 0.5) / 4294967296.0))
            if len(b_vals) == rff_bits:
                break
        ctr += 1
    b = np.asarray(b_vals, dtype=np.float32)
    return W, b


def encode_edge_rff(
    coords: np.ndarray,
    thetas: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    sigma_dist: float = 18.0,
    sigma_angle: float = 0.38,
    sigma_edge_angle: float = 0.34,
    dpi: float = 500.0,
) -> np.ndarray:
    """
    Encodes all N x N pairwise minutia relations into keyed 1-bit RFF tokens
    packed into uint64 words `(N, N, rff_bits // 64)`.

    Contains ZERO plaintext coordinates or orientations; distinct salts/apps yield
    orthogonal statistically independent bit patterns.
    """
    rff_bits = int(W.shape[0])
    w64 = rff_bits // 64
    n = len(coords)
    if n == 0:
        return np.empty((0, 0, w64), dtype=np.uint64)

    g = _build_edge_feature_tensor(
        coords,
        thetas,
        sigma_dist=sigma_dist,
        sigma_angle=sigma_angle,
        sigma_edge_angle=sigma_edge_angle,
        dpi=dpi,
    )
    proj = np.einsum("ijd,kd->ijk", g, W) + b[None, None, :]
    bits = (np.cos(proj) >= 0.0).astype(np.uint8)
    packed = np.packbits(bits, axis=-1).view(np.uint64)
    return np.ascontiguousarray(packed)


def build_sparse_edge_mask(
    coords: np.ndarray,
    top_k: int = 20,
    mode: str = "topk",
) -> np.ndarray:
    """
    Constructs an (N, N) boolean mask selecting a compact edge subset per template.
    Supported modes:
    - "topk": directed top-k spatial nearest neighbors per node (excluding self-loops).
    - "upper": strict upper triangle (i < j), cutting pairwise edge storage in half.
    """
    n = len(coords)
    if n <= 1:
        return np.zeros((n, n), dtype=bool)
    if mode == "upper":
        return np.triu(np.ones((n, n), dtype=bool), k=1)
    if mode != "topk":
        raise ValueError(f"Unsupported sparse edge mode '{mode}': expected 'topk' or 'upper'")

    coords_f = np.asarray(coords, dtype=np.float64)
    diff = coords_f[:, None, :] - coords_f[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    k_eff = min(max(1, int(top_k)), n - 1)
    nn_idx = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
    mask = np.zeros((n, n), dtype=bool)
    mask[np.arange(n)[:, None], nn_idx] = True
    np.fill_diagonal(mask, False)
    return mask


def encode_sparse_edge_rff(
    coords: np.ndarray,
    thetas: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    top_k: int = 20,
    mode: str = "topk",
    sigma_dist: float = 18.0,
    sigma_angle: float = 0.38,
    sigma_edge_angle: float = 0.34,
    dpi: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encodes keyed RFF edge tokens restricted to a sparse/compact boolean edge mask.

    Returns:
        compact_tokens: (K, w64) uint64 active tokens in row-major order over `edge_mask`.
        edge_mask: (N, N) bool array marking active edges.
    """
    dense = encode_edge_rff(
        coords=coords,
        thetas=thetas,
        W=W,
        b=b,
        sigma_dist=sigma_dist,
        sigma_angle=sigma_angle,
        sigma_edge_angle=sigma_edge_angle,
        dpi=dpi,
    )
    mask = build_sparse_edge_mask(coords=coords, top_k=top_k, mode=mode)
    return np.ascontiguousarray(dense[mask], dtype=np.uint64), mask


def decode_sparse_edge_rff(
    compact_tokens: np.ndarray,
    edge_mask: np.ndarray,
) -> np.ndarray:
    """
    Expands compact (K, w64) sparse RFF edge tokens back to (N, N, w64) uint64,
    zero-filling omitted edges (which remain excluded via `edge_mask` during matching).
    """
    mask = np.asarray(edge_mask, dtype=bool)
    n = mask.shape[0]
    if mask.ndim != 2 or mask.shape[1] != n:
        raise ValueError(f"edge_mask must be square (N, N), got {mask.shape}")
    toks = np.asarray(compact_tokens, dtype=np.uint64)
    w64 = int(toks.shape[1]) if toks.ndim == 2 else 0
    k_expected = int(mask.sum())
    if len(toks) != k_expected:
        raise ValueError(f"Expected {k_expected} active edge tokens for mask, got {len(toks)}")
    out = np.zeros((n, n, w64), dtype=np.uint64)
    if k_expected > 0:
        out[mask] = toks
    return out


def edge_rff_compatibility(
    query_edge_tokens: np.ndarray,
    enrolled_edge_tokens: np.ndarray,
    cand_u: np.ndarray,
    cand_v: np.ndarray,
    gamma: float = 1.25,
    query_edge_mask: Optional[np.ndarray] = None,
    enrolled_edge_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes vectorized M x M pairwise structural compatibility from keyed RFF
    edge tokens via native uint64 Hamming distance (np.bitwise_count).
    When sparse edge masks are supplied, omitted edges contribute 0.0 compatibility.
    """
    w64 = query_edge_tokens.shape[-1]
    rff_bits = float(w64 * 64)
    q_sub = query_edge_tokens[np.ix_(cand_u, cand_u)]
    e_sub = enrolled_edge_tokens[np.ix_(cand_v, cand_v)]
    xor_u64 = np.bitwise_xor(q_sub, e_sub)
    ham = np.bitwise_count(xor_u64).sum(axis=-1).astype(np.float32) / rff_bits
    compat = np.clip(1.0 - 2.0 * ham, 0.0, 1.0) ** gamma
    if query_edge_mask is not None:
        compat *= query_edge_mask[np.ix_(cand_u, cand_u)].astype(np.float32)
    if enrolled_edge_mask is not None:
        compat *= enrolled_edge_mask[np.ix_(cand_v, cand_v)].astype(np.float32)
    return compat


def compute_se2_edge_compatibility(
    query_coords: np.ndarray,
    query_thetas: np.ndarray,
    enrolled_coords: np.ndarray,
    enrolled_thetas: np.ndarray,
    cand_u: np.ndarray,
    cand_v: np.ndarray,
    sigma_dist: float = 18.0,
    sigma_angle: float = 0.38,
    sigma_edge_angle: float = 0.34,
    dpi_query: float = 500.0,
    dpi_enrolled: float = 500.0,
) -> np.ndarray:
    """
    Computes continuous SE(2) higher-order geometric compatibility across M
    candidate correspondences `(cand_u[k], cand_v[k])`, incorporating pairwise
    distance residual, relative orientation residual, and both endpoint
    edge-to-minutia bearing residuals.
    """
    qc = np.asarray(query_coords, dtype=np.float32) / (dpi_query / 500.0)
    ec = np.asarray(enrolled_coords, dtype=np.float32) / (dpi_enrolled / 500.0)
    qt = np.asarray(query_thetas, dtype=np.float32)
    et = np.asarray(enrolled_thetas, dtype=np.float32)

    q_cu = qc[cand_u]
    e_cv = ec[cand_v]
    q_tu = qt[cand_u]
    e_tv = et[cand_v]

    q_dv = q_cu[None, :, :] - q_cu[:, None, :]
    e_dv = e_cv[None, :, :] - e_cv[:, None, :]

    q_dist = np.linalg.norm(q_dv, axis=2)
    e_dist = np.linalg.norm(e_dv, axis=2)

    q_phi = np.arctan2(q_dv[:, :, 1], q_dv[:, :, 0])
    e_phi = np.arctan2(e_dv[:, :, 1], e_dv[:, :, 0])

    q_dth = (q_tu[None, :] - q_tu[:, None]) % np.pi
    e_dth = (e_tv[None, :] - e_tv[:, None]) % np.pi
    th_diff = np.abs(q_dth - e_dth)
    th_err = np.minimum(th_diff, np.pi - th_diff)

    q_ai = (q_phi - q_tu[:, None]) % np.pi
    e_ai = (e_phi - e_tv[:, None]) % np.pi
    ai_diff = np.abs(q_ai - e_ai)
    ai_err = np.minimum(ai_diff, np.pi - ai_diff)

    q_aj = (q_phi - q_tu[None, :]) % np.pi
    e_aj = (e_phi - e_tv[None, :]) % np.pi
    aj_diff = np.abs(q_aj - e_aj)
    aj_err = np.minimum(aj_diff, np.pi - aj_diff)

    d_err = np.abs(q_dist - e_dist)
    e_err = 0.5 * (ai_err + aj_err)

    return np.exp(
        -0.5
        * (
            (d_err / max(sigma_dist, 1e-4)) ** 2
            + (th_err / max(sigma_angle, 1e-4)) ** 2
            + (e_err / max(sigma_edge_angle, 1e-4)) ** 2
        )
    ).astype(np.float32)


class SpectralGraphMatcher:
    """
    Deformable SE(2) Spectral Graph Consensus Matcher.

    Finds the maximally self-consistent 1-to-1 minutiae correspondence subgraph via
    principal eigenvector power iteration over an affinity matrix M whose
    off-diagonal terms enforce SE(2) second-order geometric consistency (either
    from plaintext geometry or zero-plaintext keyed RFF edge tokens).
    """

    def __init__(
        self,
        sigma_dist: float = 18.0,
        sigma_angle: float = 0.38,
        power_iterations: int = 18,
        sigma_edge_angle: float = 0.34,
        top_k: int = 6,
        use_lss: bool = True,
        min_edge_compat: float = 0.15,
        rff_gamma: float = 1.00,
        max_candidates: int = 400,
        min_eigenvector_weight: float = 0.01,
        clique_prune_thresh: float = 0.16,
    ):
        self.sigma_dist = sigma_dist
        self.sigma_angle = sigma_angle
        self.power_iterations = power_iterations
        self.sigma_edge_angle = sigma_edge_angle
        self.top_k = top_k
        self.use_lss = use_lss
        self.min_edge_compat = min_edge_compat
        self.rff_gamma = rff_gamma
        self.max_candidates = max_candidates
        self.min_eigenvector_weight = min_eigenvector_weight
        self.clique_prune_thresh = clique_prune_thresh

    def _run_spectral_consensus(
        self,
        pairwise_dists: np.ndarray,
        tau: float,
        query_weights: Optional[np.ndarray],
        enrolled_weights: Optional[np.ndarray],
        compat_fn,
    ) -> Tuple[float, Dict[str, Any]]:
        n_q, n_e = pairwise_dists.shape
        if n_q < 3 or n_e < 3:
            return 0.0, {
                "num_matched": 0,
                "num_consensus_matches": 0,
                "lambda_max": 0.0,
                "num_candidates": 0,
                "selected_pairs": [],
            }

        # Bidirectional mutual top-K candidate selection under Hamming/distance cutoff tau
        k_row = min(self.top_k, n_e)
        idx_row = np.argpartition(pairwise_dists, k_row - 1, axis=1)[:, :k_row]
        cand_mask = np.zeros_like(pairwise_dists, dtype=bool)
        cand_mask[np.arange(n_q)[:, None], idx_row] = True

        if n_q >= self.top_k:
            k_col = min(self.top_k, n_q)
            idx_col = np.argpartition(pairwise_dists, k_col - 1, axis=0)[:k_col, :]
            col_mask = np.zeros_like(pairwise_dists, dtype=bool)
            col_mask[idx_col, np.arange(n_e)[None, :]] = True
            cand_mask |= col_mask

        cand_u, cand_v = np.where(cand_mask & (pairwise_dists < tau))
        if len(cand_u) < 2:
            return 0.0, {
                "num_matched": 0,
                "num_consensus_matches": 0,
                "lambda_max": 0.0,
                "num_candidates": int(len(cand_u)),
                "selected_pairs": [],
            }

        sim_raw = np.clip(1.0 - pairwise_dists / max(tau, 1e-6), 0.0, 1.0)
        sims = sim_raw[cand_u, cand_v].astype(np.float32)

        if self.use_lss:
            k_bg_row = min(n_e, 6)
            k_bg_col = min(n_q, 6)
            row_bg = np.mean(
                np.partition(pairwise_dists, k_bg_row - 1, axis=1)[:, :k_bg_row],
                axis=1,
            )
            col_bg = np.mean(
                np.partition(pairwise_dists, k_bg_col - 1, axis=0)[:k_bg_col, :],
                axis=0,
            )
            bg = 0.5 * (row_bg[cand_u] + col_bg[cand_v])
            distinct = np.clip((bg - pairwise_dists[cand_u, cand_v]) / 0.12, 0.25, 1.4).astype(np.float32)
            sims = sims * distinct

        if query_weights is not None and enrolled_weights is not None:
            qw = np.clip(np.asarray(query_weights, dtype=np.float32), 0.15, 1.0)
            ew = np.clip(np.asarray(enrolled_weights, dtype=np.float32), 0.15, 1.0)
            sims = sims * np.sqrt(qw[cand_u] * ew[cand_v])

        if len(cand_u) > self.max_candidates:
            best_idx = np.argsort(-sims, kind="mergesort")[:self.max_candidates]
            cand_u = cand_u[best_idx]
            cand_v = cand_v[best_idx]
            sims = sims[best_idx]

        m = len(cand_u)
        pcompat = compat_fn(cand_u, cand_v)
        compat = pcompat * (0.3 + 0.7 * np.sqrt(np.clip(sims[:, None], 0.0, 1.5) * np.clip(sims[None, :], 0.0, 1.5)))

        valid_pairs = (cand_u[:, None] != cand_u[None, :]) & (cand_v[:, None] != cand_v[None, :])
        M = np.zeros((m, m), dtype=np.float32)
        M[valid_pairs] = compat[valid_pairs]
        np.fill_diagonal(M, sims * 0.5)

        # Principal eigenvector via power iteration
        x = np.full((m,), 1.0 / np.sqrt(m), dtype=np.float32)
        lambda_max = 0.0
        for _ in range(self.power_iterations):
            x_next = M @ x
            nrm = float(np.linalg.norm(x_next))
            if nrm < 1e-8:
                break
            x = x_next / nrm
            lambda_max = nrm

        # Greedy 1-to-1 consensus subgraph extraction with mean edge-support gate
        order = np.argsort(-x)
        used_u, used_v = set(), set()
        selected: List[int] = []
        selected_pairs: List[Tuple[int, int]] = []
        for idx in order:
            if x[idx] < self.min_eigenvector_weight:
                break
            u = int(cand_u[idx])
            v = int(cand_v[idx])
            if u in used_u or v in used_v:
                continue
            if selected and float(np.mean(M[idx, selected])) < self.min_edge_compat:
                continue
            used_u.add(u)
            used_v.add(v)
            selected.append(int(idx))
            selected_pairs.append((u, v))

        # Iterative weakest-node clique outlier pruning
        while len(selected) >= 4 and self.clique_prune_thresh > 0.0:
            sub = M[np.ix_(selected, selected)]
            row_avg = (np.sum(sub, axis=1) - np.diag(sub)) / float(len(selected) - 1)
            worst_pos = int(np.argmin(row_avg))
            if float(row_avg[worst_pos]) >= self.clique_prune_thresh:
                break
            selected.pop(worst_pos)
            selected_pairs.pop(worst_pos)

        if len(selected) < 3:
            score = float(0.01 * len(selected) / max(1, min(n_q, n_e)))
            return score, {
                "num_matched": len(selected),
                "num_consensus_matches": len(selected),
                "lambda_max": float(lambda_max),
                "num_candidates": int(m),
                "selected_pairs": selected_pairs,
            }

        sub = M[np.ix_(selected, selected)]
        edge_sum = float((np.sum(sub) - np.trace(sub)) / max(1, len(selected) - 1))
        unary_sum = float(np.sum(sims[selected]))
        denom = 0.6 * np.sqrt(float(n_q) * float(n_e)) + 0.4 * max(12.0, float(min(n_q, n_e)))
        score = float(np.clip((0.7 * edge_sum + 0.3 * unary_sum) / max(1.0, denom), 0.0, 1.0))

        return score, {
            "num_matched": len(selected),
            "num_consensus_matches": len(selected),
            "lambda_max": float(lambda_max),
            "num_candidates": int(m),
            "edge_consistency": float(edge_sum / max(1, len(selected))),
            "selected_pairs": selected_pairs,
        }

    def match(
        self,
        query_coords: Optional[np.ndarray],
        query_thetas: Optional[np.ndarray],
        enrolled_coords: Optional[np.ndarray],
        enrolled_thetas: Optional[np.ndarray],
        pairwise_dists: np.ndarray,
        tau: float,
        query_weights: Optional[np.ndarray] = None,
        enrolled_weights: Optional[np.ndarray] = None,
        query_edge_tokens: Optional[np.ndarray] = None,
        enrolled_edge_tokens: Optional[np.ndarray] = None,
        query_edge_mask: Optional[np.ndarray] = None,
        enrolled_edge_mask: Optional[np.ndarray] = None,
        dpi_query: float = 500.0,
        dpi_enrolled: float = 500.0,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Executes deformable SE(2) spectral consensus matching.
        When `query_edge_tokens` and `enrolled_edge_tokens` are supplied, operates
        strictly in protected RFF edge space (no plaintext geometry needed).
        """
        if query_edge_tokens is not None and enrolled_edge_tokens is not None:
            return self.match_protected(
                query_edge_tokens=query_edge_tokens,
                enrolled_edge_tokens=enrolled_edge_tokens,
                pairwise_dists=pairwise_dists,
                tau=tau,
                query_weights=query_weights,
                enrolled_weights=enrolled_weights,
                query_edge_mask=query_edge_mask,
                enrolled_edge_mask=enrolled_edge_mask,
            )

        if query_coords is None or query_thetas is None or enrolled_coords is None or enrolled_thetas is None:
            raise ValueError("Plaintext coords/thetas or protected RFF edge tokens must be provided.")

        def _compat(cu: np.ndarray, cv: np.ndarray) -> np.ndarray:
            return compute_se2_edge_compatibility(
                query_coords=query_coords,
                query_thetas=query_thetas,
                enrolled_coords=enrolled_coords,
                enrolled_thetas=enrolled_thetas,
                cand_u=cu,
                cand_v=cv,
                sigma_dist=self.sigma_dist,
                sigma_angle=self.sigma_angle,
                sigma_edge_angle=self.sigma_edge_angle,
                dpi_query=dpi_query,
                dpi_enrolled=dpi_enrolled,
            )

        return self._run_spectral_consensus(pairwise_dists, tau, query_weights, enrolled_weights, _compat)

    def match_protected(
        self,
        query_edge_tokens: np.ndarray,
        enrolled_edge_tokens: np.ndarray,
        pairwise_dists: np.ndarray,
        tau: float,
        query_weights: Optional[np.ndarray] = None,
        enrolled_weights: Optional[np.ndarray] = None,
        query_edge_mask: Optional[np.ndarray] = None,
        enrolled_edge_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Zero-plaintext privacy-preserving spectral graph consensus matching using
        keyed RFF binary edge tokens and unary MCC descriptor distances.
        """
        def _compat(cu: np.ndarray, cv: np.ndarray) -> np.ndarray:
            return edge_rff_compatibility(
                query_edge_tokens=query_edge_tokens,
                enrolled_edge_tokens=enrolled_edge_tokens,
                cand_u=cu,
                cand_v=cv,
                gamma=self.rff_gamma,
                query_edge_mask=query_edge_mask,
                enrolled_edge_mask=enrolled_edge_mask,
            )

        return self._run_spectral_consensus(pairwise_dists, tau, query_weights, enrolled_weights, _compat)
