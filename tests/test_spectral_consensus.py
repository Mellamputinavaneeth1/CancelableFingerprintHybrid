import numpy as np
import pytest
from src.matching.spectral_consensus import SpectralGraphMatcher


def test_spectral_consensus_genuine_vs_impostor_separation():
    matcher = SpectralGraphMatcher(sigma_dist=15.0, sigma_angle=0.436, power_iterations=20)

    np.random.seed(42)
    # 25 genuine minutiae points
    n_pts = 25
    coords_q = np.random.uniform(50, 250, (n_pts, 2)).astype(np.float32)
    thetas_q = np.random.uniform(0, np.pi, n_pts).astype(np.float32)

    # Genuine enrolled: rotated by 30 deg + translated + small elastic shear (std = 2.0 px)
    alpha = np.radians(30.0)
    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha),  np.cos(alpha)]], dtype=np.float32)
    trans = np.array([40.0, -25.0], dtype=np.float32)
    noise = np.random.normal(0.0, 2.0, (n_pts, 2)).astype(np.float32)

    coords_e = (coords_q @ R.T) + trans + noise
    thetas_e = (thetas_q + alpha + np.random.normal(0.0, 0.05, n_pts)) % np.pi

    # Genuine distance matrix: diagonal has low distance (e.g. 0.08), off-diagonal has high distance (0.45)
    gen_dist_mat = np.full((n_pts, n_pts), 0.45, dtype=np.float32)
    np.fill_diagonal(gen_dist_mat, 0.08)

    gen_score, gen_details = matcher.match(
        coords_q, thetas_q, coords_e, thetas_e,
        pairwise_dists=gen_dist_mat, tau=0.22
    )

    # Impostor: completely random unrelated minutiae
    coords_imp = np.random.uniform(50, 250, (n_pts, 2)).astype(np.float32)
    thetas_imp = np.random.uniform(0, np.pi, n_pts).astype(np.float32)

    # Impostor distance matrix: random candidates with distance < tau scattered randomly
    imp_dist_mat = np.full((n_pts, n_pts), 0.45, dtype=np.float32)
    # Scatter 10 random false matches
    rand_pairs = np.random.choice(n_pts, (10, 2), replace=False)
    for u, v in rand_pairs:
        imp_dist_mat[u, v] = 0.12

    imp_score, imp_details = matcher.match(
        coords_q, thetas_q, coords_imp, thetas_imp,
        pairwise_dists=imp_dist_mat, tau=0.22
    )

    print(f"\n[SPECTRAL CONSENSUS TEST]")
    print(f"Genuine score:  {gen_score:.4f} (lambda_max: {gen_details['lambda_max']:.2f}, consensus: {gen_details['num_consensus_matches']})")
    print(f"Impostor score: {imp_score:.4f} (lambda_max: {imp_details['lambda_max']:.2f}, consensus: {imp_details['num_consensus_matches']})")

    assert gen_score > 0.65, f"Expected genuine score > 0.65, got {gen_score}"
    assert imp_score < 0.12, f"Expected impostor score < 0.12, got {imp_score}"
    margin = gen_score - imp_score
    assert margin > 0.55, f"Expected margin > 0.55, got {margin}"
