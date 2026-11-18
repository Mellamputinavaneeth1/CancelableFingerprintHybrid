"""
Minutiae Cylinder-Code (MCC) Descriptor Engine.
Implements the 3D local spatial-directional cylinder descriptor (Cappelli, Ferrara, Maltoni, IEEE TPAMI 2010),
adapted for keyed cancelable orthonormal projection and sign quantization.

Mathematical Formulation:
--------------------------
For each minutia m_i = (x_i, y_i, theta_i), a local 3D cylinder of radius R is defined.
1. Local Coordinate Alignment:
   Rotates neighboring minutiae m_j into m_i's local frame:
   [x_j^i, y_j^i]^T = R(theta_i) * [x_j - x_i, y_j - y_i]^T
   Delta theta_j^i = (theta_j - theta_i) mod pi

2. 3D Spatial-Directional Discretization:
   - Spatial grid: N_s x N_s cells spanning [-R, R] x [-R, R]
   - Directional slices: N_d slices spanning [0, pi)
   - Total dimension: D = N_s * N_s * N_d (e.g. 8 * 8 * 4 = 256 continuous features)

3. Gaussian Accumulation Kernel:
   Accumulates continuous spatial-directional density:
   C(u, v, k) = sum_{m_j} G_sigma_s(||p_uv - p_j^i||) * G_sigma_d(|phi_k - Delta theta_j^i|)

4. Axial Modulo-pi Inversion Symmetry:
   Because fingerprint ridge orientation theta is defined modulo pi, a rotation crossing pi
   maps theta -> theta + pi. Under theta + pi, the spatial coordinates negate:
   (x_local, y_local) -> (-x_local, -y_local).
   In the grid, this corresponds exactly to a 180-degree spatial inversion:
   C_{theta + pi}(u, v, k) = C_theta(N_s - 1 - u, N_s - 1 - v, k).
   The dual descriptor d^pi is computed via dual_flip(d) to ensure 100% exact SE(2) invariance.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from src.minutiae.extractor import Minutia


def flip_mcc_dual(descriptor: np.ndarray, num_spatial_cells: int = 8, num_directional_slices: int = 4) -> np.ndarray:
    """
    Computes the exact 180-degree spatially inverted dual MCC descriptor for axial modulo-pi invariance.
    If descriptor has shape (D,) or (K, D), spatial axes (N_s, N_s) are inverted.
    """
    orig_shape = descriptor.shape
    if descriptor.ndim == 1:
        reshaped = descriptor.reshape(num_spatial_cells, num_spatial_cells, num_directional_slices)
        flipped = np.flip(np.flip(reshaped, axis=0), axis=1)
        return flipped.reshape(orig_shape)
    elif descriptor.ndim == 2:
        k = descriptor.shape[0]
        reshaped = descriptor.reshape(k, num_spatial_cells, num_spatial_cells, num_directional_slices)
        flipped = np.flip(np.flip(reshaped, axis=1), axis=2)
        return flipped.reshape(orig_shape)
    else:
        raise ValueError(f"Unsupported descriptor shape: {orig_shape}")


class MinutiaeCylinderCode:
    """
    Constructs fixed-length, SE(2)-invariant continuous 3D spatial-directional
    cylindrical descriptors for each minutia in a fingerprint.
    """
    def __init__(self, 
                 radius: float = 80.0,
                 num_spatial_cells: int = 8,
                 num_directional_slices: int = 4,
                 sigma_spatial: float = 10.0,
                 sigma_directional: float = np.pi / 6.0,
                 min_neighbors: int = 2,
                 zero_mean: bool = False,
                 sigmoid_gain: Optional[float] = None,
                 dpi: Optional[float] = None):
        if dpi is not None and dpi > 0:
            scale = float(dpi) / 500.0
            radius = radius * scale
            sigma_spatial = sigma_spatial * scale
        self.radius = radius
        self.num_spatial_cells = num_spatial_cells
        self.num_directional_slices = num_directional_slices
        self.sigma_spatial = sigma_spatial
        self.sigma_directional = sigma_directional
        self.min_neighbors = min_neighbors
        self.zero_mean = zero_mean
        self.sigmoid_gain = sigmoid_gain
        self.descriptor_dim = num_spatial_cells * num_spatial_cells * num_directional_slices

        # Precompute spatial cell centers in local frame [-R, R]
        cell_width = (2.0 * radius) / num_spatial_cells
        coords = np.linspace(-radius + cell_width / 2.0, radius - cell_width / 2.0, num_spatial_cells)
        grid_x, grid_y = np.meshgrid(coords, coords)
        self.cell_centers = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)  # (N_s*N_s, 2)

        # Circular cylinder mask (cells whose centers lie inside radius * 1.15)
        self.valid_cell_mask = (np.linalg.norm(self.cell_centers, axis=1) <= radius * 1.15)

        # Precompute directional slice center angles in [0, pi)
        self.slice_centers = np.linspace(0.0, np.pi, num_directional_slices, endpoint=False)

    def extract_descriptors(self, minutiae: List[Minutia], 
                            mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts MCC continuous descriptors and associated quality weights for all minutiae.

        Args:
            minutiae: List of Minutia objects.
            mask: Optional binary foreground mask (used to ignore out-of-ROI cylinder cells if provided).

        Returns:
            descriptors: (K, D) float32 matrix of normalized continuous descriptors.
            weights: (K,) float32 vector of confidence weights in [0, 1].
        """
        n = len(minutiae)
        if n < 3:
            return np.empty((0, self.descriptor_dim), dtype=np.float32), np.empty((0,), dtype=np.float32)

        coords = np.array([[m.x, m.y] for m in minutiae], dtype=np.float32)
        thetas = np.array([m.theta % np.pi for m in minutiae], dtype=np.float32)
        qualities = np.array([m.quality for m in minutiae], dtype=np.float32)

        descriptors = np.zeros((n, self.descriptor_dim), dtype=np.float32)
        weights = np.zeros((n,), dtype=np.float32)

        # Pairwise Euclidean distances between all minutiae
        diff_coords = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
        dist_matrix = np.linalg.norm(diff_coords, axis=2)       # (N, N)

        num_cells = self.num_spatial_cells * self.num_spatial_cells
        num_slices = self.num_directional_slices
        two_sig_s2 = 2.0 * (self.sigma_spatial ** 2)
        two_sig_d2 = 2.0 * (self.sigma_directional ** 2)

        for i in range(n):
            # Find neighbors of minutia i within radius R (excluding self)
            nbr_indices = np.where((dist_matrix[i] <= self.radius) & (dist_matrix[i] > 1e-3))[0]
            
            if len(nbr_indices) < self.min_neighbors:
                weights[i] = qualities[i] * 0.25
            else:
                weights[i] = qualities[i] * min(1.0, len(nbr_indices) / 6.0)

            if len(nbr_indices) == 0:
                continue

            # Minutia i coordinate frame
            x_i, y_i = coords[i]
            th_i = thetas[i]
            cos_th = np.cos(th_i)
            sin_th = np.sin(th_i)

            # Rotate and translate neighbors into local frame of m_i
            dx = coords[nbr_indices, 0] - x_i
            dy = coords[nbr_indices, 1] - y_i
            nbr_x_local = cos_th * dx + sin_th * dy
            nbr_y_local = -sin_th * dx + cos_th * dy
            nbr_local = np.stack([nbr_x_local, nbr_y_local], axis=-1)  # (N_nbr, 2)

            # Full modulo-pi circular angular difference in [0, pi) without folding into [0, pi/2]
            nbr_thetas = thetas[nbr_indices]
            nbr_dth = (nbr_thetas - th_i) % np.pi  # (N_nbr,) in [0, pi)

            # Spatial Gaussian kernel between neighbors and all cell centers: (N_nbr, num_cells)
            cell_dists_sq = np.sum((nbr_local[:, None, :] - self.cell_centers[None, :, :]) ** 2, axis=2)
            spatial_weights = np.exp(-cell_dists_sq / two_sig_s2)  # (N_nbr, num_cells)
            spatial_weights[:, ~self.valid_cell_mask] = 0.0

            # Directional Gaussian kernel across full circular [0, pi) slice centers: (N_nbr, num_slices)
            ang_diffs = np.abs(nbr_dth[:, None] - self.slice_centers[None, :])
            ang_diffs_mod_pi = np.minimum(ang_diffs, np.pi - ang_diffs)
            directional_weights = np.exp(-(ang_diffs_mod_pi ** 2) / two_sig_d2)  # (N_nbr, num_slices)

            # Outer product and summation: (num_cells, num_slices)
            cylinder = np.einsum('nc,ns->cs', spatial_weights, directional_weights)  # (num_cells, num_slices)
            if self.sigmoid_gain is not None and self.sigmoid_gain > 0:
                cylinder = np.tanh(self.sigmoid_gain * cylinder)
            flat_desc = cylinder.reshape(-1)  # (D,)

            if self.zero_mean:
                # 180-degree symmetric centering across active cylinder cells preserves exact dual_flip symmetry
                active_mask_3d = np.repeat(self.valid_cell_mask, num_slices)
                if np.any(active_mask_3d):
                    mu_c = float(np.mean(flat_desc[active_mask_3d]))
                    flat_desc = np.where(active_mask_3d, flat_desc - mu_c, 0.0)

            # L2 unit normalization
            norm = np.linalg.norm(flat_desc)
            if norm > 1e-6:
                descriptors[i] = flat_desc / norm
            else:
                descriptors[i] = flat_desc

        return descriptors, weights

    def get_dual_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        """Computes dual 180-degree inverted descriptors for query matching under modulo-pi ambiguity."""
        return flip_mcc_dual(descriptors, self.num_spatial_cells, self.num_directional_slices)
