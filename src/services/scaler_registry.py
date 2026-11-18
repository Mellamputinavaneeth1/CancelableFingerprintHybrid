"""
Scaler Registry Service.

Manages feature normalization, standardization, and scaling strategies
for continuous and cancelable biometric descriptors.
"""

import os
from typing import Dict, Any, Optional, Callable
import numpy as np


# Pre-calibrated empirical reference statistics for 9-D geometric minutiae descriptors
CALIBRATED_EMPIRICAL_MEAN_9D = np.array([
    0.8200, 0.5460, 4.0000, 0.5000, 0.4800, 0.4800, 0.0600, 0.1330, 0.1210
], dtype=np.float32)

CALIBRATED_EMPIRICAL_STD_9D = np.array([
    0.1200, 0.1870, 2.3000, 0.2900, 0.2950, 0.2850, 0.0880, 0.1650, 0.1550
], dtype=np.float32)

# Pooled empirical reference statistics across FVC2002 Set B partitions (DB1_B, DB2_B, DB3_B, DB4_B; N=40 subjects)
POOLED_SET_B_EMPIRICAL_MEAN_9D = np.array([
    0.8142, 0.5418, 4.0850, 0.4982, 0.4851, 0.4812, 0.0615, 0.1342, 0.1225
], dtype=np.float32)

POOLED_SET_B_EMPIRICAL_STD_9D = np.array([
    0.1215, 0.1842, 2.3120, 0.2885, 0.2921, 0.2864, 0.0865, 0.1638, 0.1541
], dtype=np.float32)


class ScalerRegistry:
    """
    Registry for descriptor transformation and scaling pipelines.
    Supports Identity (raw geometric angles/ratios), Z-Score standardization,
    and sensor-specific calibrated statistics.
    """
    def __init__(self):
        self._scalers: Dict[str, Dict[str, Any]] = {}
        # Register standard defaults
        self.register_identity("identity")
        self.register_identity("identity_9d")
        for name in ["DB1_A", "DB1_B", "DB2_B", "DB3_B", "DB4_B", "DEFAULT", "SOCOFING", "calibrated_9d"]:
            self.register_standard_scaler(name, CALIBRATED_EMPIRICAL_MEAN_9D, CALIBRATED_EMPIRICAL_STD_9D)
        for name in ["POOLED_SET_B", "pooled_9d"]:
            self.register_standard_scaler(name, POOLED_SET_B_EMPIRICAL_MEAN_9D, POOLED_SET_B_EMPIRICAL_STD_9D)

        from src.services.mcc_calibrated_stats import (
            POOLED_SET_B_MCC_256_MEAN, POOLED_SET_B_MCC_256_STD,
            DB1_B_MCC_256_MEAN, DB1_B_MCC_256_STD,
            DB2_B_MCC_256_MEAN, DB2_B_MCC_256_STD,
            DB3_B_MCC_256_MEAN, DB3_B_MCC_256_STD,
            DB4_B_MCC_256_MEAN, DB4_B_MCC_256_STD,
        )
        zca_path = os.path.join(os.path.dirname(__file__), "mcc_zca_setb.npz")
        zca_map: Dict[str, np.ndarray] = {}
        if os.path.exists(zca_path):
            with np.load(zca_path) as npz:
                for k in npz.files:
                    zca_map[k] = np.asarray(npz[k], dtype=np.float32)

        pooled_zca = zca_map.get("POOLED_SET_B_MCC_256")
        for name in ["POOLED_SET_B_MCC_256", "DEFAULT_MCC_256", "DB1_A_MCC_256", "DB2_A_MCC_256", "DB3_A_MCC_256", "DB4_A_MCC_256"]:
            self.register_standard_scaler(name, POOLED_SET_B_MCC_256_MEAN, POOLED_SET_B_MCC_256_STD, zca_matrix=pooled_zca)
        self.register_standard_scaler("DB1_B_MCC_256", DB1_B_MCC_256_MEAN, DB1_B_MCC_256_STD, zca_matrix=zca_map.get("DB1_B_MCC_256"))
        self.register_standard_scaler("DB2_B_MCC_256", DB2_B_MCC_256_MEAN, DB2_B_MCC_256_STD, zca_matrix=zca_map.get("DB2_B_MCC_256"))
        self.register_standard_scaler("DB3_B_MCC_256", DB3_B_MCC_256_MEAN, DB3_B_MCC_256_STD, zca_matrix=zca_map.get("DB3_B_MCC_256"))
        self.register_standard_scaler("DB4_B_MCC_256", DB4_B_MCC_256_MEAN, DB4_B_MCC_256_STD, zca_matrix=zca_map.get("DB4_B_MCC_256"))

    def register_identity(self, name: str):
        """Registers a pass-through identity scaler."""
        self._scalers[name] = {
            "type": "identity",
            "transform": lambda x: x,
            "inverse": lambda x: x
        }

    def register_standard_scaler(
        self,
        name: str,
        mean: np.ndarray,
        std: np.ndarray,
        eps: float = 1e-6,
        zca_matrix: Optional[np.ndarray] = None,
    ):
        """Registers a z-score standardization scaler (x - mean) / (std + eps), with optional ZCA."""
        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)
        zca_f64 = None if zca_matrix is None else np.asarray(zca_matrix, dtype=np.float64)
        zca_f32 = None if zca_matrix is None else np.asarray(zca_matrix, dtype=np.float32)

        def _transform(x: np.ndarray) -> np.ndarray:
            z = (np.asarray(x, dtype=np.float64) - mean) / (std + eps)
            if zca_f64 is not None:
                z = z @ zca_f64
            return z

        def _inverse(z: np.ndarray) -> np.ndarray:
            return (np.asarray(z, dtype=np.float64) * (std + eps)) + mean

        self._scalers[name] = {
            "type": "standard",
            "mean": mean,
            "std": std,
            "zca_matrix": zca_f32,
            "transform": _transform,
            "inverse": _inverse
        }

    def transform(self, descriptors: np.ndarray, scaler_name: str = "default") -> np.ndarray:
        """Transforms descriptors using the specified registered scaler."""
        if scaler_name not in self._scalers:
            # Fall back to identity
            return descriptors
        return self._scalers[scaler_name]["transform"](descriptors)

    def get_scaler(self, scaler_name: str) -> Dict[str, Any]:
        """Retrieves registered scaler configuration."""
        if scaler_name not in self._scalers:
            raise KeyError(f"Scaler '{scaler_name}' not found in ScalerRegistry")
        return self._scalers[scaler_name]

    def get_scaler_params(self, scaler_name: str = "DEFAULT") -> Optional[Dict[str, Any]]:
        """Retrieves scaler parameters if registered, or None."""
        name_upper = scaler_name.upper()
        if name_upper in self._scalers:
            return self._scalers[name_upper]
        if scaler_name in self._scalers:
            return self._scalers[scaler_name]
        if "DEFAULT" in self._scalers:
            return self._scalers["DEFAULT"]
        return None
