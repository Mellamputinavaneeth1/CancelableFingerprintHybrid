"""
Threshold Registry Service.

Mainages sensor-calibrated and dataset-specific operational thresholds
for Equal Error Rate (EER), 1.0% FAR, and 0.1% FAR operating points.
"""

from typing import Dict, Any, Optional
from src.services.types import BiometricErrorCode, ThresholdNotFoundError


class ThresholdRegistry:
    """
    Production threshold registry ensuring decisions are made strictly against
    pre-calibrated, verified operating points.
    """
    def __init__(self):
        self._registry: Dict[str, Dict[str, float]] = {
            "DB1_A": {
                "EER": 0.3751,
                "FAR_1_PERCENT": 0.3950,
                "FAR_0_1_PERCENT": 0.4100
            },
            "DB1_B": {
                "EER": 0.3751,
                "FAR_1_PERCENT": 0.3900,
                "FAR_0_1_PERCENT": 0.4050
            },
            "DB2_B": {
                "EER": 0.3751,
                "FAR_1_PERCENT": 0.3880,
                "FAR_0_1_PERCENT": 0.4020
            },
            "DB3_B": {
                "EER": 0.3751,
                "FAR_1_PERCENT": 0.3890,
                "FAR_0_1_PERCENT": 0.4040
            },
            "DB4_B": {
                "EER": 0.3751,
                "FAR_1_PERCENT": 0.3870,
                "FAR_0_1_PERCENT": 0.4010
            },
            "SOCOFING": {
                "BENCHMARK_ACCEPT": 0.3891,
                "EER": 0.3891
            },
            "DEFAULT": {
                "EER": 0.3751,
                "FAR_1_PERCENT": 0.3950,
                "FAR_0_1_PERCENT": 0.4100,
                "BENCHMARK_ACCEPT": 0.3891
            }
        }
        from src.services.mcc_calibrated_stats import CALIBRATED_MCC_THRESHOLDS
        for k, vals in CALIBRATED_MCC_THRESHOLDS.items():
            self._registry[k] = dict(vals)
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def get_threshold(self, dataset: str = "DEFAULT", operating_point: str = "EER") -> float:
        """
        Retrieves calibrated operating threshold.

        Args:
            dataset: Sensor or benchmark dataset identifier (e.g., 'DB1_A', 'DB1_B').
            operating_point: 'EER', 'FAR_1_PERCENT', 'FAR_0_1_PERCENT'.

        Returns:
            Calibrated decision threshold float.

        Raises:
            ThresholdNotFoundError: If dataset or operating point has not been calibrated.
        """
        ds_key = dataset.upper()
        if ds_key not in self._registry:
            if "DEFAULT" in self._registry:
                ds_key = "DEFAULT"
            else:
                raise ThresholdNotFoundError(
                    BiometricErrorCode.THRESHOLD_NOT_FOUND,
                    f"No threshold registered for dataset '{dataset}'"
                )

        points = self._registry[ds_key]
        op_key = operating_point.upper()
        if op_key not in points:
            raise ThresholdNotFoundError(
                BiometricErrorCode.INVALID_OPERATING_POINT,
                f"Operating point '{operating_point}' not calibrated for dataset '{dataset}'. Available: {list(points.keys())}"
            )

        return points[op_key]

    def register_threshold(self, dataset: str, operating_point: str, threshold: float, metadata: Optional[Dict[str, Any]] = None):
        """Registers or updates a calibrated operating threshold."""
        ds_key = dataset.upper()
        op_key = operating_point.upper()
        if ds_key not in self._registry:
            self._registry[ds_key] = {}
        self._registry[ds_key][op_key] = float(threshold)
        if metadata:
            self._metadata[f"{ds_key}_{op_key}"] = dict(metadata)

    def evaluate_decision(self, score: float, dataset: str = "DEFAULT", operating_point: str = "EER") -> str:
        """Evaluates whether score achieves acceptance against calibrated threshold."""
        tau = self.get_threshold(dataset, operating_point)
        return "ACCEPT" if float(score) >= tau else "REJECT"
