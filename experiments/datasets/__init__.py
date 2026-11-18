"""
Benchmark dataset ingestion and validation module for Cancelable Fingerprint Biometrics.
Supports FVC2002 (DB1, DB2) and SOCOFing.
"""

from .fvc_loader import load_fvc_dataset, download_fvc_subset, FVC_SUBSETS
from .socofing_loader import load_socofing_dataset, generate_synthetic_socofing_benchmark
from .dataset_validator import validate_fvc_dataset, validate_socofing_dataset

__all__ = [
    "load_fvc_dataset",
    "download_fvc_subset",
    "FVC_SUBSETS",
    "load_socofing_dataset",
    "generate_synthetic_socofing_benchmark",
    "validate_fvc_dataset",
    "validate_socofing_dataset"
]
