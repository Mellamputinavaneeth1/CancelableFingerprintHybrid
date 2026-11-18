"""
Descriptor Builder Service.

Extracts canonical 9-D pure geometric invariant descriptors and directional
reliability quality weights from minutiae triangle sets.
"""

from typing import Optional, List, Tuple
import numpy as np

from src.graph.descriptors import extract_all_descriptors
from src.services.types import (
    BuiltTriangles, 
    BuiltDescriptors, 
    BiometricErrorCode, 
    BiometricServiceException
)


class DescriptorBuilder:
    """Production service for geometric descriptor extraction."""
    def __init__(self, in_dim: int = 9, tie_breaker: str = "geometry_only"):
        if in_dim not in (9, 12):
            raise ValueError(f"in_dim must be 9 (pure geometric) or 12 (with types). Got {in_dim}")
        self.in_dim = in_dim
        self.tie_breaker = tie_breaker

    def build(self, triangles: BuiltTriangles, dpi: float = 500.0) -> BuiltDescriptors:
        """
        Extracts invariant descriptors for all triangles.

        Args:
            triangles: Output of TriangleBuilder.
            dpi: Physical sensor resolution in dots per inch.

        Returns:
            BuiltDescriptors container with (N, in_dim) features and (N,) weights.
        """
        include_types = (self.in_dim == 12)
        try:
            descriptors, weights = extract_all_descriptors(
                triangles.triangles,
                dpi=dpi,
                include_types=include_types,
                canonical_tie_breaker=self.tie_breaker
            )
        except Exception as e:
            raise BiometricServiceException(
                BiometricErrorCode.DEGENERATE_TRIANGLE_SET,
                f"Failed to extract descriptors from triangles: {str(e)}"
            )

        return BuiltDescriptors(
            descriptors=descriptors,
            weights=weights,
            in_dim=self.in_dim,
            count=len(descriptors)
        )
