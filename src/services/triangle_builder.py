"""
Triangle Builder Service.

Constructs Delaunay triangulation (or k-NN graphs) over minutiae point sets,
optionally deriving the dual simplicial incidence matrix A for topological consensus.
"""

from typing import Optional, List
import numpy as np

from src.graph.delaunay import build_delaunay_triangles, build_knn_graphs
from src.graph.hybrid_graph import build_hybrid_triangles
from src.matching.graph_consensus import compute_triangle_adjacency, pack_adjacency_matrix
from src.services.types import (
    ExtractedMinutiae, 
    BuiltTriangles, 
    BiometricErrorCode, 
    TriangulationError
)


class TriangleBuilder:
    """Production service for minutiae mesh and graph construction."""
    def __init__(self, method: str = "delaunay", knn_k: int = 5):
        if method not in ("delaunay", "knn", "hybrid"):
            raise ValueError(f"Unsupported meshing method: '{method}'. Choose 'delaunay', 'knn', or 'hybrid'")
        self.method = method
        self.knn_k = knn_k

    def build(self, minutiae: ExtractedMinutiae, compute_topology: bool = False) -> BuiltTriangles:
        """
        Builds triangles from minutiae set.

        Args:
            minutiae: Output of MinutiaeExtractor.
            compute_topology: If True, computes symmetric dual incidence matrix A.

        Returns:
            BuiltTriangles container with triangle tuples and optional adjacency matrix.

        Raises:
            TriangulationError: If triangulation fails or yields 0 triangles.
        """
        try:
            if self.method == "hybrid":
                triangles = build_hybrid_triangles(minutiae.minutiae, knn_k=self.knn_k)
            elif self.method == "knn":
                triangles = build_knn_graphs(minutiae.minutiae, k=self.knn_k)
            else:
                try:
                    triangles = build_delaunay_triangles(minutiae.minutiae)
                except Exception:
                    # Fallback to local k-NN star topology if planar Delaunay fails (e.g., near-collinear set)
                    triangles = build_knn_graphs(minutiae.minutiae, k=min(self.knn_k, len(minutiae.minutiae)))
        except Exception as e:
            raise TriangulationError(
                BiometricErrorCode.TRIANGULATION_FAILED,
                f"Triangulation construction failed: {str(e)}"
            )

        if len(triangles) == 0:
            raise TriangulationError(
                BiometricErrorCode.DEGENERATE_TRIANGLE_SET,
                "Minutiae distribution is degenerate (collinear or insufficient for Delaunay meshing)"
            )

        adj = None
        if compute_topology:
            adj = compute_triangle_adjacency(triangles)

        return BuiltTriangles(
            triangles=triangles,
            count=len(triangles),
            adjacency_matrix=adj
        )
