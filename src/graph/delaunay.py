from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial import Delaunay
from src.minutiae.extractor import Minutia

def is_valid_triangle(m1: Minutia, m2: Minutia, m3: Minutia,
                      max_edge_length: float = 120.0,
                      min_edge_length: float = 0.0,
                      min_area: float = 0.0,
                      max_aspect_ratio: Optional[float] = None) -> bool:
    """
    Evaluates whether a triangle simplex satisfies geometric stability constraints:
    1. Maximum edge length: max(l1, l2, l3) <= max_edge_length.
    2. Minimum edge length: min(l1, l2, l3) >= min_edge_length (suppresses high-frequency jitter).
    3. Minimum triangle area: area >= min_area (suppresses near-collinear degenerate triangles).
    4. Maximum aspect ratio: (longest edge / shortest edge) <= max_aspect_ratio (suppresses sliver triangles).
    """
    l1 = float(np.hypot(m1.x - m2.x, m1.y - m2.y))
    l2 = float(np.hypot(m2.x - m3.x, m2.y - m3.y))
    l3 = float(np.hypot(m3.x - m1.x, m3.y - m1.y))
    
    shortest = min(l1, l2, l3)
    longest = max(l1, l2, l3)
    
    if longest > max_edge_length or shortest < min_edge_length:
        return False
        
    if min_area > 0.0 or max_aspect_ratio is not None:
        # 2D cross-product area: 0.5 * |(x2 - x1)(y3 - y1) - (x3 - x1)(y2 - y1)|
        area = 0.5 * abs((m2.x - m1.x) * (m3.y - m1.y) - (m3.x - m1.x) * (m2.y - m1.y))
        if area < min_area:
            return False
        if max_aspect_ratio is not None:
            aspect = longest / max(shortest, 1e-6)
            if aspect > max_aspect_ratio:
                return False
                
    return True

def build_delaunay_triangles(minutiae: List[Minutia], 
                            max_edge_length: float = 120.0,
                            min_edge_length: float = 0.0,
                            min_area: float = 0.0,
                            max_aspect_ratio: Optional[float] = None) -> List[Tuple[Minutia, Minutia, Minutia]]:
    """
    Constructs Delaunay triangles from detected minutiae points, filtering triangles
    with overly long edges and optional degeneracy constraints (sliver / near-collinear / aspect ratio).
    
    Args:
        minutiae: List of Minutia objects.
        max_edge_length: Maximum allowed edge length for a valid local triangle.
        min_edge_length: Minimum allowed edge length (default: 0.0).
        min_area: Minimum allowed triangle area in square pixels (default: 0.0).
        max_aspect_ratio: Maximum allowed aspect ratio (longest / shortest edge) (default: None).
        
    Returns:
        List of 3-tuples of Minutia representing valid local Delaunay triangles.
    """
    if len(minutiae) < 3:
        return []
        
    points = np.array([[m.x, m.y] for m in minutiae], dtype=np.float32)
    tri = Delaunay(points)
    
    valid_triangles = []
    for simplex in tri.simplices:
        m1, m2, m3 = minutiae[simplex[0]], minutiae[simplex[1]], minutiae[simplex[2]]
        if is_valid_triangle(m1, m2, m3, 
                              max_edge_length=max_edge_length,
                              min_edge_length=min_edge_length,
                              min_area=min_area,
                              max_aspect_ratio=max_aspect_ratio):
            valid_triangles.append((m1, m2, m3))
            
    return valid_triangles

def build_knn_graphs(minutiae: List[Minutia], k: int = 5, 
                     max_edge_length: float = 120.0,
                     min_edge_length: float = 0.0,
                     min_area: float = 0.0,
                     max_aspect_ratio: Optional[float] = None) -> List[Tuple[Minutia, Minutia, Minutia]]:
    """
    Builds local triangle triplets from k-nearest neighbors of each minutia,
    applying edge length and optional degeneracy filtering.
    """
    if len(minutiae) < 3:
        return []
        
    points = np.array([[m.x, m.y] for m in minutiae], dtype=np.float32)
    n = len(minutiae)
    triplets = set()
    
    for i in range(n):
        dists = np.hypot(points[:, 0] - points[i, 0], points[:, 1] - points[i, 1])
        # Find k nearest (excluding self)
        nearest_indices = np.argsort(dists)[1:min(k+1, n)]
        
        # Form triplets with pairs of neighbors
        for j_idx in range(len(nearest_indices)):
            for k_idx in range(j_idx + 1, len(nearest_indices)):
                idx1, idx2, idx3 = i, nearest_indices[j_idx], nearest_indices[k_idx]
                m1, m2, m3 = minutiae[idx1], minutiae[idx2], minutiae[idx3]
                if is_valid_triangle(m1, m2, m3,
                                      max_edge_length=max_edge_length,
                                      min_edge_length=min_edge_length,
                                      min_area=min_area,
                                      max_aspect_ratio=max_aspect_ratio):
                    # Sort indices to avoid duplicates
                    key = tuple(sorted([idx1, idx2, idx3]))
                    triplets.add(key)
                    
    return [(minutiae[i], minutiae[j], minutiae[k]) for i, j, k in triplets]

