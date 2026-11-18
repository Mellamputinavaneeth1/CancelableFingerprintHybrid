import hashlib
import numpy as np

def encode_to_bloom_filter(bitstrings: np.ndarray, filter_size_bits: int = 256, 
                           num_hashes: int = 3, salt: str = "bloom_salt") -> np.ndarray:
    """
    Encodes a set of descriptor bitstrings into a single Bloom filter bit-vector.
    
    Args:
        bitstrings: (N, M) binary descriptor matrix.
        filter_size_bits: Length of Bloom filter array in bits (e.g., 256).
        num_hashes: Number of independent hash functions.
        salt: Optional salt to key the hashing.
        
    Returns:
        bloom_filter: 1D np.ndarray of shape (filter_size_bits,), dtype uint8 (0 or 1).
    """
    bloom = np.zeros(filter_size_bits, dtype=np.uint8)
    if bitstrings.size == 0:
        return bloom
        
    for row in bitstrings:
        row_bytes = row.tobytes()
        for h_idx in range(num_hashes):
            # Derive index via SHA-256 slice
            data = salt.encode('utf-8') + bytes([h_idx]) + row_bytes
            digest = hashlib.sha256(data).digest()
            idx = int.from_bytes(digest[:4], 'big') % filter_size_bits
            bloom[idx] = 1
            
    return bloom

def bloom_jaccard_similarity(bloom1: np.ndarray, bloom2: np.ndarray) -> float:
    """
    Computes Jaccard similarity between two Bloom filter bit vectors:
    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    intersection = np.sum(np.bitwise_and(bloom1, bloom2))
    union = np.sum(np.bitwise_or(bloom1, bloom2))
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)
