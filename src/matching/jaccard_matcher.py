import numpy as np
from src.protection.bloom import bloom_jaccard_similarity

def match_bloom_templates(query_bloom: np.ndarray, enrolled_bloom: np.ndarray) -> float:
    """
    Computes Jaccard similarity between two Bloom filter templates.
    """
    return bloom_jaccard_similarity(query_bloom, enrolled_bloom)
