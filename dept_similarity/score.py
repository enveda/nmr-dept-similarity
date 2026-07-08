"""Code for calculating similarity scores between vectors using various metrics, optimized with numba for performance."""

import numpy as np
from numba import njit


@njit(cache=True)
def cosine_numba(a: np.ndarray, b: np.ndarray) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(a.shape[0]):
        av = float(a[i])
        bv = float(b[i])
        dot += av * bv
        norm_a += av * av
        norm_b += bv * bv

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a**0.5) * (norm_b**0.5))


@njit(cache=True)
def euclidean_numba(a: np.ndarray, b: np.ndarray) -> float:
    sq_sum = 0.0
    for i in range(a.shape[0]):
        diff = float(a[i]) - float(b[i])
        sq_sum += diff * diff
    return sq_sum**0.5


@njit(cache=True)
def manhattan_numba(a: np.ndarray, b: np.ndarray) -> float:
    abs_sum = 0.0
    for i in range(a.shape[0]):
        abs_sum += abs(float(a[i]) - float(b[i]))
    return abs_sum


@njit(cache=True)
def jaccard_numba(a: np.ndarray, b: np.ndarray) -> float:
    intersection = 0.0
    union = 0.0
    for i in range(a.shape[0]):
        av = float(a[i])
        bv = float(b[i])
        if av > 0 and bv > 0:
            intersection += 1.0
        if av > 0 or bv > 0:
            union += 1.0
    if union == 0.0:
        return 0.0
    return intersection / union


def compute_metric_numba(
    metric_name: str, query_vector: np.ndarray, candidate_vector: np.ndarray
) -> float:
    if metric_name == "cosine":
        return float(cosine_numba(query_vector, candidate_vector))
    if metric_name == "euclidean":
        return float(euclidean_numba(query_vector, candidate_vector))
    if metric_name == "manhattan":
        return float(manhattan_numba(query_vector, candidate_vector))
    if metric_name == "jaccard":
        return float(jaccard_numba(query_vector, candidate_vector))
    raise ValueError(f"Unsupported similarity metric: {metric_name}")
