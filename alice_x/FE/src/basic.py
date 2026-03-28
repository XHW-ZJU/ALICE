# from numpy.random import random
import math
import random
import os
import numpy as np
from typing import List, Union

def create_directory_if_not_exists(path: str) -> None:
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def calculate_norm(dim: Union[int, float], distances: List[float]) -> float:
    """
    Calculate the norm from one n-dimensional point to another.
    """
    if not distances:
        raise ValueError('The distances list is empty!')
    if dim == math.inf:
        return max(abs(d) for d in distances)
    return sum(d ** dim for d in distances) ** (1 / dim)

def calculate_mean(dim: Union[int, float], values: List[float]) -> float:
    """
    Calculate the mean of a list of float numbers.
    """
    if not values:
        raise ValueError('The values list is empty!')
    if dim == math.inf:
        return max(abs(v) for v in values)
    return (sum(v ** dim for v in values) / len(values)) ** (1 / dim)

def weighted_choice(weights: np.ndarray) -> int:
    """
    Perform a weighted random choice from an array of weights.

    Args:
        weights (np.ndarray): Array of weights (can contain negatives; they are clamped to 0).

    Returns:
        int: Randomly chosen index in [0, len(weights)-1].
    """
    # Fast path for all-zero or negative weights: fall back to uniform
    if weights.size == 0:
        raise ValueError("weights must be non-empty")
    positive_weights = np.maximum(weights, 0.0)
    total = positive_weights.sum()
    if total <= 0:
        return int(random.random() * len(weights))

    # Cumulative once, searchsorted
    cumsum = np.cumsum(positive_weights)
    r = random.random() * total
    return int(np.searchsorted(cumsum, r))
