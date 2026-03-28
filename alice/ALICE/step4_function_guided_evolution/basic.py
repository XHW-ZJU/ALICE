from numpy.random import random
import math
import random
import os
import numpy as np
from typing import List, Union


def create_directory_if_not_exists(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
        path (str): The path of the directory to be created.
    """
    os.makedirs(path, exist_ok=True)


def calculate_norm(dim: Union[int, float], distances: List[float]) -> float:
    """
    Calculate the norm from one n-dimensional point to another.

    Args:
        dim (int or float): The dimension of the norm (use math.inf for infinity norm).
        distances (List[float]): The differences in each dimension.

    Returns:
        float: The calculated norm.

    Raises:
        ValueError: If the distances list is empty.
    """
    if not distances:
        raise ValueError('The distances list is empty!')

    if dim == math.inf:
        return max(abs(d) for d in distances)
    else:
        return sum(d ** dim for d in distances) ** (1 / dim)


def calculate_mean(dim: Union[int, float], values: List[float]) -> float:
    """
    Calculate the mean of a list of float numbers.

    Args:
        dim (int or float): The dimension for mean calculation (use math.inf for max absolute value).
        values (List[float]): The list of values.

    Returns:
        float: The calculated mean.

    Raises:
        ValueError: If the values list is empty.
    """
    if not values:
        raise ValueError('The values list is empty!')

    if dim == math.inf:
        return max(abs(v) for v in values)
    else:
        return (sum(v ** dim for v in values) / len(values)) ** (1 / dim)


def weighted_choice(weights: np.ndarray) -> int:
    """
    Perform a weighted random choice from an array of weights.

    Args:
    weights (np.ndarray): Array of weights.

    Returns:
    int: Randomly chosen index from 0 to len(weights)-1.
    """
    # Ensure weights are non-negative
    positive_weights = np.maximum(weights, 0)

    # Compute cumulative sum of weights
    cumsum = np.cumsum(positive_weights)

    # Generate a random number and find its position in the cumulative sum
    return np.searchsorted(cumsum, random() * cumsum[-1])