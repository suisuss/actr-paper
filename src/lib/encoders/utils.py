"""
Shared utility functions for encoder types.

Provides angular and magnitude transformations used across multiple encoders.
"""

import numpy as np


def to_angles(embeddings: np.ndarray) -> np.ndarray:
    """
    Convert embedding dimension pairs to angles.

    Treats adjacent dimension pairs as (x, y) coordinates and computes arctan2.
    This discards magnitude and keeps only direction in each 2D subspace.

    Args:
        embeddings: Array of shape (n_tokens, emb_dim)

    Returns:
        Array of shape (n_tokens, emb_dim // 2) with angles in [-pi, pi]
    """
    n_tokens, n_dims = embeddings.shape
    n_pairs = n_dims // 2
    pairs = embeddings[:, :n_pairs * 2].reshape(n_tokens, n_pairs, 2)
    return np.arctan2(pairs[:, :, 1], pairs[:, :, 0])


def to_magnitudes(embeddings: np.ndarray) -> np.ndarray:
    """
    Convert embedding dimension pairs to magnitudes.

    Treats adjacent dimension pairs as (x, y) coordinates and computes Euclidean norm.
    This discards direction and keeps only magnitude in each 2D subspace.

    Args:
        embeddings: Array of shape (n_tokens, emb_dim)

    Returns:
        Array of shape (n_tokens, emb_dim // 2) with magnitudes >= 0
    """
    n_tokens, n_dims = embeddings.shape
    n_pairs = n_dims // 2
    pairs = embeddings[:, :n_pairs * 2].reshape(n_tokens, n_pairs, 2)
    return np.sqrt(pairs[:, :, 0]**2 + pairs[:, :, 1]**2)


def coeff_to_angles(arr: np.ndarray) -> np.ndarray:
    """Convert a 1D coefficient array to angular representation using adjacent dimension pairs."""
    n_pairs = len(arr) // 2
    pairs = arr[:n_pairs * 2].reshape(n_pairs, 2)
    return np.arctan2(pairs[:, 1], pairs[:, 0])


def coeff_to_magnitudes(arr: np.ndarray) -> np.ndarray:
    """Convert a 1D coefficient array to magnitude representation using adjacent dimension pairs."""
    n_pairs = len(arr) // 2
    pairs = arr[:n_pairs * 2].reshape(n_pairs, 2)
    return np.sqrt(pairs[:, 0]**2 + pairs[:, 1]**2)


def apply_rope(embeddings: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """
    Apply Rotary Position Embeddings (RoPE) to embeddings.

    RoPE rotates each dimension pair by an angle proportional to position,
    with different frequencies for different dimension pairs.

    Args:
        embeddings: Array of shape (n_tokens, emb_dim)
        base: Base frequency for position encoding (default: 10000.0)

    Returns:
        Rotated embeddings with same shape (n_tokens, emb_dim)
    """
    n_tokens, d = embeddings.shape
    if n_tokens == 0:
        return embeddings.copy()

    positions = np.arange(n_tokens)
    dim_indices = np.arange(0, d, 2)
    theta = base ** (-dim_indices / d)

    angles = positions[:, None] * theta[None, :]  # (n_tokens, d//2)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    x_even = embeddings[:, 0::2]  # (n_tokens, d//2)
    x_odd = embeddings[:, 1::2]   # (n_tokens, d//2)

    rotated = np.empty_like(embeddings)
    rotated[:, 0::2] = x_even * cos_angles - x_odd * sin_angles
    rotated[:, 1::2] = x_even * sin_angles + x_odd * cos_angles

    return rotated


def apply_random_rotation(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Apply random orthogonal rotation to embeddings.

    This is an ablation baseline for RoPE. By applying a random (but fixed)
    orthogonal rotation, we can test whether improvements from RoPE are due
    to its specific positional structure, or just from rotating the space.

    Args:
        embeddings: Array of shape (n_tokens, emb_dim)
        seed: Random seed for reproducibility

    Returns:
        Rotated embeddings with same shape (n_tokens, emb_dim)
    """
    if embeddings.shape[0] == 0:
        return embeddings.copy()

    d = embeddings.shape[1]
    rng = np.random.RandomState(seed)

    random_matrix = rng.randn(d, d)
    Q, _ = np.linalg.qr(random_matrix)

    return embeddings @ Q
