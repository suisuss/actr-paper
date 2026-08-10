"""
Simple pooling baseline methods.

Implements permutation-invariant and position-aware pooling:
- mean_pool: Mean pooling (equivalent to Chebyshev c0r)
- max_pool: Max pooling
- min_pool: Min pooling
- first_last_pool: First + last token concatenation
- minmax_pool: Min + max concatenation
- position_weighted_mean: Position-weighted mean with various schemes
"""

import numpy as np


def mean_pool(embeddings: np.ndarray) -> np.ndarray:
    """
    Mean pooling baseline.

    Equivalent to Chebyshev c0r. This is the simplest baseline
    and should always be compared against.

    Args:
        embeddings: (n_tokens, embedding_dim)

    Returns:
        (embedding_dim,) mean vector
    """
    if len(embeddings) == 0:
        raise ValueError("Cannot pool empty sequence")
    return embeddings.mean(axis=0)


def max_pool(embeddings: np.ndarray) -> np.ndarray:
    """
    Max pooling baseline.

    Takes maximum value per dimension. Captures "most activated"
    features regardless of position. Permutation invariant.

    Args:
        embeddings: (n_tokens, embedding_dim)

    Returns:
        (embedding_dim,) max-pooled vector
    """
    if len(embeddings) == 0:
        raise ValueError("Cannot pool empty sequence")
    return embeddings.max(axis=0)


def min_pool(embeddings: np.ndarray) -> np.ndarray:
    """
    Min pooling baseline.

    Takes minimum value per dimension.

    Args:
        embeddings: (n_tokens, embedding_dim)

    Returns:
        (embedding_dim,) min-pooled vector
    """
    if len(embeddings) == 0:
        raise ValueError("Cannot pool empty sequence")
    return embeddings.min(axis=0)


def first_last_pool(embeddings: np.ndarray) -> np.ndarray:
    """
    First + Last token concatenation baseline.

    Simple position-aware baseline that captures sequence boundaries.
    Output dimension is 2x embedding dimension.

    Args:
        embeddings: (n_tokens, embedding_dim)

    Returns:
        (2 * embedding_dim,) concatenated first and last
    """
    if len(embeddings) == 0:
        raise ValueError("Cannot pool empty sequence")
    if len(embeddings) == 1:
        return np.concatenate([embeddings[0], embeddings[0]])
    return np.concatenate([embeddings[0], embeddings[-1]])


def minmax_pool(embeddings: np.ndarray) -> np.ndarray:
    """
    Min + Max pooling concatenation baseline.

    Captures the range of activation per dimension.
    Output dimension is 2x embedding dimension.

    Args:
        embeddings: (n_tokens, embedding_dim)

    Returns:
        (2 * embedding_dim,) concatenated min and max
    """
    if len(embeddings) == 0:
        raise ValueError("Cannot pool empty sequence")
    return np.concatenate([embeddings.min(axis=0), embeddings.max(axis=0)])


def position_weighted_mean(
    embeddings: np.ndarray,
    scheme: str = 'recency',
    decay: float = 0.9
) -> np.ndarray:
    """
    Position-weighted mean pooling baseline.

    Weights tokens by position. Simple position injection without
    learning or complex transforms.

    Args:
        embeddings: (n_tokens, embedding_dim)
        scheme: Weighting scheme
            - 'recency': Later tokens weighted more (exponential decay from start)
            - 'primacy': Earlier tokens weighted more (exponential decay from end)
            - 'ends': First and last tokens weighted more than middle
            - 'linear_up': Linear increase toward end
            - 'linear_down': Linear decrease toward end
        decay: Decay factor for exponential schemes (default 0.9)

    Returns:
        (embedding_dim,) weighted mean vector
    """
    if len(embeddings) == 0:
        raise ValueError("Cannot pool empty sequence")

    n = len(embeddings)

    if n == 1:
        return embeddings[0].copy()

    if scheme == 'recency':
        weights = decay ** np.arange(n - 1, -1, -1)
    elif scheme == 'primacy':
        weights = decay ** np.arange(n)
    elif scheme == 'ends':
        positions = np.arange(n)
        center = (n - 1) / 2
        weights = np.abs(positions - center) / center if center > 0 else np.ones(n)
        weights = weights + 0.1
    elif scheme == 'linear_up':
        weights = (np.arange(n) + 1) / n
    elif scheme == 'linear_down':
        weights = (n - np.arange(n)) / n
    else:
        raise ValueError(f"Unknown scheme: {scheme}. Valid: recency, primacy, ends, linear_up, linear_down")

    weights = weights / weights.sum()

    return (embeddings * weights[:, None]).sum(axis=0)
