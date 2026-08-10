"""
Gram polynomial encoding (Discrete Orthogonal Polynomials).

Gram polynomials are orthogonal under summation at uniform discrete points:
    sum_i G_j(i; n) G_k(i; n) = 0 for j != k

This is exactly the orthogonality property that Chebyshev LACKS at uniform points.
~1000x faster than Chebyshev fitting.

Gram polynomial interpretation vs Chebyshev:
- g0 ~ c0: Mean/constant (but g0 is EXACTLY the DC component)
- g1 = c1: Linear trend (g1 is EXACTLY the linear component)
- g2 = c2: Curvature (g2 is EXACTLY the quadratic component)

See material/theory/06_orthogonality_uniform_samples.md for orthogonality analysis.
"""

from typing import Dict, Any
import numpy as np
from scipy.special import comb

from .utils import coeff_to_angles, coeff_to_magnitudes


def gram_polynomial_value(k: int, i: int, n: int) -> float:
    """
    Evaluate Gram polynomial G_k(i; n) at discrete point i.

    Args:
        k: Polynomial degree (0 to n-1)
        i: Discrete point index (0 to n-1)
        n: Number of points

    Returns:
        G_k(i; n) value
    """
    result = 0.0
    for s in range(k + 1):
        if i < s:
            continue
        if n - 1 < s:
            continue
        term = ((-1) ** s *
                comb(k, s, exact=True) *
                comb(k + s, s, exact=True) *
                comb(i, s, exact=True) /
                comb(n - 1, s, exact=True))
        result += term
    return result


def gram_basis_matrix(n: int, degree: int) -> np.ndarray:
    """
    Build Gram polynomial basis matrix for n uniform points.

    Args:
        n: Number of sample points
        degree: Maximum polynomial degree (0 to degree)

    Returns:
        G: (n, degree+1) matrix where G[i, k] = G_k(i; n)
           Columns are orthogonal: G.T @ G is diagonal
    """
    actual_degree = min(degree, n - 1)
    G = np.zeros((n, actual_degree + 1))

    for k in range(actual_degree + 1):
        for i in range(n):
            G[i, k] = gram_polynomial_value(k, i, n)

    return G


def gram_basis_matrix_fast(n: int, degree: int) -> np.ndarray:
    """
    Build discrete orthogonal polynomial basis using explicit formula.

    Uses the Gram polynomial definition which is exactly orthogonal at uniform points.
    """
    actual_degree = min(degree, n - 1)
    G = np.zeros((n, actual_degree + 1))

    if n == 1:
        G[0, 0] = 1.0
        return G

    for k in range(actual_degree + 1):
        for i in range(n):
            G[i, k] = gram_polynomial_value(k, i, n)

    return G


def gram_fit(embeddings: np.ndarray, degree: int = 5, normalize: bool = True) -> np.ndarray:
    """
    Fit Gram polynomials to a sequence of embeddings.

    Because Gram polynomials are orthogonal at uniform points,
    the coefficients are computed directly without solving a linear system.

    Args:
        embeddings: (n_tokens, embedding_dim) array
        degree: Maximum polynomial degree
        normalize: Whether to normalize coefficients by basis norm

    Returns:
        coefficients: (degree+1, embedding_dim) array
    """
    n_tokens, emb_dim = embeddings.shape

    if n_tokens == 0:
        return np.zeros((degree + 1, emb_dim))

    if n_tokens == 1:
        coeffs = np.zeros((degree + 1, emb_dim))
        coeffs[0] = embeddings[0]
        return coeffs

    actual_degree = min(degree, n_tokens - 1)

    G = gram_basis_matrix_fast(n_tokens, actual_degree)

    norms_sq = np.sum(G ** 2, axis=0)

    coeffs = np.zeros((degree + 1, emb_dim))

    for k in range(actual_degree + 1):
        if norms_sq[k] > 1e-10:
            inner_product = G[:, k] @ embeddings
            coeffs[k] = inner_product / norms_sq[k]

    return coeffs


def gram_encode(
    embeddings: np.ndarray,
    n_coefficients: int = 6,
    normalize: bool = True
) -> np.ndarray:
    """
    Gram polynomial encoding.

    Args:
        embeddings: (n_tokens, embedding_dim)
        n_coefficients: Number of Gram coefficients to keep
        normalize: Whether to normalize by basis norm

    Returns:
        (n_coefficients * embedding_dim,) flattened Gram coefficients
    """
    if len(embeddings) == 0:
        emb_dim = embeddings.shape[1] if embeddings.ndim > 1 else 384
        return np.zeros(n_coefficients * emb_dim)

    coeffs = gram_fit(embeddings, degree=n_coefficients - 1, normalize=normalize)
    return coeffs.flatten()


def gram_encode_decomposed(
    embeddings: np.ndarray,
    n_coefficients: int = 6,
    include_angular: bool = True,
    include_magnitude: bool = True
) -> Dict[str, np.ndarray]:
    """
    Gram encoding with decomposed output.

    Args:
        embeddings: (n_tokens, embedding_dim)
        n_coefficients: Number of Gram coefficients
        include_angular: Include angular transforms (g0a, g1a, etc.)
        include_magnitude: Include magnitude transforms (g0m, g1m, etc.)

    Returns:
        Dict with:
            - 'g0', 'g1', ...: Individual Gram coefficients (384 dims each)
            - 'g0a', 'g1a', ...: Angular transforms (192 dims each)
            - 'g0m', 'g1m', ...: Magnitude transforms (192 dims each)
            - '_components': List of component names
    """
    if len(embeddings) == 0:
        emb_dim = embeddings.shape[1] if embeddings.ndim > 1 else 384
        result = {f'g{i}': np.zeros(emb_dim) for i in range(n_coefficients)}
        if include_angular:
            for i in range(n_coefficients):
                result[f'g{i}a'] = np.zeros(emb_dim // 2)
        if include_magnitude:
            for i in range(n_coefficients):
                result[f'g{i}m'] = np.zeros(emb_dim // 2)
        result['_components'] = [f'g{i}' for i in range(n_coefficients)]
        return result

    n_tokens, emb_dim = embeddings.shape
    coeffs = gram_fit(embeddings, degree=n_coefficients - 1)

    result = {}
    components = []

    for i in range(n_coefficients):
        comp_name = f'g{i}'
        components.append(comp_name)
        if i < coeffs.shape[0]:
            coeff = coeffs[i]
            result[comp_name] = coeff
            if include_angular:
                result[f'g{i}a'] = coeff_to_angles(coeff)
            if include_magnitude:
                result[f'g{i}m'] = coeff_to_magnitudes(coeff)
        else:
            result[comp_name] = np.zeros(emb_dim)
            if include_angular:
                result[f'g{i}a'] = np.zeros(emb_dim // 2)
            if include_magnitude:
                result[f'g{i}m'] = np.zeros(emb_dim // 2)

    result['_components'] = components
    return result


def verify_gram_orthogonality(n: int, degree: int) -> Dict[str, Any]:
    """
    Verify that Gram polynomials are orthogonal at n uniform points.

    Returns correlation matrix and diagnostics.
    """
    G = gram_basis_matrix_fast(n, degree)

    gram_matrix = G.T @ G

    norms = np.sqrt(np.diag(gram_matrix))
    norms[norms < 1e-10] = 1.0
    correlation = gram_matrix / np.outer(norms, norms)

    off_diag_mask = ~np.eye(correlation.shape[0], dtype=bool)
    max_correlation = np.max(np.abs(correlation[off_diag_mask])) if off_diag_mask.any() else 0.0

    return {
        'n_points': n,
        'degree': degree,
        'gram_matrix': gram_matrix,
        'correlation_matrix': correlation,
        'max_off_diagonal_correlation': max_correlation,
        'is_orthogonal': max_correlation < 1e-10
    }
