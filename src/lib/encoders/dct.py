"""
Discrete Cosine Transform (DCT-II) encoding.

DCT decomposes sequences into orthogonal frequency components:
- Coefficient 0: DC component (EXACTLY the mean, by the projection below)
- Coefficient 1: Fundamental frequency
- Coefficient k: k-th harmonic

These are FEATURE coefficients: the least-squares projection of the sequence
onto DCT-II columns, normalized by each column's squared norm (see
lib/encoders/projection.py). This is the same convention as gram_fit, so
d0 = g0 = mean exactly and coefficients are length-comparable -- it is what
the papers use. It is NOT the orthonormal scipy.fftpack.dct(norm='ortho')
transform (whose d0 is sqrt(n)*mean); that transform is used only by the
reconstruction encoders (rope/circular_fourier/hyperspherical), which invert
it with idct and need orthonormality.

See material/methodology/05_baselines.md for rationale.
"""

from typing import Dict
import numpy as np

from .projection import dct2_basis, basis_fit
from .utils import coeff_to_angles, coeff_to_magnitudes


def dct_encode(
    embeddings: np.ndarray,
    n_coefficients: int = 6,
    norm: str = 'ortho'
) -> np.ndarray:
    """
    Discrete Cosine Transform (DCT-II) encoding.

    Args:
        embeddings: (n_tokens, embedding_dim)
        n_coefficients: Number of DCT coefficients to keep
        norm: Accepted for backward compatibility and IGNORED. Coefficients
            are the least-squares projection normalized by each column's
            squared norm (d0 = mean exactly), not an orthonormal transform.

    Returns:
        (n_coefficients * embedding_dim,) flattened DCT coefficients
    """
    if len(embeddings) == 0:
        return np.zeros(n_coefficients * embeddings.shape[1] if embeddings.ndim > 1 else n_coefficients)

    coeffs = basis_fit(embeddings, dct2_basis, n_coefficients)
    return coeffs.flatten()


def dct_encode_decomposed(
    embeddings: np.ndarray,
    n_coefficients: int = 6,
    norm: str = 'ortho',
    include_angular: bool = True,
    include_magnitude: bool = True
) -> Dict[str, np.ndarray]:
    """
    DCT encoding with decomposed output.

    Returns individual coefficient arrays for component-wise analysis,
    including optional angular and magnitude transforms.

    Args:
        embeddings: (n_tokens, embedding_dim)
        n_coefficients: Number of DCT coefficients
        norm: Accepted for backward compatibility and IGNORED (see dct_encode).
        include_angular: Include angular transforms (d0a, d1a, etc.)
        include_magnitude: Include magnitude transforms (d0m, d1m, etc.)

    Returns:
        Dict with:
            - 'd0', 'd1', ...: Individual DCT coefficients (384 dims each)
            - 'd0a', 'd1a', ...: Angular transforms (192 dims each)
            - 'd0m', 'd1m', ...: Magnitude transforms (192 dims each)
            - '_components': List of component names
    """
    if len(embeddings) == 0:
        emb_dim = embeddings.shape[1] if embeddings.ndim > 1 else 384
        result = {f'd{i}': np.zeros(emb_dim) for i in range(n_coefficients)}
        if include_angular:
            for i in range(n_coefficients):
                result[f'd{i}a'] = np.zeros(emb_dim // 2)
        if include_magnitude:
            for i in range(n_coefficients):
                result[f'd{i}m'] = np.zeros(emb_dim // 2)
        result['_components'] = [f'd{i}' for i in range(n_coefficients)]
        return result

    # (n_coefficients, emb_dim); rows beyond min(n_coefficients, n_tokens)
    # are zero-padded by basis_fit.
    dct_coeffs = basis_fit(embeddings, dct2_basis, n_coefficients)

    result = {}
    components = []

    for i in range(n_coefficients):
        comp_name = f'd{i}'
        components.append(comp_name)
        coeff = dct_coeffs[i]
        result[comp_name] = coeff
        if include_angular:
            result[f'd{i}a'] = coeff_to_angles(coeff)
        if include_magnitude:
            result[f'd{i}m'] = coeff_to_magnitudes(coeff)

    result['_components'] = components
    return result
