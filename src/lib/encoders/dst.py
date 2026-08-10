"""
Discrete Sine Transform (DST-II) encoding.

DST is the odd counterpart to DCT, with different boundary behavior.
DST assumes odd symmetry (zero at boundaries), while DCT assumes even symmetry.
It has NO constant basis function, so it carries no mean of its own.

These are FEATURE coefficients: the least-squares projection of the sequence
onto DST-II columns, normalized by each column's squared norm (see
lib/encoders/projection.py) -- the same convention as gram_fit/dct_encode,
NOT the orthonormal scipy.fftpack.dst(norm='ortho') transform.

Components are 1-indexed to match the papers: s{k} is DST-II harmonic j = k
(so s1 is the first half-sine). There is NO s0 -- the DST has no DC/mean term.

Better for: Asymmetric patterns where start and end differ semantically.
"""

from typing import Dict
import numpy as np

from .projection import dst2_basis, basis_fit
from .utils import coeff_to_angles, coeff_to_magnitudes


def dst_encode(
    embeddings: np.ndarray,
    n_coefficients: int = 6,
    norm: str = 'ortho'
) -> np.ndarray:
    """
    Discrete Sine Transform (DST-II) encoding.

    Args:
        embeddings: (n_tokens, embedding_dim)
        n_coefficients: Number of DST coefficients to keep (harmonics j=1..n)
        norm: Accepted for backward compatibility and IGNORED. Coefficients
            are the least-squares projection normalized by each column's
            squared norm, not an orthonormal transform.

    Returns:
        (n_coefficients * embedding_dim,) flattened DST coefficients
    """
    if len(embeddings) == 0:
        return np.zeros(n_coefficients * embeddings.shape[1] if embeddings.ndim > 1 else n_coefficients)

    coeffs = basis_fit(embeddings, dst2_basis, n_coefficients)
    return coeffs.flatten()


def dst_encode_decomposed(
    embeddings: np.ndarray,
    n_coefficients: int = 6,
    norm: str = 'ortho',
    include_angular: bool = True,
    include_magnitude: bool = True
) -> Dict[str, np.ndarray]:
    """
    DST encoding with decomposed output.

    Args:
        embeddings: (n_tokens, embedding_dim)
        n_coefficients: Number of DST coefficients (harmonics j=1..n)
        norm: Accepted for backward compatibility and IGNORED (see dst_encode).
        include_angular: Include angular transforms (s1a, s2a, etc.)
        include_magnitude: Include magnitude transforms (s1m, s2m, etc.)

    Returns:
        Dict with:
            - 's1', 's2', ...: Individual DST coefficients (384 dims each),
              1-indexed: s{k} is DST-II harmonic j = k. There is no s0/DC term.
            - 's1a', 's2a', ...: Angular transforms (192 dims each)
            - 's1m', 's2m', ...: Magnitude transforms (192 dims each)
            - '_components': List of component names
    """
    if len(embeddings) == 0:
        emb_dim = embeddings.shape[1] if embeddings.ndim > 1 else 384
        result = {f's{i + 1}': np.zeros(emb_dim) for i in range(n_coefficients)}
        if include_angular:
            for i in range(n_coefficients):
                result[f's{i + 1}a'] = np.zeros(emb_dim // 2)
        if include_magnitude:
            for i in range(n_coefficients):
                result[f's{i + 1}m'] = np.zeros(emb_dim // 2)
        result['_components'] = [f's{i + 1}' for i in range(n_coefficients)]
        return result

    # (n_coefficients, emb_dim); rows beyond min(n_coefficients, n_tokens)
    # are zero-padded by basis_fit. Row i is DST-II harmonic j = i + 1.
    dst_coeffs = basis_fit(embeddings, dst2_basis, n_coefficients)

    result = {}
    components = []

    for i in range(n_coefficients):
        comp_name = f's{i + 1}'
        components.append(comp_name)
        coeff = dst_coeffs[i]
        result[comp_name] = coeff
        if include_angular:
            result[f's{i + 1}a'] = coeff_to_angles(coeff)
        if include_magnitude:
            result[f's{i + 1}m'] = coeff_to_magnitudes(coeff)

    result['_components'] = components
    return result
