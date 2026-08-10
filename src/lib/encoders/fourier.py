"""
Fourier magnitude encoding - the primary frequency decomposition method.

Fourier magnitude provides phase-invariant frequency decomposition:
    f_k = sqrt(d_k^2 + s_k^2)

where d_k is the DCT coefficient and s_k is the DST coefficient at frequency k.

This is the recommended approach for word-order and trajectory analysis because:
1. Phase-invariant: Captures frequency strength regardless of pattern position
2. Optimal for word-order: f1 achieves 0.7654 AUC on PAWS (outperforming d1 and s1)
3. Clean interpretation: Each f_k represents strength of k-th frequency

Key finding: Phase varies across sentences (~38 deg mean, 64 deg std),
so phase-invariant magnitude captures "how much linear trajectory"
regardless of where the pattern starts.

Reference:
    material/theory/fourier_magnitude_sentence_embeddings.md
"""

from typing import Dict
import numpy as np

from .utils import coeff_to_angles, coeff_to_magnitudes


def fourier_magnitude_coefficient(embeddings: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Compute Fourier magnitude of coefficient k.

    Combines DCT and DST as real/imaginary parts of DFT:
        f_k = sqrt(d_k^2 + s_k^2) = |d_k + i*s_k|

    This is phase-invariant: captures the STRENGTH of frequency k
    regardless of where the pattern starts in the sequence.

    Args:
        embeddings: (n_tokens, embedding_dim)
        k: Frequency index (0 = DC, 1 = linear/fundamental, 2 = quadratic, etc.)

    Returns:
        (embedding_dim,) Fourier magnitude vector
    """
    n_tokens, emb_dim = embeddings.shape

    if n_tokens == 0:
        return np.zeros(emb_dim)

    if n_tokens == 1:
        if k == 0:
            return np.abs(embeddings[0])
        return np.zeros(emb_dim)

    n = np.arange(n_tokens)
    theta = np.pi * k * (2*n + 1) / (2 * n_tokens)

    w = np.exp(1j * theta)

    f_k = np.sqrt(2 / n_tokens) * np.abs(embeddings.T @ w)

    return f_k.real


def fourier_magnitude_encode(
    embeddings: np.ndarray,
    n_coefficients: int = 6
) -> np.ndarray:
    """
    Fourier magnitude encoding for multiple coefficients.

    Computes f_k = sqrt(d_k^2 + s_k^2) for k = 0, 1, ..., n_coefficients-1.

    Args:
        embeddings: (n_tokens, embedding_dim)
        n_coefficients: Number of Fourier magnitude coefficients

    Returns:
        (n_coefficients * embedding_dim,) flattened Fourier magnitudes
    """
    if len(embeddings) == 0:
        emb_dim = embeddings.shape[1] if embeddings.ndim > 1 else 384
        return np.zeros(n_coefficients * emb_dim)

    n_tokens, emb_dim = embeddings.shape

    coeffs = np.zeros((n_coefficients, emb_dim))
    for k in range(min(n_coefficients, n_tokens)):
        coeffs[k] = fourier_magnitude_coefficient(embeddings, k)

    return coeffs.flatten()


def fourier_magnitude_encode_decomposed(
    embeddings: np.ndarray,
    n_coefficients: int = 6,
    include_angular: bool = True,
    include_magnitude: bool = True
) -> Dict[str, np.ndarray]:
    """
    Fourier magnitude encoding with decomposed output.

    Returns individual Fourier magnitude coefficients for component-wise analysis.

    Args:
        embeddings: (n_tokens, embedding_dim)
        n_coefficients: Number of Fourier magnitude coefficients
        include_angular: Include angular transforms (f0a, f1a, etc.)
        include_magnitude: Include magnitude-of-magnitude transforms (f0m, f1m, etc.)

    Returns:
        Dict with:
            - 'f0', 'f1', ...: Individual Fourier magnitude coefficients (384 dims each)
            - 'f0a', 'f1a', ...: Angular transforms (192 dims each)
            - 'f0m', 'f1m', ...: Magnitude transforms (192 dims each)
            - '_components': List of component names
    """
    if len(embeddings) == 0:
        emb_dim = embeddings.shape[1] if embeddings.ndim > 1 else 384
        result = {f'f{i}': np.zeros(emb_dim) for i in range(n_coefficients)}
        if include_angular:
            for i in range(n_coefficients):
                result[f'f{i}a'] = np.zeros(emb_dim // 2)
        if include_magnitude:
            for i in range(n_coefficients):
                result[f'f{i}m'] = np.zeros(emb_dim // 2)
        result['_components'] = [f'f{i}' for i in range(n_coefficients)]
        return result

    n_tokens, emb_dim = embeddings.shape

    result = {}
    components = []

    for k in range(n_coefficients):
        comp_name = f'f{k}'
        components.append(comp_name)

        if k < n_tokens:
            coeff = fourier_magnitude_coefficient(embeddings, k)
            result[comp_name] = coeff
            if include_angular:
                result[f'f{k}a'] = coeff_to_angles(coeff)
            if include_magnitude:
                result[f'f{k}m'] = coeff_to_magnitudes(coeff)
        else:
            result[comp_name] = np.zeros(emb_dim)
            if include_angular:
                result[f'f{k}a'] = np.zeros(emb_dim // 2)
            if include_magnitude:
                result[f'f{k}m'] = np.zeros(emb_dim // 2)

    result['_components'] = components
    return result


def fourier_magnitude_from_dct_dst(
    dct_coeffs: np.ndarray,
    dst_coeffs: np.ndarray
) -> np.ndarray:
    """
    Compute Fourier magnitude from precomputed DCT and DST coefficients.

    IMPORTANT: DCT and DST coefficients must be at the SAME frequency.

    Args:
        dct_coeffs: DCT coefficient at frequency k (embedding_dim,)
        dst_coeffs: DST coefficient at SAME frequency k (embedding_dim,)

    Returns:
        (embedding_dim,) Fourier magnitude = sqrt(dct^2 + dst^2)
    """
    return np.sqrt(dct_coeffs**2 + dst_coeffs**2)


def fourier_magnitude_from_scipy(
    dct_coeffs: np.ndarray,
    dst_coeffs: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Compute Fourier magnitude from scipy DCT/DST outputs, handling the index offset.

    scipy's DST-II uses frequency k+1 at index k, so this function
    correctly pairs DCT[k] with DST[k-1] for k>=1.

    Args:
        dct_coeffs: Full DCT output array (n_coeffs, embedding_dim) from scipy with norm='ortho'
        dst_coeffs: Full DST output array (n_coeffs, embedding_dim) from scipy with norm='ortho'
        k: Frequency index (0 = DC, 1 = fundamental, etc.)

    Returns:
        (embedding_dim,) Fourier magnitude at frequency k
    """
    d_k = dct_coeffs[k]

    if k == 0:
        return np.abs(d_k)
    else:
        s_k = dst_coeffs[k - 1]
        return np.sqrt(d_k**2 + s_k**2)


def fourier_phase(embeddings: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Compute Fourier phase of coefficient k.

    Phase = arctan2(s_k, d_k) = angle of d_k + i*s_k

    Phase tells you WHERE in the sequence the frequency pattern is strongest.

    Args:
        embeddings: (n_tokens, embedding_dim)
        k: Frequency index

    Returns:
        (embedding_dim,) phase angles in [-pi, pi]
    """
    n_tokens, emb_dim = embeddings.shape

    if n_tokens == 0 or n_tokens == 1:
        return np.zeros(emb_dim)

    n = np.arange(n_tokens)
    theta = np.pi * k * (2*n + 1) / (2 * n_tokens)

    cos_proj = embeddings.T @ np.cos(theta)
    sin_proj = embeddings.T @ np.sin(theta)

    return np.arctan2(sin_proj, cos_proj)


def fourier_complex(embeddings: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Compute full complex Fourier coefficient at frequency k.

    Returns d_k + i*s_k where:
    - d_k = DCT-like (cosine projection)
    - s_k = DST-like (sine projection)

    Use np.abs() for magnitude, np.angle() for phase.

    Args:
        embeddings: (n_tokens, embedding_dim)
        k: Frequency index

    Returns:
        (embedding_dim,) complex array
    """
    n_tokens, emb_dim = embeddings.shape

    if n_tokens == 0:
        return np.zeros(emb_dim, dtype=complex)

    if n_tokens == 1:
        if k == 0:
            return embeddings[0].astype(complex)
        return np.zeros(emb_dim, dtype=complex)

    n = np.arange(n_tokens)
    theta = np.pi * k * (2*n + 1) / (2 * n_tokens)

    w = np.exp(1j * theta)

    return np.sqrt(2 / n_tokens) * (embeddings.T @ w)
