"""
Circular Fourier encoding for RoPE dimension pairs.

Uses circular Fourier series (e^{ik*theta}) instead of DCT to decompose
angular waves across a token sequence. This is the native harmonic
basis for angles on S^1 -- it handles +/-pi wrapping correctly.

Pipeline:
    1. Convert each token to (angle, magnitude) per dimension pair
    2. Angle waves: decompose with circular Fourier (handles wrapping)
    3. Magnitude waves: decompose with standard DCT
    4. Combine into fixed-size encoding

The key insight: DCT treats angles as linear values and breaks at
+/-pi boundaries. Circular Fourier treats them as points on a circle,
where pi and -pi are the same point.
"""

from typing import Dict, Tuple
import numpy as np


def circular_fourier_coefficients(
    angles: np.ndarray,
    n_harmonics: int = 6,
) -> np.ndarray:
    """
    Compute circular Fourier coefficients of an angle sequence.

    For a sequence of angles theta_0, ..., theta_{n-1} on S^1:
        c_k = (1/n) sum_j exp(-i k theta_j)

    We store as (real, imag) pairs: each harmonic produces 2 values.

    Args:
        angles: (n_tokens, n_pairs) -- angles in [-pi, pi]
        n_harmonics: Number of harmonics (k = 0, 1, ..., n_harmonics-1)

    Returns:
        (n_harmonics * 2, n_pairs) -- real and imag parts interleaved
    """
    n_tokens, n_pairs = angles.shape
    coeffs = np.zeros((n_harmonics * 2, n_pairs))

    for k in range(n_harmonics):
        cos_part = np.cos(k * angles)
        sin_part = np.sin(k * angles)

        coeffs[2*k]     = cos_part.mean(axis=0)
        coeffs[2*k + 1] = -sin_part.mean(axis=0)

    return coeffs


def circular_fourier_reconstruct_angles(
    coeffs: np.ndarray,
    n_tokens: int,
    positions: np.ndarray = None,
) -> np.ndarray:
    """
    Reconstruct angles from circular Fourier coefficients.

    NOTE: This is approximate for the general case.

    Args:
        coeffs: (n_harmonics * 2, n_pairs)
        n_tokens: number of positions to reconstruct
        positions: optional position values (default: linspace)

    Returns:
        (n_tokens, n_pairs) reconstructed angles
    """
    n_harmonics = coeffs.shape[0] // 2
    n_pairs = coeffs.shape[1]

    if positions is None:
        positions = np.linspace(0, 2 * np.pi * (1 - 1/n_tokens), n_tokens)

    angles = np.zeros((n_tokens, n_pairs))
    for k in range(n_harmonics):
        re = coeffs[2*k]
        im = coeffs[2*k + 1]
        for j in range(n_tokens):
            angles[j] += re * np.cos(k * positions[j]) - im * np.sin(k * positions[j])

    return angles


def positional_circular_encode(
    angles: np.ndarray,
    n_harmonics: int = 6,
) -> np.ndarray:
    """
    Position-aware circular Fourier encoding.

    Instead of DCT on theta (which wraps),
    DCT on (cos theta, sin theta) (which is smooth and wrapping-free).

    Args:
        angles: (n_tokens, n_pairs) angles in [-pi, pi]
        n_harmonics: Number of positional frequency components

    Returns:
        (n_harmonics * 2, n_pairs) -- [cos_coeffs; sin_coeffs] per frequency
    """
    from scipy.fftpack import dct

    n_tokens, n_pairs = angles.shape

    cos_signal = np.cos(angles)
    sin_signal = np.sin(angles)

    K = min(n_harmonics, n_tokens)
    cos_dct = dct(cos_signal, type=2, axis=0, norm='ortho')
    sin_dct = dct(sin_signal, type=2, axis=0, norm='ortho')

    coeffs = np.zeros((n_harmonics * 2, n_pairs))
    coeffs[:K] = cos_dct[:K]
    coeffs[n_harmonics:n_harmonics + K] = sin_dct[:K]

    return coeffs


def positional_circular_reconstruct(
    coeffs: np.ndarray,
    n_tokens: int,
) -> np.ndarray:
    """
    Reconstruct angles from positional circular encoding.

    Args:
        coeffs: (n_harmonics * 2, n_pairs)
        n_tokens: sequence length

    Returns:
        (n_tokens, n_pairs) reconstructed angles
    """
    from scipy.fftpack import idct

    n_harmonics = coeffs.shape[0] // 2
    n_pairs = coeffs.shape[1]
    K = min(n_harmonics, n_tokens)

    cos_coeffs = np.zeros((n_tokens, n_pairs))
    sin_coeffs = np.zeros((n_tokens, n_pairs))
    cos_coeffs[:K] = coeffs[:K]
    sin_coeffs[:K] = coeffs[n_harmonics:n_harmonics + K]

    cos_signal = idct(cos_coeffs, type=2, axis=0, norm='ortho')
    sin_signal = idct(sin_coeffs, type=2, axis=0, norm='ortho')

    return np.arctan2(sin_signal, cos_signal)


# =============================================================================
# Full encoding pipeline
# =============================================================================

def to_pair_polar(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split into (angle, magnitude) per dimension pair."""
    n, d = embeddings.shape
    pairs = embeddings.reshape(n, d // 2, 2)
    angles = np.arctan2(pairs[:, :, 1], pairs[:, :, 0])
    magnitudes = np.sqrt(pairs[:, :, 0]**2 + pairs[:, :, 1]**2)
    return angles, magnitudes


def from_pair_polar(angles: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
    """Recover Cartesian from (angle, magnitude) per pair."""
    x = magnitudes * np.cos(angles)
    y = magnitudes * np.sin(angles)
    return np.stack([x, y], axis=-1).reshape(angles.shape[0], -1)


def circular_polar_encode(
    embeddings: np.ndarray,
    n_coefficients: int = 6,
) -> dict:
    """
    Circular-Fourier polar encoding.

    1. Split into (angle, magnitude) per dimension pair
    2. Angles: positional circular encoding (DCT on cos/sin)
    3. Magnitudes: standard DCT
    4. Pack into fixed output

    Args:
        embeddings: (n, d) -- d must be even
        n_coefficients: K

    Returns:
        encoding dict with components
    """
    from scipy.fftpack import dct

    n, d = embeddings.shape
    n_pairs = d // 2
    K = min(n_coefficients, n)

    angles, magnitudes = to_pair_polar(embeddings)

    cos_sig = np.cos(angles)
    sin_sig = np.sin(angles)
    cos_dct = dct(cos_sig, type=2, axis=0, norm='ortho')
    sin_dct = dct(sin_sig, type=2, axis=0, norm='ortho')

    mag_dct = dct(magnitudes, type=2, axis=0, norm='ortho')

    return {
        'angle_cos': cos_dct[:K],
        'angle_sin': sin_dct[:K],
        'magnitude': mag_dct[:K],
        'n_tokens': n,
        'hidden_dim': d,
        'K': n_coefficients,
    }


def circular_polar_reconstruct(enc: dict) -> np.ndarray:
    """
    Reconstruct embeddings from circular polar encoding.

    Args:
        enc: dict from circular_polar_encode

    Returns:
        (n_tokens, hidden_dim) reconstructed embeddings
    """
    from scipy.fftpack import idct

    n = enc['n_tokens']
    d = enc['hidden_dim']
    n_pairs = d // 2
    K = enc['K']
    Kn = min(K, n)

    cos_padded = np.zeros((n, n_pairs))
    sin_padded = np.zeros((n, n_pairs))
    mag_padded = np.zeros((n, n_pairs))
    cos_padded[:Kn] = enc['angle_cos'][:Kn]
    sin_padded[:Kn] = enc['angle_sin'][:Kn]
    mag_padded[:Kn] = enc['magnitude'][:Kn]

    cos_sig = idct(cos_padded, type=2, axis=0, norm='ortho')
    sin_sig = idct(sin_padded, type=2, axis=0, norm='ortho')
    magnitudes = idct(mag_padded, type=2, axis=0, norm='ortho')

    angles = np.arctan2(sin_sig, cos_sig)

    return from_pair_polar(angles, magnitudes)


def compare_all(
    embeddings: np.ndarray,
    n_coefficients: int = 6,
) -> Dict[str, float]:
    """
    Compare reconstruction error: Cartesian DCT vs Circular Polar.

    Args:
        embeddings: (n, d)
        n_coefficients: K

    Returns:
        Dict with RMSE per method and diagnostics
    """
    from scipy.fftpack import dct, idct

    n, d = embeddings.shape
    K = min(n_coefficients, n)
    results = {'n_tokens': n, 'dim': d, 'K': n_coefficients}

    # Cartesian DCT baseline
    cart = dct(embeddings, type=2, axis=0, norm='ortho')
    trunc = np.zeros_like(embeddings)
    trunc[:K] = cart[:K]
    recon = idct(trunc, type=2, axis=0, norm='ortho')
    results['cartesian_dct'] = float(np.sqrt(np.mean((embeddings - recon) ** 2)))

    # Circular polar
    if d % 2 == 0:
        enc = circular_polar_encode(embeddings, n_coefficients)
        recon_cp = circular_polar_reconstruct(enc)
        results['circular_polar'] = float(np.sqrt(np.mean((embeddings - recon_cp) ** 2)))

        # Naive polar DCT (for comparison)
        angles, mags = to_pair_polar(embeddings)
        ang_dct = dct(angles, type=2, axis=0, norm='ortho')
        mag_dct_v = dct(mags, type=2, axis=0, norm='ortho')
        ang_t = np.zeros_like(angles); ang_t[:K] = ang_dct[:K]
        mag_t = np.zeros_like(mags); mag_t[:K] = mag_dct_v[:K]
        ang_r = idct(ang_t, type=2, axis=0, norm='ortho')
        mag_r = idct(mag_t, type=2, axis=0, norm='ortho')
        recon_naive = from_pair_polar(ang_r, mag_r)
        results['naive_polar_dct'] = float(np.sqrt(np.mean((embeddings - recon_naive) ** 2)))

        # Smoothness diagnostics
        cos_sig = np.cos(angles)
        sin_sig = np.sin(angles)
        results['smoothness_cartesian'] = float(np.mean(np.abs(np.diff(embeddings, axis=0))))
        results['smoothness_angles'] = float(np.mean(np.abs(np.diff(angles, axis=0))))
        results['smoothness_cos_sin'] = float(np.mean(np.abs(np.diff(cos_sig, axis=0)) + np.abs(np.diff(sin_sig, axis=0))))
        results['smoothness_magnitudes'] = float(np.mean(np.abs(np.diff(mags, axis=0))))

    return results
