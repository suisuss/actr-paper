"""
Canonical orthogonal-basis projection for sequence representation.

SINGLE source of truth for turning a length-n sequence of embeddings into
fixed-size basis coefficients by least-squares projection: each coefficient
is the sequence's projection onto one basis column, normalized by that
column's squared norm. This mirrors ``lib/encoders/gram.py::gram_fit``
exactly, so across every basis in this family the degree-0 coefficient is the
arithmetic mean of the sequence -- d0 (DCT) = g0 (Gram) = mean, EXACTLY, not
up to a length-dependent scale -- and the higher coefficients are the
least-squares amplitudes (length-comparable across sequences of different n).

This is the convention the papers use (the Gram/DCT head-to-head and the
basis study). ``lib/encoders/dct.py`` and ``lib/encoders/dst.py`` build their
feature coefficients through this module, and the paper scripts import the
basis functions from here, so there is exactly one implementation.

It is DELIBERATELY DISTINCT from the orthonormal ``scipy.fftpack.dct/dst
(norm='ortho')`` transform used by the reconstruction encoders
(``rope.py``, ``circular_fourier.py``, ``hyperspherical.py``), which need a
norm-preserving invertible round-trip. Do NOT substitute one for the other:
the orthonormal transform's d0 is sqrt(n)*mean (length-dependent), which
breaks both cross-length comparability and the d0 = mean identity. See
``tests/test_projection_consistency.py`` for the guard that pins this.
"""

import numpy as np


def dct2_basis(n, ncoef):
    """(n, min(ncoef, n)) DCT-II columns cos(pi k (i+1/2)/n), k = 0..;
    exactly orthogonal on the uniform grid; column k=0 is the constant, so
    its projection coefficient is the arithmetic mean."""
    k = np.arange(min(ncoef, n))
    i = np.arange(n)
    return np.cos(np.pi * np.outer(i + 0.5, k) / n)


def dst2_basis(n, ncoef):
    """(n, min(ncoef, n)) DST-II columns sin(pi j (i+1/2)/n), j = 1..;
    exactly orthogonal on the uniform grid; contains NO constant term. Note
    the columns start at harmonic j=1: column index c corresponds to
    harmonic j = c + 1 (so the library's DST component s{c} is the paper's
    s{c+1}); there is no DC/mean column."""
    j = np.arange(1, min(ncoef, n) + 1)
    i = np.arange(n)
    return np.sin(np.pi * np.outer(i + 0.5, j) / n)


def basis_fit(x, basis_fn, ncoef):
    """Least-squares projection of x (n, d) onto orthogonal-on-grid columns,
    normalized by column norm^2 (mirrors gram_fit), zero-padded to ncoef rows.

    Returns (ncoef, d). For the constant column (dct2_basis, k=0) the
    coefficient equals the arithmetic mean of x exactly (the 1/n from the
    column norm^2 cancels the sum), independent of sequence length n."""
    coeffs = np.zeros((ncoef, x.shape[1]))
    n = len(x)
    if n == 0:
        return coeffs
    B = basis_fn(n, ncoef)
    norms_sq = np.sum(B ** 2, axis=0)
    for k in range(B.shape[1]):
        if norms_sq[k] > 1e-10:
            coeffs[k] = (B[:, k] @ x) / norms_sq[k]
    return coeffs
