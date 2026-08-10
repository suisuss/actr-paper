"""Basis encoders used by the pipeline.

    - gram: Gram polynomials (orthogonal at uniform sample points)
    - projection: shared basis-projection helpers (DCT-II / DST-II bases)
    - dct, dst, fourier, circular_fourier: frequency bases
    - pooling: simple pooling baselines (mean, max, ...)
    - utils: shared helpers

Import submodules directly (from lib.encoders.gram import gram_fit).
"""
