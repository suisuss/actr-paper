"""Spike 5: MDL/BIC adaptive degree selection prototype.

Two-part code per dialogue trajectory: choose degree k minimizing
    L(k) = elias_gamma_bits(k+1) + model_bits(k) + residual_bits(k)
with a Gaussian residual code:  n*d/2 * log2(RSS/(n*d)) + (k+1)*d/2 * log2(n).
(The (k+1)*d/2*log2(n) parameter cost is the BIC penalty in bits.)

Reports the selected-degree distribution and compares fixed-degree-3 raw Gram
features vs adaptive-degree (zero-padded to degree 6) on the preference task.
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))

from lib.campaign import cap, make_run_sample
from lib.encoders.gram import gram_fit, gram_basis_matrix_fast
from trace_h2h import (load_corpus, oof_predictions, pairwise_accuracy,
                       bootstrap_diff, GAPS)

MAX_DEGREE = 6


def elias_gamma_bits(n):
    return 2 * int(np.floor(np.log2(n))) + 1


def mdl_degree(en, max_degree=MAX_DEGREE):
    """Pick trajectory degree by two-part MDL. en: (n, d) normalized turns."""
    n, d = en.shape
    kmax = min(max_degree, n - 1)
    B = gram_basis_matrix_fast(n, kmax)
    coeffs = gram_fit(en, degree=kmax)
    best_k, best_L = 0, np.inf
    total = float(np.sum(en ** 2))
    for k in range(kmax + 1):
        recon = B[:, :k + 1] @ coeffs[:k + 1]
        rss = float(np.sum((en - recon) ** 2))
        var = max(rss / (n * d), 1e-12)
        L = (elias_gamma_bits(k + 1)
             + (k + 1) * d / 2 * np.log2(n)
             + n * d / 2 * np.log2(var))
        if L < best_L:
            best_L, best_k = L, k
    return best_k, coeffs


def run(dataset="uss_sgd", max_samples=None):
    emb, offsets, meta = load_corpus(dataset)
    emb, offsets, meta = cap(emb, offsets, meta, max_samples)
    y = np.asarray(meta["overall_mean"], float)
    n_dialogues = len(offsets) - 1

    fixed_rows, adaptive_rows, degrees = [], [], []
    for i in range(n_dialogues):
        e = emb[offsets[i]:offsets[i + 1]]
        en = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        k, coeffs_k = mdl_degree(en)
        degrees.append(k)
        fixed = gram_fit(en, degree=3)
        fixed_rows.append(fixed.ravel())
        # adaptive: keep MDL-selected degrees, zero-pad to MAX_DEGREE
        adaptive = np.zeros((MAX_DEGREE + 1, en.shape[1]))
        adaptive[:k + 1] = coeffs_k[:k + 1]
        adaptive_rows.append(adaptive.ravel())

    Xf = np.asarray(fixed_rows)
    Xa = np.asarray(adaptive_rows)
    dist = {int(k): int(v) for k, v in
            zip(*np.unique(degrees, return_counts=True))}
    print(f"[{dataset}] MDL-selected degree distribution: {dist}")

    pf = oof_predictions(Xf, y, 5, 42)
    pa = oof_predictions(Xa, y, 5, 42)
    out = {"dataset": dataset, "degree_distribution": dist}
    for gap in GAPS:
        af, aa = pairwise_accuracy(pf, y, gap), pairwise_accuracy(pa, y, gap)
        bd = bootstrap_diff(pa, pf, y, gap, 1000, 42)
        out[f"gap_{gap}"] = {"fixed3": round(af, 4), "adaptive_mdl": round(aa, 4),
                             "adaptive_vs_fixed": bd}
        print(f"  gap {gap}: fixed3={af:.4f} adaptive={aa:.4f} "
              f"CI [{bd['ci_lo']:+.4f},{bd['ci_hi']:+.4f}] p={bd['p_boot']:.3f}")

    if max_samples is None:
        from campaign_io import save_result
        save_result(f"mdl_degree_{dataset}", out, dataset=dataset,
                    config="minilm", params={"analysis": "mdl_degree"})
    return out


run_sample = make_run_sample(run)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    run(dataset=argv[0] if argv else "uss_sgd")
    return 0


if __name__ == "__main__":
    sys.exit(main())
