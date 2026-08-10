"""Spike 4 addendum: is design_b_raw geometry or content?

Ablates raw Gram coefficients into G0-only (mean embedding = content) vs
G1..G3-only (trajectory shape = geometry), and tests design_b_raw against the
n_turns floor directly.
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))

from lib.campaign import cap, make_run_sample
from trace_h2h import (load_corpus, build_all_arms, oof_predictions,
                       pairwise_accuracy, bootstrap_diff, GAPS)

N_FOLDS, N_BOOT, SEED = 5, 1000, 42


def run(dataset="uss_sgd", max_samples=None):
    emb, offsets, meta = load_corpus(dataset)
    emb, offsets, meta = cap(emb, offsets, meta, max_samples)
    y = np.asarray(meta["overall_mean"], float)
    arms = build_all_arms(emb, offsets, meta, degree=3)

    raw = arms["design_b_raw"]  # (n, 4*384) rows [G0 | G1 | G2 | G3]
    d = raw.shape[1] // 4
    X = {
        "floor1": arms["floor1"],
        "trace": arms["trace"],
        "g0_only": raw[:, :d],
        "g1to3_only": raw[:, d:],
        "design_b_raw": raw,
        "mean_pool": None,  # plain mean of turn embeddings (unnormalized)
    }
    # mean pooling baseline over raw (unnormalized) turn embeddings
    mp = np.vstack([emb[offsets[i]:offsets[i+1]].mean(axis=0)
                    for i in range(len(offsets)-1)])
    X["mean_pool"] = mp

    preds = {k: oof_predictions(v, y, N_FOLDS, SEED) for k, v in X.items()}

    out = {"dataset": dataset}
    for gap in GAPS:
        g = {}
        for k in X:
            g[k] = round(pairwise_accuracy(preds[k], y, gap), 4)
        g["g1to3_vs_g0"] = bootstrap_diff(preds["g1to3_only"], preds["g0_only"], y, gap, N_BOOT, SEED)
        g["raw_vs_g0"] = bootstrap_diff(preds["design_b_raw"], preds["g0_only"], y, gap, N_BOOT, SEED)
        g["raw_vs_floor1"] = bootstrap_diff(preds["design_b_raw"], preds["floor1"], y, gap, N_BOOT, SEED)
        g["raw_vs_meanpool"] = bootstrap_diff(preds["design_b_raw"], preds["mean_pool"], y, gap, N_BOOT, SEED)
        out[f"gap_{gap}"] = g

    if max_samples is None:
        from campaign_io import save_result
        save_result(f"feature_ablation_{dataset}", out, dataset=dataset, config="minilm",
                    params={"seed": SEED, "n_folds": N_FOLDS})

    for gap in GAPS:
        g = out[f"gap_{gap}"]
        print(f"=== gap {gap} ===")
        for k in X:
            print(f"  {k:14s} {g[k]:.4f}")
        for cmp in ["g1to3_vs_g0", "raw_vs_g0", "raw_vs_floor1", "raw_vs_meanpool"]:
            v = g[cmp]
            print(f"  {cmp:16s} CI [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] p={v['p_boot']:.3f}")
    return out


run_sample = make_run_sample(run)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    run(dataset=argv[0] if argv else "uss_sgd")
    return 0


if __name__ == "__main__":
    sys.exit(main())
