"""Combo arms: does the TRACE scalar zoo add anything on top of mean pooling?

Arms:
    g0_only          mean pooling (content channel)
    trace            26 TRACE-geometric scalars
    g0_plus_trace    [mean_pool | trace scalars]  (the fork-proposed combo)
    design_b_raw     G0..G3 vectors
    g0_plus_temporal (PRISM only) [mean_pool | trace | duration_per_turn]

Usage: g0_combo.py {uss_sgd|uss_mwoz|prism}
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
    if dataset == "prism":
        npz = np.load(WORK.parent / "data" / "embeddings" / "prism_minilm.npz")
        with open(WORK.parent / "data" / "embeddings" / "prism_meta.json") as f:
            meta = json.load(f)
        emb, offsets = npz["embeddings"], npz["offsets"]
        emb, offsets, meta = cap(emb, offsets, meta, max_samples)
        groups = meta["user_ids"]
    else:
        emb, offsets, meta = load_corpus(dataset)
        emb, offsets, meta = cap(emb, offsets, meta, max_samples)
        groups = None
    y = np.asarray(meta["overall_mean"], float)

    arms = build_all_arms(emb, offsets, meta, degree=3)
    raw = arms["design_b_raw"]
    d = raw.shape[1] // 4
    g0 = raw[:, :d]
    X = {
        "g0_only": g0,
        "trace": arms["trace"],
        "g0_plus_trace": np.hstack([g0, arms["trace"]]),
        "design_b_raw": raw,
    }
    if dataset == "prism" and "duration_per_turn" in meta:
        dur = np.asarray(meta["duration_per_turn"], float)[:, None]
        X["g0_plus_trace_temporal"] = np.hstack([g0, arms["trace"], dur])
        X["floor_temporal"] = dur

    if groups is None:
        preds = {k: oof_predictions(v, y, N_FOLDS, SEED) for k, v in X.items()}
    else:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GroupKFold
        def grouped(Xm):
            p = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=N_FOLDS).split(Xm, y, groups):
                rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                           random_state=SEED, n_jobs=-1)
                rf.fit(Xm[tr], y[tr])
                p[te] = rf.predict(Xm[te])
            return p
        preds = {k: grouped(v) for k, v in X.items()}

    out = {"dataset": dataset, "arms": list(X)}
    for gap in GAPS:
        g = {k: round(pairwise_accuracy(preds[k], y, gap), 4) for k in X}
        g["combo_vs_g0"] = bootstrap_diff(preds["g0_plus_trace"],
                                          preds["g0_only"], y, gap, N_BOOT, SEED)
        g["combo_vs_raw"] = bootstrap_diff(preds["g0_plus_trace"],
                                           preds["design_b_raw"], y, gap, N_BOOT, SEED)
        if "g0_plus_trace_temporal" in X:
            g["temporal_gain"] = bootstrap_diff(preds["g0_plus_trace_temporal"],
                                                preds["g0_plus_trace"], y, gap,
                                                N_BOOT, SEED)
        out[f"gap_{gap}"] = g
        print(f"=== {dataset} gap {gap} ===")
        for k in X:
            print(f"  {k:24s} {g[k]:.4f}")
        for c in ["combo_vs_g0", "combo_vs_raw", "temporal_gain"]:
            if c in g:
                v = g[c]
                print(f"  {c:24s} CI [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] p={v['p_boot']:.3f}")

    if max_samples is None:
        from campaign_io import save_result
        save_result(f"g0_combo_{dataset}", out, dataset=dataset, config="minilm",
                    params={"seed": SEED, "n_folds": N_FOLDS})
    return out


run_sample = make_run_sample(run)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    run(dataset=argv[0] if argv else "uss_sgd")
    return 0


if __name__ == "__main__":
    sys.exit(main())
