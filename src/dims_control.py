"""Issue #22: dimensionality-matched control.

Does the projection advantage survive at TRACE's feature count (26 dims)?
PCA and random projection are fit INSIDE each CV fold (train-only) to avoid
transduction. Arms: trace (26), raw->PCA26, rolesplit->PCA26, raw->RP26.
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))

from lib.campaign import cap, make_run_sample
from trace_h2h import (load_corpus, build_all_arms, pairwise_accuracy,
                       bootstrap_diff, GAPS)
from extension_arms import build_rolesplit

N_FOLDS, N_BOOT, SEED, K = 5, 1000, 42, 26


def oof_reduced(X, y, reducer, seed=SEED):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    preds = np.zeros(len(y))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        red = reducer()
        Xtr = red.fit_transform(X[tr])
        Xte = red.transform(X[te])
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                   random_state=seed, n_jobs=-1)
        rf.fit(Xtr, y[tr])
        preds[te] = rf.predict(Xte)
    return preds


def run(max_samples=None):
    from sklearn.decomposition import PCA
    from sklearn.random_projection import GaussianRandomProjection
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    results = {}
    for ds in ["uss_sgd", "uss_mwoz"]:
        emb, offsets, meta = load_corpus(ds)
        emb, offsets, meta = cap(emb, offsets, meta, max_samples)
        y = np.asarray(meta["overall_mean"], float)
        arms = build_all_arms(emb, offsets, meta, degree=3)
        role_raw, _ = build_rolesplit(emb, offsets, meta)

        # trace baseline (no reduction)
        preds_trace = np.zeros(len(y))
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for tr, te in kf.split(arms["trace"]):
            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                       random_state=SEED, n_jobs=-1)
            rf.fit(arms["trace"][tr], y[tr])
            preds_trace[te] = rf.predict(arms["trace"][te])

        reduced = {
            "raw_pca26": (arms["design_b_raw"], lambda: PCA(n_components=K, random_state=SEED)),
            "rolesplit_pca26": (role_raw, lambda: PCA(n_components=K, random_state=SEED)),
            "raw_rp26": (arms["design_b_raw"],
                         lambda: GaussianRandomProjection(n_components=K, random_state=SEED)),
        }
        entry = {}
        for name, (X, mk) in reduced.items():
            p = oof_reduced(X, y, mk)
            e = {}
            for gap in GAPS:
                e[f"gap_{gap}"] = {
                    "acc": round(pairwise_accuracy(p, y, gap), 4),
                    "trace": round(pairwise_accuracy(preds_trace, y, gap), 4),
                    "vs_trace": bootstrap_diff(p, preds_trace, y, gap, N_BOOT, SEED),
                }
            entry[name] = e
            for gap in GAPS:
                v = e[f"gap_{gap}"]["vs_trace"]
                print(f"[{ds}] {name:16s} gap {gap}: acc={e[f'gap_{gap}']['acc']:.4f} "
                      f"trace={e[f'gap_{gap}']['trace']:.4f} "
                      f"CI [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] p={v['p_boot']:.3f}",
                      flush=True)
        results[ds] = entry
    if max_samples is None:
        from campaign_io import save_result
        save_result("dims_control", results, dataset="uss_sgd.uss_mwoz",
                    config="minilm",
                    params={"seed": SEED, "n_folds": N_FOLDS, "pca_k": K})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
