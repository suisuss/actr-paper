"""Which TRACE features are strong, are they contained in the Gram block,
and can we get the best of both worlds?

Analyses per dataset:
1. STRENGTH  - univariate pairwise accuracy per TRACE feature + RF permutation
               importance (5-fold OOF).
2. CONTAINMENT - OOF R^2 of ridge regression predicting each TRACE feature
               from the design_b_raw block (G0..G3). High R^2 = the projection
               already carries the feature; low R^2 + high strength =
               constructive candidate.
3. BEST-OF-BOTH arms:
     - order_stats: the max/min family from the TRACE set (known
       non-polynomial functionals)
     - residual_extremes: extremes of the degree-3 reconstruction residual -
       orthogonal to the projection by construction (zero double-counting)
     - [design_b_raw | order_stats], [design_b_raw | residual_extremes],
       [design_b_raw | trace]
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))

from lib.campaign import cap, make_run_sample
from lib.trace_features import FEATURE_NAMES
from lib.encoders.gram import gram_fit, gram_basis_matrix_fast
from trace_h2h import (load_corpus, build_all_arms, dialogue_iter,
                       oof_predictions, pairwise_accuracy, bootstrap_diff, GAPS)

N_FOLDS, N_BOOT, SEED = 5, 500, 42

ORDER_STAT_FEATURES = [
    "model_self_sim_max", "model_user_dist_max", "model_user_dist_min",
    "user_model_dist_max", "turn_dist_max", "model_goal_dist_min",
    "model_goal_dist_max",
]


def build_residual_extremes(emb, offsets, meta, degree=3):
    """Extremes of the part of the trajectory the projection discarded."""
    rows = []
    for i, e, roles in dialogue_iter(emb, offsets, meta):
        en = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        n = len(en)
        K = min(degree, n - 1)
        B = gram_basis_matrix_fast(n, K)
        coeffs = gram_fit(en, degree=K)
        resid = en - B @ coeffs[:K + 1]
        norms = np.linalg.norm(resid, axis=1)
        total = float(np.linalg.norm(en))
        rows.append([
            float(norms.max()),
            float(norms.mean()),
            float(np.argmax(norms)) / max(n - 1, 1),   # where the worst moment is
            float(norms[-1]),                          # rough ending?
            float(norms.max() / (total + 1e-12)),
        ])
    return np.asarray(rows, float)


def univariate_strength(trace, y, gap):
    out = {}
    for j, name in enumerate(FEATURE_NAMES):
        acc = pairwise_accuracy(trace[:, j], y, gap)
        out[name] = round(max(acc, 1 - acc), 4)  # orientation-free
    return out


def permutation_importance_oof(trace, y, gap, seed=SEED):
    """Permutation importance measured in pairwise accuracy, OOF."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    rng = np.random.default_rng(seed)
    base_preds = oof_predictions(trace, y, N_FOLDS, seed)
    base = pairwise_accuracy(base_preds, y, gap)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    drops = {name: [] for name in FEATURE_NAMES}
    models, splits = [], []
    for tr, te in kf.split(trace):
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                   random_state=seed, n_jobs=-1)
        rf.fit(trace[tr], y[tr])
        models.append(rf)
        splits.append(te)
    for j, name in enumerate(FEATURE_NAMES):
        for rep in range(3):
            preds = np.zeros(len(y))
            for rf, te in zip(models, splits):
                Xp = trace[te].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                preds[te] = rf.predict(Xp)
            drops[name].append(base - pairwise_accuracy(preds, y, gap))
    return {k: round(float(np.mean(v)), 4) for k, v in drops.items()}, round(base, 4)


def containment(trace, raw, seed=SEED):
    """OOF R^2 of ridge predicting each TRACE feature from the Gram block."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    out = {}
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for j, name in enumerate(FEATURE_NAMES):
        t = trace[:, j]
        if np.std(t) < 1e-12:
            out[name] = None  # constant (goal-proxy degeneracy)
            continue
        preds = np.zeros(len(t))
        for tr, te in kf.split(raw):
            m = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
            m.fit(raw[tr], t[tr])
            preds[te] = m.predict(raw[te])
        out[name] = round(float(r2_score(t, preds)), 3)
    return out


def run(dataset="uss_sgd", max_samples=None):
    emb, offsets, meta = load_corpus(dataset)
    emb, offsets, meta = cap(emb, offsets, meta, max_samples)
    y = np.asarray(meta["overall_mean"], float)
    arms = build_all_arms(emb, offsets, meta, degree=3)
    trace, raw = arms["trace"], arms["design_b_raw"]
    resid_x = build_residual_extremes(emb, offsets, meta)
    order_ix = [FEATURE_NAMES.index(k) for k in ORDER_STAT_FEATURES]

    print(f"\n================ {dataset} ================")
    uni = univariate_strength(trace, y, 1.0)
    perm, base = permutation_importance_oof(trace, y, 1.0)
    cont = containment(trace, raw)

    print(f"{'feature':28s} {'uni_acc':>8s} {'perm_drop':>9s} {'R2_in_gram':>10s}")
    ranked = sorted(FEATURE_NAMES, key=lambda k: -perm[k])
    for name in ranked:
        c = cont[name]
        print(f"{name:28s} {uni[name]:8.4f} {perm[name]:9.4f} "
              f"{'const' if c is None else f'{c:10.3f}'}")

    X = {
        "trace": trace,
        "design_b_raw": raw,
        "raw_plus_orderstats": np.hstack([raw, trace[:, order_ix]]),
        "raw_plus_resid_extremes": np.hstack([raw, resid_x]),
        "raw_plus_trace": np.hstack([raw, trace]),
        "resid_extremes_only": resid_x,
    }
    preds = {k: oof_predictions(v, y, N_FOLDS, SEED) for k, v in X.items()}
    out = {"dataset": dataset, "univariate": uni, "perm_importance": perm,
           "perm_base_acc": base, "containment_r2": cont}
    for gap in GAPS:
        g = {k: round(pairwise_accuracy(preds[k], y, gap), 4) for k in X}
        for k in ["raw_plus_orderstats", "raw_plus_resid_extremes", "raw_plus_trace"]:
            g[k + "_vs_raw"] = bootstrap_diff(preds[k], preds["design_b_raw"],
                                              y, gap, N_BOOT, SEED)
        out[f"gap_{gap}"] = g
        print(f"--- gap {gap} ---")
        for k in X:
            print(f"  {k:26s} {g[k]:.4f}")
        for k in ["raw_plus_orderstats", "raw_plus_resid_extremes", "raw_plus_trace"]:
            v = g[k + "_vs_raw"]
            print(f"  {k:26s} vs raw: CI [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] p={v['p_boot']:.3f}")

    if max_samples is None:
        from campaign_io import save_result
        save_result(f"containment_graft_{dataset}", out, dataset=dataset,
                    config="minilm", params={"seed": SEED, "n_folds": N_FOLDS})
    return out


run_sample = make_run_sample(run)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    for ds in (argv or ["uss_sgd", "uss_mwoz"]):
        run(dataset=ds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
