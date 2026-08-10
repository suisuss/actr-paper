"""Causal per-role surprise on the role-split paths.

For each role path (user turns u_1..u_p, model turns m_1..m_q):
    pred_j   = degree-<=1 polynomial extrapolation fitted on the PREFIX
               (causal - no future turns)
    surprise_j = 1 - cos(pred_j, actual_j)          (trend-break surprise)
    naive_j    = 1 - cos(actual_{j-1}, actual_j)    (volatility control)

Per-dialogue features: mean/max surprise per role, mean bits per role,
surprise-minus-naive (does the trend predictor beat the static one), and the
Gram-0..3 expansion of each surprise series.

Outputs: descriptive Spearman vs satisfaction + marginal predictive check
vs the role-split block.
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))

from lib.campaign import cap, make_run_sample
from lib.encoders.gram import gram_fit
from trace_h2h import (load_corpus, build_all_arms, dialogue_iter,
                       oof_predictions, pairwise_accuracy, bootstrap_diff, GAPS)
from extension_arms import build_rolesplit

N_FOLDS, N_BOOT, SEED = 5, 1000, 42
D = 384


def causal_surprise(path):
    """path: (n, d) normalized. Returns (surprise, naive) series of len n-2."""
    n = len(path)
    if n < 3:
        return np.zeros(0), np.zeros(0)
    sur, nai = [], []
    for j in range(2, n):
        x = np.arange(j)
        deg = 1 if j >= 2 else 0
        coef = np.polyfit(x, path[:j], deg)          # (deg+1, d)
        pred = np.polyval(coef, j)
        pred = pred / (np.linalg.norm(pred) + 1e-12)
        sur.append(1.0 - float(pred @ path[j]))
        nai.append(1.0 - float(path[j - 1] @ path[j]))
    return np.asarray(sur), np.asarray(nai)


def build(emb, offsets, meta):
    scalars, series_feats = [], []
    for i, e, roles in dialogue_iter(emb, offsets, meta):
        en = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        u = en[[j for j, r in enumerate(roles) if r == "USER"]]
        m = en[[j for j, r in enumerate(roles) if r != "USER"]]
        su, nu = causal_surprise(u)
        sm, nm = causal_surprise(m)
        def stats(s, nctl):
            if len(s) == 0:
                return [0.0] * 4
            return [float(s.mean()), float(s.max()),
                    float(s.mean() - nctl.mean()),   # trend gain vs naive
                    float(D / 2 * np.log2(s.mean() + 1e-9))]  # ~bits scale
        scalars.append(stats(su, nu) + stats(sm, nm))
        feats = []
        for s in (su, sm):
            if len(s) == 0:
                feats.extend([0.0] * 4)
            else:
                feats.extend(gram_fit(s[:, None], degree=3)[:, 0].tolist())
        series_feats.append(feats)
    return np.asarray(scalars), np.asarray(series_feats)


def run(max_samples=None):
    from scipy import stats as st
    emb, offsets, meta = load_corpus("uss_sgd")
    emb, offsets, meta = cap(emb, offsets, meta, max_samples)
    y = np.asarray(meta["overall_mean"], float)
    S, F = build(emb, offsets, meta)
    names = ["user_sur_mean", "user_sur_max", "user_trend_gain", "user_bits",
             "model_sur_mean", "model_sur_max", "model_trend_gain", "model_bits"]
    print("descriptive Spearman vs satisfaction (n=%d):" % len(y))
    for j, nm in enumerate(names):
        r, p = st.spearmanr(S[:, j], y)
        print(f"  {nm:18s} rho={r:+.3f} p={p:.1e}")

    # marginal predictive check vs role-split block
    role_raw, _ = build_rolesplit(emb, offsets, meta)
    X = {"rolesplit": role_raw,
         "rolesplit_plus_surprise": np.hstack([role_raw, S, F])}
    preds = {k: oof_predictions(v, y, N_FOLDS, SEED) for k, v in X.items()}
    out = {"spearman": {nm: float(st.spearmanr(S[:, j], y)[0])
                        for j, nm in enumerate(names)}}
    for gap in GAPS:
        a = pairwise_accuracy(preds["rolesplit"], y, gap)
        b = pairwise_accuracy(preds["rolesplit_plus_surprise"], y, gap)
        bd = bootstrap_diff(preds["rolesplit_plus_surprise"], preds["rolesplit"],
                            y, gap, N_BOOT, SEED)
        out[f"gap_{gap}"] = {"rolesplit": round(a, 4), "plus_surprise": round(b, 4),
                             "diff": bd}
        print(f"gap {gap}: rolesplit={a:.4f} +surprise={b:.4f} "
              f"CI [{bd['ci_lo']:+.4f},{bd['ci_hi']:+.4f}] p={bd['p_boot']:.3f}")
    if max_samples is None:
        from campaign_io import save_result
        save_result("role_surprise_sgd", out, dataset="uss_sgd", config="minilm",
                    params={"seed": SEED, "n_folds": N_FOLDS})
    return out


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
