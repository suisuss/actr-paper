"""Spike 6: PRISM generalization with user-grouped CV.

First-person labels (helpfulness 0-100 rescaled to 1-5), goal = opening prompt
(= first user turn), groups = user_id. GroupKFold(5) approximates TRACE's
leave-one-user-out at tractable cost (8k conversations, 1.5k users).

Also reports within-user pairwise accuracy (mean/std across users), mirroring
TRACE's variance-across-users comparison.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))

from lib.campaign import cap, make_run_sample
from trace_h2h import (build_all_arms, pairwise_accuracy, bootstrap_diff, GAPS)

N_BOOT, SEED = 1000, 42


def grouped_oof(X, y, groups, seed=SEED, n_folds=5):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    preds = np.zeros(len(y))
    gkf = GroupKFold(n_splits=n_folds)
    for tr, te in gkf.split(X, y, groups):
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                   random_state=seed, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        preds[te] = rf.predict(X[te])
    return preds


def within_user_accuracy(preds, y, user_ids, gap):
    """Pairwise accuracy computed only within each user's own conversations."""
    by_user = defaultdict(list)
    for i, u in enumerate(user_ids):
        by_user[u].append(i)
    accs = []
    for u, ix in by_user.items():
        if len(ix) < 2:
            continue
        correct = total = 0
        for a in range(len(ix)):
            for b in range(a + 1, len(ix)):
                i, j = ix[a], ix[b]
                if abs(y[i] - y[j]) >= gap:
                    total += 1
                    if (preds[i] - preds[j]) * (y[i] - y[j]) > 0:
                        correct += 1
        if total > 0:
            accs.append(correct / total)
    return {"mean": round(float(np.mean(accs)), 4),
            "std": round(float(np.std(accs)), 4),
            "n_users": len(accs)}


def run(max_samples=None):
    npz = np.load(WORK.parent / "data" / "embeddings" / "prism_minilm.npz")
    with open(WORK.parent / "data" / "embeddings" / "prism_meta.json") as f:
        meta = json.load(f)
    emb, offsets = npz["embeddings"], npz["offsets"]
    emb, offsets, meta = cap(emb, offsets, meta, max_samples)
    y = np.asarray(meta["overall_mean"], float)
    users = meta["user_ids"]
    print(f"PRISM: {len(y)} conversations, {len(set(users))} users, "
          f"label range [{y.min():.2f}, {y.max():.2f}]")
    nt = np.asarray(meta["n_turns"])
    print(f"turns: min {nt.min()} mean {nt.mean():.1f} max {nt.max()} | "
          f"convs with < 4 turns: {(nt < 4).sum()}")

    arms = build_all_arms(emb, offsets, meta, degree=3)
    raw = arms["design_b_raw"]
    d = raw.shape[1] // 4
    X = {"floor1": arms["floor1"], "trace": arms["trace"],
         "design_a": arms["design_a"], "design_b": arms["design_b"],
         "design_b_raw": raw, "g0_only": raw[:, :d], "g1to3_only": raw[:, d:]}

    preds = {k: grouped_oof(v, y, users) for k, v in X.items()}
    out = {"n_conversations": len(y), "n_users": len(set(users)),
           "cv": "GroupKFold(5) by user_id",
           "label": "performance_attributes.helpfulness rescaled 1-5"}
    for gap in GAPS:
        g = {}
        for k in X:
            g[k] = {"acc": round(pairwise_accuracy(preds[k], y, gap), 4),
                    "within_user": within_user_accuracy(preds[k], y, users, gap)}
        for k in ["design_a", "design_b", "design_b_raw", "g0_only",
                  "g1to3_only", "floor1"]:
            g[k]["vs_trace"] = bootstrap_diff(preds[k], preds["trace"], y, gap,
                                              N_BOOT, SEED)
        g["raw_vs_g0"] = bootstrap_diff(preds["design_b_raw"], preds["g0_only"],
                                        y, gap, N_BOOT, SEED)
        out[f"gap_{gap}"] = g
        print(f"=== gap {gap} ===")
        for k in X:
            v = g[k]
            sig = ""
            if "vs_trace" in v:
                p = v["vs_trace"]["p_boot"]
                sig = f"  vs_trace p={p:.3f}"
            print(f"  {k:14s} acc={v['acc']:.4f} within-user "
                  f"{v['within_user']['mean']:.4f}±{v['within_user']['std']:.4f}{sig}")

    if max_samples is None:
        from campaign_io import save_result
        save_result("prism_reversal", out, dataset="prism", config="minilm",
                    params={"seed": SEED, "n_boot": N_BOOT})
    return out


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
