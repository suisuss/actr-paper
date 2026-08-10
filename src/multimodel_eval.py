"""Issue #19: multi-model replication of the core arms (GTE-small, bge-base).

Per model tag and corpus: rebuild TRACE-geometric features and the projection
arms from that model's embeddings, evaluate at seed 42, and check the sign of
each headline comparison against the MiniLM conclusions.

PRISM: rolesplit excluded (not in the primary family there; bounds compute at
768-d). All arms use the existing meta jsons (offsets asserted at embed time).
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))

from lib.campaign import Checkpoint, cap, make_run_sample
from trace_h2h import (build_all_arms, oof_predictions, pairwise_accuracy,
                       bootstrap_diff, GAPS)
from extension_arms import build_rolesplit

N_FOLDS, N_BOOT, SEED = 5, 1000, 42
EMB = WORK.parent / "data" / "embeddings"
CKPT = WORK.parent / "data" / "experiment-caches" / "multimodel_eval"


_CKPT = Checkpoint("multimodel_eval", script=__file__)
ckpt_load = _CKPT.load
ckpt_save = _CKPT.save



def load(name, tag):
    npz = np.load(EMB / f"{name}_{tag}.npz")
    meta_file = ("prism_meta.json" if name == "prism"
                 else f"{name}_meta.json")
    with open(EMB / meta_file) as f:
        meta = json.load(f)
    groups = meta["user_ids"] if name == "prism" else None
    return npz["embeddings"], npz["offsets"], meta, groups


def oof(X, y, groups):
    if groups is None:
        return oof_predictions(X, y, N_FOLDS, SEED)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                   random_state=SEED, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        p[te] = rf.predict(X[te])
    return p


def run(max_samples=None):
    sampled = max_samples is not None
    if not sampled:
        CKPT.mkdir(parents=True, exist_ok=True)
    results = {}
    for tag in ["gte", "bge"]:
        results[tag] = {}
        for name in ["uss_sgd", "uss_mwoz", "prism"]:
            emb, offsets, meta, groups = load(name, tag)
            emb, offsets, meta = cap(emb, offsets, meta, max_samples)
            if groups is not None:
                groups = meta["user_ids"]
            y = np.asarray(meta["overall_mean"], float)
            base = build_all_arms(emb, offsets, meta, degree=3)
            raw = base["design_b_raw"]
            d = raw.shape[1] // 4
            arms = {"trace": base["trace"], "raw": raw, "g0": raw[:, :d],
                    "combo": np.hstack([raw[:, :d], base["trace"]])}
            if name != "prism":
                arms["rolesplit"], _ = build_rolesplit(emb, offsets, meta)
            ck = None if sampled else ckpt_load(f"{tag}_{name}")
            if ck is not None and all(k in ck for k in arms):
                preds = {k: ck[k] for k in arms}
                print(f"[{tag}/{name}] loaded from checkpoint", flush=True)
            else:
                preds = {k: oof(X, y, groups) for k, X in arms.items()}
                if not sampled:
                    ckpt_save(f"{tag}_{name}", preds)
            entry = {}
            for gap in GAPS:
                g = {k: round(pairwise_accuracy(p, y, gap), 4)
                     for k, p in preds.items()}
                g["raw_vs_trace"] = bootstrap_diff(preds["raw"], preds["trace"],
                                                   y, gap, N_BOOT, SEED)
                g["combo_vs_g0"] = bootstrap_diff(preds["combo"], preds["g0"],
                                                  y, gap, N_BOOT, SEED)
                if "rolesplit" in preds:
                    g["rolesplit_vs_raw"] = bootstrap_diff(
                        preds["rolesplit"], preds["raw"], y, gap, N_BOOT, SEED)
                entry[f"gap_{gap}"] = g
                row = " ".join(f"{k}={v}" for k, v in g.items()
                               if isinstance(v, float))
                print(f"[{tag}/{name}] gap {gap}: {row}", flush=True)
                for c in ["raw_vs_trace", "rolesplit_vs_raw", "combo_vs_g0"]:
                    if c in g:
                        v = g[c]
                        print(f"    {c}: CI [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] "
                              f"p={v['p_boot']:.3f}", flush=True)
            results[tag][name] = entry
    if not sampled:
        from campaign_io import save_result
        save_result("multimodel_eval", results,
                    dataset="uss_sgd.uss_mwoz.prism", config="gte_bge",
                    params={"seed": SEED, "n_folds": N_FOLDS})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
