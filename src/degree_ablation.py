"""Issue #30: role-split degree ablation - smallest K preserving the advantage.

Pure slicing: the (degree+1, d) per-role blocks are prefix-truncatable, so
K = 0..3 arms are column subsets of the existing role-split matrix.
Runs on all three corpora vs the interleaved raw block and TRACE baselines.

Checkpoints: per-corpus OOF predictions under
experiment-caches/degree_ablation/<corpus>.npz (the 5-arm x n_dialogues
vectors are the expensive part - ~1.7h full-corpus). Sampled runs
(max_samples) bypass the cache entirely and never persist, per the
lib/campaign.py contract.
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))

from lib.campaign import Checkpoint, cap, make_run_sample
from trace_h2h import (load_corpus, build_all_arms, oof_predictions,
                       pairwise_accuracy, bootstrap_diff, GAPS)
from extension_arms import build_rolesplit

N_FOLDS, N_BOOT, SEED, D = 5, 1000, 42, 384

_CKPT = Checkpoint("degree_ablation", script=__file__)
ckpt_load = _CKPT.load
ckpt_save = _CKPT.save


def role_k_slice(role_raw, K):
    """role_raw rows = [gu0..gu3 | gm0..gm3] flattened (4,384) each."""
    cols = []
    for base in (0, 4 * D):
        cols.extend(range(base, base + (K + 1) * D))
    return role_raw[:, cols]


def run_corpus(name, max_samples=None):
    sampled = max_samples is not None
    if name == "prism":
        npz = np.load(WORK.parent / "data" / "embeddings" / "prism_minilm.npz")
        with open(WORK.parent / "data" / "embeddings" / "prism_meta.json") as f:
            meta = json.load(f)
        emb, offsets = npz["embeddings"], npz["offsets"]
        emb, offsets, meta = cap(emb, offsets, meta, max_samples)
        groups = meta["user_ids"]
    else:
        emb, offsets, meta = load_corpus(name)
        emb, offsets, meta = cap(emb, offsets, meta, max_samples)
        groups = None
    y = np.asarray(meta["overall_mean"], float)

    def oof(X):
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

    arm_names = ["interleaved_raw"] + [f"rolesplit_K{K}" for K in range(4)]
    cached = None if sampled else ckpt_load(name)
    if cached is not None and all(k in cached for k in arm_names):
        preds = {k: cached[k] for k in arm_names}
        print(f"[{name}] reusing checkpointed OOF predictions", flush=True)
    else:
        raw = build_all_arms(emb, offsets, meta, degree=3)["design_b_raw"]
        role_raw, _ = build_rolesplit(emb, offsets, meta)
        preds = {"interleaved_raw": oof(raw)}
        for K in range(4):
            preds[f"rolesplit_K{K}"] = oof(role_k_slice(role_raw, K))
        if not sampled:
            ckpt_save(name, preds)

    out = {}
    for gap in GAPS:
        g = {k: round(pairwise_accuracy(p, y, gap), 4) for k, p in preds.items()}
        for K in range(3):
            g[f"K{K}_vs_K3"] = bootstrap_diff(preds[f"rolesplit_K{K}"],
                                              preds["rolesplit_K3"], y, gap,
                                              N_BOOT, SEED)
        g["K3_vs_raw"] = bootstrap_diff(preds["rolesplit_K3"],
                                        preds["interleaved_raw"], y, gap,
                                        N_BOOT, SEED)
        out[f"gap_{gap}"] = g
        row = " ".join(f"{k}={v}" for k, v in g.items() if isinstance(v, float))
        print(f"[{name}] gap {gap}: {row}", flush=True)
        for K in range(3):
            v = g[f"K{K}_vs_K3"]
            print(f"    K{K}_vs_K3: CI [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] "
                  f"p={v['p_boot']:.3f}", flush=True)
    return out


def run(max_samples=None):
    results = {ds: run_corpus(ds, max_samples)
               for ds in ["uss_sgd", "uss_mwoz", "prism"]}
    if max_samples is None:
        from campaign_io import save_result
        save_result("degree_ablation", results, dataset="uss_sgd.uss_mwoz",
                    config="minilm", params={"seed": SEED, "n_folds": N_FOLDS})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
