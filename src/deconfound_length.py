"""Issue #20: decompose the PRISM regime reversal (length vs label construct).

Part A - USS truncation: cut each dialogue to its first 7 turns (PRISM-matched
length), keep the full-dialogue satisfaction label, re-run the core arms.
If shape (g1to3, role-split advantage) collapses here too, LENGTH drives the
reversal; whatever survives at matched length is attributable to label
construct / domain.

Part B - PRISM length bins: median split by turn count, core arms per bin
(user-grouped CV within bin).
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

N_FOLDS, N_BOOT, SEED, TRUNC = 5, 1000, 42, 7

# Per-tag OOF prediction checkpoints (the RF fits are the ~1h part of a
# full-corpus run). Sampled runs bypass the cache, per lib/campaign.py.
_CKPT = Checkpoint("deconfound_length", script=__file__)
ckpt_load = _CKPT.load
ckpt_save = _CKPT.save


def truncate(emb, offsets, meta, k=TRUNC):
    """First-k-turns view of a corpus, preserving dialogue labels."""
    chunks, new_off, roles = [], [0], []
    for i in range(len(offsets) - 1):
        s, e = offsets[i], offsets[i + 1]
        n = min(k, e - s)
        chunks.append(emb[s:s + n])
        new_off.append(new_off[-1] + n)
        roles.append(meta["roles"][i][:n])
    m = dict(meta)
    m["roles"] = roles
    return np.vstack(chunks), np.asarray(new_off), m


def core_arms(emb, offsets, meta):
    arms = build_all_arms(emb, offsets, meta, degree=3)
    role_raw, _ = build_rolesplit(emb, offsets, meta)
    raw = arms["design_b_raw"]
    d = raw.shape[1] // 4
    return {"trace": arms["trace"], "raw": raw, "rolesplit": role_raw,
            "g0": raw[:, :d], "g1to3": raw[:, d:],
            "g0_plus_trace": np.hstack([raw[:, :d], arms["trace"]])}


def evaluate(tag, X, y, groups=None, sampled=False):
    cached = None if sampled else ckpt_load(tag)
    reused = cached is not None and all(k in cached for k in X)
    if reused:
        preds = {k: cached[k] for k in X}
        print(f"[{tag}] reusing checkpointed OOF predictions", flush=True)
    elif groups is None:
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
    if not reused and not sampled:
        ckpt_save(tag, preds)
    out = {}
    for gap in GAPS:
        g = {k: round(pairwise_accuracy(preds[k], y, gap), 4) for k in X}
        g["shape_vs_g0"] = bootstrap_diff(preds["g1to3"], preds["g0"], y, gap, N_BOOT, SEED)
        g["rolesplit_vs_raw"] = bootstrap_diff(preds["rolesplit"], preds["raw"], y, gap, N_BOOT, SEED)
        g["combo_vs_g0"] = bootstrap_diff(preds["g0_plus_trace"], preds["g0"], y, gap, N_BOOT, SEED)
        out[f"gap_{gap}"] = g
        row = " ".join(f"{k}={v}" for k, v in g.items() if isinstance(v, float))
        print(f"[{tag}] gap {gap}: {row}", flush=True)
        for c in ["shape_vs_g0", "rolesplit_vs_raw", "combo_vs_g0"]:
            v = g[c]
            print(f"    {c}: CI [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] p={v['p_boot']:.3f}",
                  flush=True)
    return out


def run(max_samples=None):
    results = {}
    # Part A: USS truncation
    for ds in ["uss_sgd", "uss_mwoz"]:
        emb, offsets, meta = load_corpus(ds)
        emb, offsets, meta = cap(emb, offsets, meta, max_samples)
        y = np.asarray(meta["overall_mean"], float)
        te, to, tm = truncate(emb, offsets, meta)
        results[f"{ds}_trunc{TRUNC}"] = evaluate(
            f"{ds}-trunc{TRUNC}", core_arms(te, to, tm), y,
            sampled=max_samples is not None)
    # Part B: PRISM length bins
    npz = np.load(WORK.parent / "data" / "embeddings" / "prism_minilm.npz")
    with open(WORK.parent / "data" / "embeddings" / "prism_meta.json") as f:
        meta = json.load(f)
    emb, offsets = npz["embeddings"], npz["offsets"]
    emb, offsets, meta = cap(emb, offsets, meta, max_samples)
    y_all = np.asarray(meta["overall_mean"], float)
    nt = np.asarray(meta["n_turns"])
    med = int(np.median(nt))
    for tag, mask in [("prism_short", nt <= med), ("prism_long", nt > med)]:
        idx = np.where(mask)[0]
        sub_off = [0]
        chunks, roles, users = [], [], []
        for i in idx:
            s, e = offsets[i], offsets[i + 1]
            chunks.append(emb[s:e])
            sub_off.append(sub_off[-1] + (e - s))
            roles.append(meta["roles"][i])
            users.append(meta["user_ids"][i])
        sub_meta = {"roles": roles}
        X = core_arms(np.vstack(chunks), np.asarray(sub_off), sub_meta)
        print(f"[{tag}] n={len(idx)} turn range {nt[mask].min()}-{nt[mask].max()}",
              flush=True)
        results[tag] = evaluate(tag, X, y_all[idx], groups=users,
                                sampled=max_samples is not None)
    if max_samples is None:
        from campaign_io import save_result
        save_result("deconfound_length", results, dataset="prism.uss_trunc",
                    config="minilm",
                    params={"seed": SEED, "n_folds": N_FOLDS, "trunc": TRUNC})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
