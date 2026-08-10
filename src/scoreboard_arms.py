"""Complete Table 1: five-seed replication of the remaining single-seed arms.

The flagship's Table 1 carries five legacy single-seed-42 arms (dagger rows):
turn-count floor, projected TRACE series, scalarized coefficients, pooled
scalars, and the PRISM-only [mean pooling | TRACE | duration/turn]. This
script re-runs them under the standard multi-seed protocol so the dagger
convention can be retired.

Protocol (identical to multiseed_probes.py / rolesplit_graft):
    seeds {42, 7, 123, 2024, 31337}; RandomForestRegressor(300 trees,
    min_samples_leaf=2); 5-fold OOF (KFold shuffled by seed; GroupKFold by
    user on PRISM); pairwise preference accuracy at gaps 0.5 and 1.0.

Arms per corpus:
    uss_sgd / uss_mwoz: floor1, design_a, design_b, hybrid
    prism:              the same four, plus combo_duration =
                        [g0 | trace | duration_per_turn] (411-d)
    (hybrid on PRISM fills a cell the legacy campaign never ran)

Checkpoints: data/experiment-caches/scoreboard_arms/<corpus>_seed<seed>.npz
Registry: campaign_io.save_result("scoreboard_arms", ...)
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "scripts"))

from lib.campaign import Checkpoint, cap, make_run_sample
from trace_h2h import load_corpus, build_all_arms, pairwise_accuracy, GAPS

SEEDS = [42, 7, 123, 2024, 31337]
N_FOLDS = 5

CKPT = WORK.parent / "data" / "experiment-caches" / "scoreboard_arms"


_CKPT = Checkpoint("scoreboard_arms", script=__file__)
ckpt_load = _CKPT.load
ckpt_save = _CKPT.save


def load(name):
    if name == "prism":
        npz = np.load(WORK.parent / "data" / "embeddings" / "prism_minilm.npz")
        with open(WORK.parent / "data" / "embeddings" / "prism_meta.json") as f:
            meta = json.load(f)
        return npz["embeddings"], npz["offsets"], meta, meta["user_ids"]
    emb, offsets, meta = load_corpus(name)
    return emb, offsets, meta, None


def oof(X, y, seed, groups):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, GroupKFold
    preds = np.zeros(len(y))
    if groups is None:
        splits = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(X)
    else:
        splits = GroupKFold(n_splits=N_FOLDS).split(X, y, groups)
    for tr, te in splits:
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                  random_state=seed, n_jobs=-1)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return preds


def run(max_samples=None):
    sampled = max_samples is not None
    if not sampled:
        CKPT.mkdir(parents=True, exist_ok=True)
    results = {"seeds": SEEDS, "protocol": "multiseed_probes-identical"}
    for name in ["uss_sgd", "uss_mwoz", "prism"]:
        emb, offsets, meta, groups = load(name)
        emb, offsets, meta = cap(emb, offsets, meta, max_samples)
        if groups is not None:
            groups = meta["user_ids"]
        y = np.asarray(meta["overall_mean"], float)
        base = build_all_arms(emb, offsets, meta, degree=3)
        arms = {"floor1": base["floor1"], "design_a": base["design_a"],
                "design_b": base["design_b"], "hybrid": base["hybrid"]}
        if name == "prism" and "duration_per_turn" in meta:
            raw = base["design_b_raw"]
            g0 = raw[:, :raw.shape[1] // 4]
            dur = np.asarray(meta["duration_per_turn"], float)[:, None]
            arms["combo_duration"] = np.hstack([g0, base["trace"], dur])
        dims = {k: int(v.shape[1]) for k, v in arms.items()}
        print(f"[{name}] arms: {dims}", flush=True)

        seed_accs = {k: [] for k in arms}
        for seed in SEEDS:
            tag = f"{name}_seed{seed}"
            cached = None if sampled else ckpt_load(tag)
            if cached is not None and all(k in cached for k in arms):
                preds_this = {k: cached[k] for k in arms}
                print(f"[{name}] seed {seed} loaded from checkpoint", flush=True)
            elif sampled:
                preds_this = {k: oof(X, y, seed, groups) for k, X in arms.items()}
                print(f"[{name}] seed {seed} done", flush=True)
            else:
                preds_this = {k: oof(X, y, seed, groups) for k, X in arms.items()}
                ckpt_save(tag, preds_this)
                print(f"[{name}] seed {seed} done (checkpointed)", flush=True)
            for k in arms:
                seed_accs[k].append({f"gap_{gap}": pairwise_accuracy(
                    preds_this[k], y, gap) for gap in GAPS})
        ms = {}
        for k, accs in seed_accs.items():
            ms[k] = {f"gap_{gap}": {
                "mean": round(float(np.mean([a[f"gap_{gap}"] for a in accs])), 4),
                "std": round(float(np.std([a[f"gap_{gap}"] for a in accs])), 4)}
                for gap in GAPS}
            print(f"[{name}] {k:15s} " + " ".join(
                f"gap{gap}={ms[k][f'gap_{gap}']['mean']:.4f}+-"
                f"{ms[k][f'gap_{gap}']['std']:.4f}" for gap in GAPS), flush=True)
        results[name] = {"multiseed": ms, "dims": dims}

    if not sampled:
        from campaign_io import save_result
        save_result("scoreboard_arms", results,
                    dataset="uss_sgd.uss_mwoz.prism", config="minilm",
                    params={"seeds": SEEDS, "n_folds": N_FOLDS,
                            "arms": ["floor1", "design_a", "design_b", "hybrid",
                                     "combo_duration(prism)"]})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
