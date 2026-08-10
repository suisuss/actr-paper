"""Issue #63: the two cheap defenses from the codex assessment v2.

1. Uniqueness contrast (assessment 3.4): role-split(1) vs mean-pooling+TRACE
   (the combo) on USS-SGD, seed-ensembled OOF predictions loaded from the
   rolesplit_graft checkpoint caches, dialogue-resampling bootstrap at gaps 0.5/1.0.
   Decides whether the never-distinguishably-worse property is unique to the
   split or shared by the combo.
2. PCA-9 / random-9 capacity control (assessment 3.2): the scalarized-
   coefficients arm (9 hand-crafted norms/angles) confounds direction loss
   with a ~170x capacity drop from the 1,536-d block. Project the same
   interleaved coefficient block to 9 dims per fold (train-only PCA and
   Gaussian random projection), same protocol; if 9 projected dims recover
   what the 9 hand-crafted scalars lose, the deficit is the scalarization,
   not the dimensionality.
3. Scalarized-vs-TRACE, seed-ensembled: the scalarization-losses table's
   "worse than TRACE" p on USS-SGD previously traced to the single-seed
   designs run, against the Section 5.2 rule that inferential claims are
   seed-ensembled. Recorded here from the same ensembles the other
   contrasts load (design_b from scoreboard_arms, trace from
   multiseed_probes).

Protocol: seeds {42, 7, 123, 2024, 31337}; RandomForestRegressor(300,
min_samples_leaf=2); 5-fold OOF; contrasts seed-ensembled via bootstrap_diff.
Checkpoints: data/experiment-caches/capacity_uniqueness/. Registry: campaign_io.
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "scripts"))

from lib.campaign import Checkpoint, make_run_sample
from trace_h2h import (load_corpus, build_all_arms, pairwise_accuracy,
                       bootstrap_diff, GAPS)

SEEDS = [42, 7, 123, 2024, 31337]
N_FOLDS, N_BOOT, K = 5, 1000, 9

CKPT = WORK.parent / "data" / "experiment-caches" / "capacity_uniqueness"


def ensemble(cache_dir, corpus, arm):
    ck = Checkpoint(cache_dir)
    preds = [ck.load(f"{corpus}_seed{s}")[arm] for s in SEEDS]
    return np.mean(preds, axis=0)


def uniqueness_contrast():
    _, _, meta = load_corpus("uss_sgd")
    y = np.asarray(meta["overall_mean"], float)
    rs1 = ensemble("rolesplit_graft", "uss_sgd", "rolesplit_k1")
    combo = ensemble("rolesplit_graft", "uss_sgd", "combo")
    out = {}
    for gap in GAPS:
        bd = bootstrap_diff(rs1, combo, y, gap, N_BOOT, 42)
        out[f"gap_{gap}"] = {
            "rs1_acc": round(pairwise_accuracy(rs1, y, gap), 4),
            "combo_acc": round(pairwise_accuracy(combo, y, gap), 4),
            "diff": round(pairwise_accuracy(rs1, y, gap)
                          - pairwise_accuracy(combo, y, gap), 4),
            "ci": [bd["ci_lo"], bd["ci_hi"]], "p_boot": bd["p_boot"]}
        print(f"[uniqueness sgd] gap{gap} rs1={out[f'gap_{gap}']['rs1_acc']} "
              f"combo={out[f'gap_{gap}']['combo_acc']} "
              f"ci={out[f'gap_{gap}']['ci']} p={bd['p_boot']}", flush=True)
    return out


def oof_reduced(X, y, make_reducer, seed):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    preds = np.zeros(len(y))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        r = make_reducer(seed)
        Xtr = r.fit_transform(X[tr])
        Xte = r.transform(X[te])
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                   random_state=seed, n_jobs=-1)
        rf.fit(Xtr, y[tr])
        preds[te] = rf.predict(Xte)
    return preds


def pca9_control():
    from sklearn.decomposition import PCA
    from sklearn.random_projection import GaussianRandomProjection
    reducers = {
        "raw_pca9": lambda seed: PCA(n_components=K, random_state=seed),
        "raw_rp9": lambda seed: GaussianRandomProjection(n_components=K,
                                                         random_state=seed),
    }
    results = {}
    for corpus in ("uss_sgd", "uss_mwoz"):
        emb, offsets, meta = load_corpus(corpus)
        y = np.asarray(meta["overall_mean"], float)
        raw = build_all_arms(emb, offsets, meta, degree=3)["design_b_raw"]
        seed_preds = {a: [] for a in reducers}
        for seed in SEEDS:
            tag = f"{corpus}_seed{seed}"
            f = CKPT / f"{tag}.npz"
            if f.exists():
                z = np.load(f)
                preds = {a: z[a] for a in reducers}
                print(f"[pca9 {corpus}] seed {seed} from checkpoint", flush=True)
            else:
                preds = {a: oof_reduced(raw, y, mk, seed)
                         for a, mk in reducers.items()}
                tmp = f.with_suffix(".tmp.npz")
                np.savez(tmp, **preds)
                tmp.rename(f)
                print(f"[pca9 {corpus}] seed {seed} done", flush=True)
            for a in reducers:
                seed_preds[a].append(preds[a])
        block = {}
        for a in reducers:
            accs = [{f"gap_{g}": pairwise_accuracy(p, y, g) for g in GAPS}
                    for p in seed_preds[a]]
            block[a] = {f"gap_{g}": {
                "mean": round(float(np.mean([x[f"gap_{g}"] for x in accs])), 4),
                "std": round(float(np.std([x[f"gap_{g}"] for x in accs])), 4)}
                for g in GAPS}
            print(f"[pca9 {corpus}] {a} " + " ".join(
                f"gap{g}={block[a][f'gap_{g}']['mean']:.4f}"
                f"+-{block[a][f'gap_{g}']['std']:.4f}" for g in GAPS),
                flush=True)
        # seed-ensembled contrasts vs the hand-scalarized arm and TRACE
        pca_ens = np.mean(seed_preds["raw_pca9"], axis=0)
        db_ens = ensemble("scoreboard_arms", corpus, "design_b")
        tr_ens = ensemble("multiseed_probes", corpus, "trace")
        block["contrasts"] = {}
        for name, a, b in [("pca9_vs_scalarized", pca_ens, db_ens),
                           ("pca9_vs_trace", pca_ens, tr_ens),
                           ("scalarized_vs_trace", db_ens, tr_ens)]:
            bd = bootstrap_diff(a, b, y, 1.0, N_BOOT, 42)
            block["contrasts"][name] = {
                "diff": round(pairwise_accuracy(a, y, 1.0)
                              - pairwise_accuracy(b, y, 1.0), 4),
                "ci": [bd["ci_lo"], bd["ci_hi"]], "p_boot": bd["p_boot"]}
            print(f"[pca9 {corpus}] {name} diff="
                  f"{block['contrasts'][name]['diff']} "
                  f"ci={block['contrasts'][name]['ci']} p={bd['p_boot']}",
                  flush=True)
        results[corpus] = block
    return results


def run(max_samples=None):
    if max_samples is not None:
        raise ValueError(
            "capacity_uniqueness re-analyzes full-corpus seed-ensembled OOF "
            "prediction caches (rolesplit_graft, scoreboard_arms, multiseed) "
            "and cannot recompute them in-process; for sampled runs use the "
            "sandbox path, reproduce.py --max-samples N, which caps the raw "
            "layer (lib/corpus_sample.py) and regenerates those caches "
            "upstream.")
    CKPT.mkdir(parents=True, exist_ok=True)
    results = {"seeds": SEEDS, "k": K,
               "uniqueness_sgd": uniqueness_contrast(),
               "pca9": pca9_control()}
    from campaign_io import save_result
    save_result("capacity_uniqueness", results,
                dataset="uss_sgd.uss_mwoz", config="minilm",
                params={"seeds": SEEDS, "n_folds": N_FOLDS, "k": K,
                        "issue": 63})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
