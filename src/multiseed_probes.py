"""Issues #21 + #24: multi-seed variance, Holm-corrected primary family,
and linear-probe arms.

Primary family (5 hypotheses, all at gap >= 1.0, bootstrap
p-values at the canonical seed 42, Holm-corrected):
    H1  design_b_raw > TRACE-geometric        on USS-SGD
    H2  design_b_raw > TRACE-geometric        on USS-MWOZ
    H3  rolesplit    > design_b_raw           on USS-SGD
    H4  rolesplit    > design_b_raw           on USS-MWOZ
    H5  [g0|TRACE]   > g0                     on PRISM

Multi-seed: seeds {42, 7, 123, 2024, 31337} for CV shuffling and RF; reported
as mean +- std pairwise accuracy per arm (stability, not inference).
PRISM multi-seed limited to the H5 arms (grouped CV; rolesplit excluded to
bound compute - it is not in the primary family there).

Linear probes (#24): RidgeCV heads, same OOF protocol, seed 42, all arms,
USS corpora + PRISM H5 arms. Reports the RF-minus-linear gap per
representation.
"""
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))

from trace_h2h import (build_all_arms, pairwise_accuracy,
                       bootstrap_diff, GAPS)
from extension_arms import build_rolesplit
from lib.campaign import Checkpoint, cap, load_corpus, make_run_sample

SEEDS = [42, 7, 123, 2024, 31337]
N_FOLDS, N_BOOT = 5, 1000

_CKPT = Checkpoint("multiseed_probes", script=__file__)
ckpt_load = _CKPT.load
ckpt_save = _CKPT.save


def load(name):
    return load_corpus(name)


def oof(X, y, seed, groups, linear=False):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    preds = np.zeros(len(y))
    if groups is None:
        splits = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(X)
    else:
        splits = GroupKFold(n_splits=N_FOLDS).split(X, y, groups)
    for tr, te in splits:
        if linear:
            m = make_pipeline(StandardScaler(),
                              RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0]))
        else:
            m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                      random_state=seed, n_jobs=-1)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return preds


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def run(max_samples=None):
    """#21/#24: multi-seed RF + linear probes + Holm primary family.

    max_samples is not None -> ephemeral capped run: corpora truncated to
    their first max_samples dialogues, checkpoint caches neither read nor
    written, no save_result (lib/campaign.py contract).
    """
    ephemeral = max_samples is not None
    results = {"seeds": SEEDS}
    arms_cache = {}
    for name in ["uss_sgd", "uss_mwoz", "prism"]:
        emb, offsets, meta, groups = load(name)
        if ephemeral:
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
        arms_cache[name] = (arms, y, groups)

        # --- multi-seed RF ---
        seed_accs = {k: [] for k in arms}
        seed_preds_42 = {}
        for seed in SEEDS:
            tag = f"{name}_seed{seed}"
            cached = None if ephemeral else ckpt_load(tag)
            if cached is not None and all(k in cached for k in arms):
                preds_this = {k: cached[k] for k in arms}
                print(f"[{name}] seed {seed} loaded from checkpoint", flush=True)
            else:
                preds_this = {k: oof(X, y, seed, groups) for k, X in arms.items()}
                if not ephemeral:
                    ckpt_save(tag, preds_this)
                    print(f"[{name}] seed {seed} done (checkpointed)", flush=True)
                else:
                    print(f"[{name}] seed {seed} done (ephemeral)", flush=True)
            for k in arms:
                p = preds_this[k]
                if seed == 42:
                    seed_preds_42[k] = p
                seed_accs[k].append(
                    {f"gap_{gap}": pairwise_accuracy(p, y, gap) for gap in GAPS})
        ms = {}
        for k, accs in seed_accs.items():
            ms[k] = {f"gap_{gap}": {
                "mean": round(float(np.mean([a[f"gap_{gap}"] for a in accs])), 4),
                "std": round(float(np.std([a[f"gap_{gap}"] for a in accs])), 4)}
                for gap in GAPS}
            print(f"[{name}] {k:10s} " + " ".join(
                f"gap{gap}={ms[k][f'gap_{gap}']['mean']:.4f}+-{ms[k][f'gap_{gap}']['std']:.4f}"
                for gap in GAPS), flush=True)
        results[name] = {"multiseed": ms}
        results[name]["_preds42"] = {k: p.tolist() for k, p in seed_preds_42.items()}

        # --- linear probes (seed 42) ---
        lin = {}
        lin_cached = None if ephemeral else ckpt_load(f"{name}_linear")
        lin_preds = {}
        for k, X in arms.items():
            if lin_cached is not None and k in lin_cached:
                p = lin_cached[k]
            else:
                p = oof(X, y, 42, groups, linear=True)
            lin_preds[k] = p
            lin[k] = {f"gap_{gap}": round(pairwise_accuracy(p, y, gap), 4)
                      for gap in GAPS}
            rf_acc = ms[k]["gap_1.0"]["mean"]
            print(f"[{name}] linear {k:10s} gap1.0={lin[k]['gap_1.0']:.4f} "
                  f"(RF mean {rf_acc:.4f})", flush=True)
        if not ephemeral:
            ckpt_save(f"{name}_linear", lin_preds)
        results[name]["linear_probes"] = lin

    # --- primary family, Holm-corrected (seed-42 bootstrap p at gap 1.0) ---
    fam = []
    def add(name, a, b, label):
        arms, y, groups = arms_cache[name]
        pa = np.asarray(results[name]["_preds42"][a])
        pb = np.asarray(results[name]["_preds42"][b])
        bd = bootstrap_diff(pa, pb, y, 1.0, N_BOOT, 42)
        fam.append({"h": label, "corpus": name, "diff_ci": [bd["ci_lo"], bd["ci_hi"]],
                    "p_raw": bd["p_boot"]})
    add("uss_sgd", "raw", "trace", "H1 raw>trace SGD")
    add("uss_mwoz", "raw", "trace", "H2 raw>trace MWOZ")
    add("uss_sgd", "rolesplit", "raw", "H3 rolesplit>raw SGD")
    add("uss_mwoz", "rolesplit", "raw", "H4 rolesplit>raw MWOZ")
    add("prism", "combo", "g0", "H5 combo>g0 PRISM")
    adj = holm([f["p_raw"] for f in fam])
    for f, a in zip(fam, adj):
        f["p_holm"] = round(float(a), 4)
        print(f"[family] {f['h']:24s} p_raw={f['p_raw']:.4f} p_holm={f['p_holm']:.4f}",
              flush=True)
    results["primary_family"] = fam

    for name in ["uss_sgd", "uss_mwoz", "prism"]:
        del results[name]["_preds42"]
    if not ephemeral:
        from campaign_io import save_result
        save_result("multiseed_probes", results,
                    dataset="uss_sgd.uss_mwoz.prism", config="minilm",
                    params={"seeds": SEEDS, "n_folds": 5})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    """CLI shim - historical invocation: python scripts/multiseed_probes.py"""
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
