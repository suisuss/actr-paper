"""Issue #51: role-split(1) vs raw at matched capacity, across embedders.

Retires (or bounds) draft Section 6.5's stated limit: the matched-capacity
role-split(1) form had only been measured under MiniLM. Re-runs the eqdim
contrast under GTE-small (384-d) and bge-base (768-d) on all three corpora,
including the PRISM role-split cells that #19 excluded to bound compute.

Design (issue #51, calibration-compliant): seeds [42, 7, 123];
inference on seed-ensembled predictions, bootstrap n=1000. Primary family
(Holm over 2): rs1 > raw on SGD under each embedder (the cell where MiniLM
shows +4.7, above the ~+-2.5 pt instrument resolution). All other cells
(MWOZ expected null, PRISM expected small positive) reported as estimates.

Reuses the multimodel cache's seed-42 raw predictions per embedder; new
compute is rs1 x 3 seeds and raw x 2 extra seeds per (embedder, corpus).
"""
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "scripts"))

from lib.campaign import Checkpoint, cap, make_run_sample
from trace_h2h import build_all_arms, pairwise_accuracy, bootstrap_diff, GAPS
from extension_arms import build_rolesplit
from multiseed_probes import oof, holm
from multimodel_eval import load as load_mm

SEEDS = [42, 7, 123]
N_BOOT = 1000
GAP_PRIMARY = 1.0
TAGS = ["gte", "bge"]

CKPT = WORK.parent / "data" / "experiment-caches" / "rolesplit_embedders"
_CKPT = Checkpoint("rolesplit_embedders", script=__file__)


def ckpt_load(tag):
    return _CKPT.load(tag) or {}


ckpt_save = _CKPT.save

_MM = Checkpoint("multimodel_eval")


def mm_raw(tag, name):
    """Seed-42 raw OOF predictions from the #19 multimodel cache."""
    d = _MM.load(f"{tag}_{name}")
    if d is not None and "raw" in d:
        return d["raw"]
    return None


def run(max_samples=None):
    sampled = max_samples is not None
    if not sampled:
        CKPT.mkdir(parents=True, exist_ok=True)
    results = {"seeds": SEEDS, "gap_primary": GAP_PRIMARY, "embedders": TAGS}
    ens, ys = {}, {}
    for tag in TAGS:
        results[tag] = {}
        for name in ["uss_sgd", "uss_mwoz", "prism"]:
            emb, offsets, meta, groups = load_mm(name, tag)
            emb, offsets, meta = cap(emb, offsets, meta, max_samples)
            if groups is not None:
                groups = meta["user_ids"]
            y = np.asarray(meta["overall_mean"], float)
            ys[(tag, name)] = y
            raw = build_all_arms(emb, offsets, meta, degree=3)["design_b_raw"]
            # role_k_slice hardcodes D=384; slice K=1 for arbitrary d instead
            d = emb.shape[1]
            role_raw = build_rolesplit(emb, offsets, meta)[0]
            cols = []
            for base_ix in (0, 4 * d):
                cols.extend(range(base_ix, base_ix + 2 * d))
            rs1 = role_raw[:, cols]
            assert rs1.shape[1] == raw.shape[1], (rs1.shape, raw.shape)
            feats = {"raw": raw, "rolesplit_k1": rs1}

            seed_accs = {k: [] for k in feats}
            per_seed = {}
            for seed in SEEDS:
                ck = f"{tag}_{name}_seed{seed}"
                done = {} if sampled else ckpt_load(ck)
                for k in feats:
                    if k in done:
                        continue
                    p = (mm_raw(tag, name)
                         if (not sampled and k == "raw" and seed == 42)
                         else None)
                    done[k] = p if p is not None else oof(feats[k], y, seed,
                                                          groups)
                    if not sampled:
                        ckpt_save(ck, done)
                per_seed[seed] = done
                for k in feats:
                    seed_accs[k].append(
                        {f"gap_{g}": pairwise_accuracy(done[k], y, g)
                         for g in GAPS})
                print(f"[{tag}/{name}] seed {seed} done", flush=True)
            ms = {}
            for k, accs in seed_accs.items():
                ms[k] = {f"gap_{g}": {
                    "mean": round(float(np.mean([a[f"gap_{g}"] for a in accs])), 4),
                    "std": round(float(np.std([a[f"gap_{g}"] for a in accs])), 4)}
                    for g in GAPS}
                print(f"[{tag}/{name}] {k:13s} " + " ".join(
                    f"gap{g}={ms[k][f'gap_{g}']['mean']:.4f}"
                    f"+-{ms[k][f'gap_{g}']['std']:.4f}" for g in GAPS),
                    flush=True)
            results[tag][name] = {"multiseed": ms}
            ens[(tag, name)] = {k: np.mean([per_seed[s][k] for s in SEEDS],
                                           axis=0) for k in feats}
            results[tag][name]["accuracy_seedens"] = {
                k: round(pairwise_accuracy(ens[(tag, name)][k], y,
                                           GAP_PRIMARY), 4) for k in feats}

    def contrast(tag, name, label, test):
        e, y = ens[(tag, name)], ys[(tag, name)]
        bd = bootstrap_diff(e["rolesplit_k1"], e["raw"], y,
                            GAP_PRIMARY, N_BOOT, 42)
        point = round(pairwise_accuracy(e["rolesplit_k1"], y, GAP_PRIMARY)
                      - pairwise_accuracy(e["raw"], y, GAP_PRIMARY), 4)
        return {"h": label, "embedder": tag, "corpus": name, "test": test,
                "point": point, "diff_ci": [bd["ci_lo"], bd["ci_hi"]],
                "p_raw": bd["p_boot"]}

    fam = [contrast(tag, "uss_sgd", f"E1-{tag} rs1>raw SGD", "superiority")
           for tag in TAGS]
    for f, a in zip(fam, holm([f["p_raw"] for f in fam])):
        f["p_holm"] = round(float(a), 4)
        print(f"[F_emb] {f['h']:24s} point={f['point']:+.4f} "
              f"CI [{f['diff_ci'][0]:+.4f},{f['diff_ci'][1]:+.4f}] "
              f"p_holm={f['p_holm']:.4f}", flush=True)
    results["F_emb"] = fam

    results["estimates"] = [
        contrast(tag, name, f"S rs1>raw {tag}/{name}", "estimation")
        for tag in TAGS for name in ["uss_mwoz", "prism"]]
    for s in results["estimates"]:
        print(f"[estimate] {s['h']:26s} point={s['point']:+.4f} "
              f"CI [{s['diff_ci'][0]:+.4f},{s['diff_ci'][1]:+.4f}] "
              f"p={s['p_raw']:.4f}", flush=True)

    if not sampled:
        from campaign_io import save_result
        save_result("rolesplit_embedders", results,
                    dataset="uss_sgd.uss_mwoz.prism", config="gte_bge",
                    params={"seeds": SEEDS, "n_folds": 5, "n_boot": N_BOOT,
                            "inference": "seed-ensembled", "issues": "#51"})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
