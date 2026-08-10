"""Equal-dimensionality, seed-robust role-split vs raw (spec on issues #30/#21).

The seed-robust H3/H4 test compared role-split K=3 (3,072-d) against raw
(1,536-d) - a 2:1 capacity mismatch that conflates "per-speaker structure"
with "twice the parameters". This isolates the confound: role-split K=1
(1,536-d, per-role degrees 0-1) vs raw (1,536-d), across 5 seeds, RF and
RidgeCV heads, Holm-corrected.

    rolesplit_k1 = role_k_slice(build_rolesplit(...), 1)  = [gu0|gu1|gm0|gm1]  1536-d
    raw          = build_all_arms(...)["design_b_raw"]    = [G0|G1|G2|G3]      1536-d

Family F_eq (Holm within itself, kept SEPARATE from H1-H5):
    E1  rolesplit_k1 > raw   on USS-SGD    gap >= 1.0
    E2  rolesplit_k1 > raw   on USS-MWOZ   gap >= 1.0
Secondary (reported, outside Holm): E3 PRISM; linear-probe versions of all.

Decision rules (pre-stated):
  E1/E2 significant  -> role-split carries signal at matched capacity (upgrade claim)
  E1/E2 flat null    -> contribution is interpretability, not accuracy (confirm reframe)
  k1 > raw LINEAR but ties/loses RF -> direct evidence for the legibility story
"""
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "scripts"))

from trace_h2h import (load_corpus, build_all_arms, pairwise_accuracy,
                       bootstrap_diff, GAPS)
from extension_arms import build_rolesplit
from degree_ablation import role_k_slice
from multiseed_probes import load, oof, holm
from lib.campaign import Checkpoint, cap, make_run_sample

SEEDS = [42, 7, 123, 2024, 31337]
N_FOLDS, N_BOOT = 5, 1000

_CKPT = Checkpoint("eqdim_probes", script=__file__)
ckpt_load = _CKPT.load
ckpt_save = _CKPT.save
_MS = Checkpoint("multiseed_probes")


def ms_raw(name, seed):
    """Reuse multiseed's already-computed raw OOF predictions if present."""
    d = _MS.load(f"{name}_seed{seed}")
    if d is not None and "raw" in d:
        return d["raw"]
    return None


def run(max_samples=None):
    """#30/#21: equal-dimensionality role-split K=1 vs raw, 5 seeds.

    max_samples is not None -> ephemeral capped run: corpora truncated to
    their first max_samples dialogues, checkpoint caches and the multiseed
    raw-pred reuse bypassed (compute fresh), no save_result
    (lib/campaign.py contract).
    """
    ephemeral = max_samples is not None
    results = {"seeds": SEEDS, "arms": "rolesplit_k1(1536) vs raw(1536)"}
    cache = {}
    for name in ["uss_sgd", "uss_mwoz", "prism"]:
        emb, offsets, meta, groups = load(name)
        if ephemeral:
            emb, offsets, meta = cap(emb, offsets, meta, max_samples)
            if groups is not None:
                groups = meta["user_ids"]
        y = np.asarray(meta["overall_mean"], float)
        raw = build_all_arms(emb, offsets, meta, degree=3)["design_b_raw"]
        role_raw, _ = build_rolesplit(emb, offsets, meta)
        rs_k1 = role_k_slice(role_raw, 1)
        assert rs_k1.shape[1] == raw.shape[1], (rs_k1.shape, raw.shape)
        arms = {"raw": raw, "rolesplit_k1": rs_k1}
        cache[name] = (arms, y, groups)

        # --- multi-seed RF (reuse multiseed raw; compute rolesplit_k1) ---
        seed_accs = {k: [] for k in arms}
        preds42 = {}
        for seed in SEEDS:
            tag = f"{name}_seed{seed}"
            cached = None if ephemeral else ckpt_load(tag)
            preds_this = {}
            for k in arms:
                if cached is not None and k in cached:
                    preds_this[k] = cached[k]
                elif (not ephemeral and k == "raw"
                      and ms_raw(name, seed) is not None):
                    preds_this[k] = ms_raw(name, seed)
                else:
                    preds_this[k] = oof(arms[k], y, seed, groups)
            if not ephemeral:
                ckpt_save(tag, preds_this)
            for k in arms:
                if seed == 42:
                    preds42[k] = preds_this[k]
                seed_accs[k].append(
                    {f"gap_{g}": pairwise_accuracy(preds_this[k], y, g) for g in GAPS})
            print(f"[{name}] seed {seed} done", flush=True)
        ms = {}
        for k, accs in seed_accs.items():
            ms[k] = {f"gap_{g}": {
                "mean": round(float(np.mean([a[f"gap_{g}"] for a in accs])), 4),
                "std": round(float(np.std([a[f"gap_{g}"] for a in accs])), 4)}
                for g in GAPS}
            print(f"[{name}] {k:13s} " + " ".join(
                f"gap{g}={ms[k][f'gap_{g}']['mean']:.4f}+-{ms[k][f'gap_{g}']['std']:.4f}"
                for g in GAPS), flush=True)
        results[name] = {"multiseed": ms,
                         "_preds42": {k: p.tolist() for k, p in preds42.items()}}

        # --- linear probes (RidgeCV, seed 42) ---
        lin_cached = None if ephemeral else ckpt_load(f"{name}_linear")
        lin, lin_preds = {}, {}
        for k in arms:
            if lin_cached is not None and k in lin_cached:
                p = lin_cached[k]
            else:
                p = oof(arms[k], y, 42, groups, linear=True)
            lin_preds[k] = p
            lin[k] = {f"gap_{g}": round(pairwise_accuracy(p, y, g), 4) for g in GAPS}
            print(f"[{name}] linear {k:13s} gap1.0={lin[k]['gap_1.0']:.4f} "
                  f"(RF {ms[k]['gap_1.0']['mean']:.4f})", flush=True)
        if not ephemeral:
            ckpt_save(f"{name}_linear", lin_preds)
        results[name]["linear_probes"] = lin

    # --- E1/E2 family (RF, gap 1.0, Holm across the 2) ---
    fam = []
    for name, label in [("uss_sgd", "E1 rolesplit_k1>raw SGD"),
                        ("uss_mwoz", "E2 rolesplit_k1>raw MWOZ")]:
        _, y, _ = cache[name]
        pa = np.asarray(results[name]["_preds42"]["rolesplit_k1"])
        pb = np.asarray(results[name]["_preds42"]["raw"])
        bd = bootstrap_diff(pa, pb, y, 1.0, N_BOOT, 42)
        fam.append({"h": label, "corpus": name,
                    "diff_ci": [bd["ci_lo"], bd["ci_hi"]], "p_raw": bd["p_boot"]})
    for f, a in zip(fam, holm([f["p_raw"] for f in fam])):
        f["p_holm"] = round(float(a), 4)
        print(f"[F_eq] {f['h']:26s} p_raw={f['p_raw']:.4f} p_holm={f['p_holm']:.4f}",
              flush=True)
    results["primary_family_eqdim"] = fam

    # --- E3 PRISM (secondary, reported) + linear E1/E2 bootstraps ---
    _, y, _ = cache["prism"]
    pa = np.asarray(results["prism"]["_preds42"]["rolesplit_k1"])
    pb = np.asarray(results["prism"]["_preds42"]["raw"])
    results["E3_prism_rf"] = bootstrap_diff(pa, pb, y, 1.0, N_BOOT, 42)
    print(f"[E3] PRISM rolesplit_k1>raw (RF) p={results['E3_prism_rf']['p_boot']:.4f}",
          flush=True)

    for name in cache:
        del results[name]["_preds42"]
    if not ephemeral:
        from campaign_io import save_result
        save_result("eqdim_probes", results, dataset="uss_sgd.uss_mwoz.prism",
                    config="minilm",
                    params={"seeds": SEEDS, "n_folds": 5,
                            "arms": "rolesplit_k1(1536) vs raw(1536)"})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    """CLI shim - historical invocation: python scripts/eqdim_probes.py"""
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
