"""Issue #50: [role-split(1) | TRACE] graft - the regime-unification cell.

Tests whether grafting the 26 curated scalars onto the role-aware projection
(rs1_trace, 1,562-d) unifies the regimes, and runs the previously untested
head-to-head between role-split(1) and the PRISM-winning [g0 | TRACE] combo.

Design (issue #50, calibration-compliant per #43's lesson):
inference on SEED-ENSEMBLED predictions (mean of 5 OOF vectors per arm),
bootstrap n=1000; F_graft = {G1: rs1_trace > combo, G2: rs1_trace >
rolesplit_k1} on PRISM, Holm within; rolesplit_k1 vs combo reported as an
ESTIMATE (CI, no band - the measured resolution is ~+-2.5 pts). USS
contrasts secondary (graft expected inert there: combo = g0 on USS).

Reuses cached OOF predictions: rolesplit_k1 from eqdim/basis caches;
combo/g0/trace/raw from the multiseed cache. Only rs1_trace is new compute.
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
from degree_ablation import role_k_slice
from multiseed_probes import load, oof, holm

SEEDS = [42, 7, 123, 2024, 31337]
N_BOOT = 1000
GAP_PRIMARY = 1.0

CKPT = WORK.parent / "data" / "experiment-caches" / "rolesplit_graft"
REF_CACHES = [WORK.parent / "data" / "experiment-caches" / d
              for d in ("eqdim_probes", "basis_probes", "multiseed_probes")]


_CKPT = Checkpoint("rolesplit_graft", script=__file__)


def ckpt_load(tag):
    return _CKPT.load(tag) or {}


ckpt_save = _CKPT.save


def ref_pred(name, seed, key, linear=False):
    tag = f"{name}_linear" if linear else f"{name}_seed{seed}"
    for cache in REF_CACHES:
        f = cache / f"{tag}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            if key in z.files:
                return z[key]
    return None


ARMS = ["rs1_trace", "rolesplit_k1", "combo", "g0", "trace", "raw"]


def run(max_samples=None):
    sampled = max_samples is not None
    if not sampled:
        CKPT.mkdir(parents=True, exist_ok=True)
    results = {"seeds": SEEDS, "gap_primary": GAP_PRIMARY,
               "arms": "rs1_trace(1562) vs rolesplit_k1(1536)/combo(410)/refs"}
    ens = {}
    ys = {}
    for name in ["uss_sgd", "uss_mwoz", "prism"]:
        emb, offsets, meta, groups = load(name)
        emb, offsets, meta = cap(emb, offsets, meta, max_samples)
        if groups is not None:
            groups = meta["user_ids"]
        y = np.asarray(meta["overall_mean"], float)
        ys[name] = y
        base = build_all_arms(emb, offsets, meta, degree=3)
        raw, trace = base["design_b_raw"], base["trace"]
        d = raw.shape[1] // 4
        rs_k1 = role_k_slice(build_rolesplit(emb, offsets, meta)[0], 1)
        feats = {"rs1_trace": np.hstack([rs_k1, trace]),
                 "rolesplit_k1": rs_k1,
                 "combo": np.hstack([raw[:, :d], trace]),
                 "g0": raw[:, :d], "trace": trace, "raw": raw}
        assert feats["rs1_trace"].shape[1] == rs_k1.shape[1] + trace.shape[1]

        seed_accs = {k: [] for k in ARMS}
        per_seed = {}
        for seed in SEEDS:
            tag = f"{name}_seed{seed}"
            done = {} if sampled else ckpt_load(tag)
            for k in ARMS:
                if k in done:
                    continue
                p = None if sampled else ref_pred(name, seed, k)
                done[k] = p if p is not None else oof(feats[k], y, seed, groups)
                if not sampled:
                    ckpt_save(tag, done)   # per-arm: bounds loss to one OOF fit
            per_seed[seed] = done
            for k in ARMS:
                seed_accs[k].append(
                    {f"gap_{g}": pairwise_accuracy(done[k], y, g) for g in GAPS})
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
        results[name] = {"multiseed": ms}
        ens[name] = {k: np.mean([per_seed[s][k] for s in SEEDS], axis=0)
                     for k in ARMS}
        results[name]["accuracy_seedens"] = {
            k: round(pairwise_accuracy(ens[name][k], y, GAP_PRIMARY), 4)
            for k in ARMS}

        # linear probes (RidgeCV, seed 42), checkpointed per arm
        lin_tag = f"{name}_linear"
        lin_done = {} if sampled else ckpt_load(lin_tag)
        lin = {}
        for k in ARMS:
            if k not in lin_done:
                p = None if sampled else ref_pred(name, 42, k, linear=True)
                lin_done[k] = (p if p is not None
                               else oof(feats[k], y, 42, groups, linear=True))
                if not sampled:
                    ckpt_save(lin_tag, lin_done)
            lin[k] = {f"gap_{g}": round(
                pairwise_accuracy(lin_done[k], y, g), 4) for g in GAPS}
            print(f"[{name}] linear {k:13s} gap1.0={lin[k]['gap_1.0']:.4f}",
                  flush=True)
        results[name]["linear_probes"] = lin

    def contrast(name, a, b, label, test):
        bd = bootstrap_diff(ens[name][a], ens[name][b], ys[name],
                            GAP_PRIMARY, N_BOOT, 42)
        point = round(pairwise_accuracy(ens[name][a], ys[name], GAP_PRIMARY)
                      - pairwise_accuracy(ens[name][b], ys[name], GAP_PRIMARY), 4)
        return {"h": label, "corpus": name, "test": test, "point": point,
                "diff_ci": [bd["ci_lo"], bd["ci_hi"]], "p_raw": bd["p_boot"]}

    fam = [contrast("prism", "rs1_trace", "combo",
                    "G1 rs1_trace>combo PRISM", "superiority"),
           contrast("prism", "rs1_trace", "rolesplit_k1",
                    "G2 rs1_trace>rolesplit_k1 PRISM", "superiority")]
    for f, a in zip(fam, holm([f["p_raw"] for f in fam])):
        f["p_holm"] = round(float(a), 4)
        print(f"[F_graft] {f['h']:32s} point={f['point']:+.4f} "
              f"CI [{f['diff_ci'][0]:+.4f},{f['diff_ci'][1]:+.4f}] "
              f"p_holm={f['p_holm']:.4f}", flush=True)
    results["F_graft"] = fam

    results["headtohead_estimate"] = contrast(
        "prism", "rolesplit_k1", "combo",
        "E rolesplit_k1 vs combo PRISM", "estimation")
    h = results["headtohead_estimate"]
    print(f"[estimate] {h['h']:32s} point={h['point']:+.4f} "
          f"CI [{h['diff_ci'][0]:+.4f},{h['diff_ci'][1]:+.4f}]", flush=True)

    results["secondary_uss"] = [
        contrast(name, a, b, f"S {a}>{b} {name}", "secondary")
        for name in ["uss_sgd", "uss_mwoz"]
        for a, b in [("rs1_trace", "combo"), ("rs1_trace", "rolesplit_k1"),
                     ("rolesplit_k1", "combo")]]
    for s in results["secondary_uss"]:
        print(f"[secondary] {s['h']:36s} point={s['point']:+.4f} "
              f"p={s['p_raw']:.4f}", flush=True)

    if not sampled:
        from campaign_io import save_result
        save_result("rolesplit_graft", results,
                    dataset="uss_sgd.uss_mwoz.prism", config="minilm",
                    params={"seeds": SEEDS, "n_folds": 5, "n_boot": N_BOOT,
                            "inference": "seed-ensembled", "issues": "#50"})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
