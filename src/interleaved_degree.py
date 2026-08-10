"""Completing the degree x split factorial: the interleaved projection at K=1/K=2.

Table 2 (eqdim_probes) compares role-split(1) [gu0|gu1|gm0|gm1] (1,536-d)
against the interleaved projection [G0|G1|G2|G3] (1,536-d) - matched
capacity, mismatched polynomial degree. The degree ablation shows
truncation alone helps the role-split arm (K=1 >= K=3 on every corpus),
and Table 1 shows it helps the interleaved family too on SGD (mean
pooling, identically interleaved K=0, 0.729 > raw K=3, 0.705). The cell
that decides whether Table 2's gap is a split effect or a truncation
effect - the interleaved projection truncated to K=1 ([G0|G1], 768-d) -
was never run. This script runs it (plus K=2 for the dose-response
curve): 5 seeds, RF head, all three corpora, standard OOF protocol.

Contrasts on seed-ensembled OOF predictions (Section 5.2 inference),
reusing the per-seed prediction vectors already checkpointed by
eqdim_probes (raw, rolesplit_k1) and multiseed_probes (g0) so the
comparison arms are bit-identical to Tables 1 and 2:

    C1  rolesplit_k1 vs inter_k1   the decisive degree-matched contrast
    C2  inter_k1     vs raw        the truncation effect in the
                                   interleaved family
    C3  inter_k1     vs g0         does interleaved drift add over the
                                   interleaved mean
    C4  rolesplit_k1 vs g0         split vs mean pooling (previously
                                   never bootstrap-tested)
    C5  inter_k2     vs raw        dose-response point

This is a follow-up diagnostic with fixed decision rules: if inter_k1 on USS-SGD stays at or below mean
pooling (~0.73), Table 2's split interpretation stands (the gap is not a
truncation artifact) and Section 6.5 gains a sentence closing the
loophole; if inter_k1 approaches role-split(1) (~0.75), the SGD claim
must be reframed from "splitting" to "degree truncation + splitting".
Holm is applied to C1 across the three corpora (gap 1.0).
"""
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))

from trace_h2h import (build_all_arms, pairwise_accuracy, bootstrap_diff,
                       GAPS)
from extension_arms import build_rolesplit
from degree_ablation import role_k_slice
from multiseed_probes import oof, holm
from lib.campaign import Checkpoint, load_corpus, make_run_sample

SEEDS = [42, 7, 123, 2024, 31337]
N_FOLDS, N_BOOT, D = 5, 1000, 384
GAP_PRIMARY = 1.0

_CKPT = Checkpoint("interleaved_degree", script=__file__)
# passive readers of other producers' caches (never invalidate them)
_EQ = Checkpoint("eqdim_probes")
_MS = Checkpoint("multiseed_probes")

# arm -> (producing cache, key in its per-seed npz); None = computed here
REUSED = {"raw": (_EQ, "raw"), "rolesplit_k1": (_EQ, "rolesplit_k1"),
          "g0": (_MS, "g0")}
CONTRASTS = [("rolesplit_k1", "inter_k1", "C1 decisive degree-matched"),
             ("inter_k1", "raw", "C2 truncation effect"),
             ("inter_k1", "g0", "C3 drift over mean, interleaved"),
             ("rolesplit_k1", "g0", "C4 split vs mean pooling"),
             ("inter_k2", "raw", "C5 dose-response")]


def reused_pred(arm, name, seed, n):
    """A comparison arm's per-seed OOF vector from its producer's cache,
    or None when absent/mismatched (caller recomputes)."""
    ckpt, key = REUSED[arm]
    d = ckpt.load(f"{name}_seed{seed}")
    if d is not None and key in d and len(d[key]) == n:
        return np.asarray(d[key], float)
    return None


def run(max_samples=None):
    """Interleaved K=1/K=2 arms + degree-matched contrasts, 5 seeds.

    max_samples is not None -> ephemeral capped run: corpora truncated
    to their first max_samples dialogues, every cache (own and reused)
    bypassed and recomputed fresh, no save_result (lib/campaign.py
    contract).
    """
    ephemeral = max_samples is not None
    results = {"seeds": SEEDS,
               "arms": {"inter_k1": "[G0|G1] 768-d", "inter_k2":
                        "[G0|G1|G2] 1152-d", "raw": "[G0..G3] 1536-d",
                        "rolesplit_k1": "[gu0|gu1|gm0|gm1] 1536-d",
                        "g0": "mean pooling 384-d"},
               "protocol": "RF 300 trees, 5-fold OOF (grouped on PRISM), "
                           "seed-ensembled inference (Section 5.2)"}
    for name in ["uss_sgd", "uss_mwoz", "prism"]:
        emb, offsets, meta, groups = load_corpus(name, max_samples)
        y = np.asarray(meta["overall_mean"], float)
        raw = build_all_arms(emb, offsets, meta, degree=3)["design_b_raw"]
        role_raw, _ = build_rolesplit(emb, offsets, meta)
        arms = {"inter_k1": raw[:, :2 * D], "inter_k2": raw[:, :3 * D],
                "raw": raw, "rolesplit_k1": role_k_slice(role_raw, 1),
                "g0": raw[:, :D]}
        assert arms["rolesplit_k1"].shape[1] == raw.shape[1]

        per_seed = {k: [] for k in arms}
        for seed in SEEDS:
            tag = f"{name}_seed{seed}"
            cached = None if ephemeral else _CKPT.load(tag)
            own = {}
            for k in arms:
                if cached is not None and k in cached:
                    p = np.asarray(cached[k], float)
                elif not ephemeral and k in REUSED:
                    p = reused_pred(k, name, seed, len(y))
                    if p is None:
                        print(f"[{name}] {k} seed {seed}: cache miss, "
                              "recomputing", flush=True)
                        p = oof(arms[k], y, seed, groups)
                else:
                    p = oof(arms[k], y, seed, groups)
                own[k] = p
                per_seed[k].append(p)
            if not ephemeral:
                _CKPT.save(tag, {k: own[k] for k in ("inter_k1", "inter_k2")})
            print(f"[{name}] seed {seed} done", flush=True)

        ms = {}
        for k in arms:
            accs = {f"gap_{g}": [pairwise_accuracy(p, y, g)
                                 for p in per_seed[k]] for g in GAPS}
            ms[k] = {gk: {"mean": round(float(np.mean(a)), 4),
                          "std": round(float(np.std(a)), 4)}
                     for gk, a in accs.items()}
            print(f"[{name}] {k:13s} " + " ".join(
                f"{gk}={v['mean']:.4f}+-{v['std']:.4f}"
                for gk, v in ms[k].items()), flush=True)

        ens = {k: np.mean(v, axis=0) for k, v in per_seed.items()}
        seedens = {k: {f"gap_{g}": round(pairwise_accuracy(p, y, g), 4)
                       for g in GAPS} for k, p in ens.items()}
        contrasts = {}
        for a, b, label in CONTRASTS:
            block = {"label": label, "arms": [a, b]}
            for gap in GAPS:
                bd = bootstrap_diff(ens[a], ens[b], y, gap, N_BOOT, 42)
                block[f"gap_{gap}"] = {
                    "delta": round(float(pairwise_accuracy(ens[a], y, gap)
                                         - pairwise_accuracy(ens[b], y, gap)),
                                   4),
                    "ci": [round(bd["ci_lo"], 4), round(bd["ci_hi"], 4)],
                    "p_boot": bd["p_boot"]}
            contrasts[f"{a}_vs_{b}"] = block
            g = block[f"gap_{GAP_PRIMARY}"]
            print(f"[{name}] {label:33s} delta={g['delta']:+.4f} "
                  f"CI [{g['ci'][0]:+.4f},{g['ci'][1]:+.4f}] "
                  f"p={g['p_boot']:.4f}", flush=True)
        results[name] = {"multiseed": ms, "seedens_accuracy": seedens,
                         "contrasts": contrasts}

    fam = [{"corpus": name, "h": f"C1 rolesplit_k1>inter_k1 {name}",
            **results[name]["contrasts"]["rolesplit_k1_vs_inter_k1"]
            [f"gap_{GAP_PRIMARY}"]}
           for name in ["uss_sgd", "uss_mwoz", "prism"]]
    for f, a in zip(fam, holm([f["p_boot"] for f in fam])):
        f["p_holm"] = round(float(a), 4)
        print(f"[C1 family] {f['h']:35s} p_raw={f['p_boot']:.4f} "
              f"p_holm={f['p_holm']:.4f}", flush=True)
    results["decisive_family_c1"] = fam

    if not ephemeral:
        from campaign_io import save_result
        save_result("interleaved_degree", results,
                    dataset="uss_sgd.uss_mwoz.prism", config="minilm",
                    params={"seeds": SEEDS, "n_folds": N_FOLDS,
                            "arms": "inter_k1(768) inter_k2(1152) vs "
                                    "raw/rolesplit_k1/g0 (reused)"})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
