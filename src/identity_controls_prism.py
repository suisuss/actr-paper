"""Issue #97 Analysis B: same-model pair restriction on the PRISM role-split gain.

B is the headline control for H_confound (see identity_probes_prism.py for the
hypothesis and the A gate). It requires NO retraining: it re-scores the
existing seed-ensembled OOF prediction vectors under a pair filter.

Falsifiable prediction of H_confound: the split's advantage shrinks or
vanishes when model identity is held constant. Holding it constant at
evaluation time = restricting the pairwise-preference metric to pairs of
conversations that used the SAME chosen model. If the gain is identity, it
cannot survive a partition where identity carries no information.

Two quantities, both bootstrapped:

    Delta_same   = role-split(1) - interleaved on same-model pairs
    attenuation  = Delta_all - Delta_same        <- how much of the gain is
                                                    model identity

The attenuation CI is why this script does not just call bootstrap_diff
twice: Delta_all and Delta_same must be computed inside the SAME resample,
or their difference gets no valid paired interval. bootstrap_partitions()
below is a mirror of trace_h2h.bootstrap_diff (same m-out-of-n dialogue
resampling, same seeding, same CUDA/numpy equivalence) that evaluates every
partition per resample.

Pair construction is IMPORTED, never reimplemented: pair_matrices/
pairwise_accuracy/bootstrap_diff come from trace_h2h, so any filter quirk is
shared across arms by construction. The same-model partition is one extra
mask ANDed into `valid`.

Regression test (must hold, or the re-scoring is not measuring Table 2):
  - per-seed mean accuracies reproduce Table 2's PRISM MiniLM cells
    (interleaved 0.610, role-split(1) 0.638)
  - bootstrap_diff on the seed-42 vectors reproduces eqdim_probes'
    E3_prism_rf exactly (ci [0.0038, 0.0541], p=0.018)

PROTOCOL NOTE (discrepancy found 2026-07-15, issue #97). Section 5.2 defines
seed-ensembled as "the mean of per-seed OOF vectors" and Limitation 6 states
"All inferential claims here are seed-ensembled", but eqdim_probes computes
Table 2's MiniLM CIs (E1/E2/E3) from the SEED-42 vectors alone;
rolesplit_embedders (the GTE/bge rows of the same table) does ensemble. So
Table 2 mixes both conventions. This script reports BOTH:
  inference_seedens  - the paper's STATED protocol (primary here, and what
                       the #97 spec asks for)
  inference_seed42   - what Table 2's MiniLM row actually printed (the
                       regression test)
Neither is "wrong"; they must simply not be confused, and Delta_all differs
slightly between them.

Free ride (scope discipline, #97): mean_pool and combo re-score at zero
marginal cost from the same partition, so they appear in the table for
completeness. They do NOT get C-level controls - the combo is not the
paper's recommendation.

Arms (all n=7951, 5 seeds, from existing caches - no retraining):
  interleaved  eqdim_probes/prism_seed{S}.npz  "raw"           (Table 2 producer)
  rolesplit_1  eqdim_probes/prism_seed{S}.npz  "rolesplit_k1"  (Table 2 producer)
  mean_pool    rolesplit_graft/prism_seed{S}.npz "g0"
  combo        rolesplit_graft/prism_seed{S}.npz "combo"

Checkpoints: none needed for the re-scoring itself (it is seconds of GPU
work per contrast); the expensive OOF layer it consumes is already cached
upstream. Registry: lib.campaign.save_result.
"""
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))

from lib.campaign import Checkpoint, load_corpus, make_run_sample, save_result
from trace_h2h import pair_matrices, pairwise_accuracy, bootstrap_diff, GAPS
from identity_probes_prism import prism_model_labels

SEEDS = [42, 7, 123, 2024, 31337]
N_BOOT, MAX_M = 1000, 2000
GAP_PRIMARY = 1.0

# (arm label, producing script's cache, arm key inside the npz)
ARMS = [("interleaved", "eqdim_probes", "raw"),
        ("rolesplit_1", "eqdim_probes", "rolesplit_k1"),
        ("mean_pool", "rolesplit_graft", "g0"),
        ("combo", "rolesplit_graft", "combo")]

# the contrast #97 protects, then the free ride
CONTRASTS = [("rolesplit_1", "interleaved", "primary"),
             ("combo", "mean_pool", "free_ride")]


def per_seed(cache, arm):
    """The per-seed OOF vectors for one arm, from another script's cache.

    Reader instances (no script=) never invalidate: freshness is the
    producer's job, enforced by the driver's depends_on ordering.
    """
    ck = Checkpoint(cache)
    out = []
    for s in SEEDS:
        d = ck.load(f"prism_seed{s}")
        if d is None or arm not in d:
            raise FileNotFoundError(
                f"missing {cache}/prism_seed{s}.npz[{arm}] - run the "
                f"producing stage ({cache}) first; B re-scores its OOF "
                "predictions and cannot recompute them.")
        out.append(np.asarray(d[arm], float))
    return out


def bootstrap_partitions(preds_a, preds_b, y, gap, masks, n_boot=N_BOOT,
                         seed=42, max_m=MAX_M):
    """acc(A)-acc(B) per pair-partition, evaluated inside a shared resample.

    A mirror of trace_h2h.bootstrap_diff: m-out-of-n dialogue resampling
    (m=max_m), indices drawn with numpy so CPU and CUDA agree bit-for-bit.
    masks: {name: (n,n) bool} extra pair masks ANDed into `valid`; the
    per-resample differences of every partition are returned together, which
    is what makes attenuation = diff[all] - diff[same] a paired quantity.

    Duplicate indices from resampling with replacement are self-excluding:
    a dialogue paired with its own copy has |dy| = 0 < gap, so it never
    enters `valid` (the same property bootstrap_diff relies on).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    m = min(n, max_m)
    dy, valid = pair_matrices(y, gap)
    da = (preds_a[:, None] - preds_a[None, :]) * dy > 0
    db = (preds_b[:, None] - preds_b[None, :]) * dy > 0
    valid_p = {name: (valid & mask) for name, mask in masks.items()}
    idx_all = rng.integers(0, n, (n_boot, m))
    out = {name: np.zeros(n_boot) for name in valid_p}

    use_cuda = False
    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    if use_cuda:
        import torch
        dev = torch.device("cuda")
        tda = torch.from_numpy(da).to(dev)
        tdb = torch.from_numpy(db).to(dev)
        tval = {k: torch.from_numpy(v).to(dev) for k, v in valid_p.items()}
        tidx = torch.from_numpy(idx_all.astype(np.int64)).to(dev)
        for b in range(n_boot):
            i = tidx[b]
            A = tda.index_select(0, i).index_select(1, i)
            B = tdb.index_select(0, i).index_select(1, i)
            for k, v_full in tval.items():
                v = v_full.index_select(0, i).index_select(1, i)
                nv = v.sum()
                if nv.item() == 0:
                    continue
                out[k][b] = float(
                    ((A & v).sum() - (B & v).sum()).item() / nv.item())
    else:
        for b in range(n_boot):
            i = idx_all[b]
            A = da[np.ix_(i, i)]
            B = db[np.ix_(i, i)]
            for k, v_full in valid_p.items():
                v = v_full[np.ix_(i, i)]
                nv = v.sum()
                if nv == 0:
                    continue
                out[k][b] = (A[v].sum() - B[v].sum()) / nv
    return out


def ci_p(diffs):
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"point": round(float(diffs.mean()), 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)],
            "p_boot": float(min(p, 1.0))}


def partition_masks(labels):
    same = labels[:, None] == labels[None, :]
    return {"all": np.ones_like(same, dtype=bool), "same_model": same,
            "diff_model": ~same}


def pair_counts(y, labels, masks):
    out = {}
    for gap in GAPS:
        _, valid = pair_matrices(y, gap)
        block = {k: int((valid & m).sum() // 2) for k, m in masks.items()}
        block["same_model_share"] = round(
            block["same_model"] / block["all"], 4) if block["all"] else 0.0
        out[f"gap_{gap}"] = block
    return out


def decide(delta_all, delta_same, atten):
    """The #97 decision rules for reading B."""
    excludes_zero = not (atten["ci"][0] <= 0 <= atten["ci"][1])
    half = delta_same["point"] >= 0.5 * delta_all["point"]
    if not excludes_zero and half:
        v = ("CLEAN - the split's gain survives holding model identity "
             "constant; not a fingerprint effect")
    elif excludes_zero and not half:
        v = ("DIRTY - a material share of the PRISM gain is model identity; "
             "rescope contribution 3 per #97 (SGD untouched: single-system)")
    else:
        v = ("UNDECIDABLE at this instrument - report the gap-0.5 replica as "
             "primary for this control and lean on C (Analysis C1)")
    return {"verdict": v, "attenuation_ci_excludes_zero": bool(excludes_zero),
            "delta_same_at_least_half_of_delta_all": bool(half),
            "rule": ("clean = attenuation CI includes zero AND Delta_same >= "
                     "0.5*Delta_all; dirty = CI excludes zero AND Delta_same "
                     "< 0.5*Delta_all; else undecidable")}


def run(max_samples=None):
    """#97 B: same-model pair restriction; Delta_same and the attenuation.

    max_samples is rejected: B re-scores full-corpus OOF prediction caches
    keyed without N (lib/campaign.py contract). For a sampled structural run
    use the sandbox driver, reproduce.py --max-samples N, which caps the raw
    layer and regenerates the upstream caches at that size.
    """
    if max_samples is not None:
        raise ValueError(
            "identity_controls_prism re-scores full-corpus seed-ensembled OOF "
            "prediction caches (eqdim_probes, rolesplit_graft) whose keys do "
            "not encode N and cannot be recomputed in-process; for sampled "
            "runs use the sandbox path, reproduce.py --max-samples N, which "
            "caps the raw layer (lib/corpus_sample.py) and regenerates those "
            "caches upstream.")
    _, offsets, meta, _ = load_corpus("prism")
    y = np.asarray(meta["overall_mean"], float)
    labels = prism_model_labels()
    assert len(labels) == len(offsets) - 1 == len(y), \
        f"label/corpus misalignment: {len(labels)} vs {len(offsets) - 1}"

    seeds_by_arm = {a: per_seed(c, k) for a, c, k in ARMS}
    ens = {a: np.mean(v, axis=0) for a, v in seeds_by_arm.items()}
    masks = partition_masks(labels)

    results = {
        "seeds": SEEDS, "n_boot": N_BOOT, "max_m": MAX_M,
        "n_dialogues": int(len(y)), "n_models": int(len(set(labels))),
        "arms": {a: f"{c}/prism_seed*.npz[{k}]" for a, c, k in ARMS},
        "pair_counts": pair_counts(y, labels, masks),
    }
    pc = results["pair_counts"][f"gap_{GAP_PRIMARY}"]
    print(f"[pairs] gap{GAP_PRIMARY}: all={pc['all']:,} "
          f"same={pc['same_model']:,} ({100 * pc['same_model_share']:.1f}%) "
          f"diff={pc['diff_model']:,}", flush=True)

    # per-model same-model pair counts (composition shift is a caveat to
    # state, not a bug to fix - #97 scope note)
    top = sorted(set(labels), key=lambda m: -(labels == m).sum())[:6]
    _, valid1 = pair_matrices(y, GAP_PRIMARY)
    results["top_models_same_pairs"] = {}
    for mname in top:
        sel = labels == mname
        results["top_models_same_pairs"][mname] = {
            "n_conversations": int(sel.sum()),
            "same_model_pairs_gap1": int(valid1[np.ix_(sel, sel)].sum() // 2)}

    # --- regression test: does this reproduce Table 2 / E3 exactly? ----
    reg = {}
    for arm in ("interleaved", "rolesplit_1"):
        accs = [pairwise_accuracy(p, y, GAP_PRIMARY)
                for p in seeds_by_arm[arm]]
        reg[arm] = {"per_seed_mean": round(float(np.mean(accs)), 4),
                    "per_seed_std": round(float(np.std(accs)), 4)}
    e3 = bootstrap_diff(seeds_by_arm["rolesplit_1"][0],
                        seeds_by_arm["interleaved"][0],
                        y, GAP_PRIMARY, N_BOOT, 42)
    reg["E3_prism_rf_seed42"] = {"ci": [e3["ci_lo"], e3["ci_hi"]],
                                 "p_boot": e3["p_boot"]}
    reg["expected"] = {"interleaved": 0.610, "rolesplit_1": 0.6382,
                       "E3_ci": [0.0037898257871672378, 0.054114279212513904],
                       "E3_p": 0.018,
                       "source": "data/results/trace_h2h/eqdim_probes.json"}
    reg["reproduces_table2"] = bool(
        reg["interleaved"]["per_seed_mean"] == 0.610
        and reg["rolesplit_1"]["per_seed_mean"] == 0.6382
        and abs(e3["ci_lo"] - reg["expected"]["E3_ci"][0]) < 1e-9
        and abs(e3["ci_hi"] - reg["expected"]["E3_ci"][1]) < 1e-9
        and e3["p_boot"] == reg["expected"]["E3_p"])
    results["regression_test"] = reg
    print(f"[regression] interleaved={reg['interleaved']['per_seed_mean']} "
          f"rolesplit_1={reg['rolesplit_1']['per_seed_mean']} "
          f"E3 p={e3['p_boot']} -> reproduces Table 2: "
          f"{reg['reproduces_table2']}", flush=True)

    # --- the control, on both inference conventions --------------------
    for label, vecs in (("inference_seedens", ens),
                        ("inference_seed42",
                         {a: seeds_by_arm[a][0] for a in ens})):
        block = {}
        for a, b, kind in CONTRASTS:
            cblock = {"kind": kind, "arms": [a, b]}
            for gap in GAPS:
                d = bootstrap_partitions(vecs[a], vecs[b], y, gap, masks)
                g = {"n_pairs": results["pair_counts"][f"gap_{gap}"]}
                for part in masks:
                    _, valid = pair_matrices(y, gap)
                    vm = valid & masks[part]
                    g[part] = {
                        "acc_a": round(float(_acc_masked(vecs[a], y, vm)), 4),
                        "acc_b": round(float(_acc_masked(vecs[b], y, vm)), 4),
                        "delta": ci_p(d[part])}
                atten = ci_p(d["all"] - d["same_model"])
                g["attenuation"] = atten
                g["attenuation"]["definition"] = "Delta_all - Delta_same"
                if kind == "primary":
                    g["decision"] = decide(g["all"]["delta"],
                                           g["same_model"]["delta"], atten)
                cblock[f"gap_{gap}"] = g
                print(f"[{label}] {a}>{b} gap{gap} "
                      f"D_all={g['all']['delta']['point']:+.4f} "
                      f"{g['all']['delta']['ci']} | "
                      f"D_same={g['same_model']['delta']['point']:+.4f} "
                      f"{g['same_model']['delta']['ci']} | "
                      f"atten={atten['point']:+.4f} {atten['ci']}",
                      flush=True)
            block[f"{a}_vs_{b}"] = cblock
        results[label] = block

    prim = results["inference_seedens"]["rolesplit_1_vs_interleaved"]
    results["verdict"] = prim[f"gap_{GAP_PRIMARY}"]["decision"]["verdict"]
    print(f"[verdict] {results['verdict']}", flush=True)

    save_result("identity_controls_prism", results, dataset="prism",
                config="minilm",
                params={"seeds": SEEDS, "n_boot": N_BOOT, "issue": 97,
                        "analysis": "B", "gap_primary": GAP_PRIMARY})
    return results


def _acc_masked(preds, y, valid_mask):
    """pairwise_accuracy restricted to a pair mask (trace_h2h's metric with
    `valid` narrowed); mirrors trace_h2h.pairwise_accuracy exactly."""
    dy = y[:, None] - y[None, :]
    dp = preds[:, None] - preds[None, :]
    correct = (dp * dy) > 0
    return float(correct[valid_mask].sum() / valid_mask.sum())


run_sample = make_run_sample(run)


def main(argv=None):
    """CLI shim: python scripts/identity_controls_prism.py"""
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
