"""Issue #97 Analysis C1: identity augmentation - the mechanism test.

B (identity_controls_prism.py) controls the EVALUATION: it restricts the
metric to same-model pairs. C controls the INFORMATION: it hands model
identity to both arms explicitly and asks whether the split still wins.

    interleaved + model-ID   vs   role-split(1)      <- the diagnostic contrast
    role-split(1) + model-ID vs   role-split(1)      <- secondary (see below)
    role-split(1) + model-ID vs   interleaved + model-ID

Reading the diagnostic contrast (pre-stated, #97):
  - if handing the interleaved arm identity for free CLOSES the gap to the
    split, the split's gain was substantially identity;
  - if role-split(1) + model-ID still beats interleaved + model-ID by roughly
    the original margin, the gain is content beyond identity.

Why C is a GENEROUS test of H_confound (and therefore a strong exoneration
if the gap survives): Analysis A measured that the interleaved pooled mean
already recovers model ID well above chance on its own, while the model-turn
mean recovers it substantially better. The one-hot block hands the
interleaved arm BOTH the identity it cannot recover unaided AND perfect
precision on the rest - strictly more identity than role-splitting could
ever supply.

The secondary contrast (role-split(1)+ID vs plain role-split(1)) is
deliberately NOT read alone: "model-ID adds nothing on top of the split" is
consistent with BOTH readings (the split already encodes identity; or
identity is simply not very useful here). It is interpreted jointly with the
diagnostic contrast, per #97.

Leakage: the one-hot block is a fixed encoding of an observed conversation
attribute - no fitting, so nothing leaks across folds. (C2, fingerprint
removal by within-model centering, WOULD need per-fold fitting; it is the
alternative variant in #97 and is not implemented here - C1 was chosen for
exactly that simplicity.)

Protocol: identical to eqdim_probes/Table 2 - RandomForestRegressor(300,
min_samples_leaf=2) via multiseed_probes.oof, GroupKFold(5) by user_id,
seeds {42, 7, 123, 2024, 31337}, seed-ensembled inference (Section 5.2),
m-out-of-n dialogue bootstrap. The plain (un-augmented) arms are REPLAYED
from the eqdim_probes cache rather than recomputed, so this script's
baseline is bit-identical to the published Table 2 arms.

Checkpoints: data/experiment-caches/identity_augment_prism/ (per seed; the
augmented arms are the only real compute). Registry: lib.campaign.save_result.
"""
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))

from lib.campaign import Checkpoint, cap, load_corpus, make_run_sample, save_result
from trace_h2h import build_all_arms, pairwise_accuracy, bootstrap_diff, GAPS
from extension_arms import build_rolesplit
from degree_ablation import role_k_slice
from multiseed_probes import oof
from identity_probes_prism import prism_model_labels

SEEDS = [42, 7, 123, 2024, 31337]
N_BOOT = 1000
GAP_PRIMARY = 1.0

_CKPT = Checkpoint("identity_augment_prism", script=__file__)
_EQ = Checkpoint("eqdim_probes")            # reader: plain arms, bit-identical

CONTRASTS = [
    ("interleaved_id", "rolesplit_1", "diagnostic"),
    ("rolesplit_1_id", "rolesplit_1", "secondary_not_alone"),
    ("rolesplit_1_id", "interleaved_id", "both_augmented"),
    ("rolesplit_1", "interleaved", "reference_table2"),
]


def one_hot(labels):
    """(n, n_models) one-hot of the chosen model. Fixed encoding - no fit,
    so no cross-fold leakage."""
    levels = sorted(set(labels))
    ix = {m: i for i, m in enumerate(levels)}
    M = np.zeros((len(labels), len(levels)))
    M[np.arange(len(labels)), [ix[m] for m in labels]] = 1.0
    return M, levels


def plain_arm(seed, arm, ephemeral):
    """Replay a published Table 2 arm from the eqdim_probes cache."""
    if ephemeral:
        return None
    d = _EQ.load(f"prism_seed{seed}")
    key = {"interleaved": "raw", "rolesplit_1": "rolesplit_k1"}[arm]
    if d is None or key not in d:
        return None
    return np.asarray(d[key], float)


def run(max_samples=None):
    """#97 C1: retrain both arms with a one-hot model-ID block.

    max_samples is not None -> ephemeral capped run (first-N dialogues, no
    checkpoint reads/writes, no save_result; lib/campaign.py contract). The
    plain-arm replay is bypassed too - full-N cache entries must never be
    mixed with sampled features.
    """
    ephemeral = max_samples is not None
    emb, offsets, meta, groups = load_corpus("prism", max_samples=max_samples)
    labels = prism_model_labels(max_samples=max_samples)
    y = np.asarray(meta["overall_mean"], float)
    assert len(labels) == len(offsets) - 1 == len(y), \
        f"label/corpus misalignment: {len(labels)} vs {len(offsets) - 1}"

    interleaved = build_all_arms(emb, offsets, meta, degree=3)["design_b_raw"]
    role_raw, _ = build_rolesplit(emb, offsets, meta)
    rolesplit_1 = role_k_slice(role_raw, 1)
    assert rolesplit_1.shape[1] == interleaved.shape[1], \
        (rolesplit_1.shape, interleaved.shape)
    ident, levels = one_hot(labels)

    X = {"interleaved": interleaved, "rolesplit_1": rolesplit_1,
         "interleaved_id": np.hstack([interleaved, ident]),
         "rolesplit_1_id": np.hstack([rolesplit_1, ident])}
    results = {
        "seeds": SEEDS, "n_boot": N_BOOT, "n_dialogues": int(len(y)),
        "n_models": len(levels), "models": levels,
        "dims": {k: int(v.shape[1]) for k, v in X.items()},
        "protocol": {"head": "RandomForestRegressor(300, min_samples_leaf=2)",
                     "cv": "GroupKFold(5) by user_id",
                     "inference": "seed-ensembled (Section 5.2)"},
    }
    print(f"[dims] {results['dims']} ({len(levels)} models)", flush=True)

    per_seed = {k: [] for k in X}
    for seed in SEEDS:
        tag = f"prism_seed{seed}"
        cached = None if ephemeral else _CKPT.load(tag)
        preds = {}
        for k in X:
            if cached is not None and k in cached:
                preds[k] = np.asarray(cached[k], float)
                continue
            replay = (plain_arm(seed, k, ephemeral)
                      if k in ("interleaved", "rolesplit_1") else None)
            if replay is not None:
                preds[k] = replay          # bit-identical to Table 2
            else:
                preds[k] = oof(X[k], y, seed, groups)
        if not ephemeral:
            _CKPT.save(tag, preds)
        for k in X:
            per_seed[k].append(preds[k])
        print(f"[C1] seed {seed} done", flush=True)

    ens = {k: np.mean(v, axis=0) for k, v in per_seed.items()}
    acc = {}
    for k in X:
        accs = [pairwise_accuracy(p, y, GAP_PRIMARY) for p in per_seed[k]]
        acc[k] = {"per_seed_mean": round(float(np.mean(accs)), 4),
                  "per_seed_std": round(float(np.std(accs)), 4),
                  "seedens": round(
                      float(pairwise_accuracy(ens[k], y, GAP_PRIMARY)), 4)}
        print(f"[C1] {k:16s} {acc[k]['per_seed_mean']:.4f}"
              f"+-{acc[k]['per_seed_std']:.4f} "
              f"(seedens {acc[k]['seedens']:.4f})", flush=True)
    results["accuracy"] = acc

    contrasts = {}
    for a, b, kind in CONTRASTS:
        block = {"kind": kind, "arms": [a, b]}
        for gap in GAPS:
            bd = bootstrap_diff(ens[a], ens[b], y, gap, N_BOOT, 42)
            block[f"gap_{gap}"] = {
                "delta": round(float(pairwise_accuracy(ens[a], y, gap)
                                     - pairwise_accuracy(ens[b], y, gap)), 4),
                "ci": [round(bd["ci_lo"], 4), round(bd["ci_hi"], 4)],
                "p_boot": bd["p_boot"]}
        g = block[f"gap_{GAP_PRIMARY}"]
        print(f"[C1] {a} > {b} ({kind}): d={g['delta']:+.4f} "
              f"{g['ci']} p={g['p_boot']}", flush=True)
        contrasts[f"{a}_vs_{b}"] = block
    results["contrasts"] = contrasts

    # --- verdict (pre-stated readings, #97) ----------------------------
    # The DECISIVE contrast is rolesplit_1_id vs interleaved_id, per #97:
    # "if role-split(1) + model-ID still beats interleaved + model-ID by
    # roughly the original margin, the gain is content beyond identity."
    #
    # The diagnostic contrast (interleaved_id vs rolesplit_1) is reported but
    # must NOT be read as equivalence when its CI spans zero: failure to
    # distinguish at a +-2.5-point instrument is not evidence the gap closed
    # (Section 5.2 - equivalence statements are estimation-framed). An earlier
    # version of this rule made exactly that error and inverted the verdict.
    diag = contrasts["interleaved_id_vs_rolesplit_1"][f"gap_{GAP_PRIMARY}"]
    both = contrasts["rolesplit_1_id_vs_interleaved_id"][f"gap_{GAP_PRIMARY}"]
    ref = contrasts["rolesplit_1_vs_interleaved"][f"gap_{GAP_PRIMARY}"]
    retained = (both["delta"] / ref["delta"]) if ref["delta"] else float("nan")
    survives = (not (both["ci"][0] <= 0 <= both["ci"][1])) and retained >= 0.5
    results["verdict"] = {
        "reference_delta_rolesplit_vs_interleaved": ref["delta"],
        "delta_both_augmented": both["delta"],
        "margin_retained_under_identity": round(float(retained), 3),
        "delta_interleaved_id_vs_rolesplit": diag["delta"],
        "identity_increment": {
            k: round(acc[f"{k}_id"]["seedens"] - acc[k]["seedens"], 4)
            for k in ("interleaved", "rolesplit_1")},
        "gain_is_content_beyond_identity": bool(survives),
        "rule": (
            "decisive = rolesplit_1_id vs interleaved_id: CI excludes zero "
            "AND >=50% of the reference margin retained when BOTH arms hold "
            "model identity. C1 is a GENEROUS test of H_confound: the "
            "one-hot block hands interleaved more identity than the split "
            "could encode (A: pooled mean recovers model ID unaided)."),
        "reading": (
            "role-split(1) retains its margin over the interleaved arm when "
            "BOTH are handed model identity: the gain is content beyond "
            "model identity, not a fingerprint effect - corroborates B"
            if survives else
            "the split's margin collapses once the interleaved arm holds "
            "model identity: consistent with the gain being substantially "
            "identity - read jointly with B")}
    print(f"[verdict] retained={retained:.2f} of the reference margin; "
          f"{results['verdict']['reading']}", flush=True)

    if not ephemeral:
        save_result("identity_augment_prism", results, dataset="prism",
                    config="minilm",
                    params={"seeds": SEEDS, "n_boot": N_BOOT, "issue": 97,
                            "analysis": "C1", "gap_primary": GAP_PRIMARY})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    """CLI shim: python scripts/identity_augment_prism.py"""
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
