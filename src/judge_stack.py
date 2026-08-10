"""Issue #53: [bge role-split(1) | LLM judge] - do the embedder gain and the
textual-analysis channel stack?

MiniLM role-split(1)+judge = 0.8152 on SGD (#23, haiku judge); bge
role-split(1) alone = 0.802 multiseed / 0.8112 seed-ensembled (#51). This
measures the unmeasured hybrid cell [bge role-split(1) | judge rating].

Judge channel: the frozen haiku ratings (judge_ratings.json, raw external
API output, encoder-independent). Haiku is the judge behind the referenced
0.8152; the #23 opus arm was weaker on SGD (0.7961) and is not re-run.

Arms:
    uss_sgd  (primary): bge_rs1 (cached, #51), bge_rs1_judge (new),
                        minilm_rs1 (cached, eqdim #30/#21),
                        minilm_rs1_judge (new; seed 42 must reproduce #23's
                        0.8152 - recorded as a check, not asserted)
    uss_mwoz (secondary): bge_rs1 (cached, #51), bge_rs1_judge (new)
    prism    (secondary): bge_rs1, bge_rs1_judge, BOTH refit on the
                        judge-rated 2,000-conversation subset (rng(42)
                        reproduction of #23's draw; the #51 full-corpus
                        cache is not comparable on a subset protocol),
                        GroupKFold by user.

Contrasts (issue #53, calibration regime of #43/#51):
seeds [42, 7, 123]; inference on seed-ensembled OOF predictions (mean of
the per-seed vectors); dialogue-resampling bootstrap n=1000 at seed 42;
gap 1.0 primary; NO equivalence bands (instrument resolution ~+-2.5pt).
    C1 (superiority, family of one): bge_rs1_judge > bge_rs1 on SGD.
    C2 (estimation, point + CI, no pass/fail): bge_rs1_judge vs
        minilm_rs1_judge on SGD.
Secondary estimates: the C1 analog on MWOZ (expected judge-saturated) and
on the PRISM subset (judge boosts there).

Alignment verification BEFORE any fit (the likely bug is an off-by-one
join between judge ratings and dialogues):
    1. load_uss dialogue order == embedding meta json order (overall_mean
       exact float equality, per-dialogue turn counts == offsets diffs);
    2. PRISM: load_prism_dialogues order == prism_meta.json (y, user_ids),
       subset indices re-drawn with the same rng(42) recipe as #23;
    3. judge-alone strict + ties-half accuracies recomputed from the
       ratings must reproduce llm_judge_eval.json (#23) to 4dp on every
       corpus/gap - this pins the (rating_i, dialogue_i) join end to end.

Checkpoints: per (corpus, seed) npz under experiment-caches/judge_stack/,
saved after every arm (atomic tmp+rename); foreign cached predictions
(rolesplit_embedders, eqdim) are copied in so re-emission is self-contained.
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
from trace_h2h import pairwise_accuracy, bootstrap_diff, GAPS, load_corpus
from extension_arms import build_rolesplit
from multiseed_probes import oof
from multimodel_eval import load as load_mm

SEEDS = [42, 7, 123]
N_BOOT = 1000
GAP_PRIMARY = 1.0
JUDGE = "haiku"
PRISM_SUBSET = 2000

RESULTS = WORK.parent / "data" / "results" / "trace_h2h"
CKPT = WORK.parent / "data" / "experiment-caches" / "judge_stack"
RS_MM = WORK.parent / "data" / "experiment-caches" / "rolesplit_embedders"
EQDIM = WORK.parent / "data" / "experiment-caches" / "eqdim_probes"


_CKPT = Checkpoint("judge_stack", script=__file__)


def ckpt_load(tag):
    return _CKPT.load(tag) or {}


ckpt_save = _CKPT.save


def foreign(path, key):
    if path.exists():
        z = np.load(path, allow_pickle=True)
        if key in z.files:
            return z[key]
    return None


def rs1_feats(emb, offsets, meta):
    """Role-split(1) at arbitrary embedding width d: per-role degrees 0-1."""
    d = emb.shape[1]
    role_raw = build_rolesplit(emb, offsets, meta)[0]
    cols = []
    for base in (0, 4 * d):
        cols.extend(range(base, base + 2 * d))
    rs1 = role_raw[:, cols]
    assert rs1.shape[1] == 4 * d, (rs1.shape, d)
    return rs1


def judge_col(ratings, corpus, n):
    """Judge rating column, index i == dialogue i. Full coverage asserted."""
    r = ratings[JUDGE]
    n_keys = sum(1 for k in r if k.rsplit("-", 1)[0] == corpus)
    assert n_keys == n, (corpus, n_keys, n)
    vals = [r.get(f"{corpus}-{i}") for i in range(n)]
    assert all(v is not None for v in vals), f"missing ratings on {corpus}"
    return np.asarray(vals, float)


def pairwise_ties_half(scores, y, gap):
    dy = y[:, None] - y[None, :]
    valid = np.abs(dy) >= gap
    np.fill_diagonal(valid, False)
    dp = scores[:, None] - scores[None, :]
    correct = ((dp * dy) > 0).astype(float) + 0.5 * (dp == 0)
    return float(correct[valid].sum() / valid.sum())


def verify_judge_join(corpus, vals, y, ref):
    """The (rating_i, dialogue_i) join is correct iff the deterministic
    judge-alone accuracies reproduce #23's llm_judge_eval.json exactly."""
    out = {}
    for gap in GAPS:
        strict = round(pairwise_accuracy(vals, y, gap), 4)
        ties = round(pairwise_ties_half(vals, y, gap), 4)
        r = ref[corpus][JUDGE][f"gap_{gap}"]
        assert strict == r["judge_strict"], (corpus, gap, strict, r)
        assert ties == r["judge_ties_half"], (corpus, gap, ties, r)
        out[f"gap_{gap}"] = {"judge_strict": strict, "judge_ties_half": ties}
    print(f"[{corpus}] judge join verified against #23 "
          f"(strict gap1.0={out['gap_1.0']['judge_strict']})", flush=True)
    return out


def verify_uss_order(name, offsets, meta):
    from lib.uss import load_uss
    seqs, m2 = load_uss(name.split("_", 1)[1])
    assert len(seqs) == len(meta["overall_mean"]) == len(offsets) - 1
    assert [float(v) for v in m2["overall_mean"]] == \
           [float(v) for v in meta["overall_mean"]]
    assert [len(s) for s in seqs] == list(np.diff(offsets))
    assert [len(r) for r in meta["roles"]] == list(np.diff(offsets))
    print(f"[{name}] load_uss order == embedding meta order "
          f"({len(seqs)} dialogues)", flush=True)


def prism_subset_data():
    """Judge-rated PRISM subset: #23's deterministic rng(42) draw."""
    from embed_prism import load_prism_dialogues
    emb_full, offsets_full, meta, _ = load_mm("prism", "bge")
    ds = load_prism_dialogues()
    assert len(ds) == len(offsets_full) - 1 == len(meta["overall_mean"])
    assert [float(d["y"]) for d in ds] == \
           [float(v) for v in meta["overall_mean"]]
    assert [d["user_id"] for d in ds] == meta["user_ids"]
    assert [len(d["texts"]) for d in ds] == list(np.diff(offsets_full))
    rng = np.random.default_rng(42)
    idx = sorted(rng.choice(len(ds), PRISM_SUBSET, replace=False).tolist())
    chunks, off = [], [0]
    for i in idx:
        s, e = offsets_full[i], offsets_full[i + 1]
        chunks.append(emb_full[s:e])
        off.append(off[-1] + int(e - s))
    emb = np.vstack(chunks)
    offsets = np.asarray(off, dtype=np.int64)
    meta_sub = {"roles": [meta["roles"][i] for i in idx]}
    y = np.asarray(meta["overall_mean"], float)[idx]
    groups = [meta["user_ids"][i] for i in idx]
    print(f"[prism] subset verified: {len(idx)} of {len(ds)} conversations, "
          f"{emb.shape[0]} turns", flush=True)
    return emb, offsets, meta_sub, y, groups


def run_arms(corpus_key, feats, y, groups, foreign_src, order):
    """Per-seed OOF predictions for every arm, checkpointed per arm."""
    per_seed = {}
    for seed in SEEDS:
        tag = f"{corpus_key}_seed{seed}"
        done = ckpt_load(tag)
        for arm in order:
            if arm in done:
                continue
            p = None
            if arm in foreign_src:
                p = foreign(*foreign_src[arm](seed))
            if p is None:
                p = oof(feats[arm], y, seed, groups)
            done[arm] = p
            ckpt_save(tag, done)
            print(f"[{corpus_key}] seed {seed} {arm} done", flush=True)
        per_seed[seed] = done
    return per_seed


def summarize(per_seed, y, order):
    ms, ens, seedens_acc = {}, {}, {}
    for arm in order:
        accs = [{f"gap_{g}": pairwise_accuracy(per_seed[s][arm], y, g)
                 for g in GAPS} for s in SEEDS]
        ms[arm] = {f"gap_{g}": {
            "mean": round(float(np.mean([a[f"gap_{g}"] for a in accs])), 4),
            "std": round(float(np.std([a[f"gap_{g}"] for a in accs])), 4)}
            for g in GAPS}
        ens[arm] = np.mean([per_seed[s][arm] for s in SEEDS], axis=0)
        seedens_acc[arm] = {f"gap_{g}": round(
            pairwise_accuracy(ens[arm], y, g), 4) for g in GAPS}
        print("  " + " ".join(
            [f"{arm:17s}"] +
            [f"gap{g}={ms[arm][f'gap_{g}']['mean']:.4f}"
             f"+-{ms[arm][f'gap_{g}']['std']:.4f}" for g in GAPS] +
            [f"ens1.0={seedens_acc[arm]['gap_1.0']:.4f}"]), flush=True)
    return ms, ens, seedens_acc


def contrast(pa, pb, y, label, test):
    bd = bootstrap_diff(pa, pb, y, GAP_PRIMARY, N_BOOT, 42)
    point = round(pairwise_accuracy(pa, y, GAP_PRIMARY)
                  - pairwise_accuracy(pb, y, GAP_PRIMARY), 4)
    out = {"h": label, "test": test, "gap": GAP_PRIMARY, "point": point,
           "diff_ci": [bd["ci_lo"], bd["ci_hi"]], "p_boot": bd["p_boot"]}
    print(f"[{test}] {label:42s} point={point:+.4f} "
          f"CI [{bd['ci_lo']:+.4f},{bd['ci_hi']:+.4f}] p={bd['p_boot']:.4f}",
          flush=True)
    return out


def run(max_samples=None):
    if max_samples is not None:
        raise ValueError(
            "judge_stack is full-corpus-bound: the frozen judge ratings "
            "(judge_ratings.json) are index-aligned to the full corpora "
            "(PRISM to a fixed seeded 2000-conversation subset), so "
            "max_samples cannot apply here; use reproduce.py "
            "--max-samples N, which excludes the judge stages.")
    CKPT.mkdir(parents=True, exist_ok=True)
    ratings = json.load(open(RESULTS / "judge_ratings.json"))
    ref23 = json.load(open(RESULTS / "llm_judge_eval.json"))
    results = {"seeds": SEEDS, "gap_primary": GAP_PRIMARY, "judge": JUDGE,
               "n_boot": N_BOOT, "inference": "seed-ensembled",
               "calibration": "no equivalence bands below +-2.5pt "
                              "(measured instrument resolution, #43)"}
    ens_all, ys = {}, {}

    # ---------- uss_sgd (primary) ----------
    emb, offsets, meta, groups = load_mm("uss_sgd", "bge")
    verify_uss_order("uss_sgd", offsets, meta)
    y = np.asarray(meta["overall_mean"], float)
    vals = judge_col(ratings, "uss_sgd", len(y))
    judge_alone = verify_judge_join("uss_sgd", vals, y, ref23)
    bge_rs1 = rs1_feats(emb, offsets, meta)
    del emb
    emb_m, offsets_m, meta_m = load_corpus("uss_sgd")
    assert list(np.diff(offsets_m)) == list(np.diff(offsets))
    minilm_rs1 = rs1_feats(emb_m, offsets_m, meta_m)
    del emb_m
    feats = {
        "bge_rs1": bge_rs1,
        "bge_rs1_judge": np.hstack([bge_rs1, vals[:, None]]),
        "minilm_rs1": minilm_rs1,
        "minilm_rs1_judge": np.hstack([minilm_rs1, vals[:, None]]),
    }
    order = ["bge_rs1", "bge_rs1_judge", "minilm_rs1", "minilm_rs1_judge"]
    foreign_src = {
        "bge_rs1": lambda s: (RS_MM / f"bge_uss_sgd_seed{s}.npz",
                              "rolesplit_k1"),
        "minilm_rs1": lambda s: (EQDIM / f"uss_sgd_seed{s}.npz",
                                 "rolesplit_k1"),
    }
    per_seed = run_arms("uss_sgd", feats, y, groups, foreign_src, order)
    ms, ens, seedens = summarize(per_seed, y, order)
    rep42 = round(pairwise_accuracy(per_seed[42]["minilm_rs1_judge"], y,
                                    GAP_PRIMARY), 4)
    ref_val = ref23["uss_sgd"][JUDGE]["gap_1.0"]["rolesplit_k1_plus_judge"]
    repro = {"minilm_rs1_judge_seed42_gap1.0": rep42,
             "reference_23": ref_val, "match": bool(abs(rep42 - ref_val) < 5e-4)}
    if not repro["match"]:
        print(f"WARNING: #23 reproduction off: {rep42} vs {ref_val}",
              flush=True)
    else:
        print(f"[uss_sgd] #23 reproduction check: {rep42} == {ref_val}",
              flush=True)
    results["uss_sgd"] = {"n": len(y), "judge_alone": judge_alone,
                          "multiseed": ms, "accuracy_seedens": seedens,
                          "reproduction_23": repro}
    ens_all["uss_sgd"], ys["uss_sgd"] = ens, y
    del feats, bge_rs1, minilm_rs1

    # ---------- uss_mwoz (secondary) ----------
    emb, offsets, meta, groups = load_mm("uss_mwoz", "bge")
    verify_uss_order("uss_mwoz", offsets, meta)
    y = np.asarray(meta["overall_mean"], float)
    vals = judge_col(ratings, "uss_mwoz", len(y))
    judge_alone = verify_judge_join("uss_mwoz", vals, y, ref23)
    bge_rs1 = rs1_feats(emb, offsets, meta)
    del emb
    feats = {"bge_rs1": bge_rs1,
             "bge_rs1_judge": np.hstack([bge_rs1, vals[:, None]])}
    order = ["bge_rs1", "bge_rs1_judge"]
    foreign_src = {"bge_rs1": lambda s: (RS_MM / f"bge_uss_mwoz_seed{s}.npz",
                                         "rolesplit_k1")}
    per_seed = run_arms("uss_mwoz", feats, y, groups, foreign_src, order)
    ms, ens, seedens = summarize(per_seed, y, order)
    results["uss_mwoz"] = {"n": len(y), "judge_alone": judge_alone,
                           "multiseed": ms, "accuracy_seedens": seedens}
    ens_all["uss_mwoz"], ys["uss_mwoz"] = ens, y
    del feats, bge_rs1

    # ---------- prism (secondary, judge-rated subset, refit both arms) ----
    emb, offsets, meta_sub, y, groups = prism_subset_data()
    vals = judge_col(ratings, "prism", len(y))
    judge_alone = verify_judge_join("prism", vals, y, ref23)
    bge_rs1 = rs1_feats(emb, offsets, meta_sub)
    del emb
    feats = {"bge_rs1": bge_rs1,
             "bge_rs1_judge": np.hstack([bge_rs1, vals[:, None]])}
    order = ["bge_rs1", "bge_rs1_judge"]
    per_seed = run_arms("prism", feats, y, groups, {}, order)
    ms, ens, seedens = summarize(per_seed, y, order)
    results["prism"] = {
        "n": len(y), "judge_alone": judge_alone, "multiseed": ms,
        "accuracy_seedens": seedens,
        "note": "judge-rated 2000-conversation subset (#23 rng(42) draw), "
                "GroupKFold by user (unseeded splits; seed variance is "
                "RF-only); bge_rs1 REFIT on the subset - the #51 "
                "full-corpus cache is not comparable under a subset "
                "protocol"}
    ens_all["prism"], ys["prism"] = ens, y
    del feats, bge_rs1

    # ---------- inference ----------
    c1 = contrast(ens_all["uss_sgd"]["bge_rs1_judge"],
                  ens_all["uss_sgd"]["bge_rs1"], ys["uss_sgd"],
                  "C1 bge_rs1_judge>bge_rs1 SGD", "superiority")
    c1["p_holm"] = c1["p_boot"]  # family of one
    c2 = contrast(ens_all["uss_sgd"]["bge_rs1_judge"],
                  ens_all["uss_sgd"]["minilm_rs1_judge"], ys["uss_sgd"],
                  "C2 bge_rs1_judge vs minilm_rs1_judge SGD", "estimation")
    results["primary_family"] = [c1]
    results["C2_estimation"] = c2
    results["secondary_estimates"] = [
        contrast(ens_all["uss_mwoz"]["bge_rs1_judge"],
                 ens_all["uss_mwoz"]["bge_rs1"], ys["uss_mwoz"],
                 "S1 bge_rs1_judge>bge_rs1 MWOZ", "estimation"),
        contrast(ens_all["prism"]["bge_rs1_judge"],
                 ens_all["prism"]["bge_rs1"], ys["prism"],
                 "S2 bge_rs1_judge>bge_rs1 PRISM-subset", "estimation"),
    ]

    # ---------- distance to the annotator ceiling (SGD, gap 1.0) ----------
    ceil = json.load(open(RESULTS / "annotator_ceiling.json"))
    ceiling = ceil["uss_sgd"]["gap_1.0"]["ceiling"]
    sgd_acc = results["uss_sgd"]["accuracy_seedens"]
    best = max(sgd_acc, key=lambda a: sgd_acc[a]["gap_1.0"])
    results["ceiling_distance_sgd"] = {
        "ceiling_gap_1.0": ceiling, "best_arm": best,
        "accuracy_seedens": sgd_acc[best]["gap_1.0"],
        "distance": round(ceiling - sgd_acc[best]["gap_1.0"], 4)}
    print(f"[ceiling] SGD best={best} acc={sgd_acc[best]['gap_1.0']:.4f} "
          f"ceiling={ceiling} distance={results['ceiling_distance_sgd']['distance']:.4f}",
          flush=True)

    from campaign_io import save_result
    save_result("judge_stack", results, dataset="uss_sgd.uss_mwoz.prism",
                config="bge_minilm",
                params={"seeds": SEEDS, "n_folds": 5, "n_boot": N_BOOT,
                        "judge": JUDGE, "inference": "seed-ensembled",
                        "issues": "#53"})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
