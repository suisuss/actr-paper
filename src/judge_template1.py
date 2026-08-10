"""Prompt-faithful judge re-rate: TRACE Appendix D Template 1, verbatim.

User directive (2026-07-10, recorded on #53): the judge-channel ratings must
use the prompt from the original paper. The frozen judge_ratings.json (#23)
used a minimal in-house prompt; every judge/hybrid cell (#23, #53) therefore
measured OUR judge instantiation, not TRACE's configuration. This script
regenerates the haiku ratings with Template 1 (arXiv:2511.08394 Appendix D.1,
the paper's final baseline prompt) and re-emits the judge cells.

Faithfulness notes (documented, not silent):
  - Template 1 text is verbatim from the paper PDF (material/literature/
    paper/2511.08394v1.pdf), with PDF line-wraps joined and typographic
    apostrophes normalized to ASCII. List items and the 1-5 anchors keep
    their own lines; the trailing "SATISFACTION SCORE (1-5):" scaffold is
    kept.
  - Transcript speaker labels follow the template's nouns: "User:" /
    "Model:" (the paper does not specify transcript formatting; #23 used
    USER/ASSISTANT).
  - TRACE never names its judge model; we keep claude-haiku-4-5 (the
    cheap-scalable-auto-rater role their baseline mirrors). Prompt is now
    theirs; model remains ours. Cross-paper number comparisons stay out of
    bounds either way (their corpus is proprietary).
  - judge_ratings.json stays frozen; this writes judge_ratings_t1.json.

Modes:
    submit  - build transcripts, estimate cost (abort > $10), create batch
    collect - poll batch, parse integer ratings, save judge_ratings_t1.json
    eval    - re-emit the judge cells with the T1 column:
              arms per corpus (geometry arms REUSED from judge_stack/#53
              checkpoints, never refit): SGD [bge_rs1, minilm_rs1] +
              new fits [bge_rs1_judge_t1, minilm_rs1_judge_t1]; MWOZ
              [bge_rs1] + [bge_rs1_judge_t1]; PRISM rated subset [bge_rs1]
              + [bge_rs1_judge_t1]. Old-prompt judge arms loaded from the
              same checkpoints for prompt-sensitivity contrasts.

Inference (seeds [42,7,123], seed-ensembled, bootstrap
n=1000 at seed 42, gap 1.0 primary, no equivalence bands below +-2.5pt):
    T1-C1 (superiority, family of one): bge_rs1_judge_t1 > bge_rs1 on SGD
          - the #53 primary question under the paper's prompt.
    T1-C2 (estimation): bge_rs1_judge_t1 vs bge_rs1_judge (old prompt), SGD
          - prompt sensitivity of the hybrid.
    T1-C3 (estimation): minilm_rs1_judge_t1 vs minilm_rs1_judge (the #23
          0.8152 arm), SGD.
    Secondary estimates: T1 stacking analogs on MWOZ and PRISM subset.
    Judge-alone strict/ties-half accuracies, T1 vs old, all corpora.

Checkpoints: per (corpus, seed) npz under experiment-caches/judge_template1/,
saved after every arm (atomic tmp+rename). Alignment gates identical to
judge_stack.py (dialogue-order verification; full-coverage assertion on
the T1 rating column).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "scripts"))

from lib.campaign import Checkpoint, make_run_sample
from trace_h2h import pairwise_accuracy, bootstrap_diff, GAPS, load_corpus
from multiseed_probes import oof
from multimodel_eval import load as load_mm
from judge_stack import (rs1_feats, judge_col, pairwise_ties_half,
                         verify_uss_order, prism_subset_data,
                         summarize, contrast)
import judge_stack as js

SEEDS = [42, 7, 123]
N_BOOT = 1000
GAP_PRIMARY = 1.0
PRISM_SUBSET = 2000
COST_GUARD_USD = 10.0
MODEL = "claude-haiku-4-5"

RESULTS = WORK.parent / "data" / "results" / "trace_h2h"
STATE = RESULTS / "judge_state_t1.json"
RATINGS_T1 = RESULTS / "judge_ratings_t1.json"
CKPT = WORK.parent / "data" / "experiment-caches" / "judge_template1"

# Template 1, verbatim (arXiv:2511.08394 Appendix D.1); see module docstring
# for the two normalizations (line-wrap joining, ASCII apostrophes).
PROMPT_T1 = (
    "You are an impartial and objective evaluator of user satisfaction in "
    "conversational AI interactions.\n"
    "Your task is to carefully review the following transcript of an "
    "interaction between a User and an AI Model.\n"
    "Based ONLY on this transcript, provide a numerical rating representing "
    "the User's likely satisfaction with the interaction.\n"
    "Use the following 1-5 scale:\n"
    "1: Very dissatisfied\n"
    "2: Somewhat dissatisfied\n"
    "3: Neither satisfied nor dissatisfied\n"
    "4: Somewhat satisfied\n"
    "5: Very satisfied\n"
    "When making your assessment, consider factors such as:\n"
    "- Was the AI Model's response relevant to the User's prompts?\n"
    "- Did the AI Model seem to understand the User's intent?\n"
    "- Was the User likely to achieve their goal by the end of the "
    "interaction?\n"
    "- Was the interaction efficient, or did the User have to struggle?\n"
    "- Was the AI Model polite and the tone appropriate?\n"
    "Your response MUST be a single integer from 1 to 5 only. Do not "
    "provide any explanation, additional text, or punctuation.\n"
    "TRANSCRIPT:\n"
    "{transcript_text}\n"
    "SATISFACTION SCORE (1-5):"
)


def load_env_key():
    for line in open(WORK / ".env"):
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.strip().split("=", 1)[1]


def transcript(texts, roles):
    lines = []
    for t, r in zip(texts, roles):
        speaker = "User" if r == "USER" else "Model"
        lines.append(f"{speaker}: {t}")
    return "\n".join(lines)


def corpora():
    """Same coverage and ordering as #23's haiku job."""
    from lib.uss import load_uss
    from embed_prism import load_prism_dialogues
    out = {}
    for name in ["uss_sgd", "uss_mwoz"]:
        seqs, meta = load_uss(name.split("_", 1)[1])
        out[name] = {"texts": seqs, "roles": meta["roles"],
                     "y": meta["overall_mean"]}
    ds = load_prism_dialogues()
    rng = np.random.default_rng(42)
    idx = sorted(rng.choice(len(ds), PRISM_SUBSET, replace=False).tolist())
    out["prism"] = {
        "texts": [ds[i]["texts"] for i in idx],
        "roles": [ds[i]["roles"] for i in idx],
        "y": [ds[i]["y"] for i in idx],
    }
    return out


def submit():
    load_env_key()
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic()
    data = corpora()

    est_toks = 0.0
    for corpus in data:
        for t, r in zip(data[corpus]["texts"], data[corpus]["roles"]):
            est_toks += len(PROMPT_T1.format(
                transcript_text=transcript(t, r))) / 4
    est = est_toks / 1e6 * 1.0 * 0.5  # haiku $1/MTok input, batch 50% off
    print(f"estimated batched input cost: ${est:.2f}", flush=True)
    if est > COST_GUARD_USD:
        raise SystemExit(f"cost estimate ${est:.2f} exceeds guard "
                         f"${COST_GUARD_USD}")

    requests = []
    for corpus in ["uss_sgd", "uss_mwoz", "prism"]:
        for i in range(len(data[corpus]["texts"])):
            tx = transcript(data[corpus]["texts"][i],
                            data[corpus]["roles"][i])
            requests.append(Request(
                custom_id=f"{corpus}-{i}",
                params=MessageCreateParamsNonStreaming(
                    model=MODEL, max_tokens=8,
                    messages=[{"role": "user",
                               "content": PROMPT_T1.format(
                                   transcript_text=tx)}]),
            ))
    batch = client.messages.batches.create(requests=requests)
    RESULTS.mkdir(parents=True, exist_ok=True)
    state = {"batches": {"haiku_t1": {
        "id": batch.id, "n": len(requests), "model": MODEL,
        "prompt": "trace_appendix_d_template_1"}}}
    with open(STATE, "w") as f:
        json.dump(state, f, indent=1)
    print(f"[haiku_t1] batch {batch.id} submitted "
          f"({len(requests)} requests)", flush=True)
    return state


def collect():
    load_env_key()
    import anthropic
    client = anthropic.Anthropic()
    state = json.load(open(STATE))
    b = state["batches"]["haiku_t1"]
    while True:
        batch = client.messages.batches.retrieve(b["id"])
        print(f"[haiku_t1] {batch.processing_status} "
              f"(ok={batch.request_counts.succeeded} "
              f"err={batch.request_counts.errored})", flush=True)
        if batch.processing_status == "ended":
            break
        time.sleep(60)
    r = {}
    for result in client.messages.batches.results(b["id"]):
        if result.result.type != "succeeded":
            continue
        msg = result.result.message
        text = next((blk.text for blk in msg.content
                     if blk.type == "text"), "")
        val = None
        for tok in text.split():
            t = tok.strip(".:,;")
            if t.isdigit() and 1 <= int(t) <= 5:
                val = int(t)
                break
        r[result.custom_id] = val
    print(f"[haiku_t1] collected {len(r)} ratings "
          f"({sum(1 for v in r.values() if v is None)} unparsable)",
          flush=True)
    ratings = {"haiku": r}
    with open(RATINGS_T1, "w") as f:
        json.dump(ratings, f)
    print("saved", RATINGS_T1, flush=True)
    return ratings


_CKPT = Checkpoint("judge_template1", script=__file__)


def ckpt_load(tag):
    return _CKPT.load(tag) or {}


ckpt_save = _CKPT.save

_JS = Checkpoint("judge_stack")


def js_pred(corpus, seed, key):
    """Load a #53 judge_stack cached OOF prediction vector (never refit)."""
    tag = f"{corpus}_seed{seed}"
    d = _JS.load(tag)
    assert d is not None, \
        f"missing judge_stack checkpoint {_JS.dir / (tag + '.npz')}"
    assert key in d, (_JS.dir / (tag + ".npz"), key, sorted(d))
    return d[key]


def run_arms_t1(corpus_key, feats, y, groups, reuse_keys, order):
    """Per-seed OOF preds; geometry + old-judge arms copied from #53 cache,
    T1 arms fit fresh. Checkpointed per arm under judge_template1/."""
    per_seed = {}
    for seed in SEEDS:
        tag = f"{corpus_key}_seed{seed}"
        done = ckpt_load(tag)
        for arm in order:
            if arm in done:
                continue
            if arm in reuse_keys:
                done[arm] = js_pred(corpus_key, seed, reuse_keys[arm])
            else:
                done[arm] = oof(feats[arm], y, seed, groups)
            ckpt_save(tag, done)
            print(f"[{corpus_key}] seed {seed} {arm} done", flush=True)
        per_seed[seed] = done
    return per_seed


def judge_alone(vals, y):
    return {f"gap_{g}": {
        "judge_strict": round(pairwise_accuracy(vals, y, g), 4),
        "judge_ties_half": round(pairwise_ties_half(vals, y, g), 4)}
        for g in GAPS}


def evaluate():
    CKPT.mkdir(parents=True, exist_ok=True)
    ratings_t1 = json.load(open(RATINGS_T1))
    ref23 = json.load(open(RESULTS / "llm_judge_eval.json"))
    old = json.load(open(RESULTS / "judge_ratings.json"))
    results = {"seeds": SEEDS, "gap_primary": GAP_PRIMARY,
               "judge_model": MODEL,
               "prompt": "trace_appendix_d_template_1",
               "n_boot": N_BOOT, "inference": "seed-ensembled",
               "calibration": "no equivalence bands below +-2.5pt (#43)"}
    ens_all, ys = {}, {}

    # ---------- uss_sgd (primary) ----------
    emb, offsets, meta, groups = load_mm("uss_sgd", "bge")
    verify_uss_order("uss_sgd", offsets, meta)
    y = np.asarray(meta["overall_mean"], float)
    # old-prompt join re-verified (pins dialogue order end to end), then T1
    verify_old = js.verify_judge_join("uss_sgd",
                                      judge_col(old, "uss_sgd", len(y)),
                                      y, ref23)
    vals = judge_col(ratings_t1, "uss_sgd", len(y))
    bge_rs1 = rs1_feats(emb, offsets, meta)
    del emb
    emb_m, offsets_m, meta_m = load_corpus("uss_sgd")
    assert list(np.diff(offsets_m)) == list(np.diff(offsets))
    minilm_rs1 = rs1_feats(emb_m, offsets_m, meta_m)
    del emb_m
    feats = {
        "bge_rs1_judge_t1": np.hstack([bge_rs1, vals[:, None]]),
        "minilm_rs1_judge_t1": np.hstack([minilm_rs1, vals[:, None]]),
    }
    order = ["bge_rs1", "minilm_rs1", "bge_rs1_judge", "minilm_rs1_judge",
             "bge_rs1_judge_t1", "minilm_rs1_judge_t1"]
    reuse = {"bge_rs1": "bge_rs1", "minilm_rs1": "minilm_rs1",
             "bge_rs1_judge": "bge_rs1_judge",
             "minilm_rs1_judge": "minilm_rs1_judge"}
    per_seed = run_arms_t1("uss_sgd", feats, y, groups, reuse, order)
    ms, ens, seedens = summarize(per_seed, y, order)
    results["uss_sgd"] = {
        "n": len(y),
        "judge_alone_t1": judge_alone(vals, y),
        "judge_alone_old": verify_old,
        "multiseed": ms, "accuracy_seedens": seedens}
    ens_all["uss_sgd"], ys["uss_sgd"] = ens, y
    del feats, bge_rs1, minilm_rs1

    # ---------- uss_mwoz (secondary) ----------
    emb, offsets, meta, groups = load_mm("uss_mwoz", "bge")
    verify_uss_order("uss_mwoz", offsets, meta)
    y = np.asarray(meta["overall_mean"], float)
    verify_old = js.verify_judge_join("uss_mwoz",
                                      judge_col(old, "uss_mwoz", len(y)),
                                      y, ref23)
    vals = judge_col(ratings_t1, "uss_mwoz", len(y))
    bge_rs1 = rs1_feats(emb, offsets, meta)
    del emb
    feats = {"bge_rs1_judge_t1": np.hstack([bge_rs1, vals[:, None]])}
    order = ["bge_rs1", "bge_rs1_judge", "bge_rs1_judge_t1"]
    reuse = {"bge_rs1": "bge_rs1", "bge_rs1_judge": "bge_rs1_judge"}
    per_seed = run_arms_t1("uss_mwoz", feats, y, groups, reuse, order)
    ms, ens, seedens = summarize(per_seed, y, order)
    results["uss_mwoz"] = {
        "n": len(y),
        "judge_alone_t1": judge_alone(vals, y),
        "judge_alone_old": verify_old,
        "multiseed": ms, "accuracy_seedens": seedens}
    ens_all["uss_mwoz"], ys["uss_mwoz"] = ens, y
    del feats, bge_rs1

    # ---------- prism (secondary, judge-rated subset) ----------
    emb, offsets, meta_sub, y, groups = prism_subset_data()
    verify_old = js.verify_judge_join("prism",
                                      judge_col(old, "prism", len(y)),
                                      y, ref23)
    vals = judge_col(ratings_t1, "prism", len(y))
    bge_rs1 = rs1_feats(emb, offsets, meta_sub)
    del emb
    feats = {"bge_rs1_judge_t1": np.hstack([bge_rs1, vals[:, None]])}
    order = ["bge_rs1", "bge_rs1_judge", "bge_rs1_judge_t1"]
    reuse = {"bge_rs1": "bge_rs1", "bge_rs1_judge": "bge_rs1_judge"}
    per_seed = run_arms_t1("prism", feats, y, groups, reuse, order)
    ms, ens, seedens = summarize(per_seed, y, order)
    results["prism"] = {
        "n": len(y),
        "judge_alone_t1": judge_alone(vals, y),
        "judge_alone_old": verify_old,
        "multiseed": ms, "accuracy_seedens": seedens,
        "note": "judge-rated 2000-conversation subset (#23 rng(42) draw), "
                "GroupKFold by user; geometry and old-judge arms reused "
                "from #53 checkpoints"}
    ens_all["prism"], ys["prism"] = ens, y
    del feats, bge_rs1

    # ---------- inference ----------
    c1 = contrast(ens_all["uss_sgd"]["bge_rs1_judge_t1"],
                  ens_all["uss_sgd"]["bge_rs1"], ys["uss_sgd"],
                  "T1-C1 bge_rs1_judge_t1>bge_rs1 SGD", "superiority")
    c1["p_holm"] = c1["p_boot"]  # family of one
    results["primary_family"] = [c1]
    results["estimates"] = [
        contrast(ens_all["uss_sgd"]["bge_rs1_judge_t1"],
                 ens_all["uss_sgd"]["bge_rs1_judge"], ys["uss_sgd"],
                 "T1-C2 t1 vs old-prompt hybrid SGD", "estimation"),
        contrast(ens_all["uss_sgd"]["minilm_rs1_judge_t1"],
                 ens_all["uss_sgd"]["minilm_rs1_judge"], ys["uss_sgd"],
                 "T1-C3 minilm t1 vs old-prompt (#23 arm) SGD",
                 "estimation"),
        contrast(ens_all["uss_mwoz"]["bge_rs1_judge_t1"],
                 ens_all["uss_mwoz"]["bge_rs1"], ys["uss_mwoz"],
                 "T1-S1 bge_rs1_judge_t1>bge_rs1 MWOZ", "estimation"),
        contrast(ens_all["prism"]["bge_rs1_judge_t1"],
                 ens_all["prism"]["bge_rs1"], ys["prism"],
                 "T1-S2 bge_rs1_judge_t1>bge_rs1 PRISM-subset",
                 "estimation"),
    ]

    ceil = json.load(open(RESULTS / "annotator_ceiling.json"))
    ceiling = ceil["uss_sgd"]["gap_1.0"]["ceiling"]
    sgd_acc = results["uss_sgd"]["accuracy_seedens"]
    best = max(sgd_acc, key=lambda a: sgd_acc[a]["gap_1.0"])
    results["ceiling_distance_sgd"] = {
        "ceiling_gap_1.0": ceiling, "best_arm": best,
        "accuracy_seedens": sgd_acc[best]["gap_1.0"],
        "distance": round(ceiling - sgd_acc[best]["gap_1.0"], 4)}
    print(f"[ceiling] SGD best={best} acc={sgd_acc[best]['gap_1.0']:.4f} "
          f"distance={results['ceiling_distance_sgd']['distance']:.4f}",
          flush=True)

    from campaign_io import save_result
    save_result("judge_template1", results, dataset="uss_sgd.uss_mwoz.prism",
                config="bge_minilm",
                params={"seeds": SEEDS, "n_folds": 5, "n_boot": N_BOOT,
                        "judge_model": MODEL,
                        "prompt": "trace_appendix_d_template_1",
                        "inference": "seed-ensembled", "issues": "#53"})
    return results


def run(mode=None, max_samples=None):
    if max_samples is not None:
        raise ValueError(
            "judge_template1 is full-corpus-bound: the frozen judge ratings "
            "(judge_ratings.json, judge_ratings_t1.json) are index-aligned "
            "to the full corpora (PRISM to a fixed seeded 2000-conversation "
            "subset), so max_samples cannot apply here; use reproduce.py "
            "--max-samples N, which excludes the judge stages.")
    if mode is None:
        raise ValueError("explicit mode required: submit|collect|eval")
    return {"submit": submit, "collect": collect, "eval": evaluate}[mode]()


run_sample = make_run_sample(run)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    run(mode=argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
