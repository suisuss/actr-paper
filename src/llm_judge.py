"""Issue #23: LLM-judge arm via the Batches API.

Two-judge design:
    haiku (claude-haiku-4-5): form-faithful (one-shot, transcript-only, integer 1-5), not prompt-faithful (not TRACE's Appendix D Template 1) rater,
        all of USS-SGD + USS-MWOZ + a deterministic 2,000-conversation PRISM
        subset (their baseline was deliberately a cheap auto-rater).
    opus (claude-opus-4-8): strong-judge arm, USS-SGD + first 1,000 of the
        PRISM subset - answers "would a stronger judge beat the geometry?"

Modes:
    submit  - build transcripts, estimate cost (abort > $40), create batches
    collect - poll batches, parse integer ratings, save judge_ratings.json
    eval    - judge-alone pairwise accuracy (strict + ties-as-half), hybrid
              arms (judge rating appended to g0 / role-split), disagreement
              analysis vs the geometry arm

Never prints the API key. State in data/results/trace_h2h/judge_state.json.
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

from lib.campaign import make_run_sample

RESULTS = WORK.parent / "data" / "results" / "trace_h2h"
STATE = RESULTS / "judge_state.json"
RATINGS = RESULTS / "judge_ratings.json"
PRISM_SUBSET = 2000
OPUS_PRISM = 1000
COST_GUARD_USD = 40.0

PROMPT = (
    "You will read a conversation between a user (USER) and an AI assistant "
    "(ASSISTANT). Rate the user's overall satisfaction with the assistant on "
    "a scale from 1 (very dissatisfied) to 5 (very satisfied).\n"
    "Respond with only the integer rating and nothing else.\n\n"
    "<conversation>\n{transcript}\n</conversation>"
)


def load_env_key():
    for line in open(WORK / ".env"):
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.strip().split("=", 1)[1]


def corpora():
    from lib.uss import load_uss
    from embed_prism import load_prism_dialogues
    out = {}
    for name in ["uss_sgd", "uss_mwoz"]:
        seqs, meta = load_uss(name.split("_", 1)[1])
        out[name] = {
            "texts": seqs, "roles": meta["roles"],
            "y": meta["overall_mean"], "users": None,
        }
    ds = load_prism_dialogues()
    rng = np.random.default_rng(42)
    idx = sorted(rng.choice(len(ds), PRISM_SUBSET, replace=False).tolist())
    out["prism"] = {
        "texts": [ds[i]["texts"] for i in idx],
        "roles": [ds[i]["roles"] for i in idx],
        "y": [ds[i]["y"] for i in idx],
        "users": [ds[i]["user_id"] for i in idx],
        "prism_indices": idx,
    }
    return out


def transcript(texts, roles):
    lines = []
    for t, r in zip(texts, roles):
        speaker = "USER" if r == "USER" else "ASSISTANT"
        lines.append(f"{speaker}: {t}")
    return "\n".join(lines)


def submit():
    load_env_key()
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic()
    data = corpora()

    jobs = {
        "haiku": {"model": "claude-haiku-4-5",
                  "targets": [("uss_sgd", None), ("uss_mwoz", None), ("prism", None)]},
        "opus": {"model": "claude-opus-4-8",
                 "targets": [("uss_sgd", None), ("prism", OPUS_PRISM)]},
    }

    # cost estimate (chars/4 as token proxy), batch pricing = 50% off
    price_in = {"haiku": 1.0, "opus": 5.0}  # $/MTok
    est = 0.0
    for jname, job in jobs.items():
        for corpus, limit in job["targets"]:
            texts = data[corpus]["texts"][:limit] if limit else data[corpus]["texts"]
            roles = data[corpus]["roles"][:limit] if limit else data[corpus]["roles"]
            toks = sum(len(transcript(t, r)) for t, r in zip(texts, roles)) / 4
            est += toks / 1e6 * price_in[jname] * 0.5
    print(f"estimated batched input cost: ${est:.2f}", flush=True)
    if est > COST_GUARD_USD:
        raise SystemExit(f"cost estimate ${est:.2f} exceeds guard ${COST_GUARD_USD}")

    state = {"batches": {}}
    for jname, job in jobs.items():
        requests = []
        for corpus, limit in job["targets"]:
            n = limit or len(data[corpus]["texts"])
            for i in range(n):
                tx = transcript(data[corpus]["texts"][i], data[corpus]["roles"][i])
                requests.append(Request(
                    custom_id=f"{corpus}-{i}",
                    params=MessageCreateParamsNonStreaming(
                        model=job["model"], max_tokens=8,
                        messages=[{"role": "user",
                                   "content": PROMPT.format(transcript=tx)}]),
                ))
        batch = client.messages.batches.create(requests=requests)
        state["batches"][jname] = {"id": batch.id, "n": len(requests),
                                   "model": job["model"]}
        print(f"[{jname}] batch {batch.id} submitted ({len(requests)} requests)",
              flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(state, f, indent=1)
    return state


def collect():
    load_env_key()
    import anthropic
    client = anthropic.Anthropic()
    state = json.load(open(STATE))
    ratings = {}
    for jname, b in state["batches"].items():
        while True:
            batch = client.messages.batches.retrieve(b["id"])
            print(f"[{jname}] {batch.processing_status} "
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
            text = next((blk.text for blk in msg.content if blk.type == "text"), "")
            val = None
            for tok in text.split():
                t = tok.strip(".:,;")
                if t.isdigit() and 1 <= int(t) <= 5:
                    val = int(t)
                    break
            r[result.custom_id] = val
        ratings[jname] = r
        print(f"[{jname}] collected {len(r)} ratings "
              f"({sum(1 for v in r.values() if v is None)} unparsable)", flush=True)
    with open(RATINGS, "w") as f:
        json.dump(ratings, f)
    return ratings


def evaluate():
    from trace_h2h import (build_all_arms, oof_predictions, pairwise_accuracy,
                           bootstrap_diff, GAPS)
    from extension_arms import build_rolesplit
    ratings = json.load(open(RATINGS))
    data = corpora()
    EMB = WORK.parent / "data" / "embeddings"
    out = {}

    def pairwise_ties_half(scores, y, gap):
        dy = y[:, None] - y[None, :]
        valid = np.abs(dy) >= gap
        np.fill_diagonal(valid, False)
        dp = scores[:, None] - scores[None, :]
        correct = ((dp * dy) > 0).astype(float) + 0.5 * (dp == 0)
        return float(correct[valid].sum() / valid.sum())

    for corpus in ["uss_sgd", "uss_mwoz", "prism"]:
        y = np.asarray(data[corpus]["y"], float)
        groups = data[corpus]["users"]
        # embeddings aligned to corpus rows
        if corpus == "prism":
            npz = np.load(EMB / "prism_minilm.npz")
            # read the embeddings member ONCE: indexing npz[...] inside the loop
            # re-decompresses the whole array each iteration, and each basic-slice
            # view pins its own fresh copy alive -> unbounded memory over 7,951 convs
            full_emb = npz["embeddings"]
            full_off = npz["offsets"]
            idx = data[corpus]["prism_indices"]
            chunks, off, roles = [], [0], []
            for i in idx:
                s, e = full_off[i], full_off[i + 1]
                chunks.append(full_emb[s:e])
                off.append(off[-1] + (e - s))
            emb = np.vstack(chunks)
            offsets = np.asarray(off)
            meta = {"roles": data[corpus]["roles"]}
        else:
            npz = np.load(EMB / f"{corpus}_minilm.npz")
            emb, offsets = npz["embeddings"], npz["offsets"]
            meta = {"roles": data[corpus]["roles"]}

        base = build_all_arms(emb, offsets, meta, degree=3)
        raw = base["design_b_raw"]
        d = raw.shape[1] // 4
        role_raw, _ = build_rolesplit(emb, offsets, meta)
        # role-split K=1 slice (the recommended configuration)
        cols = list(range(0, 2 * d)) + list(range(4 * d, 6 * d))
        role_k1 = role_raw[:, cols]

        entry = {}
        for jname in ratings:
            r = ratings[jname]
            vals = np.asarray([r.get(f"{corpus}-{i}") if r.get(f"{corpus}-{i}")
                               is not None else np.nan
                               for i in range(len(y))], float)
            mask = ~np.isnan(vals)
            if mask.sum() < 100:
                continue  # this judge did not cover this corpus
            ym, vm = y[mask], vals[mask]
            gm = [groups[i] for i in range(len(y)) if mask[i]] if groups else None
            je = {"n_rated": int(mask.sum())}
            for gap in GAPS:
                je[f"gap_{gap}"] = {
                    "judge_strict": round(pairwise_accuracy(vm, ym, gap), 4),
                    "judge_ties_half": round(pairwise_ties_half(vm, ym, gap), 4),
                }
            # hybrid arms on the rated subset
            def oof(X):
                if gm is None:
                    return oof_predictions(X, ym, 5, 42)
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import GroupKFold
                p = np.zeros(len(ym))
                for tr, te in GroupKFold(n_splits=5).split(X, ym, gm):
                    rf = RandomForestRegressor(n_estimators=300,
                                               min_samples_leaf=2,
                                               random_state=42, n_jobs=-1)
                    rf.fit(X[tr], ym[tr])
                    p[te] = rf.predict(X[te])
                return p
            arms = {
                "g0": raw[mask][:, :d],
                "g0_plus_judge": np.hstack([raw[mask][:, :d], vm[:, None]]),
                "rolesplit_k1": role_k1[mask],
                "rolesplit_k1_plus_judge": np.hstack([role_k1[mask], vm[:, None]]),
            }
            preds = {k: oof(X) for k, X in arms.items()}
            for gap in GAPS:
                g = {k: round(pairwise_accuracy(p, ym, gap), 4)
                     for k, p in preds.items()}
                g["judge_gain_rolesplit"] = bootstrap_diff(
                    preds["rolesplit_k1_plus_judge"], preds["rolesplit_k1"],
                    ym, gap, 1000, 42)
                je[f"gap_{gap}"].update(g)
            # disagreement vs geometry (rolesplit_k1 OOF), gap >= 1.0
            geo = preds["rolesplit_k1"]
            dy = ym[:, None] - ym[None, :]
            valid = np.abs(dy) >= 1.0
            np.fill_diagonal(valid, False)
            js = np.sign(vm[:, None] - vm[None, :])
            gs = np.sign(geo[:, None] - geo[None, :])
            both = valid & (js != 0) & (gs != 0)
            je["disagreement_frac_gap1.0"] = round(
                float((js[both] != gs[both]).sum() / max(both.sum(), 1)), 4)
            entry[jname] = je
            out[corpus] = entry
            with open(RESULTS / "llm_judge_eval.json", "w") as f:
                json.dump(out, f, indent=1)  # incremental write per cell
            print(f"[{corpus}/{jname}] " + json.dumps(
                {k: v for k, v in je.items() if not isinstance(v, dict)}),
                flush=True)
            for gap in GAPS:
                print(f"    gap {gap}: {json.dumps({k: v for k, v in je[f'gap_{gap}'].items() if isinstance(v, (int, float))})}",
                      flush=True)
        out[corpus] = entry
        del base, raw, role_raw, role_k1

    from lib.campaign import save_result
    save_result("llm_judge_eval", out, dataset="uss_sgd.uss_mwoz.prism",
                config="minilm", tags=["trace_h2h", "judge"])
    return out


def run(mode=None, max_samples=None):
    if max_samples is not None:
        raise ValueError(
            "llm_judge is full-corpus-bound: the frozen judge ratings "
            "(judge_ratings.json) are index-aligned to the full corpora "
            "(PRISM to a fixed seeded 2000-conversation subset), so "
            "max_samples cannot apply here; use reproduce.py "
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
