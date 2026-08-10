"""Issue #16: cross-encoder relevance series.

encode mode: score (preceding user turn, model turn) pairs with a cross-encoder
  -> data/embeddings/{corpus}_xenc.npz {scores, pair_dialogue, pair_model_pos}
eval mode: swap cross-encoder relevance into the model_relevance series;
  rebuild the TRACE arm and a Design-A-style expansion; compare vs bi-encoder.
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))

from lib.campaign import cap, make_run_sample

EMB = WORK.parent / "data" / "embeddings"
CORPORA = ["uss_sgd", "uss_mwoz", "prism"]


def corpus_dialogues(name):
    from lib.uss import load_uss
    if name == "prism":
        sys.path.insert(0, str(WORK / "scripts"))
        from embed_prism import load_prism_dialogues
        ds = load_prism_dialogues()
        return [d["texts"] for d in ds], [d["roles"] for d in ds]
    seqs, meta = load_uss(name.split("_", 1)[1])
    return seqs, meta["roles"]


def encode():
    from lib.device import get_device
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                      device=get_device())
    summary = {}
    for name in CORPORA:
        texts, roles = corpus_dialogues(name)
        pairs, pd, pp = [], [], []
        for di, (seq, rs) in enumerate(zip(texts, roles)):
            last_user = None
            for ti, (t, r) in enumerate(zip(seq, rs)):
                if r == "USER":
                    last_user = t
                elif last_user is not None:
                    pairs.append((last_user, t))
                    pd.append(di)
                    pp.append(ti)
        print(f"[{name}] scoring {len(pairs)} pairs...", flush=True)
        scores = ce.predict(pairs, batch_size=128, show_progress_bar=False)
        np.savez(EMB / f"{name}_xenc.npz", scores=np.asarray(scores, np.float32),
                 pair_dialogue=np.asarray(pd, np.int32),
                 pair_model_pos=np.asarray(pp, np.int32))
        print(f"[{name}] saved {len(scores)} scores", flush=True)
        summary[name] = {"n_pairs": int(len(scores)),
                         "file": str(EMB / f"{name}_xenc.npz")}
    return summary


def eval_corpus(name, max_samples=None):
    from lib.encoders.gram import gram_fit
    from lib.trace_features import features_matrix, FEATURE_NAMES
    from trace_h2h import (load_corpus, oof_predictions, pairwise_accuracy,
                           bootstrap_diff, GAPS, build_design_a)
    emb, offsets, meta = load_corpus(name)
    emb, offsets, meta = cap(emb, offsets, meta, max_samples)
    y = np.asarray(meta["overall_mean"], float)
    # pair_dialogue indices are corpus positions, so a first-N cap simply
    # ignores the cached scores of dialogues >= N
    x = np.load(EMB / f"{name}_xenc.npz")
    n_dialogues = len(offsets) - 1

    # per-dialogue cross-encoder relevance series -> gram 0-3 + mean/min/slope
    from collections import defaultdict
    per_d = defaultdict(list)
    for s, di in zip(x["scores"], x["pair_dialogue"]):
        per_d[int(di)].append(float(s))
    xe_rows = []
    for i in range(n_dialogues):
        s = np.asarray(per_d.get(i, [0.0]))
        g = gram_fit(s[:, None], degree=3)[:, 0]
        xe_rows.append(list(g) + [float(s.mean()), float(s.min()),
                                  float(s.max())])
    xe = np.asarray(xe_rows, float)

    trace = features_matrix(emb, offsets, meta["roles"])
    a = build_design_a(emb, offsets, meta)
    arms = {
        "trace_bienc": trace,
        "trace_plus_xenc": np.hstack([trace, xe]),
        "design_a_bienc": a,
        "design_a_plus_xenc": np.hstack([a, xe]),
        "xenc_only": xe,
    }
    preds = {k: oof_predictions(v, y, 5, 42) for k, v in arms.items()}
    out = {}
    for gap in GAPS:
        g = {k: round(pairwise_accuracy(preds[k], y, gap), 4) for k in arms}
        g["trace_gain"] = bootstrap_diff(preds["trace_plus_xenc"],
                                         preds["trace_bienc"], y, gap, 1000, 42)
        g["design_a_gain"] = bootstrap_diff(preds["design_a_plus_xenc"],
                                            preds["design_a_bienc"], y, gap, 1000, 42)
        out[f"gap_{gap}"] = g
        print(f"[{name}] gap {gap}: " +
              " ".join(f"{k}={v}" for k, v in g.items() if isinstance(v, float)),
              flush=True)
        for c in ["trace_gain", "design_a_gain"]:
            v = g[c]
            print(f"    {c}: CI [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] p={v['p_boot']:.3f}",
                  flush=True)
    return out


def run(mode="encode", max_samples=None):
    if mode == "encode":
        if max_samples is not None:
            raise ValueError(
                "cross_encoder_eval encode mode cannot be capped in memory: "
                "the {corpus}_xenc.npz cache contract is full-corpus "
                "index-aligned; use reproduce.py --max-samples N, which "
                "truncates the raw corpora instead")
        return encode()
    results = {ds: eval_corpus(ds, max_samples=max_samples)
               for ds in ["uss_sgd", "uss_mwoz"]}
    if max_samples is None:
        from campaign_io import save_result
        save_result("cross_encoder_eval", results, dataset="uss_sgd.uss_mwoz",
                    config="minilm", params={"seed": 42, "n_folds": 5})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    mode = args[0] if args else "encode"
    run(mode=mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
