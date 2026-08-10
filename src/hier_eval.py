"""Issues #14/#15 evaluation: within-turn structure trajectories.

Loads {corpus}_hier.npz (within-turn g0/g1/g2 per turn + residual fractions)
and builds:
    struct_g1_traj:  between-turn Gram 0-3 over the sequence of within-turn g1
                     vectors (word-order/structure channel), 1536-d
    struct_g2_traj:  same over within-turn g2, 1536-d
    resid_series:    Gram 0-3 expansion of the per-turn residual-fraction
                     series + token-count controls (#15)
Arms vs design_b_raw (content trajectories from sentence embeddings):
    structure alone, [content | structure], residual block marginal.
Confound check (#15): residual-fraction features vs n_tokens correlation.
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))

from lib.campaign import cap, make_run_sample
from lib.encoders.gram import gram_fit
from trace_h2h import (load_corpus, build_all_arms, oof_predictions,
                       pairwise_accuracy, bootstrap_diff, GAPS)

EMB = WORK.parent / "data" / "embeddings"
N_FOLDS, N_BOOT, SEED = 5, 1000, 42


def build_structure_blocks(name, offsets):
    h = np.load(EMB / f"{name}_hier.npz")
    wg, rf, ntok = h["within_g"], h["resid_frac"], h["n_tokens"]
    # capped runs pass first-N offsets; the per-utterance arrays are
    # position-aligned, so the shared offset prefix slices them consistently
    # (the loop below only reads below offsets[-1])
    assert np.array_equal(h["offsets"][:len(offsets)], offsets)
    n_dialogues = len(offsets) - 1

    g1_rows, g2_rows, resid_rows, tok_rows = [], [], [], []
    for i in range(n_dialogues):
        s, e = offsets[i], offsets[i + 1]
        for out_rows, k in ((g1_rows, 1), (g2_rows, 2)):
            traj = wg[s:e, k, :].astype(np.float64)
            norms = np.linalg.norm(traj, axis=1, keepdims=True)
            traj = traj / (norms + 1e-12)
            out_rows.append(gram_fit(traj, degree=3).ravel())
        r = rf[s:e, 1].astype(np.float64)  # residual above degree 1
        g = gram_fit(r[:, None], degree=3)[:, 0]
        resid_rows.append(list(g) + [float(r.mean()), float(r.max())])
        t = ntok[s:e].astype(np.float64)
        tok_rows.append([float(t.mean()), float(t.std()), float(t.max()),
                         float(t.sum())])
    return (np.asarray(g1_rows), np.asarray(g2_rows),
            np.asarray(resid_rows), np.asarray(tok_rows))


def run(max_samples=None):
    results = {}
    for name in ["uss_sgd", "uss_mwoz"]:
        emb, offsets, meta = load_corpus(name)
        emb, offsets, meta = cap(emb, offsets, meta, max_samples)
        y = np.asarray(meta["overall_mean"], float)
        raw = build_all_arms(emb, offsets, meta, degree=3)["design_b_raw"]
        g1t, g2t, resid, toks = build_structure_blocks(name, offsets)

        # confound: residual block vs token counts
        rr = np.corrcoef(resid[:, 4], toks[:, 0])[0, 1]
        print(f"[{name}] mean resid-frac vs mean tokens r={rr:.3f}", flush=True)

        arms = {
            "struct_g1_traj": g1t,
            "struct_g2_traj": g2t,
            "content_plus_g1": np.hstack([raw, g1t]),
            "content_plus_g1g2": np.hstack([raw, g1t, g2t]),
            "resid_block": resid,
            "content_plus_resid": np.hstack([raw, resid]),
            "token_ctrl": toks,
        }
        preds_raw = oof_predictions(raw, y, N_FOLDS, SEED)
        entry = {"resid_vs_tokens_r": float(rr)}
        for k, X in arms.items():
            p = oof_predictions(X, y, N_FOLDS, SEED)
            e = {}
            for gap in GAPS:
                e[f"gap_{gap}"] = {
                    "acc": round(pairwise_accuracy(p, y, gap), 4),
                    "raw": round(pairwise_accuracy(preds_raw, y, gap), 4),
                }
                if k.startswith("content_plus"):
                    e[f"gap_{gap}"]["vs_raw"] = bootstrap_diff(
                        p, preds_raw, y, gap, N_BOOT, SEED)
            entry[k] = e
            line = " ".join(f"g{gap}={e[f'gap_{gap}']['acc']}" for gap in GAPS)
            print(f"[{name}] {k:20s} {line}", flush=True)
            for gap in GAPS:
                v = e[f"gap_{gap}"].get("vs_raw")
                if v:
                    print(f"    gap {gap} vs raw: CI [{v['ci_lo']:+.4f},"
                          f"{v['ci_hi']:+.4f}] p={v['p_boot']:.3f}", flush=True)
        results[name] = entry

    if max_samples is None:
        from campaign_io import save_result
        save_result("hier_eval", results, dataset="uss_sgd.uss_mwoz",
                    config="minilm", params={"seed": SEED, "n_folds": N_FOLDS})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
