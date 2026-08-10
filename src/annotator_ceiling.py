"""Issue #25: inter-annotator ceiling on USS.

For each dialogue pair qualifying at a gap threshold (on the mean rating),
score each annotator's preference against the leave-one-out mean of the
remaining annotators. The mean agreement is the achievable ceiling for any
predictor evaluated against mean-rating preferences.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
from lib.campaign import make_run_sample
from lib.uss import load_uss

GAPS = (0.5, 1.0)
RNG = np.random.default_rng(42)
MAX_PAIRS = 200_000


def ceiling(name, max_samples=None):
    _, meta = load_uss(name.split("_", 1)[1] if "_" in name else name)
    overall = meta["overall"]           # per-dialogue list of annotator ratings
    mean = np.asarray(meta["overall_mean"], float)
    if max_samples is not None:
        # first-N dialogues, mirroring the raw-layer truncation
        overall = overall[:int(max_samples)]
        mean = mean[:int(max_samples)]
    n = len(mean)
    out = {}
    for gap in GAPS:
        pairs = [(i, j) for i, j in combinations(range(n), 2)
                 if abs(mean[i] - mean[j]) >= gap]
        if len(pairs) > MAX_PAIRS:
            idx = RNG.choice(len(pairs), MAX_PAIRS, replace=False)
            pairs = [pairs[k] for k in idx]
        agree = total = 0
        for i, j in pairs:
            true_sign = np.sign(mean[i] - mean[j])
            for side, other in ((i, j), (j, i)):
                # annotator a's implied preference: their rating of `side`
                # versus the group mean of `other`, oriented as (side - other);
                # flip to the pair's (i - j) orientation when side == j
                orient = 1.0 if side == i else -1.0
                for r_a in overall[side]:
                    pred_sign = orient * np.sign(r_a - mean[other])
                    total += 1
                    if pred_sign == 0:
                        agree += 0.5
                    else:
                        agree += float(pred_sign == true_sign)
        out[f"gap_{gap}"] = {"ceiling": round(agree / total, 4),
                             "n_pairs_scored": total}
        print(f"[{name}] gap {gap}: annotator-vs-mean agreement "
              f"{agree/total:.4f} over {total} judgments", flush=True)
    return out


def run(max_samples=None):
    results = {ds: ceiling(ds, max_samples) for ds in ["uss_sgd", "uss_mwoz"]}
    results["note"] = ("Leave-one-annotator-out agreement with mean-rating "
                      "preferences; PRISM has one first-person rating per "
                      "conversation, no ceiling estimable.")
    if max_samples is None:
        from campaign_io import save_result
        save_result("annotator_ceiling", results, dataset="uss_sgd.uss_mwoz",
                    config="none", params={"seed": 42})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
