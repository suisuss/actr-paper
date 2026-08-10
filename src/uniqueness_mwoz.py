"""Post-#63 follow-up: decide the uniqueness of the never-worse property.

Issue #63's uniqueness contrast (role-split(1) vs mean-pooling+TRACE,
seed-ensembled) ran on USS-SGD and came back undecided (+1.7, CI
[-1.0, +4.7], p=0.25) [capacity_uniqueness]. On USS-MWOZ the combo trails
role-split(1) by 3.3 points (draft Table 1) - outside the instrument's
~+-2.5-point resolution - so the same contrast there can decide whether
the never-distinguishably-worse property is unique to the split. PRISM
is already measured (rolesplit_graft: CI [-2.8, +1.5], indistinguishable).

Same protocol and caches as capacity_uniqueness.py: seed-ensembled OOF
predictions from data/experiment-caches/rolesplit_graft/, dialogue-resampling
bootstrap (N=1000, seed 42) at gaps 0.5/1.0. Near-zero compute: no
model is fit, predictions are loaded from the checkpoint caches.
"""
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "scripts"))

from lib.campaign import make_run_sample
from trace_h2h import load_corpus, pairwise_accuracy, bootstrap_diff, GAPS

SEEDS = [42, 7, 123, 2024, 31337]
N_BOOT = 1000
CACHES = WORK.parent / "data" / "experiment-caches"


def ensemble(cache_dir, corpus, arm):
    preds = [np.load(CACHES / cache_dir / f"{corpus}_seed{s}.npz")[arm]
             for s in SEEDS]
    return np.mean(preds, axis=0)


def uniqueness_contrast(corpus):
    _, _, meta = load_corpus(corpus)
    y = np.asarray(meta["overall_mean"], float)
    rs1 = ensemble("rolesplit_graft", corpus, "rolesplit_k1")
    combo = ensemble("rolesplit_graft", corpus, "combo")
    out = {}
    for gap in GAPS:
        bd = bootstrap_diff(rs1, combo, y, gap, N_BOOT, 42)
        out[f"gap_{gap}"] = {
            "rs1_acc": round(pairwise_accuracy(rs1, y, gap), 4),
            "combo_acc": round(pairwise_accuracy(combo, y, gap), 4),
            "diff": round(pairwise_accuracy(rs1, y, gap)
                          - pairwise_accuracy(combo, y, gap), 4),
            "ci": [bd["ci_lo"], bd["ci_hi"]], "p_boot": bd["p_boot"]}
        print(f"[uniqueness {corpus}] gap{gap} "
              f"rs1={out[f'gap_{gap}']['rs1_acc']} "
              f"combo={out[f'gap_{gap}']['combo_acc']} "
              f"ci={out[f'gap_{gap}']['ci']} p={bd['p_boot']}", flush=True)
    return out


def run(max_samples=None):
    if max_samples is not None:
        raise ValueError(
            "uniqueness_mwoz only re-analyzes full-corpus seed-ensembled OOF "
            "predictions from the rolesplit_graft checkpoint cache and cannot "
            "recompute them in-process; for sampled runs use the sandbox "
            "path, reproduce.py --max-samples N, which caps the raw layer "
            "(lib/corpus_sample.py) and regenerates that cache upstream.")
    results = {"seeds": SEEDS,
               "uniqueness_mwoz": uniqueness_contrast("uss_mwoz")}
    from campaign_io import save_result
    save_result("uniqueness_mwoz", results,
                dataset="uss_mwoz", config="minilm",
                params={"seeds": SEEDS, "n_boot": N_BOOT,
                        "follows_issue": 63})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
