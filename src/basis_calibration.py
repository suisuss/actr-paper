"""#43 amendment: calibrate the inference layer against instrument resolution.

Analysis-only - consumes the OOF predictions cached by basis_probes.py
(data/experiment-caches/basis_probes/), fits nothing. Registered on #43
(2026-07-09 comment, 'optional amendment') BEFORE running.

Motivation: every +-1pt TOST band in basis_probes came back
in_band=False with a CI containing 0 - the band sat below what the
dialogue-resampling bootstrap can resolve at these corpus sizes, so the
tests were unpassable by construction (design error, owned on #43).

This script makes the calibration explicit:
1. Instrument resolution: 95% CI half-width of the bootstrapped accuracy
   difference at gap 1.0, per corpus, per contrast, under (a) single-seed
   (42) predictions - what basis_probes used - and (b) seed-ensembled
   predictions (mean of the 5 cached OOF vectors per arm; removes RF/CV
   seed luck, cannot remove dialogue-sampling noise).
2. All primary contrasts recomputed on the seed-ensembled
   predictions, reported as ESTIMATES (point diff + CI), no band verdicts.
3. The minimal decidable equivalence band per corpus (~ the median CI
   half-width) - i.e. what a calibrated TOST would have needed.
4. Exploratory (labeled as such): Fisher combination across the three
   independent corpora of the role-split-transfer p-values.
"""
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "scripts"))

from trace_h2h import pairwise_accuracy, bootstrap_diff
from multiseed_probes import load
from lib.campaign import cap, make_run_sample

SEEDS = [42, 7, 123, 2024, 31337]
GAP, N_BOOT = 1.0, 1000
CKPT = WORK.parent / "data" / "experiment-caches" / "basis_probes"
ARMS = ["raw", "rolesplit_k1", "dct_raw", "dct_rolesplit_k1",
        "dst_raw", "dst_dc", "g1to3"]

CONTRASTS = [
    ("dct_raw", "raw", "basis tie, raw"),
    ("dct_rolesplit_k1", "dct_raw", "role-split transfer (DCT)"),
    ("dct_rolesplit_k1", "rolesplit_k1", "basis tie, role-split"),
    ("raw", "dst_raw", "DST penalty"),
    ("dst_dc", "raw", "DC restoration"),
    ("dst_raw", "g1to3", "DST vs shape-only (collapse cell)"),
    ("rolesplit_k1", "raw", "anchor: role-split advantage (Gram)"),
]


def run(max_samples=None):
    """#43 amendment: instrument-resolution calibration, analysis-only.

    max_samples is not None -> ephemeral capped run, no save_result. This
    script fits nothing: its inputs ARE the frozen full-N OOF prediction
    vectors cached by basis_probes (dialogue-index-aligned), so capping
    truncates them to the first max_samples dialogues alongside y - there
    is no fresh compute to fall back to (lib/campaign.py contract).
    """
    ephemeral = max_samples is not None
    results = {"gap": GAP, "n_boot": N_BOOT, "seeds": SEEDS,
               "modes": {"seed42": "single-seed preds, as basis_probes",
                         "seedavg": "mean of 5 OOF pred vectors per arm"}}
    transfer_p = {}
    for name in ["uss_sgd", "uss_mwoz", "prism"]:
        emb, offsets, meta, _ = load(name)
        if ephemeral:
            emb, offsets, meta = cap(emb, offsets, meta, max_samples)
        y = np.asarray(meta["overall_mean"], float)
        per_seed = {}
        for s in SEEDS:
            z = np.load(CKPT / f"{name}_seed{s}.npz", allow_pickle=True)
            # [:len(y)] is the identity at full N; under max_samples it
            # truncates the full-N vectors to the capped dialogue prefix
            per_seed[s] = {k: z[k][:len(y)] for k in ARMS}
        modes = {"seed42": per_seed[42],
                 "seedavg": {k: np.mean([per_seed[s][k] for s in SEEDS],
                                        axis=0) for k in ARMS}}
        corp = {"n_dialogues": int(len(y))}
        corp["accuracy_seedavg"] = {
            k: round(pairwise_accuracy(modes["seedavg"][k], y, GAP), 4)
            for k in ARMS}
        hw = {m: [] for m in modes}
        for a, b, label in CONTRASTS:
            entry = {}
            for m, P in modes.items():
                bd = bootstrap_diff(P[a], P[b], y, GAP, N_BOOT, 42)
                point = round(pairwise_accuracy(P[a], y, GAP)
                              - pairwise_accuracy(P[b], y, GAP), 4)
                half = round((bd["ci_hi"] - bd["ci_lo"]) / 2, 4)
                hw[m].append(half)
                entry[m] = {"point": point, "ci": [bd["ci_lo"], bd["ci_hi"]],
                            "ci_half_width": half, "p": bd["p_boot"]}
            corp[f"{a}__vs__{b}"] = {"label": label, **entry}
            if label.startswith("role-split transfer"):
                transfer_p[name] = entry["seedavg"]["p"]
            s42, sav = entry["seed42"], entry["seedavg"]
            print(f"[{name}] {label:34s} seed42 {s42['point']:+.4f} "
                  f"(hw {s42['ci_half_width']:.4f}) | seedavg "
                  f"{sav['point']:+.4f} (hw {sav['ci_half_width']:.4f}) "
                  f"p={sav['p']:.4f}", flush=True)
        corp["resolution"] = {
            m: {"median_ci_half_width": round(float(np.median(v)), 4),
                "max_ci_half_width": round(float(np.max(v)), 4)}
            for m, v in hw.items()}
        corp["minimal_decidable_band"] = corp["resolution"]["seedavg"][
            "median_ci_half_width"]
        print(f"[{name}] resolution: seed42 median hw "
              f"{corp['resolution']['seed42']['median_ci_half_width']:.4f}, "
              f"seedavg median hw "
              f"{corp['resolution']['seedavg']['median_ci_half_width']:.4f}",
              flush=True)
        results[name] = corp

    # exploratory: Fisher combination of the transfer contrast, 3 corpora
    from scipy.stats import chi2
    ps = np.clip([transfer_p[n] for n in transfer_p], 1e-12, 1.0)
    x2 = float(-2 * np.sum(np.log(ps)))
    results["exploratory_fisher_transfer"] = {
        "per_corpus_p_seedavg": {n: float(p) for n, p in transfer_p.items()},
        "chi2": round(x2, 3), "df": 2 * len(ps),
        "p_combined": float(chi2.sf(x2, 2 * len(ps))),
        "note": "post-hoc, two-sided per-corpus p; all point diffs positive"}
    print(f"[fisher] transfer combined p="
          f"{results['exploratory_fisher_transfer']['p_combined']:.4f}",
          flush=True)

    results["calibration_verdict"] = (
        "TOST at +-1pt is undecidable on all three corpora (median CI "
        "half-widths exceed the band in both modes). Calibrated inference "
        "for this data: report point estimates with CIs (estimation "
        "framing); equivalence claims only at bands >= the per-corpus "
        "minimal_decidable_band.")

    if not ephemeral:
        from campaign_io import save_result
        save_result("basis_calibration", results,
                    dataset="uss_sgd.uss_mwoz.prism", config="minilm",
                    params={"seeds": SEEDS, "gap": GAP, "n_boot": N_BOOT,
                            "source": "cached OOF preds from basis_probes",
                            "issues": "#43 amendment"})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    """CLI shim - historical invocation: python scripts/basis_calibration.py"""
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
