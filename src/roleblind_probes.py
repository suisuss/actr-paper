"""Issue #45: role-blind TRACE control - the missing factorial cell.

Audit (material/assessment/2026-07-10_roleblind_trace_audit.md): 19/26
features admit role-blind analogues (bar met); the 7 cross-role relational
features (response relevance in all forms) have none. The goal anchor is
role-blind in these corpora (100% of dialogues start with a USER turn).

Arms, all grafted on g0 mirroring the combo construction:
    combo26 = [g0 | trace26]        cached OOF preds (multiseed 'combo')
    combo19 = [g0 | trace19]        role-aware admitting subset (403-d)
    comboRB = [g0 | roleblind19]    role-blind analogues (403-d)
    g0                              cached (multiseed 'g0')

Design (issue #45 comment; calibrated
regime - seed-ensembled contrasts, no bands below +-2.5pt):
    R1  combo26 > comboRB   PRISM  (role information matters)     - Holm pair
    R2  combo19 > comboRB   PRISM  (matched-form role-awareness)  - Holm pair
    R3  combo26 vs combo19  PRISM  (worth of the cross-role 7)    - estimation
    Prediction: comboRB's gain over g0 ~ +1.1 pts on PRISM (one-third of the
    combo's +3.2), from the #50 dose-response.
USS-SGD runs the same arms as secondary (scalars add nothing there; expected
all-null).

Invariant asserted per corpus: the 7 class-A columns are IDENTICAL between
trace19 and roleblind19 (they are already role-blind), and the role-blind
user_self_consistency column equals the role-blind volatility_mean column
(documented collapse).
"""
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "scripts"))

from lib.trace_features import FEATURE_NAMES, features_matrix, _normalize, _slope
from trace_h2h import build_all_arms, pairwise_accuracy, bootstrap_diff, GAPS
from multiseed_probes import load, oof, holm
from lib.campaign import Checkpoint, cap, make_run_sample

SEEDS = [42, 7, 123, 2024, 31337]
N_BOOT = 1000
GAP_PRIMARY = 1.0

MS_CKPT = WORK.parent / "data" / "experiment-caches" / "multiseed_probes"

ADMITTING = [
    "n_turns", "volatility_mean", "turn_dist_max", "late_volatility",
    "final_turn_goal_dist", "conv_drift_from_goal", "goal_initial_prompt_dist",
    "model_self_sim_mean", "model_self_sim_max", "user_self_consistency",
    "semantic_cohesion", "model_goal_dist_mean", "user_goal_dist_mean",
    "model_goal_dist_min", "model_goal_dist_max", "final_model_goal_dist",
    "model_initial_prompt_dist", "goal_adherence_trend",
    "goal_convergence_ratio",
]
CLASS_A = ["n_turns", "volatility_mean", "turn_dist_max", "late_volatility",
           "final_turn_goal_dist", "conv_drift_from_goal",
           "goal_initial_prompt_dist"]


_CKPT = Checkpoint("roleblind_probes", script=__file__)


def ckpt_load(tag):
    return _CKPT.load(tag) or {}


ckpt_save = _CKPT.save


def ms_pred(name, seed, key):
    f = MS_CKPT / f"{name}_seed{seed}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if key in z.files:
            return z[key]
    return None


def roleblind_row(emb):
    """The 19 role-blind analogues for one dialogue, in ADMITTING order."""
    e = _normalize(emb)
    n = len(e)
    goal = e[0]
    vol = 1.0 - np.sum(e[:-1] * e[1:], axis=1) if n > 1 else np.zeros(0)
    vol_mean = float(vol.mean()) if len(vol) else 0.0
    vol_max = float(vol.max()) if len(vol) else 0.0
    k = max(2, int(np.ceil(n / 4)))
    late = float(vol[-k:].mean()) if len(vol) else 0.0
    final_goal = 1.0 - float(e[-1] @ goal)
    center = e.mean(axis=0)
    center = center / (np.linalg.norm(center) + 1e-12)
    drift = 1.0 - float(center @ goal)

    if n >= 2:
        sims = e @ e.T
        iu = np.triu_indices(n, 1)
        ss_mean, ss_max = float(sims[iu].mean()), float(sims[iu].max())
    else:
        ss_mean = ss_max = 0.0

    cohesion_vals = []
    if n > 1:
        cum = np.cumsum(e, axis=0)
        for i in range(1, n):
            ctx = cum[i - 1] / i
            ctx = ctx / (np.linalg.norm(ctx) + 1e-12)
            cohesion_vals.append(float(e[i] @ ctx))
    cohesion = float(np.mean(cohesion_vals)) if cohesion_vals else 0.0

    # all-turns goal series over i >= 1 (anchor excluded; audit construction note)
    g_series = 1.0 - e[1:] @ goal if n > 1 else np.zeros(0)
    gmean = float(g_series.mean()) if len(g_series) else 0.0
    gmax = float(g_series.max()) if len(g_series) else 0.0
    gmin = float(g_series.min()) if len(g_series) else 0.0
    trend = _slope(g_series)
    conv = final_goal / gmin if gmin > 1e-9 else 0.0

    vals = {
        "n_turns": float(n),
        "volatility_mean": vol_mean, "turn_dist_max": vol_max,
        "late_volatility": late,
        "final_turn_goal_dist": final_goal, "conv_drift_from_goal": drift,
        # computed as in trace_features (float32 noise ~1e-7, not exact 0),
        # so the class-A identity assertion holds column-for-column
        "goal_initial_prompt_dist": 1.0 - float(goal @ goal),
        "model_self_sim_mean": ss_mean, "model_self_sim_max": ss_max,
        "user_self_consistency": vol_mean,      # documented collapse
        "semantic_cohesion": cohesion,
        "model_goal_dist_mean": gmean, "user_goal_dist_mean": gmean,  # merge
        "model_goal_dist_min": gmin, "model_goal_dist_max": gmax,
        "final_model_goal_dist": final_goal,    # documented collapse
        "model_initial_prompt_dist": gmean,     # collapse (G == u1)
        "goal_adherence_trend": trend,
        "goal_convergence_ratio": conv,
    }
    return [vals[k] for k in ADMITTING]


def build_roleblind(emb, offsets):
    return np.asarray([roleblind_row(emb[offsets[i]:offsets[i + 1]])
                       for i in range(len(offsets) - 1)], float)


ARMS = ["combo26", "combo19", "comboRB", "g0"]


def run(max_samples=None):
    """#45: role-blind TRACE control, grafted arms on g0.

    max_samples is not None -> ephemeral capped run: corpora truncated to
    their first max_samples dialogues, checkpoint caches and the multiseed
    pred reuse bypassed (compute fresh), no save_result
    (lib/campaign.py contract).
    """
    ephemeral = max_samples is not None
    results = {"seeds": SEEDS, "gap_primary": GAP_PRIMARY,
               "admitting": ADMITTING,
               "arms": "combo26(410) combo19(403) comboRB(403) g0(384)"}
    ens, ys = {}, {}
    adm_ix = [FEATURE_NAMES.index(k) for k in ADMITTING]
    a_ix_in_adm = [ADMITTING.index(k) for k in CLASS_A]
    for name in ["prism", "uss_sgd"]:
        emb, offsets, meta, groups = load(name)
        if ephemeral:
            emb, offsets, meta = cap(emb, offsets, meta, max_samples)
            if groups is not None:
                groups = meta["user_ids"]
        y = np.asarray(meta["overall_mean"], float)
        ys[name] = y
        raw = build_all_arms(emb, offsets, meta, degree=3)["design_b_raw"]
        d = raw.shape[1] // 4
        g0 = raw[:, :d]
        trace26 = features_matrix(emb, offsets, meta["roles"])
        trace19 = trace26[:, adm_ix]
        rb19 = build_roleblind(emb, offsets)
        # invariants: class-A columns identical role-aware vs role-blind;
        # role-blind self-consistency == role-blind volatility (collapse)
        assert np.allclose(trace19[:, a_ix_in_adm], rb19[:, a_ix_in_adm],
                           atol=1e-5), "class-A columns diverged"
        assert np.allclose(rb19[:, ADMITTING.index("user_self_consistency")],
                           rb19[:, ADMITTING.index("volatility_mean")])
        feats = {"combo26": np.hstack([g0, trace26]),
                 "combo19": np.hstack([g0, trace19]),
                 "comboRB": np.hstack([g0, rb19]),
                 "g0": g0}

        seed_accs = {k: [] for k in ARMS}
        per_seed = {}
        for seed in SEEDS:
            tag = f"{name}_seed{seed}"
            done = ckpt_load(tag) if not ephemeral else {}
            for k in ARMS:
                if k in done:
                    continue
                p = (ms_pred(name, seed, "combo" if k == "combo26" else
                             ("g0" if k == "g0" else "___"))
                     if not ephemeral else None)
                done[k] = p if p is not None else oof(feats[k], y, seed, groups)
                if not ephemeral:
                    ckpt_save(tag, done)
            per_seed[seed] = done
            for k in ARMS:
                seed_accs[k].append(
                    {f"gap_{g}": pairwise_accuracy(done[k], y, g) for g in GAPS})
            print(f"[{name}] seed {seed} done", flush=True)
        ms = {}
        for k, accs in seed_accs.items():
            ms[k] = {f"gap_{g}": {
                "mean": round(float(np.mean([a[f"gap_{g}"] for a in accs])), 4),
                "std": round(float(np.std([a[f"gap_{g}"] for a in accs])), 4)}
                for g in GAPS}
            print(f"[{name}] {k:9s} " + " ".join(
                f"gap{g}={ms[k][f'gap_{g}']['mean']:.4f}+-{ms[k][f'gap_{g}']['std']:.4f}"
                for g in GAPS), flush=True)
        results[name] = {"multiseed": ms}
        ens[name] = {k: np.mean([per_seed[s][k] for s in SEEDS], axis=0)
                     for k in ARMS}
        results[name]["accuracy_seedens"] = {
            k: round(pairwise_accuracy(ens[name][k], y, GAP_PRIMARY), 4)
            for k in ARMS}

    def contrast(name, a, b, label, test):
        bd = bootstrap_diff(ens[name][a], ens[name][b], ys[name],
                            GAP_PRIMARY, N_BOOT, 42)
        point = round(pairwise_accuracy(ens[name][a], ys[name], GAP_PRIMARY)
                      - pairwise_accuracy(ens[name][b], ys[name], GAP_PRIMARY), 4)
        return {"h": label, "corpus": name, "test": test, "point": point,
                "diff_ci": [bd["ci_lo"], bd["ci_hi"]], "p_raw": bd["p_boot"]}

    fam = [contrast("prism", "combo26", "comboRB",
                    "R1 combo26>comboRB PRISM", "superiority"),
           contrast("prism", "combo19", "comboRB",
                    "R2 combo19>comboRB PRISM", "superiority")]
    for f, a in zip(fam, holm([f["p_raw"] for f in fam])):
        f["p_holm"] = round(float(a), 4)
        print(f"[F_rb] {f['h']:28s} point={f['point']:+.4f} "
              f"CI [{f['diff_ci'][0]:+.4f},{f['diff_ci'][1]:+.4f}] "
              f"p_holm={f['p_holm']:.4f}", flush=True)
    results["F_roleblind"] = fam

    results["R3_estimate"] = contrast("prism", "combo26", "combo19",
                                      "R3 combo26 vs combo19 PRISM",
                                      "estimation")
    r3 = results["R3_estimate"]
    print(f"[R3] {r3['h']:32s} point={r3['point']:+.4f} "
          f"CI [{r3['diff_ci'][0]:+.4f},{r3['diff_ci'][1]:+.4f}]", flush=True)

    gains = {k: contrast("prism", k, "g0", f"gain {k}>g0 PRISM", "descriptive")
             for k in ["combo26", "combo19", "comboRB"]}
    results["prism_gains_over_g0"] = gains
    for k, v in gains.items():
        print(f"[gain] {v['h']:28s} point={v['point']:+.4f} "
              f"CI [{v['diff_ci'][0]:+.4f},{v['diff_ci'][1]:+.4f}] "
              f"p={v['p_raw']:.4f}", flush=True)

    results["secondary_sgd"] = [
        contrast("uss_sgd", a, b, f"S {a} vs {b} SGD", "secondary")
        for a, b in [("combo26", "comboRB"), ("combo19", "comboRB"),
                     ("comboRB", "g0")]]
    for s in results["secondary_sgd"]:
        print(f"[secondary] {s['h']:28s} point={s['point']:+.4f} "
              f"p={s['p_raw']:.4f}", flush=True)

    if not ephemeral:
        from campaign_io import save_result
        save_result("roleblind_probes", results, dataset="prism.uss_sgd",
                    config="minilm",
                    params={"seeds": SEEDS, "n_folds": 5, "n_boot": N_BOOT,
                            "inference": "seed-ensembled", "issues": "#45"})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    """CLI shim - historical invocation: python scripts/roleblind_probes.py"""
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
