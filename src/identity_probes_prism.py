"""Issue #97 Analysis A: is the model-identity confound live on PRISM?

H_confound (issue #97): the PRISM role-split(1) gain over the interleaved
projection (+2.8, Table 2) is carried by model-identity information. PRISM
conversations come from 21 LLMs; the per-role model mean G0^m is a
near-fingerprint of which model the user talked to. If mean satisfaction
differs by model, part of the split's gain is "the regressor learned which
models get rated well" - which the interleaved mean blurs by pooling both
speakers.

A is the cheap gate that decides whether B (same-model pair restriction,
identity_controls_prism.py) and C (identity augmentation,
identity_augment_prism.py) are worth their cost. Two questions:

  A1  Does model identity predict the label?   one-way ANOVA eta^2 of
      helpfulness ~ chosen model over the 7,951 filtered conversations.
      eta^2 ~ 0.01 (negligible) would mean the confound has almost no
      channel to operate through -> report and stop.
  A2  Is the fingerprint real, and is access DIFFERENTIAL? Multinomial
      logistic regression recovering chosen-model ID from a 384-d mean,
      GroupKFold(5) by user, against the majority-class baseline:
        G0^m    role-split(1)'s model-turn mean  - what the split exposes
        G0^all  the interleaved arm's pooled mean - what interleaved sees
        G0^u    the user-turn mean               - NEGATIVE CONTROL, should
                                                   sit at chance
      G0^all is the crux: the confound needs DIFFERENTIAL access. If the
      interleaved arm identifies the model just as well, the split cannot
      be winning by identifying it better. The G0^m - G0^all margin bounds
      how much identity the split can possibly add.

Gate (pre-stated): non-trivial eta^2 AND a real fingerprint -> B and C are
mandatory. Either absent -> the confound has no mechanism and #97 closes on
A alone.

Labels: PRISM turn records carry model_name/model_provider/if_chosen. Turn 0
presents several candidate responses; exactly one is if_chosen, and the
chosen model is constant within every conversation (asserted below), so the
per-conversation label is well defined. embed_prism.py keeps only if_chosen
model turns, so every model turn in the trajectory belongs to the chosen
model - the label is consistent with the embedded turns.

WARNING for anyone extending this: prism_meta.json stores roles as
USER/SYSTEM, not user/model. Filtering with r=="model" silently yields an
all-zero feature matrix and a classifier that collapses to the majority
class - which reads exactly like a clean "no fingerprint" null. The role
masks below assert non-degeneracy for that reason.

Protocol: GroupKFold(5) by user_id (identical to the PRISM CV protocol in
Section 5.2), LogisticRegression(max_iter=3000) on standardized features.
Checkpoints: data/experiment-caches/identity_probes_prism/. Registry:
lib.campaign.save_result.
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))

from lib.campaign import Checkpoint, load_corpus, make_run_sample, save_result

N_FOLDS = 5
MAX_ITER = 3000

_CKPT = Checkpoint("identity_probes_prism", script=__file__)


def prism_model_labels(max_samples=None):
    """Chosen-model name per PRISM conversation, aligned with load_corpus.

    Re-walks the raw jsonl under embed_prism.py's exact filter (helpfulness
    present, >= 2 kept turns), which is the same order the embeddings and
    prism_meta.json were built in - the alignment build_prism_meta.py also
    relies on. Returns an (n_dialogues,) array of model names, truncated to
    the first max_samples to mirror lib.campaign.cap().
    """
    labels = []
    raw = WORK.parent / "data" / "raw" / "prism" / "conversations.jsonl"
    with open(raw) as f:
        for line in f:
            rec = json.loads(line)
            help_score = (rec.get("performance_attributes") or {}).get(
                "helpfulness")
            if help_score is None:
                continue
            turns = [t for t in rec["conversation_history"]
                     if t["role"] == "user"
                     or (t["role"] == "model" and t.get("if_chosen"))]
            if len(turns) < 2:
                continue
            chosen = [t["model_name"] for t in turns
                      if t["role"] == "model" and t.get("if_chosen")]
            # the label is only well defined if one model carries the whole
            # conversation; verified 0/8011 exceptions on the full corpus
            assert chosen, "conversation kept with no if_chosen model turn"
            assert len(set(chosen)) == 1, (
                f"non-constant chosen model in a conversation: {set(chosen)}")
            labels.append(chosen[0])
    labels = np.asarray(labels)
    if max_samples is not None:
        labels = labels[:int(max_samples)]
    return labels


def role_means(emb, offsets, roles):
    """(G0^m, G0^all, G0^u) per dialogue: model-turn mean, pooled mean,
    user-turn mean. Roles are USER/SYSTEM (see module docstring)."""
    n = len(offsets) - 1
    d = emb.shape[1]
    g0m, g0all, g0u = (np.zeros((n, d)) for _ in range(3))
    for i in range(n):
        seg = emb[offsets[i]:offsets[i + 1]]
        r = np.asarray(roles[i])
        m, u = seg[r == "SYSTEM"], seg[r == "USER"]
        g0all[i] = seg.mean(0)
        g0m[i] = m.mean(0) if len(m) else 0.0
        g0u[i] = u.mean(0) if len(u) else 0.0
    return {"G0_model": g0m, "G0_pooled": g0all, "G0_user": g0u}


def eta_squared(y, groups):
    """One-way ANOVA eta^2 + F/p of a continuous y across group labels."""
    from scipy import stats
    levels = sorted(set(groups))
    grand = y.mean()
    ss_tot = float(((y - grand) ** 2).sum())
    ss_between = float(sum(
        (groups == g).sum() * (y[groups == g].mean() - grand) ** 2
        for g in levels))
    F, p = stats.f_oneway(*[y[groups == g] for g in levels])
    return {"eta_squared": round(ss_between / ss_tot, 4),
            "F": round(float(F), 3), "p": float(p),
            "df_between": len(levels) - 1, "df_within": len(y) - len(levels),
            "n_levels": len(levels), "n": int(len(y))}


def probe(X, labels, users, seed=42):
    """Model-ID recovery from a 384-d mean; GroupKFold(5) by user."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # a degenerate (all-zero) matrix would silently read as "no fingerprint"
    assert int((np.abs(X).sum(1) == 0).sum()) == 0, \
        "all-zero rows in probe features - check the USER/SYSTEM role mask"
    accs, npred, oof = [], [], np.empty(len(labels), dtype=object)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, labels, groups=users):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=MAX_ITER,
                                               random_state=seed))
        clf.fit(X[tr], labels[tr])
        pred = clf.predict(X[te])
        oof[te] = pred
        accs.append(float((pred == labels[te]).mean()))
        npred.append(int(len(set(pred))))
    return {"accs": np.asarray(accs, float),
            "distinct_preds": np.asarray(npred, int),
            "oof": oof.astype(str)}


def run(max_samples=None):
    """#97 A: eta^2 of the label on model ID + differential fingerprint probes.

    max_samples is not None -> ephemeral capped run (first-N dialogues, no
    checkpoint reads/writes, no save_result; lib/campaign.py contract).
    """
    ephemeral = max_samples is not None
    emb, offsets, meta, groups = load_corpus("prism", max_samples=max_samples)
    labels = prism_model_labels(max_samples=max_samples)
    y = np.asarray(meta["overall_mean"], float)
    users = np.asarray(groups)
    assert len(labels) == len(offsets) - 1 == len(y) == len(users), (
        f"label/corpus misalignment: {len(labels)} labels vs "
        f"{len(offsets) - 1} dialogues")

    results = {"n_dialogues": int(len(y)), "n_models": int(len(set(labels))),
               "protocol": {"cv": "GroupKFold(5) by user_id",
                            "head": f"LogisticRegression(max_iter={MAX_ITER})",
                            "n_folds": N_FOLDS}}

    # --- A1: does model identity predict the label? -------------------
    results["A1_label_by_model"] = eta_squared(y, labels)
    counts = {m: int((labels == m).sum()) for m in sorted(set(labels))}
    results["A1_label_by_model"]["per_model"] = {
        m: {"n": counts[m], "mean_label": round(float(y[labels == m].mean()), 4)}
        for m in sorted(counts, key=lambda m: -y[labels == m].mean())}
    a1 = results["A1_label_by_model"]
    print(f"[A1] eta^2={a1['eta_squared']} F={a1['F']} p={a1['p']:.3e} "
          f"({a1['n_levels']} models, n={a1['n']})", flush=True)

    # --- A2: is the fingerprint real, and differential? ---------------
    feats = role_means(emb, offsets, meta["roles"])
    _, counts_l = np.unique(labels, return_counts=True)
    majority = float(counts_l.max() / len(labels))
    results["A2_fingerprint"] = {"majority_baseline": round(majority, 4)}
    for name, X in feats.items():
        tag = f"probe_{name}"
        cached = None if ephemeral else _CKPT.load(tag)
        if cached is not None:
            out = cached
            print(f"[A2] {name} from checkpoint", flush=True)
        else:
            out = probe(X, labels, users)
            if not ephemeral:
                _CKPT.save(tag, out)
        accs = np.asarray(out["accs"], float)
        results["A2_fingerprint"][name] = {
            "acc_mean": round(float(accs.mean()), 4),
            "acc_std": round(float(accs.std()), 4),
            "lift_over_majority": round(float(accs.mean() - majority), 4),
            "folds": [round(float(a), 4) for a in accs],
            "distinct_preds_per_fold": [int(v) for v in out["distinct_preds"]]}
        r = results["A2_fingerprint"][name]
        print(f"[A2] {name:9s} acc={r['acc_mean']:.4f}+-{r['acc_std']:.4f} "
              f"lift={r['lift_over_majority']:+.4f} "
              f"(majority {majority:.4f})", flush=True)

    # differential access: what the split exposes beyond the pooled mean
    fp = results["A2_fingerprint"]
    fp["differential_access"] = {
        "G0_model_minus_G0_pooled": round(
            fp["G0_model"]["acc_mean"] - fp["G0_pooled"]["acc_mean"], 4),
        "note": ("the confound needs DIFFERENTIAL access; this margin bounds "
                 "the identity the split can add over the interleaved mean")}
    print(f"[A2] differential (G0_model - G0_pooled) = "
          f"{fp['differential_access']['G0_model_minus_G0_pooled']:+.4f}",
          flush=True)

    # --- gate verdict --------------------------------------------------
    live = (a1["eta_squared"] >= 0.02
            and fp["G0_model"]["lift_over_majority"] > 0.05)
    results["gate"] = {
        "confound_live": bool(live),
        "rule": ("non-trivial eta^2 (>=0.02) AND a real fingerprint "
                 "(G0_model lift >5pt) -> B/C mandatory"),
        "verdict": ("B/C mandatory" if live
                    else "no mechanism - #97 closes on A")}
    print(f"[gate] {results['gate']['verdict']}", flush=True)

    if not ephemeral:
        save_result("identity_probes_prism", results, dataset="prism",
                    config="minilm",
                    params={"n_folds": N_FOLDS, "head": "logreg",
                            "issue": 97, "analysis": "A"})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    """CLI shim: python scripts/identity_probes_prism.py"""
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
