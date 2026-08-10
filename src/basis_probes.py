"""Issues #43 + #44 (tracking #46): basis invariance and the DC mechanism.

Section 4.2 of the draft claims (a) DCT and Gram tie at matched truncation
[doc:-tagged, never run under this protocol] and (b) DST is disqualified
because its basis contains no constant function. This script measures both
under the exact E1/E2 protocol (eqdim_probes.py): 5 seeds, RF primary +
RidgeCV secondary, OOF predictions, pairwise accuracy, gap 1.0 primary,
Holm within each family, bootstrap n=1000 at seed 42.

Arms (all matched at 1,536-d, same normalization as design_b_raw):
    dct_raw           first 4 orthonormal-grid DCT-II coefficients (k=0..3)
    dct_rolesplit_k1  per-role DCT-II [du0|du1|dm0|dm1] (mirror rolesplit_k1)
    dst_raw           first 4 DST-II harmonics (j=1..4 - the basis has no DC)
    dst_dc            [g0 | s1 s2 s3] - DC restored, smooth budget matched
    g1to3             raw[:, 384:] - shape-without-content reference (#44)
    raw, rolesplit_k1 references, OOF preds reused from eqdim/multiseed caches

Inference families (gap 1.0, RF, seed-42 bootstrap; Holm within each
superiority family; equivalence = 95% CI of the accuracy difference inside
+-BAND, TOST-style - "p > 0.05" is not evidence of a tie):
    F_basis_raw_eq   B1/B2  dct_raw ~ raw               SGD, MWOZ  (band)
    F_basis_rs_sup   B3/B4  dct_rolesplit_k1 > dct_raw  SGD, MWOZ  (Holm)
    F_basis_rs_eq    B5/B6  dct_rolesplit_k1 ~ rolesplit_k1  SGD, MWOZ (band)
    F_dst_sup        D1/D2  raw > dst_raw               SGD, MWOZ  (Holm)
    F_dst_eq         D3/D4  dst_dc ~ raw                SGD, MWOZ  (band)
Secondary (reported, outside Holm): PRISM versions of all contrasts (the
role-split transfer on PRISM is the crux cell); dst_raw vs g1to3
(collapse-to-shape prediction); linear probes of every arm.

Deterministic mechanism artifact (no training): principal angles between
span{Gram 0..3} and span{DCT-II 0..3} on n = 4..40 uniform points, plus the
fraction of the constant vector's energy captured by the truncated DST span.
NOTE (measured before launch): that capture is ~0.90 at n=8..26 - the j=1
half-sine alone carries 8/pi^2 ~ 0.81 of DC - so the #44 collapse prediction
is at real risk in exactly the direction its decision criteria pre-stated
("mean smuggled through low-order harmonics"). The RF arms decide it.
Saved as its own registry artifact.

Basis conventions mirror lib/encoders/gram.py::gram_fit exactly: projection
normalized by basis-column norm^2, coefficients zero-padded when
n_turns < n_coefficients, turn embeddings L2-normalized first. Edge case
documented: at n=1 the j=1 DST-II function is the single point [1], so a
one-turn sequence puts its content into s1; no corpus here has one-turn
dialogues, and per-role blocks are not built for DST.
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "scripts"))

from trace_h2h import (build_all_arms, pairwise_accuracy, bootstrap_diff,
                       GAPS)
from extension_arms import build_rolesplit
from degree_ablation import role_k_slice
from multiseed_probes import load, oof, holm
from lib.encoders.gram import gram_basis_matrix_fast
from lib.encoders.projection import dct2_basis, dst2_basis, basis_fit
from lib.campaign import Checkpoint, cap, make_run_sample

SEEDS = [42, 7, 123, 2024, 31337]
N_FOLDS, N_BOOT = 5, 1000
BAND = 0.01          # +-1.0 point equivalence band at gap 1.0 (confirmed on #46)
GAP_PRIMARY = 1.0
D = 384
NCOEF = 4

_CKPT = Checkpoint("basis_probes", script=__file__)
ckpt_load = _CKPT.load
ckpt_save = _CKPT.save
_EQDIM = Checkpoint("eqdim_probes")
_MS = Checkpoint("multiseed_probes")


def ref_pred(name, seed, key, linear=False):
    """Reuse raw / rolesplit_k1 OOF predictions from earlier campaign caches."""
    tag = f"{name}_linear" if linear else f"{name}_seed{seed}"
    for cache in (_EQDIM, _MS):
        d = cache.load(tag)
        if d is not None and key in d:
            return d[key]
    return None


# Bases (dct2_basis, dst2_basis, basis_fit): the canonical norm^2 least-squares
# projection, imported above from lib.encoders.projection. That module is the
# SINGLE source of truth shared by lib.encoders.dct/dst and every paper script,
# so this convention (d0 = g0 = mean exactly) cannot diverge from the library.


def _norm(e):
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


def build_basis_raw(emb, offsets, meta, basis_fn, ncoef=NCOEF):
    rows = []
    for i in range(len(offsets) - 1):
        en = _norm(emb[offsets[i]:offsets[i + 1]])
        rows.append(basis_fit(en, basis_fn, ncoef).ravel())
    return np.asarray(rows, float)


def build_basis_rolesplit(emb, offsets, meta, basis_fn, ncoef=NCOEF):
    """Per-role blocks [cu0..cu3 | cm0..cm3], mirroring build_rolesplit."""
    rows = []
    for i in range(len(offsets) - 1):
        en = _norm(emb[offsets[i]:offsets[i + 1]])
        roles = meta["roles"][i]
        u = en[[j for j, r in enumerate(roles) if r == "USER"]]
        m = en[[j for j, r in enumerate(roles) if r != "USER"]]
        cu = basis_fit(u, basis_fn, ncoef)
        cm = basis_fit(m, basis_fn, ncoef)
        rows.append(np.concatenate([cu.ravel(), cm.ravel()]))
    return np.asarray(rows, float)


# ---------- deterministic mechanism artifact ----------

def subspace_report(n_lo=4, n_hi=40, ncoef=NCOEF):
    """Principal angles Gram-span vs DCT-span, and the truncated DST span's
    capture of the constant vector, per sequence length. No training."""
    out = {}
    for n in range(n_lo, n_hi + 1):
        G = gram_basis_matrix_fast(n, ncoef - 1)
        Qg = np.linalg.qr(G)[0]
        Qc = np.linalg.qr(dct2_basis(n, ncoef))[0]
        Qs = np.linalg.qr(dst2_basis(n, ncoef))[0]
        sv_gc = np.clip(np.linalg.svd(Qg.T @ Qc, compute_uv=False), 0.0, 1.0)
        sv_gs = np.clip(np.linalg.svd(Qg.T @ Qs, compute_uv=False), 0.0, 1.0)
        one = np.ones(n) / np.sqrt(n)
        out[str(n)] = {
            "angles_gram_dct_deg": [round(float(a), 2) for a in
                                    np.degrees(np.arccos(sv_gc))],
            "overlap_gram_dct": round(float(np.mean(sv_gc ** 2)), 4),
            "overlap_gram_dst": round(float(np.mean(sv_gs ** 2)), 4),
            "dst_dc_capture": round(float(np.sum((Qs.T @ one) ** 2)), 4),
        }
    return out


# ---------- evaluation ----------

def eq_test(pa, pb, y, label, corpus):
    bd = bootstrap_diff(pa, pb, y, GAP_PRIMARY, N_BOOT, 42)
    return {"h": label, "corpus": corpus, "test": "equivalence",
            "band": BAND, "diff_ci": [bd["ci_lo"], bd["ci_hi"]],
            "p_raw": bd["p_boot"],
            "in_band": bool(bd["ci_lo"] >= -BAND and bd["ci_hi"] <= BAND)}


def sup_test(pa, pb, y, label, corpus):
    bd = bootstrap_diff(pa, pb, y, GAP_PRIMARY, N_BOOT, 42)
    return {"h": label, "corpus": corpus, "test": "superiority",
            "diff_ci": [bd["ci_lo"], bd["ci_hi"]], "p_raw": bd["p_boot"]}


def run(max_samples=None):
    """#43/#44/#46: basis arms + deterministic subspace artifact.

    max_samples is not None -> ephemeral capped run: corpora truncated to
    their first max_samples dialogues, checkpoint caches and the
    cross-campaign ref_pred reuse bypassed (compute fresh), no save_result
    (lib/campaign.py contract). Returns {"basis_probes": ...,
    "basis_subspace": ...} - the two dicts this script persists.
    """
    ephemeral = max_samples is not None
    results = {"seeds": SEEDS, "band": BAND, "gap_primary": GAP_PRIMARY,
               "arms": "dct_raw/dct_rolesplit_k1/dst_raw/dst_dc/g1to3 "
                       "(all 1536-d except g1to3=1152) vs raw/rolesplit_k1"}

    subspace = subspace_report()
    caps = [subspace[str(n)]["dst_dc_capture"] for n in range(8, 27)]
    print(f"[subspace] gram-dct overlap at n=8: "
          f"{subspace['8']['overlap_gram_dct']:.4f}; "
          f"dst dc-capture n=8..26: {min(caps):.3f}-{max(caps):.3f}",
          flush=True)

    cache = {}
    for name in ["uss_sgd", "uss_mwoz", "prism"]:
        emb, offsets, meta, groups = load(name)
        if ephemeral:
            emb, offsets, meta = cap(emb, offsets, meta, max_samples)
            if groups is not None:
                groups = meta["user_ids"]
        y = np.asarray(meta["overall_mean"], float)
        raw = build_all_arms(emb, offsets, meta, degree=3)["design_b_raw"]
        role_raw, _ = build_rolesplit(emb, offsets, meta)
        rs_k1 = role_k_slice(role_raw, 1)
        dct_raw = build_basis_raw(emb, offsets, meta, dct2_basis)
        dct_rs_k1 = role_k_slice(
            build_basis_rolesplit(emb, offsets, meta, dct2_basis), 1)
        dst_raw = build_basis_raw(emb, offsets, meta, dst2_basis)
        dst_dc = np.hstack([raw[:, :D], dst_raw[:, :3 * D]])
        g1to3 = raw[:, D:]

        # invariants: DCT k=0 IS the mean (= gram g0); dims matched
        assert np.allclose(dct_raw[:, :D], raw[:, :D], atol=1e-8), \
            "DCT d0 != gram g0 - basis conventions diverged"
        for k, X in [("dct_raw", dct_raw), ("dct_rolesplit_k1", dct_rs_k1),
                     ("dst_raw", dst_raw), ("dst_dc", dst_dc)]:
            assert X.shape == raw.shape, (k, X.shape, raw.shape)

        arms = {"raw": raw, "rolesplit_k1": rs_k1, "dct_raw": dct_raw,
                "dct_rolesplit_k1": dct_rs_k1, "dst_raw": dst_raw,
                "dst_dc": dst_dc, "g1to3": g1to3}
        cache[name] = (arms, y, groups)

        # --- multi-seed RF, checkpointed per arm within each seed ---
        seed_accs = {k: [] for k in arms}
        preds42 = {}
        for seed in SEEDS:
            tag = f"{name}_seed{seed}"
            done = (ckpt_load(tag) or {}) if not ephemeral else {}
            for k in arms:
                if k in done:
                    continue
                p = ref_pred(name, seed, k) if not ephemeral else None
                done[k] = p if p is not None else oof(arms[k], y, seed, groups)
                if not ephemeral:
                    ckpt_save(tag, done)   # per-arm: bounds loss to one OOF fit
            for k in arms:
                if seed == 42:
                    preds42[k] = done[k]
                seed_accs[k].append(
                    {f"gap_{g}": pairwise_accuracy(done[k], y, g) for g in GAPS})
            print(f"[{name}] seed {seed} done", flush=True)
        ms = {}
        for k, accs in seed_accs.items():
            ms[k] = {f"gap_{g}": {
                "mean": round(float(np.mean([a[f"gap_{g}"] for a in accs])), 4),
                "std": round(float(np.std([a[f"gap_{g}"] for a in accs])), 4)}
                for g in GAPS}
            print(f"[{name}] {k:17s} " + " ".join(
                f"gap{g}={ms[k][f'gap_{g}']['mean']:.4f}+-{ms[k][f'gap_{g}']['std']:.4f}"
                for g in GAPS), flush=True)
        results[name] = {"multiseed": ms,
                         "_preds42": {k: p for k, p in preds42.items()}}

        # --- linear probes (RidgeCV, seed 42), checkpointed per arm ---
        lin_tag = f"{name}_linear"
        lin_done = (ckpt_load(lin_tag) or {}) if not ephemeral else {}
        lin = {}
        for k in arms:
            if k not in lin_done:
                p = (ref_pred(name, 42, k, linear=True)
                     if not ephemeral else None)
                lin_done[k] = (p if p is not None
                               else oof(arms[k], y, 42, groups, linear=True))
                if not ephemeral:
                    ckpt_save(lin_tag, lin_done)
            lin[k] = {f"gap_{g}": round(
                pairwise_accuracy(lin_done[k], y, g), 4) for g in GAPS}
            print(f"[{name}] linear {k:17s} gap1.0={lin[k]['gap_1.0']:.4f} "
                  f"(RF {ms[k]['gap_1.0']['mean']:.4f})", flush=True)
        results[name]["linear_probes"] = lin

    # --- inference families (seed-42 preds, gap 1.0) ---
    def preds(name, k):
        return np.asarray(results[name]["_preds42"][k])

    def y_of(name):
        return cache[name][1]

    fam_eq_raw, fam_rs_sup, fam_rs_eq, fam_dst_sup, fam_dst_eq = [], [], [], [], []
    for name, tag in [("uss_sgd", "SGD"), ("uss_mwoz", "MWOZ")]:
        y = y_of(name)
        fam_eq_raw.append(eq_test(preds(name, "dct_raw"), preds(name, "raw"),
                                  y, f"B dct_raw~raw {tag}", name))
        fam_rs_sup.append(sup_test(preds(name, "dct_rolesplit_k1"),
                                   preds(name, "dct_raw"), y,
                                   f"B dct_rs_k1>dct_raw {tag}", name))
        fam_rs_eq.append(eq_test(preds(name, "dct_rolesplit_k1"),
                                 preds(name, "rolesplit_k1"), y,
                                 f"B dct_rs_k1~rs_k1 {tag}", name))
        fam_dst_sup.append(sup_test(preds(name, "raw"), preds(name, "dst_raw"),
                                    y, f"D raw>dst_raw {tag}", name))
        fam_dst_eq.append(eq_test(preds(name, "dst_dc"), preds(name, "raw"),
                                  y, f"D dst_dc~raw {tag}", name))
    for fam in (fam_rs_sup, fam_dst_sup):
        for f, a in zip(fam, holm([f["p_raw"] for f in fam])):
            f["p_holm"] = round(float(a), 4)
    for label, fam in [("F_basis_raw_eq", fam_eq_raw),
                       ("F_basis_rs_sup", fam_rs_sup),
                       ("F_basis_rs_eq", fam_rs_eq),
                       ("F_dst_sup", fam_dst_sup),
                       ("F_dst_eq", fam_dst_eq)]:
        results[label] = fam
        for f in fam:
            verdict = (f"in_band={f['in_band']}" if "in_band" in f
                       else f"p_holm={f['p_holm']:.4f}")
            print(f"[{label}] {f['h']:28s} CI [{f['diff_ci'][0]:+.4f},"
                  f"{f['diff_ci'][1]:+.4f}] {verdict}", flush=True)

    # --- secondary: PRISM contrasts (crux cell) + dst collapse target ---
    y = y_of("prism")
    results["secondary_prism"] = {
        "dct_raw_vs_raw": eq_test(preds("prism", "dct_raw"),
                                  preds("prism", "raw"), y,
                                  "S dct_raw~raw PRISM", "prism"),
        "dct_rs_vs_dct_raw": sup_test(preds("prism", "dct_rolesplit_k1"),
                                      preds("prism", "dct_raw"), y,
                                      "S dct_rs_k1>dct_raw PRISM", "prism"),
        "dct_rs_vs_rs": eq_test(preds("prism", "dct_rolesplit_k1"),
                                preds("prism", "rolesplit_k1"), y,
                                "S dct_rs_k1~rs_k1 PRISM", "prism"),
        "raw_vs_dst": sup_test(preds("prism", "raw"), preds("prism", "dst_raw"),
                               y, "S raw>dst_raw PRISM", "prism"),
        "dst_dc_vs_raw": eq_test(preds("prism", "dst_dc"),
                                 preds("prism", "raw"), y,
                                 "S dst_dc~raw PRISM", "prism"),
    }
    results["secondary_dst_collapse"] = {
        name: sup_test(preds(name, "dst_raw"), preds(name, "g1to3"),
                       y_of(name), f"S dst_raw vs g1to3 {name}", name)
        for name in ["uss_sgd", "uss_mwoz", "prism"]}
    for k, v in {**results["secondary_prism"],
                 **results["secondary_dst_collapse"]}.items():
        print(f"[secondary] {v['h']:30s} CI [{v['diff_ci'][0]:+.4f},"
              f"{v['diff_ci'][1]:+.4f}] p={v['p_raw']:.4f}", flush=True)

    for name in cache:
        del results[name]["_preds42"]
    subspace_results = {"ncoef": NCOEF, "per_n": subspace}
    if not ephemeral:
        from campaign_io import save_result
        save_result("basis_subspace", subspace_results,
                    dataset="synthetic", config="none",
                    params={"n_lo": 4, "n_hi": 40, "ncoef": NCOEF,
                            "note": "deterministic, no seeds involved"})
        save_result("basis_probes", results,
                    dataset="uss_sgd.uss_mwoz.prism", config="minilm",
                    params={"seeds": SEEDS, "n_folds": N_FOLDS, "band": BAND,
                            "gap_primary": GAP_PRIMARY, "n_boot": N_BOOT,
                            "issues": "#43 #44 #46"})
    return {"basis_probes": results, "basis_subspace": subspace_results}


run_sample = make_run_sample(run)


def main(argv=None):
    """CLI shim - historical invocation: python scripts/basis_probes.py"""
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
