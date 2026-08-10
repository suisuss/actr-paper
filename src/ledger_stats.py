"""Verification-ledger statistics (#37): regenerate the draft's [doc:] numbers
as a provenanced JSON artifact.

Covers ledger items 1-5 (6 and 7 are out of scope: the DCT=Gram tie needs a
re-run under the h2h protocol, and the Chebyshev degradation belongs to the
Paper 1 technical note). All role-split geometry is computed on USS-SGD, the
corpus the draft's role-split section reports.

Definitions (from material/methodology/07_role_split_encoding.md):
  path         per-speaker L2-normalized turn embeddings (USER vs non-USER)
  G_k          degree-k Gram coefficient of a path (gram_fit, degree 3)
  drift align  cos(G1_user, G1_model) per dialogue
  offset d_k   G_k_user - G_k_model; gap(tau) = ||sum_k d_k phi_k(tau)||
  residual@3   fraction of a path's energy not captured by its degree-0..3 fit
  causal surprise  prefix-only degree<=1 extrapolation error (role_surprise.py)
Descriptive Spearman correlations are against dialogue-level satisfaction.
"""
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))

from scipy import stats as st

from lib.campaign import cap, make_run_sample
from lib.encoders.gram import gram_fit, gram_basis_matrix_fast
from lib.embedding_cache import EmbeddingCache
from trace_h2h import load_corpus, dialogue_iter
from extension_arms import build_rolesplit  # noqa: F401 (kept for parity/import check)
from role_surprise import build as build_surprise, causal_surprise  # noqa: F401


def _norm(e):
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


def residual_above_deg3(path):
    """Fraction of a path's energy above its degree-0..3 Gram reconstruction."""
    n = len(path)
    if n < 2:
        return None
    deg = min(3, n - 1)
    coeffs = gram_fit(path, degree=deg)
    G = gram_basis_matrix_fast(n, deg)
    recon = G @ coeffs[:deg + 1]
    num = float(np.sum((path - recon) ** 2))
    den = float(np.sum(path ** 2)) + 1e-12
    return num / den


def gap_profile(gu, gm, n_eval=12):
    """||d(tau)|| over uniform tau in [-1,1] from offset coeffs d_k=G_k^u-G_k^m."""
    d = gu - gm                                  # (deg+1, dim)
    deg = d.shape[0] - 1
    B = gram_basis_matrix_fast(n_eval, deg)      # (n_eval, deg+1)
    traj = B @ d                                 # (n_eval, dim)
    return np.linalg.norm(traj, axis=1)          # (n_eval,)


def role_geometry(emb, offsets, meta):
    y = np.asarray(meta["overall_mean"], float)
    drift, gmean, gslope = [], [], []
    r_user, r_model, r_mid, r_inter = [], [], [], []
    keep = []
    for i, e, roles in dialogue_iter(emb, offsets, meta):
        en = _norm(e)
        u = en[[j for j, r in enumerate(roles) if r == "USER"]]
        m = en[[j for j, r in enumerate(roles) if r != "USER"]]
        if len(u) < 2 or len(m) < 2:
            continue
        gu = gram_fit(u, degree=3)
        gm = gram_fit(m, degree=3)
        # drift alignment: cos(G1_user, G1_model)
        a, b = gu[1], gm[1]
        cos = float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))
        drift.append(cos)
        # gap process from the offset coefficients
        gp = gap_profile(gu, gm)
        gmean.append(float(gp.mean()))
        gslope.append(float(np.polyfit(np.arange(len(gp)), gp, 1)[0]))
        # residual-above-degree-3 for each path + interleaved + midline
        mid = (gu + gm) / 2.0
        Bm = gram_basis_matrix_fast(12, 3)
        mid_traj = Bm @ mid
        r_user.append(residual_above_deg3(u))
        r_model.append(residual_above_deg3(m))
        r_inter.append(residual_above_deg3(en))
        r_mid.append(residual_above_deg3(_norm(mid_traj)))
        keep.append(i)
    yk = y[keep]

    def sp(vals):
        r, p = st.spearmanr(vals, yk)
        return {"rho": round(float(r), 4), "p": float(p), "n": len(vals)}

    return {
        "n_dialogues_used": len(keep),
        # item 1: three-path descriptive Spearman
        "drift_alignment": sp(drift),
        "gap_slope": sp(gslope),
        "gap_mean": sp(gmean),
        "resid_user": sp(r_user),
        "resid_model": sp(r_model),
        "resid_midline": sp(r_mid),
        # item 3: mean residual-above-degree-3 fractions
        "residual_above_deg3_mean": {
            "interleaved": round(float(np.mean(r_inter)), 4),
            "user": round(float(np.mean(r_user)), 4),
            "model": round(float(np.mean(r_model)), 4),
        },
    }


def causal_surprise_pvalues(emb, offsets, meta):
    """Item 2: Spearman rho AND p-value for the causal-surprise scalars."""
    y = np.asarray(meta["overall_mean"], float)
    S, _ = build_surprise(emb, offsets, meta)
    names = ["user_sur_mean", "user_sur_max", "user_trend_gain", "user_bits",
             "model_sur_mean", "model_sur_max", "model_trend_gain", "model_bits"]
    out = {}
    for j, nm in enumerate(names):
        r, p = st.spearmanr(S[:, j], y)
        out[nm] = {"rho": round(float(r), 4), "p": float(p)}
    return out


def _load_any(corpus):
    """load_corpus handles USS ('uss_sgd'); PRISM loads from its own npz/meta."""
    if corpus == "prism":
        import json
        npz = np.load(WORK.parent / "data" / "embeddings" / "prism_minilm.npz")
        with open(WORK.parent / "data" / "embeddings" / "prism_meta.json") as f:
            meta = json.load(f)
        return npz["embeddings"], npz["offsets"], meta
    return load_corpus(corpus)


def corpus_stats(max_samples=None):
    """Items 4-5: PRISM mean length and USS corpus statistics."""
    stats = {}
    for corpus in ["uss_sgd", "uss_mwoz", "prism"]:
        emb_c, offsets, meta = _load_any(corpus)
        _, offsets, meta = cap(emb_c, offsets, meta, max_samples)
        n_turns = np.diff(offsets)
        y = np.asarray(meta["overall_mean"], float)
        stats[corpus] = {
            "n_dialogues": int(len(n_turns)),
            "total_turns": int(n_turns.sum()),
            "mean_turns": round(float(n_turns.mean()), 2),
            "median_turns": float(np.median(n_turns)),
            "turn_range": [int(n_turns.min()), int(n_turns.max())],
            "rating_mean": round(float(y.mean()), 3),
            "rating_std": round(float(y.std()), 3),
        }
    return stats


def run(max_samples=None):
    emb, offsets, meta = load_corpus("uss_sgd")
    emb, offsets, meta = cap(emb, offsets, meta, max_samples)
    results = {
        "role_geometry_sgd": role_geometry(emb, offsets, meta),
        "causal_surprise_sgd": causal_surprise_pvalues(emb, offsets, meta),
        "corpus_stats": corpus_stats(max_samples),
        "_scope_note": ("ledger items 1-5; items 6 (DCT=Gram tie) and 7 "
                        "(Chebyshev degradation) are out of scope - see docstring"),
    }
    import json
    print(json.dumps(results, indent=1))
    if max_samples is None:
        from campaign_io import save_result
        save_result("ledger_stats", results, dataset="uss_sgd.uss_mwoz.prism",
                    config="minilm", params={"seed": 42, "ledger_items": "1-5"})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
