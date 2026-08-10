"""Issue #27: per-category descriptive profiles with FDR + JSON persistence.

Also resolves the draft's verification ledger: every descriptive statistic
previously computed in throwaway sessions is recomputed here and persisted:
  - three-path geometry (drift alignment, gap mean/slope, residuals) with
    Spearman rho AND p-values
  - causal surprise stats with p-values
  - per-service profiles on USS-SGD (top services from the SGD text match),
    Kruskal-Wallis across services per geometry, BH-FDR corrected
  - corpus statistics table (turn counts, per-role counts)
  - the example-figure residual fractions
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "scripts"))

from lib.campaign import cap, make_run_sample
from lib.encoders.gram import gram_fit, gram_basis_matrix_fast
from trace_h2h import load_corpus, dialogue_iter
from role_surprise import causal_surprise

BH_FAMILY = []


def _norm(e):
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


def resid_frac(pts, degree=3):
    n = len(pts)
    K = min(degree, n - 1)
    coefs = gram_fit(pts, degree=K)
    B = gram_basis_matrix_fast(n, K)
    return float(np.sum((pts - B @ coefs[:K + 1]) ** 2) / np.sum(pts ** 2))


def geometry_rows(emb, offsets, meta):
    rows = []
    for i, e, roles in dialogue_iter(emb, offsets, meta):
        en = _norm(e)
        u = en[[j for j, r in enumerate(roles) if r == "USER"]]
        m = en[[j for j, r in enumerate(roles) if r != "USER"]]
        k = min(len(u), len(m))
        d = u[:k] - m[:k]
        c = (u[:k] + m[:k]) / 2
        gu, gm = gram_fit(u, degree=3), gram_fit(m, degree=3)
        def cos(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0
        dn = np.linalg.norm(d, axis=1)
        x = np.arange(len(dn), dtype=float)
        x -= x.mean()
        slope = float(x @ (dn - dn.mean()) / (x @ x)) if len(dn) > 1 else 0.0
        su, nu = causal_surprise(u)
        sm, nm = causal_surprise(m)
        rows.append({
            "cos_drift": cos(gu[1], gm[1]),
            "gap_mean": float(dn.mean()),
            "gap_slope": slope,
            "resid_user": resid_frac(u), "resid_model": resid_frac(m),
            "resid_midline": resid_frac(c), "resid_interleaved": resid_frac(en),
            "user_sur_mean": float(su.mean()) if len(su) else 0.0,
            "model_sur_mean": float(sm.mean()) if len(sm) else 0.0,
            "user_trend_gain": float(su.mean() - nu.mean()) if len(su) else 0.0,
            "model_trend_gain": float(sm.mean() - nm.mean()) if len(sm) else 0.0,
        })
    return rows


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        prev = min(prev, p[idx] * m / (rank + 1))
        adj[idx] = prev
    return adj


def run(max_samples=None):
    from scipy import stats as st
    emb, offsets, meta = load_corpus("uss_sgd")
    emb, offsets, meta = cap(emb, offsets, meta, max_samples)
    y = np.asarray(meta["overall_mean"], float)
    rows = geometry_rows(emb, offsets, meta)
    keys = list(rows[0].keys())
    M = {k: np.asarray([r[k] for r in rows]) for k in keys}

    out = {"n": len(y)}
    # 1. overall Spearman with p-values, BH-FDR across the descriptive family
    fam = []
    for k in keys:
        rho, p = st.spearmanr(M[k], y)
        fam.append({"stat": k, "rho": round(float(rho), 4), "p_raw": float(p)})
    adj = bh_fdr([f["p_raw"] for f in fam])
    for f, a in zip(fam, adj):
        f["p_fdr"] = float(a)
        print(f"[overall] {f['stat']:18s} rho={f['rho']:+.4f} "
              f"p={f['p_raw']:.2e} p_fdr={f['p_fdr']:.2e}", flush=True)
    out["overall_spearman"] = fam

    # 2. per-service profiles + Kruskal-Wallis, FDR-corrected
    with open(WORK.parent / "data" / "embeddings" / "uss_sgd_original_map.json") as f:
        mapping = json.load(f)
    if max_samples is not None:
        mapping = mapping[:len(y)]  # index-aligned per-dialogue list: first-N
    svc = np.asarray([m["services"][0] if m["services"] else "other"
                      for m in mapping])
    top = [s for s, _ in Counter(svc).most_common(6)]
    per_service = {}
    kw_fam = []
    for k in keys:
        groups = [M[k][svc == s] for s in top]
        h, p = st.kruskal(*groups)
        kw_fam.append({"stat": k, "H": round(float(h), 2), "p_raw": float(p)})
        per_service[k] = {s: round(float(M[k][svc == s].mean()), 4) for s in top}
    adj = bh_fdr([f["p_raw"] for f in kw_fam])
    for f, a in zip(kw_fam, adj):
        f["p_fdr"] = float(a)
        print(f"[kruskal] {f['stat']:18s} H={f['H']:8.2f} p_fdr={f['p_fdr']:.2e}",
              flush=True)
    out["per_service_means"] = per_service
    out["kruskal_across_services"] = kw_fam
    out["services"] = top

    # 3. corpus statistics
    for name in ["uss_sgd", "uss_mwoz"]:
        e2, o2, m2 = load_corpus(name)
        e2, o2, m2 = cap(e2, o2, m2, max_samples)
        nt = np.asarray([len(r) for r in m2["roles"]])
        nu = np.asarray([sum(1 for x in r if x == "USER") for r in m2["roles"]])
        out[f"corpus_{name}"] = {
            "n_dialogues": int(len(nt)),
            "turns_mean": round(float(nt.mean()), 1),
            "turns_min": int(nt.min()), "turns_max": int(nt.max()),
            "user_turns_mean": round(float(nu.mean()), 1),
            "n_annotators_dist": dict(Counter(len(o) for o in m2["overall"])),
        }
    # 4. example-figure residual fractions (dialogue 0, as in the figure)
    e0 = _norm(emb[offsets[0]:offsets[1]])
    roles0 = meta["roles"][0]
    u0 = e0[[j for j, r in enumerate(roles0) if r == "USER"]]
    m0 = e0[[j for j, r in enumerate(roles0) if r != "USER"]]
    out["figure2_residuals"] = {
        "interleaved": round(resid_frac(e0), 3),
        "user": round(resid_frac(u0), 3),
        "model": round(resid_frac(m0), 3),
    }
    if max_samples is None:
        from campaign_io import save_result
        save_result("descriptive_profiles", out, dataset="uss_sgd.uss_mwoz",
                    config="minilm", params={"test": "kruskal_fdr"})
    return out


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
