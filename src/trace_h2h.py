"""
TRACE head-to-head: Gram trajectory coefficients vs hand-crafted geometric
features on conversation-level satisfaction prediction.

Arms:
    floor1          n_turns only
    floor2          turn counts + utterance length stats
    trace           TRACE-geometric feature set (lib/trace_features.py)
    trace_nocount   TRACE-geometric minus count features
    design_a        Gram degrees 0-3 of TRACE's own scalar series (subsumption)
    design_b        derived scalars from vector-level Gram coefficients
    design_b_raw    raw flattened G0..G3 coefficients (1536-d)
    hybrid          trace + design_a + design_b

Protocol: RandomForestRegressor on dialogue-level mean satisfaction,
5-fold out-of-fold predictions, pairwise preference accuracy on pairs with
mean-rating gap >= {0.5, 1.0}, bootstrap CI on arm differences (resampling
dialogues, not pairs).
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

WORK_DIR = Path(__file__).parent
sys.path.insert(0, str(WORK_DIR))

from lib.trace_features import (FEATURE_NAMES, features_matrix,
                                trace_scalar_series)
from lib.encoders.gram import gram_fit, gram_basis_matrix_fast

EMB_DIR = WORK_DIR.parent / "data" / "embeddings"

MANIFEST = {
    "id": "trace_h2h",
    "name": "TRACE Head-to-Head",
    "category": "trace",
    "description": "Gram trajectory coefficients vs TRACE-geometric features "
                   "on USS satisfaction prediction (issues #1-#8)",
    "modes": {
        "features": "Spike 1: TRACE-geometric extraction + redundancy analysis",
        "floors": "Spike 2: preference harness, floors vs TRACE-geometric (kill switch)",
        "designs": "Spikes 3-4: Design A (subsumption) and Design B (vector-level)",
        "hybrid": "Spike 5: hybrid arm + bucket ablations",
    },
    "configs": None,
    "datasets": ["uss_sgd", "uss_mwoz"],
    "params": {
        "n_folds": {"default": 5, "type": "int"},
        "n_bootstrap": {"default": 1000, "type": "int"},
        "degree": {"default": 3, "type": "int"},
        "seed": {"default": 42, "type": "int"},
    },
    "tags": ["trace", "paper2"],
    "doc_ref": "material/experiments/trace_h2h.md",
    "save_mode": "auto",
}

GAPS = (0.5, 1.0)


# ---------- data ----------

def load_corpus(dataset):
    name = dataset.split("_", 1)[1]
    npz = np.load(EMB_DIR / f"uss_{name}_minilm.npz")
    with open(EMB_DIR / f"uss_{name}_meta.json") as f:
        meta = json.load(f)
    return npz["embeddings"], npz["offsets"], meta


def dialogue_iter(emb, offsets, meta):
    for i in range(len(offsets) - 1):
        yield i, emb[offsets[i]:offsets[i + 1]], meta["roles"][i]


# ---------- feature arms ----------

def build_floors(emb, offsets, meta, seqs_lens=None):
    rows1, rows2 = [], []
    for i, e, roles in dialogue_iter(emb, offsets, meta):
        n = len(roles)
        nu = sum(1 for r in roles if r == "USER")
        rows1.append([n])
        rows2.append([n, nu, n - nu])
    return np.asarray(rows1, float), np.asarray(rows2, float)


def build_trace(emb, offsets, meta):
    return features_matrix(emb, offsets, meta["roles"])


DESIGN_A_SERIES = ["volatility", "model_relevance", "goal_dist_user",
                   "goal_dist_model", "user_from_model"]
DESIGN_A_NAMES = [f"{s}_g{k}" for s in DESIGN_A_SERIES for k in range(4)]


def build_design_a(emb, offsets, meta, degree=3):
    rows = []
    for i, e, roles in dialogue_iter(emb, offsets, meta):
        s = trace_scalar_series(e, roles)
        feats = []
        for key in DESIGN_A_SERIES:
            series = s[key]
            if len(series) == 0:
                feats.extend([0.0] * (degree + 1))
            else:
                c = gram_fit(series[:, None], degree=degree)  # (degree+1, 1)
                feats.extend(c[:, 0].tolist())
        rows.append(feats)
    return np.asarray(rows, float)


DESIGN_B_NAMES = ["g1_norm", "g2_norm", "g3_norm",
                  "cos_g1_goal", "cos_g1_g2", "cos_g1_g3", "cos_g2_g3",
                  "residual_frac", "g0_goal_dist"]


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-10 and nb > 1e-10 else 0.0


def build_design_b(emb, offsets, meta, degree=3):
    scalar_rows, raw_rows = [], []
    for i, e, roles in dialogue_iter(emb, offsets, meta):
        en = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        n = len(en)
        goal_idx = next((j for j, r in enumerate(roles) if r == "USER"), 0)
        goal = en[goal_idx]
        coeffs = gram_fit(en, degree=degree)  # (degree+1, 384)
        G0, G1, G2, G3 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]

        # residual energy fraction of degree-`degree` reconstruction
        K = min(degree, n - 1)
        B = gram_basis_matrix_fast(n, K)
        recon = B @ coeffs[:K + 1]
        total = float(np.sum(en ** 2))
        residual = float(np.sum((en - recon) ** 2)) / total if total > 0 else 0.0

        scalar_rows.append([
            float(np.linalg.norm(G1)), float(np.linalg.norm(G2)),
            float(np.linalg.norm(G3)),
            _cos(G1, goal - G0), _cos(G1, G2), _cos(G1, G3), _cos(G2, G3),
            residual, float(1.0 - G0 @ goal / (np.linalg.norm(G0) + 1e-12)),
        ])
        raw_rows.append(coeffs.ravel())
    return np.asarray(scalar_rows, float), np.asarray(raw_rows, float)


# ---------- evaluation ----------

def oof_predictions(X, y, n_folds, seed):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    preds = np.zeros(len(y))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                   random_state=seed, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        preds[te] = rf.predict(X[te])
    return preds


def pair_matrices(y, gap):
    dy = y[:, None] - y[None, :]
    valid = np.abs(dy) >= gap
    np.fill_diagonal(valid, False)
    return dy, valid


def pairwise_accuracy(preds, y, gap):
    dy, valid = pair_matrices(y, gap)
    dp = preds[:, None] - preds[None, :]
    correct = (dp * dy) > 0
    return float(correct[valid].sum() / valid.sum())


def bootstrap_diff(preds_a, preds_b, y, gap, n_boot, seed, max_m=2000):
    """Bootstrap CI for accuracy(A) - accuracy(B), resampling dialogues.

    For corpora larger than max_m, uses m-out-of-n subsample bootstrap
    (m = max_m) to keep the per-resample n x n pair matrices tractable;
    CIs are then conservative for the full-corpus difference.

    The resample indices are always drawn with numpy (seeded), so results are
    bit-identical whether the matrix reductions run on CPU (numpy) or GPU
    (torch CUDA, used automatically when available - the reductions are
    deterministic boolean sums).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    m = min(n, max_m)
    dy = y[:, None] - y[None, :]
    da = (preds_a[:, None] - preds_a[None, :]) * dy > 0
    db = (preds_b[:, None] - preds_b[None, :]) * dy > 0
    valid = np.abs(dy) >= gap
    np.fill_diagonal(valid, False)
    diffs = np.empty(n_boot)

    use_cuda = False
    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    if use_cuda:
        import torch
        dev = torch.device("cuda")
        tda = torch.from_numpy(da).to(dev)
        tdb = torch.from_numpy(db).to(dev)
        tval = torch.from_numpy(valid).to(dev)
        # draw all indices up front (numpy, seeded) - one host->device copy
        all_idx = torch.from_numpy(
            rng.integers(0, n, (n_boot, m)).astype(np.int64)).to(dev)
        tdiffs = torch.empty(n_boot, dtype=torch.float64, device=dev)
        for b in range(n_boot):
            idx = all_idx[b]
            v = tval.index_select(0, idx).index_select(1, idx)
            nv = v.sum()
            a_hits = (tda.index_select(0, idx).index_select(1, idx) & v).sum()
            b_hits = (tdb.index_select(0, idx).index_select(1, idx) & v).sum()
            tdiffs[b] = torch.where(
                nv > 0, (a_hits - b_hits).double() / nv.double(),
                torch.zeros((), dtype=torch.float64, device=dev))
        diffs = tdiffs.cpu().numpy()
    else:
        for b in range(n_boot):
            idx = rng.integers(0, n, m)
            v = valid[np.ix_(idx, idx)]
            nv = v.sum()
            if nv == 0:
                diffs[b] = 0.0
                continue
            diffs[b] = (da[np.ix_(idx, idx)][v].sum() - db[np.ix_(idx, idx)][v].sum()) / nv

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"ci_lo": float(lo), "ci_hi": float(hi), "p_boot": float(min(p, 1.0))}


def correlation_report(X, names, thresh=0.7):
    C = np.corrcoef(X, rowvar=False)
    pairs = []
    for i, j in combinations(range(len(names)), 2):
        r = C[i, j]
        if np.isfinite(r) and abs(r) > thresh:
            pairs.append({"a": names[i], "b": names[j], "r": round(float(r), 3)})
    pairs.sort(key=lambda d: -abs(d["r"]))
    offdiag = np.abs(C[np.triu_indices(len(names), 1)])
    offdiag = offdiag[np.isfinite(offdiag)]
    return {"n_features": len(names), "n_pairs_over_thresh": len(pairs),
            "mean_abs_offdiag_r": round(float(offdiag.mean()), 3),
            "pairs": pairs}


# ---------- arms + shared runner ----------

def build_all_arms(emb, offsets, meta, degree):
    floor1, floor2 = build_floors(emb, offsets, meta)
    trace = build_trace(emb, offsets, meta)
    count_ix = [FEATURE_NAMES.index("n_turns")]
    keep = [i for i in range(len(FEATURE_NAMES)) if i not in count_ix]
    a = build_design_a(emb, offsets, meta, degree)
    b_scalars, b_raw = build_design_b(emb, offsets, meta, degree)
    return {
        "floor1": floor1, "floor2": floor2,
        "trace": trace, "trace_nocount": trace[:, keep],
        "design_a": a, "design_b": b_scalars, "design_b_raw": b_raw,
        "hybrid": np.hstack([trace, a, b_scalars]),
    }


def evaluate_arms(arms, y, arm_names, n_folds, n_boot, seed, baseline="trace"):
    preds = {k: oof_predictions(arms[k], y, n_folds, seed) for k in arm_names}
    out = {}
    for gap in GAPS:
        g = {}
        for k in arm_names:
            g[k] = {"acc": round(pairwise_accuracy(preds[k], y, gap), 4)}
        for k in arm_names:
            if k != baseline and baseline in arm_names:
                g[k]["vs_" + baseline] = bootstrap_diff(
                    preds[k], preds[baseline], y, gap, n_boot, seed)
        dy = y[:, None] - y[None, :]
        valid = np.abs(dy) >= gap
        np.fill_diagonal(valid, False)
        g["n_pairs"] = int(valid.sum() // 2)
        out[f"gap_{gap}"] = g
    return out, preds


# ---------- modes ----------

def run(mode, config, dataset, n_folds=5, n_bootstrap=1000, degree=3,
        seed=42, **params):
    emb, offsets, meta = load_corpus(dataset)
    y = np.asarray(meta["overall_mean"], float)
    results = {"dataset": dataset, "n_dialogues": len(y), "mode": mode,
               "params": {"n_folds": n_folds, "n_bootstrap": n_bootstrap,
                          "degree": degree, "seed": seed}}

    if mode == "features":
        trace = build_trace(emb, offsets, meta)
        results["correlation_trace"] = correlation_report(trace, FEATURE_NAMES)
        results["feature_names"] = FEATURE_NAMES
        results["goal_analog"] = "first_user_utterance"
        # the paper's four Temporal Dynamics features (Table 3), all
        # timestamp-dependent and unavailable in public corpora
        results["excluded_temporal"] = ["model_turn_duration",
                                        "user_turn_duration",
                                        "gap_time_median", "gap_time_mad"]
        return results

    arms = build_all_arms(emb, offsets, meta, degree)

    if mode == "floors":
        arm_names = ["floor1", "floor2", "trace", "trace_nocount"]
        ev, _ = evaluate_arms(arms, y, arm_names, n_folds, n_bootstrap, seed)
        results["evaluation"] = ev
        # kill switch: does trace beat floor1 at both gaps?
        verdicts = {}
        for gap in GAPS:
            g = ev[f"gap_{gap}"]
            diff = g["trace"]["acc"] - g["floor1"]["acc"]
            ci = g["floor1"]["vs_trace"]
            verdicts[f"gap_{gap}"] = {
                "trace_minus_floor1": round(diff, 4),
                "floor1_vs_trace_ci": [ci["ci_lo"], ci["ci_hi"]],
                "significant": ci["p_boot"] < 0.05,
            }
        results["kill_switch"] = verdicts
        return results

    if mode == "designs":
        arm_names = ["trace", "design_a", "design_b", "design_b_raw"]
        ev, _ = evaluate_arms(arms, y, arm_names, n_folds, n_bootstrap, seed)
        results["evaluation"] = ev
        results["correlation_trace"] = correlation_report(
            arms["trace"], FEATURE_NAMES)
        results["correlation_design_a"] = correlation_report(
            arms["design_a"], DESIGN_A_NAMES)
        results["correlation_design_b"] = correlation_report(
            arms["design_b"], DESIGN_B_NAMES)
        return results

    if mode == "hybrid":
        arm_names = ["trace", "design_a", "design_b", "hybrid"]
        ev, _ = evaluate_arms(arms, y, arm_names, n_folds, n_bootstrap, seed)
        results["evaluation"] = ev

        # bucket ablations on the trace arm
        buckets = {
            "counts": ["n_turns"],
            "repetition": ["model_self_sim_mean", "model_self_sim_max"],
            "cohesion_relevance": [
                "initial_response_dist", "model_user_dist_mean",
                "model_user_dist_max", "model_user_dist_min",
                "user_model_dist_mean", "user_model_dist_max",
                "relevance_trend", "semantic_cohesion", "volatility_mean",
                "turn_dist_max", "late_volatility", "user_self_consistency"],
            "goal": [
                "model_goal_dist_mean", "user_goal_dist_mean",
                "model_goal_dist_min", "model_goal_dist_max",
                "final_turn_goal_dist", "final_model_goal_dist",
                "model_initial_prompt_dist", "goal_initial_prompt_dist",
                "conv_drift_from_goal", "goal_adherence_trend",
                "goal_convergence_ratio"],
        }
        ablate = {}
        for bname, keys in buckets.items():
            ix = [FEATURE_NAMES.index(k) for k in keys]
            p = oof_predictions(arms["trace"][:, ix], y, n_folds, seed)
            ablate[bname] = {f"gap_{g}": round(pairwise_accuracy(p, y, g), 4)
                             for g in GAPS}
        results["bucket_ablation"] = ablate
        return results

    raise ValueError(f"unknown mode {mode}")


if __name__ == "__main__":
    from lib.experiment import main
    main(run)
