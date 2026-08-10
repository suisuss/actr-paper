"""
TRACE-geometric feature set, reimplemented from Appendix A (Table 3) of:
    Gooding & Grefenstette, "Interaction Dynamics as a Reward Signal for LLMs",
    arXiv:2511.08394 (TRACE).
Reimplementation spec: material/literature/trace_interaction_dynamics.md

Scope: the 26 geometric features. TRACE's 4 Temporal Dynamics features
(turn durations, median/MAD gap time) require timestamps that public
satisfaction datasets do not provide. Results from this module must be
reported as "TRACE-geometric", not TRACE.

Notation (paper's): d(A,B) = cosine distance; U_i = user turn i paired with
M_i = the model turn that answers it; T_i = turn i (either speaker); G = goal
embedding; T = number of turns.

Goal-embedding analog: USS has no stated-goal field; G = FIRST USER UTTERANCE.
Documented degeneracies under this proxy: goal_initial_prompt_dist == 0 and
model_goal_dist_mean == model_initial_prompt_dist. On PRISM, G = opening
prompt, which IS the stated goal (same two features still merge).

Documented decisions for the paper's underspecified features:
    - semantic_cohesion "Context_{<i}": mean of all prior turn embeddings.
    - late_volatility window k: last max(2, ceil(T/4)) turn-to-turn steps.
    - Model/user pairing: M_i = model turn following user turn U_i.
    - goal_convergence_ratio guard: 0.0 when min_i d(M_i, G) <= 1e-9 (the
      paper leaves the vanishing denominator undefined). Measured 2026-07-12:
      fires on exactly one dialogue corpus-wide - PRISM #3847, where a model
      turn verbatim-echoes the opening prompt - and never on either USS
      corpus (smallest USS denominator 0.045).

All distances are cosine distances on L2-normalized embeddings.
"""

from typing import Dict, List

import numpy as np

FEATURE_NAMES = [
    # Category 1: Inefficiency and Repetition
    "n_turns", "model_self_sim_mean", "model_self_sim_max",
    # Category 3: Semantic Cohesion and Relevance
    "initial_response_dist",
    "model_user_dist_mean", "model_user_dist_max", "model_user_dist_min",
    "user_model_dist_mean", "user_model_dist_max",
    "relevance_trend", "semantic_cohesion",
    "volatility_mean", "turn_dist_max", "late_volatility",
    "user_self_consistency",
    # Category 4: Goal Orientation
    "model_goal_dist_mean", "user_goal_dist_mean",
    "model_goal_dist_min", "model_goal_dist_max",
    "final_turn_goal_dist", "final_model_goal_dist",
    "model_initial_prompt_dist", "goal_initial_prompt_dist",
    "conv_drift_from_goal", "goal_adherence_trend", "goal_convergence_ratio",
]

# Floor baseline (Spike 2 kill switch): trivial counts only.
FLOOR_NAMES = ["n_turns"]


def _normalize(emb: np.ndarray) -> np.ndarray:
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)


def _slope(y: np.ndarray) -> float:
    """OLS slope of y over 0..n-1; 0 for fewer than 2 points."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x = x - x.mean()
    denom = float(x @ x)
    return float(x @ (y - y.mean()) / denom) if denom > 0 else 0.0


def _stats(a, default=0.0):
    a = np.asarray(a, dtype=np.float64)
    if len(a) == 0:
        return default, default, default
    return float(a.mean()), float(a.max()), float(a.min())


def trace_scalar_series(emb: np.ndarray, roles: List[str],
                        goal_idx: int = None) -> Dict[str, np.ndarray]:
    """The scalar series TRACE summarizes with means, slopes, and extremes.

    Design A (Spike 3) expands these same series with Gram degrees 0-3.
    NOTE: order-statistic features (max/min) are NOT polynomial functionals of
    these series; subsumption claims cover the mean/slope/endpoint family only.

    Returns dict of 1-D arrays over their natural index:
        volatility:      d(T_i, T_{i-1})                       (len T-1)
        model_relevance: 1 - d(M_i, U_i)  (paired user turn)
        goal_dist_user:  d(U_i, G)
        goal_dist_model: d(M_i, G)
        user_from_model: d(U_i, M_{i-1})                        (len <= users-1)
    """
    e = _normalize(emb)
    roles = list(roles)
    n = len(roles)
    if goal_idx is None:
        goal_idx = next((i for i, r in enumerate(roles) if r == "USER"), 0)
    goal = e[goal_idx]

    volatility = 1.0 - np.sum(e[:-1] * e[1:], axis=1) if n > 1 else np.zeros(0)

    model_relevance, goal_dist_user, goal_dist_model, user_from_model = [], [], [], []
    last_user, last_model = None, None
    for i, r in enumerate(roles):
        if r == "USER":
            if last_model is not None:
                user_from_model.append(1.0 - float(e[i] @ last_model))
            last_user = e[i]
            goal_dist_user.append(1.0 - float(e[i] @ goal))
        else:
            goal_dist_model.append(1.0 - float(e[i] @ goal))
            if last_user is not None:
                model_relevance.append(float(e[i] @ last_user))
            last_model = e[i]

    return {k: np.asarray(v, dtype=np.float64) for k, v in {
        "volatility": volatility,
        "model_relevance": model_relevance,
        "goal_dist_user": goal_dist_user,
        "goal_dist_model": goal_dist_model,
        "user_from_model": user_from_model,
    }.items()}


def trace_geometric_features(emb: np.ndarray, roles: List[str],
                             goal_idx: int = None) -> Dict[str, float]:
    """Compute the 26 TRACE-geometric features (Table 3) for one dialogue."""
    e = _normalize(emb)
    roles = list(roles)
    n = len(roles)
    user_ix = [i for i, r in enumerate(roles) if r == "USER"]
    model_ix = [i for i, r in enumerate(roles) if r != "USER"]
    if goal_idx is None:
        goal_idx = user_ix[0] if user_ix else 0
    goal = e[goal_idx]
    s = trace_scalar_series(emb, roles, goal_idx)

    # Category 1 - all-pairs model self-similarity (paper definition)
    if len(model_ix) >= 2:
        M = e[model_ix]
        sims = M @ M.T
        iu = np.triu_indices(len(model_ix), 1)
        mss_mean, mss_max = float(sims[iu].mean()), float(sims[iu].max())
    else:
        mss_mean = mss_max = 0.0

    # Category 3
    mu_dist = 1.0 - s["model_relevance"]  # d(M_i, U_i) per answered user turn
    mu_mean, mu_max, mu_min = _stats(mu_dist)
    um_mean, um_max, _ = _stats(s["user_from_model"])
    initial_response = float(mu_dist[0]) if len(mu_dist) else 0.0

    # semantic cohesion: avg sim(U_i, mean of ALL prior turns), i > first
    cohesion_vals = []
    if n > 1:
        cum = np.cumsum(e, axis=0)
        for i in user_ix:
            if i == 0:
                continue
            ctx = cum[i - 1] / i
            ctx = ctx / (np.linalg.norm(ctx) + 1e-12)
            cohesion_vals.append(float(e[i] @ ctx))
    cohesion = float(np.mean(cohesion_vals)) if cohesion_vals else 0.0

    vol = s["volatility"]
    vol_mean, vol_max, _ = _stats(vol)
    k = max(2, int(np.ceil(n / 4)))
    late_vol = float(vol[-k:].mean()) if len(vol) else 0.0

    if len(user_ix) >= 2:
        U = e[user_ix]
        usc = float(np.mean(1.0 - np.sum(U[:-1] * U[1:], axis=1)))
    else:
        usc = 0.0

    # Category 4
    gdm, gdu = s["goal_dist_model"], s["goal_dist_user"]
    gdm_mean, gdm_max, gdm_min = _stats(gdm)
    gdu_mean, _, _ = _stats(gdu)
    final_turn_goal = 1.0 - float(e[-1] @ goal)
    final_model_goal = float(gdm[-1]) if len(gdm) else 0.0
    u1 = e[user_ix[0]] if user_ix else e[0]
    mip = float(np.mean(1.0 - e[model_ix] @ u1)) if model_ix else 0.0
    gip = 1.0 - float(goal @ u1)
    center = e.mean(axis=0)
    center = center / (np.linalg.norm(center) + 1e-12)
    drift = 1.0 - float(center @ goal)
    convergence_ratio = final_turn_goal / gdm_min if gdm_min > 1e-9 else 0.0

    return {
        "n_turns": float(n),
        "model_self_sim_mean": mss_mean, "model_self_sim_max": mss_max,
        "initial_response_dist": initial_response,
        "model_user_dist_mean": mu_mean, "model_user_dist_max": mu_max,
        "model_user_dist_min": mu_min,
        "user_model_dist_mean": um_mean, "user_model_dist_max": um_max,
        "relevance_trend": _slope(mu_dist),
        "semantic_cohesion": cohesion,
        "volatility_mean": vol_mean, "turn_dist_max": vol_max,
        "late_volatility": late_vol,
        "user_self_consistency": usc,
        "model_goal_dist_mean": gdm_mean, "user_goal_dist_mean": gdu_mean,
        "model_goal_dist_min": gdm_min, "model_goal_dist_max": gdm_max,
        "final_turn_goal_dist": final_turn_goal,
        "final_model_goal_dist": final_model_goal,
        "model_initial_prompt_dist": mip,
        "goal_initial_prompt_dist": gip,
        "conv_drift_from_goal": drift,
        "goal_adherence_trend": _slope(gdm),
        "goal_convergence_ratio": convergence_ratio,
    }


def features_matrix(embeddings: np.ndarray, offsets: np.ndarray,
                    roles: List[List[str]]) -> np.ndarray:
    """Vectorize a corpus: (n_dialogues, len(FEATURE_NAMES)) feature matrix."""
    rows = []
    for i in range(len(offsets) - 1):
        emb = embeddings[offsets[i]:offsets[i + 1]]
        f = trace_geometric_features(emb, roles[i])
        rows.append([f[k] for k in FEATURE_NAMES])
    return np.asarray(rows, dtype=np.float64)
