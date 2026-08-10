"""Flagship figures 1-4 (paper.md), generated from artifacts and cached embeddings.

Design contract (2026-07-10): figures state facts, captions in the draft carry
the narrative. Titles are factual (corpus, n, exemplar id), axis labels are
self-explanatory, and color is meaning-mapped consistently across figures:
    red    = scalar features / the cost of interleaving
    blue   = vector projections
    purple = vector + scalar combinations
    gray   = floors and raw paths
    teal / orange = user / model roles (complementary pair)

Figure 1 (Sec. 1/6.1): the Table 1 scoreboard grouped by representation class,
    five-seed error bars on every bar (rolesplit_graft.json; floor and scalarized-
    coefficients arms from scoreboard_arms.json).
Figure 2 (Sec. 4.4): the aliasing mechanism. Panel A: turns of one strictly-
    alternating dialogue projected onto the user-model axis (a square wave,
    every turn labeled with its index) with the degree-3 least-squares fit
    overlaid — exactly the interleaved projection's representation along that
    axis, since projection is linear per dimension. Panel B: per-dialogue
    residual energy above degree 3, interleaved minus per-role, all of USS-SGD.
Figure 3 (Sec. 7): forest plot of the corpus-level Spearman correlations with
    95% CIs (Fisher z, n=1000) from ledger_stats.json.
Figure 4 (Sec. 6.8): permutation importance vs containment R^2, one panel per
    USS corpus, from containment_graft_uss_{sgd,mwoz}.json.
Encoder scaling (Sec. 6.5): role-split vs interleaved projection accuracy
    across MiniLM/GTE/bge-base on all three corpora, from
    rolesplit_embedders.json and rolesplit_graft.json (the MiniLM anchor);
    color encodes corpus in this figure only, so no clash with the arm
    meaning-map above.

Output: paper/figures/*.{svg,png} (300 dpi); figure
numbers live in the draft, not in file names.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
from lib.campaign import make_run_sample

RES = WORK.parent / "data" / "results" / "trace_h2h"
OUT = WORK.parent / "paper" / "figures"

# Google palette (blue / green / red / yellow + greys). Blue is the positive /
# "our-quantities" colour; red is reserved for negative / contrast emphasis only
# (e.g. the gap in Figure 6). User and model turn colours are kept (teal/orange).
C_SCALAR = "#FBBC04"     # Google yellow - hand-crafted scalar features
C_PROJ = "#4285F4"       # Google blue - our lines / projections / positive
C_COMBO = "#34A853"      # Google green - projection + scalar combinations
C_FLOOR = "#9AA0A6"      # Google grey - trivial floor
C_USER = "#35978f"       # teal - user turns
C_MODEL = "#e08214"      # orange - model turns
C_COST = "#EA4335"       # Google red - negative / contrast emphasis only

CLASS_COLORS = {"floor": C_FLOOR, "scalars": C_SCALAR,
                "projection": C_PROJ, "combo": C_COMBO}


def _save(fig, stem):
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{stem} written")


# ---------------------------------------------------------------- Figure 1


def fig1():
    from matplotlib.patches import Patch
    g = json.load(open(RES / "rolesplit_graft.json"))
    leg = json.load(open(RES / "scoreboard_arms.json"))
    LEG_KEYS = {"floor1": "floor1", "designB": "design_b"}

    def val(corpus, key):
        src = leg if key in LEG_KEYS else g
        b = src[corpus]["multiseed"][LEG_KEYS.get(key, key)]["gap_1.0"]
        return b["mean"], b["std"]

    # one compact panel: arms grouped into class clusters along x, each arm a
    # group of three corpus columns (light->dark = MWOZ/SGD/PRISM).
    arms = [
        ("turn-count\nfloor", "floor1", "floor"),
        ("scalarized\ncoeffs", "designB", "scalars"),
        ("TRACE-\ngeometric", "trace", "scalars"),
        ("mean\npooling", "g0", "projection"),
        ("interleaved\nprojection", "raw", "projection"),
        ("role-split\nprojection", "rolesplit_k1", "projection"),
        ("mean pool\n+ TRACE", "combo", "combo"),
        ("role-split\n+ TRACE", "rs1_trace", "combo"),
    ]
    corpora = [("uss_mwoz", "USS-MWOZ", 0.5), ("uss_sgd", "USS-SGD", 0.75),
               ("prism", "PRISM", 1.0)]
    class_label = {"floor": "floor", "scalars": "scalar features",
                   "projection": "vector projections", "combo": "vector + scalar"}

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    w, base = 0.27, 0.45
    xc, x, prev, cluster = [], 0.0, None, {}
    for label, key, cls in arms:
        if prev is not None and cls != prev:
            x += 0.7
        xc.append(x)
        cluster.setdefault(cls, []).append(x)
        emph = key == "rolesplit_k1"
        for j, (corpus, _, a) in enumerate(corpora):
            v, s = val(corpus, key)
            bx = x + (j - 1) * w
            ax.bar(bx, v - base, width=w * 0.9, bottom=base, color=CLASS_COLORS[cls],
                   alpha=a, edgecolor="k" if emph else "none",
                   lw=1.0 if emph else 0, zorder=3)
            if s:
                ax.errorbar(bx, v, yerr=s, fmt="none", ecolor="#333333",
                            elinewidth=0.7, capsize=1.5, zorder=4)
            ax.text(bx, v + (s or 0) + 0.007, f"{v:.2f}"[1:], ha="center",
                    va="bottom", fontsize=5.4, rotation=90, color="#333333")
        prev = cls
        x += 1.0

    ax.axhline(0.5, color="k", lw=0.7, ls=":", zorder=1)
    ax.set_ylim(base, 0.83)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8])
    ax.set_ylabel("pairwise preference accuracy\n(rating gap ≥ 1.0; dotted = chance)",
                  fontsize=9)
    ax.set_xticks(xc)
    ax.set_xticklabels([a[0] for a in arms], fontsize=7.5)
    ax.set_xlim(-0.7, x - 0.3)
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    for cls, xs in cluster.items():
        ax.text(sum(xs) / len(xs), 0.845, class_label[cls], ha="center",
                fontsize=8, style="italic", color="#444444")
    ax.legend(handles=[Patch(facecolor="#666666", alpha=a, label=n)
                       for _, n, a in corpora],
              fontsize=8, frameon=False, loc="upper left", ncol=1,
              handlelength=1.2, labelspacing=0.35)
    fig.tight_layout()
    _save(fig, "teaser")


# --------------------------------------------------- shared embedding access


def _load_sgd():
    d = np.load(WORK.parent / "data" / "embeddings" / "uss_sgd_minilm.npz")
    meta = json.load(open(WORK.parent / "data" / "embeddings" / "uss_sgd_meta.json"))
    return d["embeddings"], d["offsets"], meta


def _norm(e):
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


def _dialogue(emb, offsets, meta, i):
    e = _norm(emb[offsets[i]:offsets[i + 1]].astype(np.float64))
    roles = meta["roles"][i]
    u_ix = [j for j, r in enumerate(roles) if r == "USER"]
    m_ix = [j for j, r in enumerate(roles) if r != "USER"]
    return e, u_ix, m_ix


def _resid_frac(X, deg=3):
    """Residual energy fraction after the degree<=deg least-squares fit over
    the turn index (identical to truncating the Gram projection at deg)."""
    T = X.shape[0]
    if T <= deg + 1:
        return None
    x = np.arange(T, dtype=np.float64)
    V = np.polynomial.polynomial.polyvander(x, deg)
    coef, *_ = np.linalg.lstsq(V, X, rcond=None)
    R = X - V @ coef
    return float((R ** 2).sum() / (X ** 2).sum())


def _pick_alternating(meta, offsets, emb, min_T=14, max_T=24):
    """Strictly-alternating exemplar maximizing role separation along the
    user-model axis, so the mechanism is visible; Figure 2's panel B carries
    the corpus-wide claim."""
    best, besti = -1e9, None
    for i in range(len(offsets) - 1):
        T = offsets[i + 1] - offsets[i]
        roles = meta["roles"][i]
        if not (min_T <= T <= max_T):
            continue
        if any(roles[j] == roles[j + 1] for j in range(len(roles) - 1)):
            continue
        e, u_ix, m_ix = _dialogue(emb, offsets, meta, i)
        u, m = e[u_ix], e[m_ix]
        between = np.linalg.norm(u.mean(0) - m.mean(0))
        within = (u.std(0).mean() + m.std(0).mean()) / 2 + 1e-9
        if between / within > best:
            best, besti = between / within, i
    return besti


# ---------------------------------------------------------------- Figure 2


def fig2():
    emb, offsets, meta = _load_sgd()

    # Panel A: one dialogue's turns projected onto the user-model axis
    i = _pick_alternating(meta, offsets, emb)
    e, u_ix, m_ix = _dialogue(emb, offsets, meta, i)
    axis = e[u_ix].mean(0) - e[m_ix].mean(0)
    axis /= np.linalg.norm(axis)
    s = e @ axis
    T = len(s)
    x = np.arange(T)
    coef = np.polynomial.polynomial.polyfit(x, s, 3)
    xf = np.linspace(0, T - 1, 240)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    axA.plot(x, s, "-", color="#bbbbbb", lw=1.0, zorder=1)
    axA.plot(x[u_ix], s[u_ix], "o", color=C_USER, ms=7, label="user turns",
             zorder=3)
    axA.plot(x[m_ix], s[m_ix], "s", color=C_MODEL, ms=7, label="model turns",
             zorder=3)
    for j in range(T):  # every turn labeled with its index
        up = j in u_ix
        axA.annotate(str(j), (x[j], s[j]), fontsize=6.5, ha="center",
                     va="bottom" if up else "top",
                     xytext=(0, 6 if up else -7), textcoords="offset points",
                     color=C_USER if up else C_MODEL)
    axA.axhline(s[u_ix].mean(), color=C_USER, lw=0.8, ls=":", alpha=0.8)
    axA.axhline(s[m_ix].mean(), color=C_MODEL, lw=0.8, ls=":", alpha=0.8)
    axA.plot(xf, np.polynomial.polynomial.polyval(xf, coef), "-",
             color=C_COST, lw=2.0, zorder=2,
             label="degree-3 fit = the interleaved\nprojection, on this axis")
    axA.set_xlabel("turn index within the dialogue", fontsize=9)
    axA.set_ylabel("projection of turn embedding\nonto the user−model axis",
                   fontsize=9)
    axA.set_title(f"A. USS-SGD dialogue {i} (T = {T}, strictly alternating; "
                  "dotted lines = role means)", fontsize=8.5)
    lo, hi = s.min(), s.max()
    axA.set_ylim(lo - 0.10, hi + 0.30)  # headroom so the legend clears data
    axA.legend(fontsize=7.5, loc="upper left", frameon=False)
    axA.spines[["top", "right"]].set_visible(False)

    # Panel B: corpus-wide residual energy, interleaved vs per-role
    d_inter, d_role = [], []
    for k in range(len(offsets) - 1):
        e, u_ix, m_ix = _dialogue(emb, offsets, meta, k)
        ri = _resid_frac(e)
        ru = _resid_frac(e[u_ix])
        rm = _resid_frac(e[m_ix])
        if ri is None or ru is None or rm is None:
            continue
        d_inter.append(ri)
        d_role.append((ru + rm) / 2)
    d_inter, d_role = np.array(d_inter), np.array(d_role)
    delta = d_inter - d_role
    axB.hist(delta, bins=40, color=C_COST, alpha=0.8)
    axB.axvline(0, color="k", lw=0.9, ls=":")
    axB.axvline(delta.mean(), color="k", lw=1.4, ls="--",
                label=f"mean +{delta.mean():.3f}\n({100*(delta>0).mean():.0f}%"
                      " of dialogues > 0)")
    axB.set_xlabel("extra residual energy above degree 3 from interleaving\n"
                   "(interleaved − mean per-role, per dialogue)", fontsize=9)
    axB.set_ylabel("number of dialogues", fontsize=9)
    axB.set_title(f"B. all USS-SGD dialogues with ≥ 5 turns per role "
                  f"(n = {len(delta):,})", fontsize=8.5)
    axB.legend(fontsize=8, frameon=False)
    axB.spines[["top", "right"]].set_visible(False)
    print(f"fig2 stats: exemplar={i}, means interleaved={d_inter.mean():.3f} "
          f"per-role={d_role.mean():.3f}, delta>0 {100*(delta>0).mean():.1f}%")
    fig.tight_layout()
    _save(fig, "aliasing_mechanism")


# ---------------------------------------------------------------- Figure 3


def fig3():
    L = json.load(open(RES / "ledger_stats.json"))
    geo, sur = L["role_geometry_sgd"], L["causal_surprise_sgd"]
    n = geo["n_dialogues_used"]
    ci = 1.96 / np.sqrt(n - 3)  # Fisher-z 95% half-width (rho scale, small rho)

    rows = [
        ("three-path geometry", None, None),
        ("parties drift the same way  cos(G₁ᵘ, G₁ᵐ)",
         geo["drift_alignment"]["rho"], geo["drift_alignment"]["p"]),
        ("mean user–model gap  ‖d₀‖",
         geo["gap_mean"]["rho"], geo["gap_mean"]["p"]),
        ("gap growth per turn  slope ‖u−m‖",
         geo["gap_slope"]["rho"], geo["gap_slope"]["p"]),
        ("user path roughness (residual energy)",
         geo["resid_user"]["rho"], geo["resid_user"]["p"]),
        ("model path roughness",
         geo["resid_model"]["rho"], geo["resid_model"]["p"]),
        ("causal per-role surprise", None, None),
        ("model surprise vs static control (trend gain)",
         sur["model_trend_gain"]["rho"], sur["model_trend_gain"]["p"]),
        ("user mean surprise (pushed off own line)",
         sur["user_sur_mean"]["rho"], sur["user_sur_mean"]["p"]),
        ("user max single-step surprise",
         sur["user_sur_max"]["rho"], sur["user_sur_max"]["p"]),
        ("model max single-step surprise",
         sur["model_sur_max"]["rho"], sur["model_sur_max"]["p"]),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    y = 0
    ypos, ylabels = [], []
    for label, rho, p in rows:
        y -= 1
        if rho is None:
            ax.text(-0.245, y, label, fontsize=8.5, style="italic",
                    color="#444444", va="center")
            continue
        sig = p < 0.05
        color = C_PROJ if rho > 0 else C_COST
        ax.errorbar(rho, y, xerr=ci, fmt="o", ms=6,
                    mfc=color if sig else "white", mec=color, ecolor=color,
                    elinewidth=1.2, capsize=2.5)
        ax.text(0.225, y, f"ρ={rho:+.3f}" + ("" if sig else "  n.s."),
                fontsize=7.5, va="center", color="#333333")
        ypos.append(y)
        ylabels.append(label)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_ylim(y - 0.8, 0)
    ax.set_xlim(-0.25, 0.36)
    ax.set_xlabel(f"Spearman ρ between the quantity and dialogue satisfaction"
                  f"\n(USS-SGD, n = {n:,}; bars = 95% CI; filled = p < 0.05)",
                  fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "descriptive_geometry")


# ---------------------------------------------------------------- Figure 4

PLAIN = {
    "user_model_dist_max": "max user→model distance",
    "model_user_dist_max": "max model→user distance",
    "model_self_sim_mean": "mean model self-similarity",
    "model_self_sim_max": "max model self-similarity",
    "n_turns": "turn count",
    "turn_dist_max": "max turn-to-turn jump",
}


def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    # Label positions are data coordinates placed in empty regions of each
    # panel's point cloud, so every leader line is visible and ends on
    # exactly one dot (checked against the artifact values).
    annotate = {
        "uss_sgd": {"model_self_sim_max": (0.030, 0.0145),
                    "user_model_dist_max": (0.000, 0.0080),
                    "n_turns": (0.545, -0.0060)},
        "uss_mwoz": {"model_self_sim_mean": (0.055, 0.0140),
                     "model_user_dist_max": (0.100, 0.0085),
                     "n_turns": (0.390, 0.0025)},
    }
    star = "user_model_dist_max"
    for ax, corpus, title in [(axes[0], "uss_sgd", "USS-SGD"),
                              (axes[1], "uss_mwoz", "USS-MWOZ")]:
        d = json.load(open(RES / f"containment_graft_{corpus}.json"))
        r2, imp = d["containment_r2"], d["perm_importance"]
        feats = sorted(set(r2) & set(imp))
        ax.scatter([r2[f] for f in feats], [imp[f] for f in feats],
                   s=34, c=C_SCALAR, alpha=0.8, lw=0)
        for f, pos in annotate[corpus].items():
            if f in feats:
                ax.annotate(PLAIN.get(f, f), (r2[f], imp[f]), fontsize=7.5,
                            xytext=pos, textcoords="data",
                            ha="left", va="bottom", zorder=5,
                            arrowprops=dict(arrowstyle="-", lw=0.6,
                                            color="#888888", shrinkB=2.5))
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("containment R² — how well the projection "
                      "block linearly\npredicts the feature", fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("permutation importance inside TRACE-geometric\n"
                       "(accuracy lost when the feature is shuffled)",
                       fontsize=8.5)
    mwoz = json.load(open(RES / "containment_graft_uss_mwoz.json"))
    axes[1].scatter([mwoz["containment_r2"][star]],
                    [mwoz["perm_importance"][star]], s=60, facecolors="none",
                    edgecolors="k", lw=1.2, zorder=4)
    axes[1].annotate("max user→model distance:\nrank-4 importance, R² = 0.03",
                     (mwoz["containment_r2"][star],
                      mwoz["perm_importance"][star]),
                     xytext=(0.012, 0.0175), fontsize=8, color="#333333",
                     ha="left", va="bottom",
                     arrowprops=dict(arrowstyle="->", lw=0.9,
                                     color="#333333", shrinkB=5))
    fig.tight_layout()
    _save(fig, "importance_vs_containment")


# ------------------------------------------------- Figure (Sec. 6.5): encoders

EMB_ORDER = [("minilm", "MiniLM\n22M · 384-d"),
             ("gte", "GTE-small\n33M · 384-d"),
             ("bge", "bge-base\n109M · 768-d")]
# corpus-categorical colors, this figure only (role/combo hues unused here)
EMB_CORPORA = [("uss_sgd", "USS-SGD", C_USER),
               ("uss_mwoz", "USS-MWOZ", C_MODEL),
               ("prism", "PRISM", C_COMBO)]


def fig_embedder_scaling():
    graft = json.load(open(RES / "rolesplit_graft.json"))
    emb = json.load(open(RES / "rolesplit_embedders.json"))

    def val(tag, corpus, arm):
        src = (graft[corpus]["multiseed"] if tag == "minilm"
               else emb[tag][corpus]["multiseed"])
        b = src[arm]["gap_1.0"]
        return b["mean"], b["std"]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = list(range(len(EMB_ORDER)))
    for corpus, clabel, color in EMB_CORPORA:
        rs = [val(t, corpus, "rolesplit_k1") for t, _ in EMB_ORDER]
        rw = [val(t, corpus, "raw") for t, _ in EMB_ORDER]
        ax.errorbar(x, [m for m, _ in rs], yerr=[s for _, s in rs],
                    marker="o", ms=6, lw=2.0, color=color, capsize=2.5,
                    label=clabel, zorder=3)
        ax.plot(x, [m for m, _ in rw], marker="o", ms=3.5, lw=1.0, ls="--",
                color=color, alpha=0.45, zorder=2)  # interleaved control
        m, _ = val("bge", corpus, "rolesplit_k1")
        ax.annotate(f"{m:.3f}", (x[-1], m), color=color, fontsize=9,
                    fontweight="bold", xytext=(8, 0),
                    textcoords="offset points", va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in EMB_ORDER], fontsize=8.5)
    ax.set_xlim(-0.25, len(EMB_ORDER) - 1 + 0.6)
    ax.set_ylabel("pairwise preference accuracy, rating gap ≥ 1.0", fontsize=9)
    ax.set_title("Role-split projection scales with encoder strength on the "
                 "USS corpora\n(solid = role-split, dashed = interleaved; the "
                 "gain persists at every encoder)", fontsize=9)
    ax.legend(title="corpus", fontsize=8.5, title_fontsize=8.5, frameon=False,
              loc="center left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.3)
    fig.tight_layout()
    _save(fig, "embedder_scaling")


# --------------------------------------- Path-trajectory figures (Sec. 6.5, 7)
# Merged from the former flagship_path_figures.py (2026-07-13): two exemplar
# dialogues in PCA-2D - role-split paths, three-path geometry, drift-gap
# profiles. Colors follow the same meaning map (teal user / orange model /
# red = gap or interleaving cost).


def _pca2(X):
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T


def _fit1(P):
    """Degree-1 least-squares fit of a path over its index (rows = points).
    Identical to the Gram degree-0+1 reconstruction on that grid."""
    x = np.arange(len(P), dtype=float)
    V = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(V, P, rcond=None)
    return V @ coef, coef[1]  # fitted points, slope vector


def _alternating_ids(meta, offsets, min_T=14, max_T=24, exclude=()):
    out = []
    for i in range(len(offsets) - 1):
        T = offsets[i + 1] - offsets[i]
        roles = meta["roles"][i]
        if i in exclude or not (min_T <= T <= max_T):
            continue
        if any(roles[j] == roles[j + 1] for j in range(len(roles) - 1)):
            continue
        out.append(i)
    return out


def _pick_short_extremes(meta, offsets, max_T=12):
    """Highest- and lowest-rated strictly-alternating dialogues among the
    shortest (T <= max_T); ties broken toward fewer turns."""
    ids = _alternating_ids(meta, offsets, min_T=6, max_T=max_T)
    ys = meta["overall_mean"]
    hi = max(ids, key=lambda i: (ys[i], -(offsets[i + 1] - offsets[i])))
    lo = min(ids, key=lambda i: (ys[i], offsets[i + 1] - offsets[i]))
    return hi, lo


def _um_labels(ax, Pu, Pm, fs=10.5):
    for j in range(len(Pu)):
        ax.annotate(f"U{j + 1}", Pu[j], fontsize=fs, color=C_USER,
                    fontweight="bold", ha="center", va="bottom",
                    xytext=(0, 9), textcoords="offset points")
    for j in range(len(Pm)):
        ax.annotate(f"M{j + 1}", Pm[j], fontsize=fs, color=C_MODEL,
                    fontweight="bold", ha="center", va="top",
                    xytext=(0, -10), textcoords="offset points")


def _pair_labels(ax, Pu, Pm, color_u=C_USER, color_m=C_MODEL, fs=7):
    # alternate a small horizontal jitter so labels of near-coincident
    # consecutive turns stay legible (from methodology_figures)
    for j in range(len(Pu)):
        dx = 5 if j % 2 else -5
        ax.annotate(str(j + 1), Pu[j], fontsize=fs, ha="center", va="bottom",
                    xytext=(dx, 6), textcoords="offset points", color=color_u)
    for j in range(len(Pm)):
        dx = 5 if j % 2 else -5
        ax.annotate(str(j + 1), Pm[j], fontsize=fs, ha="center", va="top",
                    xytext=(dx, -7), textcoords="offset points", color=color_m)


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def _exemplars():
    emb, offsets, meta = _load_sgd()
    hi, lo = _pick_short_extremes(meta, offsets)
    return emb, offsets, meta, hi, lo


def _pair_data(emb, offsets, meta, i):
    e, u_ix, m_ix = _dialogue(emb, offsets, meta, i)
    k = min(len(u_ix), len(m_ix))
    U, M = e[u_ix[:k]], e[m_ix[:k]]
    return U, M, (U + M) / 2.0, np.linalg.norm(U - M, axis=1)


def role_split_paths():
    emb, offsets, meta, hi, lo = _exemplars()
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 10.6))
    for row, i, tag in [(0, hi, "high-rated"), (1, lo, "low-rated")]:
        e, u_ix, m_ix = _dialogue(emb, offsets, meta, i)
        P = _pca2(e)
        k = min(len(u_ix), len(m_ix))
        Pu, Pm = P[u_ix[:k]], P[m_ix[:k]]
        axA, axB = axes[row]

        axA.plot(P[:, 0], P[:, 1], "-", color="#aaaaaa", lw=1.4, zorder=1,
                 label="alternating path")
        fitA, _ = _fit1(P)
        axA.plot(fitA[:, 0], fitA[:, 1], "-", color=C_COST, lw=2.6, zorder=2,
                 label="degree-1 fit")
        axA.plot(Pu[:, 0], Pu[:, 1], "o", color=C_USER, ms=10, zorder=3)
        axA.plot(Pm[:, 0], Pm[:, 1], "s", color=C_MODEL, ms=10, zorder=3)
        _um_labels(axA, Pu, Pm)
        axA.set_title(f"{tag} dialogue {i} (satisfaction "
                      f"{meta['overall_mean'][i]:.2f} / 5) — "
                      "Gram over alternating turns", fontsize=11.5)

        axB.plot(Pu[:, 0], Pu[:, 1], "-o", color=C_USER, lw=2.4, ms=10,
                 zorder=3, label="user path")
        axB.plot(Pm[:, 0], Pm[:, 1], "-s", color=C_MODEL, lw=2.4, ms=10,
                 zorder=3, label="model path")
        fitU, _ = _fit1(Pu)
        fitM, _ = _fit1(Pm)
        axB.plot(fitU[:, 0], fitU[:, 1], "--", color=C_USER, lw=2.4,
                 alpha=0.75, zorder=2, label="degree-1 fits (per role)")
        axB.plot(fitM[:, 0], fitM[:, 1], "--", color=C_MODEL, lw=2.4,
                 alpha=0.75, zorder=2)
        _um_labels(axB, Pu, Pm)
        axB.set_title("Gram role-split(1)", fontsize=11.5)

        for ax in (axA, axB):
            ax.margins(0.12)
            ax.legend(fontsize=10, frameon=False, loc="best")
            _bare(ax)
    fig.tight_layout()
    _save(fig, "role_split_paths")


def three_path_geometry():
    emb, offsets, meta, hi, lo = _exemplars()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.0))
    for ax, i, tag in [(axes[0], hi, "high-rated"), (axes[1], lo, "low-rated")]:
        U, M, C, gap = _pair_data(emb, offsets, meta, i)
        k = len(U)
        P = _pca2(np.vstack([U, M, C]))
        Pu, Pm, Pc = P[:k], P[k:2 * k], P[2 * k:]
        widest = int(np.argmax(gap))  # one annotated gap, TRACE-style
        for j in range(k):
            emph = j == widest
            ax.plot([Pu[j, 0], Pm[j, 0]], [Pu[j, 1], Pm[j, 1]],
                    color=C_COST, lw=2.0 if emph else 1.3,
                    alpha=0.9 if emph else 0.5, zorder=1)
        mid = (Pu[widest] + Pm[widest]) / 2
        ax.annotate("gap ‖u − m‖", mid, fontsize=10, color=C_COST,
                    style="italic", ha="left", va="center",
                    xytext=(10, 0), textcoords="offset points")
        ax.plot(Pc[:, 0], Pc[:, 1], "--", color="#888888", lw=2.0, zorder=2,
                label="midline (u+m)/2")
        ax.plot(Pu[:, 0], Pu[:, 1], "-", color=C_USER, lw=2.4, zorder=3,
                label="user path")
        ax.plot(Pm[:, 0], Pm[:, 1], "-", color=C_MODEL, lw=2.4, zorder=3,
                label="model path")
        ax.plot(Pu[:, 0], Pu[:, 1], "o", color=C_USER, ms=10, zorder=4)
        ax.plot(Pm[:, 0], Pm[:, 1], "s", color=C_MODEL, ms=10, zorder=4)
        _um_labels(ax, Pu, Pm)
        ax.set_title(f"{tag} — satisfaction {meta['overall_mean'][i]:.2f} / 5"
                     f"    (USS-SGD dialogue {i}, T = {2 * k})", fontsize=12)
        ax.margins(0.12)
        ax.legend(fontsize=10, frameon=False, loc="lower left")
        _bare(ax)
    fig.tight_layout()
    _save(fig, "three_path_geometry")


def drift_gap_profiles():
    emb, offsets, meta, hi, lo = _exemplars()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, i, tag in [(axes[0], hi, "high-rated"), (axes[1], lo, "low-rated")]:
        U, M, _, gap = _pair_data(emb, offsets, meta, i)
        _, g1u = _fit1(U)
        _, g1m = _fit1(M)
        drift_cos = float(g1u @ g1m /
                          (np.linalg.norm(g1u) * np.linalg.norm(g1m) + 1e-12))
        x = np.arange(len(gap))
        ax.plot(x, gap, "-o", color=C_PROJ, lw=1.6, ms=5)
        z = np.polyfit(x, gap, 1)
        ax.plot(x, np.polyval(z, x), "--", color="k", lw=1.2,
                label=f"gap trend {z[0]:+.3f} / pair")
        ax.set_xlabel("turn pair (in order)", fontsize=9)
        ax.set_ylabel("gap ‖user − model‖ per pair", fontsize=9)
        ax.set_title(f"{tag} dialogue {i} (satisfaction "
                     f"{meta['overall_mean'][i]:.2f} / 5)\n"
                     f"drift alignment cos(G₁ᵘ, G₁ᵐ) = {drift_cos:+.2f};  "
                     f"mean gap {gap.mean():.2f}", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "drift_gap_profiles")


def fig_concept():
    """Two opening intuition figures (no per-panel titles), one USS-SGD dialogue
    in PCA-2D, each a 1x3 row of raw vectors / degree-3 shape / spline.
    trajectory_concept_interleaved (paper Figure 1): the whole conversation as a
    single interleaved path. trajectory_concept_split (paper Figure 2): the same
    dialogue split by speaker. Both share one set of axis limits so the dialogue
    sits identically across the pair; each carries its own user/model key.
    Reading lives in the captions."""
    from scipy.interpolate import splprep, splev
    emb, offsets, meta = _load_sgd()
    i = 667
    e, u_ix, m_ix = _dialogue(emb, offsets, meta, i)
    P = _pca2(e)
    is_user = np.zeros(len(P), bool)
    is_user[u_ix] = True
    Pu, Pm = P[is_user], P[~is_user]
    span = np.hypot(*np.ptp(P, axis=0))

    def fit3(pts):
        idx = np.arange(len(pts), dtype=float)
        tt = np.linspace(0, len(pts) - 1, 400)
        return np.column_stack([np.polyval(np.polyfit(idx, pts[:, c], 3), tt)
                                for c in range(2)])

    def spline(pts):
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, k=min(3, len(pts) - 1))
        return np.column_stack(splev(np.linspace(0, 1, 400), tck))

    def arrow(ax, curve, color):
        d = curve[-1] - curve[-2]
        d = d / (np.linalg.norm(d) + 1e-12)
        ax.annotate("", curve[-1] + d * 0.05 * span, curve[-1], zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2,
                                    shrinkA=0, shrinkB=0, mutation_scale=12))

    def points(ax):
        for t in range(len(P)):
            c = C_USER if is_user[t] else C_MODEL
            ax.plot(*P[t], "o" if is_user[t] else "s", color=c, ms=7,
                    mec="white", mew=0.9, zorder=4)
            dx = 5 if t % 2 else -5
            ax.annotate(str(t + 1), P[t], color=c, fontsize=5.5,
                        fontweight="bold", ha="center", va="center",
                        xytext=(dx, 8), textcoords="offset points", zorder=5)

    def faint(ax, *paths):
        for p in paths:
            ax.plot(p[:, 0], p[:, 1], "-", color="#dcdcdc", lw=1.0, zorder=1)

    # shared limits: union of every plotted curve across both figures, so the
    # dialogue lands in exactly the same place in Figure 1 and Figure 2.
    allpts = np.vstack([P, fit3(P), spline(P),
                        fit3(Pu), fit3(Pm), spline(Pu), spline(Pm)])
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.08 * (hi - lo)
    xlim = (lo[0] - pad[0], hi[0] + pad[0])
    ylim = (lo[1] - pad[1], hi[1] + pad[1])

    def dress(axes, ylabel):
        for ax in axes:
            points(ax)
        axes[0].set_ylabel(ylabel, fontsize=11)
        axes[0].plot([], [], "o", color=C_USER, ms=7, label="user turn")
        axes[0].plot([], [], "s", color=C_MODEL, ms=7, label="model turn")
        axes[0].legend(fontsize=8, frameon=False, loc="lower right")
        for ax, letter in zip(axes, "abc"):
            ax.text(0.03, 0.95, f"({letter})", transform=ax.transAxes,
                    fontsize=10.5, fontweight="bold", va="top", color="#333333")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#cfcfcf")

    # Figure 1 - the whole conversation interleaved
    figA, (iV, iS, iP) = plt.subplots(1, 3, figsize=(13.5, 4.9),
                                      sharex=True, sharey=True)
    iV.plot(P[:, 0], P[:, 1], "-", color=C_PROJ, lw=2.2, zorder=2)
    arrow(iV, P, C_PROJ)
    faint(iS, P)
    iS.plot(*fit3(P).T, "-", color=C_PROJ, lw=2.6, zorder=2)
    arrow(iS, fit3(P), C_PROJ)
    faint(iP, P)
    iP.plot(*spline(P).T, "-", color=C_PROJ, lw=2.6, zorder=2)
    arrow(iP, spline(P), C_PROJ)
    dress((iV, iS, iP), "interleaved")
    figA.tight_layout()
    _save(figA, "trajectory_concept_interleaved")

    # Figure 2 - the same dialogue split by speaker
    figB, (rV, rS, rP) = plt.subplots(1, 3, figsize=(13.5, 4.9),
                                      sharex=True, sharey=True)
    rV.plot(Pu[:, 0], Pu[:, 1], "-", color=C_USER, lw=2.0, zorder=2)
    rV.plot(Pm[:, 0], Pm[:, 1], "-", color=C_MODEL, lw=2.0, zorder=2)
    arrow(rV, Pu, C_USER)
    arrow(rV, Pm, C_MODEL)
    for ax, fn in ((rS, fit3), (rP, spline)):
        faint(ax, Pu, Pm)
        ax.plot(*fn(Pu).T, "-", color=C_USER, lw=2.4, zorder=2)
        ax.plot(*fn(Pm).T, "-", color=C_MODEL, lw=2.4, zorder=2)
        arrow(ax, fn(Pu), C_USER)
        arrow(ax, fn(Pm), C_MODEL)
    dress((rV, rS, rP), "split by speaker")
    figB.tight_layout()
    _save(figB, "trajectory_concept_split")


def run(max_samples=None):
    if max_samples is not None:
        raise ValueError(
            "actr_figures renders assets from flat artifacts and cached "
            "embeddings and cannot cap in memory; use the sandbox driver "
            "instead: reproduce.py --max-samples N caps the raw layer "
            "(lib/corpus_sample.py).")
    OUT.mkdir(parents=True, exist_ok=True)
    fig_concept()
    fig1()
    fig2()
    fig3()
    fig4()
    fig_embedder_scaling()
    role_split_paths()
    three_path_geometry()
    drift_gap_profiles()
    return {stem: [str(OUT / f"{stem}.{ext}") for ext in ("svg", "png")]
            for stem in ("trajectory_concept_interleaved",
                         "trajectory_concept_split", "teaser",
                         "aliasing_mechanism", "descriptive_geometry",
                         "importance_vs_containment", "embedder_scaling",
                         "role_split_paths", "three_path_geometry",
                         "drift_gap_profiles")}


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
