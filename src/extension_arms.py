"""Feature arms for issues #9-#13: haar, rolesplit, rqa, spectrum, surprisal.

Each block is evaluated two ways on the standard harness:
    block alone, and [design_b_raw | block] vs design_b_raw (marginal value).

Surprisal (#13) additionally gets a cross-corpus transfer check
(train on SGD, test on MWOZ) against a volatility-feature control.
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))

from lib.campaign import cap, make_run_sample
from lib.encoders.gram import gram_fit
from trace_h2h import (load_corpus, build_all_arms, dialogue_iter,
                       oof_predictions, pairwise_accuracy, bootstrap_diff, GAPS)

N_FOLDS, N_BOOT, SEED = 5, 1000, 42


def _norm(e):
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


# ---------- #9 haar: localized linear functionals (levels 0-2, any n) ----------

def _haar_rows(n):
    """4 fixed localized functionals valid for any n: mean, half contrast,
    quarter contrasts. Rows are L2-normalized weight vectors over turns."""
    rows = np.zeros((4, n))
    rows[0] = 1.0 / n
    h = n // 2 or 1
    rows[1, :h] = 1.0 / h
    rows[1, h:] = -1.0 / max(n - h, 1)
    q = n // 4 or 1
    rows[2, :q] = 1.0 / q
    rows[2, q:2 * q] = -1.0 / q
    rows[3, -q:] = 1.0 / q
    rows[3, -2 * q:-q] = -1.0 / q
    for i in range(1, 4):
        nz = np.linalg.norm(rows[i])
        if nz > 0:
            rows[i] /= nz
    return rows


def build_haar(emb, offsets, meta):
    rows = []
    for i, e, roles in dialogue_iter(emb, offsets, meta):
        en = _norm(e)
        rows.append((_haar_rows(len(en)) @ en).ravel())
    return np.asarray(rows, float)


# ---------- #10 role-split + coupling ----------

def build_rolesplit(emb, offsets, meta, degree=3):
    raw_rows, cpl_rows = [], []
    for i, e, roles in dialogue_iter(emb, offsets, meta):
        en = _norm(e)
        u = en[[j for j, r in enumerate(roles) if r == "USER"]]
        m = en[[j for j, r in enumerate(roles) if r != "USER"]]
        gu = gram_fit(u, degree=degree) if len(u) else np.zeros((degree + 1, en.shape[1]))
        gm = gram_fit(m, degree=degree) if len(m) else np.zeros((degree + 1, en.shape[1]))
        raw_rows.append(np.concatenate([gu.ravel(), gm.ravel()]))

        def cos(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(a @ b / (na * nb)) if na > 1e-10 and nb > 1e-10 else 0.0
        du = np.diff(u, axis=0)
        dm = np.diff(m, axis=0)
        k = min(len(du), len(dm))
        lag = float(np.mean(np.sum(_norm(du[:k]) * _norm(dm[:k]), axis=1))) if k > 0 else 0.0
        cpl_rows.append([cos(gu[1], gm[1]), cos(gu[2], gm[2]),
                         cos(gu[1], gm[0] - gu[0]), lag])
    return np.asarray(raw_rows, float), np.asarray(cpl_rows, float)


# ---------- #11 RQA ----------

def build_rqa(emb, offsets, meta, thresh=0.7):
    rows = []
    for i, e, roles in dialogue_iter(emb, offsets, meta):
        en = _norm(e)
        n = len(en)
        R = (en @ en.T) >= thresh
        np.fill_diagonal(R, False)
        rr = float(R.sum()) / max(n * (n - 1), 1)
        diag_lengths = []
        for off in range(1, n):
            d = np.diagonal(R, offset=off)
            run = 0
            for v in list(d) + [False]:
                if v:
                    run += 1
                elif run:
                    diag_lengths.append(run)
                    run = 0
        det = (sum(l for l in diag_lengths if l >= 2) / max(R.sum() / 2, 1)) if diag_lengths else 0.0
        lmax = max(diag_lengths) if diag_lengths else 0
        lmean = float(np.mean(diag_lengths)) if diag_lengths else 0.0
        if diag_lengths:
            _, counts = np.unique(diag_lengths, return_counts=True)
            p = counts / counts.sum()
            ent = float(-(p * np.log(p)).sum())
        else:
            ent = 0.0
        rows.append([rr, det, float(lmax), lmean, ent])
    return np.asarray(rows, float)


# ---------- #12 spectrum ----------

def build_spectrum(emb, offsets, meta, k=8):
    rows = []
    for i, e, roles in dialogue_iter(emb, offsets, meta):
        en = _norm(e)
        c = en - en.mean(axis=0, keepdims=True)
        ev = np.linalg.eigvalsh(c @ c.T)[::-1]
        ev = np.clip(ev, 0, None)
        tot = ev.sum() or 1.0
        p = ev / tot
        top = np.zeros(k)
        top[:min(k, len(p))] = p[:k]
        pr = float(1.0 / np.sum(p ** 2)) if tot > 0 else 0.0
        pe = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        rows.append(list(top) + [pr, pe])
    return np.asarray(rows, float)


# ---------- #13 surprisal ----------

def build_surprisal(emb, offsets, meta, degree=3):
    rows = []
    for i, e, roles in dialogue_iter(emb, offsets, meta):
        en = _norm(e)
        if len(en) < 3:
            rows.append([0.0] * (degree + 1 + 2))
            continue
        x, ynext = en[:-1], en[1:]
        a = float(np.sum(x * ynext) / (np.sum(x * x) + 1e-12))  # per-dialogue AR(1)
        resid = ynext - a * x
        bits = 0.5 * en.shape[1] * np.log2(np.sum(resid ** 2, axis=1) / en.shape[1] + 1e-9)
        g = gram_fit(bits[:, None], degree=degree)[:, 0]
        rows.append(list(g) + [float(bits.sum()), float(bits.mean())])
    return np.asarray(rows, float)


# ---------- evaluation ----------

def evaluate(dataset, blocks, raw, y):
    preds_raw = oof_predictions(raw, y, N_FOLDS, SEED)
    out = {}
    for bname, X in blocks.items():
        p_alone = oof_predictions(X, y, N_FOLDS, SEED)
        p_aug = oof_predictions(np.hstack([raw, X]), y, N_FOLDS, SEED)
        entry = {}
        for gap in GAPS:
            entry[f"gap_{gap}"] = {
                "alone": round(pairwise_accuracy(p_alone, y, gap), 4),
                "augmented": round(pairwise_accuracy(p_aug, y, gap), 4),
                "raw_baseline": round(pairwise_accuracy(preds_raw, y, gap), 4),
                "aug_vs_raw": bootstrap_diff(p_aug, preds_raw, y, gap, N_BOOT, SEED),
            }
        out[bname] = entry
        for gap in GAPS:
            g = entry[f"gap_{gap}"]
            v = g["aug_vs_raw"]
            print(f"[{dataset}] {bname:12s} gap {gap}: alone={g['alone']:.4f} "
                  f"aug={g['augmented']:.4f} raw={g['raw_baseline']:.4f} "
                  f"CI [{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] p={v['p_boot']:.3f}",
                  flush=True)
    return out


def transfer_check(max_samples=None):
    """#13: train on SGD, test on MWOZ - surprisal features vs volatility control."""
    from sklearn.ensemble import RandomForestRegressor
    res = {}
    data = {}
    for ds in ["uss_sgd", "uss_mwoz"]:
        emb, offsets, meta = load_corpus(ds)
        emb, offsets, meta = cap(emb, offsets, meta, max_samples)
        y = np.asarray(meta["overall_mean"], float)
        sur = build_surprisal(emb, offsets, meta)
        vol_rows = []
        for i, e, roles in dialogue_iter(emb, offsets, meta):
            en = _norm(e)
            v = 1.0 - np.sum(en[:-1] * en[1:], axis=1)
            g = gram_fit(v[:, None], degree=3)[:, 0]
            vol_rows.append(list(g) + [float(v.sum()), float(v.mean())])
        data[ds] = (sur, np.asarray(vol_rows), y)
    for fname, fi in [("surprisal", 0), ("volatility_ctrl", 1)]:
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                   random_state=SEED, n_jobs=-1)
        rf.fit(data["uss_sgd"][fi], data["uss_sgd"][2])
        preds = rf.predict(data["uss_mwoz"][fi])
        res[fname] = {f"gap_{g}": round(pairwise_accuracy(
            preds, data["uss_mwoz"][2], g), 4) for g in GAPS}
    print("[transfer sgd->mwoz]", res, flush=True)
    return res


def run(max_samples=None):
    results = {}
    for ds in ["uss_sgd", "uss_mwoz"]:
        emb, offsets, meta = load_corpus(ds)
        emb, offsets, meta = cap(emb, offsets, meta, max_samples)
        y = np.asarray(meta["overall_mean"], float)
        arms = build_all_arms(emb, offsets, meta, degree=3)
        raw = arms["design_b_raw"]
        role_raw, coupling = build_rolesplit(emb, offsets, meta)
        spec = build_spectrum(emb, offsets, meta)
        nt = np.asarray([len(r) for r in meta["roles"]], float)
        spec_conf = float(np.corrcoef(spec[:, -2], nt)[0, 1])  # PR vs n_turns
        print(f"[{ds}] spectrum participation-ratio vs n_turns r={spec_conf:.3f}",
              flush=True)
        blocks = {
            "haar": build_haar(emb, offsets, meta),
            "rolesplit": role_raw,
            "coupling": coupling,
            "rqa": build_rqa(emb, offsets, meta),
            "spectrum": spec,
            "surprisal": build_surprisal(emb, offsets, meta),
        }
        results[ds] = evaluate(ds, blocks, raw, y)
        results[ds]["spectrum_nturns_r"] = spec_conf
    results["transfer"] = transfer_check(max_samples)
    if max_samples is None:
        from campaign_io import save_result
        save_result("extension_arms_uss", results, dataset="uss_sgd.uss_mwoz",
                    config="minilm", params={"seed": SEED, "n_folds": N_FOLDS})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
