"""Issue #31: embedding-dimension reduction frontier.

Per-fold PCA of the MiniLM turn embeddings to d in {192, 96, 48}, role-split
blocks rebuilt at each d, evaluated on both USS corpora. Charts the joint
accuracy-vs-total-dims frontier with #30 (degree) and #22 (PCA-26).
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))

from lib.campaign import Checkpoint, cap, make_run_sample
from lib.encoders.gram import gram_fit
from trace_h2h import load_corpus, pairwise_accuracy, bootstrap_diff, GAPS
from extension_arms import build_rolesplit

N_FOLDS, N_BOOT, SEED = 5, 1000, 42
DIMS = [192, 96, 48]
_CKPT = Checkpoint("dim_frontier", script=__file__)
ckpt_load = _CKPT.load
ckpt_save = _CKPT.save



def rolesplit_from_turns(emb, offsets, roles_list, degree=3):
    rows = []
    for i in range(len(offsets) - 1):
        e = emb[offsets[i]:offsets[i + 1]]
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
        roles = roles_list[i]
        u = e[[j for j, r in enumerate(roles) if r == "USER"]]
        m = e[[j for j, r in enumerate(roles) if r != "USER"]]
        gu = gram_fit(u, degree=degree) if len(u) else np.zeros((degree + 1, e.shape[1]))
        gm = gram_fit(m, degree=degree) if len(m) else np.zeros((degree + 1, e.shape[1]))
        rows.append(np.concatenate([gu.ravel(), gm.ravel()]))
    return np.asarray(rows)


def run(max_samples=None):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    from sklearn.decomposition import PCA
    results = {}
    for name in ["uss_sgd", "uss_mwoz"]:
        emb, offsets, meta = load_corpus(name)
        emb, offsets, meta = cap(emb, offsets, meta, max_samples)
        y = np.asarray(meta["overall_mean"], float)
        roles = meta["roles"]
        full, _ = build_rolesplit(emb, offsets, meta)

        entry = {}
        # reference: full-dim role-split at this protocol
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        p_full = np.zeros(len(y))
        for tr, te in kf.split(full):
            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                       random_state=SEED, n_jobs=-1)
            rf.fit(full[tr], y[tr])
            p_full[te] = rf.predict(full[te])
        entry["d384"] = {f"gap_{gap}": round(pairwise_accuracy(p_full, y, gap), 4)
                         for gap in GAPS}
        print(f"[{name}] d=384 (full): {entry['d384']}", flush=True)

        for dd in DIMS:
            ck = ckpt_load(f"{name}_d{dd}") if max_samples is None else None
            if ck is not None and "preds" in ck:
                preds = ck["preds"]
                print(f"[{name}] d={dd} loaded from checkpoint", flush=True)
                e = {f"gap_{gap}": round(pairwise_accuracy(preds, y, gap), 4)
                     for gap in GAPS}
                bd = bootstrap_diff(preds, p_full, y, 1.0, N_BOOT, SEED)
                e["vs_full_gap1.0"] = bd
                entry[f"d{dd}"] = e
                continue
            preds = np.zeros(len(y))
            for tr, te in kf.split(np.arange(len(y))):
                # fit PCA on train-fold TURN embeddings only
                tr_turn_idx = np.concatenate(
                    [np.arange(offsets[i], offsets[i + 1]) for i in tr])
                pca = PCA(n_components=dd, random_state=SEED)
                pca.fit(emb[tr_turn_idx])
                red = pca.transform(emb).astype(np.float64)
                X = rolesplit_from_turns(red, offsets, roles)
                rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                           random_state=SEED, n_jobs=-1)
                rf.fit(X[tr], y[tr])
                preds[te] = rf.predict(X[te])
            if max_samples is None:
                ckpt_save(f"{name}_d{dd}", {"preds": preds})
            e = {f"gap_{gap}": round(pairwise_accuracy(preds, y, gap), 4)
                 for gap in GAPS}
            bd = bootstrap_diff(preds, p_full, y, 1.0, N_BOOT, SEED)
            e["vs_full_gap1.0"] = bd
            entry[f"d{dd}"] = e
            print(f"[{name}] d={dd} (rolesplit {8*dd} dims): {e['gap_1.0']} "
                  f"vs full CI [{bd['ci_lo']:+.4f},{bd['ci_hi']:+.4f}] "
                  f"p={bd['p_boot']:.3f}", flush=True)
        results[name] = entry
    if max_samples is None:
        from campaign_io import save_result
        save_result("dim_frontier", results, dataset="uss_sgd.uss_mwoz",
                    config="minilm", params={"seed": SEED, "n_folds": N_FOLDS})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
