"""Issue #26: role-split generalization - second task.

Task: dominant-service prediction on USS-SGD (multi-class), labels from the
1000/1000 text-match to original SGD (uss_sgd_original_map.json). Top-8
services + "other". This is a content-leaning task by construction - the
question is whether role-split still matches/beats the interleaved block,
i.e. whether the representation transfers beyond satisfaction.

PRISM generalization was already delivered by the #20 deconfound (role-split
significant in the short bin, trend in the long); this script completes #26.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK))

from lib.campaign import cap, make_run_sample
from trace_h2h import load_corpus, build_all_arms
from extension_arms import build_rolesplit

N_FOLDS, SEED, TOP = 5, 42, 8


def run(max_samples=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score

    emb, offsets, meta = load_corpus("uss_sgd")
    emb, offsets, meta = cap(emb, offsets, meta, max_samples)
    with open(WORK.parent / "data" / "embeddings" / "uss_sgd_original_map.json") as f:
        mapping = json.load(f)
    if max_samples is not None:
        mapping = mapping[:len(offsets) - 1]
    labels = []
    for m in mapping:
        svc = m["services"][0] if m["services"] else "other"
        labels.append(svc)
    top = {s for s, _ in Counter(labels).most_common(TOP)}
    y = np.asarray([l if l in top else "other" for l in labels])
    print("label distribution:", Counter(y), flush=True)

    base = build_all_arms(emb, offsets, meta, degree=3)
    raw = base["design_b_raw"]
    d = raw.shape[1] // 4
    role_raw, _ = build_rolesplit(emb, offsets, meta)
    arms = {"trace": base["trace"], "g0": raw[:, :d], "raw": raw,
            "rolesplit": role_raw}

    results = {"task": "dominant service, top-8 + other",
               "n": len(y), "n_classes": len(set(y))}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, X in arms.items():
        preds = np.empty(len(y), dtype=object)
        for tr, te in skf.split(X, y):
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                         random_state=SEED, n_jobs=-1)
            clf.fit(X[tr], y[tr])
            preds[te] = clf.predict(X[te])
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="macro")
        results[k] = {"acc": round(float(acc), 4), "macro_f1": round(float(f1), 4)}
        print(f"[service_transfer] {k:10s} acc={acc:.4f} macroF1={f1:.4f}", flush=True)

    if max_samples is None:
        from campaign_io import save_result
        save_result("service_transfer", results, dataset="uss_sgd",
                    config="minilm", params={"seed": SEED, "n_folds": N_FOLDS})
    return results


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
