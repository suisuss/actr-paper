"""Regenerate prism_meta.json with timing_duration_s (no re-embedding).

Must reproduce the exact dialogue filter used by embed_prism.py so the meta
stays aligned with the saved offsets; asserts the count matches.
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "scripts"))

from lib.campaign import make_run_sample
from embed_prism import load_prism_dialogues


def run(max_samples=None):
    if max_samples is not None:
        raise ValueError(
            "build_prism_meta re-walks the raw PRISM jsonl against the saved "
            "offsets and cannot cap in memory; use the sandbox driver "
            "instead: reproduce.py --max-samples N caps the raw layer "
            "(lib/corpus_sample.py).")
    npz = np.load(WORK.parent / "data" / "embeddings" / "prism_minilm.npz")
    offsets = npz["offsets"]

    # re-walk conversations.jsonl in the same order to attach durations
    durations = []
    with open(WORK.parent / "data" / "raw" / "prism" / "conversations.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            help_score = (rec.get("performance_attributes") or {}).get("helpfulness")
            if help_score is None:
                continue
            turns = [t for t in rec["conversation_history"]
                     if t["role"] == "user" or (t["role"] == "model" and t.get("if_chosen"))]
            if len(turns) < 2:
                continue
            durations.append(rec.get("timing_duration_s"))

    ds = load_prism_dialogues()
    assert len(ds) == len(offsets) - 1 == len(durations), \
        f"mismatch: {len(ds)} vs {len(offsets)-1} vs {len(durations)}"
    for i, d in enumerate(ds):
        assert offsets[i + 1] - offsets[i] == len(d["texts"]), f"offset mismatch at {i}"

    meta = {
        "roles": [d["roles"] for d in ds],
        "overall_mean": [d["y"] for d in ds],
        "user_ids": [d["user_id"] for d in ds],
        "n_turns": [len(d["texts"]) for d in ds],
        "duration_per_turn": [
            (dur / len(d["texts"])) if dur else 0.0
            for dur, d in zip(durations, ds)],
    }
    out = WORK.parent / "data" / "embeddings" / "prism_meta.json"
    with open(out, "w") as f:
        json.dump(meta, f)
    print(f"regenerated meta for {len(ds)} conversations with duration_per_turn")
    return {"n_dialogues": len(ds), "meta": str(out)}


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
