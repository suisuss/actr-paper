"""Embed PRISM conversations (user turns + chosen model responses) for Spike 6.

Trajectory = user turns and if_chosen model responses in dialogue order.
Label = performance_attributes.helpfulness (first-person, 0-100), rescaled to
1-5 so USS gap thresholds carry over. Goal = the first user turn, as on USS:
opening_prompt is byte-identical to it in all 8,011 raw conversations, so it is
a copy of the transcript's opening, not an independently elicited goal field.

Saves data/embeddings/prism_minilm.npz + prism_meta.json
(roles, helpfulness_1to5, user_ids, n_turns).
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
OUT = WORK.parent / "data" / "embeddings"

sys.path.insert(0, str(WORK))
from lib.campaign import make_run_sample


def load_prism_dialogues():
    dialogues = []
    with open(WORK.parent / "data" / "raw" / "prism" / "conversations.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            help_score = (rec.get("performance_attributes") or {}).get("helpfulness")
            if help_score is None:
                continue
            turns, roles = [], []
            for t in rec["conversation_history"]:
                if t["role"] == "user":
                    turns.append(t["content"]); roles.append("USER")
                elif t["role"] == "model" and t.get("if_chosen"):
                    turns.append(t["content"]); roles.append("SYSTEM")
            if len(turns) < 2:
                continue
            dialogues.append({
                "texts": turns, "roles": roles,
                "y": 1.0 + 4.0 * float(help_score) / 100.0,
                "user_id": rec["user_id"],
            })
    return dialogues


def run(max_samples=None):
    if max_samples is not None:
        raise ValueError(
            "embed_prism builds embeddings from the raw PRISM jsonl and "
            "cannot cap in memory; use the sandbox driver instead: "
            "reproduce.py --max-samples N caps the raw layer "
            "(lib/corpus_sample.py).")
    from lib.device import load_sentence_transformer
    model = load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
    ds = load_prism_dialogues()
    flat = [t for d in ds for t in d["texts"]]
    offsets = np.zeros(len(ds) + 1, dtype=np.int64)
    for i, d in enumerate(ds):
        offsets[i + 1] = offsets[i] + len(d["texts"])
    print(f"encoding {len(flat)} utterances from {len(ds)} conversations...")
    emb = model.encode(flat, batch_size=128, show_progress_bar=True,
                       convert_to_numpy=True).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "prism_minilm.npz", embeddings=emb, offsets=offsets)
    meta = {
        "roles": [d["roles"] for d in ds],
        "overall_mean": [d["y"] for d in ds],
        "user_ids": [d["user_id"] for d in ds],
        "n_turns": [len(d["texts"]) for d in ds],
    }
    with open(OUT / "prism_meta.json", "w") as f:
        json.dump(meta, f)
    print(f"saved {emb.shape} -> prism_minilm.npz")
    return {"n_dialogues": len(ds), "n_utterances": len(flat),
            "npz": str(OUT / "prism_minilm.npz"),
            "meta": str(OUT / "prism_meta.json")}


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
