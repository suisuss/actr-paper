"""Embed USS dialogue turns with MiniLM for the TRACE head-to-head spikes.

Saves per subset:
    data/embeddings/uss_{name}_minilm.npz   - embeddings (n_utts, 384) float32,
                                              offsets (n_dialogues+1,) int64
    data/embeddings/uss_{name}_meta.json    - roles, ratings, overall, n_turns
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lib.campaign import make_run_sample
from lib.uss import load_uss

OUT_DIR = Path(__file__).parents[1] / "data" / "embeddings"


def run(max_samples=None):
    if max_samples is not None:
        raise ValueError(
            "embed_uss builds embeddings from the raw USS files and cannot "
            "cap in memory; use the sandbox driver instead: reproduce.py "
            "--max-samples N caps the raw layer (lib/corpus_sample.py).")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from lib.device import load_sentence_transformer
    model = load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")

    summary = {}
    for name in ["sgd", "mwoz"]:
        seqs, meta = load_uss(name)
        flat = [t for d in seqs for t in d]
        offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
        for i, d in enumerate(seqs):
            offsets[i + 1] = offsets[i] + len(d)
        print(f"[{name}] encoding {len(flat)} utterances from {len(seqs)} dialogues...")
        emb = model.encode(flat, batch_size=128, show_progress_bar=True,
                           convert_to_numpy=True).astype(np.float32)
        np.savez(OUT_DIR / f"uss_{name}_minilm.npz", embeddings=emb, offsets=offsets)
        side = {k: meta[k] for k in ["roles", "actions", "turn_ratings", "overall",
                                     "overall_mean", "overall_median", "n_turns"]}
        with open(OUT_DIR / f"uss_{name}_meta.json", "w") as f:
            json.dump(side, f)
        print(f"[{name}] saved {emb.shape} -> uss_{name}_minilm.npz")
        summary[name] = {
            "n_dialogues": len(seqs), "n_utterances": len(flat),
            "npz": str(OUT_DIR / f"uss_{name}_minilm.npz"),
            "meta": str(OUT_DIR / f"uss_{name}_meta.json"),
        }
    return summary


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
