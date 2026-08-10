"""Issue #19: embed all three corpora with additional models (GPU).

Usage: embed_multimodel.py <model_tag>
    gte   -> thenlper/gte-small       (384-d)
    bge   -> BAAI/bge-base-en-v1.5    (768-d)

Saves data/embeddings/{corpus}_{tag}.npz with the same offsets as the minilm
reference (asserted), so all existing meta json files remain valid.
"""
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "scripts"))

from lib.campaign import make_run_sample

MODELS = {"gte": "thenlper/gte-small", "bge": "BAAI/bge-base-en-v1.5"}
EMB = WORK.parent / "data" / "embeddings"


def corpus_texts(name):
    from lib.uss import load_uss
    if name == "prism":
        from embed_prism import load_prism_dialogues
        return [d["texts"] for d in load_prism_dialogues()]
    seqs, _ = load_uss(name.split("_", 1)[1])
    return seqs


def run(tag=None, max_samples=None):
    if max_samples is not None:
        raise ValueError(
            "embed_multimodel builds embeddings from the raw corpus files "
            "and cannot cap in memory; use the sandbox driver instead: "
            "reproduce.py --max-samples N caps the raw layer "
            "(lib/corpus_sample.py).")
    from lib.device import load_sentence_transformer
    model = load_sentence_transformer(MODELS[tag])
    summary = {}
    for name in ["uss_sgd", "uss_mwoz", "prism"]:
        ref = np.load(EMB / (f"{name}_minilm.npz" if name != "prism"
                             else "prism_minilm.npz"))
        seqs = corpus_texts(name)
        flat = [t for d in seqs for t in d]
        assert len(flat) == ref["embeddings"].shape[0]
        print(f"[{tag}/{name}] encoding {len(flat)} utterances...", flush=True)
        emb = model.encode(flat, batch_size=128, show_progress_bar=False,
                           convert_to_numpy=True).astype(np.float32)
        np.savez(EMB / f"{name}_{tag}.npz", embeddings=emb,
                 offsets=ref["offsets"])
        print(f"[{tag}/{name}] saved {emb.shape}", flush=True)
        summary[name] = {"n_utterances": len(flat), "shape": list(emb.shape),
                         "npz": str(EMB / f"{name}_{tag}.npz")}
    return {"tag": tag, "corpora": summary}


run_sample = make_run_sample(run)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    run(tag=argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
