"""Token-level hierarchical encoding for issues #14/#15.

GPU pass: for every turn in USS-SGD, USS-MWOZ, PRISM, compute token-level
embeddings (MiniLM) and reduce IN THE LOOP to within-turn Gram coefficients
g0/g1/g2 plus residual-energy fractions - only the compact per-turn summary is
stored, never raw token embeddings.

Output per corpus: data/embeddings/{name}_hier.npz
    within_g:   (n_utts, 3, 384) float32 - within-turn g0, g1, g2
    resid_frac: (n_utts, 3) float32 - residual fraction above degrees 0,1,2
    n_tokens:   (n_utts,) int32
    offsets:    same dialogue offsets as the sentence-level npz (asserted)
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))

from lib.campaign import make_run_sample
from lib.device import get_device
from lib.encoders.gram import gram_fit, gram_basis_matrix_fast
from lib.uss import load_uss

EMB = WORK.parent / "data" / "embeddings"
DEGREE = 2


def within_turn_reduce(tok_emb):
    """(n_tokens, d) -> (g coeffs (3, d), residual fractions (3,), n_tokens)."""
    n, d = tok_emb.shape
    K = min(DEGREE, n - 1)
    coeffs = gram_fit(tok_emb, degree=DEGREE)  # zero-pads above K
    total = float(np.sum(tok_emb ** 2)) or 1.0
    fracs = []
    if n > 1:
        B = gram_basis_matrix_fast(n, K)
        for k in range(DEGREE + 1):
            kk = min(k, K)
            recon = B[:, :kk + 1] @ coeffs[:kk + 1]
            fracs.append(float(np.sum((tok_emb - recon) ** 2)) / total)
    else:
        fracs = [0.0, 0.0, 0.0]
    return coeffs.astype(np.float32), np.asarray(fracs, dtype=np.float32), n


def corpus_texts(name):
    if name == "prism":
        sys.path.insert(0, str(WORK / "scripts"))
        from embed_prism import load_prism_dialogues
        ds = load_prism_dialogues()
        return [d["texts"] for d in ds]
    seqs, _ = load_uss(name.split("_", 1)[1])
    return seqs


def run(max_samples=None):
    if max_samples is not None:
        raise ValueError(
            "embed_hierarchical builds token-level summaries from the raw "
            "corpus files and cannot cap in memory; use the sandbox driver "
            "instead: reproduce.py --max-samples N caps the raw layer "
            "(lib/corpus_sample.py).")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                device=get_device())

    summary = {}
    for name in ["uss_sgd", "uss_mwoz", "prism"]:
        ref = np.load(EMB / f"{name.replace('uss_', 'uss_')}_minilm.npz"
                      if name != "prism" else EMB / "prism_minilm.npz")
        seqs = corpus_texts(name)
        flat = [t for d in seqs for t in d]
        assert len(flat) == ref["embeddings"].shape[0], \
            f"{name}: {len(flat)} texts vs {ref['embeddings'].shape[0]} embeddings"

        out_g = np.zeros((len(flat), DEGREE + 1, 384), dtype=np.float32)
        out_f = np.zeros((len(flat), DEGREE + 1), dtype=np.float32)
        out_n = np.zeros(len(flat), dtype=np.int32)

        BATCH = 256
        for start in range(0, len(flat), BATCH):
            chunk = flat[start:start + BATCH]
            toks = model.encode(chunk, output_value="token_embeddings",
                                batch_size=64, show_progress_bar=False)
            for j, t in enumerate(toks):
                arr = t.detach().cpu().numpy().astype(np.float64)
                # strip [CLS]/[SEP] specials if present
                if arr.shape[0] > 2:
                    arr = arr[1:-1]
                g, f, n = within_turn_reduce(arr)
                out_g[start + j], out_f[start + j], out_n[start + j] = g, f, n
            if (start // BATCH) % 20 == 0:
                print(f"[{name}] {start}/{len(flat)}", flush=True)

        np.savez(EMB / f"{name}_hier.npz", within_g=out_g, resid_frac=out_f,
                 n_tokens=out_n, offsets=ref["offsets"])
        print(f"[{name}] saved {out_g.shape} -> {name}_hier.npz", flush=True)
        summary[name] = {"n_utterances": len(flat),
                         "shape": list(out_g.shape),
                         "npz": str(EMB / f"{name}_hier.npz")}
    return summary


run_sample = make_run_sample(run)


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
