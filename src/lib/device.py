"""Device selection for embedding models.

Central helper so every embedding path picks up the GPU when present
(RTX PRO 2000, 8GB) and falls back to CPU silently otherwise.

Override with EXPERIMENT_DEVICE=cpu|cuda when pinning is needed
(e.g. keeping the GPU free during parallel CPU encoding jobs).
"""

import os


def get_device() -> str:
    forced = os.environ.get("EXPERIMENT_DEVICE")
    if forced:
        return forced
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_sentence_transformer(model_name: str):
    """Load a SentenceTransformer on the best available device."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=get_device())
