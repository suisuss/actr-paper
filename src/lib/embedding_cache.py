"""
Raw embedding cache for reusable sequence embeddings.

Stores per-sequence embeddings so we can compute different pooling
strategies without re-embedding.

Usage:
    from lib.embedding_cache import EmbeddingCache

    cache = EmbeddingCache()

    # Check if embeddings exist
    if cache.exists('sgd', 'minilm', 'pre_tool'):
        embeddings = cache.load('sgd', 'minilm', 'pre_tool')
    else:
        embeddings = compute_embeddings(sequences, model)
        cache.save('sgd', 'minilm', 'pre_tool', embeddings, metadata)

    # Now compute any pooling method from cached embeddings
    mean_pool = [emb.mean(axis=0) for emb in embeddings]
    max_pool = [emb.max(axis=0) for emb in embeddings]
    cls_pool = [emb[0] for emb in embeddings]
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import tempfile
import os

from .ioutils import atomic_savez


def get_cache_root() -> Path:
    """Get the embedding cache directory."""
    return Path(__file__).parent.parent.parent / "data" / "embeddings"


class EmbeddingCache:
    """
    Cache for raw sequence embeddings.

    Stores variable-length sequence embeddings using object arrays,
    allowing different pooling strategies to be computed later.
    """

    def __init__(self):
        self.root = get_cache_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def _get_path(self, dataset: str, model: str, variant: str) -> Path:
        """Get cache file path."""
        # Normalize model name
        model_short = model.split('/')[-1].lower().replace('-', '_')
        return self.root / dataset / model_short / f"{variant}.npz"

    def _get_checkpoint_path(self, dataset: str, model: str, variant: str) -> Path:
        """Get checkpoint path."""
        return self._get_path(dataset, model, variant).with_suffix('.checkpoint.npz')

    def exists(self, dataset: str, model: str, variant: str) -> bool:
        """Check if cache exists."""
        return self._get_path(dataset, model, variant).exists()

    def load(self, dataset: str, model: str, variant: str) -> Tuple[List[np.ndarray], Dict]:
        """
        Load cached embeddings.

        Returns:
            embeddings: List of arrays, each (n_turns, emb_dim)
            metadata: Dict with labels, etc.
        """
        path = self._get_path(dataset, model, variant)
        if not path.exists():
            raise FileNotFoundError(f"Cache not found: {path}")

        data = np.load(path, allow_pickle=True)

        # Load embeddings (stored as object array of arrays)
        embeddings = list(data['embeddings'])

        # Load metadata
        metadata = {}
        for k in data.keys():
            if k.startswith('meta_'):
                metadata[k[5:]] = data[k]
            elif k not in ['embeddings', 'model', 'dataset', 'variant', 'created']:
                metadata[k] = data[k]

        return embeddings, metadata

    def save(
        self,
        dataset: str,
        model: str,
        variant: str,
        embeddings: List[np.ndarray],
        metadata: Optional[Dict] = None
    ):
        """
        Save embeddings to cache.

        Args:
            embeddings: List of arrays, each (n_turns, emb_dim)
            metadata: Optional metadata dict
        """
        path = self._get_path(dataset, model, variant)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare save data
        save_data = {
            'embeddings': np.array(embeddings, dtype=object),
            'model': model,
            'dataset': dataset,
            'variant': variant,
            'created': datetime.now().isoformat(),
            'n_sequences': len(embeddings),
        }

        # Add metadata with prefix
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (list, np.ndarray)):
                    save_data[f'meta_{k}'] = np.array(v) if isinstance(v, list) else v
                else:
                    save_data[f'meta_{k}'] = v

        atomic_savez(path, save_data)
        print(f"Saved cache: {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")

    def encode_and_cache(
        self,
        dataset: str,
        model_name: str,
        variant: str,
        sequences: List[List[str]],
        metadata: Optional[Dict] = None,
        batch_size: int = 32,
        checkpoint_interval: int = 500,
        force: bool = False
    ) -> List[np.ndarray]:
        """
        Encode sequences and cache the raw embeddings.

        Args:
            sequences: List of text sequences (each is list of strings)
            metadata: Metadata to save with cache
            batch_size: Batch size for encoding
            checkpoint_interval: Save checkpoint every N sequences
            force: Overwrite existing cache

        Returns:
            List of embedding arrays
        """
        from sentence_transformers import SentenceTransformer

        if self.exists(dataset, model_name, variant) and not force:
            print(f"Cache exists. Loading...")
            embeddings, _ = self.load(dataset, model_name, variant)
            return embeddings

        from .device import get_device
        print(f"Encoding {len(sequences)} sequences with {model_name} on {get_device()}...")
        model = SentenceTransformer(model_name, device=get_device())

        checkpoint_path = self._get_checkpoint_path(dataset, model_name, variant)

        # Check for checkpoint
        start_idx = 0
        embeddings = []

        if checkpoint_path.exists():
            checkpoint = np.load(checkpoint_path, allow_pickle=True)
            start_idx = int(checkpoint['processed_count'])
            embeddings = list(checkpoint['embeddings'])
            print(f"Resuming from checkpoint: {start_idx}/{len(sequences)}")

        # Process sequences
        for i in tqdm(range(start_idx, len(sequences)), initial=start_idx, total=len(sequences)):
            seq = sequences[i]

            if not seq:
                # Empty sequence
                emb_dim = model.get_sentence_embedding_dimension()
                embeddings.append(np.zeros((1, emb_dim)))
                continue

            # Encode all turns in sequence
            emb = model.encode(seq, batch_size=batch_size, show_progress_bar=False)
            embeddings.append(np.array(emb))

            # Checkpoint
            if (i + 1) % checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_path, i + 1, embeddings, len(sequences))

        # Save final cache
        self.save(dataset, model_name, variant, embeddings, metadata)

        # Cleanup checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        return embeddings

    def _save_checkpoint(self, path: Path, processed_count: int, embeddings: List, total: int):
        """Save checkpoint atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_fd, temp_path = tempfile.mkstemp(suffix='.npz', dir=path.parent)
        os.close(temp_fd)

        np.savez(
            temp_path,
            processed_count=processed_count,
            total_count=total,
            embeddings=np.array(embeddings, dtype=object),
            last_updated=datetime.now().isoformat()
        )
        os.replace(temp_path, path)
        print(f"Checkpoint: {processed_count}/{total}")

    def list_cached(self) -> List[Dict[str, str]]:
        """List all cached embeddings."""
        cached = []
        for npz_path in self.root.rglob("*.npz"):
            if ".checkpoint" in npz_path.name:
                continue

            rel_path = npz_path.relative_to(self.root)
            parts = rel_path.parts
            if len(parts) == 3:
                cached.append({
                    'dataset': parts[0],
                    'model': parts[1],
                    'variant': parts[2].replace('.npz', ''),
                    'path': str(npz_path),
                    'size_mb': npz_path.stat().st_size / 1024 / 1024
                })

        return cached

    def print_status(self):
        """Print cache status."""
        print("\n" + "=" * 60)
        print("EMBEDDING CACHE STATUS")
        print("=" * 60)
        print(f"Cache root: {self.root}")

        cached = self.list_cached()
        if not cached:
            print("\n  (no cached embeddings)")
            return

        print(f"\nCached: {len(cached)} embeddings")
        for c in sorted(cached, key=lambda x: (x['dataset'], x['model'], x['variant'])):
            print(f"  {c['dataset']}/{c['model']}/{c['variant']} ({c['size_mb']:.1f} MB)")


# Convenience functions

def get_or_create_embeddings(
    dataset: str,
    model: str,
    variant: str,
    sequences: Optional[List[List[str]]] = None,
    metadata: Optional[Dict] = None,
    force: bool = False
) -> Tuple[List[np.ndarray], Dict]:
    """
    Get cached embeddings or create them.

    Args:
        dataset: Dataset name
        model: Model name (e.g., 'all-MiniLM-L6-v2')
        variant: Variant name (e.g., 'pre_tool')
        sequences: Sequences to encode if cache doesn't exist
        metadata: Metadata to save with cache
        force: Force re-encoding

    Returns:
        (embeddings, metadata)
    """
    cache = EmbeddingCache()

    if cache.exists(dataset, model, variant) and not force:
        return cache.load(dataset, model, variant)

    if sequences is None:
        raise ValueError(f"Cache not found and no sequences provided: {dataset}/{model}/{variant}")

    embeddings = cache.encode_and_cache(
        dataset, model, variant, sequences, metadata, force=force
    )

    # Reload to get properly formatted metadata
    return cache.load(dataset, model, variant)


def compute_poolings(embeddings: List[np.ndarray], emb_dim: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Compute all pooling methods from raw embeddings.

    Args:
        embeddings: List of (n_turns, emb_dim) arrays
        emb_dim: Embedding dimension (inferred from first non-empty embedding if not provided)

    Returns:
        Dict with all pooling results
    """
    from .encoder import to_angles, to_magnitudes

    # Determine embedding dimension from first non-empty embedding if not provided
    if emb_dim is None:
        for emb in embeddings:
            if emb.ndim >= 2 and emb.shape[1] > 0:
                emb_dim = emb.shape[1]
                break
        if emb_dim is None:
            raise ValueError("Cannot infer embedding dimension: all embeddings are empty and emb_dim not provided")

    results = {
        'mean': [],
        'cls': [],
        'max': [],
        'weighted': [],
        'c0_ang': [],
        'c0_mag': [],
    }

    for emb in embeddings:
        if emb.shape[0] == 0:
            results['mean'].append(np.zeros(emb_dim))
            results['cls'].append(np.zeros(emb_dim))
            results['max'].append(np.zeros(emb_dim))
            results['weighted'].append(np.zeros(emb_dim))
            results['c0_ang'].append(np.zeros(emb_dim // 2))
            results['c0_mag'].append(np.zeros(emb_dim // 2))
            continue

        # Standard poolings
        results['mean'].append(np.mean(emb, axis=0))
        results['cls'].append(emb[0])
        results['max'].append(np.max(emb, axis=0))

        # Weighted (by L2 norm)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        weights = norms / (norms.sum() + 1e-8)
        results['weighted'].append((emb * weights).sum(axis=0))

        # Angular
        angles = to_angles(emb)
        results['c0_ang'].append(np.mean(angles, axis=0))

        # Magnitude
        mags = to_magnitudes(emb)
        results['c0_mag'].append(np.mean(mags, axis=0))

    return {k: np.array(v) for k, v in results.items()}


# =============================================================================
# Token-Level Embedding Cache
# =============================================================================

def get_token_cache_root() -> Path:
    """Get the token embedding cache directory."""
    return Path(__file__).parent.parent.parent / "data" / "embeddings" / "tokens"


class TokenEmbeddingCache:
    """
    Cache for token-level embeddings (per-token, not per-sentence).

    Unlike EmbeddingCache which stores sentence embeddings for sequences,
    this stores the raw token embeddings WITHIN each sentence, enabling
    different encoding methods (Chebyshev, DCT, pooling) to be computed later.

    REUSE FEATURE: When encoding N sentences, automatically reuses embeddings
    from smaller cached files. E.g., if pairs_n500 exists and you request
    pairs_n2000, it loads the first 1000 sentences and only extracts the
    remaining 3000.

    Usage:
        cache = TokenEmbeddingCache()

        # Check if embeddings exist
        if cache.exists('paws', 'minilm', 'pairs_n500'):
            embeddings, metadata = cache.load('paws', 'minilm', 'pairs_n500')
        else:
            # embeddings is List[np.ndarray], each (n_tokens, emb_dim)
            embeddings = [get_token_embeddings(text, model) for text in texts]
            cache.save('paws', 'minilm', 'pairs_n500', embeddings, metadata)

        # Now compute any encoding from cached token embeddings
        from lib.encoders.pooling import mean_pool
        from lib.encoders.dct import dct_encode_decomposed
        mean_pooled = [mean_pool(emb) for emb in embeddings]
        dct_encoded = [dct_encode_decomposed(emb) for emb in embeddings]
    """

    def __init__(self):
        self.root = get_token_cache_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def _get_path(self, dataset: str, model: str, variant: str) -> Path:
        """Get cache file path."""
        model_short = model.split('/')[-1].lower().replace('-', '_')
        return self.root / dataset / model_short / f"{variant}.npz"

    def _get_checkpoint_path(self, dataset: str, model: str, variant: str) -> Path:
        """Get checkpoint path."""
        return self._get_path(dataset, model, variant).with_suffix('.checkpoint.npz')

    def _find_reusable_cache(self, dataset: str, model: str, variant: str, n_needed: int) -> Optional[Tuple[str, int]]:
        """
        Find a smaller cache that can be reused.

        For variants like 'pairs_n500', 'pairs_n2000', finds the largest
        existing cache with fewer sentences that can be reused.

        Args:
            dataset: Dataset name
            model: Model name
            variant: Target variant name (e.g., 'pairs_n2000')
            n_needed: Number of sentences needed

        Returns:
            (variant_name, n_sentences) of best reusable cache, or None
        """
        import re

        model_short = model.split('/')[-1].lower().replace('-', '_')
        cache_dir = self.root / dataset / model_short

        if not cache_dir.exists():
            return None

        # Extract the base pattern (e.g., 'pairs_n' from 'pairs_n2000')
        match = re.match(r'^(.+_n)(\d+)$', variant)
        if not match:
            return None

        base_pattern = match.group(1)  # e.g., 'pairs_n'

        # Find all matching caches
        best_cache = None
        best_count = 0

        for npz_path in cache_dir.glob(f"{base_pattern}*.npz"):
            if ".checkpoint" in npz_path.name:
                continue

            cache_variant = npz_path.stem
            cache_match = re.match(r'^(.+_n)(\d+)$', cache_variant)
            if not cache_match or cache_match.group(1) != base_pattern:
                continue

            cache_n = int(cache_match.group(2))

            # Check if this cache has fewer sentences than we need
            # For pairs: pairs_n500 means 500 pairs = 1000 sentences
            # We need to load and check the actual count
            try:
                data = np.load(npz_path, allow_pickle=True)
                n_cached = int(data.get('n_sentences', len(data['embeddings'])))

                if n_cached < n_needed and n_cached > best_count:
                    best_count = n_cached
                    best_cache = cache_variant
            except Exception:
                continue

        if best_cache:
            return (best_cache, best_count)
        return None

    def exists(self, dataset: str, model: str, variant: str) -> bool:
        """Check if cache exists."""
        return self._get_path(dataset, model, variant).exists()

    def load(self, dataset: str, model: str, variant: str) -> Tuple[List[np.ndarray], Dict]:
        """
        Load cached token embeddings.

        Returns:
            embeddings: List of arrays, each (n_tokens, emb_dim) per sentence
            metadata: Dict with labels, etc.
        """
        path = self._get_path(dataset, model, variant)
        if not path.exists():
            raise FileNotFoundError(f"Token cache not found: {path}")

        data = np.load(path, allow_pickle=True)

        # Load embeddings (stored as object array of variable-length arrays)
        embeddings = list(data['embeddings'])

        # Load metadata
        metadata = {}
        for k in data.keys():
            if k.startswith('meta_'):
                metadata[k[5:]] = data[k]
            elif k not in ['embeddings', 'model', 'dataset', 'variant', 'created', 'n_sentences']:
                metadata[k] = data[k]

        return embeddings, metadata

    def save(
        self,
        dataset: str,
        model: str,
        variant: str,
        embeddings: List[np.ndarray],
        metadata: Optional[Dict] = None
    ):
        """
        Save token embeddings to cache.

        Args:
            embeddings: List of arrays, each (n_tokens, emb_dim) per sentence
            metadata: Optional metadata dict (labels, etc.)
        """
        path = self._get_path(dataset, model, variant)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare save data
        save_data = {
            'embeddings': np.array(embeddings, dtype=object),
            'model': model,
            'dataset': dataset,
            'variant': variant,
            'created': datetime.now().isoformat(),
            'n_sentences': len(embeddings),
        }

        # Add metadata with prefix
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (list, np.ndarray)):
                    save_data[f'meta_{k}'] = np.array(v) if isinstance(v, list) else v
                else:
                    save_data[f'meta_{k}'] = v

        atomic_savez(path, save_data)
        print(f"Saved token cache: {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")

    def encode_and_cache(
        self,
        dataset: str,
        model_name: str,
        variant: str,
        texts: List[str],
        metadata: Optional[Dict] = None,
        checkpoint_interval: int = 100,
        force: bool = False
    ) -> List[np.ndarray]:
        """
        Extract token embeddings and cache them.

        REUSE: Automatically reuses embeddings from smaller caches.
        E.g., if pairs_n500 exists and you encode pairs_n2000, it loads
        the first 1000 sentences and only extracts the remaining 3000.

        Args:
            texts: List of text strings (one per sentence)
            metadata: Metadata to save with cache
            checkpoint_interval: Save checkpoint every N sentences
            force: Overwrite existing cache

        Returns:
            List of token embedding arrays
        """
        from sentence_transformers import SentenceTransformer
        import torch

        if self.exists(dataset, model_name, variant) and not force:
            print(f"Token cache exists. Loading...")
            embeddings, _ = self.load(dataset, model_name, variant)
            return embeddings

        n_needed = len(texts)
        checkpoint_path = self._get_checkpoint_path(dataset, model_name, variant)

        # Check for checkpoint first
        start_idx = 0
        embeddings = []

        if checkpoint_path.exists() and not force:
            checkpoint = np.load(checkpoint_path, allow_pickle=True)
            start_idx = int(checkpoint['processed_count'])
            embeddings = list(checkpoint['embeddings'])
            print(f"Resuming from checkpoint: {start_idx}/{n_needed}")

        # If no checkpoint, try to reuse a smaller cache
        elif not force:
            reusable = self._find_reusable_cache(dataset, model_name, variant, n_needed)
            if reusable:
                reuse_variant, n_cached = reusable
                print(f"Reusing {n_cached} embeddings from {reuse_variant}...")
                cached_embeddings, _ = self.load(dataset, model_name, reuse_variant)
                embeddings = list(cached_embeddings)
                start_idx = n_cached
                print(f"Loaded {n_cached}/{n_needed} sentences from cache, extracting {n_needed - n_cached} more...")

        if start_idx == 0:
            print(f"Extracting token embeddings for {n_needed} sentences with {model_name}...")

        from .device import get_device
        model = SentenceTransformer(model_name, device=get_device())

        # Process sentences
        for i in tqdm(range(start_idx, len(texts)), initial=start_idx, total=len(texts),
                      desc="Extracting tokens"):
            text = texts[i]

            # Get token embeddings (skip [CLS] and [SEP])
            encoded = model.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=False
            )

            with torch.no_grad():
                output = model[0].auto_model(**encoded)
                token_emb = output.last_hidden_state[0].cpu().numpy()

            # Skip [CLS] and [SEP]
            embeddings.append(token_emb[1:-1])

            # Checkpoint
            if (i + 1) % checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_path, i + 1, embeddings, len(texts))

        # Save final cache
        self.save(dataset, model_name, variant, embeddings, metadata)

        # Cleanup checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        return embeddings

    def _save_checkpoint(self, path: Path, processed_count: int, embeddings: List, total: int):
        """Save checkpoint atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_fd, temp_path = tempfile.mkstemp(suffix='.npz', dir=path.parent)
        os.close(temp_fd)

        np.savez(
            temp_path,
            processed_count=processed_count,
            total_count=total,
            embeddings=np.array(embeddings, dtype=object),
            last_updated=datetime.now().isoformat()
        )
        os.replace(temp_path, path)
        print(f"Checkpoint: {processed_count}/{total}")

    def list_cached(self) -> List[Dict[str, str]]:
        """List all cached token embeddings."""
        cached = []
        for npz_path in self.root.rglob("*.npz"):
            if ".checkpoint" in npz_path.name:
                continue

            rel_path = npz_path.relative_to(self.root)
            parts = rel_path.parts
            if len(parts) == 3:
                cached.append({
                    'dataset': parts[0],
                    'model': parts[1],
                    'variant': parts[2].replace('.npz', ''),
                    'path': str(npz_path),
                    'size_mb': npz_path.stat().st_size / 1024 / 1024
                })

        return cached

    def print_status(self):
        """Print cache status."""
        print("\n" + "=" * 60)
        print("TOKEN EMBEDDING CACHE STATUS")
        print("=" * 60)
        print(f"Cache root: {self.root}")

        cached = self.list_cached()
        if not cached:
            print("\n  (no cached token embeddings)")
            return

        print(f"\nCached: {len(cached)} token embeddings")
        for c in sorted(cached, key=lambda x: (x['dataset'], x['model'], x['variant'])):
            print(f"  {c['dataset']}/{c['model']}/{c['variant']} ({c['size_mb']:.1f} MB)")


# Convenience function for token embeddings

def get_or_create_token_embeddings(
    dataset: str,
    model: str,
    variant: str,
    texts: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
    force: bool = False
) -> Tuple[List[np.ndarray], Dict]:
    """
    Get cached token embeddings or create them.

    Args:
        dataset: Dataset name (e.g., 'paws')
        model: Model name (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        variant: Variant name (e.g., 'pairs_n500')
        texts: Texts to encode if cache doesn't exist
        metadata: Metadata to save with cache
        force: Force re-encoding

    Returns:
        (embeddings, metadata) where embeddings is List[(n_tokens, emb_dim)]
    """
    cache = TokenEmbeddingCache()

    if cache.exists(dataset, model, variant) and not force:
        return cache.load(dataset, model, variant)

    if texts is None:
        raise ValueError(f"Token cache not found and no texts provided: {dataset}/{model}/{variant}")

    embeddings = cache.encode_and_cache(
        dataset, model, variant, texts, metadata, force=force
    )

    # Reload to get properly formatted metadata
    return cache.load(dataset, model, variant)
