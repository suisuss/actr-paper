"""
Core encoder for polynomial/frequency decomposition of embeddings.

Provides the shared encoding logic for all datasets. Supports multiple bases:
- Chebyshev polynomials (c): Traditional polynomial fitting
- Fourier magnitude (f): Phase-invariant frequency decomposition
- DCT (d): Discrete Cosine Transform
- DST (s): Discrete Sine Transform
- Gram polynomials (g): Orthogonal at uniform points

Component Naming:
    Format: {basis}{degree}{type}
    - basis: c/f/d/s/g
    - degree: 0, 1, 2, ...
    - type: r/a/m (raw/angular/magnitude), optional for f/d/s/g (defaults to raw)

    Examples:
        c0r, c0a, c0m, c1a, c2a  (Chebyshev)
        f0, f1, f2, f1a, f1m     (Fourier magnitude)
        d0, d1, d2               (DCT)
        s0, s1, s2               (DST)
        g0, g1, g2               (Gram)
"""

import warnings
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np
from pathlib import Path

from .config import (
    EncodingConfig, load_config, ENCODING_METHODS,
    get_method_components, COMPONENT_CODES, COMPONENT_NAMES,
    ProcessingConfig
)

from .encoders.utils import to_angles, to_magnitudes, apply_rope, apply_random_rotation

warnings.filterwarnings('ignore', message='.*poorly conditioned.*')


class ChebyshevEncoder:
    """
    Chebyshev polynomial encoder for token sequences.

    Encodes a sequence of token embeddings into decomposed Chebyshev coefficients:
    - c0: Semantic component (approximately equal to mean pooling)
    - c1_plus: Positional components (captures word order, trajectory)
    """

    def __init__(self, config: Union[EncodingConfig, str]):
        """
        Initialize encoder.

        Args:
            config: EncodingConfig object or config name string
        """
        if isinstance(config, str):
            config = load_config(config)

        self.config = config
        self.degree = config.chebyshev.degree
        self.embedding_dim = config.model.dim
        self.model_name = config.model.name
        self.normalize = config.model.normalize

        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        """Lazy load embedding model."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """Load the sentence transformer model."""
        from sentence_transformers import SentenceTransformer
        from .device import get_device
        self._model = SentenceTransformer(self.model_name, device=get_device())

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            Array of shape (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize
        )
        return np.array(embeddings)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed_texts([text])[0]

    # DEPRECATED: Use Gram polynomials instead - they are ~1000x faster and truly orthogonal
    # at uniform sample points. Chebyshev fitting via chebfit is slow and Chebyshev polynomials
    # are NOT orthogonal at uniform sample points (only at Chebyshev nodes).
    #
    # def fit_chebyshev(self, embeddings: np.ndarray) -> np.ndarray:
    #     """
    #     Fit Chebyshev polynomials to a sequence of embeddings.
    #
    #     Args:
    #         embeddings: Array of shape (seq_len, embedding_dim)
    #
    #     Returns:
    #         Coefficients of shape (degree+1, embedding_dim)
    #     """
    #     seq_len = len(embeddings)
    #     if seq_len == 0:
    #         return np.zeros((self.degree + 1, self.embedding_dim))
    #
    #     if seq_len == 1:
    #         # Single item: c0 = embedding, rest = 0
    #         coeffs = np.zeros((self.degree + 1, self.embedding_dim))
    #         coeffs[0] = embeddings[0]
    #         return coeffs
    #
    #     # Map positions to [-1, 1]
    #     positions = np.linspace(-1, 1, seq_len)
    #
    #     # Fit each dimension independently
    #     coeffs = np.zeros((self.degree + 1, self.embedding_dim))
    #     for d in range(self.embedding_dim):
    #         coeffs[:, d] = C.chebfit(positions, embeddings[:, d], self.degree)
    #
    #     return coeffs

    def preprocess(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing transformations before Chebyshev fitting.

        Applies RoPE rotation or random rotation based on config.processing settings.

        Args:
            embeddings: Array of shape (seq_len, embedding_dim)

        Returns:
            Preprocessed embeddings with same shape
        """
        processing = self.config.processing
        if processing.rope:
            return apply_rope(embeddings, base=processing.rope_base)
        elif processing.random_rotation:
            return apply_random_rotation(embeddings, seed=processing.random_seed)
        return embeddings

    def decompose(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decompose embeddings based on configured method.

        Supports multiple bases:
        - Chebyshev (c): Polynomial fitting in [-1, 1]
        - Fourier magnitude (f): Phase-invariant √(d_k² + s_k²)
        - DCT (d): Discrete Cosine Transform
        - DST (s): Discrete Sine Transform
        - Gram (g): Polynomials orthogonal at uniform points

        If config.processing specifies RoPE or random rotation, applies that
        transformation before fitting.

        Args:
            embeddings: Array of shape (seq_len, embedding_dim)

        Returns:
            Dict with ONLY the individual components specified by the method.
            Component names: c0r, c0a, f0, f1, d0, d1, g0, g1, etc.

            NO combined/grouped arrays - combining happens at experiment time.
        """
        method = self.config.chebyshev.method
        degree = self.config.effective_degree

        result = {}

        # Get the component list for this method
        try:
            components = get_method_components(method, degree)
        except ValueError:
            # Fallback for unknown methods
            components = ['c0r']

        # Apply preprocessing (RoPE, random rotation, etc.) before fitting
        processed_embeddings = self.preprocess(embeddings)

        # Group components by basis type
        cheb_comps = [c for c in components if c.startswith('c')]
        fourier_comps = [c for c in components if c.startswith('f')]
        dct_comps = [c for c in components if c.startswith('d')]
        dst_comps = [c for c in components if c.startswith('s')]
        gram_comps = [c for c in components if c.startswith('g')]
        circular_comps = [c for c in components if c.startswith('h')]

        # Process Chebyshev components (c0r, c0a, c0m, c1a, etc.)
        # DEPRECATED: Chebyshev fitting is slow. Use Gram polynomials (g0, g1, etc.) instead.
        if cheb_comps:
            raise NotImplementedError(
                "Chebyshev components (c0r, c0a, etc.) are deprecated due to slow fitting. "
                "Use Gram polynomials (g0, g1, g0a, g1a, etc.) instead - they are ~1000x faster "
                "and truly orthogonal at uniform sample points."
            )

        # Process Fourier magnitude components (f0, f1, f2, f0a, f1a, etc.)
        if fourier_comps:
            fourier_result = self._decompose_fourier(processed_embeddings, fourier_comps, degree)
            result.update(fourier_result)

        # Process DCT components (d0, d1, d2, d0a, d1a, etc.)
        if dct_comps:
            dct_result = self._decompose_dct(processed_embeddings, dct_comps, degree)
            result.update(dct_result)

        # Process DST components (s1, s2, s3, s1a, s2a, etc.; 1-indexed, no s0)
        if dst_comps:
            dst_result = self._decompose_dst(processed_embeddings, dst_comps, degree)
            result.update(dst_result)

        # Process Gram polynomial components (g0, g1, g2, g0a, g1a, etc.)
        if gram_comps:
            gram_result = self._decompose_gram(processed_embeddings, gram_comps, degree)
            result.update(gram_result)

        # Process Circular Fourier components (h0, h1, h2, etc.)
        if circular_comps:
            circular_result = self._decompose_circular(processed_embeddings, circular_comps, degree)
            result.update(circular_result)

        # Store the component list for reference (useful for combining later)
        result['_components'] = components

        return result

    # DEPRECATED: Chebyshev decomposition is slow (~1000x slower than Gram).
    # Use Gram polynomials instead - they are orthogonal at uniform sample points.
    #
    # def _decompose_chebyshev(
    #     self,
    #     embeddings: np.ndarray,
    #     components: List[str],
    #     degree: int
    # ) -> Dict[str, np.ndarray]:
    #     """Decompose using Chebyshev polynomials."""
    #     result = {}
    #
    #     # Determine which coefficient types we need
    #     needs_raw = any(c.endswith('r') for c in components)
    #     needs_ang = any(c.endswith('a') for c in components)
    #     needs_mag = any(c.endswith('m') for c in components)
    #
    #     # Compute raw Chebyshev coefficients
    #     coeffs_raw = None
    #     if needs_raw:
    #         coeffs_raw = self.fit_chebyshev(embeddings)
    #
    #     # Compute angular coefficients if needed
    #     coeffs_ang = None
    #     if needs_ang:
    #         angles = to_angles(embeddings)
    #         coeffs_ang = self._fit_chebyshev_data(angles, degree)
    #
    #     # Compute magnitude coefficients if needed
    #     coeffs_mag = None
    #     if needs_mag:
    #         magnitudes = to_magnitudes(embeddings)
    #         coeffs_mag = self._fit_chebyshev_data(magnitudes, degree)
    #
    #     # Store requested components
    #     for comp in components:
    #         if comp.endswith('r') and coeffs_raw is not None:
    #             idx = int(comp[1:-1])  # c0r -> 0, c1r -> 1
    #             if idx <= degree:
    #                 result[comp] = coeffs_raw[idx]
    #         elif comp.endswith('a') and coeffs_ang is not None:
    #             idx = int(comp[1:-1])
    #             if idx <= degree:
    #                 result[comp] = coeffs_ang[idx]
    #         elif comp.endswith('m') and coeffs_mag is not None:
    #             idx = int(comp[1:-1])
    #             if idx <= degree:
    #                 result[comp] = coeffs_mag[idx]
    #
    #     return result

    def _decompose_fourier(
        self,
        embeddings: np.ndarray,
        components: List[str],
        degree: int
    ) -> Dict[str, np.ndarray]:
        """
        Decompose using Fourier magnitude (phase-invariant).

        f_k = √(d_k² + s_k²) where d_k is DCT and s_k is DST.
        """
        from .encoders.fourier import fourier_magnitude_encode_decomposed

        result = {}

        # Compute Fourier magnitude decomposition
        n_coefficients = degree + 1
        fourier = fourier_magnitude_encode_decomposed(
            embeddings,
            n_coefficients=n_coefficients,
            include_angular=True,
            include_magnitude=True
        )

        # Map requested components
        for comp in components:
            # Parse component: f0, f1, f0a, f1m, etc.
            if comp[-1] in ('a', 'm'):
                idx = int(comp[1:-1])
                suffix = comp[-1]
                key = f'f{idx}{suffix}'
            else:
                idx = int(comp[1:])
                key = f'f{idx}'

            if key in fourier:
                result[comp] = fourier[key]

        return result

    def _decompose_dct(
        self,
        embeddings: np.ndarray,
        components: List[str],
        degree: int
    ) -> Dict[str, np.ndarray]:
        """Decompose using Discrete Cosine Transform."""
        from .encoders.dct import dct_encode_decomposed

        result = {}

        # Compute DCT decomposition
        n_coefficients = degree + 1
        dct = dct_encode_decomposed(
            embeddings,
            n_coefficients=n_coefficients,
            include_angular=True,
            include_magnitude=True
        )

        # Map requested components
        for comp in components:
            # Parse component: d0, d1, d0a, d1m, etc.
            if comp[-1] in ('a', 'm'):
                idx = int(comp[1:-1])
                suffix = comp[-1]
                key = f'd{idx}{suffix}'
            else:
                idx = int(comp[1:])
                key = f'd{idx}'

            if key in dct:
                result[comp] = dct[key]

        return result

    def _decompose_dst(
        self,
        embeddings: np.ndarray,
        components: List[str],
        degree: int
    ) -> Dict[str, np.ndarray]:
        """Decompose using Discrete Sine Transform."""
        from .encoders.dst import dst_encode_decomposed

        result = {}

        # Compute DST decomposition
        n_coefficients = degree + 1
        dst = dst_encode_decomposed(
            embeddings,
            n_coefficients=n_coefficients,
            include_angular=True,
            include_magnitude=True
        )

        # Map requested components
        for comp in components:
            # Parse component: s1, s2, s1a, s2m, etc. (1-indexed; no s0)
            if comp[-1] in ('a', 'm'):
                idx = int(comp[1:-1])
                suffix = comp[-1]
                key = f's{idx}{suffix}'
            else:
                idx = int(comp[1:])
                key = f's{idx}'

            if key in dst:
                result[comp] = dst[key]

        return result

    def _decompose_gram(
        self,
        embeddings: np.ndarray,
        components: List[str],
        degree: int
    ) -> Dict[str, np.ndarray]:
        """Decompose using Gram polynomials (orthogonal at uniform points)."""
        from .encoders.gram import gram_encode_decomposed

        result = {}

        # Compute Gram decomposition
        n_coefficients = degree + 1
        gram = gram_encode_decomposed(
            embeddings,
            n_coefficients=n_coefficients,
            include_angular=True,
            include_magnitude=True
        )

        # Map requested components
        for comp in components:
            # Parse component: g0, g1, g0a, g1m, etc.
            if comp[-1] in ('a', 'm'):
                idx = int(comp[1:-1])
                suffix = comp[-1]
                key = f'g{idx}{suffix}'
            else:
                idx = int(comp[1:])
                key = f'g{idx}'

            if key in gram:
                result[comp] = gram[key]

        return result

    def _decompose_circular(
        self,
        embeddings: np.ndarray,
        components: List[str],
        degree: int
    ) -> Dict[str, np.ndarray]:
        """Decompose using circular Fourier polar encoding.

        Each h_k = concat(angle_cos[k], angle_sin[k], magnitude[k]).
        Dimension per harmonic: 3 * (embedding_dim / 2).
        """
        from .encoders.circular_fourier import circular_polar_encode

        n_coefficients = degree + 1
        enc = circular_polar_encode(embeddings, n_coefficients=n_coefficients)
        actual_K = enc['angle_cos'].shape[0]  # may be < n_coefficients for short sequences
        n_pairs = enc['angle_cos'].shape[1]

        result = {}
        for comp in components:
            idx = int(comp[1:])  # h0 -> 0, h1 -> 1
            if idx < actual_K:
                result[comp] = np.concatenate([
                    enc['angle_cos'][idx],
                    enc['angle_sin'][idx],
                    enc['magnitude'][idx],
                ])
            else:
                # Short sequence: zero-pad missing harmonics
                result[comp] = np.zeros(3 * n_pairs)

        return result

    def combine(self, decomposed: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine individual components into a single feature vector.

        Use this at experiment time to create the combined representation.

        Args:
            decomposed: Dict from decompose() with individual components

        Returns:
            Concatenated feature vector
        """
        components = decomposed.get('_components', [])
        if not components:
            # Try to infer from keys - look for any valid component pattern
            import re
            components = sorted([
                k for k in decomposed.keys()
                if re.match(r'^[cfdsgh]\d+[ram]?$', k)
            ])

        parts = [decomposed[c] for c in components if c in decomposed]
        return np.concatenate(parts) if parts else np.array([])

    # DEPRECATED: Chebyshev fitting is slow. Use Gram polynomials instead.
    #
    # def _fit_chebyshev_data(self, data: np.ndarray, degree: int) -> np.ndarray:
    #     """Fit Chebyshev to arbitrary data (raw embeddings or angles)."""
    #     seq_len, n_features = data.shape
    #
    #     if seq_len == 0:
    #         return np.zeros((degree + 1, n_features))
    #
    #     if seq_len == 1:
    #         coeffs = np.zeros((degree + 1, n_features))
    #         coeffs[0] = data[0]
    #         return coeffs
    #
    #     # Map positions to [-1, 1]
    #     positions = np.linspace(-1, 1, seq_len)
    #
    #     # Fit each dimension independently
    #     actual_degree = min(degree, seq_len - 1)
    #     coeffs = np.zeros((degree + 1, n_features))
    #
    #     for d in range(n_features):
    #         fit = C.chebfit(positions, data[:, d], actual_degree)
    #         coeffs[:len(fit), d] = fit
    #
    #     return coeffs

    def encode_sequence(
        self,
        texts: List[str],
        return_embeddings: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Encode a sequence of texts into decomposed components.

        Args:
            texts: List of text strings (e.g., conversation turns)
            return_embeddings: Whether to also return raw embeddings

        Returns:
            Dict with individual components (c0r, c0a, f0, f1, etc.) and metadata
        """
        # Get component list for empty result
        method = self.config.chebyshev.method
        degree = self.config.effective_degree
        try:
            components = get_method_components(method, degree)
        except ValueError:
            components = ['c0r']

        if not texts:
            # Return zeros for each component
            result = {'n_items': 0, '_components': components}
            for comp in components:
                result[comp] = self._get_empty_component(comp)
            return result

        embeddings = self.embed_texts(texts)
        result = self.decompose(embeddings)
        result['n_items'] = len(texts)

        if return_embeddings:
            result['embeddings'] = embeddings

        return result

    def _get_empty_component(self, comp: str) -> np.ndarray:
        """Return zero array of correct size for a component."""
        # Circular Fourier: 3 * (d/2) per harmonic
        if comp.startswith('h'):
            return np.zeros(3 * (self.embedding_dim // 2))
        # Chebyshev components require explicit type
        if comp.startswith('c'):
            comp_type = comp[-1]  # r, a, or m
            if comp_type == 'r':
                return np.zeros(self.embedding_dim)
            else:
                return np.zeros(self.embedding_dim // 2)
        # Other bases: check if type suffix present
        elif comp[-1] in ('a', 'm'):
            return np.zeros(self.embedding_dim // 2)
        else:
            # Raw (no suffix or 'r' suffix) = full embedding dim
            return np.zeros(self.embedding_dim)

    def encode_batch(
        self,
        sequences: List[List[str]],
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Encode multiple sequences.

        Args:
            sequences: List of text sequences
            show_progress: Show progress bar

        Returns:
            Dict with stacked arrays for each individual component (c0r, c0a, etc.)
            Plus metadata: n_items, _components
        """
        from tqdm import tqdm

        # Get component list for this method
        method = self.config.chebyshev.method
        degree = self.config.effective_degree
        try:
            components = get_method_components(method, degree)
        except ValueError:
            components = ['c0r']

        # Initialize lists for each component
        component_lists = {comp: [] for comp in components}
        n_items_list = []

        iterator = tqdm(sequences, desc="Encoding") if show_progress else sequences

        for seq in iterator:
            result = self.encode_sequence(seq)
            for comp in components:
                if comp in result:
                    component_lists[comp].append(result[comp])
            n_items_list.append(result['n_items'])

        # Stack into arrays
        output = {}
        for comp in components:
            if component_lists[comp]:
                output[comp] = np.array(component_lists[comp])

        output['n_items'] = np.array(n_items_list)
        output['_components'] = components

        return output

    def encode_with_checkpoints(
        self,
        sequences: List[List[str]],
        checkpoint_path: Path,
        checkpoint_interval: int = 500,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Encode with periodic checkpoint saves for resumability.

        Args:
            sequences: List of text sequences
            checkpoint_path: Path to save checkpoints
            checkpoint_interval: Save every N sequences
            show_progress: Show progress bar

        Returns:
            Dict with stacked arrays for each individual component (c0r, c0a, etc.)
        """
        import tempfile
        import os
        from tqdm import tqdm

        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Get component list for this method
        method = self.config.chebyshev.method
        degree = self.config.effective_degree
        try:
            components = get_method_components(method, degree)
        except ValueError:
            components = ['c0r']

        # Check for existing checkpoint
        start_idx = 0
        component_lists = {comp: [] for comp in components}
        n_items_list = []

        if checkpoint_path.exists():
            checkpoint = np.load(checkpoint_path, allow_pickle=True)
            start_idx = int(checkpoint['processed_count'])
            for comp in components:
                if comp in checkpoint:
                    component_lists[comp] = list(checkpoint[comp])
            n_items_list = list(checkpoint['n_items'])
            print(f"Resuming from checkpoint: {start_idx}/{len(sequences)} processed")

        total = len(sequences)

        # Process remaining
        iterator = range(start_idx, total)
        if show_progress:
            iterator = tqdm(iterator, initial=start_idx, total=total, desc="Encoding")

        for i in iterator:
            result = self.encode_sequence(sequences[i])
            for comp in components:
                if comp in result:
                    component_lists[comp].append(result[comp])
            n_items_list.append(result['n_items'])

            # Save checkpoint
            if (i + 1) % checkpoint_interval == 0:
                self._save_checkpoint(
                    checkpoint_path,
                    i + 1,
                    component_lists,
                    n_items_list,
                    components,
                    total_count=total
                )

        # Final save
        self._save_checkpoint(
            checkpoint_path,
            total,
            component_lists,
            n_items_list,
            components,
            total_count=total
        )

        # Stack into arrays
        output = {}
        for comp in components:
            if component_lists[comp]:
                output[comp] = np.array(component_lists[comp])

        output['n_items'] = np.array(n_items_list)
        output['_components'] = components

        return output

    def _save_checkpoint(
        self,
        path: Path,
        processed_count: int,
        component_lists: Dict[str, List],
        n_items_list: List,
        components: List[str],
        total_count: Optional[int] = None
    ):
        """Atomically save checkpoint."""
        import tempfile
        import os
        from datetime import datetime

        temp_fd, temp_path = tempfile.mkstemp(suffix='.npz', dir=path.parent)
        os.close(temp_fd)

        save_dict = {
            'processed_count': processed_count,
            'total_count': total_count or 0,
            'n_items': np.array(n_items_list),
            '_components': components,
            'last_updated': datetime.now().isoformat(),
            'config_name': self.config.name,
            'model_name': self.model_name,
            'degree': self.degree
        }

        # Save each component
        for comp in components:
            if component_lists[comp]:
                save_dict[comp] = np.array(component_lists[comp])

        np.savez(temp_path, **save_dict)
        os.replace(temp_path, path)

    @staticmethod
    def get_checkpoint_status(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """
        Get status of an encoding checkpoint.

        Returns:
            Dict with progress info, or None if no checkpoint exists
        """
        if not checkpoint_path.exists():
            return None

        try:
            data = np.load(checkpoint_path, allow_pickle=True)
            processed = int(data.get('processed_count', 0))
            total = int(data.get('total_count', 0))

            return {
                'processed': processed,
                'total': total,
                'progress': processed / total if total > 0 else 0,
                'remaining': total - processed if total > 0 else None,
                'last_updated': str(data.get('last_updated', 'unknown')),
                'config_name': str(data.get('config_name', 'unknown')),
                'checkpoint_path': str(checkpoint_path)
            }
        except Exception as e:
            return {'error': str(e), 'checkpoint_path': str(checkpoint_path)}


class TrajectoryEncoder(ChebyshevEncoder):
    """
    Extended encoder for conversation trajectories.

    Adds trajectory-specific features like deltas and statistics.
    """

    def encode_trajectory(
        self,
        turns: List[str],
        compute_deltas: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Encode a conversation trajectory with additional features.

        Args:
            turns: List of turn texts
            compute_deltas: Whether to compute turn-to-turn deltas

        Returns:
            Dict with:
                - Individual components (c0r, c0a, etc.)
                - combined: Concatenated components for convenience
                - embeddings: Raw turn embeddings
                - deltas: Turn-to-turn embedding changes (if compute_deltas)
                - trajectory_stats: Summary statistics
        """
        if not turns:
            return self._empty_trajectory_result()

        embeddings = self.embed_texts(turns)
        result = self.decompose(embeddings)
        result['combined'] = self.combine(result)
        result['embeddings'] = embeddings
        result['n_turns'] = len(turns)

        if compute_deltas and len(turns) > 1:
            deltas = np.diff(embeddings, axis=0)
            result['deltas'] = deltas
            result['delta_norms'] = np.linalg.norm(deltas, axis=1)
            result['mean_delta'] = np.mean(deltas, axis=0)
            result['total_distance'] = np.sum(result['delta_norms'])

        # Trajectory statistics
        c0r = result.get('c0r', np.zeros(self.embedding_dim))
        c0r_norm = np.linalg.norm(c0r) if c0r.size > 0 else 0.0

        # Compute c1+ norm from individual angular components if available
        c1_parts = [result[c] for c in result.get('_components', []) if c.startswith('c') and c != 'c0r' and c != 'c0a' and c != 'c0m']
        c1_plus_norm = np.linalg.norm(np.concatenate(c1_parts)) if c1_parts else 0.0

        result['trajectory_stats'] = {
            'c0r_norm': c0r_norm,
            'c1_plus_norm': c1_plus_norm,
            'embedding_mean_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
        }

        return result

    def _empty_trajectory_result(self) -> Dict[str, Any]:
        """Return empty result for empty input."""
        method = self.config.chebyshev.method
        degree = self.config.effective_degree
        try:
            components = get_method_components(method, degree)
        except ValueError:
            components = ['c0r']

        result = {'_components': components}
        for comp in components:
            comp_type = comp[-1]  # r, a, or m
            if comp_type == 'r':
                result[comp] = np.zeros(self.embedding_dim)
            else:
                result[comp] = np.zeros(self.embedding_dim // 2)

        result['combined'] = np.zeros(1)  # Placeholder
        result['embeddings'] = np.zeros((0, self.embedding_dim))
        result['n_turns'] = 0
        result['trajectory_stats'] = {
            'c0r_norm': 0.0,
            'c1_plus_norm': 0.0,
            'embedding_mean_norm': 0.0,
        }
        return result


class TokenLevelEncoder:
    """
    Encode single sentences at token level for word-order analysis.

    Unlike ChebyshevEncoder (which fits Chebyshev to a sequence of sentence embeddings),
    this class fits Chebyshev/DCT to the token embeddings WITHIN a single sentence.

    Use case: PAWS word-order discrimination, where we analyze how word arrangement
    affects meaning within individual sentences.

    Example:
        encoder = TokenLevelEncoder("minilm_c0r.c0a.c1a.c2a")
        result = encoder.encode_sentence_all_methods("The dog chased the cat.")

        # Returns individual components:
        # - Chebyshev: c0r, c0a, c1a, c2a (fitted to token sequence)
        # - DCT: d0, d1, d2, ... (frequency decomposition)
        # - Baselines: mean_pool, max_pool, first_last, weighted_*
    """

    def __init__(self, config: Union[EncodingConfig, str]):
        """
        Initialize token-level encoder.

        Args:
            config: EncodingConfig object or config name string (e.g., "minilm_c0r.c0a.c1a.c2a")
        """
        if isinstance(config, str):
            config = load_config(config)

        self.config = config
        self.degree = config.chebyshev.degree
        self.embedding_dim = config.model.dim
        self.model_name = config.model.name
        self.normalize = config.model.normalize

        self._model = None

    @property
    def model(self):
        """Lazy load embedding model."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """Load the sentence transformer model."""
        from sentence_transformers import SentenceTransformer
        from .device import get_device
        self._model = SentenceTransformer(self.model_name, device=get_device())

    def get_token_embeddings(self, text: str) -> np.ndarray:
        """
        Get per-token embeddings (before pooling) for a single text.

        Args:
            text: Input text string

        Returns:
            Array of shape (n_tokens, embedding_dim) - tokens excluding [CLS]/[SEP]
        """
        import torch

        encoded = self.model.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=False
        )

        with torch.no_grad():
            output = self.model[0].auto_model(**encoded)
            embeddings = output.last_hidden_state[0].cpu().numpy()

        # Skip [CLS] and [SEP] tokens
        return embeddings[1:-1]

    # DEPRECATED: Chebyshev fitting in TokenLevelEncoder is slow.
    # Use Gram polynomials or Fourier magnitude instead.
    #
    # def _fit_chebyshev(self, embeddings: np.ndarray) -> np.ndarray:
    #     """Fit Chebyshev polynomials to embeddings."""
    #     ...
    #
    # def _decompose_tokens(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    #     """Decompose token embeddings into Chebyshev components."""
    #     ...
    #
    # def _fit_chebyshev_data(self, data: np.ndarray, degree: int) -> np.ndarray:
    #     """Fit Chebyshev to arbitrary data."""
    #     ...

    def encode_sentence(self, text: str) -> Dict[str, np.ndarray]:
        """
        DEPRECATED: Chebyshev encoding is deprecated.
        Use encode_sentence_all_methods() for DCT/Gram components or Fourier magnitude.
        """
        raise NotImplementedError(
            "Chebyshev encoding in TokenLevelEncoder is deprecated. "
            "Use encode_sentence_all_methods() which includes DCT, Gram, and Fourier components."
        )

    def encode_sentence_all_methods(self, text: str) -> Dict[str, np.ndarray]:
        """
        Encode with ALL methods for comparison experiments.

        Args:
            text: Input text string

        Returns:
            Dict with:
                - DCT raw: d0, d1, d2, ...
                - DCT angular: d0a, d1a, d2a, ...
                - DCT magnitude: d0m, d1m, d2m, ...
                - Gram: g0, g1, g2, ... (orthogonal polynomials)
                - Fourier magnitude: f0, f1, f2, ... (phase-invariant)
                - Baselines: mean_pool, max_pool, first_last, weighted_*
                - n_tokens: number of tokens in sentence

        Note: Chebyshev components (c0r, c1a, etc.) are deprecated. Use Gram instead.
        """
        from .encoders.pooling import mean_pool, max_pool, first_last_pool, position_weighted_mean
        from .encoders.dct import dct_encode_decomposed
        from .encoders.gram import gram_encode_decomposed
        from .encoders.fourier import fourier_magnitude_encode_decomposed

        embeddings = self.get_token_embeddings(text)

        if len(embeddings) == 0:
            return self._empty_result_all()

        result = {}
        n_coeffs = self.degree + 1

        # DCT components (raw, angular, magnitude)
        dct = dct_encode_decomposed(embeddings, n_coefficients=n_coeffs,
                                     include_angular=True, include_magnitude=True)
        result.update({k: v for k, v in dct.items() if k.startswith('d')})

        # Gram polynomial components (orthogonal at uniform points)
        gram = gram_encode_decomposed(embeddings, n_coefficients=n_coeffs,
                                       include_angular=True, include_magnitude=True)
        result.update({k: v for k, v in gram.items() if k.startswith('g')})

        # Fourier magnitude components (phase-invariant)
        fourier = fourier_magnitude_encode_decomposed(embeddings, n_coefficients=n_coeffs,
                                                       include_angular=True, include_magnitude=True)
        result.update({k: v for k, v in fourier.items() if k.startswith('f')})

        # Baselines
        result['mean_pool'] = mean_pool(embeddings)
        result['max_pool'] = max_pool(embeddings)
        result['first_last'] = first_last_pool(embeddings)
        result['weighted_recency'] = position_weighted_mean(embeddings, 'recency')
        result['weighted_ends'] = position_weighted_mean(embeddings, 'ends')

        result['n_tokens'] = len(embeddings)
        # Primary components are DCT for now
        result['_components'] = [f'd{i}' for i in range(n_coeffs)]

        return result

    # DEPRECATED: Chebyshev decomposition removed. Use encode_sentence_all_methods() instead.
    #
    # def _decompose_tokens_full(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    #     """Decompose token embeddings into ALL Chebyshev components."""
    #     ...

    def encode_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Encode multiple sentences.

        Args:
            texts: List of text strings
            show_progress: Show progress bar

        Returns:
            Dict with stacked arrays for each component
        """
        from tqdm import tqdm

        results = []
        iterator = tqdm(texts, desc="Encoding sentences") if show_progress else texts

        for text in iterator:
            results.append(self.encode_sentence_all_methods(text))

        if not results:
            return self._empty_result_all()

        # Get component names from first result
        component_names = [k for k in results[0].keys() if not k.startswith('_') and k != 'n_tokens']

        # Stack into arrays
        output = {}
        for comp in component_names:
            output[comp] = np.array([r[comp] for r in results])

        output['n_tokens'] = np.array([r['n_tokens'] for r in results])
        output['_components'] = results[0].get('_components', [])

        return output

    def encode_batch_with_checkpoints(
        self,
        texts: List[str],
        checkpoint_path: Path,
        checkpoint_interval: int = 100,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Encode with periodic checkpoint saves for resumability.

        Args:
            texts: List of text strings
            checkpoint_path: Path to save checkpoints
            checkpoint_interval: Save every N sentences
            show_progress: Show progress bar

        Returns:
            Dict with stacked arrays for each component
        """
        import tempfile
        import os
        from tqdm import tqdm
        from datetime import datetime

        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for existing checkpoint
        start_idx = 0
        results = []

        if checkpoint_path.exists():
            checkpoint = np.load(checkpoint_path, allow_pickle=True)
            start_idx = int(checkpoint['processed_count'])
            results = list(checkpoint['results'])
            print(f"Resuming from checkpoint: {start_idx}/{len(texts)} processed")

        total = len(texts)

        # Process remaining
        iterator = range(start_idx, total)
        if show_progress:
            iterator = tqdm(iterator, initial=start_idx, total=total, desc="Encoding")

        for i in iterator:
            result = self.encode_sentence_all_methods(texts[i])
            results.append(result)

            # Save checkpoint
            if (i + 1) % checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_path, i + 1, results, total)

        # Final save
        self._save_checkpoint(checkpoint_path, total, results, total)

        if not results:
            return self._empty_result_all()

        # Stack into arrays
        component_names = [k for k in results[0].keys() if not k.startswith('_') and k != 'n_tokens']

        output = {}
        for comp in component_names:
            output[comp] = np.array([r[comp] for r in results])

        output['n_tokens'] = np.array([r['n_tokens'] for r in results])
        output['_components'] = results[0].get('_components', [])

        return output

    def _save_checkpoint(self, path: Path, processed_count: int, results: List, total_count: int):
        """Atomically save checkpoint."""
        import tempfile
        import os
        from datetime import datetime

        temp_fd, temp_path = tempfile.mkstemp(suffix='.npz', dir=path.parent)
        os.close(temp_fd)

        np.savez(
            temp_path,
            processed_count=processed_count,
            total_count=total_count,
            results=np.array(results, dtype=object),
            last_updated=datetime.now().isoformat(),
            config_name=self.config.name,
            model_name=self.model_name,
            degree=self.degree
        )
        os.replace(temp_path, path)

    def _empty_result(self) -> Dict[str, np.ndarray]:
        """Return empty result for Chebyshev-only encoding."""
        method = self.config.chebyshev.method
        degree = self.config.effective_degree

        try:
            components = get_method_components(method, degree)
        except ValueError:
            components = ['c0r']

        result = {'_components': components}
        for comp in components:
            comp_type = comp[-1]
            if comp_type == 'r':
                result[comp] = np.zeros(self.embedding_dim)
            else:
                result[comp] = np.zeros(self.embedding_dim // 2)

        result['n_tokens'] = 0
        return result

    def _empty_result_all(self) -> Dict[str, np.ndarray]:
        """Return empty result for all-methods encoding."""
        result = {}
        n_coeffs = self.config.effective_degree + 1

        # DCT components (raw, angular, magnitude)
        for i in range(n_coeffs):
            result[f'd{i}'] = np.zeros(self.embedding_dim)
            result[f'd{i}a'] = np.zeros(self.embedding_dim // 2)
            result[f'd{i}m'] = np.zeros(self.embedding_dim // 2)

        # Gram polynomial components (orthogonal at uniform points)
        for i in range(n_coeffs):
            result[f'g{i}'] = np.zeros(self.embedding_dim)
            result[f'g{i}a'] = np.zeros(self.embedding_dim // 2)
            result[f'g{i}m'] = np.zeros(self.embedding_dim // 2)

        # Fourier magnitude components (phase-invariant)
        for i in range(n_coeffs):
            result[f'f{i}'] = np.zeros(self.embedding_dim)
            result[f'f{i}a'] = np.zeros(self.embedding_dim // 2)
            result[f'f{i}m'] = np.zeros(self.embedding_dim // 2)

        # Baselines
        result['mean_pool'] = np.zeros(self.embedding_dim)
        result['max_pool'] = np.zeros(self.embedding_dim)
        result['first_last'] = np.zeros(self.embedding_dim * 2)
        result['weighted_recency'] = np.zeros(self.embedding_dim)
        result['weighted_ends'] = np.zeros(self.embedding_dim)

        result['n_tokens'] = 0
        return result
