"""
Encoding configuration management.

Defines the schema for embedding model + encoding combinations.

Component Naming Convention:
    Components use short codes: {basis}{degree}{type}
    - basis: c=Chebyshev, f=Fourier magnitude, d=DCT, s=DST, g=Gram
    - degree: 0, 1, 2, ... (coefficient index)
    - type: r=raw, a=angular, m=magnitude (optional for f/d/s/g, defaults to raw)

    Examples:
        c0r = Chebyshev c0 raw (384 dims for 384d model)
        c0a = Chebyshev c0 angular (192 dims)
        f0  = Fourier magnitude f0 raw (384 dims)
        f1  = Fourier magnitude f1 raw (384 dims) - KEY for word order
        f1a = Fourier magnitude f1 angular (192 dims)
        d0  = DCT coefficient 0 (384 dims)
        d1  = DCT coefficient 1 (384 dims)
        s1  = first DST harmonic j=1 (384 dims); DST is 1-indexed, no s0/DC

    Method names are component lists joined by '.':
        c0r           = just Chebyshev c0 raw (mean pooling equivalent)
        f0.f1.f2      = Fourier magnitude coefficients 0-2
        c0r.f1        = Chebyshev semantic + Fourier positional (hybrid)

    Full config names: {model}_{components}
        minilm_c0r.c0a.c1a.c2a
        minilm_f0.f1.f2
        gte-small_c0r
"""

import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import re


# =============================================================================
# Component Code System
# =============================================================================

# Component code to full name mapping
# Chebyshev: c{degree}{type} where type is r/a/m
# Fourier magnitude: f{degree} or f{degree}{type}
# DCT: d{degree} or d{degree}{type}
# DST: s{k} (k>=1) or s{k}{type}  -- s{k}=harmonic j=k, no s0/DC term
# Gram: g{degree} or g{degree}{type}
COMPONENT_CODES = {
    # Chebyshev components (require explicit type: r/a/m)
    'c0r': 'c0_raw',
    'c0a': 'c0_ang',
    'c0m': 'c0_mag',
    'c1r': 'c1_raw',
    'c1a': 'c1_ang',
    'c1m': 'c1_mag',
    'c2r': 'c2_raw',
    'c2a': 'c2_ang',
    'c2m': 'c2_mag',
    'c3r': 'c3_raw',
    'c3a': 'c3_ang',
    'c4r': 'c4_raw',
    'c4a': 'c4_ang',
    'c5r': 'c5_raw',
    'c5a': 'c5_ang',
    # Fourier magnitude components (f{k} = √(d_k² + s_k²), phase-invariant)
    'f0': 'f0_raw', 'f0a': 'f0_ang', 'f0m': 'f0_mag',
    'f1': 'f1_raw', 'f1a': 'f1_ang', 'f1m': 'f1_mag',
    'f2': 'f2_raw', 'f2a': 'f2_ang', 'f2m': 'f2_mag',
    'f3': 'f3_raw', 'f3a': 'f3_ang', 'f3m': 'f3_mag',
    'f4': 'f4_raw', 'f4a': 'f4_ang', 'f4m': 'f4_mag',
    'f5': 'f5_raw', 'f5a': 'f5_ang', 'f5m': 'f5_mag',
    'f6': 'f6_raw', 'f6a': 'f6_ang', 'f6m': 'f6_mag',
    # DCT components (d{k} = DCT coefficient k)
    'd0': 'd0_raw', 'd0a': 'd0_ang', 'd0m': 'd0_mag',
    'd1': 'd1_raw', 'd1a': 'd1_ang', 'd1m': 'd1_mag',
    'd2': 'd2_raw', 'd2a': 'd2_ang', 'd2m': 'd2_mag',
    'd3': 'd3_raw', 'd3a': 'd3_ang', 'd3m': 'd3_mag',
    'd4': 'd4_raw', 'd4a': 'd4_ang', 'd4m': 'd4_mag',
    'd5': 'd5_raw', 'd5a': 'd5_ang', 'd5m': 'd5_mag',
    'd6': 'd6_raw', 'd6a': 'd6_ang', 'd6m': 'd6_mag',
    # DST components (s{k} = DST harmonic j=k, 1-indexed; no s0/DC term)
    's1': 's1_raw', 's1a': 's1_ang', 's1m': 's1_mag',
    's2': 's2_raw', 's2a': 's2_ang', 's2m': 's2_mag',
    's3': 's3_raw', 's3a': 's3_ang', 's3m': 's3_mag',
    's4': 's4_raw', 's4a': 's4_ang', 's4m': 's4_mag',
    's5': 's5_raw', 's5a': 's5_ang', 's5m': 's5_mag',
    's6': 's6_raw', 's6a': 's6_ang', 's6m': 's6_mag',
    's7': 's7_raw', 's7a': 's7_ang', 's7m': 's7_mag',
    # Gram polynomial components (g{k} = Gram coefficient k, orthogonal at uniform points)
    'g0': 'g0_raw', 'g0a': 'g0_ang', 'g0m': 'g0_mag',
    'g1': 'g1_raw', 'g1a': 'g1_ang', 'g1m': 'g1_mag',
    'g2': 'g2_raw', 'g2a': 'g2_ang', 'g2m': 'g2_mag',
    'g3': 'g3_raw', 'g3a': 'g3_ang', 'g3m': 'g3_mag',
    'g4': 'g4_raw', 'g4a': 'g4_ang', 'g4m': 'g4_mag',
    'g5': 'g5_raw', 'g5a': 'g5_ang', 'g5m': 'g5_mag',
    'g6': 'g6_raw', 'g6a': 'g6_ang', 'g6m': 'g6_mag',
    # Circular Fourier components (h{k} = [angle_cos_k, angle_sin_k, magnitude_k])
    'h0': 'h0_raw', 'h1': 'h1_raw', 'h2': 'h2_raw',
    'h3': 'h3_raw', 'h4': 'h4_raw', 'h5': 'h5_raw', 'h6': 'h6_raw',
}

# Reverse mapping
COMPONENT_NAMES = {v: k for k, v in COMPONENT_CODES.items()}

# Valid basis prefixes
VALID_BASES = {'c', 'f', 'd', 's', 'g', 'h'}

# Component dimensions (multiplier of embedding_dim)
# r (raw) = 1.0, a (angular) = 0.5, m (magnitude) = 0.5
COMPONENT_DIM_MULT = {
    'r': 1.0,    # raw: full embedding dim
    'a': 0.5,    # angular: half (dimension pairs → angles)
    'm': 0.5,    # magnitude: half (dimension pairs → magnitudes)
}


def parse_components(method_str: str) -> List[str]:
    """Parse component string like 'c0r.c0a.c1a.c2a' into list of codes."""
    if not method_str:
        return []
    return method_str.split('.')


def serialize_components(components: List[str]) -> str:
    """Serialize list of component codes into method string."""
    return '.'.join(components)


def components_to_full_names(components: List[str]) -> List[str]:
    """Convert component codes to full names for internal use."""
    return [COMPONENT_CODES.get(c, c) for c in components]


def full_names_to_components(names: List[str]) -> List[str]:
    """Convert full names to component codes."""
    return [COMPONENT_NAMES.get(n, n) for n in names]


def compute_dims(components: List[str], embedding_dim: int) -> int:
    """Compute total dimensions for a component list."""
    total = 0
    for comp in components:
        if len(comp) >= 2:
            # Circular Fourier (h): each harmonic has 3 * (d/2) dims
            if comp.startswith('h'):
                total += 3 * (embedding_dim // 2)
                continue
            # Check if last char is a type specifier
            if comp[-1] in ('r', 'a', 'm'):
                comp_type = comp[-1]
            else:
                # No type suffix (e.g., 'f1', 'd0') defaults to raw
                comp_type = 'r'
            mult = COMPONENT_DIM_MULT.get(comp_type, 1.0)
            total += int(embedding_dim * mult)
    return total


def infer_degree(components: List[str]) -> int:
    """Infer the maximum degree from component list."""
    max_degree = 0
    for comp in components:
        # Match any valid basis: c/f/d/s/g followed by degree and optional type
        match = re.match(r'[cfdsgh](\d+)[ram]?', comp)
        if match:
            degree = int(match.group(1))
            max_degree = max(max_degree, degree)
    return max_degree


def validate_components(components: List[str]) -> Tuple[bool, str]:
    """Validate a component list. Returns (is_valid, error_message)."""
    for comp in components:
        # Chebyshev requires explicit type (r/a/m)
        # Other bases (f/d/s/g) can omit type (defaults to raw)
        if comp.startswith('c'):
            if not re.match(r'^c\d+[ram]$', comp):
                return False, f"Invalid Chebyshev component: {comp}. Expected format: c{{degree}}{{type}} where type is r/a/m"
        else:
            # f/d/s/g/h can be just the number or number + type
            if not re.match(r'^[fdsgh]\d+[ram]?$', comp):
                return False, f"Invalid component code: {comp}. Expected format: {{basis}}{{degree}}[{{type}}] where basis is c/f/d/s/g/h"

        if comp not in COMPONENT_CODES:
            # Check if it's a valid but unlisted high-degree component
            match = re.match(r'([cfdsg])(\d+)([ram])?', comp)
            if not match:
                return False, f"Invalid component: {comp}"
            degree = int(match.group(2))
            if degree > 10:
                return False, f"Degree {degree} too high (max 10)"
    return True, ""


# =============================================================================
# Legacy method aliases (maps old names to component lists)
# =============================================================================

LEGACY_METHOD_ALIASES = {
    'c0-only': 'c0r',
    'c0-dual': 'c0r.c0a',
    'c0-full': 'c0r.c0a.c0m',
    'dual-d1': 'c0r.c0a.c1a',
    'dual-d2': 'c0r.c0a.c1a.c2a',
    'dual-d2-mag': 'c0r.c0a.c0m.c1a.c2a',
    # Variable degree methods - these need special handling
    'raw': None,      # Depends on degree: c0r.c1r.c2r...
    'angular': None,  # Depends on degree: c0a.c1a.c2a...
    'hybrid': None,   # c0r + angular c1+
    'dual': None,     # c0r.c0a + angular c1+
}


def resolve_method(method: str, degree: int) -> str:
    """Resolve a method name (possibly legacy) to component string."""
    # If it looks like a component string already, return as-is
    if '.' in method or re.match(r'^c\d+[ram]$', method):
        return method

    # Check legacy aliases
    if method in LEGACY_METHOD_ALIASES:
        resolved = LEGACY_METHOD_ALIASES[method]
        if resolved is not None:
            return resolved

        # Handle variable-degree methods
        if method == 'raw':
            comps = [f'c{i}r' for i in range(degree + 1)]
            return serialize_components(comps)
        elif method == 'angular':
            comps = [f'c{i}a' for i in range(degree + 1)]
            return serialize_components(comps)
        elif method == 'hybrid':
            comps = ['c0r'] + [f'c{i}a' for i in range(1, degree + 1)]
            return serialize_components(comps)
        elif method == 'dual':
            comps = ['c0r', 'c0a'] + [f'c{i}a' for i in range(1, degree + 1)]
            return serialize_components(comps)

    # Unknown method
    raise ValueError(f"Unknown method: {method}")


@dataclass
class ModelConfig:
    """Embedding model configuration."""
    name: str                          # HuggingFace model name
    dim: int                           # Embedding dimension
    max_seq_length: int = 512          # Max tokens per input
    normalize: bool = True             # L2 normalize embeddings

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ChebyshevConfig:
    """Chebyshev polynomial encoding configuration."""
    degree: int = 5                    # Polynomial degree
    normalize_positions: bool = True   # Map positions to [-1, 1]
    method: str = 'raw'                # Encoding method (see ENCODING_METHODS)
    components: List[str] = field(default_factory=lambda: ['c0', 'c1_plus'])
    # Options: 'full', 'c0', 'c1_plus', 'c0_c1_c2_c3_c4_c5' (individual)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ChebyshevConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def output_dim(self) -> int:
        """Calculate output dimension based on components."""
        # Will be multiplied by embedding dim
        if 'full' in self.components:
            return self.degree + 1
        dim = 0
        if 'c0' in self.components:
            dim += 1
        if 'c1_plus' in self.components:
            dim += self.degree  # c1 through c_degree
        return dim


@dataclass
class ProcessingConfig:
    """Pre-processing configuration applied before Chebyshev fitting.

    Modifiers in config names (e.g., +rope, +random) map to this config.
    """
    rope: bool = False                 # Apply RoPE rotation before Chebyshev
    rope_base: float = 10000.0         # RoPE base frequency
    random_rotation: bool = False      # Apply random orthogonal rotation (ablation)
    random_seed: int = 42              # Seed for random rotation

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ProcessingConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_modifier(cls, modifier: Optional[str]) -> 'ProcessingConfig':
        """Create ProcessingConfig from a modifier string like 'rope' or 'random'."""
        if modifier is None:
            return cls()
        elif modifier == 'rope':
            return cls(rope=True)
        elif modifier == 'random':
            return cls(random_rotation=True)
        else:
            raise ValueError(f"Unknown modifier: {modifier}. Valid modifiers: rope, random")

    @property
    def modifier_string(self) -> Optional[str]:
        """Return the modifier string representation, or None if no processing."""
        if self.rope:
            return 'rope'
        elif self.random_rotation:
            return 'random'
        return None


def parse_config_name(name: str) -> Tuple[str, str, Optional[str]]:
    """
    Parse config name into (model, method, modifier).

    Examples:
        "minilm_c0r.c0a.c1a.c2a" -> ("minilm", "c0r.c0a.c1a.c2a", None)
        "minilm_c0r.c0a.c1a.c2a+rope" -> ("minilm", "c0r.c0a.c1a.c2a", "rope")
        "gte-small_c0r+random" -> ("gte-small", "c0r", "random")
    """
    # Split off modifier first
    if '+' in name:
        base, modifier = name.rsplit('+', 1)
    else:
        base, modifier = name, None

    # Split model and method
    if '_' in base:
        model, method = base.split('_', 1)
    else:
        # Fallback for legacy names without underscore
        model, method = base, 'raw'

    return model, method, modifier


def build_config_name(model_short: str, method: str, modifier: Optional[str] = None) -> str:
    """Build config name from components."""
    name = f"{model_short}_{method}"
    if modifier:
        name += f"+{modifier}"
    return name


# =============================================================================
# Encoding Methods
# =============================================================================
#
# Key concepts (see material/theory/):
# - c₀: Zeroth Chebyshev coefficient ≈ mean pooling. Captures semantic content.
# - c₁: Linear trend coefficient. Captures start-to-end change within turn.
# - c₂: Curvature coefficient. Captures whether middle differs from endpoints.
# - c₁+: Higher-degree coefficients. Capture WITHIN-TURN token arrangement.
#
# Component types:
# - r (raw): Direct coefficient values. Preserves magnitude.
# - a (angular): arctan2 of dimension pairs. Direction only.
# - m (magnitude): Euclidean norm of pairs. Intensity only.
#
# NEW: Method names ARE the component list: 'c0r.c0a.c1a.c2a'
# LEGACY: Old names like 'dual-d2' are aliases (see LEGACY_METHOD_ALIASES)
#

ENCODING_METHODS = {
    # === Fixed component methods (new canonical names) ===
    'c0r': {
        'description': 'c0 raw only (≈ mean pooling). No positional info.',
        'components': ['c0r'],
        'dim_formula': lambda emb_dim, degree: emb_dim,
    },
    'c0r.c0a': {
        'description': 'c0 raw + c0 angular. Semantic mean + direction. No positional info.',
        'components': ['c0r', 'c0a'],
        'dim_formula': lambda emb_dim, degree: emb_dim + (emb_dim // 2),
    },
    'c0r.c0a.c0m': {
        'description': 'c0 raw + angular + magnitude. Full c0 decomposition. No positional info.',
        'components': ['c0r', 'c0a', 'c0m'],
        'dim_formula': lambda emb_dim, degree: emb_dim + (emb_dim // 2) + (emb_dim // 2),
    },
    'c0r.c0a.c1a': {
        'description': 'c0 raw+angular + c1 angular. Mean + linear trend.',
        'components': ['c0r', 'c0a', 'c1a'],
        'dim_formula': lambda emb_dim, degree: emb_dim + (emb_dim // 2) + (emb_dim // 2),
        'fixed_degree': 1,
    },
    'c0r.c0a.c1a.c2a': {
        'description': 'c0 raw+angular + c1+c2 angular. Mean + linear + curvature.',
        'components': ['c0r', 'c0a', 'c1a', 'c2a'],
        'dim_formula': lambda emb_dim, degree: emb_dim + (emb_dim // 2) + (emb_dim // 2) + (emb_dim // 2),
        'fixed_degree': 2,
    },
    'c0r.c0a.c0m.c1a.c2a': {
        'description': 'c0 full + c1+c2 angular. Mean (all forms) + linear + curvature.',
        'components': ['c0r', 'c0a', 'c0m', 'c1a', 'c2a'],
        'dim_formula': lambda emb_dim, degree: emb_dim + (emb_dim // 2) * 4,
        'fixed_degree': 2,
    },

    # === Legacy aliases (for backwards compatibility) ===
    # These map to the new canonical names above
    'c0-only': {
        'description': '[LEGACY] Alias for c0r',
        'components': ['c0r'],
        'dim_formula': lambda emb_dim, degree: emb_dim,
        'canonical': 'c0r',
    },
    'c0-dual': {
        'description': '[LEGACY] Alias for c0r.c0a',
        'components': ['c0r', 'c0a'],
        'dim_formula': lambda emb_dim, degree: emb_dim + (emb_dim // 2),
        'canonical': 'c0r.c0a',
    },
    'c0-full': {
        'description': '[LEGACY] Alias for c0r.c0a.c0m',
        'components': ['c0r', 'c0a', 'c0m'],
        'dim_formula': lambda emb_dim, degree: emb_dim + (emb_dim // 2) + (emb_dim // 2),
        'canonical': 'c0r.c0a.c0m',
    },
    'dual-d1': {
        'description': '[LEGACY] Alias for c0r.c0a.c1a',
        'components': ['c0r', 'c0a', 'c1a'],
        'dim_formula': lambda emb_dim, degree: emb_dim + (emb_dim // 2) + (emb_dim // 2),
        'fixed_degree': 1,
        'canonical': 'c0r.c0a.c1a',
    },
    'dual-d2': {
        'description': '[LEGACY] Alias for c0r.c0a.c1a.c2a',
        'components': ['c0r', 'c0a', 'c1a', 'c2a'],
        'dim_formula': lambda emb_dim, degree: emb_dim + (emb_dim // 2) + (emb_dim // 2) + (emb_dim // 2),
        'fixed_degree': 2,
        'canonical': 'c0r.c0a.c1a.c2a',
    },
    'dual-d2-mag': {
        'description': '[LEGACY] Alias for c0r.c0a.c0m.c1a.c2a',
        'components': ['c0r', 'c0a', 'c0m', 'c1a', 'c2a'],
        'dim_formula': lambda emb_dim, degree: emb_dim + (emb_dim // 2) * 4,
        'fixed_degree': 2,
        'canonical': 'c0r.c0a.c0m.c1a.c2a',
    },

    # === Variable degree methods (legacy, require degree param) ===
    'raw': {
        'description': '[LEGACY] All coefficients raw. Use c0r.c1r.c2r... instead.',
        'components': ['c0r', 'c1r+'],  # c1r+ means c1r through c{degree}r
        'dim_formula': lambda emb_dim, degree: emb_dim * (degree + 1),
        'variable_degree': True,
    },
    'angular': {
        'description': '[LEGACY] All coefficients angular. Use c0a.c1a.c2a... instead.',
        'components': ['c0a', 'c1a+'],
        'dim_formula': lambda emb_dim, degree: (emb_dim // 2) * (degree + 1),
        'variable_degree': True,
    },
    'hybrid': {
        'description': '[LEGACY] c0 raw + c1+ angular. Use c0r.c1a.c2a... instead.',
        'components': ['c0r', 'c1a+'],
        'dim_formula': lambda emb_dim, degree: emb_dim + (degree * emb_dim // 2),
        'variable_degree': True,
    },
    'dual': {
        'description': '[LEGACY] c0 raw+angular + c1+ angular. Use c0r.c0a.c1a... instead.',
        'components': ['c0r', 'c0a', 'c1a+'],
        'dim_formula': lambda emb_dim, degree: emb_dim + (emb_dim // 2) + (degree * emb_dim // 2),
        'variable_degree': True,
    },

    # =========================================================================
    # FOURIER MAGNITUDE METHODS
    # f_k = √(d_k² + s_k²) - phase-invariant, optimal for word-order detection
    # Key finding: f1 achieves 0.7654 AUC on PAWS, outperforming d1 and s1
    # =========================================================================
    'f0.f1': {
        'description': 'Fourier magnitude f0+f1. Semantic + phase-invariant linear.',
        'components': ['f0', 'f1'],
        'dim_formula': lambda emb_dim, degree: emb_dim * 2,
        'fixed_degree': 1,
        'basis': 'fourier',
    },
    'f0.f1.f2': {
        'description': 'Fourier magnitude f0+f1+f2. Semantic + linear + curvature.',
        'components': ['f0', 'f1', 'f2'],
        'dim_formula': lambda emb_dim, degree: emb_dim * 3,
        'fixed_degree': 2,
        'basis': 'fourier',
    },
    'f1': {
        'description': 'Fourier magnitude f1 only. Phase-invariant linear (word-order).',
        'components': ['f1'],
        'dim_formula': lambda emb_dim, degree: emb_dim,
        'fixed_degree': 1,
        'basis': 'fourier',
    },
    'f1.f2': {
        'description': 'Fourier magnitude f1+f2. Positional only (no semantic).',
        'components': ['f1', 'f2'],
        'dim_formula': lambda emb_dim, degree: emb_dim * 2,
        'fixed_degree': 2,
        'basis': 'fourier',
    },
    'c0r.f1': {
        'description': 'Chebyshev semantic + Fourier positional. Best of both worlds.',
        'components': ['c0r', 'f1'],
        'dim_formula': lambda emb_dim, degree: emb_dim * 2,
        'fixed_degree': 1,
        'basis': 'hybrid',
    },
    'c0r.f1.f2': {
        'description': 'Chebyshev semantic + Fourier positional (linear + curvature).',
        'components': ['c0r', 'f1', 'f2'],
        'dim_formula': lambda emb_dim, degree: emb_dim * 3,
        'fixed_degree': 2,
        'basis': 'hybrid',
    },
    'd0.f1': {
        'description': 'DCT semantic + Fourier positional. d0 preserves sign (true mean).',
        'components': ['d0', 'f1'],
        'dim_formula': lambda emb_dim, degree: emb_dim * 2,
        'fixed_degree': 1,
        'basis': 'hybrid',
    },
    'd0.f1.f2': {
        'description': 'DCT semantic + Fourier positional (linear + curvature).',
        'components': ['d0', 'f1', 'f2'],
        'dim_formula': lambda emb_dim, degree: emb_dim * 3,
        'fixed_degree': 2,
        'basis': 'hybrid',
    },

    # =========================================================================
    # DCT METHODS (Discrete Cosine Transform)
    # d_k = DCT coefficient k - frequency interpretation, orthogonal at uniform points
    # =========================================================================
    'd0.d1': {
        'description': 'DCT d0+d1. DC + fundamental frequency.',
        'components': ['d0', 'd1'],
        'dim_formula': lambda emb_dim, degree: emb_dim * 2,
        'fixed_degree': 1,
        'basis': 'dct',
    },
    'd0.d1.d2': {
        'description': 'DCT d0+d1+d2. DC + fundamental + second harmonic.',
        'components': ['d0', 'd1', 'd2'],
        'dim_formula': lambda emb_dim, degree: emb_dim * 3,
        'fixed_degree': 2,
        'basis': 'dct',
    },

    # =========================================================================
    # DST METHODS (Discrete Sine Transform)
    # s_k = DST harmonic j=k (1-indexed; no s0/DC) - odd symmetry, complements DCT
    # =========================================================================
    's1.s2': {
        'description': 'DST s1+s2. First two sine harmonics.',
        'components': ['s1', 's2'],
        'dim_formula': lambda emb_dim, degree: emb_dim * 2,
        'fixed_degree': 1,
        'basis': 'dst',
    },
    's1.s2.s3': {
        'description': 'DST s1+s2+s3. First three sine harmonics.',
        'components': ['s1', 's2', 's3'],
        'dim_formula': lambda emb_dim, degree: emb_dim * 3,
        'fixed_degree': 2,
        'basis': 'dst',
    },

    # =========================================================================
    # GRAM POLYNOMIAL METHODS
    # g_k = Gram coefficient k - orthogonal at uniform points, polynomial interpretation
    # =========================================================================
    'g0.g1': {
        'description': 'Gram g0+g1. Truly orthogonal mean + linear.',
        'components': ['g0', 'g1'],
        'dim_formula': lambda emb_dim, degree: emb_dim * 2,
        'fixed_degree': 1,
        'basis': 'gram',
    },
    'g0.g1.g2': {
        'description': 'Gram g0+g1+g2. Truly orthogonal mean + linear + curvature.',
        'components': ['g0', 'g1', 'g2'],
        'dim_formula': lambda emb_dim, degree: emb_dim * 3,
        'fixed_degree': 2,
        'basis': 'gram',
    },
}


def get_method_components(method: str, degree: int = 5) -> List[str]:
    """
    Get the actual component list for a method.

    For fixed methods like 'c0r.c0a.c1a.c2a', returns as-is.
    For variable degree methods like 'raw', expands based on degree.
    """
    if method not in ENCODING_METHODS:
        # Try parsing as component string directly
        # Matches: c0r, f1, d0.d1, g0.g1.g2, c0r.f1, etc.
        if '.' in method or re.match(r'^[cfdsg]\d+[ram]?$', method):
            return parse_components(method)
        raise ValueError(f"Unknown method: {method}")

    info = ENCODING_METHODS[method]
    components = info['components']

    # Handle variable degree methods
    if info.get('variable_degree'):
        expanded = []
        for comp in components:
            if comp.endswith('+'):
                # Expand c1r+ to c1r, c2r, ..., c{degree}r
                base = comp[:-1]  # Remove '+'
                prefix = base[:-1]  # c1
                suffix = base[-1]   # r or a
                start_degree = int(prefix[1:])  # 1
                for d in range(start_degree, degree + 1):
                    expanded.append(f'c{d}{suffix}')
            else:
                expanded.append(comp)
        return expanded

    return components.copy()


@dataclass
class EncodingConfig:
    """Complete encoding configuration (model + chebyshev + processing)."""
    name: str                          # Config identifier (e.g., 'minilm_c0r.c0a.c1a.c2a+rope')
    model: ModelConfig
    chebyshev: ChebyshevConfig
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    description: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EncodingConfig':
        # Handle processing config - can come from dict or be inferred from name
        processing_dict = d.get('processing', {})
        if processing_dict:
            processing = ProcessingConfig.from_dict(processing_dict)
        else:
            # Infer from config name if modifier present
            _, _, modifier = parse_config_name(d.get('name', ''))
            processing = ProcessingConfig.from_modifier(modifier)

        return cls(
            name=d['name'],
            model=ModelConfig.from_dict(d.get('model', {})),
            chebyshev=ChebyshevConfig.from_dict(d.get('chebyshev', {})),
            processing=processing,
            description=d.get('description', '')
        )

    @classmethod
    def from_yaml(cls, path: Path) -> 'EncodingConfig':
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_json(cls, path: Path) -> 'EncodingConfig':
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'name': self.name,
            'model': asdict(self.model),
            'chebyshev': asdict(self.chebyshev),
            'description': self.description
        }
        # Only include processing if non-default
        if self.processing.rope or self.processing.random_rotation:
            d['processing'] = asdict(self.processing)
        return d

    def to_yaml(self, path: Path) -> None:
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @property
    def method_info(self) -> Dict[str, Any]:
        """Get encoding method definition."""
        method = self.chebyshev.method
        if method in ENCODING_METHODS:
            return ENCODING_METHODS[method]

        # Try parsing as component string (e.g., 'c0r.c1r.c2r')
        if '.' in method or re.match(r'^c\d+[ram]$', method):
            components = parse_components(method)
            if components:
                # Dynamically generate method info
                def calc_dim(emb_dim, degree):
                    total = 0
                    for c in components:
                        if c.endswith('r'):
                            total += emb_dim
                        else:  # 'a' or 'm'
                            total += emb_dim // 2
                    return total

                # Infer fixed_degree from component names
                max_degree = 0
                for c in components:
                    if len(c) >= 3 and c[1].isdigit():
                        max_degree = max(max_degree, int(c[1:-1]))

                info = {
                    'description': f'Custom component combination: {method}',
                    'components': components,
                    'dim_formula': calc_dim,
                }
                if max_degree > 0:
                    info['fixed_degree'] = max_degree

                return info

        raise ValueError(f"Unknown method: {method}. Available: {list(ENCODING_METHODS.keys())}")

    @property
    def effective_degree(self) -> int:
        """Get effective degree (some methods override)."""
        info = self.method_info
        if 'fixed_degree' in info:
            return info['fixed_degree']
        return self.chebyshev.degree

    @property
    def total_dim(self) -> int:
        """Total output dimension based on method."""
        info = self.method_info
        return info['dim_formula'](self.model.dim, self.effective_degree)

    @property
    def output_components(self) -> List[str]:
        """List of output component names for this method."""
        return self.method_info['components']

    @property
    def c0_dim(self) -> int:
        """Dimension of c0 component (raw)."""
        return self.model.dim

    @property
    def c0_ang_dim(self) -> int:
        """Dimension of angular c0 component."""
        return self.model.dim // 2

    @property
    def c1_plus_dim(self) -> int:
        """Dimension of c1+ component (raw)."""
        return self.effective_degree * self.model.dim

    @property
    def c1_plus_ang_dim(self) -> int:
        """Dimension of angular c1+ component."""
        return self.effective_degree * (self.model.dim // 2)


# --- Config directory management ---

def get_config_dir() -> Path:
    """Get the configs directory."""
    return Path(__file__).parent.parent.parent / "data" / "configs"


def load_config(name: str) -> EncodingConfig:
    """Load a config by name."""
    config_dir = get_config_dir()

    # Try YAML first, then JSON
    yaml_path = config_dir / f"{name}.yaml"
    json_path = config_dir / f"{name}.json"

    if yaml_path.exists():
        return EncodingConfig.from_yaml(yaml_path)
    elif json_path.exists():
        return EncodingConfig.from_json(json_path)
    else:
        raise FileNotFoundError(f"Config not found: {name} (looked in {config_dir})")


def list_configs() -> List[str]:
    """List available config names."""
    config_dir = get_config_dir()
    if not config_dir.exists():
        return []

    configs = set()
    for f in config_dir.iterdir():
        if f.suffix in ('.yaml', '.json'):
            configs.add(f.stem)
    return sorted(configs)


def save_config(config: EncodingConfig, format: str = 'yaml') -> Path:
    """Save a config to the configs directory."""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    if format == 'yaml':
        path = config_dir / f"{config.name}.yaml"
        config.to_yaml(path)
    else:
        path = config_dir / f"{config.name}.json"
        config.to_json(path)

    return path


# =============================================================================
# Preset Configs
# =============================================================================

# Model short names for config naming
MODEL_SHORT_NAMES = {
    'thenlper/gte-small': 'gte-small',
    'sentence-transformers/all-MiniLM-L6-v2': 'minilm',
    'BAAI/bge-base-en-v1.5': 'bge-base',
}

# Common model definitions
_GTE_SMALL = ModelConfig(name='thenlper/gte-small', dim=384)
_MINILM = ModelConfig(name='sentence-transformers/all-MiniLM-L6-v2', dim=384)
_BGE_BASE = ModelConfig(name='BAAI/bge-base-en-v1.5', dim=768)


def make_config_name(model: ModelConfig, method: str) -> str:
    """Generate canonical config name: {model}_{components}"""
    short_name = MODEL_SHORT_NAMES.get(model.name, model.name.split('/')[-1])
    return f"{short_name}_{method}"


PRESET_CONFIGS = {
    # =========================================================================
    # NEW CANONICAL NAMES: {model}_{components}
    # =========================================================================

    # === c0r: Mean pooling equivalent ===
    'gte-small_c0r': EncodingConfig(
        name='gte-small_c0r',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=0, method='c0r'),
        description='GTE-small c0r: semantic mean only (384 dims)'
    ),
    'minilm_c0r': EncodingConfig(
        name='minilm_c0r',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=0, method='c0r'),
        description='MiniLM c0r: semantic mean only (384 dims)'
    ),

    # === c0r.c0a: Mean + direction ===
    'gte-small_c0r.c0a': EncodingConfig(
        name='gte-small_c0r.c0a',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=0, method='c0r.c0a'),
        description='GTE-small c0r.c0a: mean + direction (576 dims)'
    ),
    'minilm_c0r.c0a': EncodingConfig(
        name='minilm_c0r.c0a',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=0, method='c0r.c0a'),
        description='MiniLM c0r.c0a: mean + direction (576 dims)'
    ),

    # === c0r.c0a.c0m: Full c0 decomposition ===
    'gte-small_c0r.c0a.c0m': EncodingConfig(
        name='gte-small_c0r.c0a.c0m',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=0, method='c0r.c0a.c0m'),
        description='GTE-small c0r.c0a.c0m: full c0 (768 dims)'
    ),
    'minilm_c0r.c0a.c0m': EncodingConfig(
        name='minilm_c0r.c0a.c0m',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=0, method='c0r.c0a.c0m'),
        description='MiniLM c0r.c0a.c0m: full c0 (768 dims)'
    ),

    # === c0r.c0a.c1a: Mean + direction + linear trend ===
    'gte-small_c0r.c0a.c1a': EncodingConfig(
        name='gte-small_c0r.c0a.c1a',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=1, method='c0r.c0a.c1a'),
        description='GTE-small c0r.c0a.c1a: mean + linear (768 dims)'
    ),
    'minilm_c0r.c0a.c1a': EncodingConfig(
        name='minilm_c0r.c0a.c1a',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=1, method='c0r.c0a.c1a'),
        description='MiniLM c0r.c0a.c1a: mean + linear (768 dims)'
    ),

    # === c0r.c0a.c1a.c2a: Mean + direction + linear + curvature ===
    'gte-small_c0r.c0a.c1a.c2a': EncodingConfig(
        name='gte-small_c0r.c0a.c1a.c2a',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=2, method='c0r.c0a.c1a.c2a'),
        description='GTE-small c0r.c0a.c1a.c2a: mean + linear + curvature (960 dims)'
    ),
    'minilm_c0r.c0a.c1a.c2a': EncodingConfig(
        name='minilm_c0r.c0a.c1a.c2a',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=2, method='c0r.c0a.c1a.c2a'),
        description='MiniLM c0r.c0a.c1a.c2a: mean + linear + curvature (960 dims)'
    ),
    'bge-base_c0r.c0a.c1a.c2a': EncodingConfig(
        name='bge-base_c0r.c0a.c1a.c2a',
        model=_BGE_BASE,
        chebyshev=ChebyshevConfig(degree=2, method='c0r.c0a.c1a.c2a'),
        description='BGE-base c0r.c0a.c1a.c2a: mean + linear + curvature (1920 dims)'
    ),

    # === c0r.c0a.c0m.c1a.c2a: Full c0 + positional ===
    'gte-small_c0r.c0a.c0m.c1a.c2a': EncodingConfig(
        name='gte-small_c0r.c0a.c0m.c1a.c2a',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=2, method='c0r.c0a.c0m.c1a.c2a'),
        description='GTE-small c0r.c0a.c0m.c1a.c2a: full c0 + positional (1152 dims)'
    ),
    'minilm_c0r.c0a.c0m.c1a.c2a': EncodingConfig(
        name='minilm_c0r.c0a.c0m.c1a.c2a',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=2, method='c0r.c0a.c0m.c1a.c2a'),
        description='MiniLM c0r.c0a.c0m.c1a.c2a: full c0 + positional (1152 dims)'
    ),

    # =========================================================================
    # LEGACY ALIASES: Old names → New canonical names
    # These allow old code to continue working during transition
    # =========================================================================

    # c0-only → c0r
    'gte-small-c0': EncodingConfig(
        name='gte-small-c0',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=0, method='c0-only'),
        description='[LEGACY → gte-small_c0r] semantic mean (384 dims)'
    ),
    'minilm-c0': EncodingConfig(
        name='minilm-c0',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=0, method='c0-only'),
        description='[LEGACY → minilm_c0r] semantic mean (384 dims)'
    ),

    # dual-d1 → c0r.c0a.c1a
    'gte-small-dual-d1': EncodingConfig(
        name='gte-small-dual-d1',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=1, method='dual-d1'),
        description='[LEGACY → gte-small_c0r.c0a.c1a] (768 dims)'
    ),
    'minilm-dual-d1': EncodingConfig(
        name='minilm-dual-d1',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=1, method='dual-d1'),
        description='[LEGACY → minilm_c0r.c0a.c1a] (768 dims)'
    ),

    # dual-d2 → c0r.c0a.c1a.c2a
    'gte-small-dual-d2': EncodingConfig(
        name='gte-small-dual-d2',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=2, method='dual-d2'),
        description='[LEGACY → gte-small_c0r.c0a.c1a.c2a] (960 dims)'
    ),
    'minilm-dual-d2': EncodingConfig(
        name='minilm-dual-d2',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=2, method='dual-d2'),
        description='[LEGACY → minilm_c0r.c0a.c1a.c2a] (960 dims)'
    ),
    'bge-base-dual-d2': EncodingConfig(
        name='bge-base-dual-d2',
        model=_BGE_BASE,
        chebyshev=ChebyshevConfig(degree=2, method='dual-d2'),
        description='[LEGACY → bge-base_c0r.c0a.c1a.c2a] (1920 dims)'
    ),

    # Other legacy configs
    'gte-small-c0-dual': EncodingConfig(
        name='gte-small-c0-dual',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=0, method='c0-dual'),
        description='[LEGACY → gte-small_c0r.c0a] (576 dims)'
    ),
    'minilm-c0-dual': EncodingConfig(
        name='minilm-c0-dual',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=0, method='c0-dual'),
        description='[LEGACY → minilm_c0r.c0a] (576 dims)'
    ),
    'gte-small-c0-full': EncodingConfig(
        name='gte-small-c0-full',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=0, method='c0-full'),
        description='[LEGACY → gte-small_c0r.c0a.c0m] (768 dims)'
    ),
    'minilm-c0-full': EncodingConfig(
        name='minilm-c0-full',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=0, method='c0-full'),
        description='[LEGACY → minilm_c0r.c0a.c0m] (768 dims)'
    ),
    'gte-small-dual-d2-mag': EncodingConfig(
        name='gte-small-dual-d2-mag',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=2, method='dual-d2-mag'),
        description='[LEGACY → gte-small_c0r.c0a.c0m.c1a.c2a] (1152 dims)'
    ),
    'minilm-dual-d2-mag': EncodingConfig(
        name='minilm-dual-d2-mag',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=2, method='dual-d2-mag'),
        description='[LEGACY → minilm_c0r.c0a.c0m.c1a.c2a] (1152 dims)'
    ),

    # Variable degree legacy configs
    'gte-small-d5-raw': EncodingConfig(
        name='gte-small-d5-raw',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=5, method='raw'),
        description='[LEGACY] raw d=5 (2304 dims)'
    ),
    'minilm-d5-raw': EncodingConfig(
        name='minilm-d5-raw',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=5, method='raw'),
        description='[LEGACY] raw d=5 (2304 dims)'
    ),
    'gte-small-d5-hybrid': EncodingConfig(
        name='gte-small-d5-hybrid',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=5, method='hybrid'),
        description='[LEGACY] hybrid d=5 (1344 dims)'
    ),
    'gte-small-d5-angular': EncodingConfig(
        name='gte-small-d5-angular',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=5, method='angular'),
        description='[LEGACY] angular d=5 (1152 dims)'
    ),
    'gte-small-d3-raw': EncodingConfig(
        name='gte-small-d3-raw',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=3, method='raw'),
        description='[LEGACY] raw d=3 (1536 dims)'
    ),
    'gte-small-d7-raw': EncodingConfig(
        name='gte-small-d7-raw',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=7, method='raw'),
        description='[LEGACY] raw d=7 (3072 dims)'
    ),
    'gte-small-d5': EncodingConfig(
        name='gte-small-d5',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=5, method='raw'),
        description='[LEGACY] alias for gte-small-d5-raw'
    ),
    'minilm-d5': EncodingConfig(
        name='minilm-d5',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=5, method='raw'),
        description='[LEGACY] alias for minilm-d5-raw'
    ),

    # =========================================================================
    # PROCESSING MODIFIERS: +rope (principled angular), +random (ablation)
    # =========================================================================

    # === +rope: RoPE pre-rotation for principled angular extraction ===
    'minilm_c0r.c0a.c1a.c2a+rope': EncodingConfig(
        name='minilm_c0r.c0a.c1a.c2a+rope',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=2, method='c0r.c0a.c1a.c2a'),
        processing=ProcessingConfig(rope=True),
        description='MiniLM c0r.c0a.c1a.c2a with RoPE pre-rotation (960 dims, principled angular)'
    ),
    'gte-small_c0r.c0a.c1a.c2a+rope': EncodingConfig(
        name='gte-small_c0r.c0a.c1a.c2a+rope',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=2, method='c0r.c0a.c1a.c2a'),
        processing=ProcessingConfig(rope=True),
        description='GTE-small c0r.c0a.c1a.c2a with RoPE pre-rotation (960 dims, principled angular)'
    ),
    'bge-base_c0r.c0a.c1a.c2a+rope': EncodingConfig(
        name='bge-base_c0r.c0a.c1a.c2a+rope',
        model=_BGE_BASE,
        chebyshev=ChebyshevConfig(degree=2, method='c0r.c0a.c1a.c2a'),
        processing=ProcessingConfig(rope=True),
        description='BGE-base c0r.c0a.c1a.c2a with RoPE pre-rotation (1920 dims, principled angular)'
    ),

    # === +random: Random rotation baseline for ablation ===
    'minilm_c0r.c0a.c1a.c2a+random': EncodingConfig(
        name='minilm_c0r.c0a.c1a.c2a+random',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=2, method='c0r.c0a.c1a.c2a'),
        processing=ProcessingConfig(random_rotation=True),
        description='MiniLM c0r.c0a.c1a.c2a with random rotation (960 dims, ablation baseline)'
    ),
    'gte-small_c0r.c0a.c1a.c2a+random': EncodingConfig(
        name='gte-small_c0r.c0a.c1a.c2a+random',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=2, method='c0r.c0a.c1a.c2a'),
        processing=ProcessingConfig(random_rotation=True),
        description='GTE-small c0r.c0a.c1a.c2a with random rotation (960 dims, ablation baseline)'
    ),

    # =========================================================================
    # FOURIER MAGNITUDE CONFIGS
    # f_k = √(d_k² + s_k²) - phase-invariant, optimal for word-order detection
    # =========================================================================
    'minilm_f0.f1': EncodingConfig(
        name='minilm_f0.f1',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=1, method='f0.f1'),
        description='MiniLM Fourier f0+f1: semantic + linear (768 dims)'
    ),
    'minilm_f0.f1.f2': EncodingConfig(
        name='minilm_f0.f1.f2',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=2, method='f0.f1.f2'),
        description='MiniLM Fourier f0+f1+f2: semantic + linear + curvature (1152 dims)'
    ),
    'minilm_f1': EncodingConfig(
        name='minilm_f1',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=1, method='f1'),
        description='MiniLM Fourier f1 only: phase-invariant linear (384 dims)'
    ),
    'gte-small_f0.f1': EncodingConfig(
        name='gte-small_f0.f1',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=1, method='f0.f1'),
        description='GTE-small Fourier f0+f1: semantic + linear (768 dims)'
    ),
    'gte-small_f0.f1.f2': EncodingConfig(
        name='gte-small_f0.f1.f2',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=2, method='f0.f1.f2'),
        description='GTE-small Fourier f0+f1+f2: semantic + linear + curvature (1152 dims)'
    ),

    # === Hybrid: Chebyshev semantic + Fourier positional ===
    'minilm_c0r.f1': EncodingConfig(
        name='minilm_c0r.f1',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=1, method='c0r.f1'),
        description='MiniLM hybrid: Chebyshev c0r + Fourier f1 (768 dims)'
    ),
    'minilm_c0r.f1.f2': EncodingConfig(
        name='minilm_c0r.f1.f2',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=2, method='c0r.f1.f2'),
        description='MiniLM hybrid: Chebyshev c0r + Fourier f1+f2 (1152 dims)'
    ),
    'gte-small_c0r.f1': EncodingConfig(
        name='gte-small_c0r.f1',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=1, method='c0r.f1'),
        description='GTE-small hybrid: Chebyshev c0r + Fourier f1 (768 dims)'
    ),

    # === Hybrid: DCT semantic + Fourier positional ===
    'minilm_d0.f1': EncodingConfig(
        name='minilm_d0.f1',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=1, method='d0.f1'),
        description='MiniLM hybrid: DCT d0 (true mean) + Fourier f1 (768 dims)'
    ),
    'minilm_d0.f1.f2': EncodingConfig(
        name='minilm_d0.f1.f2',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=2, method='d0.f1.f2'),
        description='MiniLM hybrid: DCT d0 (true mean) + Fourier f1+f2 (1152 dims)'
    ),
    'gte-small_d0.f1': EncodingConfig(
        name='gte-small_d0.f1',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=1, method='d0.f1'),
        description='GTE-small hybrid: DCT d0 (true mean) + Fourier f1 (768 dims)'
    ),
    'gte-small_d0.f1.f2': EncodingConfig(
        name='gte-small_d0.f1.f2',
        model=_GTE_SMALL,
        chebyshev=ChebyshevConfig(degree=2, method='d0.f1.f2'),
        description='GTE-small hybrid: DCT d0 (true mean) + Fourier f1+f2 (1152 dims)'
    ),

    # =========================================================================
    # DCT CONFIGS (Discrete Cosine Transform)
    # =========================================================================
    'minilm_d0.d1': EncodingConfig(
        name='minilm_d0.d1',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=1, method='d0.d1'),
        description='MiniLM DCT d0+d1: DC + fundamental (768 dims)'
    ),
    'minilm_d0.d1.d2': EncodingConfig(
        name='minilm_d0.d1.d2',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=2, method='d0.d1.d2'),
        description='MiniLM DCT d0+d1+d2: DC + fundamental + second harmonic (1152 dims)'
    ),

    # =========================================================================
    # GRAM POLYNOMIAL CONFIGS (Orthogonal at uniform points)
    # =========================================================================
    'minilm_g0.g1': EncodingConfig(
        name='minilm_g0.g1',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=1, method='g0.g1'),
        description='MiniLM Gram g0+g1: truly orthogonal mean + linear (768 dims)'
    ),
    'minilm_g0.g1.g2': EncodingConfig(
        name='minilm_g0.g1.g2',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=2, method='g0.g1.g2'),
        description='MiniLM Gram g0+g1+g2: truly orthogonal mean + linear + curvature (1152 dims)'
    ),

    # =========================================================================
    # FULL DEGREE-6 CONFIGS (d0-d6, g0-g6, f0-f6)
    # =========================================================================
    'minilm_d0.d1.d2.d3.d4.d5.d6': EncodingConfig(
        name='minilm_d0.d1.d2.d3.d4.d5.d6',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=6, method='d0.d1.d2.d3.d4.d5.d6'),
        description='MiniLM DCT d0-d6: full 7-coefficient DCT (2688 dims)'
    ),
    'minilm_g0.g1.g2.g3.g4.g5.g6': EncodingConfig(
        name='minilm_g0.g1.g2.g3.g4.g5.g6',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=6, method='g0.g1.g2.g3.g4.g5.g6'),
        description='MiniLM Gram g0-g6: full 7-coefficient Gram (2688 dims)'
    ),
    'minilm_f0.f1.f2.f3.f4.f5.f6': EncodingConfig(
        name='minilm_f0.f1.f2.f3.f4.f5.f6',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=6, method='f0.f1.f2.f3.f4.f5.f6'),
        description='MiniLM Fourier f0-f6: full 7-coefficient Fourier magnitude (2688 dims)'
    ),

    # =========================================================================
    # CIRCULAR FOURIER CONFIGS (h = circular polar encoding)
    # Each h_k = [angle_cos_k, angle_sin_k, magnitude_k] = 3 * (d/2) dims
    # =========================================================================
    'minilm_h0.h1.h2.h3.h4.h5.h6': EncodingConfig(
        name='minilm_h0.h1.h2.h3.h4.h5.h6',
        model=_MINILM,
        chebyshev=ChebyshevConfig(degree=6, method='h0.h1.h2.h3.h4.h5.h6'),
        description='MiniLM Circular Fourier h0-h6: 7-harmonic polar encoding (4032 dims)'
    ),
}


def init_preset_configs(overwrite: bool = False) -> List[Path]:
    """Initialize preset config files."""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    created = []
    for name, config in PRESET_CONFIGS.items():
        path = config_dir / f"{name}.yaml"
        if not path.exists() or overwrite:
            config.to_yaml(path)
            created.append(path)

    return created
