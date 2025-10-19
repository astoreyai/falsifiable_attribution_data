"""
Core Falsification Framework for XAI Dissertation.

This package implements the theoretical framework from Chapter 3:
- Counterfactual generation on hyperspheres (Theorem 3.6, 3.8)
- Falsification test (Definition 3.1)
- Statistical metrics for evaluation
"""

from .counterfactual_generation import (
    generate_counterfactuals_hypersphere,
    compute_geodesic_distance,
)
from .falsification_test import (
    falsification_test,
    compute_falsification_rate,
)
from .metrics import (
    compute_separation_margin,
    compute_effect_size,
    statistical_significance_test,
)

__all__ = [
    'generate_counterfactuals_hypersphere',
    'compute_geodesic_distance',
    'falsification_test',
    'compute_falsification_rate',
    'compute_separation_margin',
    'compute_effect_size',
    'statistical_significance_test',
]
