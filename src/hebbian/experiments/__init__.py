"""Experimental framework for Hebbian consolidation research.

Enforces statistical rigor:
- Pre-registered hypotheses
- Power analysis for sample sizing
- Proper null hypothesis testing
- Effect sizes with confidence intervals
- Reproducible experiment configs
"""

from .framework import Experiment, ExperimentResult, ExperimentSuite
from .metrics import compute_effect_size, compute_confidence_interval, power_analysis

__all__ = [
    'Experiment',
    'ExperimentResult',
    'ExperimentSuite',
    'compute_effect_size',
    'compute_confidence_interval',
    'power_analysis',
]
