"""Hebbian consolidation during inference - weight updates without backprop."""

from .config import HebbianConfig, BenchmarkConfig, BASELINE, HEBBIAN
from .functional_engine import FunctionalHebbianEngine
from .minimal_engine import MinimalHebbianEngine

__all__ = [
    # Config
    'HebbianConfig',
    'BenchmarkConfig',
    'BASELINE',
    'HEBBIAN',
    # Engines
    'FunctionalHebbianEngine',
    'MinimalHebbianEngine',
]
