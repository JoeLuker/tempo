"""Hebbian consolidation during inference - weight updates without backprop."""

from .attention_hooks import AttentionHooks, HebbianWeightUpdater
from .sparse_context import SparseContext, SparseKVCache
from .sparse_engine import SparseHebbianEngine

__all__ = [
    'AttentionHooks',
    'HebbianWeightUpdater',
    'SparseContext',
    'SparseKVCache',
    'SparseHebbianEngine',
]
