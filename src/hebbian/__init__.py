"""Hebbian consolidation during inference - weight updates without backprop.

This module implements the core hypothesis: inference can be learning.
When tokens leave the context window, they leave Hebbian traces in the weights,
proportional to their importance (measured by attention received).

Core components:
- HebbianConfig: Configuration for consolidation behavior
- HebbianMLXEngine: MLX-based engine for Apple Silicon (primary implementation)

Theory:
- Eviction by relevance, not recency (least-attended tokens evict first)
- Weight update: ΔW = scale × importance × outer(projection, hidden)
- Attention patterns serve as credit assignment
- Context window = working memory, weights = long-term memory
"""

from .config import HebbianConfig, BenchmarkConfig, BASELINE, HEBBIAN
from .mlx import HebbianMLXEngine

__all__ = [
    # Config
    'HebbianConfig',
    'BenchmarkConfig',
    'BASELINE',
    'HEBBIAN',
    # Engine
    'HebbianMLXEngine',
]
