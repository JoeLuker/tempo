"""MLX-based Hebbian consolidation engine.

This module provides a fast MLX implementation with memory bank
for transformers on Apple Silicon.
"""

from .engine import HebbianMLXEngine
from .cache import HebbianKVCache

__all__ = ["HebbianMLXEngine", "HebbianKVCache"]
