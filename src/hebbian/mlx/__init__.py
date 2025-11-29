"""MLX-based Hebbian consolidation engine.

This module provides a fast MLX implementation of Hebbian consolidation
for transformers on Apple Silicon.
"""

from .engine import HebbianMLXEngine
from .cache import HebbianKVCache
from .modifications import HebbianModifications

__all__ = ["HebbianMLXEngine", "HebbianKVCache", "HebbianModifications"]
