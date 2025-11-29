"""Sparse KV cache with eviction support for Hebbian consolidation.

Unlike standard KV caches that only append, this cache supports eviction
of old positions when they leave the sliding window. Evicted positions
trigger Hebbian weight updates.

Implements "attention sink" pattern from StreamingLLM - keeps initial tokens
as anchors that are never evicted. This is critical for models not trained
with sliding window attention.
"""

from dataclasses import dataclass, field
import mlx.core as mx


@dataclass
class HebbianKVCache:
    """KV cache supporting eviction for sliding window with importance tracking.

    This cache stores KV pairs per position across all layers, allowing
    O(1) eviction of specific positions. When positions are evicted,
    their KV values are returned for Hebbian consolidation.

    Implements "attention sink" pattern - keeps the first N positions as
    anchors that are never evicted. This is critical for models not trained
    with sliding window attention (like Llama, Gemma, etc.).

    Attributes:
        n_layers: Number of transformer layers
        n_kv_heads: Number of KV attention heads
        head_dim: Dimension of each attention head
        window_size: Maximum number of positions to keep in cache
        n_sink_tokens: Number of initial tokens to keep as attention sinks
    """

    n_layers: int
    n_kv_heads: int
    head_dim: int
    window_size: int = 32
    n_sink_tokens: int = 4  # Keep first 4 tokens as anchors (StreamingLLM default)

    # Internal state - initialized post-init
    # layer -> position -> (k, v) where k,v are (n_kv_heads, head_dim)
    _cache: dict[int, dict[int, tuple[mx.array, mx.array]]] = field(
        default_factory=dict, repr=False
    )
    _active_positions: set[int] = field(default_factory=set, repr=False)
    _importance: dict[int, float] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Initialize per-layer cache dictionaries."""
        self._cache = {layer: {} for layer in range(self.n_layers)}
        self._active_positions = set()
        self._importance = {}

    def add(
        self,
        layer: int,
        position: int,
        k: mx.array,
        v: mx.array,
    ) -> None:
        """Add KV pair for a position at a specific layer.

        Args:
            layer: Layer index
            position: Absolute position in sequence
            k: Key tensor of shape (n_kv_heads, head_dim)
            v: Value tensor of shape (n_kv_heads, head_dim)
        """
        self._cache[layer][position] = (k, v)
        self._active_positions.add(position)

    def evict(self, position: int) -> dict[int, tuple[mx.array, mx.array]]:
        """Remove a position from all layers and return evicted KV pairs.

        Args:
            position: Position to evict

        Returns:
            Dictionary mapping layer index to (k, v) tuple for the evicted position
        """
        evicted = {}
        for layer in range(self.n_layers):
            if position in self._cache[layer]:
                evicted[layer] = self._cache[layer].pop(position)

        self._active_positions.discard(position)
        self._importance.pop(position, None)

        return evicted

    def update_importance(self, position: int, importance: float) -> None:
        """Update the importance score for a position.

        Args:
            position: Position to update
            importance: New importance score (accumulated attention received)
        """
        self._importance[position] = importance

    def get_importance(self, position: int) -> float:
        """Get the importance score for a position."""
        return self._importance.get(position, 0.0)

    def get_kv_for_attention(
        self, layer: int
    ) -> tuple[mx.array | None, mx.array | None, list[int]]:
        """Stack all cached K,V for a layer into tensors for attention.

        Args:
            layer: Layer index

        Returns:
            keys: Stacked keys of shape (1, n_kv_heads, n_positions, head_dim) or None
            values: Stacked values of shape (1, n_kv_heads, n_positions, head_dim) or None
            positions: Sorted list of positions in the cache
        """
        layer_cache = self._cache[layer]
        if not layer_cache:
            return None, None, []

        positions = sorted(layer_cache.keys())

        # Stack K and V for all positions
        # Each entry is (n_kv_heads, head_dim)
        keys = mx.stack([layer_cache[p][0] for p in positions], axis=1)  # (n_kv_heads, n_pos, head_dim)
        values = mx.stack([layer_cache[p][1] for p in positions], axis=1)

        # Add batch dimension: (1, n_kv_heads, n_pos, head_dim)
        keys = mx.expand_dims(keys, axis=0)
        values = mx.expand_dims(values, axis=0)

        return keys, values, positions

    @property
    def active_positions(self) -> set[int]:
        """Get the set of currently active positions."""
        return self._active_positions.copy()

    @property
    def size(self) -> int:
        """Get the number of active positions."""
        return len(self._active_positions)

    def clear(self) -> None:
        """Clear all cached values."""
        for layer in range(self.n_layers):
            self._cache[layer].clear()
        self._active_positions.clear()
        self._importance.clear()

    def get_oldest_position(self) -> int | None:
        """Get the oldest evictable position in the cache.

        Returns the oldest position that is NOT a sink token.
        Sink tokens (positions 0 to n_sink_tokens-1) are never evicted.
        """
        if not self._active_positions:
            return None

        # Get positions that can be evicted (not sink tokens)
        evictable = [p for p in self._active_positions if p >= self.n_sink_tokens]
        if not evictable:
            return None

        return min(evictable)

    def should_evict(self) -> bool:
        """Check if eviction is needed based on window size."""
        return len(self._active_positions) > self.window_size
