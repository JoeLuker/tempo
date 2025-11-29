"""KV cache that evicts by attention importance and triggers consolidation."""

import torch
import logging
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass

from .importance_tracker import ImportanceTracker, EvictionCandidate

logger = logging.getLogger(__name__)


@dataclass
class CachedToken:
    """Data for a single cached token across all layers."""
    position: int
    input_embedding: torch.Tensor  # The input that produced this token
    keys: List[torch.Tensor]  # Key per layer
    values: List[torch.Tensor]  # Value per layer


class ConsolidatingCache:
    """
    KV cache that evicts based on attention importance, not recency.

    When full:
    1. Find token with lowest cumulative attention importance
    2. Call consolidation callback (for Hebbian updates)
    3. Evict the token

    Stores input embeddings alongside KV pairs so we can compute
    outer(key, input) for the Hebbian update.
    """

    def __init__(
        self,
        max_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        decay: float = 0.99,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
        on_eviction: Optional[Callable] = None,
    ):
        """
        Args:
            max_size: Maximum tokens in cache
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
            decay: Attention importance decay per step
            device: Device for tensors
            dtype: Data type for tensors
            on_eviction: Callback(position, keys, values, input_emb, importance)
                        Called before eviction for Hebbian update
        """
        self.max_size = max_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.on_eviction = on_eviction

        # Importance tracker
        self.importance = ImportanceTracker(max_size, decay=decay, device=device)

        # KV storage: [num_layers, batch=1, num_heads, max_size, head_dim]
        self.key_cache = torch.zeros(
            num_layers, 1, num_heads, max_size, head_dim,
            device=device, dtype=dtype
        )
        self.value_cache = torch.zeros(
            num_layers, 1, num_heads, max_size, head_dim,
            device=device, dtype=dtype
        )

        # Input embeddings that produced each position [max_size, hidden_dim]
        # We'll set hidden_dim on first use
        self.input_embeddings: Optional[torch.Tensor] = None
        self.hidden_dim: Optional[int] = None

        # Track which positions are filled
        self.filled = torch.zeros(max_size, dtype=torch.bool, device=device)
        self.current_size = 0

        # Statistics
        self.total_evictions = 0
        self.eviction_ages = []  # Track age at eviction

    def add_token(
        self,
        position: int,
        keys: List[torch.Tensor],
        values: List[torch.Tensor],
        input_embedding: torch.Tensor,
    ) -> Optional[EvictionCandidate]:
        """
        Add a token to the cache.

        Args:
            position: Position in sequence
            keys: Key tensors per layer [num_heads, head_dim]
            values: Value tensors per layer [num_heads, head_dim]
            input_embedding: The input that produced this token [hidden_dim]

        Returns:
            EvictionCandidate if eviction occurred, None otherwise
        """
        evicted = None

        # Initialize input embedding storage if needed
        if self.input_embeddings is None:
            self.hidden_dim = input_embedding.size(-1)
            self.input_embeddings = torch.zeros(
                self.max_size, self.hidden_dim,
                device=self.device, dtype=self.dtype
            )

        # Check if we need to evict
        evicted_slot = None
        if self.current_size >= self.max_size:
            evicted = self._evict_one()
            if evicted:
                evicted_slot = evicted.position

        # Find slot: use evicted slot, or position if it fits, or find any free slot
        if evicted_slot is not None:
            slot = evicted_slot
        elif position < self.max_size and not self.filled[position]:
            slot = position
        else:
            slot = self._find_free_slot()

        if slot is None:
            raise RuntimeError("No free slot available after eviction")

        # Track if this is a new slot
        was_empty = not self.filled[slot]

        # Store KV pairs
        for layer_idx, (k, v) in enumerate(zip(keys, values)):
            self.key_cache[layer_idx, 0, :, slot, :] = k.to(self.dtype)
            self.value_cache[layer_idx, 0, :, slot, :] = v.to(self.dtype)

        # Store input embedding
        self.input_embeddings[slot] = input_embedding.flatten()[:self.hidden_dim].to(self.dtype)

        # Mark filled and update count
        self.filled[slot] = True
        if was_empty:
            self.current_size += 1

        # Register with importance tracker
        self.importance.add_token(slot)

        # Return eviction info if one occurred
        if evicted_slot is not None:
            return EvictionCandidate(
                position=evicted_slot,
                importance=evicted.importance if evicted else 0.0,
                age=evicted.age if evicted else 0
            )
        return None

    def _find_free_slot(self) -> Optional[int]:
        """Find first unfilled slot."""
        unfilled = (~self.filled).nonzero(as_tuple=True)[0]
        if len(unfilled) == 0:
            return None
        return unfilled[0].item()

    def _evict_one(self) -> Optional[EvictionCandidate]:
        """Evict the least important token."""
        candidate = self.importance.get_eviction_candidate()

        if candidate is None:
            logger.warning("No eviction candidate found (all protected?)")
            return None

        pos = candidate.position

        # Gather data for consolidation
        if self.on_eviction is not None:
            keys = [self.key_cache[l, 0, :, pos, :].clone() for l in range(self.num_layers)]
            values = [self.value_cache[l, 0, :, pos, :].clone() for l in range(self.num_layers)]
            input_emb = self.input_embeddings[pos].clone()

            # Call consolidation callback
            self.on_eviction(
                position=pos,
                keys=keys,
                values=values,
                input_embedding=input_emb,
                importance=candidate.importance
            )

        # Actually evict
        self.importance.evict(pos)
        self.filled[pos] = False
        self.current_size -= 1

        # Track stats
        self.total_evictions += 1
        self.eviction_ages.append(candidate.age)

        logger.debug(
            f"Evicted position {pos}: importance={candidate.importance:.4f}, "
            f"age={candidate.age}"
        )

        return candidate

    def update_attention(self, attention_weights: torch.Tensor) -> None:
        """
        Update importance scores from attention weights.

        Args:
            attention_weights: [batch, heads, queries, keys]
        """
        self.importance.update(attention_weights)

    def protect_range(self, start: int, end: int) -> None:
        """Protect positions from eviction (e.g., prompt tokens)."""
        self.importance.protect_positions(range(start, end))

    def get_kv_for_attention(
        self,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get key/value tensors for attention computation.

        Returns only filled positions, maintaining order.

        Returns:
            (keys, values) each of shape [batch, heads, filled_len, head_dim]
        """
        filled_positions = self.filled.nonzero(as_tuple=True)[0]

        if len(filled_positions) == 0:
            return (
                torch.zeros(1, self.num_heads, 0, self.head_dim, device=self.device, dtype=self.dtype),
                torch.zeros(1, self.num_heads, 0, self.head_dim, device=self.device, dtype=self.dtype)
            )

        keys = self.key_cache[layer_idx, :, :, filled_positions, :]
        values = self.value_cache[layer_idx, :, :, filled_positions, :]

        return keys, values

    def get_stats(self) -> dict:
        """Get cache statistics."""
        importance_stats = self.importance.get_importance_stats()
        return {
            "current_size": self.current_size,
            "max_size": self.max_size,
            "total_evictions": self.total_evictions,
            "mean_eviction_age": sum(self.eviction_ages) / len(self.eviction_ages) if self.eviction_ages else 0,
            "importance": importance_stats,
        }

    def clear(self) -> None:
        """Clear the cache."""
        self.key_cache.zero_()
        self.value_cache.zero_()
        if self.input_embeddings is not None:
            self.input_embeddings.zero_()
        self.filled.zero_()
        self.current_size = 0
        self.importance = ImportanceTracker(
            self.max_size,
            decay=self.importance.decay,
            device=self.device
        )
