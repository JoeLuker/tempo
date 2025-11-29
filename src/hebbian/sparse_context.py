"""
Sparse context management - positions have holes where tokens were evicted.

Key insight: RoPE positions are just numbers. They don't need to be contiguous.
When we evict token at position 5, positions become [0,1,2,3,4,_,6,7,8...]
The model sees gaps. New tokens append at the end.

Benefits:
- No KV cache recomputation
- Positions are stable identities
- Attention mask handles the gaps
- Cleaner Hebbian tracking
"""

import torch
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TokenSlot:
    """A slot in the sparse context."""
    position: int
    token_id: int
    importance: float = 0.0
    step_added: int = 0


class SparseContext:
    """
    Manages a context window with sparse positions.

    Evicted positions become holes - not filled until we wrap around.
    New tokens always append at next_position.
    Attention mask excludes empty slots.
    """

    def __init__(
        self,
        max_positions: int,
        decay: float = 0.99,
        device: str = "cpu",
    ):
        self.max_positions = max_positions
        self.decay = decay
        self.device = device

        # Position -> TokenSlot (only filled positions)
        self.slots: Dict[int, TokenSlot] = {}

        # Set of positions that are empty (were evicted)
        self.empty_positions: Set[int] = set()

        # Protected positions (can't be evicted)
        self.protected: Set[int] = set()

        # Next position for new tokens
        self.next_position = 0

        # Current step (for age tracking)
        self.current_step = 0

        # Eviction log
        self.evictions: List[dict] = []

    def add_token(self, token_id: int) -> int:
        """
        Add a new token at the next position.

        Returns:
            The position assigned to this token
        """
        position = self.next_position
        self.next_position += 1

        self.slots[position] = TokenSlot(
            position=position,
            token_id=token_id,
            importance=0.0,
            step_added=self.current_step,
        )

        return position

    def protect(self, positions: range) -> None:
        """Mark positions as protected from eviction."""
        self.protected.update(positions)

    def update_importance(self, attention_weights: torch.Tensor) -> None:
        """
        Update importance from attention weights.

        Args:
            attention_weights: (batch, heads, seq_len, seq_len)
                              where seq_len includes ALL positions (including empty)
        """
        self.current_step += 1

        # Decay existing importance
        for slot in self.slots.values():
            slot.importance *= self.decay

        # attention[:, :, q, k] = attention from query q to key k
        # Incoming attention to position k = sum over all q
        incoming = attention_weights.sum(dim=2).mean(dim=(0, 1))  # (seq_len,)

        for pos, slot in self.slots.items():
            if pos < incoming.size(0):
                slot.importance += incoming[pos].item()

    def get_eviction_candidate(self) -> Optional[int]:
        """
        Get position with lowest importance that isn't protected.

        Returns:
            Position to evict, or None if all protected
        """
        candidates = [
            (pos, slot.importance)
            for pos, slot in self.slots.items()
            if pos not in self.protected
        ]

        if not candidates:
            return None

        return min(candidates, key=lambda x: x[1])[0]

    def evict(self, position: int) -> Optional[TokenSlot]:
        """
        Evict token at position, leaving a hole.

        Returns:
            The evicted TokenSlot, or None if position was empty
        """
        if position not in self.slots:
            return None

        slot = self.slots.pop(position)
        self.empty_positions.add(position)

        self.evictions.append({
            "position": position,
            "token_id": slot.token_id,
            "importance": slot.importance,
            "age": self.current_step - slot.step_added,
        })

        logger.debug(
            f"Evicted pos={position}, token={slot.token_id}, "
            f"importance={slot.importance:.4f}"
        )

        return slot

    def get_filled_positions(self) -> List[int]:
        """Get sorted list of filled positions."""
        return sorted(self.slots.keys())

    def get_position_ids(self) -> torch.Tensor:
        """
        Get position IDs tensor for the model.

        Returns positions for filled slots only, in order.
        The model will see gaps in the position sequence.
        """
        positions = self.get_filled_positions()
        return torch.tensor(positions, device=self.device, dtype=torch.long)

    def get_attention_mask(self) -> torch.Tensor:
        """
        Get attention mask that excludes empty positions.

        Returns:
            (1, 1, seq_len, seq_len) causal mask with empty positions masked
        """
        positions = self.get_filled_positions()
        seq_len = len(positions)

        # Start with causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))

        # The mask is already correct because we only include filled positions
        # in the sequence passed to the model

        return mask.unsqueeze(0).unsqueeze(0)

    def num_filled(self) -> int:
        """Number of filled positions."""
        return len(self.slots)

    def get_stats(self) -> dict:
        """Get context statistics."""
        if not self.slots:
            return {
                "filled": 0,
                "empty": len(self.empty_positions),
                "next_position": self.next_position,
                "importance_mean": 0,
                "importance_max": 0,
            }

        importances = [s.importance for s in self.slots.values()]
        return {
            "filled": len(self.slots),
            "empty": len(self.empty_positions),
            "next_position": self.next_position,
            "protected": len(self.protected),
            "importance_mean": sum(importances) / len(importances),
            "importance_max": max(importances),
            "importance_min": min(importances),
        }


class SparseKVCache:
    """
    KV cache that supports sparse positions with holes.

    Instead of a contiguous tensor, we store KV pairs keyed by position.
    When building tensors for attention, we only include filled positions.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # layer_idx -> {position -> (key, value)}
        # key/value shape: (num_kv_heads, head_dim)
        self.cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {
            i: {} for i in range(num_layers)
        }

        # Also store the input hidden states for Hebbian updates
        # position -> hidden_state (hidden_dim,)
        self.inputs: Dict[int, torch.Tensor] = {}

    def store(
        self,
        position: int,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        input_hidden: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Store KV pair for a position.

        Args:
            position: Token position
            layer_idx: Transformer layer
            key: Key tensor (num_kv_heads, head_dim) or (kv_dim,)
            value: Value tensor (num_kv_heads, head_dim) or (kv_dim,)
            input_hidden: Input that produced this KV (for Hebbian)
        """
        # Reshape if needed
        if key.dim() == 1:
            key = key.view(self.num_kv_heads, self.head_dim)
        if value.dim() == 1:
            value = value.view(self.num_kv_heads, self.head_dim)

        self.cache[layer_idx][position] = (
            key.to(self.device, self.dtype),
            value.to(self.device, self.dtype),
        )

        if input_hidden is not None and layer_idx == 0:
            # Only store input once (same for all layers)
            self.inputs[position] = input_hidden.to(self.device, self.dtype)

    def get_kv_tensors(
        self,
        layer_idx: int,
        positions: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get KV tensors for specific positions.

        Args:
            layer_idx: Which layer
            positions: Ordered list of positions to include

        Returns:
            (keys, values) each of shape (1, num_kv_heads, len(positions), head_dim)
        """
        if not positions:
            return (
                torch.zeros(1, self.num_kv_heads, 0, self.head_dim,
                           device=self.device, dtype=self.dtype),
                torch.zeros(1, self.num_kv_heads, 0, self.head_dim,
                           device=self.device, dtype=self.dtype),
            )

        keys = []
        values = []

        for pos in positions:
            if pos in self.cache[layer_idx]:
                k, v = self.cache[layer_idx][pos]
                keys.append(k)
                values.append(v)
            else:
                # Missing position - use zeros (shouldn't happen if managed correctly)
                keys.append(torch.zeros(self.num_kv_heads, self.head_dim,
                                       device=self.device, dtype=self.dtype))
                values.append(torch.zeros(self.num_kv_heads, self.head_dim,
                                         device=self.device, dtype=self.dtype))

        # Stack: list of (num_kv_heads, head_dim) -> (seq_len, num_kv_heads, head_dim)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)

        # Reshape to (1, num_kv_heads, seq_len, head_dim)
        keys = keys.permute(1, 0, 2).unsqueeze(0)
        values = values.permute(1, 0, 2).unsqueeze(0)

        return keys, values

    def get_for_hebbian(self, position: int) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Get K, V, and input for Hebbian update.

        Returns:
            Dict mapping layer_idx -> (key, value, input)
        """
        result = {}
        input_hidden = self.inputs.get(position)

        if input_hidden is None:
            return result

        for layer_idx in range(self.num_layers):
            if position in self.cache[layer_idx]:
                k, v = self.cache[layer_idx][position]
                result[layer_idx] = (k.flatten(), v.flatten(), input_hidden)

        return result

    def remove(self, position: int) -> None:
        """Remove a position from the cache."""
        for layer_idx in range(self.num_layers):
            self.cache[layer_idx].pop(position, None)
        self.inputs.pop(position, None)

    def clear(self) -> None:
        """Clear all cached data."""
        for layer_idx in range(self.num_layers):
            self.cache[layer_idx].clear()
        self.inputs.clear()
