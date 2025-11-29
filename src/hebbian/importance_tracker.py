"""Track cumulative attention importance for each token position."""

import torch
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvictionCandidate:
    """Information about a token to be evicted."""
    position: int
    importance: float
    age: int  # Steps since token entered


class ImportanceTracker:
    """
    Track cumulative attention importance per token position.

    Each token accumulates importance based on how much attention it receives.
    Importance decays over time, so tokens that stop being attended to fade.
    Eviction targets the token with lowest cumulative importance (not oldest).
    """

    def __init__(
        self,
        max_positions: int,
        decay: float = 0.99,
        device: str = "cpu"
    ):
        """
        Args:
            max_positions: Maximum sequence length to track
            decay: Decay factor applied each step (0.99 = 1% decay per step)
            device: Device for importance tensor
        """
        self.max_positions = max_positions
        self.decay = decay
        self.device = device

        # Cumulative importance per position [seq_len]
        self.importance = torch.zeros(max_positions, device=device)

        # Track when each position was filled (for age calculation)
        self.entry_step = torch.full((max_positions,), -1, device=device, dtype=torch.long)

        # Current step counter
        self.current_step = 0

        # How many positions are currently filled
        self.filled_positions = 0

        # Positions that are protected from eviction (e.g., prompt)
        self.protected_positions = set()

    def update(self, attention_weights: torch.Tensor) -> None:
        """
        Update importance scores based on attention weights.

        Args:
            attention_weights: Attention from latest step
                Shape: [batch, num_heads, num_queries, num_keys]
                We sum attention received by each key position across all queries and heads.
        """
        self.current_step += 1

        # Decay existing importance
        self.importance[:self.filled_positions] *= self.decay

        # Sum attention each key position receives
        # attention_weights: [batch, heads, queries, keys]
        # We want: sum over queries, mean over heads and batch
        incoming = attention_weights.sum(dim=2).mean(dim=(0, 1))  # [keys]

        # Add to cumulative importance
        num_keys = min(incoming.size(0), self.filled_positions)
        self.importance[:num_keys] += incoming[:num_keys].to(self.device)

        if logger.isEnabledFor(logging.DEBUG):
            top_k = min(5, self.filled_positions)
            top_positions = self.importance[:self.filled_positions].topk(top_k)
            logger.debug(
                f"Step {self.current_step}: Top importance positions: "
                f"{list(zip(top_positions.indices.tolist(), top_positions.values.tolist()))}"
            )

    def add_token(self, position: int) -> None:
        """Register a new token at position."""
        if position >= self.max_positions:
            raise ValueError(f"Position {position} exceeds max {self.max_positions}")

        self.entry_step[position] = self.current_step
        self.importance[position] = 0.0  # Start with no importance
        self.filled_positions = max(self.filled_positions, position + 1)

    def protect_positions(self, positions: range) -> None:
        """Mark positions as protected from eviction (e.g., prompt tokens)."""
        self.protected_positions.update(positions)
        logger.debug(f"Protected positions 0-{max(positions)} from eviction")

    def get_eviction_candidate(self) -> Optional[EvictionCandidate]:
        """
        Find the token with lowest importance that isn't protected.

        Returns:
            EvictionCandidate with position, importance, and age, or None if no candidates
        """
        if self.filled_positions == 0:
            return None

        # Mask protected positions with infinity
        masked_importance = self.importance[:self.filled_positions].clone()
        for pos in self.protected_positions:
            if pos < self.filled_positions:
                masked_importance[pos] = float('inf')

        # Find minimum
        min_idx = masked_importance.argmin().item()

        if masked_importance[min_idx] == float('inf'):
            # All positions are protected
            return None

        age = self.current_step - self.entry_step[min_idx].item()

        return EvictionCandidate(
            position=min_idx,
            importance=self.importance[min_idx].item(),
            age=age
        )

    def evict(self, position: int) -> float:
        """
        Mark position as evicted, return its final importance.

        Note: We don't shift positions - we leave a "hole" that will be reused.
        This matches how KV cache works with position-based updates.
        """
        importance = self.importance[position].item()
        self.importance[position] = 0.0
        self.entry_step[position] = -1

        logger.debug(
            f"Evicted position {position} with importance {importance:.4f}"
        )

        return importance

    def get_importance_stats(self) -> dict:
        """Get statistics about current importance distribution."""
        if self.filled_positions == 0:
            return {"min": 0, "max": 0, "mean": 0, "std": 0}

        active = self.importance[:self.filled_positions]
        return {
            "min": active.min().item(),
            "max": active.max().item(),
            "mean": active.mean().item(),
            "std": active.std().item() if self.filled_positions > 1 else 0,
            "filled": self.filled_positions,
            "protected": len(self.protected_positions),
        }
