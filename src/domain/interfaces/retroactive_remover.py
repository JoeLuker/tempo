"""Retroactive remover interface for pruning parallel tokens."""

from typing import Protocol, Dict, List, Tuple, Optional
import torch


class RetroactiveRemoverInterface(Protocol):
    """Interface for retroactive pruning of parallel tokens based on attention."""

    def should_remove_tokens(
        self,
        step: int,
        target_step: int,
        token_ids: List[int],
        attention_scores: torch.Tensor
    ) -> List[int]:
        """Determine which tokens should be removed based on attention.

        Args:
            step: Current generation step
            target_step: Step containing tokens to evaluate
            token_ids: Token IDs to evaluate
            attention_scores: Attention scores from future tokens

        Returns:
            List of token IDs that should be removed
        """
        ...

    def get_threshold(self, step: int, max_steps: int) -> float:
        """Get the pruning threshold for a given step.

        Args:
            step: Current generation step
            max_steps: Maximum number of steps

        Returns:
            Threshold value for pruning
        """
        ...
