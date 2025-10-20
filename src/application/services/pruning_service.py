"""Retroactive pruning service for refining parallel token sets.

This service uses attention patterns to retroactively prune less
coherent token choices from parallel sets.
"""

from typing import Dict, List, Tuple, Optional
import torch

from ...algorithms.pruning.attention_pruner import AttentionBasedPruner
from ...algorithms.pruning.threshold_manager import DynamicThresholdManager
from ...utils.logging_utils import LoggingMixin


class PruningService(LoggingMixin):
    """Service for retroactive pruning of parallel tokens based on attention."""

    def __init__(self,
                 attention_threshold: float = 0.01,
                 use_relative_attention: bool = True,
                 relative_threshold: float = 0.5,
                 use_multi_scale: bool = True,
                 num_layers_to_use: Optional[int] = None,
                 use_sigmoid_threshold: bool = True,
                 sigmoid_steepness: float = 10.0,
                 debug_mode: bool = False):
        """Initialize the pruning service.

        Args:
            attention_threshold: Absolute attention threshold for pruning
            use_relative_attention: Whether to use relative thresholding
            relative_threshold: Relative threshold multiplier
            use_multi_scale: Whether to use multi-scale attention analysis
            num_layers_to_use: Number of layers to analyze (None = all)
            use_sigmoid_threshold: Whether to use sigmoid-based thresholding
            sigmoid_steepness: Steepness of sigmoid function
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        self.setup_logging("pruning_service", "pruning_service.log", debug_mode)

        self.attention_threshold = attention_threshold
        self.use_multi_scale = use_multi_scale
        self.num_layers_to_use = num_layers_to_use

        # Create pruning components
        self.attention_pruner = AttentionBasedPruner(
            attention_threshold=attention_threshold,
            use_relative_attention=use_relative_attention,
            relative_threshold=relative_threshold
        )

        # Multi-scale pruning not yet implemented
        self.multi_scale_pruner = None
        self.use_sigmoid = use_sigmoid_threshold
        self.sigmoid_steepness = sigmoid_steepness

        # Track pruning statistics
        self.pruning_stats = {
            "total_pruned": 0,
            "total_evaluated": 0,
            "pruning_by_step": {}
        }

    def prune_parallel_set(
        self,
        token_set: List[Tuple[int, float]],
        logical_step: int,
        attention_weights: Optional[torch.Tensor],
        source_positions: List[int],
        target_positions: List[int]
    ) -> List[Tuple[int, float]]:
        """Prune a parallel token set based on attention patterns.

        Args:
            token_set: List of (token_id, probability) tuples
            logical_step: The logical step number
            attention_weights: Attention tensor from the model
            source_positions: Positions that attend to targets
            target_positions: Positions being evaluated

        Returns:
            Pruned list of (token_id, probability) tuples
        """
        if not token_set or attention_weights is None:
            return token_set

        if len(token_set) == 1:
            # Can't prune a single token
            return token_set

        # Compute attention scores
        scores = self.attention_pruner.compute_pruning_scores(
            attention_weights,
            target_positions,
            source_positions
        )

        # Apply pruning decision using relative pruning
        keep_mask = self.attention_pruner.apply_relative_pruning(
            scores,
            (0, len(target_positions))
        )

        # Filter token set
        pruned_set = [
            token for i, token in enumerate(token_set)
            if i < len(keep_mask) and keep_mask[i]
        ]

        # Update statistics
        num_pruned = len(token_set) - len(pruned_set)
        self.pruning_stats["total_pruned"] += num_pruned
        self.pruning_stats["total_evaluated"] += len(token_set)
        self.pruning_stats["pruning_by_step"][logical_step] = num_pruned

        if self.debug_mode:
            self.log(f"Step {logical_step}: Pruned {num_pruned}/{len(token_set)} tokens "
                    f"(kept {len(pruned_set)})")

        # Ensure at least one token survives
        if not pruned_set:
            pruned_set = [max(token_set, key=lambda x: x[1])]  # Keep highest probability
            if self.debug_mode:
                self.log(f"Step {logical_step}: All tokens pruned, keeping highest probability token")

        return pruned_set

    def should_prune_step(
        self,
        logical_step: int,
        max_steps: int,
        min_steps_before_pruning: int = 1
    ) -> bool:
        """Determine if pruning should be applied at this step.

        Args:
            logical_step: Current logical step
            max_steps: Maximum number of steps
            min_steps_before_pruning: Minimum steps before starting pruning

        Returns:
            True if pruning should be applied
        """
        # Don't prune the first few steps to allow exploration
        if logical_step < min_steps_before_pruning:
            return False

        return True

    def get_pruning_stats(self) -> Dict:
        """Get pruning statistics.

        Returns:
            Dictionary with pruning statistics
        """
        stats = self.pruning_stats.copy()

        if stats["total_evaluated"] > 0:
            stats["pruning_rate"] = stats["total_pruned"] / stats["total_evaluated"]
        else:
            stats["pruning_rate"] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset pruning statistics."""
        self.pruning_stats = {
            "total_pruned": 0,
            "total_evaluated": 0,
            "pruning_by_step": {}
        }

        if self.debug_mode:
            self.log("Reset pruning statistics")
