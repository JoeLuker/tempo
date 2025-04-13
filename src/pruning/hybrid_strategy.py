import torch
from typing import List, Tuple, Optional
from .pruning_strategy import PruningStrategy
from .coherence_strategy import CoherencePruningStrategy
from .diversity_strategy import DiversityPruningStrategy


class HybridPruningStrategy(PruningStrategy):
    """
    Hybrid pruning strategy that combines diversity and coherence pruning.
    Uses diversity for initial steps, then switches to coherence.
    """

    def __init__(
        self,
        model,
        tokenizer,
        coherence_threshold: float = 0.3,
        num_clusters: int = 3,
        diversity_steps: int = 5,
        device: str = "mps",
    ):
        """
        Initialize the hybrid pruning strategy.

        Args:
            model: The language model
            tokenizer: HuggingFace tokenizer
            coherence_threshold: Threshold for pruning tokens based on attention coherence
            num_clusters: Number of clusters to use for diversity-optimized pruning
            diversity_steps: Number of steps to use diversity pruning before switching to coherence
            device: Device to use for computation
        """
        super().__init__(model, tokenizer, device)
        self.diversity_steps = diversity_steps
        self.current_step = 0

        # Create the component strategies
        self.coherence_strategy = CoherencePruningStrategy(
            model, tokenizer, coherence_threshold, device
        )
        self.diversity_strategy = DiversityPruningStrategy(
            model, tokenizer, num_clusters, device
        )

    def prune_tokens(
        self, input_ids: torch.Tensor, parallel_tokens: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Prune tokens using the hybrid strategy.

        Args:
            input_ids: Current input token IDs
            parallel_tokens: List of (token_id, probability) tuples

        Returns:
            List[Tuple[int, float]]: Pruned list of (token_id, probability) tuples
        """
        # Use diversity pruning for initial steps
        if self.current_step < self.diversity_steps:
            pruned_tokens = self.diversity_strategy.prune_tokens(
                input_ids, parallel_tokens
            )
        else:
            # Switch to coherence pruning after diversity_steps
            pruned_tokens = self.coherence_strategy.prune_tokens(
                input_ids, parallel_tokens
            )

        # Increment step counter
        self.current_step += 1

        return pruned_tokens

    def get_scored_tokens(
        self, input_ids: torch.Tensor, parallel_tokens: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Get tokens with their scores based on the current active strategy.

        Args:
            input_ids: Current input token IDs
            parallel_tokens: List of (token_id, probability) tuples

        Returns:
            List[Tuple[int, float]]: List of (token_id, score) tuples
        """
        # Use scoring from the appropriate strategy based on current step
        if self.current_step < self.diversity_steps:
            return self.diversity_strategy.get_scored_tokens(input_ids, parallel_tokens)
        else:
            return self.coherence_strategy.get_scored_tokens(input_ids, parallel_tokens)

    def reset(self):
        """Reset the step counter for a new generation."""
        self.current_step = 0
