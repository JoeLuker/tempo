"""Threshold-based token selector implementation.

This module implements token selection based on probability thresholds.
"""

import torch
from typing import Optional
from src.domain.interfaces.token_selection import TokenSelectorInterface
from src.utils.logging_utils import LoggingMixin


class ThresholdTokenSelector(LoggingMixin, TokenSelectorInterface):
    """Selects tokens based on probability threshold."""

    def __init__(self, debug_mode: bool = False):
        """Initialize the threshold token selector.

        Args:
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        self.setup_logging("threshold_token_selector", "token_selector.log", debug_mode)

    def select_tokens(
        self,
        logits: torch.Tensor,
        threshold: float,
        max_tokens: Optional[int] = None
    ) -> tuple[list[tuple[int, float]], int]:
        """Select tokens based on a probability threshold.

        Args:
            logits: Raw logits from the model [vocab_size]
            threshold: Probability threshold for selection
            max_tokens: Optional maximum number of tokens to select

        Returns:
            Tuple of:
            - List of (token_id, probability) tuples
            - Number of tokens that met the threshold
        """
        # Ensure logits is 1D
        if logits.dim() > 1:
            logits = logits.squeeze()

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Find tokens above threshold
        above_threshold = probs >= threshold
        num_above_threshold = above_threshold.sum().item()

        # Get indices and probabilities - use boolean indexing
        token_ids = torch.arange(len(probs), device=probs.device)[above_threshold]
        token_probs = probs[above_threshold]

        # Sort by probability (descending)
        sorted_indices = torch.argsort(token_probs, descending=True)
        token_ids = token_ids[sorted_indices]
        token_probs = token_probs[sorted_indices]

        # Apply max_tokens limit if specified
        if max_tokens is not None and len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]
            token_probs = token_probs[:max_tokens]

        # Convert to list of tuples
        selected_tokens = [
            (token_ids[i].item(), token_probs[i].item())
            for i in range(len(token_ids))
        ]

        if self.debug_mode:
            self.log(f"Selected {len(selected_tokens)} tokens above threshold {threshold} "
                    f"(total above threshold: {num_above_threshold})")

        return selected_tokens, num_above_threshold
