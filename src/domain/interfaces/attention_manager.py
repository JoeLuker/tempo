"""Attention manager interface for controlling parallel token visibility."""

from typing import Protocol, Optional
import torch


class AttentionManagerInterface(Protocol):
    """Interface for attention management during parallel generation."""

    def initialize(self, prompt_length: int) -> None:
        """Initialize the attention manager for a new generation session.

        Args:
            prompt_length: Length of the prompt in tokens
        """
        ...

    def register_parallel_set(self, start_idx: int, end_idx: int) -> None:
        """Register a new parallel token set.

        Args:
            start_idx: Starting physical index (inclusive)
            end_idx: Ending physical index (inclusive)
        """
        ...

    def build_attention_mask(
        self,
        seq_length: int,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Build attention mask for current sequence.

        Args:
            seq_length: Current sequence length
            dtype: Data type for the mask

        Returns:
            Attention mask tensor [seq_len, seq_len]
        """
        ...

    def reset(self) -> None:
        """Reset the attention manager state."""
        ...
