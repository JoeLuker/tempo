"""Attention management service for controlling parallel token visibility.

This service coordinates attention mask construction and management
to control how parallel tokens can attend to each other.
"""

from typing import List, Tuple, Optional
import torch

from ...algorithms.attention.mask_builder import AttentionMaskBuilder
from ...utils.logging_utils import LoggingMixin
from ...domain.entities.parallel_generation import LogicalPosition


class AttentionService(LoggingMixin):
    """Service for managing attention patterns during parallel generation."""

    def __init__(self,
                 isolate_parallel_tokens: bool = True,
                 device: str = "mps",
                 debug_mode: bool = False):
        """Initialize the attention service.

        Args:
            isolate_parallel_tokens: Whether to prevent parallel tokens from seeing each other
            device: Device to use for computations
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        self.setup_logging("attention_service", "attention_service.log", debug_mode)

        self.device = device
        self.isolate_parallel_tokens = isolate_parallel_tokens
        self.mask_builder = AttentionMaskBuilder(isolate_parallel_tokens=isolate_parallel_tokens)

        # Track parallel sets for mask construction
        self.parallel_sets: List[Tuple[int, int]] = []
        self.current_mask: Optional[torch.Tensor] = None

    def initialize(self, prompt_length: int):
        """Initialize the attention service for a new generation session.

        Args:
            prompt_length: Length of the prompt in tokens
        """
        self.reset()

        if self.debug_mode:
            self.log(f"Initialized attention service with prompt length {prompt_length}")
            if self.isolate_parallel_tokens:
                self.log("Parallel token isolation: ENABLED")
            else:
                self.log("Parallel tokens can attend to each other")

    def register_parallel_set(self, start_idx: int, end_idx: int) -> None:
        """Register a new parallel token set.

        Args:
            start_idx: Starting physical index (inclusive)
            end_idx: Ending physical index (inclusive)
        """
        self.parallel_sets.append((start_idx, end_idx + 1))  # Convert to exclusive end

        if self.debug_mode:
            self.log(f"Registered parallel set: positions {start_idx}-{end_idx}")

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
        device = torch.device(self.device)

        if not self.isolate_parallel_tokens:
            # Just use standard causal mask
            mask = self.mask_builder.create_causal_mask(seq_length, device, dtype)
        else:
            # Build mask with parallel token isolation
            mask = self.mask_builder.create_parallel_mask(
                seq_length,
                self.parallel_sets,
                device,
                dtype
            )

        self.current_mask = mask

        if self.debug_mode:
            num_masked = (mask < -1000).sum().item()
            total_entries = seq_length * seq_length
            self.log(f"Built attention mask: {num_masked}/{total_entries} positions masked")

        return mask

    def get_current_mask(self) -> Optional[torch.Tensor]:
        """Get the currently active attention mask.

        Returns:
            Current attention mask or None
        """
        return self.current_mask

    def update_mask_for_new_tokens(
        self,
        num_new_tokens: int,
        are_parallel: bool = False
    ) -> Optional[torch.Tensor]:
        """Update mask when new tokens are added.

        Args:
            num_new_tokens: Number of tokens being added
            are_parallel: Whether the new tokens are a parallel set

        Returns:
            Updated attention mask or None
        """
        if self.current_mask is None:
            return None

        old_len = self.current_mask.size(0)
        new_len = old_len + num_new_tokens

        # Expand mask
        device = self.current_mask.device
        dtype = self.current_mask.dtype

        new_mask = torch.zeros(new_len, new_len, dtype=dtype, device=device)
        new_mask[:old_len, :old_len] = self.current_mask

        # Apply causal masking for new positions
        for i in range(old_len, new_len):
            new_mask[i, i+1:] = -10000.0

        # If new tokens are parallel and isolated, prevent mutual attention
        if are_parallel and self.isolate_parallel_tokens and num_new_tokens > 1:
            start = old_len
            end = new_len
            for i in range(start, end):
                for j in range(start, end):
                    if i != j:
                        new_mask[i, j] = -10000.0

        self.current_mask = new_mask
        return new_mask

    def set_isolation_mode(self, enabled: bool) -> None:
        """Enable or disable parallel token isolation.

        Args:
            enabled: Whether to isolate parallel tokens
        """
        self.isolate_parallel_tokens = enabled
        self.mask_builder.isolate_parallel_tokens = enabled

        if self.debug_mode:
            mode = "enabled" if enabled else "disabled"
            self.log(f"Isolation mode {mode}")

    def reset(self) -> None:
        """Reset the attention service state."""
        self.parallel_sets.clear()
        self.current_mask = None

        if self.debug_mode:
            self.log("Reset attention service state")

    def reset_cache(self) -> None:
        """Alias for reset() for compatibility."""
        self.reset()
