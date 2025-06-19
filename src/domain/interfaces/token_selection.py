"""Token selection interfaces for the TEMPO system.

This module defines interfaces for token selection strategies.
"""

from typing import Protocol, Optional
import torch
from abc import abstractmethod


class TokenSelectorInterface(Protocol):
    """Interface for token selection operations."""
    
    @abstractmethod
    def select_tokens(
        self,
        logits: torch.Tensor,
        threshold: float,
        max_tokens: Optional[int] = None
    ) -> tuple[list[tuple[torch.Tensor, float]], int]:
        """Select tokens based on a probability threshold.
        
        Args:
            logits: Raw logits from the model
            threshold: Probability threshold for selection
            max_tokens: Optional maximum number of tokens to select
            
        Returns:
            Tuple of:
            - List of (token_id, probability) tuples
            - Number of tokens that met the threshold
        """
        ...