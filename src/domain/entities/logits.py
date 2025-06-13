"""Logits value objects for the TEMPO generation system.

This module defines immutable value objects representing model logits
and their transformations.
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass(frozen=True)
class TokenLogits:
    """Value object representing raw logits from the model."""
    tensor: torch.Tensor
    sequence_position: int
    batch_index: int = 0
    
    def __post_init__(self):
        """Validate logits properties."""
        if self.tensor.dim() < 1:
            raise ValueError(f"Logits tensor must have at least 1 dimension, got {self.tensor.dim()}")
        if self.sequence_position < 0:
            raise ValueError(f"Sequence position must be non-negative, got {self.sequence_position}")
        if self.batch_index < 0:
            raise ValueError(f"Batch index must be non-negative, got {self.batch_index}")
        if torch.isnan(self.tensor).any() or torch.isinf(self.tensor).any():
            raise ValueError("Logits tensor contains NaN or Inf values")

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size from the logits tensor."""
        return self.tensor.shape[-1]

    def to_probabilities(self, temperature: float = 1.0) -> torch.Tensor:
        """Convert logits to probabilities with optional temperature scaling."""
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        scaled_logits = self.tensor / temperature
        return torch.softmax(scaled_logits, dim=-1)