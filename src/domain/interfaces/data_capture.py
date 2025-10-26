"""Data capture interface for experiment tracking."""

from typing import Protocol, Dict, List, Any, Optional
import torch


class DataCaptureInterface(Protocol):
    """Interface for capturing experimental data during generation."""

    def capture_parallel_set(
        self,
        logical_step: int,
        token_ids: List[int],
        probabilities: List[float]
    ) -> None:
        """Capture a parallel token set.

        Args:
            logical_step: Logical generation step
            token_ids: List of token IDs in the parallel set
            probabilities: Corresponding probabilities
        """
        ...

    def capture_attention_weights(
        self,
        step: int,
        attention_weights: torch.Tensor
    ) -> None:
        """Capture attention weights at a generation step.

        Args:
            step: Generation step
            attention_weights: Attention weight tensor
        """
        ...

    def save_all(self) -> None:
        """Save all captured data to disk."""
        ...
