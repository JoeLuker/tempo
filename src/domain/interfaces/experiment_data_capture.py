"""Interface for experiment data capture.

This interface defines the contract for capturing data during text generation
for mechanistic interpretability experiments.
"""

from typing import Protocol, Optional, List, Any
import torch


class ExperimentDataCaptureInterface(Protocol):
    """Protocol for capturing experiment data during generation."""

    def capture_step_data(
        self,
        logical_step: int,
        physical_positions: List[int],
        token_ids: List[int],
        logits: Optional[torch.Tensor] = None,
        attention: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None
    ) -> None:
        """Capture data for a single generation step.

        Args:
            logical_step: Logical generation step
            physical_positions: Physical positions in sequence
            token_ids: Token IDs generated at this step
            logits: Full logit tensor (optional)
            attention: Attention tensor (optional)
            kv_cache: KV cache state (optional)
        """
        ...

    def save_all(self) -> None:
        """Save all captured data to disk."""
        ...
