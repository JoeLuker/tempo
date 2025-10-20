"""RoPE modification service for TEMPO parallel token processing.

This service coordinates RoPE modifications to enable parallel tokens
to share the same logical position while maintaining distinct identities.
"""

from typing import Dict, List, Optional
import torch

from ...algorithms.rope.position_mapper import PositionMapper
from ...algorithms.rope.embedding_modifier import RoPECache, modify_positions_for_parallel_tokens
from ...algorithms.rope.model_patcher import ModelPatcher
from ...utils.logging_utils import LoggingMixin
from ...domain.entities.parallel_generation import LogicalPosition


class RoPEService(LoggingMixin):
    """Service for managing RoPE modifications during parallel generation."""

    def __init__(self, device: str = "mps", debug_mode: bool = False):
        """Initialize the RoPE service.

        Args:
            device: Device to use for computations
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        self.setup_logging("rope_service", "rope_service.log", debug_mode)

        self.device = device
        self.position_mapper = PositionMapper()
        self.model_patcher = ModelPatcher(device=device)
        self.rope_cache: Optional[RoPECache] = None

        # Track current position mapping
        self.current_position_map: Dict[int, int] = {}
        self.logical_layout: List[LogicalPosition] = []

    def initialize(self, prompt_length: int, model_dim: int = 128):
        """Initialize the RoPE service for a new generation session.

        Args:
            prompt_length: Length of the prompt in tokens
            model_dim: Dimension of the model (for RoPE cache)
        """
        # Clear previous state
        self.reset()

        # Initialize RoPE cache
        self.rope_cache = RoPECache(dim=model_dim)

        # Initialize position map with prompt positions (1:1 mapping)
        for i in range(prompt_length):
            self.current_position_map[i] = i

        if self.debug_mode:
            self.log(f"Initialized RoPE service with prompt length {prompt_length}")

    def update_for_parallel_tokens(
        self,
        logical_step: int,
        physical_start_idx: int,
        num_tokens: int
    ) -> None:
        """Update position mappings for a new set of parallel tokens.

        Args:
            logical_step: The logical step number
            physical_start_idx: Starting physical index for these tokens
            num_tokens: Number of parallel tokens at this step
        """
        # All parallel tokens at this step map to the same logical position
        for i in range(num_tokens):
            physical_pos = physical_start_idx + i
            self.current_position_map[physical_pos] = logical_step

        # Update position mapper
        self.position_mapper.position_map.update(self.current_position_map)

        # Track logical layout
        physical_end_idx = physical_start_idx + num_tokens - 1
        self.logical_layout.append(
            LogicalPosition(logical_step, physical_start_idx, physical_end_idx)
        )

        if self.debug_mode:
            self.log(f"Mapped {num_tokens} parallel tokens at physical positions "
                    f"{physical_start_idx}-{physical_end_idx} to logical step {logical_step}")

    def get_modified_position_ids(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Get position IDs modified for parallel token processing.

        Args:
            position_ids: Original position IDs tensor

        Returns:
            Modified position IDs with parallel tokens sharing positions
        """
        return modify_positions_for_parallel_tokens(
            position_ids,
            self.current_position_map,
            torch.device(self.device)
        )

    def get_position_map(self) -> Dict[int, int]:
        """Get the current position mapping.

        Returns:
            Dictionary mapping physical to logical positions
        """
        return self.current_position_map.copy()

    def get_logical_layout(self) -> List[LogicalPosition]:
        """Get the logical layout of parallel token sets.

        Returns:
            List of LogicalPosition objects
        """
        return self.logical_layout.copy()

    def reset(self) -> None:
        """Reset the RoPE service state."""
        self.position_mapper.clear()
        self.current_position_map.clear()
        self.logical_layout.clear()

        if self.rope_cache:
            self.rope_cache.clear()

        if self.debug_mode:
            self.log("Reset RoPE service state")

    def enable_isolation_mode(self, enabled: bool = True) -> None:
        """Enable or disable isolation mode for parallel tokens.

        In isolation mode, parallel tokens at the same logical position
        cannot attend to each other.

        Args:
            enabled: Whether to enable isolation mode
        """
        # This would coordinate with attention manager
        # For now, just log
        if self.debug_mode:
            mode = "enabled" if enabled else "disabled"
            self.log(f"Isolation mode {mode}")
