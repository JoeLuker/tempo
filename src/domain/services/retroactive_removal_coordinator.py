"""Coordinator for retroactive token removal operations.

This module handles coordination of retroactive removal of tokens
based on attention patterns.
"""

from typing import Any, Optional
from ...utils.logging_utils import LoggingMixin


class RetroactiveRemovalCoordinator(LoggingMixin):
    """Coordinates retroactive removal of tokens."""
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the removal coordinator.
        
        Args:
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        self.setup_logging("removal_coordinator", "removal_coordinator.log", debug_mode)
    
    def apply_retroactive_removal(
        self,
        remover: Any,
        prompt_length: int,
        all_token_sets: dict[int, list[tuple[int, float]]],
        current_step: int
    ) -> dict[int, list[tuple[int, float]]]:
        """Apply retroactive removal to historical token sets.
        
        Args:
            remover: The retroactive remover instance
            prompt_length: Length of the initial prompt
            all_token_sets: Dictionary of all token sets by step
            current_step: Current generation step
            
        Returns:
            Dictionary of surviving token sets after removal
        """
        try:
            # Update remover step if it supports it
            if hasattr(remover, 'update_step'):
                remover.update_step(current_step)
                self.log(f"Updated remover to step {current_step}")
            
            # Try new method name first, fall back to old
            if hasattr(remover, 'retroactively_remove'):
                surviving_sets = remover.retroactively_remove(
                    prompt_length=prompt_length,
                    all_parallel_tokens=all_token_sets,
                    step=current_step
                )
            elif hasattr(remover, 'retroactively_prune'):
                surviving_sets = remover.retroactively_prune(
                    prompt_length=prompt_length,
                    all_parallel_tokens=all_token_sets,
                    step=current_step
                )
            else:
                self.log("Remover has no removal method", "error")
                return {}
            
            # Log removal statistics
            removed_count = self._count_removed_tokens(all_token_sets, surviving_sets, current_step)
            if removed_count > 0:
                self.log(f"Removed {removed_count} tokens at step {current_step}")
            
            return surviving_sets
            
        except Exception as e:
            self.log(f"Error during retroactive removal: {e}", "error")
            return {}
    
    def _count_removed_tokens(
        self,
        original_sets: dict[int, list[tuple[int, float]]],
        surviving_sets: dict[int, list[tuple[int, float]]],
        up_to_step: int
    ) -> int:
        """Count the number of tokens removed.
        
        Args:
            original_sets: Original token sets
            surviving_sets: Surviving token sets after removal
            up_to_step: Count removals up to this step
            
        Returns:
            Number of tokens removed
        """
        removed_count = 0
        
        for step in range(up_to_step):
            original = original_sets.get(step, [])
            surviving = surviving_sets.get(step, original)
            
            # Count tokens in original but not in surviving
            surviving_ids = {tid for tid, _ in surviving}
            for tid, _ in original:
                if tid not in surviving_ids:
                    removed_count += 1
        
        return removed_count