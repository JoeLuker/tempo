"""Sequence tracking service for generation progress.

This module handles tracking of sequence lengths and generation progress.
"""

from typing import List, Dict, Any, Optional, Callable
from ...utils.logging_utils import LoggingMixin


class SequenceTracker(LoggingMixin):
    """Tracks sequence length and generation progress."""
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the sequence tracker.
        
        Args:
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        self.setup_logging("sequence_tracker", "sequence_tracker.log", debug_mode)
        
        self.sequence_length = 0
        self.initial_prompt_length = 0
        self.step_count = 0
        self.sequence_length_history: List[int] = []
    
    def initialize(self, prompt_length: int) -> None:
        """Initialize sequence tracking with prompt length.
        
        Args:
            prompt_length: Length of the initial prompt
        """
        self.sequence_length = 0
        self.initial_prompt_length = prompt_length
        self.step_count = 0
        self.sequence_length_history = []
        self.log(f"Initialized sequence tracking with prompt length: {prompt_length}")
    
    def update_sequence_length(
        self, 
        new_length: int, 
        callback: Optional[Callable[[int, int, int], None]] = None
    ) -> None:
        """Update sequence length and notify callback.
        
        Args:
            new_length: New sequence length (excluding prompt)
            callback: Optional callback to notify of progress
        """
        if new_length > self.sequence_length:
            self.sequence_length = new_length
            self.step_count += 1
            self.sequence_length_history.append(new_length)
            
            if callback:
                try:
                    callback(new_length, self.step_count, self.initial_prompt_length)
                except Exception as e:
                    self.log(f"Error in sequence callback: {e}", "error")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about the generation sequence.
        
        Returns:
            Dictionary containing sequence metrics
        """
        return {
            "sequence_length": self.sequence_length,
            "initial_prompt_length": self.initial_prompt_length,
            "step_count": self.step_count,
            "sequence_length_history": self.sequence_length_history,
            "total_length": self.initial_prompt_length + self.sequence_length
        }