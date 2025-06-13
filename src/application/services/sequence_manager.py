"""Sequence management service for tracking generation progress.

This service manages sequence length tracking, callbacks, and
generation metrics throughout the generation process.
"""

from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass, field

from ...utils.logging_utils import LoggingMixin


@dataclass
class SequenceMetrics:
    """Metrics about the generation sequence."""
    sequence_length: int = 0
    initial_prompt_length: int = 0
    step_count: int = 0
    sequence_length_history: List[int] = field(default_factory=list)
    
    @property
    def total_length(self) -> int:
        """Get total sequence length including prompt."""
        return self.initial_prompt_length + self.sequence_length
    
    @property
    def average_tokens_per_step(self) -> float:
        """Calculate average tokens generated per step."""
        if self.step_count == 0:
            return 0.0
        return self.sequence_length / self.step_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "sequence_length": self.sequence_length,
            "initial_prompt_length": self.initial_prompt_length,
            "step_count": self.step_count,
            "total_length": self.total_length,
            "average_tokens_per_step": self.average_tokens_per_step,
            "sequence_length_history": self.sequence_length_history
        }


class SequenceManager(LoggingMixin):
    """Manages sequence tracking and callbacks during generation."""
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the sequence manager.
        
        Args:
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        self.setup_logging("sequence_manager", "sequence.log", debug_mode)
        
        self.metrics = SequenceMetrics()
        self.callbacks: List[Callable[[int, int, int], None]] = []
    
    def initialize(self, prompt_length: int) -> None:
        """Initialize sequence tracking with prompt length.
        
        Args:
            prompt_length: Length of the initial prompt in tokens
        """
        if prompt_length < 0:
            raise ValueError(f"Prompt length must be non-negative, got {prompt_length}")
        
        self.metrics = SequenceMetrics(
            initial_prompt_length=prompt_length
        )
        
        self.log(f"Initialized sequence tracking with prompt length: {prompt_length}")
    
    def update(self, new_tokens: int) -> bool:
        """Update sequence with new tokens.
        
        Args:
            new_tokens: Number of new tokens added
            
        Returns:
            True if sequence was updated, False otherwise
        """
        if new_tokens <= 0:
            return False
        
        old_length = self.metrics.sequence_length
        self.metrics.sequence_length += new_tokens
        self.metrics.step_count += 1
        self.metrics.sequence_length_history.append(self.metrics.sequence_length)
        
        # Validate consistency
        if len(self.metrics.sequence_length_history) != self.metrics.step_count:
            self.log(
                f"Inconsistency detected: history length {len(self.metrics.sequence_length_history)} "
                f"!= step count {self.metrics.step_count}",
                "warning"
            )
        
        # Log update
        if self.debug_mode:
            self.log(
                f"Sequence updated: {old_length} → {self.metrics.sequence_length} "
                f"(+{new_tokens} tokens)"
            )
        
        # Notify callbacks
        self._notify_callbacks()
        
        return True
    
    def add_callback(self, callback: Callable[[int, int, int], None]) -> None:
        """Add a callback for sequence updates.
        
        Args:
            callback: Function called with (sequence_length, step_count, prompt_length)
        """
        self.callbacks.append(callback)
        self.log(f"Added sequence callback, total callbacks: {len(self.callbacks)}")
    
    def remove_callback(self, callback: Callable[[int, int, int], None]) -> bool:
        """Remove a callback.
        
        Args:
            callback: Callback to remove
            
        Returns:
            True if callback was removed, False if not found
        """
        try:
            self.callbacks.remove(callback)
            self.log(f"Removed sequence callback, remaining: {len(self.callbacks)}")
            return True
        except ValueError:
            return False
    
    def get_metrics(self) -> SequenceMetrics:
        """Get current sequence metrics.
        
        Returns:
            Current sequence metrics
        """
        return self.metrics
    
    def get_current_length(self) -> int:
        """Get current sequence length (excluding prompt).
        
        Returns:
            Current sequence length
        """
        return self.metrics.sequence_length
    
    def get_total_length(self) -> int:
        """Get total sequence length (including prompt).
        
        Returns:
            Total sequence length
        """
        return self.metrics.total_length
    
    def get_step_count(self) -> int:
        """Get number of generation steps.
        
        Returns:
            Number of steps taken
        """
        return self.metrics.step_count
    
    def reset(self) -> None:
        """Reset sequence tracking."""
        self.metrics = SequenceMetrics()
        self.log("Sequence tracking reset")
    
    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks of sequence update."""
        for callback in self.callbacks:
            try:
                callback(
                    self.metrics.sequence_length,
                    self.metrics.step_count,
                    self.metrics.initial_prompt_length
                )
            except Exception as e:
                self.log(f"Error in sequence callback: {e}", "error")
    
    def validate_state(self) -> bool:
        """Validate internal state consistency.
        
        Returns:
            True if state is valid, False otherwise
        """
        checks = [
            (self.metrics.sequence_length >= 0, "sequence_length must be non-negative"),
            (self.metrics.initial_prompt_length >= 0, "initial_prompt_length must be non-negative"),
            (self.metrics.step_count >= 0, "step_count must be non-negative"),
            (
                len(self.metrics.sequence_length_history) == self.metrics.step_count,
                "history length must match step count"
            )
        ]
        
        all_valid = True
        for check, message in checks:
            if not check:
                self.log(f"Validation failed: {message}", "error")
                all_valid = False
        
        if all_valid and self.debug_mode:
            self.log("State validation passed")
        
        return all_valid
