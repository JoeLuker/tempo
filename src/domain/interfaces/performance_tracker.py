"""Performance tracker interface for the TEMPO system.

This module defines the interface for tracking performance metrics.
"""

from typing import Protocol
from abc import abstractmethod


class PerformanceTrackerInterface(Protocol):
    """Interface for tracking performance metrics."""
    
    @abstractmethod
    def track_tokenization(self, duration: float, cache_hit: bool) -> None:
        """Track tokenization performance."""
        ...
    
    @abstractmethod
    def track_model_call(self, duration: float, num_tokens: int = 1) -> None:
        """Track model forward pass performance."""
        ...
    
    @abstractmethod
    def track_decode(self, duration: float, num_tokens: int, cache_hits: int) -> None:
        """Track token decoding performance."""
        ...
    
    @abstractmethod
    def get_stats(self) -> dict:
        """Get performance statistics."""
        ...
    
    @abstractmethod
    def print_stats(self) -> None:
        """Print performance statistics."""
        ...