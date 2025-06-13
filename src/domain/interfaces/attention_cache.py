"""Attention cache interface for the TEMPO system.

This module defines the interface for caching and retrieving attention patterns.
"""

from typing import Protocol, Optional, Tuple
from abc import abstractmethod

from ..entities.generation_state import AttentionPattern


class AttentionCacheInterface(Protocol):
    """Interface for caching and retrieving attention patterns."""
    
    @abstractmethod
    def cache_attention(self, attention: AttentionPattern, sequence_length: int) -> None:
        """Cache attention patterns from a forward pass.
        
        Args:
            attention: Attention patterns to cache
            sequence_length: Sequence length for this attention
        """
        ...
    
    @abstractmethod
    def get_cached_attention(self) -> Optional[Tuple[AttentionPattern, int]]:
        """Retrieve the most recently cached attention patterns.
        
        Returns:
            Tuple of (AttentionPattern, sequence_length) or None if not cached
        """
        ...
    
    @abstractmethod
    def clear_cache(self) -> None:
        """Clear all cached attention patterns."""
        ...