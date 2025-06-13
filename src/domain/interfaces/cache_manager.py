"""Cache manager interface for the TEMPO system.

This module defines the interface for managing all caching operations.
"""

from typing import Protocol, Optional
from abc import abstractmethod

from ..entities.generation_state import TokenizationResult


class CacheManagerInterface(Protocol):
    """Interface for managing all caching operations."""
    
    @abstractmethod
    def get_tokenized_prompt(self, prompt: str) -> Optional[TokenizationResult]:
        """Get cached tokenization result for a prompt.
        
        Args:
            prompt: The prompt to look up
            
        Returns:
            Cached TokenizationResult or None if not cached
        """
        ...
    
    @abstractmethod
    def cache_tokenized_prompt(self, prompt: str, result: TokenizationResult) -> None:
        """Cache a tokenization result.
        
        Args:
            prompt: The prompt that was tokenized
            result: The tokenization result to cache
        """
        ...
    
    @abstractmethod
    def get_decoded_token(self, token_id: int) -> Optional[str]:
        """Get cached decoded text for a token ID.
        
        Args:
            token_id: The token ID to look up
            
        Returns:
            Cached decoded text or None if not cached
        """
        ...
    
    @abstractmethod
    def cache_decoded_token(self, token_id: int, text: str) -> None:
        """Cache decoded text for a token ID.
        
        Args:
            token_id: The token ID
            text: The decoded text
        """
        ...
    
    @abstractmethod
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        ...