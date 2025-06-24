"""Token decoding cache implementation for the TEMPO system.

This module implements caching for decoded token text to avoid
redundant decoding operations.
"""

from typing import Optional, Tuple, List
from .base_cache import BaseCache

# Cache configuration
DEFAULT_DECODE_CACHE_SIZE = 10000  # Covers most common tokens in vocabulary


class DecodeCache(BaseCache[int, str]):
    """Cache for decoded token text."""
    
    def __init__(self, max_size: int = DEFAULT_DECODE_CACHE_SIZE):
        """Initialize the decode cache.
        
        Args:
            max_size: Maximum number of tokens to cache
        """
        super().__init__(max_size, "decode_cache")
    
    def get(self, token_id: int) -> Optional[str]:
        """Get cached decoded text for a token ID.
        
        Args:
            token_id: The token ID to look up
            
        Returns:
            Decoded text or None if not cached
        """
        # Ensure consistent key type
        token_key = int(token_id)
        return super().get(token_key)
    
    def get_batch(self, token_ids: List[int]) -> Tuple[List[Optional[str]], List[int]]:
        """Get cached decoded text for multiple token IDs.
        
        Args:
            token_ids: List of token IDs to look up
            
        Returns:
            Tuple of (cached texts, uncached token IDs)
        """
        cached_results = []
        uncached_ids = []
        
        for token_id in token_ids:
            cached_text = self.get(token_id)
            cached_results.append(cached_text)
            if cached_text is None:
                uncached_ids.append(token_id)
        
        return cached_results, uncached_ids
    
    def put(self, token_id: int, text: str) -> None:
        """Cache decoded text for a token ID.
        
        Args:
            token_id: The token ID
            text: The decoded text
        """
        # Ensure consistent types
        token_key = int(token_id)
        text_value = str(text)
        
        # Validate inputs
        if not isinstance(text_value, str):
            raise ValueError("Decoded text must be a string")
        
        super().put(token_key, text_value)
    
    def put_batch(self, token_ids: List[int], texts: List[str]) -> None:
        """Cache multiple token-text pairs.
        
        Args:
            token_ids: List of token IDs
            texts: List of decoded texts
        """
        if len(token_ids) != len(texts):
            raise ValueError("token_ids and texts must have the same length")
        
        for token_id, text in zip(token_ids, texts):
            self.put(token_id, text)