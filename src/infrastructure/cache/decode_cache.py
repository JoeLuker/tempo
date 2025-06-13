"""Token decoding cache implementation for the TEMPO system.

This module implements caching for decoded token text to avoid
redundant decoding operations.
"""

from typing import Dict, Optional, List, Tuple
from src.utils.logging_utils import LoggingMixin


class DecodeCache(LoggingMixin):
    """Cache for decoded token text."""
    
    def __init__(self, max_size: int = 10000):
        """Initialize the decode cache.
        
        Args:
            max_size: Maximum number of tokens to cache
        """
        super().__init__()
        self.cache: Dict[int, str] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
        # Setup logging
        self.setup_logging("decode_cache", "decode_cache_debug.log")
    
    def get(self, token_id: int) -> Optional[str]:
        """Get cached decoded text for a token ID.
        
        Args:
            token_id: The token ID to look up
            
        Returns:
            Decoded text or None if not cached
        """
        # Ensure consistent key type
        token_key = int(token_id)
        
        if token_key in self.cache:
            self.hits += 1
            return self.cache[token_key]
        
        self.misses += 1
        return None
    
    def get_batch(self, token_ids: List[int]) -> Tuple[List[Optional[str]], List[int]]:
        """Get cached decoded text for multiple token IDs.
        
        Args:
            token_ids: List of token IDs to look up
            
        Returns:
            Tuple of (cached_results, uncached_token_ids)
            cached_results contains the decoded text or None for each token
            uncached_token_ids contains the IDs that weren't in cache
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
        # Ensure consistent key type
        token_key = int(token_id)
        
        # Validate inputs
        assert isinstance(text, str), "Decoded text must be a string"
        
        # Evict oldest entry if cache is full (simple FIFO)
        if len(self.cache) >= self.max_size:
            # Remove the first (oldest) entry
            oldest_token = next(iter(self.cache))
            del self.cache[oldest_token]
        
        self.cache[token_key] = text
    
    def put_batch(self, token_ids: List[int], texts: List[str]) -> None:
        """Cache decoded text for multiple token IDs.
        
        Args:
            token_ids: List of token IDs
            texts: List of decoded texts (must be same length as token_ids)
        """
        assert len(token_ids) == len(texts), "token_ids and texts must have same length"
        
        for token_id, text in zip(token_ids, texts):
            self.put(token_id, text)
    
    def clear(self) -> None:
        """Clear all cached tokens."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        if self.debug_mode:
            self.log("Cleared decode cache")
    
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate
        }
    
    def __len__(self) -> int:
        """Get number of cached tokens."""
        return len(self.cache)


# Add missing import
from typing import Tuple
