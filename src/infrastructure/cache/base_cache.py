"""Base cache implementation for TEMPO system.

Provides a simple FIFO cache that can be extended for specific use cases.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, Dict, List, Tuple
from src.utils.logging_utils import LoggingMixin

K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type


class BaseCache(Generic[K, V], LoggingMixin, ABC):
    """Abstract base class for FIFO caches."""
    
    def __init__(self, max_size: int, cache_name: str):
        """Initialize the base cache.
        
        Args:
            max_size: Maximum number of items to cache
            cache_name: Name for logging purposes
        """
        super().__init__()
        self.cache: Dict[K, V] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.cache_name = cache_name
        
        # Setup logging
        self.setup_logging(cache_name, f"{cache_name}_debug.log")
    
    def get(self, key: K) -> Optional[V]:
        """Get cached value for a key.
        
        Args:
            key: The key to look up
            
        Returns:
            Cached value or None if not cached
        """
        if key in self.cache:
            self.hits += 1
            self.log(f"Cache hit for key: {key}", level="debug")
            return self.cache[key]
        
        self.misses += 1
        self.log(f"Cache miss for key: {key}", level="debug")
        return None
    
    def put(self, key: K, value: V) -> None:
        """Add or update a value in the cache.
        
        Args:
            key: The key to store
            value: The value to cache
        """
        # Check if we need to evict
        if key not in self.cache and len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = value
        self.log(f"Cached value for key: {key}", level="debug")
    
    def _evict_oldest(self) -> None:
        """Evict the oldest entry from the cache (FIFO)."""
        if self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.log(f"Evicted oldest key: {oldest_key}", level="debug")
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.log(f"{self.cache_name} cache cleared", level="info")
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }
    
    @abstractmethod
    def get_batch(self, keys: List[K]) -> Tuple[List[Optional[V]], List[K]]:
        """Get multiple values from cache.
        
        Args:
            keys: List of keys to look up
            
        Returns:
            Tuple of (cached values, uncached keys)
        """
        pass