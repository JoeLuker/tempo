"""
Cache management for TEMPO API.

This module provides functionality for caching generation results and 
cleaning up old entries to manage memory usage.
"""

import time
import logging
from typing import Dict, Any, List

# Configure logging
logger = logging.getLogger("tempo-api")

# Global cache for generation results
generation_cache: Dict[str, Dict[str, Any]] = {}

def clean_generation_cache() -> List[str]:
    """
    Clean up old generation cache entries.
    
    This is run as a background task after each generation.
    
    Returns:
        List[str]: List of removed cache entry keys
    """
    global generation_cache
    
    # Keep entries for up to 30 minutes
    max_age = 30 * 60  # 30 minutes in seconds
    current_time = time.time()
    
    # Find old entries
    to_remove = []
    for key, entry in generation_cache.items():
        if current_time - entry["timestamp"] > max_age:
            to_remove.append(key)
    
    # Remove old entries
    for key in to_remove:
        del generation_cache[key]
    
    if to_remove:
        logger.info(f"Cleaned {len(to_remove)} old entries from generation cache")
    
    return to_remove

def add_to_cache(generation_id: str, result: Dict[str, Any], response: Any) -> None:
    """
    Add a generation result to the cache.
    
    Args:
        generation_id: Unique identifier for the generation
        result: Raw generation result dictionary
        response: Formatted response object
    """
    global generation_cache
    
    generation_cache[generation_id] = {
        "result": result,
        "response": response,
        "timestamp": time.time()
    }

def clear_cache() -> int:
    """
    Clear the generation cache to free memory.
    
    Returns:
        int: Number of cache entries that were cleared
    """
    global generation_cache
    cache_size = len(generation_cache)
    generation_cache.clear()
    
    logger.info(f"Cache cleared: {cache_size} entries removed")
    return cache_size

def get_cache_entry(generation_id: str) -> Dict[str, Any]:
    """
    Get a cached generation entry.
    
    Args:
        generation_id: Unique identifier for the generation
        
    Returns:
        Dict[str, Any]: The cached entry or None if not found
    """
    global generation_cache
    return generation_cache.get(generation_id)