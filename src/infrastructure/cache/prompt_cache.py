"""Prompt caching implementation for the TEMPO system.

This module implements caching for tokenized prompts to avoid
redundant tokenization operations.
"""

from typing import Optional
import torch
from src.utils.logging_utils import LoggingMixin
from src.domain.entities.generation_state import TokenizationResult


class PromptCache(LoggingMixin):
    """Cache for tokenized prompts."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize the prompt cache.
        
        Args:
            max_size: Maximum number of prompts to cache
        """
        super().__init__()
        self.cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
        # Setup logging
        self.setup_logging("prompt_cache", "prompt_cache_debug.log")
    
    def get(self, prompt: str) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Get cached tokenization for a prompt.
        
        Args:
            prompt: The prompt to look up
            
        Returns:
            Tuple of (input_ids, attention_mask) or None if not cached
        """
        if prompt in self.cache:
            self.hits += 1
            if self.debug_mode:
                self.log(f"Cache hit for prompt: {prompt[:50]}...")
            return self.cache[prompt]
        
        self.misses += 1
        if self.debug_mode:
            self.log(f"Cache miss for prompt: {prompt[:50]}...")
        return None
    
    def put(self, prompt: str, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        """Cache tokenization result for a prompt.
        
        Args:
            prompt: The prompt that was tokenized
            input_ids: The tokenized input IDs
            attention_mask: The attention mask
        """
        # Validate inputs
        assert prompt and isinstance(prompt, str), "Prompt must be a non-empty string"
        assert isinstance(input_ids, torch.Tensor), "input_ids must be a tensor"
        assert isinstance(attention_mask, torch.Tensor), "attention_mask must be a tensor"
        assert input_ids.shape == attention_mask.shape, "input_ids and attention_mask must have same shape"
        
        # Evict oldest entry if cache is full (simple FIFO)
        if len(self.cache) >= self.max_size:
            # Remove the first (oldest) entry
            oldest_prompt = next(iter(self.cache))
            del self.cache[oldest_prompt]
            if self.debug_mode:
                self.log(f"Evicted oldest prompt from cache: {oldest_prompt[:50]}...")
        
        # Clone tensors to ensure they're not modified
        self.cache[prompt] = (input_ids.clone(), attention_mask.clone())
        
        if self.debug_mode:
            self.log(f"Cached tokenization for prompt: {prompt[:50]}... (shape: {input_ids.shape})")
    
    def clear(self) -> None:
        """Clear all cached prompts."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        if self.debug_mode:
            self.log("Cleared prompt cache")
    
    def get_stats(self) -> dict[str, int]:
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
        """Get number of cached prompts."""
        return len(self.cache)
