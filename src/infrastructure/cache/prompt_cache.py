"""Prompt caching implementation for the TEMPO system.

This module implements caching for tokenized prompts to avoid
redundant tokenization operations.
"""

from typing import Optional, Tuple, List
import torch
from .base_cache import BaseCache
from src.domain.entities.generation_state import TokenizationResult

# Cache configuration
DEFAULT_PROMPT_CACHE_SIZE = 1000  # Reasonable size for prompt variations


class PromptCache(BaseCache[str, TokenizationResult]):
    """Cache for tokenized prompts."""
    
    def __init__(self, max_size: int = DEFAULT_PROMPT_CACHE_SIZE):
        """Initialize the prompt cache.
        
        Args:
            max_size: Maximum number of prompts to cache
        """
        super().__init__(max_size, "prompt_cache")
    
    def get_batch(self, prompts: List[str]) -> Tuple[List[Optional[TokenizationResult]], List[str]]:
        """Get cached tokenization for multiple prompts.
        
        Args:
            prompts: List of prompts to look up
            
        Returns:
            Tuple of (cached results, uncached prompts)
        """
        cached_results = []
        uncached_prompts = []
        
        for prompt in prompts:
            cached_result = self.get(prompt)
            cached_results.append(cached_result)
            if cached_result is None:
                uncached_prompts.append(prompt)
        
        return cached_results, uncached_prompts
    
    def put(self, prompt: str, result: TokenizationResult) -> None:
        """Cache tokenization result for a prompt.
        
        Args:
            prompt: The prompt text
            result: The tokenization result
        """
        # Validate input
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")
        if not isinstance(result, TokenizationResult):
            raise ValueError("Result must be a TokenizationResult")
        
        super().put(prompt, result)
    
    def put_batch(self, prompts: List[str], results: List[TokenizationResult]) -> None:
        """Cache multiple prompt-result pairs.
        
        Args:
            prompts: List of prompts
            results: List of tokenization results
        """
        if len(prompts) != len(results):
            raise ValueError("prompts and results must have the same length")
        
        for prompt, result in zip(prompts, results):
            self.put(prompt, result)