"""Cache manager that coordinates all caching operations for TEMPO.

This module provides a unified interface for managing prompt, decode,
and attention caches.
"""

from typing import Optional
import torch
from src.utils.logging_utils import LoggingMixin
from src.domain.entities.generation_state import TokenizationResult, AttentionPattern
from src.domain.interfaces.cache_manager import CacheManagerInterface

from .prompt_cache import PromptCache
from .decode_cache import DecodeCache
from .attention_cache import AttentionCache


class CacheManager(LoggingMixin, CacheManagerInterface):
    """Manages all caching operations for the token generation system."""
    
    def __init__(self, 
                 prompt_cache_size: int = 1000,
                 decode_cache_size: int = 10000):
        """Initialize the cache manager.
        
        Args:
            prompt_cache_size: Maximum number of prompts to cache
            decode_cache_size: Maximum number of decoded tokens to cache
        """
        super().__init__()
        
        # Initialize individual caches
        self.prompt_cache = PromptCache(max_size=prompt_cache_size)
        self.decode_cache = DecodeCache(max_size=decode_cache_size)
        self.attention_cache = AttentionCache()
        
        # Setup logging
        self.setup_logging("cache_manager", "cache_manager_debug.log")
    
    # Prompt cache operations
    def get_tokenized_prompt(self, prompt: str) -> Optional[TokenizationResult]:
        """Get cached tokenization result for a prompt.
        
        Args:
            prompt: The prompt to look up
            
        Returns:
            Cached TokenizationResult or None if not cached
        """
        cached = self.prompt_cache.get(prompt)
        if cached is not None:
            input_ids, attention_mask = cached
            return TokenizationResult(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt=prompt,
                token_count=input_ids.shape[1]
            )
        return None
    
    def cache_tokenized_prompt(self, prompt: str, result: TokenizationResult) -> None:
        """Cache a tokenization result.
        
        Args:
            prompt: The prompt that was tokenized
            result: The tokenization result to cache
        """
        self.prompt_cache.put(prompt, result.input_ids, result.attention_mask)
    
    # Decode cache operations
    def get_decoded_token(self, token_id: int) -> Optional[str]:
        """Get cached decoded text for a token ID.
        
        Args:
            token_id: The token ID to look up
            
        Returns:
            Cached decoded text or None if not cached
        """
        return self.decode_cache.get(token_id)
    
    def cache_decoded_token(self, token_id: int, text: str) -> None:
        """Cache decoded text for a token ID.
        
        Args:
            token_id: The token ID
            text: The decoded text
        """
        self.decode_cache.put(token_id, text)
    
    def get_decoded_tokens_batch(self, token_ids: list[int]) -> tuple[list[Optional[str]], list[int]]:
        """Get cached decoded text for multiple token IDs.
        
        Args:
            token_ids: List of token IDs to look up
            
        Returns:
            Tuple of (cached_results, uncached_token_ids)
        """
        return self.decode_cache.get_batch(token_ids)
    
    def cache_decoded_tokens_batch(self, token_ids: list[int], texts: list[str]) -> None:
        """Cache decoded text for multiple token IDs.
        
        Args:
            token_ids: List of token IDs
            texts: List of decoded texts
        """
        self.decode_cache.put_batch(token_ids, texts)
    
    # Attention cache operations
    def cache_attention(self, attention_layers: list[torch.Tensor], sequence_length: int) -> None:
        """Cache attention patterns from a forward pass.
        
        Args:
            attention_layers: List of attention tensors from each layer
            sequence_length: The sequence length for this attention
        """
        self.attention_cache.cache(attention_layers, sequence_length)
    
    def get_cached_attention(self) -> Optional[tuple[AttentionPattern, int]]:
        """Get cached attention patterns.
        
        Returns:
            Tuple of (AttentionPattern, sequence_length) or None if not cached
        """
        return self.attention_cache.get()
    
    def validate_attention_cache(self, expected_length: int) -> bool:
        """Check if cached attention is valid for a given sequence length.
        
        Args:
            expected_length: The expected sequence length
            
        Returns:
            True if cache is valid for this length, False otherwise
        """
        return self.attention_cache.validate_for_sequence_length(expected_length)
    
    # Global operations
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        self.prompt_cache.clear()
        self.decode_cache.clear()
        self.attention_cache.clear()
        
        if self.debug_mode:
            self.log("Cleared all caches")
    
    def get_all_stats(self) -> dict:
        """Get statistics from all caches.
        
        Returns:
            Dictionary with statistics from all caches
        """
        return {
            "prompt_cache": self.prompt_cache.get_stats(),
            "decode_cache": self.decode_cache.get_stats(),
            "attention_cache": self.attention_cache.get_stats()
        }
    
    def print_stats(self) -> None:
        """Print statistics from all caches."""
        stats = self.get_all_stats()
        
        print("\nCache Manager Statistics:")
        
        # Prompt cache stats
        prompt_stats = stats["prompt_cache"]
        print(f"\n  Prompt Cache:")
        print(f"    Size: {prompt_stats['size']}/{prompt_stats['max_size']}")
        print(f"    Hits: {prompt_stats['hits']}")
        print(f"    Misses: {prompt_stats['misses']}")
        print(f"    Hit Rate: {prompt_stats['hit_rate']:.1f}%")
        
        # Decode cache stats
        decode_stats = stats["decode_cache"]
        print(f"\n  Decode Cache:")
        print(f"    Size: {decode_stats['size']}/{decode_stats['max_size']}")
        print(f"    Hits: {decode_stats['hits']}")
        print(f"    Misses: {decode_stats['misses']}")
        print(f"    Hit Rate: {decode_stats['hit_rate']:.1f}%")
        
        # Attention cache stats
        attention_stats = stats["attention_cache"]
        print(f"\n  Attention Cache:")
        print(f"    Has Cached Attention: {attention_stats['has_cached_attention']}")
        if attention_stats['has_cached_attention']:
            print(f"    Layers: {attention_stats['num_layers']}")
            print(f"    Sequence Length: {attention_stats['sequence_length']}")
            print(f"    Updates: {attention_stats['cache_updates']}")
