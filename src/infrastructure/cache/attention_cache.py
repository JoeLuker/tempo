"""Attention pattern caching implementation for the TEMPO system.

This module implements caching for attention patterns from model forward passes
to support retroactive pruning and analysis.
"""

from typing import Optional
import torch
from src.utils.logging_utils import LoggingMixin
from src.domain.entities.generation_state import AttentionPattern


class AttentionCache(LoggingMixin):
    """Cache for attention patterns from model forward passes."""
    
    def __init__(self):
        """Initialize the attention cache."""
        super().__init__()
        self.cached_attention: Optional[list[torch.Tensor]] = None
        self.cached_sequence_length: Optional[int] = None
        self.cache_updates = 0
        
        # Setup logging
        self.setup_logging("attention_cache", "attention_cache_debug.log")
    
    def cache(self, attention_layers: list[torch.Tensor], sequence_length: int) -> None:
        """Cache attention patterns from a forward pass.
        
        Args:
            attention_layers: List of attention tensors from each layer
            sequence_length: The sequence length for this attention
        """
        # Validate inputs
        assert attention_layers, "attention_layers cannot be empty"
        assert all(isinstance(layer, torch.Tensor) for layer in attention_layers), "All layers must be tensors"
        assert sequence_length > 0, "sequence_length must be positive"
        
        # Validate tensor shapes
        for i, layer in enumerate(attention_layers):
            if layer.dim() != 4:  # [batch, heads, seq, seq]
                raise ValueError(f"Layer {i} attention must be 4D, got {layer.dim()}D")
            if torch.isnan(layer).any() or torch.isinf(layer).any():
                raise ValueError(f"Layer {i} attention contains NaN or Inf values")
        
        # Store attention patterns
        self.cached_attention = [layer.clone() for layer in attention_layers]
        self.cached_sequence_length = sequence_length
        self.cache_updates += 1
        
        if self.debug_mode:
            self.log(f"Cached attention for {len(attention_layers)} layers, sequence length {sequence_length}")
            if len(attention_layers) > 0:
                first_shape = attention_layers[0].shape
                self.log(f"First layer shape: {first_shape}")
    
    def get(self) -> Optional[tuple[AttentionPattern, int]]:
        """Get cached attention patterns.
        
        Returns:
            Tuple of (AttentionPattern, sequence_length) or None if not cached
        """
        if self.cached_attention is None or self.cached_sequence_length is None:
            if self.debug_mode:
                self.log("No cached attention available")
            return None
        
        # Extract shape information from first layer
        if len(self.cached_attention) > 0 and self.cached_attention[0].dim() == 4:
            _, num_heads, _, _ = self.cached_attention[0].shape
        else:
            # Fallback if shape extraction fails
            num_heads = 1
        
        # Create AttentionPattern entity
        try:
            pattern = AttentionPattern(
                layers=self.cached_attention,
                sequence_length=self.cached_sequence_length,
                num_heads=num_heads,
                num_layers=len(self.cached_attention)
            )
            
            if self.debug_mode:
                self.log(f"Retrieved cached attention: {len(self.cached_attention)} layers, seq_len={self.cached_sequence_length}")
            
            return pattern, self.cached_sequence_length
        
        except Exception as e:
            self.log(f"Failed to create AttentionPattern: {e}", level="error")
            return None
    
    def clear(self) -> None:
        """Clear cached attention patterns."""
        self.cached_attention = None
        self.cached_sequence_length = None
        
        if self.debug_mode:
            self.log("Cleared attention cache")
    
    def get_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        has_cache = self.cached_attention is not None
        num_layers = len(self.cached_attention) if has_cache else 0
        
        stats = {
            "has_cached_attention": has_cache,
            "num_layers": num_layers,
            "sequence_length": self.cached_sequence_length,
            "cache_updates": self.cache_updates
        }
        
        if has_cache and num_layers > 0:
            first_layer = self.cached_attention[0]
            if first_layer.dim() == 4:
                batch_size, num_heads, seq_len, _ = first_layer.shape
                stats.update({
                    "batch_size": batch_size,
                    "num_heads": num_heads,
                    "attention_shape": list(first_layer.shape)
                })
        
        return stats
    
    def validate_for_sequence_length(self, expected_length: int) -> bool:
        """Check if cached attention is valid for a given sequence length.
        
        Args:
            expected_length: The expected sequence length
            
        Returns:
            True if cache is valid for this length, False otherwise
        """
        if self.cached_attention is None or self.cached_sequence_length is None:
            return False
        
        return self.cached_sequence_length == expected_length
