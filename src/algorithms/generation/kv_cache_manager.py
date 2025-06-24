"""KV cache management for efficient generation."""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KVCache:
    """Container for key-value cache."""
    key_cache: List[torch.Tensor]  # List of [batch, heads, seq, dim]
    value_cache: List[torch.Tensor]
    

class KVCacheManager:
    """Manages key-value caches for transformer layers."""
    
    def __init__(self, num_layers: int, device: str = "cuda"):
        self.num_layers = num_layers
        self.device = device
        self.cache: Optional[KVCache] = None
        self.cache_size = 0
        
    def initialize_cache(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16
    ):
        """Initialize empty KV cache."""
        self.cache = KVCache(
            key_cache=[],
            value_cache=[]
        )
        
        # Pre-allocate for each layer
        for _ in range(self.num_layers):
            # Start with zero-sized cache
            k = torch.zeros(
                batch_size, num_heads, 0, head_dim,
                dtype=dtype, device=self.device
            )
            v = torch.zeros(
                batch_size, num_heads, 0, head_dim,
                dtype=dtype, device=self.device
            )
            self.cache.key_cache.append(k)
            self.cache.value_cache.append(v)
            
    def update_cache(
        self,
        layer_idx: int,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value pairs.
        
        Args:
            layer_idx: Layer index
            new_keys: New keys [batch, heads, seq, dim]
            new_values: New values [batch, heads, seq, dim]
            positions: Specific positions to update (for parallel tokens)
            
        Returns:
            Updated full key and value tensors
        """
        if self.cache is None:
            raise RuntimeError("Cache not initialized")
            
        if positions is None:
            # Standard append
            self.cache.key_cache[layer_idx] = torch.cat(
                [self.cache.key_cache[layer_idx], new_keys], dim=2
            )
            self.cache.value_cache[layer_idx] = torch.cat(
                [self.cache.value_cache[layer_idx], new_values], dim=2
            )
        else:
            # Update specific positions (for parallel tokens)
            self._update_positions(
                layer_idx, new_keys, new_values, positions
            )
            
        return self.cache.key_cache[layer_idx], self.cache.value_cache[layer_idx]
    
    def _update_positions(
        self,
        layer_idx: int,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        positions: torch.Tensor
    ):
        """Update cache at specific positions."""
        current_k = self.cache.key_cache[layer_idx]
        current_v = self.cache.value_cache[layer_idx]
        
        # Ensure cache is large enough
        max_pos = positions.max().item() + 1
        if current_k.size(2) < max_pos:
            # Expand cache
            expansion_size = max_pos - current_k.size(2)
            k_expansion = torch.zeros(
                current_k.size(0), current_k.size(1), 
                expansion_size, current_k.size(3),
                dtype=current_k.dtype, device=current_k.device
            )
            v_expansion = torch.zeros(
                current_v.size(0), current_v.size(1),
                expansion_size, current_v.size(3),
                dtype=current_v.dtype, device=current_v.device
            )
            
            self.cache.key_cache[layer_idx] = torch.cat(
                [current_k, k_expansion], dim=2
            )
            self.cache.value_cache[layer_idx] = torch.cat(
                [current_v, v_expansion], dim=2
            )
        
        # Update at positions
        for i, pos in enumerate(positions):
            self.cache.key_cache[layer_idx][:, :, pos] = new_keys[:, :, i]
            self.cache.value_cache[layer_idx][:, :, pos] = new_values[:, :, i]
    
    def get_cache_for_layer(
        self, 
        layer_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cache for specific layer."""
        if self.cache is None or layer_idx >= len(self.cache.key_cache):
            return None
            
        return self.cache.key_cache[layer_idx], self.cache.value_cache[layer_idx]
    
    def trim_cache(self, max_length: int):
        """Trim cache to maximum length."""
        if self.cache is None:
            return
            
        for i in range(self.num_layers):
            if self.cache.key_cache[i].size(2) > max_length:
                self.cache.key_cache[i] = self.cache.key_cache[i][:, :, :max_length]
                self.cache.value_cache[i] = self.cache.value_cache[i][:, :, :max_length]
    
    def clear_cache(self):
        """Clear all cached values."""
        if self.cache is not None:
            for i in range(self.num_layers):
                self.cache.key_cache[i] = self.cache.key_cache[i][:, :, :0]
                self.cache.value_cache[i] = self.cache.value_cache[i][:, :, :0]
        self.cache_size = 0
    
    def get_cache_size(self) -> int:
        """Get current cache size in tokens."""
        if self.cache is None or not self.cache.key_cache:
            return 0
        return self.cache.key_cache[0].size(2)