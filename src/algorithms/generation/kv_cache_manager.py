"""KV cache management for efficient generation."""

import torch
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KVCache:
    """Container for key-value cache."""
    key_cache: List[torch.Tensor]  # List of [batch, heads, seq, dim]
    value_cache: List[torch.Tensor]


class KVCacheManager:
    """Manages key-value caches for transformer layers with memory controls."""

    def __init__(
        self,
        num_layers: int,
        device: str = "cuda",
        max_memory_gb: Optional[float] = None,
        max_cache_tokens: Optional[int] = None
    ):
        """Initialize KV cache manager.

        Args:
            num_layers: Number of transformer layers
            device: Device for cache storage
            max_memory_gb: Maximum memory for cache in GB (optional)
            max_cache_tokens: Maximum tokens in cache (optional)
        """
        self.num_layers = num_layers
        self.device = device
        self.cache: Optional[KVCache] = None
        self.cache_size = 0
        self.max_memory_gb = max_memory_gb
        self.max_cache_tokens = max_cache_tokens

        # Track cache memory usage
        self._cache_memory_bytes = 0
        
    def initialize_cache(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16
    ):
        """Initialize empty KV cache with memory tracking."""
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

        self._cache_memory_bytes = 0

        logger.debug(
            f"Initialized KV cache: {self.num_layers} layers, "
            f"batch={batch_size}, heads={num_heads}, dim={head_dim}"
        )
            
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

        # Check memory limits before expansion
        if positions is None:
            new_size = self.cache.key_cache[layer_idx].size(2) + new_keys.size(2)
        else:
            new_size = max(self.cache.key_cache[layer_idx].size(2), positions.max().item() + 1)

        self._check_cache_limits(new_size)

        if positions is None:
            # Standard append
            old_size = self.cache.key_cache[layer_idx].size(2)
            self.cache.key_cache[layer_idx] = torch.cat(
                [self.cache.key_cache[layer_idx], new_keys], dim=2
            )
            self.cache.value_cache[layer_idx] = torch.cat(
                [self.cache.value_cache[layer_idx], new_values], dim=2
            )

            # Update memory tracking
            if layer_idx == 0:  # Only count once
                added_bytes = new_keys.element_size() * new_keys.numel() * 2  # K and V
                self._cache_memory_bytes += added_bytes
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

    def get_cache_memory_gb(self) -> float:
        """Get current cache memory usage in GB."""
        return self._cache_memory_bytes / (1024**3)

    def _check_cache_limits(self, new_size: int) -> None:
        """Check if cache expansion would violate limits.

        Args:
            new_size: Proposed new cache size in tokens

        Raises:
            RuntimeError: If limits would be exceeded
        """
        # Check token limit
        if self.max_cache_tokens is not None and new_size > self.max_cache_tokens:
            raise RuntimeError(
                f"Cache token limit exceeded: {new_size} > {self.max_cache_tokens}. "
                f"Consider reducing max_tokens or enabling aggressive pruning."
            )

        # Check memory limit (approximate)
        if self.max_memory_gb is not None and self.cache is not None:
            # Estimate memory for new size
            if self.cache.key_cache:
                sample_tensor = self.cache.key_cache[0]
                bytes_per_token = (
                    sample_tensor.element_size() *
                    sample_tensor.size(0) *  # batch
                    sample_tensor.size(1) *  # heads
                    sample_tensor.size(3)    # dim
                )
                estimated_bytes = bytes_per_token * new_size * self.num_layers * 2  # K and V
                estimated_gb = estimated_bytes / (1024**3)

                if estimated_gb > self.max_memory_gb:
                    raise RuntimeError(
                        f"Cache memory limit would be exceeded: {estimated_gb:.2f}GB > {self.max_memory_gb:.2f}GB. "
                        f"Consider reducing max_tokens or sequence length."
                    )

    def set_memory_limit(self, max_memory_gb: float) -> None:
        """Set maximum memory limit for cache.

        Args:
            max_memory_gb: Maximum memory in GB
        """
        assert max_memory_gb > 0, "Memory limit must be positive"
        self.max_memory_gb = max_memory_gb
        logger.info(f"Set KV cache memory limit to {max_memory_gb:.2f}GB")

    def set_token_limit(self, max_tokens: int) -> None:
        """Set maximum token limit for cache.

        Args:
            max_tokens: Maximum number of tokens
        """
        assert max_tokens > 0, "Token limit must be positive"
        self.max_cache_tokens = max_tokens
        logger.info(f"Set KV cache token limit to {max_tokens}")