"""RoPE embedding modification for TEMPO."""

import torch
import math
from typing import Optional, Tuple

# Standard RoPE configuration constants
# These values are commonly used in LLaMA-style models
DEFAULT_ROPE_BASE = 10000.0  # Base frequency for RoPE
DEFAULT_MAX_POSITION = 8192  # Maximum sequence length


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor
        k: Key tensor  
        cos: Cosine component of rotary embedding
        sin: Sine component of rotary embedding
        position_ids: Position indices for each token
        unsqueeze_dim: Dimension to unsqueeze for broadcasting
        
    Returns:
        Rotated query and key tensors
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    if position_ids is not None:
        cos = cos.gather(0, position_ids.unsqueeze(-1).expand_as(q))
        sin = sin.gather(0, position_ids.unsqueeze(-1).expand_as(q))
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class RoPECache:
    """Cache for rotary position embeddings to avoid recomputation."""
    
    def __init__(self, dim: int, max_position: int = DEFAULT_MAX_POSITION, base: float = DEFAULT_ROPE_BASE):
        self.dim = dim
        self.max_position = max_position
        self.base = base
        self._cache = {}
        
    def get_cos_sin(
        self, 
        seq_len: int, 
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached cos/sin embeddings or compute if not cached."""
        cache_key = (seq_len, device, dtype)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Compute embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        inv_freq = inv_freq.to(device)
        
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        
        self._cache[cache_key] = (cos, sin)
        return cos, sin
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()


def modify_positions_for_parallel_tokens(
    position_ids: torch.Tensor,
    position_map: dict,
    device: torch.device
) -> torch.Tensor:
    """
    Modify position IDs to assign same position to parallel tokens.
    
    Args:
        position_ids: Original position IDs
        position_map: Mapping from physical to logical positions
        device: Computation device
        
    Returns:
        Modified position IDs
    """
    modified_positions = position_ids.clone()
    
    # Apply position mapping
    for physical_pos, logical_pos in position_map.items():
        mask = position_ids == physical_pos
        modified_positions[mask] = logical_pos
    
    return modified_positions.to(device)