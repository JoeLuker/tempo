"""Generation state entities for the TEMPO system.

This module defines entities that represent the state of text generation,
including sequences, attention masks, and KV caches.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
import torch
from datetime import datetime


@dataclass
class GenerationState:
    """Entity representing the current state of text generation."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Optional[Any] = None  # Can be list of tuples or DynamicCache
    sequence_length: int = 0
    generated_tokens: List[int] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate generation state."""
        if self.input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D, got {self.input_ids.dim()}D")
        if self.attention_mask.dim() != 2:
            raise ValueError(f"attention_mask must be 2D, got {self.attention_mask.dim()}D")
        if self.input_ids.shape != self.attention_mask.shape:
            raise ValueError(f"input_ids and attention_mask shapes must match: {self.input_ids.shape} vs {self.attention_mask.shape}")
        if self.sequence_length < 0:
            raise ValueError(f"sequence_length must be non-negative, got {self.sequence_length}")
        
        # Update sequence length if not set
        if self.sequence_length == 0:
            self.sequence_length = self.input_ids.shape[1]
    
    def add_token(self, token_id: int) -> 'GenerationState':
        """Create a new state with an additional token."""
        # This returns a new state to maintain immutability principles where possible
        new_input_ids = torch.cat([self.input_ids, torch.tensor([[token_id]], device=self.input_ids.device)], dim=1)
        new_attention_mask = torch.cat([self.attention_mask, torch.ones((1, 1), device=self.attention_mask.device)], dim=1)
        new_generated_tokens = self.generated_tokens + [token_id]
        
        return GenerationState(
            input_ids=new_input_ids,
            attention_mask=new_attention_mask,
            past_key_values=self.past_key_values,  # This will be updated by the infrastructure layer
            sequence_length=self.sequence_length + 1,
            generated_tokens=new_generated_tokens,
            timestamp=self.timestamp
        )
    
    def get_current_sequence_length(self) -> int:
        """Get the current total sequence length including KV cache."""
        base_length = self.input_ids.shape[1]
        
        # If we have KV cache, add its length
        if self.past_key_values is not None:
            if hasattr(self.past_key_values, 'get_seq_length'):
                # DynamicCache format
                cache_length = self.past_key_values.get_seq_length()
                if cache_length is not None:
                    return cache_length
            elif isinstance(self.past_key_values, list) and len(self.past_key_values) > 0:
                # Legacy list format
                if isinstance(self.past_key_values[0], tuple) and len(self.past_key_values[0]) >= 2:
                    key_tensor = self.past_key_values[0][0]
                    if hasattr(key_tensor, 'shape') and len(key_tensor.shape) >= 3:
                        return key_tensor.shape[2]
        
        return base_length


@dataclass
class AttentionPattern:
    """Entity representing attention patterns from model layers."""
    layers: List[torch.Tensor]
    sequence_length: int
    num_heads: int
    num_layers: int
    
    def __post_init__(self):
        """Validate attention pattern."""
        if not self.layers:
            raise ValueError("AttentionPattern must have at least one layer")
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.num_layers != len(self.layers):
            raise ValueError(f"num_layers {self.num_layers} doesn't match actual layers {len(self.layers)}")
        
        # Validate each layer tensor
        for i, layer in enumerate(self.layers):
            if layer.dim() != 4:  # [batch, heads, seq, seq]
                raise ValueError(f"Layer {i} attention must be 4D, got {layer.dim()}D")
            if torch.isnan(layer).any() or torch.isinf(layer).any():
                raise ValueError(f"Layer {i} attention contains NaN or Inf values")
    
    def get_layer_attention(self, layer_idx: int) -> torch.Tensor:
        """Get attention pattern for a specific layer."""
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")
        return self.layers[layer_idx]
    
    def get_head_attention(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        """Get attention pattern for a specific head in a specific layer."""
        if not 0 <= head_idx < self.num_heads:
            raise ValueError(f"Head index {head_idx} out of range [0, {self.num_heads})")
        layer_attention = self.get_layer_attention(layer_idx)
        return layer_attention[:, head_idx, :, :]


@dataclass
class TokenizationResult:
    """Result of tokenizing a prompt."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt: str
    token_count: int
    
    def __post_init__(self):
        """Validate tokenization result."""
        if self.input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D, got {self.input_ids.dim()}D")
        if self.attention_mask.dim() != 2:
            raise ValueError(f"attention_mask must be 2D, got {self.attention_mask.dim()}D")
        if self.input_ids.shape != self.attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have the same shape")
        if self.token_count != self.input_ids.shape[1]:
            raise ValueError(f"token_count {self.token_count} doesn't match actual tokens {self.input_ids.shape[1]}")
        if not self.prompt:
            raise ValueError("prompt cannot be empty")
