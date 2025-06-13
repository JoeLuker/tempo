"""Model interface for the TEMPO system.

This module defines the interface for language model operations.
"""

from typing import Protocol, Optional, Any
import torch
from abc import abstractmethod


class ModelInterface(Protocol):
    """Interface for model operations."""
    
    @abstractmethod
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                past_key_values: Optional[Any] = None,
                use_cache: bool = True,
                output_attentions: bool = True) -> Any:
        """Run forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            past_key_values: Optional KV cache
            use_cache: Whether to use/return KV cache
            output_attentions: Whether to output attention patterns
            
        Returns:
            Model outputs (implementation specific)
        """
        ...
    
    @abstractmethod
    def clear_kv_cache(self) -> None:
        """Clear the model's KV cache."""
        ...
    
    @property
    @abstractmethod
    def config(self) -> Any:
        """Get model configuration."""
        ...