"""Token generator interface for the TEMPO system.

This module defines the interface for token generation operations.
"""

from typing import Protocol, Optional, Tuple
import torch
from abc import abstractmethod

from ..entities.logits import TokenLogits
from ..entities.generation_state import GenerationState


class TokenGeneratorInterface(Protocol):
    """Interface for token generation operations."""
    
    @abstractmethod
    def generate_logits(self, state: GenerationState, custom_attention_mask: Optional[torch.Tensor] = None) -> TokenLogits:
        """Generate logits for the next token given the current state.
        
        Args:
            state: Current generation state
            custom_attention_mask: Optional custom attention mask
            
        Returns:
            TokenLogits containing the raw logits from the model
        """
        ...
    
    @abstractmethod
    def generate_logits_with_cache(self, 
                                   state: GenerationState, 
                                   custom_attention_mask: Optional[torch.Tensor] = None) -> Tuple[TokenLogits, GenerationState]:
        """Generate logits and update the generation state with new KV cache.
        
        Args:
            state: Current generation state
            custom_attention_mask: Optional custom attention mask
            
        Returns:
            Tuple of (TokenLogits, updated GenerationState with new KV cache)
        """
        ...
    
    @abstractmethod
    def generate_logits_for_isolated_parallel(self,
                                              state: GenerationState,
                                              num_parallel_tokens: int,
                                              custom_attention_mask: Optional[torch.Tensor] = None) -> Tuple[TokenLogits, GenerationState]:
        """Optimized logit generation for isolated parallel tokens.
        
        Since isolated tokens can't see each other, we only need to compute
        the forward pass once and can reuse the logits for all tokens.
        
        Args:
            state: Current generation state
            num_parallel_tokens: Number of parallel tokens to generate
            custom_attention_mask: Optional custom attention mask
            
        Returns:
            Tuple of (TokenLogits that can be reused, updated GenerationState)
        """
        ...
