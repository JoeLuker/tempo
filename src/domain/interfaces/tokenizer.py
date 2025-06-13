"""Tokenizer interface for the TEMPO system.

This module defines the interface for tokenization operations.
"""

from typing import Protocol, List
from abc import abstractmethod

from ..entities.generation_state import TokenizationResult


class TokenizerInterface(Protocol):
    """Interface for tokenization operations."""
    
    @abstractmethod
    def tokenize_prompt(self, prompt: str) -> TokenizationResult:
        """Tokenize a text prompt into input tensors.
        
        Args:
            prompt: Text prompt to tokenize
            
        Returns:
            TokenizationResult with input_ids and attention_mask
        """
        ...
    
    @abstractmethod
    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """Decode a list of token IDs to text.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            List of decoded token strings
        """
        ...
    
    @abstractmethod
    def decode_single_token(self, token_id: int) -> str:
        """Decode a single token ID to text.
        
        Args:
            token_id: Token ID to decode
            
        Returns:
            Decoded token string
        """
        ...