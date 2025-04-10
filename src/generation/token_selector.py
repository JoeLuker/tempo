import torch
from typing import List, Tuple, Optional

class TokenSelector:
    """
    Responsible for selecting tokens based on probability thresholds.
    """
    
    def __init__(self, tokenizer):
        """
        Initialize the token selector.
        
        Args:
            tokenizer: HuggingFace tokenizer for decoding tokens
        """
        self.tokenizer = tokenizer
        
    def select_tokens_above_threshold(
        self,
        logits: torch.Tensor, 
        threshold: Optional[float] = 0.1
    ) -> Tuple[List[int], List[float]]:
        """
        Get tokens with probabilities above the threshold.
        Optimized for performance using tensor operations.
        
        Args:
            logits: Token logits from model
            threshold: Probability threshold (0.0 to 1.0)
            
        Returns:
            tuple: (token_ids, token_probabilities)
        """
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)[0]  # Take first (only) batch item
        
        # Default threshold if none specified
        if threshold is None:
            threshold = 0.1
        
        # Use tensor operations to find tokens above threshold - much faster than looping
        above_threshold = probs >= threshold
        
        # Get the indices and values that are above threshold
        indices = torch.nonzero(above_threshold, as_tuple=True)[0]
        values = probs[indices]
        
        # Sort by probability (highest first) - use torch.sort for efficiency
        if len(indices) > 0:
            sorted_values, sorted_indices = torch.sort(values, descending=True)
            indices = indices[sorted_indices]
            values = sorted_values
            
            # Convert to Python lists for compatibility with rest of the codebase
            indices = indices.tolist()
            values = values.tolist()
        else:
            indices, values = [], []
        
        return indices, values
    
    def is_eos_token(self, token_id: int) -> bool:
        """
        Check if the token ID is an end-of-sequence token.
        
        Args:
            token_id: Token ID to check
            
        Returns:
            bool: True if token is EOS, False otherwise
        """
        return hasattr(self.tokenizer, 'eos_token_id') and token_id == self.tokenizer.eos_token_id
    
    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """
        Decode token IDs to human-readable strings.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List[str]: Decoded token texts
        """
        return [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids] 