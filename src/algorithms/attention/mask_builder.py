"""Attention mask construction for parallel tokens."""

import torch
from typing import List, Tuple, Optional


class AttentionMaskBuilder:
    """Builds attention masks for controlling parallel token visibility."""
    
    def __init__(self, isolate_parallel_tokens: bool = True):
        self.isolate_parallel_tokens = isolate_parallel_tokens
        
    def create_causal_mask(
        self,
        seq_length: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Create standard causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_length, seq_length, dtype=dtype, device=device),
            diagonal=1
        )
        return mask * -10000.0
    
    def create_parallel_mask(
        self,
        seq_length: int,
        parallel_sets: List[Tuple[int, int]],
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Create mask for parallel token sets.
        
        Args:
            seq_length: Total sequence length
            parallel_sets: List of (start, end) positions for parallel sets
            device: Computation device
            dtype: Data type for mask
            
        Returns:
            Attention mask tensor
        """
        # Start with causal mask
        mask = self.create_causal_mask(seq_length, device, dtype)
        
        if not self.isolate_parallel_tokens:
            return mask
            
        # Modify mask for parallel sets
        for start, end in parallel_sets:
            if end > start + 1:  # Multiple tokens in set
                # Prevent tokens in same set from attending to each other
                for i in range(start, end):
                    for j in range(start, end):
                        if i != j:
                            mask[i, j] = -10000.0
        
        return mask
    
    def create_cross_attention_mask(
        self,
        query_length: int,
        key_length: int,
        parallel_sets: List[Tuple[int, int]],
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Create mask for cross-attention between sequences."""
        mask = torch.zeros(
            query_length, key_length, 
            dtype=dtype, device=device
        )
        
        # Apply parallel set constraints
        for start, end in parallel_sets:
            if end <= key_length:
                # Queries can only attend to first token in each parallel set
                mask[:, start+1:end] = -10000.0
                
        return mask
    
    def update_mask_for_position(
        self,
        mask: torch.Tensor,
        position: int,
        parallel_set: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Update mask for a specific position during generation."""
        if parallel_set and self.isolate_parallel_tokens:
            start, end = parallel_set
            # Isolate tokens within the parallel set
            mask[position, start:end] = -10000.0
            mask[position, position] = 0.0  # Allow self-attention
            
        return mask