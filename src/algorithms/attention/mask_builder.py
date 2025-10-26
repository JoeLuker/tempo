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
        """Create attention mask that controls visibility of parallel token sets.

        In TEMPO, parallel tokens are selected from the same logit distribution and
        appended to the sequence together. The "isolation" question is about whether
        FUTURE tokens can attend back to the parallel set.

        Isolated mode: Future tokens cannot attend to ANY tokens in a parallel set
        Visible mode: Future tokens can attend to ALL tokens in a parallel set

        This is implemented by masking within-set attention when building the mask
        for the NEXT generation step.

        Args:
            seq_length: Total sequence length including all parallel tokens
            parallel_sets: List of (start, end) positions for previously registered parallel sets
            device: Computation device
            dtype: Data type for mask

        Returns:
            Attention mask [seq_len, seq_len] where -10000.0 = masked, 0.0 = visible
        """
        # Start with causal mask (prevents attending to future positions)
        mask = self.create_causal_mask(seq_length, device, dtype)

        if not self.isolate_parallel_tokens:
            # Visible mode: standard causal mask (future can attend to all past)
            return mask

        # Isolated mode: make parallel sets invisible to future tokens
        for start, end in parallel_sets:
            if end > start + 1:  # Multiple tokens in parallel set
                # 1. Mask cross-attention within the parallel set
                # (tokens in the set can't attend to each other)
                for i in range(start, end):
                    for j in range(start, end):
                        if i != j:
                            mask[i, j] = -10000.0

                # 2. Mask future tokens from attending to the parallel set
                # (tokens after the set can't attend to any token in the set)
                for future_pos in range(end, seq_length):
                    for parallel_pos in range(start, end):
                        mask[future_pos, parallel_pos] = -10000.0

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