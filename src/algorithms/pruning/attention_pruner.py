"""Attention-based pruning for parallel tokens."""

import torch
from typing import List, Tuple, Optional, Dict


class AttentionBasedPruner:
    """Prunes parallel tokens based on attention patterns."""
    
    def __init__(
        self,
        attention_threshold: float = 0.01,
        use_relative_attention: bool = True,
        relative_threshold: float = 0.5
    ):
        self.attention_threshold = attention_threshold
        self.use_relative_attention = use_relative_attention
        self.relative_threshold = relative_threshold
        
    def compute_pruning_scores(
        self,
        attention_weights: torch.Tensor,
        target_positions: List[int],
        source_positions: List[int]
    ) -> torch.Tensor:
        """
        Compute pruning scores based on attention.
        
        Args:
            attention_weights: Attention tensor [layers, heads, seq, seq]
            target_positions: Positions of tokens to score
            source_positions: Positions attending to targets
            
        Returns:
            Pruning scores for each target position
        """
        if not source_positions:
            return torch.zeros(len(target_positions))
            
        # Extract attention to targets from sources
        scores = []
        
        for target_pos in target_positions:
            # Get attention from all sources to this target
            attn_to_target = attention_weights[:, :, source_positions, target_pos]
            
            # Average across layers and heads
            avg_attention = attn_to_target.mean()
            scores.append(avg_attention)
            
        return torch.tensor(scores)
    
    def apply_relative_pruning(
        self,
        scores: torch.Tensor,
        parallel_set: Tuple[int, int]
    ) -> List[bool]:
        """
        Apply relative pruning within a parallel set.
        
        Args:
            scores: Attention scores for positions
            parallel_set: (start, end) indices of parallel set
            
        Returns:
            List of keep/prune decisions
        """
        start, end = parallel_set
        set_scores = scores[start:end]
        
        if len(set_scores) <= 1:
            return [True] * len(set_scores)
            
        # Normalize scores within set
        if self.use_relative_attention:
            # Keep tokens with above-average attention in their set
            mean_score = set_scores.mean()
            threshold = mean_score * self.relative_threshold
            keep_mask = set_scores > threshold
        else:
            # Use absolute threshold
            keep_mask = set_scores > self.attention_threshold
            
        # Always keep at least one token (highest attention)
        if not keep_mask.any():
            best_idx = set_scores.argmax()
            keep_mask[best_idx] = True
            
        return keep_mask.tolist()
    
    def compute_attention_flow(
        self,
        attention_weights: torch.Tensor,
        path: List[int]
    ) -> float:
        """
        Compute attention flow along a token path.
        
        Args:
            attention_weights: Full attention matrix
            path: List of token positions in path
            
        Returns:
            Attention flow score
        """
        if len(path) < 2:
            return 1.0
            
        flow_score = 1.0
        
        # Compute product of attention along path
        for i in range(len(path) - 1):
            from_pos = path[i]
            to_pos = path[i + 1]
            
            # Average attention across layers and heads
            attention = attention_weights[:, :, to_pos, from_pos].mean()
            flow_score *= attention.item()
            
        return flow_score
    
    def identify_coherent_paths(
        self,
        attention_weights: torch.Tensor,
        parallel_sets: List[Tuple[int, int]],
        min_flow: float = 0.1
    ) -> List[List[int]]:
        """
        Identify coherent paths through parallel tokens.
        
        Args:
            attention_weights: Attention tensor
            parallel_sets: List of parallel token sets
            min_flow: Minimum attention flow for coherent path
            
        Returns:
            List of coherent paths (token position sequences)
        """
        coherent_paths = []
        
        # Build paths recursively
        def build_paths(current_path: List[int], set_idx: int):
            if set_idx >= len(parallel_sets):
                # Evaluate complete path
                flow = self.compute_attention_flow(attention_weights, current_path)
                if flow >= min_flow:
                    coherent_paths.append(current_path.copy())
                return
                
            start, end = parallel_sets[set_idx]
            
            # Try each token in the set
            for pos in range(start, end):
                current_path.append(pos)
                build_paths(current_path, set_idx + 1)
                current_path.pop()
        
        # Start building from first set
        if parallel_sets:
            start, end = parallel_sets[0]
            for pos in range(start, end):
                build_paths([pos], 1)
                
        return coherent_paths