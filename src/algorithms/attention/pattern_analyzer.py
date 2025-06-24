"""Attention pattern analysis for pruning decisions."""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any


class AttentionPatternAnalyzer:
    """Analyzes attention patterns to identify important tokens."""
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        device: str = "cuda"
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.device = device
        self.attention_cache: Dict[int, torch.Tensor] = {}
        
    def extract_attention_weights(
        self,
        model_output: Any,
        layer_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Extract attention weights from model output.
        
        Args:
            model_output: Output from transformer model
            layer_indices: Specific layers to extract (None = all)
            
        Returns:
            Attention weights tensor [layers, heads, seq, seq]
        """
        if not hasattr(model_output, 'attentions') or model_output.attentions is None:
            raise ValueError("Model output does not contain attention weights")
            
        attentions = model_output.attentions
        
        if layer_indices:
            attentions = [attentions[i] for i in layer_indices]
            
        # Stack all layers
        attention_tensor = torch.stack(attentions)  # [layers, batch, heads, seq, seq]
        
        # Remove batch dimension (assuming batch size 1)
        if attention_tensor.size(1) == 1:
            attention_tensor = attention_tensor.squeeze(1)
            
        return attention_tensor
    
    def compute_attention_to_positions(
        self,
        attention_weights: torch.Tensor,
        target_positions: List[int],
        source_positions: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute attention from source to target positions.
        
        Args:
            attention_weights: Attention tensor [layers, heads, seq, seq]
            target_positions: Positions being attended to
            source_positions: Positions doing the attending (None = all future)
            
        Returns:
            Attention scores for target positions
        """
        num_layers, num_heads, seq_len, _ = attention_weights.shape
        
        # Default source positions: all positions after targets
        if source_positions is None:
            max_target = max(target_positions)
            source_positions = list(range(max_target + 1, seq_len))
            
        if not source_positions:
            return torch.zeros(len(target_positions), device=attention_weights.device)
            
        # Extract relevant attention values
        attention_to_targets = []
        
        for target_idx, target_pos in enumerate(target_positions):
            # Get attention from all source positions to this target
            attn_values = attention_weights[:, :, source_positions, target_pos]
            # Average across layers and heads
            avg_attention = attn_values.mean(dim=[0, 1, 2])
            attention_to_targets.append(avg_attention)
            
        return torch.stack(attention_to_targets)
    
    def compute_relative_attention(
        self,
        attention_scores: torch.Tensor,
        parallel_set: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Compute relative attention within a parallel set.
        
        Args:
            attention_scores: Attention scores for positions
            parallel_set: (start, end) of parallel token set
            
        Returns:
            Relative attention scores (normalized within set)
        """
        start, end = parallel_set
        set_scores = attention_scores[start:end]
        
        if len(set_scores) == 0:
            return set_scores
            
        # Normalize within the set
        if set_scores.sum() > 0:
            relative_scores = set_scores / set_scores.sum()
        else:
            # Uniform if all zeros
            relative_scores = torch.ones_like(set_scores) / len(set_scores)
            
        return relative_scores
    
    def aggregate_multi_layer_attention(
        self,
        attention_weights: torch.Tensor,
        layer_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate attention across multiple layers.
        
        Args:
            attention_weights: [layers, heads, seq, seq]
            layer_weights: Weights for each layer (None = uniform)
            
        Returns:
            Aggregated attention [seq, seq]
        """
        num_layers = attention_weights.size(0)
        
        # Default: weight later layers more heavily
        if layer_weights is None:
            layer_weights = torch.linspace(
                0.5, 1.0, num_layers, 
                device=attention_weights.device
            )
            layer_weights = layer_weights / layer_weights.sum()
            
        # First average across heads
        layer_attention = attention_weights.mean(dim=1)  # [layers, seq, seq]
        
        # Then weighted average across layers
        aggregated = torch.zeros_like(layer_attention[0])
        for i, weight in enumerate(layer_weights):
            aggregated += weight * layer_attention[i]
            
        return aggregated
    
    def identify_low_attention_tokens(
        self,
        attention_scores: torch.Tensor,
        threshold: float,
        use_sigmoid: bool = True,
        sigmoid_steepness: float = 10.0
    ) -> List[int]:
        """
        Identify tokens with low attention scores.
        
        Args:
            attention_scores: Attention scores for each position
            threshold: Attention threshold
            use_sigmoid: Use soft sigmoid decision boundary
            sigmoid_steepness: Steepness of sigmoid curve
            
        Returns:
            List of positions to prune
        """
        positions_to_prune = []
        
        for i, score in enumerate(attention_scores):
            if use_sigmoid:
                # Soft decision with sigmoid
                prune_prob = 1 / (1 + torch.exp(-sigmoid_steepness * (threshold - score)))
                if prune_prob > 0.5:
                    positions_to_prune.append(i)
            else:
                # Hard threshold
                if score < threshold:
                    positions_to_prune.append(i)
                    
        return positions_to_prune
    
    def clear_cache(self):
        """Clear attention cache."""
        self.attention_cache.clear()