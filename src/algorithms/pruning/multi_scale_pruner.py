"""Multi-scale attention analysis for pruning."""

import torch
from typing import List, Dict, Optional, Tuple


class MultiScaleAttentionPruner:
    """Analyzes attention at multiple scales for pruning decisions."""
    
    def __init__(
        self,
        num_layers_to_use: Optional[int] = None,
        layer_weights: Optional[List[float]] = None,
        use_sigmoid_decision: bool = True,
        sigmoid_steepness: float = 10.0
    ):
        self.num_layers_to_use = num_layers_to_use
        self.layer_weights = layer_weights
        self.use_sigmoid_decision = use_sigmoid_decision
        self.sigmoid_steepness = sigmoid_steepness
        
    def extract_multi_scale_attention(
        self,
        attention_weights: torch.Tensor,
        target_positions: List[int],
        source_positions: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention patterns at multiple scales.
        
        Args:
            attention_weights: [layers, heads, seq, seq]
            target_positions: Positions to analyze
            source_positions: Source positions
            
        Returns:
            Dictionary with attention at different scales
        """
        num_layers = attention_weights.size(0)
        layers_to_use = self.num_layers_to_use or num_layers
        
        # Select layers (default: use later layers)
        if layers_to_use < num_layers:
            layer_indices = list(range(num_layers - layers_to_use, num_layers))
        else:
            layer_indices = list(range(num_layers))
            
        scales = {}
        
        # Layer-wise attention
        layer_attention = []
        for layer_idx in layer_indices:
            layer_attn = attention_weights[layer_idx]  # [heads, seq, seq]
            # Average across heads
            layer_attn = layer_attn.mean(dim=0)  # [seq, seq]
            layer_attention.append(layer_attn)
        scales['layer_wise'] = torch.stack(layer_attention)
        
        # Head-wise attention (for each layer)
        head_attention = []
        for layer_idx in layer_indices:
            head_attn = attention_weights[layer_idx]  # [heads, seq, seq]
            head_attention.append(head_attn)
        scales['head_wise'] = torch.stack(head_attention)
        
        # Global attention (all layers and heads averaged)
        global_attn = attention_weights[layer_indices].mean(dim=[0, 1])
        scales['global'] = global_attn
        
        # Local attention (within-layer patterns)
        local_patterns = self._compute_local_patterns(
            attention_weights[layer_indices],
            target_positions,
            source_positions
        )
        scales['local'] = local_patterns
        
        return scales
    
    def _compute_local_patterns(
        self,
        attention_weights: torch.Tensor,
        target_positions: List[int],
        source_positions: List[int],
        window_size: int = 5
    ) -> torch.Tensor:
        """Compute local attention patterns within a window."""
        num_targets = len(target_positions)
        local_scores = torch.zeros(num_targets)
        
        for i, target_pos in enumerate(target_positions):
            # Define local window around target
            window_start = max(0, target_pos - window_size)
            window_end = min(attention_weights.size(-1), target_pos + window_size + 1)
            
            # Get attention within window
            local_sources = [
                s for s in source_positions 
                if window_start <= s < window_end
            ]
            
            if local_sources:
                local_attn = attention_weights[:, :, local_sources, target_pos]
                local_scores[i] = local_attn.mean()
                
        return local_scores
    
    def aggregate_multi_scale_scores(
        self,
        scales: Dict[str, torch.Tensor],
        target_positions: List[int],
        source_positions: List[int]
    ) -> torch.Tensor:
        """
        Aggregate attention scores across scales.
        
        Returns:
            Aggregated scores for each target position
        """
        scores = []
        
        for i, target_pos in enumerate(target_positions):
            # Extract scores at different scales
            scale_scores = []
            
            # Global scale
            global_score = scales['global'][source_positions, target_pos].mean()
            scale_scores.append(global_score)
            
            # Layer-wise scale (with optional weighting)
            layer_scores = scales['layer_wise'][:, source_positions, target_pos]
            if self.layer_weights is not None:
                weights = torch.tensor(self.layer_weights[:layer_scores.size(0)])
                layer_score = (layer_scores.mean(dim=1) * weights).sum()
            else:
                layer_score = layer_scores.mean()
            scale_scores.append(layer_score)
            
            # Local scale
            if i < scales['local'].size(0):
                scale_scores.append(scales['local'][i])
                
            # Aggregate with equal weighting
            aggregated = torch.stack(scale_scores).mean()
            scores.append(aggregated)
            
        return torch.stack(scores)
    
    def make_pruning_decision(
        self,
        attention_score: torch.Tensor,
        threshold: float
    ) -> bool:
        """
        Make pruning decision based on attention score.
        
        Args:
            attention_score: Attention score for token
            threshold: Pruning threshold
            
        Returns:
            True if token should be kept, False if pruned
        """
        if self.use_sigmoid_decision:
            # Soft decision with sigmoid
            keep_prob = torch.sigmoid(
                self.sigmoid_steepness * (attention_score - threshold)
            )
            return keep_prob > 0.5
        else:
            # Hard threshold
            return attention_score > threshold
    
    def compute_layer_importance(
        self,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance weights for each layer.
        
        Returns:
            Importance scores for each layer
        """
        num_layers = attention_weights.size(0)
        
        # Compute attention entropy for each layer
        layer_entropy = []
        for layer_idx in range(num_layers):
            layer_attn = attention_weights[layer_idx].mean(dim=0)  # [seq, seq]
            
            # Compute entropy of attention distribution
            attn_probs = torch.softmax(layer_attn, dim=-1)
            entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-10), dim=-1)
            layer_entropy.append(entropy.mean())
            
        layer_entropy = torch.stack(layer_entropy)
        
        # Lower entropy = more focused = higher importance
        max_entropy = torch.log(torch.tensor(attention_weights.size(-1), dtype=torch.float))
        importance = 1.0 - (layer_entropy / max_entropy)
        
        # Weight later layers more heavily
        depth_weights = torch.linspace(0.5, 1.0, num_layers)
        importance = importance * depth_weights
        
        # Normalize
        importance = importance / importance.sum()
        
        return importance