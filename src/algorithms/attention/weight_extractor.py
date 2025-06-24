"""Utilities for extracting and processing attention weights."""

import torch
from typing import Optional, Dict, List, Any
from dataclasses import dataclass


@dataclass
class AttentionWeights:
    """Container for attention weights from multiple layers/heads."""
    weights: torch.Tensor  # [layers, heads, seq, seq]
    layer_names: List[str]
    num_heads: int
    

class AttentionWeightExtractor:
    """Extracts attention weights from various model architectures."""
    
    def __init__(self):
        self.extraction_hooks = []
        self.captured_attentions = {}
        
    def register_attention_hook(self, module: torch.nn.Module, layer_name: str):
        """Register a hook to capture attention weights."""
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                # Many models return (output, attention_weights)
                if output[1] is not None:
                    self.captured_attentions[layer_name] = output[1]
            elif hasattr(output, 'attentions'):
                self.captured_attentions[layer_name] = output.attentions
                
        handle = module.register_forward_hook(hook)
        self.extraction_hooks.append(handle)
        
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.extraction_hooks:
            hook.remove()
        self.extraction_hooks.clear()
        self.captured_attentions.clear()
        
    def extract_from_model_output(
        self, 
        outputs: Any,
        model_type: str = "auto"
    ) -> Optional[AttentionWeights]:
        """
        Extract attention weights from model outputs.
        
        Args:
            outputs: Model output (varies by architecture)
            model_type: Type of model (auto, llama, mistral, qwen)
            
        Returns:
            AttentionWeights object or None
        """
        # Handle different output formats
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            # Standard transformers output format
            attentions = outputs.attentions
            weights = torch.stack(attentions)
            
            return AttentionWeights(
                weights=weights,
                layer_names=[f"layer_{i}" for i in range(len(attentions))],
                num_heads=weights.shape[2]
            )
            
        # Check captured attentions from hooks
        if self.captured_attentions:
            layers = []
            names = []
            
            for name, attn in sorted(self.captured_attentions.items()):
                layers.append(attn)
                names.append(name)
                
            if layers:
                weights = torch.stack(layers)
                return AttentionWeights(
                    weights=weights,
                    layer_names=names,
                    num_heads=weights.shape[2]
                )
                
        return None
    
    def compute_head_importance(
        self, 
        attention_weights: AttentionWeights
    ) -> torch.Tensor:
        """
        Compute importance scores for each attention head.
        
        Returns:
            Tensor of shape [layers, heads] with importance scores
        """
        weights = attention_weights.weights  # [layers, heads, seq, seq]
        
        # Compute entropy of attention distribution for each head
        # Lower entropy = more focused attention = higher importance
        attention_probs = torch.softmax(weights, dim=-1)
        entropy = -torch.sum(
            attention_probs * torch.log(attention_probs + 1e-10), 
            dim=-1
        ).mean(dim=-1)  # [layers, heads]
        
        # Invert entropy to get importance (low entropy = high importance)
        max_entropy = torch.log(torch.tensor(weights.shape[-1], dtype=torch.float))
        importance = 1.0 - (entropy / max_entropy)
        
        return importance
    
    def get_attention_statistics(
        self, 
        attention_weights: AttentionWeights
    ) -> Dict[str, torch.Tensor]:
        """
        Compute various statistics about attention patterns.
        
        Returns:
            Dictionary with attention statistics
        """
        weights = attention_weights.weights
        
        stats = {
            'mean_attention': weights.mean(dim=[0, 1]),  # [seq, seq]
            'max_attention': weights.max(dim=1)[0].max(dim=0)[0],  # [seq, seq]
            'attention_entropy': self._compute_entropy(weights),
            'head_importance': self.compute_head_importance(attention_weights),
            'layer_avg_attention': weights.mean(dim=[1, 2, 3]),  # [layers]
        }
        
        return stats
    
    def _compute_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention distributions."""
        probs = torch.softmax(weights, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.mean(dim=[0, 1])  # Average over layers and heads