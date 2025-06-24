"""Logits processing for token generation."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class LogitsProcessor:
    """Processes raw logits from language model."""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.vocab_size = None
        
    def process_logits(
        self,
        logits: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Process raw logits with temperature scaling.
        
        Args:
            logits: Raw logits from model [batch, seq, vocab]
            temperature: Temperature for scaling (None = use default)
            
        Returns:
            Processed logits
        """
        temp = temperature if temperature is not None else self.temperature
        
        if temp != 1.0 and temp > 0:
            logits = logits / temp
            
        return logits
    
    def get_top_k_tokens(
        self,
        logits: torch.Tensor,
        k: int,
        return_probs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k tokens and their probabilities.
        
        Args:
            logits: Logits tensor [batch, vocab] or [vocab]
            k: Number of top tokens to return
            return_probs: Return probabilities instead of logits
            
        Returns:
            Tuple of (token_ids, scores)
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            
        if return_probs:
            probs = F.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, k, dim=-1)
            return top_ids.squeeze(0), top_probs.squeeze(0)
        else:
            top_logits, top_ids = torch.topk(logits, k, dim=-1)
            return top_ids.squeeze(0), top_logits.squeeze(0)
    
    def apply_threshold(
        self,
        logits: torch.Tensor,
        threshold: float,
        min_tokens: int = 1,
        max_tokens: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select tokens above probability threshold.
        
        Args:
            logits: Logits tensor
            threshold: Probability threshold
            min_tokens: Minimum tokens to return
            max_tokens: Maximum tokens to return
            
        Returns:
            Tuple of (selected_token_ids, probabilities)
        """
        probs = F.softmax(logits, dim=-1)
        
        if logits.dim() > 1:
            probs = probs[-1]  # Take last position
            
        # Find tokens above threshold
        mask = probs > threshold
        selected_ids = torch.where(mask)[0]
        selected_probs = probs[selected_ids]
        
        # Ensure minimum tokens
        if len(selected_ids) < min_tokens:
            # Add top tokens until we have enough
            top_ids, top_probs = self.get_top_k_tokens(logits, min_tokens)
            selected_ids = top_ids[:min_tokens]
            selected_probs = top_probs[:min_tokens]
            
        # Apply maximum limit
        if max_tokens and len(selected_ids) > max_tokens:
            # Keep only the highest probability tokens
            top_indices = selected_probs.topk(max_tokens).indices
            selected_ids = selected_ids[top_indices]
            selected_probs = selected_probs[top_indices]
            
        return selected_ids, selected_probs
    
    def compute_entropy(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute entropy of probability distribution."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy
    
    def apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_ids: List[int],
        penalty: float = 1.2
    ) -> torch.Tensor:
        """
        Apply repetition penalty to discourage repeating tokens.
        
        Args:
            logits: Current logits
            generated_ids: Previously generated token IDs
            penalty: Repetition penalty factor
            
        Returns:
            Modified logits
        """
        if penalty == 1.0 or not generated_ids:
            return logits
            
        logits = logits.clone()
        
        for token_id in set(generated_ids):
            if logits.dim() == 1:
                logits[token_id] /= penalty
            else:
                logits[:, token_id] /= penalty
                
        return logits