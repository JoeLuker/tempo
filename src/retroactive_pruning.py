import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

class RetroactivePruner:
    """
    Implements the Retroactive Pruning mechanism for the Parallel Threshold Output generator.
    
    This class analyzes parallel token sets and prunes tokens that are less likely to
    contribute to a coherent continuation based on attention patterns.
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        coherence_threshold: float = 0.3,
        device: str = "mps"
    ):
        """
        Initialize the pruner.
        
        Args:
            model: The Mistral-7B model
            tokenizer: HuggingFace tokenizer
            coherence_threshold: Threshold for pruning tokens based on attention coherence
            device: Device to use for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.coherence_threshold = coherence_threshold
        self.device = device
    
    def _get_attention_scores(
        self, 
        input_ids: torch.Tensor, 
        parallel_token_ids: List[int]
    ) -> torch.Tensor:
        """
        Get attention scores for the parallel tokens to analyze their coherence.
        
        Args:
            input_ids: Current input token IDs
            parallel_token_ids: List of token IDs in the current parallel set
            
        Returns:
            torch.Tensor: Attention scores for each token
        """
        # Create a sequence with each parallel token appended to input_ids
        # We'll analyze how they attend to the context
        batch_size = len(parallel_token_ids)
        sequences = []
        
        for token_id in parallel_token_ids:
            seq = torch.cat([
                input_ids, 
                torch.tensor([[token_id]], device=self.device)
            ], dim=1)
            sequences.append(seq)
            
        # Stack sequences into a batch
        batch_input_ids = torch.cat(sequences, dim=0)
        attention_mask = torch.ones_like(batch_input_ids)
        
        # Run model to get attention patterns
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            
        # Get last layer attention
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        last_layer_attention = outputs.attentions[-1]
        
        # Extract attention from last token (each parallel token) to context
        # This shows how each token relates to the context
        last_token_attention = last_layer_attention[:, :, -1, :-1]
        
        # Average across attention heads
        avg_attention = last_token_attention.mean(dim=1)  # [batch_size, seq_len-1]
        
        # Calculate an attention coherence score
        # Higher score means more focused attention (more coherent)
        # Lower score means more dispersed attention (less coherent)
        coherence_scores = avg_attention.max(dim=1)[0]  # Max attention value for each token
        
        return coherence_scores
    
    def prune_parallel_tokens(
        self, 
        input_ids: torch.Tensor, 
        parallel_tokens: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Prune parallel tokens based on attention coherence.
        
        Args:
            input_ids: Current input token IDs
            parallel_tokens: List of (token_id, probability) tuples
            
        Returns:
            List[Tuple[int, float]]: Pruned list of (token_id, probability) tuples
        """
        if len(parallel_tokens) <= 1:
            return parallel_tokens
            
        # Extract token IDs from parallel tokens
        token_ids = [t[0] for t in parallel_tokens]
        
        # Get attention-based coherence scores
        coherence_scores = self._get_attention_scores(input_ids, token_ids)
        
        # Create list of (token_id, probability, coherence_score)
        token_info = [
            (token_id, prob, coherence_scores[i].item())
            for i, (token_id, prob) in enumerate(parallel_tokens)
        ]
        
        # Normalize coherence scores to [0, 1]
        max_coherence = max(info[2] for info in token_info)
        min_coherence = min(info[2] for info in token_info)
        range_coherence = max(1e-5, max_coherence - min_coherence)  # Avoid division by zero
        
        normalized_token_info = [
            (token_id, prob, (score - min_coherence) / range_coherence)
            for token_id, prob, score in token_info
        ]
        
        # Filter based on normalized coherence score
        pruned_tokens = [
            (token_id, prob)
            for token_id, prob, norm_score in normalized_token_info
            if norm_score >= self.coherence_threshold
        ]
        
        # If all tokens were pruned, keep the one with highest coherence
        if not pruned_tokens:
            best_idx = max(range(len(normalized_token_info)), 
                          key=lambda i: normalized_token_info[i][2])
            pruned_tokens = [
                (normalized_token_info[best_idx][0], normalized_token_info[best_idx][1])
            ]
            
        return pruned_tokens 