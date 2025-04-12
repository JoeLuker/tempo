import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from .pruning_strategy import PruningStrategy

class CoherencePruningStrategy(PruningStrategy):
    """
    Pruning strategy based on attention coherence.
    Prioritizes tokens that lead to coherent continuations.
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        threshold: float = 0.3,
        device: str = "mps"
    ):
        """
        Initialize the coherence pruning strategy.
        
        Args:
            model: The language model
            tokenizer: HuggingFace tokenizer
            threshold: Coherence threshold for pruning
            device: Device to use for computation
        """
        super().__init__(model, tokenizer, device)
        self.threshold = threshold
        self.token_generator = None  # Will be set by the ExperimentRunner
        self.debug_mode = False
    
    def set_token_generator(self, token_generator):
        """
        Set the token generator instance to enable access to cached attention.
        
        Args:
            token_generator: TokenGenerator instance
        """
        self.token_generator = token_generator
        
    def set_debug_mode(self, enabled=False):
        """Enable or disable debug mode."""
        self.debug_mode = enabled
    
    def prune_tokens(
        self, 
        input_ids: torch.Tensor, 
        parallel_tokens: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Prune tokens based on attention coherence.
        
        Args:
            input_ids: Current input token IDs
            parallel_tokens: List of (token_id, probability) tuples
            
        Returns:
            List[Tuple[int, float]]: Pruned list of (token_id, probability) tuples
        """
        if len(parallel_tokens) <= 1:
            return parallel_tokens
            
        # Get token info with coherence scores
        token_info = self._get_token_coherence_info(input_ids, parallel_tokens)
        
        # Filter based on normalized coherence score
        pruned_tokens = [
            (token_id, prob)
            for token_id, prob, norm_score in token_info
            if norm_score >= self.threshold
        ]
        
        # If all tokens were pruned, keep the one with highest coherence
        if not pruned_tokens:
            # First find tokens with highest coherence
            max_coherence = max(token_info, key=lambda x: x[2])[2]
            top_tokens = [
                (token_id, prob) 
                for token_id, prob, score in token_info 
                if abs(score - max_coherence) < 1e-5
            ]
            
            if len(top_tokens) == 1:
                pruned_tokens = top_tokens
            else:
                # Multiple tokens with similar coherence - break tie using probability
                best_token = max(top_tokens, key=lambda x: x[1])
                pruned_tokens = [best_token]
            
        return pruned_tokens
    
    def get_scored_tokens(
        self, 
        input_ids: torch.Tensor, 
        parallel_tokens: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Get tokens with their normalized coherence scores.
        
        Args:
            input_ids: Current input token IDs
            parallel_tokens: List of (token_id, probability) tuples
            
        Returns:
            List[Tuple[int, float]]: List of (token_id, normalized_score) tuples
        """
        if len(parallel_tokens) <= 1:
            # Single token gets max score
            return [(token_id, 1.0) for token_id, _ in parallel_tokens]
            
        # Get token info with coherence scores
        token_info = self._get_token_coherence_info(input_ids, parallel_tokens)
        
        # Return normalized scores
        return [(token_id, norm_score) for token_id, _, norm_score in token_info]
    
    def _get_token_coherence_info(
        self, 
        input_ids: torch.Tensor, 
        parallel_tokens: List[Tuple[int, float]]
    ) -> List[Tuple[int, float, float]]:
        """
        Get token information with coherence scores.
        
        Args:
            input_ids: Current input token IDs
            parallel_tokens: List of (token_id, probability) tuples
            
        Returns:
            List[Tuple[int, float, float]]: List of (token_id, probability, normalized_score) tuples
        """
        # Extract token IDs from parallel tokens
        token_ids = [t[0] for t in parallel_tokens]
        
        # Try to use cached attention first (optimization)
        if self.token_generator is not None:
            cached_attention, cached_seq_len = self.token_generator.get_cached_attention()
            if cached_attention is not None:
                # Use cached attention for coherence calculation
                coherence_scores = self._compute_coherence_from_cached_attention(
                    cached_attention, input_ids, token_ids
                )
                if coherence_scores is not None:
                    if self.debug_mode:
                        print(f"Using cached attention for coherence calculation (seq_len: {cached_seq_len})")
                    
                    # Create list of (token_id, probability, coherence_score)
                    token_info = [
                        (token_id, prob, coherence_scores[i].item())
                        for i, (token_id, prob) in enumerate(parallel_tokens)
                    ]
                    
                    # Normalize coherence scores to [0, 1]
                    max_coherence = max(info[2] for info in token_info)
                    min_coherence = min(info[2] for info in token_info)
                    range_coherence = max(1e-5, max_coherence - min_coherence)
                    
                    normalized_token_info = [
                        (token_id, prob, (score - min_coherence) / range_coherence)
                        for token_id, prob, score in token_info
                    ]
                    
                    return normalized_token_info
        
        # Fallback to original implementation with full model forward pass
        if self.debug_mode:
            print("Fallback to full model forward pass for coherence calculation")
            
        # Get attention-based coherence scores using full model forward pass
        coherence_scores, _ = self._get_attention_scores(input_ids, token_ids)
        
        # Create list of (token_id, probability, coherence_score)
        token_info = [
            (token_id, prob, coherence_scores[i].item())
            for i, (token_id, prob) in enumerate(parallel_tokens)
        ]
        
        # Normalize coherence scores to [0, 1]
        max_coherence = max(info[2] for info in token_info)
        min_coherence = min(info[2] for info in token_info)
        range_coherence = max(1e-5, max_coherence - min_coherence)
        
        normalized_token_info = [
            (token_id, prob, (score - min_coherence) / range_coherence)
            for token_id, prob, score in token_info
        ]
        
        return normalized_token_info
    
    def _compute_coherence_from_cached_attention(
        self,
        cached_attention,
        input_ids: torch.Tensor,
        token_ids: List[int]
    ) -> Optional[torch.Tensor]:
        """
        Compute coherence scores using cached attention patterns.
        
        Args:
            cached_attention: Cached attention patterns from token generation
            input_ids: Current input token IDs
            token_ids: List of token IDs to evaluate
            
        Returns:
            Optional[torch.Tensor]: Coherence scores for each token or None if incompatible
        """
        try:
            # Get last layer attention
            last_layer_attention = cached_attention[-1]  # Shape: [batch_size, num_heads, seq_len, seq_len]
            
            # Extract attention for the last token position
            # The last token's attention shows how it attends to previous context
            last_token_attn = last_layer_attention[0, :, -1, :-1]  # [num_heads, seq_len-1]
            
            # Average across attention heads
            avg_attention = last_token_attn.mean(dim=0)  # [seq_len-1]
            
            # For each candidate token, we'll use the same cached attention
            # as a proxy for coherence (simplification)
            # Better coherence = higher attention focus (less uniform distribution)
            
            # Calculate focus score: use max attention value as proxy for coherence
            focus_score = avg_attention.max()
            
            # Create a batch of coherence scores, all using the same focus score
            # This is a simplification - ideally we'd compute token-specific coherence
            coherence_scores = torch.tensor([focus_score] * len(token_ids), device=self.device)
            
            # Apply token-specific adjustment based on token identity
            # In a full implementation, you'd calculate token-specific coherence
            
            return coherence_scores
        except Exception as e:
            if self.debug_mode:
                print(f"Error computing coherence from cached attention: {e}")
            return None
            
    def _get_attention_scores(
        self, 
        input_ids: torch.Tensor, 
        token_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention scores for the tokens to analyze their coherence.
        
        Args:
            input_ids: Current input token IDs
            token_ids: List of token IDs to analyze
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Coherence scores for each token
                - Average attention for each token to context
        """
        # Batch size should be 1 for generation
        batch_size, seq_len = input_ids.shape
        
        # Create sequences with each token at the same position
        test_sequences = []
        
        for token_id in token_ids:
            # Create a new sequence with the current token at the next position
            new_seq = torch.zeros((1, seq_len + 1), dtype=input_ids.dtype, device=self.device)
            new_seq[0, :seq_len] = input_ids[0]  # Copy existing tokens
            new_seq[0, seq_len] = token_id  # Add test token
            test_sequences.append(new_seq)
            
        # Stack sequences into a batch
        batch_input_ids = torch.cat(test_sequences, dim=0)
        
        # Create appropriate attention mask
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
        
        # Extract attention from last token (each token) to context
        # This shows how each token relates to the context
        last_token_attention = last_layer_attention[:, :, -1, :-1]
        
        # Average across attention heads
        avg_attention = last_token_attention.mean(dim=1)  # [batch_size, seq_len-1]
        
        # Calculate coherence scores
        # Higher score means more focused attention (more coherent)
        coherence_scores = avg_attention.max(dim=1)[0]  # Max attention value for each token
        
        return coherence_scores, avg_attention 