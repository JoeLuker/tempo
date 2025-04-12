import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from .pruning_strategy import PruningStrategy
import math

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
        
        # Use cached attention for coherence calculation (optimization)
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
            
            # If we reach here and have no cached attention, log the issue
            if self.debug_mode:
                print("WARNING: No cached attention available! Creating fallback coherence scores.")
        
        # FALLBACK: If no cached attention, use token probabilities as scores
        # This avoids the expensive model forward pass while still providing reasonable pruning
        token_info = []
        for i, (token_id, prob) in enumerate(parallel_tokens):
            # Use probability as fallback coherence measure
            # This way we avoid the expensive forward pass completely
            token_info.append((token_id, prob, prob))
        
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
            # Get the attention patterns from all layers
            num_layers = len(cached_attention)
            
            # Extract attention tensors from the last few layers
            # Using multiple layers tends to give better coherence signals
            layers_to_use = min(3, num_layers)  # Use last 3 layers or all if fewer
            attention_layers = cached_attention[-layers_to_use:]
            
            # Average attention patterns across selected layers
            # Shape: [batch_size, num_heads, seq_len, seq_len]
            avg_layer_attention = torch.mean(torch.stack([layer for layer in attention_layers]), dim=0)
            
            # Extract attention for the last token position
            # The last token's attention shows how it attends to previous context
            last_token_attn = avg_layer_attention[0, :, -1, :-1]  # [num_heads, seq_len-1]
            
            # Average across attention heads
            avg_attention = last_token_attn.mean(dim=0)  # [seq_len-1]
            
            # Calculate entropy of attention distribution as a coherence measure
            # Lower entropy = more focused attention = better coherence
            normalized_attn = avg_attention / (torch.sum(avg_attention) + 1e-10)
            entropy = -torch.sum(normalized_attn * torch.log(normalized_attn + 1e-10))
            
            # We don't need to run a full forward pass for each token
            # Instead, use token embedding similarity as a proxy for how well each token fits
            embedding_vectors = self.get_token_embeddings(token_ids)
            
            # Get the context representation
            context_vector = self.get_context_embedding(input_ids)
            
            # Calculate coherence scores using embedding similarity combined with attention entropy
            similarities = torch.nn.functional.cosine_similarity(embedding_vectors, context_vector.unsqueeze(0), dim=1)
            
            # Normalize similarities to [0,1] range
            min_sim = torch.min(similarities)
            max_sim = torch.max(similarities)
            range_sim = max(1e-5, max_sim - min_sim)
            norm_similarities = (similarities - min_sim) / range_sim
            
            # Combine similarity and entropy-based coherence
            # Lower entropy is better, so use 1 - normalized_entropy
            normalized_entropy = entropy / math.log(avg_attention.size(0))  # Normalize by max possible entropy
            coherence_factor = 1.0 - min(1.0, normalized_entropy.item())
            
            # Final coherence scores combine token similarity and attention pattern coherence
            coherence_scores = norm_similarities * coherence_factor
            
            if self.debug_mode:
                print(f"Coherence factor from attention: {coherence_factor:.4f}")
                print(f"Token similarities range: {min_sim.item():.4f} to {max_sim.item():.4f}")
                print(f"Resulting coherence scores: {coherence_scores}")
            
            return coherence_scores
        except Exception as e:
            if self.debug_mode:
                print(f"Error computing coherence from cached attention: {e}")
                import traceback
                traceback.print_exc()
            return None

    def get_token_embeddings(self, token_ids: List[int]) -> torch.Tensor:
        """
        Get embedding vectors for tokens to calculate similarity.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            torch.Tensor: Embedding vectors [num_tokens, embedding_dim]
        """
        # Convert token IDs to tensor
        tokens_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        
        # Get embedding layer from model
        if hasattr(self.model, "get_input_embeddings"):
            embedding_layer = self.model.get_input_embeddings()
            # Get embedding vectors
            with torch.no_grad():
                embeddings = embedding_layer(tokens_tensor)
            return embeddings
        
        # Fallback if model doesn't expose embedding layer
        return torch.zeros((len(token_ids), 768), device=self.device)

    def get_context_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get a representation of the context for similarity comparison.
        
        Args:
            input_ids: Current input token IDs
            
        Returns:
            torch.Tensor: Context representation vector [embedding_dim]
        """
        # Get embedding layer from model
        if hasattr(self.model, "get_input_embeddings"):
            embedding_layer = self.model.get_input_embeddings()
            
            # Get embeddings for all tokens in the context
            with torch.no_grad():
                # Take last few tokens as context (recency bias)
                context_length = min(20, input_ids.size(1))
                context_ids = input_ids[0, -context_length:]
                context_embeddings = embedding_layer(context_ids)
                
                # Average the embeddings to get context representation
                return torch.mean(context_embeddings, dim=0)
        
        # Fallback if model doesn't expose embedding layer
        return torch.zeros(768, device=self.device)
    
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