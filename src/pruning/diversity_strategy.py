import torch
import numpy as np
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
from .pruning_strategy import PruningStrategy

class DiversityPruningStrategy(PruningStrategy):
    """
    Pruning strategy based on semantic diversity.
    Intentionally preserves tokens that represent different semantic pathways.
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        num_clusters: int = 3,
        device: str = "mps"
    ):
        """
        Initialize the diversity pruning strategy.
        
        Args:
            model: The language model
            tokenizer: HuggingFace tokenizer
            num_clusters: Number of clusters to use for diversity-optimized pruning
            device: Device to use for computation
        """
        super().__init__(model, tokenizer, device)
        self.num_clusters = num_clusters
    
    def prune_tokens(
        self, 
        input_ids: torch.Tensor, 
        parallel_tokens: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Prune tokens based on semantic diversity.
        
        This function implements the diversity-optimized pruning strategy, which:
        1. Groups parallel tokens by semantic similarity
        2. Selects representatives from different clusters
        3. Preserves representational richness across the probability space
        
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
        
        # Get token embeddings
        token_embeddings = self._get_token_embeddings(input_ids, token_ids)
        token_embeddings_np = token_embeddings.cpu().numpy()
        
        # Determine number of clusters based on number of tokens
        n_clusters = min(self.num_clusters, len(token_ids))
        
        # Cluster token embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(token_embeddings_np)
        
        # Select representative tokens from each cluster based on probability
        pruned_tokens = []
        for cluster_id in range(n_clusters):
            # Get indices of tokens in this cluster
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if cluster_indices:
                # Select token with highest probability in this cluster
                cluster_tokens = [(token_ids[i], parallel_tokens[i][1]) for i in cluster_indices]
                best_token = max(cluster_tokens, key=lambda x: x[1])
                pruned_tokens.append(best_token)
        
        return pruned_tokens
    
    def get_scored_tokens(
        self, 
        input_ids: torch.Tensor, 
        parallel_tokens: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Get tokens with their diversity scores.
        For diversity strategy, we simply use the original probabilities as scores.
        
        Args:
            input_ids: Current input token IDs
            parallel_tokens: List of (token_id, probability) tuples
            
        Returns:
            List[Tuple[int, float]]: List of (token_id, score) tuples
        """
        # For diversity strategy, we simply use the original probabilities as scores
        return [(token_id, prob) for token_id, prob in parallel_tokens]
    
    def _get_token_embeddings(
        self, 
        input_ids: torch.Tensor, 
        token_ids: List[int]
    ) -> torch.Tensor:
        """
        Get embeddings for tokens to analyze their semantic diversity.
        
        Args:
            input_ids: Current input token IDs
            token_ids: List of token IDs to analyze
            
        Returns:
            torch.Tensor: Embeddings for each token
        """
        # Batch size should be 1 for generation
        batch_size, seq_len = input_ids.shape
        
        # Create sequences with each token at the same position
        test_sequences = []
        
        for token_id in token_ids:
            new_seq = torch.zeros((1, seq_len + 1), dtype=input_ids.dtype, device=self.device)
            new_seq[0, :seq_len] = input_ids[0]  # Copy existing tokens
            new_seq[0, seq_len] = token_id  # Add test token
            test_sequences.append(new_seq)
            
        # Stack sequences into a batch
        batch_input_ids = torch.cat(test_sequences, dim=0)
        
        # Create appropriate attention mask
        attention_mask = torch.ones_like(batch_input_ids)
        
        # Run model to get hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            
        # Get last layer hidden states
        # Shape: [batch_size, seq_len+1, hidden_size]
        last_layer_hidden = outputs.hidden_states[-1]
        
        # Extract hidden state of last token (each token)
        last_token_hidden = last_layer_hidden[:, -1, :]  # [batch_size, hidden_size]
        
        return last_token_hidden 