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
        
        # Vectorized approach to select representative tokens from each cluster
        # Create arrays for vectorized processing
        token_ids_array = np.array(token_ids)
        probs_array = np.array([prob for _, prob in parallel_tokens])
        
        # Process all clusters at once
        pruned_tokens = []
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            # Get mask for this cluster
            cluster_mask = (cluster_labels == cluster_id)
            
            if np.any(cluster_mask):
                # Find index with highest probability in this cluster
                cluster_probs = probs_array[cluster_mask]
                max_prob_idx = np.argmax(cluster_probs)
                
                # Get the original index in the token list
                original_indices = np.where(cluster_mask)[0]
                best_idx = original_indices[max_prob_idx]
                
                # Add the token with its probability
                pruned_tokens.append((token_ids[best_idx], parallel_tokens[best_idx][1]))
        
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
        
        # Vectorized approach to create all test sequences at once
        # Create a base sequence template that we'll reuse
        num_tokens = len(token_ids)
        
        # Create a batch tensor directly with all sequences
        batch_input_ids = torch.zeros((num_tokens, seq_len + 1), dtype=input_ids.dtype, device=self.device)
        
        # Fill in the common prefix (broadcast the input_ids to all rows)
        batch_input_ids[:, :seq_len] = input_ids[0].expand(num_tokens, -1)
        
        # Set the last position of each sequence to the corresponding token ID
        batch_input_ids[:, seq_len] = torch.tensor(token_ids, dtype=input_ids.dtype, device=self.device)
        
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