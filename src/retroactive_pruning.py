import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class RetroactivePruner:
    """
    Implements the Retroactive Pruning mechanism for the Parallel Threshold Output generator.
    
    This class analyzes parallel token sets and prunes tokens either based on:
    1. Coherence - prioritizing tokens that lead to coherent continuations
    2. Diversity - intentionally preserving tokens that represent different semantic pathways
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        coherence_threshold: float = 0.3,
        diversity_clusters: int = 3,
        pruning_strategy: str = "coherence",
        device: str = "mps"
    ):
        """
        Initialize the pruner.
        
        Args:
            model: The Mistral-7B model
            tokenizer: HuggingFace tokenizer
            coherence_threshold: Threshold for pruning tokens based on attention coherence
            diversity_clusters: Number of clusters to use for diversity-optimized pruning
            pruning_strategy: Strategy to use, either "coherence" or "diversity"
            device: Device to use for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.coherence_threshold = coherence_threshold
        self.diversity_clusters = diversity_clusters
        self.pruning_strategy = pruning_strategy
        self.device = device
    
    def _get_attention_scores(
        self, 
        input_ids: torch.Tensor, 
        parallel_token_ids: List[int]
    ) -> torch.Tensor:
        """
        Get attention scores for the parallel tokens to analyze their coherence.
        
        This modified implementation creates test sequences where each parallel token
        is placed at the same position (matching Option A positional encoding).
        
        Args:
            input_ids: Current input token IDs
            parallel_token_ids: List of token IDs in the current parallel set
            
        Returns:
            torch.Tensor: Attention scores for each token
        """
        # Batch size should be 1 for generation
        batch_size, seq_len = input_ids.shape
        
        # Create sequences with each parallel token at the same position
        # We'll analyze how these different tokens behave when placed at the same position
        test_sequences = []
        
        for token_id in parallel_token_ids:
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
        
        # Extract attention from last token (each parallel token) to context
        # This shows how each token relates to the context
        last_token_attention = last_layer_attention[:, :, -1, :-1]
        
        # Average across attention heads
        avg_attention = last_token_attention.mean(dim=1)  # [batch_size, seq_len-1]
        
        # Calculate coherence scores
        # Higher score means more focused attention (more coherent)
        # Lower score means more dispersed attention (less coherent)
        coherence_scores = avg_attention.max(dim=1)[0]  # Max attention value for each token
        
        return coherence_scores, avg_attention
    
    def _get_token_embeddings(
        self, 
        input_ids: torch.Tensor, 
        parallel_token_ids: List[int]
    ) -> torch.Tensor:
        """
        Get embeddings for parallel tokens to analyze their semantic diversity.
        
        Args:
            input_ids: Current input token IDs
            parallel_token_ids: List of token IDs in the current parallel set
            
        Returns:
            torch.Tensor: Embeddings for each token
        """
        # Batch size should be 1 for generation
        batch_size, seq_len = input_ids.shape
        
        # Create sequences with each parallel token at the same position
        test_sequences = []
        
        for token_id in parallel_token_ids:
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
        
        # Extract hidden state of last token (each parallel token)
        last_token_hidden = last_layer_hidden[:, -1, :]  # [batch_size, hidden_size]
        
        return last_token_hidden
    
    def _diversity_optimized_pruning(
        self, 
        input_ids: torch.Tensor, 
        parallel_tokens: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Prune parallel tokens based on semantic diversity.
        
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
        n_clusters = min(self.diversity_clusters, len(token_ids))
        
        # Apply clustering to group similar tokens
        if len(token_ids) <= n_clusters:
            # If we have fewer tokens than clusters, keep all tokens
            selected_indices = list(range(len(token_ids)))
        else:
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(token_embeddings_np)
            
            # Select representative tokens from each cluster
            # For each cluster, select the token with the highest probability
            selected_indices = []
            for cluster_id in range(n_clusters):
                # Find indices of tokens in this cluster
                cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
                if cluster_indices:
                    # Select token with highest probability in this cluster
                    best_idx = max(cluster_indices, key=lambda i: parallel_tokens[i][1])
                    selected_indices.append(best_idx)
        
        # Create pruned token list
        pruned_tokens = [parallel_tokens[i] for i in selected_indices]
        
        return pruned_tokens
    
    def _coherence_optimized_pruning(
        self, 
        input_ids: torch.Tensor, 
        parallel_tokens: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Prune parallel tokens based on attention coherence.
        
        This is the original pruning strategy that prioritizes tokens
        that lead to coherent continuations.
        
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
        coherence_scores, _ = self._get_attention_scores(input_ids, token_ids)
        
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
    
    def prune_parallel_tokens(
        self, 
        input_ids: torch.Tensor, 
        parallel_tokens: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Prune parallel tokens based on the selected strategy.
        
        Args:
            input_ids: Current input token IDs
            parallel_tokens: List of (token_id, probability) tuples
            
        Returns:
            List[Tuple[int, float]]: Pruned list of (token_id, probability) tuples
        """
        if self.pruning_strategy == "diversity":
            return self._diversity_optimized_pruning(input_ids, parallel_tokens)
        else:
            return self._coherence_optimized_pruning(input_ids, parallel_tokens) 