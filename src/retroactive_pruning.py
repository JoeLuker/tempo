import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
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
        device: str = "mps",
        use_dynamic_threshold: bool = False,
        max_steps: Optional[int] = None,
        bezier_points: Optional[List[float]] = None,
        final_threshold: float = 1.0
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
            use_dynamic_threshold: Whether to use dynamic thresholds that increase over steps
            max_steps: Maximum number of steps (for calculating dynamic threshold)
            bezier_points: Control points for Bezier curve [p1, p2] between 0-1 (default creates exponential curve)
            final_threshold: Final threshold value for dynamic threshold (default 1.0)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.coherence_threshold = coherence_threshold
        self.diversity_clusters = diversity_clusters
        self.pruning_strategy = pruning_strategy
        self.device = device
        self.use_dynamic_threshold = use_dynamic_threshold
        self.max_steps = max_steps
        self.current_step = 0
        self.final_threshold = final_threshold
        
        # Default Bezier control points for exponential-like curve that starts slow and accelerates
        self.bezier_points = bezier_points if bezier_points is not None else [0.2, 0.8]
        
        # For comprehensive dynamic threshold approach
        # Store all original token sets and their coherence scores
        self.all_token_sets = []  # List of token sets [(token_id, prob), ...]
        self.all_token_scores = []  # List of normalized coherence scores for each token in each set
        self.all_input_ids = []  # Input context for each step
    
    def _cubic_bezier(self, t: float, p0: float, p1: float, p2: float, p3: float) -> float:
        """
        Calculate a point on a cubic Bezier curve.
        
        Args:
            t: Parameter between 0 and 1
            p0, p1, p2, p3: Control points
            
        Returns:
            float: Value at point t on the Bezier curve
        """
        return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
    
    def _get_current_threshold(self) -> float:
        """
        Get the current coherence threshold based on step number when using dynamic thresholds.
        Threshold follows a Bezier curve from the initial value to final_threshold at the final step.
        
        Returns:
            float: Current coherence threshold
        """
        if not self.use_dynamic_threshold or self.max_steps is None:
            return self.coherence_threshold
            
        # Calculate progress as a value between 0 and 1
        progress = min(1.0, self.current_step / max(1, self.max_steps - 1))
        
        # Use Bezier curve for smoother threshold progression
        # p0 is initial threshold, p3 is final threshold
        # p1 and p2 are control points from self.bezier_points
        p0 = self.coherence_threshold
        p1 = self.bezier_points[0]  # First control point
        p2 = self.bezier_points[1]  # Second control point
        p3 = self.final_threshold
        
        # Calculate threshold using cubic Bezier curve
        # We're actually using a combination of linear interpolation and Bezier shape
        # This ensures we start exactly at coherence_threshold and end at exactly final_threshold
        bezier_shape = self._cubic_bezier(progress, 0.0, p1, p2, 1.0)
        dynamic_threshold = p0 + (p3 - p0) * bezier_shape
        
        return dynamic_threshold
    
    def reapply_dynamic_threshold_to_all_sets(self) -> List[List[Tuple[int, float]]]:
        """
        Reapply the current dynamic threshold to all previously processed token sets.
        This ensures that as the threshold increases, earlier parallel sets also collapse.
        
        Returns:
            List[List[Tuple[int, float]]]: Updated list of pruned token sets
        """
        if not self.use_dynamic_threshold or not self.all_token_sets:
            return []
            
        current_threshold = self._get_current_threshold()
        updated_pruned_sets = []
        
        for i, (token_set, score_set) in enumerate(zip(self.all_token_sets, self.all_token_scores)):
            # Apply current threshold to this set's scores
            if i == len(self.all_token_sets) - 1 and self.current_step >= self.max_steps:
                # For the last set at the final step, ensure a single token remains only if final_threshold is 1.0
                if self.final_threshold >= 0.999:  # Using 0.999 to account for floating point precision
                    # Force collapse to a single token
                    if token_set:
                        max_score_idx = max(range(len(score_set)), key=lambda j: score_set[j][1])
                        pruned_set = [token_set[max_score_idx]]
                    else:
                        pruned_set = []
                else:
                    # Apply threshold normally without forcing collapse
                    pruned_set = [
                        token_set[j] for j, (_, score) in enumerate(score_set) 
                        if score >= current_threshold
                    ]
                    
                    # If all tokens were pruned, keep the one with highest score
                    if not pruned_set and token_set:
                        max_score_idx = max(range(len(score_set)), key=lambda j: score_set[j][1])
                        pruned_set = [token_set[max_score_idx]]
            else:
                # For other sets, apply the current threshold without forcing single token
                pruned_set = [
                    token_set[j] for j, (_, score) in enumerate(score_set) 
                    if score >= current_threshold
                ]
                
                # If all tokens were pruned, keep the one with highest score
                if not pruned_set and token_set:
                    max_score_idx = max(range(len(score_set)), key=lambda j: score_set[j][1])
                    pruned_set = [token_set[max_score_idx]]
            
            updated_pruned_sets.append(pruned_set)
            
        return updated_pruned_sets
    
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
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        Prune parallel tokens based on attention coherence.
        
        This is the original pruning strategy that prioritizes tokens
        that lead to coherent continuations.
        
        Args:
            input_ids: Current input token IDs
            parallel_tokens: List of (token_id, probability) tuples
            
        Returns:
            Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
                - Pruned list of (token_id, probability) tuples
                - List of (token_id, normalized_score) tuples for reapplying thresholds
        """
        if len(parallel_tokens) <= 1:
            # For single token sets, still compute scores for dynamic threshold
            if len(parallel_tokens) == 1:
                token_id, prob = parallel_tokens[0]
                return parallel_tokens, [(token_id, 1.0)]  # Single token gets max score
            return parallel_tokens, []
            
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
        
        # Save normalized scores for reapplying thresholds later
        token_scores = [(token_id, norm_score) for token_id, _, norm_score in normalized_token_info]
        
        # Get the current coherence threshold (static or dynamic)
        current_threshold = self._get_current_threshold()
        
        # Filter based on normalized coherence score
        pruned_tokens = [
            (token_id, prob)
            for token_id, prob, norm_score in normalized_token_info
            if norm_score >= current_threshold
        ]
        
        # If all tokens were pruned, keep the one with highest coherence
        # If multiple tokens have nearly identical coherence, use probability as tie-breaker
        if not pruned_tokens:
            # First find tokens with highest coherence
            max_coherence = max(normalized_token_info, key=lambda x: x[2])[2]
            top_tokens = [
                (token_id, prob) 
                for token_id, prob, score in normalized_token_info 
                if abs(score - max_coherence) < 1e-5
            ]
            
            if len(top_tokens) == 1:
                pruned_tokens = top_tokens
            else:
                # Multiple tokens with similar coherence - break tie using probability
                best_token = max(top_tokens, key=lambda x: x[1])
                pruned_tokens = [best_token]
            
        return pruned_tokens, token_scores
    
    def prune_parallel_tokens(
        self, 
        input_ids: torch.Tensor, 
        parallel_tokens: List[Tuple[int, float]]
    ) -> Tuple[List[Tuple[int, float]], List[List[Tuple[int, float]]]]:
        """
        Prune parallel tokens based on the selected strategy.
        
        Args:
            input_ids: Current input token IDs
            parallel_tokens: List of (token_id, probability) tuples
            
        Returns:
            Tuple[List[Tuple[int, float]], List[List[Tuple[int, float]]]]:
                - Pruned list of (token_id, probability) tuples for current step
                - List of pruned token sets for all steps (comprehensive dynamic threshold)
        """
        # If using dynamic threshold, we need special handling
        if self.use_dynamic_threshold:
            # Store the input context for this step
            self.all_input_ids.append(input_ids.clone())
            
            # For coherence pruning with scoring
            pruned_tokens, token_scores = self._coherence_optimized_pruning(input_ids, parallel_tokens)
            
            # Store the original token set and normalized scores
            self.all_token_sets.append(parallel_tokens.copy())
            self.all_token_scores.append(token_scores)
            
            # Increment step counter after processing
            self.current_step += 1
            
            # Reapply the threshold to all previous sets with updated threshold
            all_pruned_sets = self.reapply_dynamic_threshold_to_all_sets()
            
            return pruned_tokens, all_pruned_sets
        elif self.pruning_strategy == "diversity":
            pruned_tokens = self._diversity_optimized_pruning(input_ids, parallel_tokens)
            return pruned_tokens, [pruned_tokens]
        else:
            pruned_tokens, _ = self._coherence_optimized_pruning(input_ids, parallel_tokens)
            return pruned_tokens, [pruned_tokens] 