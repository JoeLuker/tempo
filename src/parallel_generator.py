import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Set

class ParallelThresholdGenerator:
    """
    Implements the Parallel Threshold Output generation mechanism for Mistral-7B.
    
    This mechanism generates multiple tokens in parallel at each step based on a
    probability threshold, instead of the standard autoregressive approach.
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        threshold: float = 0.1,
        max_length: int = 512,
        device: str = "mps",
        pruner = None
    ):
        """
        Initialize the generator.
        
        Args:
            model: The Mistral-7B model
            tokenizer: HuggingFace tokenizer
            threshold: Probability threshold for token selection
            max_length: Maximum sequence length
            device: Device to use for computation
            pruner: Optional RetroactivePruner instance for coherence-based pruning
        """
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.max_length = max_length
        self.device = device
        self.pruner = pruner
        
    def _get_parallel_tokens(
        self, 
        logits: torch.Tensor, 
        threshold: float
    ) -> Tuple[List[int], List[float]]:
        """
        Get tokens that exceed the probability threshold.
        
        Args:
            logits: Raw logits from the model (batch_size, vocab_size)
            threshold: Probability threshold for selection
            
        Returns:
            tuple: (token_ids, probabilities)
        """
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get tokens above threshold
        tokens_above_threshold = torch.where(probs > threshold)[1]
        selected_probs = probs[0, tokens_above_threshold]
        
        # Sort by probability (highest first)
        sorted_indices = torch.argsort(selected_probs, descending=True)
        
        tokens = tokens_above_threshold[sorted_indices].tolist()
        probabilities = selected_probs[sorted_indices].tolist()
        
        return tokens, probabilities
    
    def _update_attention_mask(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        num_new_tokens: int
    ) -> torch.Tensor:
        """
        Update attention mask to handle parallel tokens.
        
        For tokens generated at the same step, they should have the same position
        in terms of what they can attend to (all previous tokens), and they can
        attend to each other.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Current attention mask
            num_new_tokens: Number of new tokens being added in parallel
            
        Returns:
            torch.Tensor: Updated attention mask
        """
        seq_len = input_ids.size(1)
        current_mask_len = attention_mask.size(1)
        
        # Create new attention mask for the expanded sequence
        new_mask = torch.ones(
            (1, seq_len), 
            dtype=attention_mask.dtype, 
            device=self.device
        )
        
        # Copy existing mask values
        new_mask[0, :current_mask_len] = attention_mask[0]
        
        return new_mask
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 100, 
        threshold: Optional[float] = None,
        return_parallel_sets: bool = False,
        use_pruning: bool = False
    ) -> Dict:
        """
        Generate text using the Parallel Threshold Output mechanism.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            threshold: Override default threshold
            return_parallel_sets: If True, return the sets of parallel tokens
            use_pruning: Whether to use retroactive pruning (if pruner is available)
            
        Returns:
            dict: Results including the generated text
        """
        if threshold is None:
            threshold = self.threshold
            
        # Encode prompt
        input_data = self.tokenizer(prompt, return_tensors="pt")
        input_ids = input_data.input_ids.to(self.device)
        attention_mask = input_data.attention_mask.to(self.device)
        
        # Track sets of parallel tokens for analysis
        parallel_token_sets = []
        pruned_token_sets = []
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Forward pass to get logits
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get last token's logits
                next_token_logits = outputs.logits[:, -1, :]
                
                # Get parallel tokens above threshold
                tokens, probs = self._get_parallel_tokens(next_token_logits, threshold)
                
                # If no tokens above threshold, fall back to standard sampling
                if not tokens:
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    tokens = [next_token.item()]
                    probs = [F.softmax(next_token_logits, dim=-1)[0, next_token].item()]
                
                # Create token set for this generation step
                token_set = [(t, p) for t, p in zip(tokens, probs)]
                parallel_token_sets.append(token_set)
                
                # Apply retroactive pruning if enabled and available
                if use_pruning and self.pruner is not None:
                    pruned_token_set = self.pruner.prune_parallel_tokens(input_ids, token_set)
                    tokens = [t[0] for t in pruned_token_set]
                    probs = [t[1] for t in pruned_token_set]
                    pruned_token_sets.append(pruned_token_set)
                else:
                    pruned_token_sets.append(token_set)
                
                # Update input_ids with parallel tokens
                # For positional encoding, these tokens will share the same position
                # We'll duplicate the last position's embedding for all tokens in the set
                for token in tokens:
                    # Append token to input_ids
                    input_ids = torch.cat([input_ids, torch.tensor([[token]], device=self.device)], dim=1)
                    
                    # Update attention mask for each token added
                    attention_mask = self._update_attention_mask(input_ids, attention_mask, 1)
                
                # Check if any token is EOS
                if self.tokenizer.eos_token_id in tokens:
                    break
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Return results
        results = {
            "generated_text": generated_text,
            "prompt": prompt,
            "threshold": threshold,
            "use_pruning": use_pruning
        }
        
        if return_parallel_sets:
            # Convert tokens to text for easier analysis
            readable_sets = []
            readable_pruned_sets = []
            
            for token_set, pruned_set in zip(parallel_token_sets, pruned_token_sets):
                # Original parallel set
                readable_set = []
                for token, prob in token_set:
                    token_text = self.tokenizer.decode([token])
                    readable_set.append((token_text, prob))
                readable_sets.append(readable_set)
                
                # Pruned set (if different)
                if use_pruning and self.pruner is not None:
                    readable_pruned = []
                    for token, prob in pruned_set:
                        token_text = self.tokenizer.decode([token])
                        readable_pruned.append((token_text, prob))
                    readable_pruned_sets.append(readable_pruned)
                
            results["parallel_sets"] = readable_sets
            if use_pruning and self.pruner is not None:
                results["pruned_sets"] = readable_pruned_sets
            
        return results 