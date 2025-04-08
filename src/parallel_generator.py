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
    
    def _create_parallel_set_input(
        self,
        base_input_ids: torch.Tensor,
        base_attention_mask: torch.Tensor,
        parallel_tokens: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create input tensors for the next forward pass where all tokens in the
        parallel set are represented with the same positional encoding.
        
        Args:
            base_input_ids: Current input token IDs
            base_attention_mask: Current attention mask
            parallel_tokens: List of token IDs in the parallel set
            
        Returns:
            tuple: (new_input_ids, new_attention_mask)
        """
        batch_size, seq_len = base_input_ids.shape
        num_parallel_tokens = len(parallel_tokens)
        
        # Create a new tensor that will hold all parallel tokens at the next position
        new_seq_len = seq_len + 1  # Only add one position for all parallel tokens
        new_input_ids = torch.zeros((batch_size, new_seq_len), dtype=base_input_ids.dtype, device=self.device)
        
        # Copy existing tokens
        new_input_ids[:, :seq_len] = base_input_ids
        
        # Set the first token of the parallel set at the next position
        # Other tokens in the set are tracked separately but not added to the input sequence
        # This implements Option A: all tokens share the exact same position
        if parallel_tokens:
            new_input_ids[:, seq_len] = parallel_tokens[0]
        
        # Create new attention mask for expanded sequence
        new_attention_mask = torch.ones((batch_size, new_seq_len), dtype=base_attention_mask.dtype, device=self.device)
        new_attention_mask[:, :seq_len] = base_attention_mask
        
        return new_input_ids, new_attention_mask, parallel_tokens
    
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
        
        # For properly tracking token positions
        position_to_tokens = {}  # Maps position -> list of parallel tokens
        actual_tokens = []  # All tokens in sequence order
        
        with torch.no_grad():
            for step in range(max_tokens):
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
                
                # Store parallel tokens at this position
                position = input_ids.size(1)
                position_to_tokens[position] = tokens
                actual_tokens.extend(tokens)
                
                # Create new input representation for next step
                # This properly implements Option A by treating all tokens at the same position
                input_ids, attention_mask, _ = self._create_parallel_set_input(
                    input_ids, attention_mask, tokens
                )
                
                # Check if any token is EOS
                if self.tokenizer.eos_token_id in tokens:
                    break
        
        # Create a raw representation of the full token sequence including all parallel tokens
        # This is for decoding the final output
        full_token_sequence = []
        for i in range(len(input_data.input_ids[0])):  # Add prompt tokens
            full_token_sequence.append(input_ids[0, i].item())
            
        # Add generated tokens
        for pos in sorted(position_to_tokens.keys()):
            full_token_sequence.extend(position_to_tokens[pos])
        
        # Decode the generated text with all parallel tokens
        generated_text = self.tokenizer.decode(full_token_sequence, skip_special_tokens=True)
        
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
                
            # Also add positional information
            position_info = {}
            for pos, tokens in position_to_tokens.items():
                position_info[str(pos)] = [self.tokenizer.decode([t]) for t in tokens]
            results["position_to_tokens"] = position_info
            
        return results 