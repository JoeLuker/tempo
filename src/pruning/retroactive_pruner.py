import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

class RetroactivePruner:
    """
    Retroactively prunes previous parallel tokens based on attention patterns from newer tokens.
    
    This pruner looks at how newly generated tokens attend to previous tokens, and prunes out
    previous parallel options that receive insufficient attention.
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        attention_threshold: float = 0.01,
        device: str = "mps",
        debug_mode: bool = False
    ):
        """
        Initialize the retroactive pruner.
        
        Args:
            model: The language model
            tokenizer: HuggingFace tokenizer
            attention_threshold: Threshold for attention-based pruning (0-1)
            device: Device to use for computation
            debug_mode: Enable detailed logging
        """
        self.model = model
        self.tokenizer = tokenizer
        self.attention_threshold = attention_threshold
        self.device = device
        self.debug_mode = debug_mode
        self.token_generator = None
        
        # For logging and debugging
        self.pruning_stats = {
            "total_tokens_considered": 0,
            "tokens_pruned": 0,
            "positions_evaluated": 0,
            "positions_pruned": 0
        }
        
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
    
    def retroactively_prune(
        self, 
        prompt_length: int,
        all_parallel_tokens: Dict[int, List[Tuple[int, float]]]
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Retroactively prune previous parallel sets based on newest token's attention.
        
        Args:
            prompt_length: Length of the original prompt
            all_parallel_tokens: Dictionary mapping positions to lists of (token_id, prob) pairs
            
        Returns:
            Dict[int, List[Tuple[int, float]]]: Updated parallel tokens after pruning
        """
        if not self.token_generator:
            if self.debug_mode:
                print("Warning: Token generator not set, cannot retrieve cached attention")
            return all_parallel_tokens
        
        # Get cached attention from most recent token
        cached_attention, _ = self.token_generator.get_cached_attention()
        if cached_attention is None:
            if self.debug_mode:
                print("Warning: No cached attention available for retroactive pruning")
            return all_parallel_tokens
        
        # Last few layers often contain most relevant attention
        layers_to_use = min(3, len(cached_attention))
        attention_layers = cached_attention[-layers_to_use:]
        
        # Average attention patterns across selected layers
        avg_layer_attention = torch.mean(torch.stack([layer for layer in attention_layers]), dim=0)
        
        # Extract attention for the last token position (the newest token)
        # This shows how the newest token attends to all previous tokens
        try:
            last_token_attn = avg_layer_attention[0, :, -1, :-1]  # [num_heads, seq_len-1]
            
            # Average across attention heads
            avg_attention = last_token_attn.mean(dim=0)  # [seq_len-1]
            
            # Normalized attention (as scores)
            normalized_attn = avg_attention / (torch.sum(avg_attention) + 1e-10)
            
            if self.debug_mode:
                print(f"Retroactive pruning with attention threshold: {self.attention_threshold}")
                print(f"Attention shape: {normalized_attn.shape}")
        except Exception as e:
            if self.debug_mode:
                print(f"Error extracting attention for retroactive pruning: {e}")
            return all_parallel_tokens
        
        # Create a copy of the dictionary to avoid modifying during iteration
        pruned_tokens = {pos: tokens[:] for pos, tokens in all_parallel_tokens.items()}
        
        # For each previous position with parallel tokens
        pruned_positions = 0
        
        for pos in sorted(all_parallel_tokens.keys()):
            if pos == max(all_parallel_tokens.keys()):  # Skip the most recent position
                continue
                
            # Get absolute position in the sequence
            abs_pos = prompt_length + pos
            
            # Get attention score to this position
            if abs_pos < len(normalized_attn):
                attention_score = normalized_attn[abs_pos].item()
                tokens_before = len(pruned_tokens[pos])
                
                # Track stats
                self.pruning_stats["positions_evaluated"] += 1
                self.pruning_stats["total_tokens_considered"] += tokens_before
                
                if self.debug_mode:
                    tokens_text = []
                    for tid, _ in pruned_tokens[pos]:
                        try:
                            tokens_text.append(self.tokenizer.decode([tid]))
                        except:
                            tokens_text.append(f"<ID:{tid}>")
                    
                    print(f"Position {pos} with tokens {tokens_text}")
                    print(f"  Attention score: {attention_score:.4f}, threshold: {self.attention_threshold}")
                
                # If below threshold, skip this position entirely
                if attention_score < self.attention_threshold and len(pruned_tokens[pos]) > 1:
                    # Find the highest probability token to keep
                    best_token = max(pruned_tokens[pos], key=lambda x: x[1])
                    pruned_tokens[pos] = [best_token]  # Keep only the best token
                    
                    tokens_pruned = tokens_before - 1
                    self.pruning_stats["tokens_pruned"] += tokens_pruned
                    self.pruning_stats["positions_pruned"] += 1
                    pruned_positions += 1
                    
                    if self.debug_mode:
                        print(f"  Pruned position {pos}: kept only '{self.tokenizer.decode([best_token[0]])}'")
                        print(f"  Removed {tokens_pruned} alternative tokens")
        
        if self.debug_mode and pruned_positions > 0:
            print(f"Retroactive pruning modified {pruned_positions} positions")
            
        return pruned_tokens
    
    def print_stats(self):
        """Print retroactive pruning statistics."""
        print("\nRetroactive Pruning Stats:")
        print(f"  Positions evaluated: {self.pruning_stats['positions_evaluated']}")
        print(f"  Positions pruned: {self.pruning_stats['positions_pruned']}")
        print(f"  Total tokens considered: {self.pruning_stats['total_tokens_considered']}")
        print(f"  Tokens pruned: {self.pruning_stats['tokens_pruned']}")
        
        if self.pruning_stats['total_tokens_considered'] > 0:
            prune_rate = (self.pruning_stats['tokens_pruned'] / self.pruning_stats['total_tokens_considered']) * 100
            print(f"  Pruning rate: {prune_rate:.1f}%")
        
        print(f"  Current attention threshold: {self.attention_threshold}") 