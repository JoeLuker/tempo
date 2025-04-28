import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import traceback

from .dynamic_threshold import DynamicThresholdManager


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
        debug_mode: bool = False,
        dynamic_threshold_manager: Optional[DynamicThresholdManager] = None,
        use_relative_attention: bool = True,
        relative_threshold: float = 0.5,
        use_multi_scale_attention: bool = True,
        num_layers_to_use: Optional[int] = None,
        use_sigmoid_threshold: bool = True,
        sigmoid_steepness: float = 10.0,
        complete_pruning_mode: str = "keep_token",
    ):
        """
        Initialize the retroactive pruner.

        Args:
            model: The language model
            tokenizer: HuggingFace tokenizer
            attention_threshold: Base threshold for attention-based pruning (0-1)
            device: Device to use for computation
            debug_mode: Enable detailed logging
            dynamic_threshold_manager: Optional DynamicThresholdManager for dynamic thresholding
            use_relative_attention: Whether to use relative attention thresholds
            relative_threshold: Threshold for relative attention-based pruning (0-1)
            use_multi_scale_attention: Whether to use multi-scale attention integration
            num_layers_to_use: Number of last layers to use (None means use all layers)
            use_sigmoid_threshold: Whether to use sigmoid-based decision boundary
            sigmoid_steepness: Controls how sharp the sigmoid transition is
            complete_pruning_mode: How to handle pruned positions. Options:
                "keep_token" - Keep the best token at the position (default)
                "keep_unattended" - Keep the best token but mark it as unattended
                "remove_position" - Remove the position entirely from generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.base_attention_threshold = attention_threshold
        self.attention_threshold = attention_threshold
        self.device = device
        self.debug_mode = debug_mode
        self.token_generator = None
        self.use_relative_attention = use_relative_attention
        self.relative_threshold = relative_threshold
        self.use_multi_scale_attention = use_multi_scale_attention
        self.num_layers_to_use = num_layers_to_use
        self.use_sigmoid_threshold = use_sigmoid_threshold
        self.sigmoid_steepness = sigmoid_steepness
        self.complete_pruning_mode = complete_pruning_mode

        # Initialize DynamicThresholdManager if not provided
        if dynamic_threshold_manager is None:
            self.dynamic_threshold_manager = DynamicThresholdManager(
                base_threshold=attention_threshold,
            )
        else:
            self.dynamic_threshold_manager = dynamic_threshold_manager

        # For logging and debugging
        self.pruning_stats = {
            "total_tokens_considered": 0,
            "tokens_pruned": 0,
            "positions_evaluated": 0,
            "positions_pruned": 0,
            "positions_removed": 0,
            "positions_unattended": 0,
        }

        if self.debug_mode:
            print(
                f"RetroactivePruner initialized with threshold={attention_threshold}, relative_threshold={relative_threshold}, "
                f"use_relative_attention={use_relative_attention}, use_multi_scale_attention={use_multi_scale_attention}, "
                f"num_layers_to_use={num_layers_to_use}, use_sigmoid_threshold={use_sigmoid_threshold}, "
                f"sigmoid_steepness={sigmoid_steepness}, complete_pruning_mode={complete_pruning_mode}, "
                f"debug_mode={debug_mode}"
            )

    def set_token_generator(self, token_generator):
        """
        Set the token generator instance to enable access to cached attention.

        Args:
            token_generator: TokenGenerator instance
        """
        self.token_generator = token_generator
        if self.debug_mode:
            print(f"Token generator set: {token_generator is not None}")

    def set_debug_mode(self, enabled=True):
        """Enable or disable debug mode."""
        self.debug_mode = enabled
        print(f"RetroactivePruner debug mode set to: {enabled}")

    def log_debug(self, message):
        """Log a debug message if debug_mode is enabled."""
        if self.debug_mode:
            print(message)

    def get_pruning_threshold(self, step: Optional[int] = None) -> float:
        """
        Get the current pruning threshold based on generation progress.
        This method now delegates to DynamicThresholdManager for all threshold calculations.

        Args:
            step: Current generation step (optional)

        Returns:
            float: Current pruning threshold
        """
        return self.dynamic_threshold_manager.get_current_threshold(step)

    def update_step(self, step: int):
        """
        Update threshold based on current generation step.
        This method now delegates to DynamicThresholdManager for all threshold calculations.

        Args:
            step: Current generation step (source of truth)
        """
        self.attention_threshold = self.get_pruning_threshold(step)

        if self.debug_mode:
            print(
                f"Updated retroactive pruner threshold to {self.attention_threshold:.4f} at step {step}"
            )

    def apply_sigmoid_threshold(self, attention_score: float, threshold: float) -> bool:
        """
        Apply sigmoid-based decision boundary to attention score.

        Args:
            attention_score: The attention score to evaluate
            threshold: The threshold value where sigmoid equals 0.5

        Returns:
            bool: True if the token should be kept, False if it should be pruned
        """
        sigmoid_value = 1.0 / (
            1.0 + math.exp(-self.sigmoid_steepness * (attention_score - threshold))
        )

        if self.debug_mode:
            print(
                f"  Sigmoid value: {sigmoid_value:.4f} (steepness={self.sigmoid_steepness}, midpoint={threshold:.4f})"
            )

        return sigmoid_value > 0.5  # Token survives if above 0.5

    def apply_vectorized_sigmoid_threshold(
        self, attention_scores: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        """
        Apply sigmoid-based decision boundary to attention scores in a vectorized way.

        Args:
            attention_scores: Tensor of attention scores to evaluate
            threshold: The threshold value where sigmoid equals 0.5

        Returns:
            torch.Tensor: Boolean tensor where True means the token should be kept
        """
        # Calculate sigmoid values for all positions at once
        sigmoid_values = 1.0 / (
            1.0 + torch.exp(-self.sigmoid_steepness * (attention_scores - threshold))
        )

        if self.debug_mode:
            print(
                f"  Vectorized sigmoid applied with steepness={self.sigmoid_steepness}, midpoint={threshold:.4f}"
            )
            print(
                f"  Min sigmoid value: {sigmoid_values.min().item():.4f}, Max: {sigmoid_values.max().item():.4f}"
            )

        # Return boolean tensor: True for positions to keep
        return sigmoid_values > 0.5  # Tokens survive if above 0.5

    def retroactively_prune(
        self,
        prompt_length: int,
        all_parallel_tokens: Dict[int, List[Tuple[int, float]]],
        step: Optional[int] = None,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Retroactively prune previous parallel sets based on newest token's attention.

        Args:
            prompt_length: Length of the original prompt
            all_parallel_tokens: Dictionary mapping positions to lists of (token_id, prob) pairs
            step: Current generation step (source of truth)

        Returns:
            Dict[int, List[Tuple[int, float]]]: Updated parallel tokens after pruning
        """
        if self.debug_mode:
            print(f"\nRetroactive pruning at step {step}")
            print(f"Number of parallel positions: {len(all_parallel_tokens)}")
            print(f"Token generator available: {self.token_generator is not None}")
            print(f"Pruning mode: {self.complete_pruning_mode}")

        if not self.token_generator:
            if self.debug_mode:
                print(
                    "Warning: Token generator not set, cannot retrieve cached attention"
                )
            return all_parallel_tokens

        # Get cached attention from most recent token
        cached_attention, seq_len = self.token_generator.get_cached_attention()
        if cached_attention is None:
            if self.debug_mode:
                print("Warning: No cached attention available for retroactive pruning")
            return all_parallel_tokens

        if self.debug_mode:
            print(f"Retrieved cached attention with {len(cached_attention)} layers")
            print(f"First layer attention shape: {cached_attention[0].shape}")
            print(f"Sequence length reported by token generator: {seq_len}")
            print(f"Current step: {step}")
            print(f"Attention threshold: {self.attention_threshold}")
            
            # More detailed attention shape analysis
            attn_shape = cached_attention[0].shape
            print(f"\nDetailed attention analysis:")
            print(f"- Batch size: {attn_shape[0]}")
            print(f"- Num heads: {attn_shape[1]}")
            print(f"- Sequence length (rows): {attn_shape[2]}")
            print(f"- Attended length (cols): {attn_shape[3]}")
            
            # List all positions to be evaluated
            print(f"\nParallel positions to evaluate: {sorted(all_parallel_tokens.keys())}")
            print(f"Prompt length: {prompt_length}")
            highest_position = prompt_length + max(all_parallel_tokens.keys(), default=0)
            print(f"Highest absolute position: {highest_position}")
            print(f"Attention window available: {attn_shape[3]}")
            
            if highest_position >= attn_shape[3]:
                print(f"WARNING: Some positions ({highest_position-attn_shape[3]+1}) are beyond attention window!")
                print(f"Positions beyond window: {[pos for pos in all_parallel_tokens.keys() if prompt_length+pos >= attn_shape[3]]}")
            else:
                print(f"All positions within attention window!")

        # Check if we have a mismatch between attention dimensions and positions
        attention_window_size = cached_attention[0].shape[3]
        positions_out_of_bounds = False
        
        # Calculate the highest position we need to access
        highest_abs_position = prompt_length + max(all_parallel_tokens.keys(), default=0)
        if highest_abs_position >= attention_window_size:
            positions_out_of_bounds = True
            if self.debug_mode:
                print(f"CRITICAL: Attention window size mismatch. Have {attention_window_size} positions, "
                      f"need {highest_abs_position+1}. Using fallback pruning.")
                print("This likely means the token_generator used by RetroactivePruner is not "
                      "the same instance that's generating tokens in the main loop.")

        # Multi-scale attention integration
        if self.use_multi_scale_attention:
            # Determine number of layers to use
            total_layers = len(cached_attention)
            if self.num_layers_to_use is None:
                # Use ALL layers by default
                layers_to_use = total_layers
                attention_layers = cached_attention
            else:
                # Use the specified number of last layers
                layers_to_use = min(self.num_layers_to_use, total_layers)
                attention_layers = cached_attention[-layers_to_use:]

            # Create weights that linearly increase from 0.2 to 1.0
            # This gives some weight to early layers but emphasizes later layers
            layer_weights = torch.linspace(0.2, 1.0, layers_to_use, device=self.device)

            # Apply weighted average - ensure we use batch index 0 for consistent attention processing
            batch_idx = 0  # Always use the first batch (main sequence) for attention analysis
            weighted_layers = [
                layer[batch_idx] * weight for layer, weight in zip(attention_layers, layer_weights)
            ]
            avg_layer_attention = (
                torch.sum(torch.stack(weighted_layers), dim=0) / layer_weights.sum()
            )

            if self.debug_mode:
                if self.num_layers_to_use is None:
                    print(
                        f"Using multi-scale attention integration with ALL {layers_to_use} layers"
                    )
                else:
                    print(
                        f"Using multi-scale attention integration with last {layers_to_use} layers"
                    )
                print(
                    f"Layer weights range: {layer_weights[0].item():.2f} to {layer_weights[-1].item():.2f}"
                )
                print(f"Average layer attention shape: {avg_layer_attention.shape}")
        else:
            # Original approach - last few layers only with simple averaging
            layers_to_use = min(3, len(cached_attention))
            batch_idx = 0  # Always use the first batch for attention
            attention_layers = [layer[batch_idx] for layer in cached_attention[-layers_to_use:]]

            # Average attention patterns across selected layers
            avg_layer_attention = torch.mean(
                torch.stack([layer for layer in attention_layers]), dim=0
            )

            if self.debug_mode:
                print(f"Using simple attention averaging with {layers_to_use} layers")
                print(f"Average layer attention shape: {avg_layer_attention.shape}")

        # Extract attention for the last token position (the newest token)
        # This shows how the newest token attends to all previous tokens
        try:
            # Get the shape of the attention tensor to understand what we're working with
            attn_shape = avg_layer_attention.shape  # [num_heads, seq_len, seq_len]
            if self.debug_mode:
                print(f"Averaged attention tensor shape: {attn_shape}")
                print(f"Total sequence length in attention: {attn_shape[-1]}")
                
            # Extract attention from the newest token (last row) to all previous tokens (all columns except the last)
            last_token_attn = avg_layer_attention[:, -1, :-1].clone()  # [num_heads, seq_len-1]
            
            if self.debug_mode:
                print(f"Last token attention shape: {last_token_attn.shape}")
                print(f"This includes attention to ALL previous {last_token_attn.shape[-1]} tokens")

            # Average across attention heads
            avg_attention = last_token_attn.mean(dim=0)  # [seq_len-1]
            
            if self.debug_mode:
                print("\nRaw Attention Analysis:")
                print(f"Raw attention shape: {avg_attention.shape}")
                print(f"Raw max attention: {avg_attention.max().item():.4f}")
                print(f"Raw min attention: {avg_attention.min().item():.4f}")
                print(f"Raw mean attention: {avg_attention.mean().item():.4f}")

            # Calculate expected attention pattern (exponential decay)
            seq_len = avg_attention.shape[0]
            positions = torch.arange(seq_len, device=avg_attention.device)
            # Expected attention decays exponentially with distance
            expected_attention = torch.exp(-positions / (seq_len / 2))
            expected_attention = expected_attention / expected_attention.sum()

            if self.debug_mode:
                print("\nExpected Attention Pattern:")
                print(f"Expected attention shape: {expected_attention.shape}")
                print(f"Expected max attention: {expected_attention.max().item():.4f}")
                print(f"Expected min attention: {expected_attention.min().item():.4f}")
                print(
                    f"Expected mean attention: {expected_attention.mean().item():.4f}"
                )
                print(f"Expected attention decay rate: {1/(seq_len/2):.4f}")

            # Calculate relative attention scores
            # This shows if a position gets more or less attention than expected
            relative_attention = avg_attention / (expected_attention + 1e-10)

            if self.debug_mode:
                print("\nRelative Attention Analysis:")
                print(f"Relative attention shape: {relative_attention.shape}")
                print(f"Relative max attention: {relative_attention.max().item():.4f}")
                print(f"Relative min attention: {relative_attention.min().item():.4f}")
                print(
                    f"Relative mean attention: {relative_attention.mean().item():.4f}"
                )

            # Scale relative attention to match the original attention range
            # First normalize to 0-1 range
            normalized_attn = relative_attention / (relative_attention.max() + 1e-10)
            # Then scale by the original attention range
            attention_range = avg_attention.max() - avg_attention.min()
            normalized_attn = normalized_attn * attention_range + avg_attention.min()

            if self.debug_mode:
                print("\nFinal Normalized Attention:")
                print(f"Final attention shape: {normalized_attn.shape}")
                print(f"Final max attention: {normalized_attn.max().item():.4f}")
                print(f"Final min attention: {normalized_attn.min().item():.4f}")
                print(f"Final mean attention: {normalized_attn.mean().item():.4f}")
                print(f"Current threshold: {self.attention_threshold}")

                # Add check for None before accessing max_steps
                if self.dynamic_threshold_manager is not None:
                    print(
                        f"Max steps: {self.dynamic_threshold_manager.max_steps}, Current step: {step or 0}"
                    )

            # Process each position with parallel tokens
            pruned_positions = {}  # Store pruned positions
            
            # Check if we're beyond the attention window
            if positions_out_of_bounds:
                # Use a fallback strategy: keep tokens with probabilities above a threshold
                fallback_thresh = 0.1  # Keep tokens with probability above 10%
                if self.debug_mode:
                    print(f"Using fallback pruning with probability threshold: {fallback_thresh}")
                
                for pos, token_list in all_parallel_tokens.items():
                    # Keep tokens with probability above threshold
                    kept_tokens = [(tid, prob) for tid, prob in token_list if prob >= fallback_thresh]
                    
                    # Ensure we keep at least one token
                    if not kept_tokens and token_list:
                        # Keep the highest probability token
                        sorted_tokens = sorted(token_list, key=lambda x: x[1], reverse=True)
                        kept_tokens = [sorted_tokens[0]]
                    
                    pruned_positions[pos] = kept_tokens
                
                # Update pruning stats
                total_tokens = sum(len(tokens) for tokens in all_parallel_tokens.values())
                kept_tokens = sum(len(tokens) for tokens in pruned_positions.values())
                
                self.pruning_stats["total_tokens_considered"] += total_tokens
                self.pruning_stats["tokens_pruned"] += (total_tokens - kept_tokens)
                self.pruning_stats["positions_evaluated"] += len(all_parallel_tokens)
                
                # Log what happened
                if self.debug_mode:
                    print(f"Fallback pruning kept {kept_tokens}/{total_tokens} tokens")
                    
                return pruned_positions
            
            # Normal processing when attention matrix matches required positions
            for pos, token_list in all_parallel_tokens.items():
                # Skip the current step itself - retroactive pruning should only apply to previous steps
                if step is not None and pos == step:
                    # Keep all tokens for the current step - it's being processed right now
                    if self.debug_mode:
                        print(f"Skipping position {pos} as it is the current step being generated")
                    pruned_positions[pos] = token_list
                    self.pruning_stats["positions_evaluated"] += 1
                    continue
                
                # Calculate the absolute position (including prompt)
                abs_pos = prompt_length + pos
                
                # Skip if the position is beyond our attention window
                if abs_pos >= attention_window_size:
                    if self.debug_mode:
                        print(f"Position {pos} (abs: {abs_pos}) is beyond attention window, using fallback")
                    kept_tokens = [(tid, prob) for tid, prob in token_list if prob >= 0.1]
                    if not kept_tokens and token_list:
                        kept_tokens = [max(token_list, key=lambda x: x[1])]
                    pruned_positions[pos] = kept_tokens
                    self.pruning_stats["positions_evaluated"] += 1
                    continue
                
                # Extract attention score
                attention_score = normalized_attn[abs_pos].item()

                # Filter tokens based on pruning mode
                if self.complete_pruning_mode == "keep_token":
                    # Keep token with highest probability (default)
                    if attention_score >= self.attention_threshold:
                        pruned_positions[pos] = token_list
                        if self.debug_mode:
                            print(
                                f"Position {pos} (abs: {abs_pos}) kept {len(token_list)} tokens - attention: {attention_score:.4f}"
                            )
                    else:
                        # Only keep highest probability token
                        pruned_positions[pos] = [max(token_list, key=lambda x: x[1])]
                        if self.debug_mode:
                            print(
                                f"Position {pos} (abs: {abs_pos}) pruned to 1 token - attention: {attention_score:.4f}"
                            )
                        self.pruning_stats["positions_pruned"] += 1

                elif self.complete_pruning_mode == "keep_unattended":
                    # Keep best token with a flag for unattended positions
                    # The flag can be used later for special visualization
                    if attention_score >= self.attention_threshold:
                        pruned_positions[pos] = token_list
                        if self.debug_mode:
                            print(
                                f"Position {pos} (abs: {abs_pos}) kept {len(token_list)} tokens - attention: {attention_score:.4f}"
                            )
                    else:
                        # Mark as unattended position
                        best_token = max(token_list, key=lambda x: x[1])
                        pruned_positions[pos] = [best_token]
                        if self.debug_mode:
                            print(
                                f"Position {pos} (abs: {abs_pos}) marked as unattended - attention: {attention_score:.4f}"
                            )
                        self.pruning_stats["positions_unattended"] += 1

                elif self.complete_pruning_mode == "remove_position":
                    # Completely remove unattended positions (more aggressive)
                    if attention_score >= self.attention_threshold:
                        pruned_positions[pos] = token_list
                        if self.debug_mode:
                            print(
                                f"Position {pos} (abs: {abs_pos}) kept {len(token_list)} tokens - attention: {attention_score:.4f}"
                            )
                    else:
                        # Don't include this position at all
                        if self.debug_mode:
                            print(
                                f"Position {pos} (abs: {abs_pos}) completely removed - attention: {attention_score:.4f}"
                            )
                        self.pruning_stats["positions_removed"] += 1

                else:
                    # Unknown mode, keep all tokens
                    pruned_positions[pos] = token_list
                    if self.debug_mode:
                        print(
                            f"Unknown pruning mode '{self.complete_pruning_mode}', keeping all tokens"
                        )

                # Update statistics
                before_tokens = len(token_list)
                after_tokens = len(pruned_positions.get(pos, []))
                self.pruning_stats["total_tokens_considered"] += before_tokens
                self.pruning_stats["tokens_pruned"] += (before_tokens - after_tokens)
                self.pruning_stats["positions_evaluated"] += 1

            return pruned_positions

        except Exception as e:
            if self.debug_mode:
                print(f"Error in retroactive pruning: {e}")
                import traceback
                traceback.print_exc()

            # Fall back to no pruning on error
            return all_parallel_tokens

    def print_stats(self):
        """Print retroactive pruning statistics."""
        print("\nRetroactive Pruning Stats:")
        print(f"  Positions evaluated: {self.pruning_stats['positions_evaluated']}")
        print(f"  Positions pruned: {self.pruning_stats['positions_pruned']}")
        print(
            f"  Positions marked as unattended: {self.pruning_stats['positions_unattended']}"
        )
        print(
            f"  Positions completely removed: {self.pruning_stats['positions_removed']}"
        )
        print(
            f"  Total tokens considered: {self.pruning_stats['total_tokens_considered']}"
        )
        print(f"  Tokens pruned: {self.pruning_stats['tokens_pruned']}")

        if self.pruning_stats["total_tokens_considered"] > 0:
            prune_rate = (
                self.pruning_stats["tokens_pruned"]
                / self.pruning_stats["total_tokens_considered"]
            ) * 100
            print(f"  Pruning rate: {prune_rate:.1f}%")

        print(f"  Current attention threshold: {self.attention_threshold}")
        print(f"  Pruning mode: {self.complete_pruning_mode}")
        if self.use_relative_attention:
            print(
                f"  Using relative attention with threshold: {self.relative_threshold}"
            )
        if self.use_multi_scale_attention:
            print(f"  Using multi-scale attention integration")
        if self.use_sigmoid_threshold:
            print(
                f"  Using sigmoid-based decision boundary (steepness={self.sigmoid_steepness})"
            )

    def extract_raw_attention(self, last_token_attn, pos, normalized_attn):
        """
        Extract raw attention score for a position.
        For positions beyond the attention window, use the last available attention.
        
        Args:
            last_token_attn: The last token's attention weights
            pos: The absolute position to get attention for
            normalized_attn: The normalized attention array
            
        Returns:
            The raw attention score for this position
        """
        # Get absolute position (adjust for attention array indexing)
        abs_pos = pos
        
        # Check if position is beyond the attention window
        if abs_pos >= len(normalized_attn):
            # Use the last available attention score
            if len(normalized_attn) > 0:
                raw_score = normalized_attn[-1].item()
                self.log_debug(f"Position {pos} beyond attention window, using last available attention: {raw_score:.4f}")
                return raw_score
            else:
                # If no attention scores available, use default value
                self.log_debug(f"No attention scores available for position {pos}, using default 0.5")
                return 0.5
        
        # Position is within attention window, get exact score
        raw_score = normalized_attn[abs_pos].item()
        return raw_score

    def extract_vectorized_attention(self, normalized_attn, position_list):
        """
        Vectorized version of attention extraction for multiple positions.
        
        Args:
            normalized_attn: The normalized attention tensor 
            position_list: List of positions to extract attention for
            
        Returns:
            Dictionary mapping positions to their attention scores
        """
        position_scores = {}
        attn_len = len(normalized_attn)
        
        for pos in position_list:
            # Check if position is beyond the attention window
            if pos >= attn_len:
                # Use the last available attention score
                if attn_len > 0:
                    position_scores[pos] = normalized_attn[-1].item()
                    self.log_debug(f"Position {pos} beyond attention window (len={attn_len}), using last score: {position_scores[pos]:.4f}")
                else:
                    # If no attention scores available, use default value
                    position_scores[pos] = 0.5
                    self.log_debug(f"No attention scores available for position {pos}, using default 0.5")
            else:
                # Position is within attention window, get exact score
                position_scores[pos] = normalized_attn[pos].item()
                
        return position_scores
