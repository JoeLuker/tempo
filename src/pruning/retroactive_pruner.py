import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional

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
        use_lci_dynamic_threshold: bool = True,
        use_sigmoid_threshold: bool = True,
        sigmoid_steepness: float = 10.0,
    ):
        """
        Initialize the retroactive pruner.

        Args:
            model: The language model
            tokenizer: HuggingFace tokenizer
            attention_threshold: Threshold for attention-based pruning (0-1)
            device: Device to use for computation
            debug_mode: Enable detailed logging
            dynamic_threshold_manager: Optional DynamicThresholdManager for dynamic thresholding
            use_relative_attention: Whether to use relative attention thresholds
            relative_threshold: Threshold for relative attention-based pruning (0-1)
            use_multi_scale_attention: Whether to use multi-scale attention integration
            num_layers_to_use: Number of last layers to use (None means use all layers)
            use_lci_dynamic_threshold: Whether to use LCI-based dynamic thresholding (vs. classic approach)
            use_sigmoid_threshold: Whether to use sigmoid-based decision boundary
            sigmoid_steepness: Controls how sharp the sigmoid transition is
        """
        self.model = model
        self.tokenizer = tokenizer
        self.base_attention_threshold = attention_threshold
        self.attention_threshold = attention_threshold
        self.device = device
        self.debug_mode = debug_mode
        self.token_generator = None
        self.dynamic_threshold_manager = dynamic_threshold_manager
        self.current_step = 0
        self.use_relative_attention = use_relative_attention
        self.relative_threshold = relative_threshold
        self.use_multi_scale_attention = use_multi_scale_attention
        self.num_layers_to_use = num_layers_to_use
        self.use_lci_dynamic_threshold = use_lci_dynamic_threshold
        self.use_sigmoid_threshold = use_sigmoid_threshold
        self.sigmoid_steepness = sigmoid_steepness

        # For logging and debugging
        self.pruning_stats = {
            "total_tokens_considered": 0,
            "tokens_pruned": 0,
            "positions_evaluated": 0,
            "positions_pruned": 0,
        }
        
        if self.debug_mode:
            print(f"RetroactivePruner initialized with threshold={attention_threshold}, relative_threshold={relative_threshold}, "
                  f"use_relative_attention={use_relative_attention}, use_multi_scale_attention={use_multi_scale_attention}, "
                  f"num_layers_to_use={num_layers_to_use}, use_lci_dynamic_threshold={use_lci_dynamic_threshold}, "
                  f"use_sigmoid_threshold={use_sigmoid_threshold}, sigmoid_steepness={sigmoid_steepness}, debug_mode={debug_mode}")

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

    def get_pruning_threshold(self, step: int, max_steps: int) -> float:
        """
        Calculate a dynamic pruning threshold based on LCI balance.
        
        LCI represents the Losslessness-Compression-Invariance tradeoff.
        This function dynamically balances these factors based on generation progress.
        
        Args:
            step: Current generation step
            max_steps: Maximum number of generation steps
            
        Returns:
            float: Dynamic pruning threshold
        """
        # Calculate where we are in generation process
        progress = step / max_steps if max_steps > 0 else 0.0
        
        # Dynamic weighting of L, C, and I based on progress
        L_weight = max(0.0, 1.0 - progress)  # Decreases over time (Losslessness)
        I_weight = min(1.0, progress * 2)    # Increases over time (Invariance)
        C_weight = min(0.5, progress)        # Slowly increases (Compression)
        
        # Threshold is inverse to L_weight and proportional to I_weight
        base = 0.01  # Minimum threshold
        dynamic_component = (I_weight * 0.1) - (L_weight * 0.05) + (C_weight * 0.05)
        
        if self.debug_mode:
            print(f"LCI weights at step {step}/{max_steps} (progress {progress:.2f}):")
            print(f"  L_weight: {L_weight:.2f}, C_weight: {C_weight:.2f}, I_weight: {I_weight:.2f}")
            print(f"  Base: {base}, Dynamic component: {dynamic_component:.4f}")
            
        return base + dynamic_component

    def update_step(self, step: int):
        """Update the current step and threshold if using dynamic thresholding."""
        self.current_step = step
        if self.dynamic_threshold_manager is not None:
            max_steps = self.dynamic_threshold_manager.max_steps
            
            if self.use_lci_dynamic_threshold:
                # Use the LCI-based dynamic threshold calculation
                self.attention_threshold = self.get_pruning_threshold(step, max_steps)
                
                if self.debug_mode:
                    print(f"Updated retroactive pruner threshold to {self.attention_threshold:.4f} using LCI model at step {step}")
            else:
                # Legacy approach - Bezier or ReLU based threshold
                progress = min(1.0, step / max_steps)
                
                # Scale threshold differently depending on whether we're using ReLU or Bezier
                if hasattr(self.dynamic_threshold_manager, 'use_relu') and self.dynamic_threshold_manager.use_relu:
                    # For ReLU, use the activation point to determine when to start increasing
                    relu_activation = self.dynamic_threshold_manager.relu_activation_point
                    if progress < relu_activation:
                        # Before activation point - use minimum threshold
                        self.attention_threshold = 0.001
                    else:
                        # After activation point - linear increase
                        relu_progress = (progress - relu_activation) / (1.0 - relu_activation) if relu_activation < 1.0 else 0.0
                        final_threshold = self.dynamic_threshold_manager.final_threshold
                        min_threshold = 0.001
                        self.attention_threshold = min_threshold + (relu_progress * (final_threshold - min_threshold))
                else:
                    # Original Bezier-based scaling
                    final_threshold = self.dynamic_threshold_manager.final_threshold
                    min_threshold = 0.001
                    self.attention_threshold = min_threshold + (progress * (final_threshold - min_threshold))
                
                if self.debug_mode:
                    print(f"Updated retroactive pruner threshold to {self.attention_threshold:.4f} using classic model at step {step}")

    def apply_sigmoid_threshold(self, attention_score: float, threshold: float) -> bool:
        """
        Apply sigmoid-based decision boundary to attention score.
        
        Args:
            attention_score: The attention score to evaluate
            threshold: The threshold value where sigmoid equals 0.5
            
        Returns:
            bool: True if the token should be kept, False if it should be pruned
        """
        sigmoid_value = 1.0 / (1.0 + math.exp(-self.sigmoid_steepness * (attention_score - threshold)))
        
        if self.debug_mode:
            print(f"  Sigmoid value: {sigmoid_value:.4f} (steepness={self.sigmoid_steepness}, midpoint={threshold:.4f})")
            
        return sigmoid_value > 0.5  # Token survives if above 0.5

    def apply_vectorized_sigmoid_threshold(self, attention_scores: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Apply sigmoid-based decision boundary to attention scores in a vectorized way.
        
        Args:
            attention_scores: Tensor of attention scores to evaluate
            threshold: The threshold value where sigmoid equals 0.5
            
        Returns:
            torch.Tensor: Boolean tensor where True means the token should be kept
        """
        # Calculate sigmoid values for all positions at once
        sigmoid_values = 1.0 / (1.0 + torch.exp(-self.sigmoid_steepness * (attention_scores - threshold)))
        
        if self.debug_mode:
            print(f"  Vectorized sigmoid applied with steepness={self.sigmoid_steepness}, midpoint={threshold:.4f}")
            print(f"  Min sigmoid value: {sigmoid_values.min().item():.4f}, Max: {sigmoid_values.max().item():.4f}")
            
        # Return boolean tensor: True for positions to keep
        return sigmoid_values > 0.5  # Tokens survive if above 0.5

    def retroactively_prune(
        self,
        prompt_length: int,
        all_parallel_tokens: Dict[int, List[Tuple[int, float]]],
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Retroactively prune previous parallel sets based on newest token's attention.

        Args:
            prompt_length: Length of the original prompt
            all_parallel_tokens: Dictionary mapping positions to lists of (token_id, prob) pairs

        Returns:
            Dict[int, List[Tuple[int, float]]]: Updated parallel tokens after pruning
        """
        if self.debug_mode:
            print(f"\nRetroactive pruning at step {self.current_step}")
            print(f"Number of parallel positions: {len(all_parallel_tokens)}")
            print(f"Token generator available: {self.token_generator is not None}")

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

        if self.debug_mode:
            print(f"Retrieved cached attention with {len(cached_attention)} layers")
            print(f"Attention threshold: {self.attention_threshold}")

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
            
            # Apply weighted average
            weighted_layers = [layer * weight for layer, weight in zip(attention_layers, layer_weights)]
            avg_layer_attention = torch.sum(
                torch.stack(weighted_layers), dim=0
            ) / layer_weights.sum()
            
            if self.debug_mode:
                if self.num_layers_to_use is None:
                    print(f"Using multi-scale attention integration with ALL {layers_to_use} layers")
                else:
                    print(f"Using multi-scale attention integration with last {layers_to_use} layers")
                print(f"Layer weights range: {layer_weights[0].item():.2f} to {layer_weights[-1].item():.2f}")
        else:
            # Original approach - last few layers only with simple averaging
            layers_to_use = min(3, len(cached_attention))
            attention_layers = cached_attention[-layers_to_use:]
            
            # Average attention patterns across selected layers
            avg_layer_attention = torch.mean(
                torch.stack([layer for layer in attention_layers]), dim=0
            )
            
            if self.debug_mode:
                print(f"Using simple attention averaging with {layers_to_use} layers")

        # Extract attention for the last token position (the newest token)
        # This shows how the newest token attends to all previous tokens
        try:
            last_token_attn = avg_layer_attention[
                0, :, -1, :-1
            ]  # [num_heads, seq_len-1]

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
                print(f"Expected mean attention: {expected_attention.mean().item():.4f}")
                print(f"Expected attention decay rate: {1/(seq_len/2):.4f}")

            # Calculate relative attention scores
            # This shows if a position gets more or less attention than expected
            relative_attention = avg_attention / (expected_attention + 1e-10)

            if self.debug_mode:
                print("\nRelative Attention Analysis:")
                print(f"Relative attention shape: {relative_attention.shape}")
                print(f"Relative max attention: {relative_attention.max().item():.4f}")
                print(f"Relative min attention: {relative_attention.min().item():.4f}")
                print(f"Relative mean attention: {relative_attention.mean().item():.4f}")
            
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
                    print(f"Threshold progression: {self.current_step}/{self.dynamic_threshold_manager.max_steps} steps")
                else:
                    print("Dynamic threshold manager not available")
        except Exception as e:
            if self.debug_mode:
                print(f"Error extracting attention for retroactive pruning: {e}")
            return all_parallel_tokens

        # Create a copy of the dictionary to avoid modifying during iteration
        pruned_tokens = {pos: tokens[:] for pos, tokens in all_parallel_tokens.items()}

        # For each previous position with parallel tokens
        pruned_positions = 0
        total_positions = 0

        # Create a list of positions and corresponding attention score indices
        positions = sorted(all_parallel_tokens.keys())
        position_indices = {}
        
        # Skip calculating for positions that don't need pruning (only have one token)
        positions_to_prune = []
        
        for pos in positions:
            if pos == max(all_parallel_tokens.keys()):  # Skip the most recent position
                continue
                
            abs_pos = prompt_length + pos
            if abs_pos < len(normalized_attn):
                position_indices[pos] = abs_pos
                if len(pruned_tokens[pos]) > 1:  # Only consider positions with multiple tokens
                    positions_to_prune.append(pos)
                
                total_positions += 1
                self.pruning_stats["positions_evaluated"] += 1
                self.pruning_stats["total_tokens_considered"] += len(pruned_tokens[pos])

        # For positions with multiple tokens, perform vectorized calculation if using sigmoid
        if self.use_sigmoid_threshold and positions_to_prune:
            if self.debug_mode:
                print(f"\nApplying vectorized sigmoid to {len(positions_to_prune)} positions with multiple tokens")
                
            if self.use_relative_attention:
                # Get relative attention scores for all positions to prune
                rel_positions = torch.tensor([position_indices[pos] for pos in positions_to_prune], 
                                            device=relative_attention.device)
                rel_scores = relative_attention[rel_positions]
                
                # Apply vectorized sigmoid
                should_keep = self.apply_vectorized_sigmoid_threshold(rel_scores, self.relative_threshold)
                
                # Process results
                for i, pos in enumerate(positions_to_prune):
                    if not should_keep[i]:  # Should prune this position
                        tokens_before = len(pruned_tokens[pos])
                        if self.debug_mode:
                            tokens_text = []
                            for tid, _ in pruned_tokens[pos]:
                                try:
                                    tokens_text.append(self.tokenizer.decode([int(tid)]))
                                except:
                                    tokens_text.append(f"<ID:{tid}>")

                            print(f"\nPosition {pos} with tokens {tokens_text}")
                            print(f"  Relative attention score: {rel_scores[i].item():.4f}, threshold: {self.relative_threshold}")
                            print(f"  Number of tokens before pruning: {tokens_before}")
                            print(f"  Position pruned by vectorized sigmoid")
                            
                        # Find the highest probability token to keep
                        best_token = max(pruned_tokens[pos], key=lambda x: x[1])
                        pruned_tokens[pos] = [best_token]  # Keep only the best token

                        tokens_pruned = tokens_before - 1
                        self.pruning_stats["tokens_pruned"] += tokens_pruned
                        self.pruning_stats["positions_pruned"] += 1
                        pruned_positions += 1

                        if self.debug_mode:
                            print(f"  Pruned position {pos}: kept only '{self.tokenizer.decode([int(best_token[0])])}'")
                            print(f"  Removed {tokens_pruned} alternative tokens")
            else:
                # Using regular attention scores
                att_positions = torch.tensor([position_indices[pos] for pos in positions_to_prune], 
                                           device=normalized_attn.device)
                att_scores = normalized_attn[att_positions]
                
                # Apply vectorized sigmoid
                should_keep = self.apply_vectorized_sigmoid_threshold(att_scores, self.attention_threshold)
                
                # Process results
                for i, pos in enumerate(positions_to_prune):
                    if not should_keep[i]:  # Should prune this position
                        tokens_before = len(pruned_tokens[pos])
                        if self.debug_mode:
                            tokens_text = []
                            for tid, _ in pruned_tokens[pos]:
                                try:
                                    tokens_text.append(self.tokenizer.decode([int(tid)]))
                                except:
                                    tokens_text.append(f"<ID:{tid}>")

                            print(f"\nPosition {pos} with tokens {tokens_text}")
                            print(f"  Attention score: {att_scores[i].item():.4f}, threshold: {self.attention_threshold}")
                            print(f"  Number of tokens before pruning: {tokens_before}")
                            print(f"  Position pruned by vectorized sigmoid")
                            
                        # Find the highest probability token to keep
                        best_token = max(pruned_tokens[pos], key=lambda x: x[1])
                        pruned_tokens[pos] = [best_token]  # Keep only the best token

                        tokens_pruned = tokens_before - 1
                        self.pruning_stats["tokens_pruned"] += tokens_pruned
                        self.pruning_stats["positions_pruned"] += 1
                        pruned_positions += 1

                        if self.debug_mode:
                            print(f"  Pruned position {pos}: kept only '{self.tokenizer.decode([int(best_token[0])])}'")
                            print(f"  Removed {tokens_pruned} alternative tokens")
        else:
            # Non-vectorized path for per-position processing
            for pos in positions:
                if pos == max(all_parallel_tokens.keys()):  # Skip the most recent position
                    continue

                # Get absolute position in the sequence
                abs_pos = prompt_length + pos

                # Get attention score to this position
                if abs_pos < len(normalized_attn):
                    attention_score = normalized_attn[abs_pos].item()
                    tokens_before = len(pruned_tokens[pos])

                    if self.debug_mode:
                        tokens_text = []
                        for tid, _ in pruned_tokens[pos]:
                            try:
                                tokens_text.append(self.tokenizer.decode([int(tid)]))
                            except:
                                tokens_text.append(f"<ID:{tid}>")

                        print(f"\nPosition {pos} with tokens {tokens_text}")
                        print(f"  Attention score: {attention_score:.4f}, threshold: {self.attention_threshold}")
                        print(f"  Number of tokens before pruning: {tokens_before}")

                    # If below threshold, skip this position entirely
                    if self.use_relative_attention:
                        # Calculate relative attention score
                        abs_pos_rel = min(abs_pos, len(relative_attention) - 1)
                        relative_score = relative_attention[abs_pos_rel].item()
                        
                        if self.debug_mode:
                            print(f"  Relative attention score: {relative_score:.4f}, threshold: {self.relative_threshold}")
                        
                        if self.use_sigmoid_threshold:
                            # Apply sigmoid decision boundary to relative attention
                            should_keep = self.apply_sigmoid_threshold(relative_score, self.relative_threshold)
                            should_prune = not should_keep and len(pruned_tokens[pos]) > 1
                        else:
                            # Linear threshold
                            should_prune = relative_score < self.relative_threshold and len(pruned_tokens[pos]) > 1
                    else:
                        if self.use_sigmoid_threshold:
                            # Apply sigmoid decision boundary to raw attention
                            should_keep = self.apply_sigmoid_threshold(attention_score, self.attention_threshold)
                            should_prune = not should_keep and len(pruned_tokens[pos]) > 1
                        else:
                            # Linear threshold
                            should_prune = attention_score < self.attention_threshold and len(pruned_tokens[pos]) > 1
                    
                    if should_prune:
                        # Find the highest probability token to keep
                        best_token = max(pruned_tokens[pos], key=lambda x: x[1])
                        pruned_tokens[pos] = [best_token]  # Keep only the best token

                        tokens_pruned = tokens_before - 1
                        self.pruning_stats["tokens_pruned"] += tokens_pruned
                        self.pruning_stats["positions_pruned"] += 1
                        pruned_positions += 1

                        if self.debug_mode:
                            print(f"  Pruned position {pos}: kept only '{self.tokenizer.decode([int(best_token[0])])}'")
                            print(f"  Removed {tokens_pruned} alternative tokens")
                    elif self.debug_mode:
                        print(f"  Position {pos} not pruned (attention score above threshold)")

        if self.debug_mode:
            print(f"\nRetroactive pruning summary:")
            print(f"  Total positions evaluated: {total_positions}")
            print(f"  Positions pruned: {pruned_positions}")
            print(f"  Pruning rate: {(pruned_positions/total_positions*100 if total_positions > 0 else 0):.1f}%")

        return pruned_tokens

    def print_stats(self):
        """Print retroactive pruning statistics."""
        print("\nRetroactive Pruning Stats:")
        print(f"  Positions evaluated: {self.pruning_stats['positions_evaluated']}")
        print(f"  Positions pruned: {self.pruning_stats['positions_pruned']}")
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
        if self.use_relative_attention:
            print(f"  Using relative attention with threshold: {self.relative_threshold}")
        if self.use_multi_scale_attention:
            print(f"  Using multi-scale attention integration")
        if self.use_lci_dynamic_threshold:
            print(f"  Using LCI-based dynamic thresholding")
        if self.use_sigmoid_threshold:
            print(f"  Using sigmoid-based decision boundary (steepness={self.sigmoid_steepness})")
