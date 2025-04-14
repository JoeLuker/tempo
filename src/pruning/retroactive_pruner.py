import torch
import numpy as np
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
        debug_mode: bool = True,
        dynamic_threshold_manager: Optional[DynamicThresholdManager] = None,
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

        # For logging and debugging
        self.pruning_stats = {
            "total_tokens_considered": 0,
            "tokens_pruned": 0,
            "positions_evaluated": 0,
            "positions_pruned": 0,
        }
        
        if self.debug_mode:
            print(f"RetroactivePruner initialized with threshold={attention_threshold}, debug_mode={debug_mode}")

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

    def update_step(self, step: int):
        """Update the current step and threshold if using dynamic thresholding."""
        self.current_step = step
        if self.dynamic_threshold_manager is not None:
            # Get the progress value (0 to 1) instead of the actual threshold
            progress = min(1.0, step / self.dynamic_threshold_manager.max_steps)
            # Scale threshold to be between 0.001 and the final threshold
            final_threshold = self.dynamic_threshold_manager.final_threshold
            min_threshold = 0.001
            self.attention_threshold = min_threshold + (progress * (final_threshold - min_threshold))
            if self.debug_mode:
                print(f"Updated retroactive pruner threshold to {self.attention_threshold:.4f} at step {step}")

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

        # Last few layers often contain most relevant attention
        layers_to_use = min(3, len(cached_attention))
        attention_layers = cached_attention[-layers_to_use:]

        # Average attention patterns across selected layers
        avg_layer_attention = torch.mean(
            torch.stack([layer for layer in attention_layers]), dim=0
        )

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
                print(f"Current threshold: {self.attention_threshold:.4f}")
                print(f"Threshold progression: {self.current_step}/{self.dynamic_threshold_manager.max_steps} steps")
        except Exception as e:
            if self.debug_mode:
                print(f"Error extracting attention for retroactive pruning: {e}")
            return all_parallel_tokens

        # Create a copy of the dictionary to avoid modifying during iteration
        pruned_tokens = {pos: tokens[:] for pos, tokens in all_parallel_tokens.items()}

        # For each previous position with parallel tokens
        pruned_positions = 0
        total_positions = 0

        for pos in sorted(all_parallel_tokens.keys()):
            if pos == max(all_parallel_tokens.keys()):  # Skip the most recent position
                continue

            # Get absolute position in the sequence
            abs_pos = prompt_length + pos

            # Get attention score to this position
            if abs_pos < len(normalized_attn):
                attention_score = normalized_attn[abs_pos].item()
                tokens_before = len(pruned_tokens[pos])
                total_positions += 1

                # Track stats
                self.pruning_stats["positions_evaluated"] += 1
                self.pruning_stats["total_tokens_considered"] += tokens_before

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
                if (
                    attention_score < self.attention_threshold
                    and len(pruned_tokens[pos]) > 1
                ):
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
