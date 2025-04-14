import torch
import time
from typing import List, Tuple, Dict, Optional
import numpy as np

from src.pruning.pruning_strategy import PruningStrategy
from src.pruning.coherence_strategy import CoherencePruningStrategy
from src.pruning.diversity_strategy import DiversityPruningStrategy
from src.pruning.hybrid_strategy import HybridPruningStrategy
from src.pruning.dynamic_threshold import DynamicThresholdManager


class Pruner:
    """
    Prunes parallel tokens using various strategies.
    Optimized for tensor operations and performance.
    """

    def __init__(
        self,
        model,
        tokenizer,
        strategy: str = "coherence",
        coherence_threshold: float = 0.3,
        diversity_clusters: int = 3,
        device: str = "mps",
        use_dynamic_threshold: bool = False,
        max_steps: Optional[int] = None,
        bezier_points: Optional[List[float]] = None,
        final_threshold: float = 1.0,
        diversity_steps: int = 0,
        skip_reapply_threshold: bool = False,
        use_relu: bool = False,
        relu_activation_point: float = 0.5,
    ):
        """
        Initialize the pruner.

        Args:
            model: The language model
            tokenizer: HuggingFace tokenizer
            strategy: Pruning strategy ("coherence", "diversity", or "hybrid")
            coherence_threshold: Threshold for coherence-based pruning (0-1)
            diversity_clusters: Number of clusters for diversity-based pruning
            device: Device to use for computation
            use_dynamic_threshold: Whether to use dynamic thresholding
            max_steps: Maximum number of steps for dynamic threshold
            bezier_points: Control points for Bezier curve [p1, p2]
            final_threshold: Final threshold value for dynamic threshold
            diversity_steps: Number of initial steps to use diversity strategy
            skip_reapply_threshold: Skip reapplying threshold to previous steps
            use_relu: Whether to use ReLU-based transitions instead of Bezier
            relu_activation_point: Point at which ReLU transition begins (0-1)
        """
        # Invariant: Model and tokenizer must be provided
        if model is None or tokenizer is None:
            raise ValueError(
                "Invariant violation: Model and tokenizer must be provided"
            )

        # Invariant: Strategy must be valid
        if strategy not in ["coherence", "diversity", "hybrid"]:
            raise ValueError(
                f"Invariant violation: Strategy must be one of 'coherence', 'diversity', or 'hybrid', got '{strategy}'"
            )

        # Invariant: Thresholds must be valid
        if not (0 <= coherence_threshold <= 1):
            raise ValueError(
                f"Invariant violation: Coherence threshold must be between 0 and 1, got {coherence_threshold}"
            )

        if not (0 <= final_threshold <= 1):
            raise ValueError(
                f"Invariant violation: Final threshold must be between 0 and 1, got {final_threshold}"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_dynamic_threshold = use_dynamic_threshold
        self.use_numpy_format = True  # Flag to determine data format consistency
        self.skip_reapply_threshold = skip_reapply_threshold  # Use the provided value

        # Performance tracking
        self.perf_stats = {
            "prune_calls": 0,
            "prune_time": 0,
            "scoring_time": 0,
            "threshold_time": 0,
            "tokens_pruned": 0,
            "total_tokens": 0,
            "steps": 0,
            "format_conversion_time": 0,
            "format_conversions": 0,
        }

        # Create the appropriate strategy
        if strategy == "diversity":
            self.strategy = DiversityPruningStrategy(
                model, tokenizer, diversity_clusters, device
            )
        elif strategy == "hybrid":
            self.strategy = HybridPruningStrategy(
                model,
                tokenizer,
                coherence_threshold,
                diversity_clusters,
                diversity_steps,
                device,
            )
        else:  # default to coherence
            self.strategy = CoherencePruningStrategy(
                model, tokenizer, coherence_threshold, device
            )

        # Create dynamic threshold manager if needed
        if use_dynamic_threshold:
            self.threshold_manager = DynamicThresholdManager(
                base_threshold=coherence_threshold,
                max_steps=max_steps,
                bezier_points=bezier_points,
                final_threshold=final_threshold,
                use_relu=use_relu,
                relu_activation_point=relu_activation_point,
            )

    def _convert_to_tuples(self, token_ids, token_probs):
        """
        Convert NumPy arrays to a list of tuples for compatibility.
        Optimized to avoid unnecessary conversions.
        """
        start_time = time.time()
        self.perf_stats["format_conversions"] += 1

        # If inputs are already tuples, return as is
        if (
            isinstance(token_ids, list)
            and isinstance(token_probs, list)
            and len(token_ids) == len(token_probs)
        ):
            if all(isinstance(item, tuple) for item in zip(token_ids, token_probs)):
                # Ensure token IDs are integers
                return [(int(tid), float(prob)) for tid, prob in zip(token_ids, token_probs)]

        # If inputs are numpy arrays, convert to list of tuples
        if isinstance(token_ids, np.ndarray) and isinstance(token_probs, np.ndarray):
            # Ensure token IDs are integers and probabilities are floats
            result = [
                (int(tid), float(prob)) for tid, prob in zip(token_ids.astype(np.int32), token_probs)
            ]
            self.perf_stats["format_conversion_time"] += time.time() - start_time
            return result

        # Already in the right format
        if isinstance(token_ids, (list, tuple)) and isinstance(
            token_probs, (list, tuple)
        ):
            # Ensure token IDs are integers
            result = [(int(tid), float(prob)) for tid, prob in zip(token_ids, token_probs)]
            self.perf_stats["format_conversion_time"] += time.time() - start_time
            return result

        # Fallback
        self.perf_stats["format_conversion_time"] += time.time() - start_time
        return [(int(tid), float(prob)) for tid, prob in zip(token_ids, token_probs)]

    def _convert_from_tuples(self, tuples_list):
        """
        Convert a list of tuples to NumPy arrays.
        Optimized to avoid unnecessary conversions.
        """
        start_time = time.time()
        self.perf_stats["format_conversions"] += 1

        # Handle empty list case
        if not tuples_list:
            self.perf_stats["format_conversion_time"] += time.time() - start_time
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        # Check if input is already numpy arrays
        if isinstance(tuples_list, tuple) and len(tuples_list) == 2:
            if isinstance(tuples_list[0], np.ndarray) and isinstance(
                tuples_list[1], np.ndarray
            ):
                # Ensure correct dtypes
                ids = tuples_list[0].astype(np.int32)
                probs = tuples_list[1].astype(np.float32)
                self.perf_stats["format_conversion_time"] += time.time() - start_time
                return ids, probs

        # Convert tuples to arrays
        try:
            ids, probs = zip(*tuples_list)
            # Ensure correct dtypes
            result = (
                np.array(ids, dtype=np.int32),
                np.array(probs, dtype=np.float32)
            )
            self.perf_stats["format_conversion_time"] += time.time() - start_time
            return result
        except Exception:
            # Fallback for malformed input
            self.perf_stats["format_conversion_time"] += time.time() - start_time
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    def prune_parallel_tokens(
        self, input_ids: torch.Tensor, parallel_tokens: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Prune parallel tokens based on the selected strategy.
        Optimized for performance with detailed tracking.

        Args:
            input_ids: Current input token IDs
            parallel_tokens: Tuple of (token_ids, token_probs) as NumPy arrays

        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
                - Pruned (token_ids, token_probs) for current step
                - List of pruned token sets for all steps (when using dynamic threshold)
        """
        # Extract token IDs and probabilities
        if isinstance(parallel_tokens, tuple) and len(parallel_tokens) == 2:
            token_ids, token_probs = parallel_tokens
        else:
            # For backward compatibility, handle list of tuples
            token_ids, token_probs = self._convert_from_tuples(parallel_tokens)

        # Invariant: Input must be valid
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("Invariant violation: input_ids must be a torch.Tensor")

        # Performance tracking
        start_time = time.time()
        self.perf_stats["prune_calls"] += 1
        self.perf_stats["total_tokens"] += len(token_ids)
        self.perf_stats["steps"] += 1

        # Convert to list of tuples for strategy compatibility
        parallel_tuples = self._convert_to_tuples(token_ids, token_probs)

        if self.use_dynamic_threshold:
            # Get token scores from the strategy
            scoring_start = time.time()
            token_scores = self.strategy.get_scored_tokens(input_ids, parallel_tuples)
            scoring_time = time.time() - scoring_start
            self.perf_stats["scoring_time"] += scoring_time

            # Store token set and scores in dynamic threshold manager
            threshold_store_start = time.time()
            self.threshold_manager.store_token_set(parallel_tuples, token_scores)
            
            # Only reapply threshold if not skipping
            if not self.skip_reapply_threshold:
                # Get updated pruned sets with current dynamic threshold
                all_pruned_sets = self.threshold_manager.reapply_threshold_to_all_sets()
                
                # Convert the last set (current step) back to numpy arrays
                if all_pruned_sets:
                    current_pruned = all_pruned_sets[-1]
                    if current_pruned:
                        pruned_ids, pruned_probs = zip(*current_pruned)
                        pruned_ids = np.array(pruned_ids, dtype=np.int32)
                        pruned_probs = np.array(pruned_probs, dtype=np.float32)
                    else:
                        # If no tokens survived pruning, use highest probability token
                        max_prob_idx = np.argmax(token_probs)
                        pruned_ids = np.array([token_ids[max_prob_idx]], dtype=np.int32)
                        pruned_probs = np.array([token_probs[max_prob_idx]], dtype=np.float32)
                else:
                    # Fallback to highest probability token
                    max_prob_idx = np.argmax(token_probs)
                    pruned_ids = np.array([token_ids[max_prob_idx]], dtype=np.int32)
                    pruned_probs = np.array([token_probs[max_prob_idx]], dtype=np.float32)

                threshold_time = time.time() - threshold_store_start
                self.perf_stats["threshold_time"] += threshold_time

                # Convert all pruned sets to numpy arrays for consistency
                all_pruned_arrays = []
                for pruned_set in all_pruned_sets[:-1]:  # Exclude current step
                    if pruned_set:
                        p_ids, p_probs = zip(*pruned_set)
                        all_pruned_arrays.append(
                            (np.array(p_ids, dtype=np.int32), 
                             np.array(p_probs, dtype=np.float32))
                        )
                    else:
                        # Empty set case
                        all_pruned_arrays.append(
                            (np.array([], dtype=np.int32), 
                             np.array([], dtype=np.float32))
                        )

                return (pruned_ids, pruned_probs), all_pruned_arrays
            else:
                # Apply pruning using the strategy with current threshold
                pruning_start = time.time()
                current_threshold = self.threshold_manager.get_current_threshold()
                self.strategy.coherence_threshold = current_threshold  # Update strategy threshold
                pruned_tuples = self.strategy.prune_tokens(input_ids, parallel_tuples)
                strategy_time = time.time() - pruning_start

                # Convert pruned tuples back to numpy arrays
                if pruned_tuples:
                    pruned_ids, pruned_probs = zip(*pruned_tuples)
                    pruned_ids = np.array(pruned_ids, dtype=np.int32)
                    pruned_probs = np.array(pruned_probs, dtype=np.float32)
                else:
                    # If no tokens survived pruning, use highest probability token
                    max_prob_idx = np.argmax(token_probs)
                    pruned_ids = np.array([token_ids[max_prob_idx]], dtype=np.int32)
                    pruned_probs = np.array([token_probs[max_prob_idx]], dtype=np.float32)

                return (pruned_ids, pruned_probs), []

        else:
            # Non-dynamic threshold case - use strategy directly
            pruning_start = time.time()
            pruned_tuples = self.strategy.prune_tokens(input_ids, parallel_tuples)
            strategy_time = time.time() - pruning_start

            # Convert pruned tuples back to numpy arrays
            if pruned_tuples:
                pruned_ids, pruned_probs = zip(*pruned_tuples)
                pruned_ids = np.array(pruned_ids, dtype=np.int32)
                pruned_probs = np.array(pruned_probs, dtype=np.float32)
            else:
                # If no tokens survived pruning, use highest probability token
                max_prob_idx = np.argmax(token_probs)
                pruned_ids = np.array([token_ids[max_prob_idx]], dtype=np.int32)
                pruned_probs = np.array([token_probs[max_prob_idx]], dtype=np.float32)

            return (pruned_ids, pruned_probs), []

    def get_final_pruned_sets(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the final pruned sets after applying dynamic thresholding.

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: Final pruned token sets as NumPy arrays
        """
        if self.use_dynamic_threshold:
            threshold_start = time.time()
            final_sets = self.threshold_manager.reapply_threshold_to_all_sets()
            self.perf_stats["threshold_time"] += time.time() - threshold_start

            # Convert to NumPy arrays
            final_array_sets = []
            for pruned_set in final_sets:
                set_ids, set_probs = self._convert_from_tuples(pruned_set)
                final_array_sets.append((set_ids, set_probs))

            return final_array_sets
        return []

    def reset(self):
        """Reset the pruner for a new generation."""
        # Reset strategy and threshold manager
        if hasattr(self.strategy, "reset"):
            self.strategy.reset()

        if self.use_dynamic_threshold:
            self.threshold_manager.reset()

        # Optionally reset performance stats
        if hasattr(self, "reset_perf") and self.reset_perf:
            self.perf_stats = {
                "prune_calls": 0,
                "prune_time": 0,
                "scoring_time": 0,
                "threshold_time": 0,
                "tokens_pruned": 0,
                "total_tokens": 0,
                "steps": 0,
                "format_conversion_time": 0,
                "format_conversions": 0,
            }

    def print_performance_stats(self):
        """Print performance statistics for pruning operations."""
        print("\nPruner Performance Stats:")
        print(f"  Pruning calls: {self.perf_stats['prune_calls']}")
        print(f"  Total pruning time: {self.perf_stats['prune_time']:.4f}s")

        if self.perf_stats["prune_calls"] > 0:
            avg_time = self.perf_stats["prune_time"] / self.perf_stats["prune_calls"]
            print(f"  Average pruning time: {avg_time:.4f}s per call")

        if self.use_dynamic_threshold:
            print(f"  Token scoring time: {self.perf_stats['scoring_time']:.4f}s")
            print(
                f"  Threshold application time: {self.perf_stats['threshold_time']:.4f}s"
            )

        total_tokens = self.perf_stats["total_tokens"]
        tokens_pruned = self.perf_stats["tokens_pruned"]
        if total_tokens > 0:
            prune_pct = (tokens_pruned / total_tokens) * 100
            print(f"  Tokens pruned: {tokens_pruned}/{total_tokens} ({prune_pct:.1f}%)")

        # Print format conversion stats
        conversions = self.perf_stats["format_conversions"]
        if conversions > 0:
            conversion_time = self.perf_stats["format_conversion_time"]
            avg_conversion = conversion_time / conversions
            conversion_pct = (
                (conversion_time / self.perf_stats["prune_time"]) * 100
                if self.perf_stats["prune_time"] > 0
                else 0
            )
            print(f"  Format conversions: {conversions}")
            print(
                f"  Format conversion time: {conversion_time:.4f}s ({conversion_pct:.1f}% of total)"
            )
            print(f"  Avg conversion time: {avg_conversion*1000:.2f}ms")

        print(f"  Steps processed: {self.perf_stats['steps']}")

        # Print strategy-specific stats if available
        if hasattr(self.strategy, "print_performance_stats"):
            self.strategy.print_performance_stats()

    def enable_detailed_perf(self, enabled=True):
        """Enable or disable detailed per-call performance logging."""
        self.detailed_perf = enabled

        # Also enable for strategy if it supports it
        if hasattr(self.strategy, "enable_detailed_perf"):
            self.strategy.enable_detailed_perf(enabled)

    def set_reset_perf(self, reset=True):
        """Set whether to reset performance stats on reset()."""
        self.reset_perf = reset
