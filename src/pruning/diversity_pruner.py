import torch
import time
import numpy as np
from typing import List, Tuple

from src.pruning.diversity_strategy import DiversityPruningStrategy


class DiversityPruner:
    """
    A pruner that clusters and selects tokens based on semantic diversity.
    """

    def __init__(
        self,
        model,
        tokenizer,
        diversity_clusters: int = 3,
        device: str = "mps",
        debug_mode: bool = False,
    ):
        """
        Initialize the diversity pruner.

        Args:
            model: The language model
            tokenizer: HuggingFace tokenizer
            diversity_clusters: Number of clusters for diversity pruning
            device: Device to use for computation
            debug_mode: Enable detailed logging
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.debug_mode = debug_mode

        # Create diversity pruning strategy
        self.diversity_strategy = DiversityPruningStrategy(
            model, tokenizer, diversity_clusters, device
        )

        # Performance tracking
        self.perf_stats = {
            "prune_calls": 0,
            "prune_time": 0,
            "diversity_time": 0,
            "total_tokens": 0,
            "tokens_pruned": 0,
        }

    def prune_parallel_tokens(
        self, input_ids: torch.Tensor, parallel_tokens: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prune parallel tokens using diversity strategy.

        Args:
            input_ids: Current input token IDs
            parallel_tokens: Tuple of (token_ids, token_probs) as NumPy arrays

        Returns:
            Tuple[np.ndarray, np.ndarray]: Pruned (token_ids, token_probs)
        """
        # Performance tracking
        start_time = time.time()
        self.perf_stats["prune_calls"] += 1

        # Extract token IDs and probabilities
        if isinstance(parallel_tokens, tuple) and len(parallel_tokens) == 2:
            token_ids, token_probs = parallel_tokens
        else:
            # Handle unexpected input format
            if self.debug_mode:
                print("Warning: unexpected format for parallel_tokens")
            # Return input as is
            return parallel_tokens

        # Count tokens before pruning
        total_tokens = len(token_ids)
        self.perf_stats["total_tokens"] += total_tokens

        # Skip if there's only one token or no tokens
        if total_tokens <= 1:
            self.perf_stats["prune_time"] += time.time() - start_time
            return parallel_tokens

        # Convert to list of tuples for diversity strategy
        token_tuples = [
            (int(tid), float(prob)) for tid, prob in zip(token_ids, token_probs)
        ]

        # Apply diversity pruning
        diversity_start = time.time()
        pruned_tuples = self.diversity_strategy.prune_tokens(input_ids, token_tuples)
        self.perf_stats["diversity_time"] += time.time() - diversity_start

        # Convert back to NumPy arrays
        if pruned_tuples:
            pruned_ids, pruned_probs = zip(*pruned_tuples)
            pruned_ids_np = np.array(pruned_ids, dtype=np.int32)
            pruned_probs_np = np.array(pruned_probs, dtype=np.float32)
        else:
            # Return empty arrays if no tokens left
            pruned_ids_np = np.array([], dtype=np.int32)
            pruned_probs_np = np.array([], dtype=np.float32)

        # Count pruned tokens
        tokens_pruned = total_tokens - len(pruned_ids_np)
        self.perf_stats["tokens_pruned"] += tokens_pruned

        # Track total time
        self.perf_stats["prune_time"] += time.time() - start_time

        return pruned_ids_np, pruned_probs_np

    def set_diversity_clusters(self, clusters: int):
        """
        Update the number of diversity clusters.

        Args:
            clusters: New number of clusters
        """
        self.diversity_strategy.num_clusters = clusters

    def reset(self):
        """Reset pruner stats for a new generation."""
        self.perf_stats = {
            "prune_calls": 0,
            "prune_time": 0,
            "diversity_time": 0,
            "total_tokens": 0,
            "tokens_pruned": 0,
        }

    def print_performance_stats(self):
        """Print performance statistics."""
        print("\nDiversity Pruner Performance Stats:")
        print(f"  Pruning calls: {self.perf_stats['prune_calls']}")
        print(f"  Total pruning time: {self.perf_stats['prune_time']:.4f}s")
        print(f"  Diversity pruning time: {self.perf_stats['diversity_time']:.4f}s")

        if self.perf_stats["prune_calls"] > 0:
            avg_time = self.perf_stats["prune_time"] / self.perf_stats["prune_calls"]
            print(f"  Average pruning time: {avg_time:.4f}s per call")

        if self.perf_stats["total_tokens"] > 0:
            prune_pct = (
                self.perf_stats["tokens_pruned"] / self.perf_stats["total_tokens"]
            ) * 100
            print(
                f"  Tokens pruned: {self.perf_stats['tokens_pruned']}/{self.perf_stats['total_tokens']} ({prune_pct:.1f}%)"
            )
            print(f"  Diversity clusters: {self.diversity_strategy.num_clusters}")
