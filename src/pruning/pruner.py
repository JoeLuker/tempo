import torch
from typing import Dict, List, Tuple, Optional, Any, Type
from .pruning_strategy import PruningStrategy
from .coherence_strategy import CoherencePruningStrategy
from .diversity_strategy import DiversityPruningStrategy
from .hybrid_strategy import HybridPruningStrategy
from .dynamic_threshold import DynamicThresholdManager
import time

class Pruner:
    """
    Main pruner class that implements token pruning using strategy pattern.
    Optimized for performance and efficient memory usage.
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
        diversity_steps: int = 0
    ):
        """
        Initialize the pruner.
        
        Args:
            model: The language model
            tokenizer: HuggingFace tokenizer
            strategy: Pruning strategy, one of "coherence", "diversity", or "hybrid"
            coherence_threshold: Threshold for pruning tokens based on attention coherence
            diversity_clusters: Number of clusters to use for diversity-optimized pruning
            device: Device to use for computation
            use_dynamic_threshold: Whether to use dynamic thresholds
            max_steps: Maximum number of steps (for calculating dynamic threshold)
            bezier_points: Control points for Bezier curve [p1, p2] between 0-1
            final_threshold: Final threshold value for dynamic threshold
            diversity_steps: Number of steps to use diversity pruning before switching to coherence
        """
        # Invariant: Model and tokenizer must exist
        if model is None or tokenizer is None:
            raise ValueError("Invariant violation: Model and tokenizer must be provided")
            
        # Invariant: Strategy must be valid
        if strategy not in ["coherence", "diversity", "hybrid"]:
            raise ValueError(f"Invariant violation: Strategy must be one of 'coherence', 'diversity', or 'hybrid', got '{strategy}'")
            
        # Invariant: Thresholds must be valid
        if not (0 <= coherence_threshold <= 1):
            raise ValueError(f"Invariant violation: Coherence threshold must be between 0 and 1, got {coherence_threshold}")
            
        if not (0 <= final_threshold <= 1):
            raise ValueError(f"Invariant violation: Final threshold must be between 0 and 1, got {final_threshold}")
            
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_dynamic_threshold = use_dynamic_threshold
        
        # Performance tracking
        self.perf_stats = {
            "prune_calls": 0,
            "prune_time": 0,
            "scoring_time": 0,
            "threshold_time": 0,
            "tokens_pruned": 0,
            "total_tokens": 0,
            "steps": 0
        }
        
        # Create the appropriate strategy
        if strategy == "diversity":
            self.strategy = DiversityPruningStrategy(
                model, tokenizer, diversity_clusters, device
            )
        elif strategy == "hybrid":
            self.strategy = HybridPruningStrategy(
                model, tokenizer, coherence_threshold, diversity_clusters, 
                diversity_steps, device
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
                final_threshold=final_threshold
            )
    
    def prune_parallel_tokens(
        self, 
        input_ids: torch.Tensor, 
        parallel_tokens: List[Tuple[int, float]]
    ) -> Tuple[List[Tuple[int, float]], List[List[Tuple[int, float]]]]:
        """
        Prune parallel tokens based on the selected strategy.
        Optimized for performance with detailed tracking.
        
        Args:
            input_ids: Current input token IDs
            parallel_tokens: List of (token_id, probability) tuples
            
        Returns:
            Tuple[List[Tuple[int, float]], List[List[Tuple[int, float]]]]:
                - Pruned list of (token_id, probability) tuples for current step
                - List of pruned token sets for all steps (when using dynamic threshold)
        """
        # Invariant: Input must be valid
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("Invariant violation: input_ids must be a torch.Tensor")
            
        # Invariant: Parallel tokens must be valid tuples of (token_id, probability)
        if not all(isinstance(t, tuple) and len(t) == 2 and isinstance(t[0], int) and isinstance(t[1], (int, float)) for t in parallel_tokens):
            raise ValueError("Invariant violation: parallel_tokens must be a list of (token_id, probability) tuples")
            
        # Invariant: Probabilities must be between 0 and 1
        if any(not (0 <= t[1] <= 1) for t in parallel_tokens):
            raise ValueError("Invariant violation: Token probabilities must be between 0 and 1")
            
        # Performance tracking
        start_time = time.time()
        self.perf_stats["prune_calls"] += 1
        self.perf_stats["total_tokens"] += len(parallel_tokens)
        self.perf_stats["steps"] += 1
        
        if self.use_dynamic_threshold:
            # Get token scores from the strategy
            scoring_start = time.time()
            token_scores = self.strategy.get_scored_tokens(input_ids, parallel_tokens)
            self.perf_stats["scoring_time"] += time.time() - scoring_start
            
            # Apply pruning using the strategy
            pruning_start = time.time()
            pruned_tokens = self.strategy.prune_tokens(input_ids, parallel_tokens)
            pruning_time = time.time() - pruning_start
            
            # Store token set and scores in dynamic threshold manager
            threshold_start = time.time()
            self.threshold_manager.store_token_set(parallel_tokens, token_scores)
            
            # Reapply threshold to all previous sets with updated threshold
            all_pruned_sets = self.threshold_manager.reapply_threshold_to_all_sets()
            self.perf_stats["threshold_time"] += time.time() - threshold_start
            
            # Track pruning stats
            self.perf_stats["tokens_pruned"] += len(parallel_tokens) - len(pruned_tokens)
            
            total_time = time.time() - start_time
            self.perf_stats["prune_time"] += total_time
            
            # Log detailed performance if enabled
            if hasattr(self, 'detailed_perf') and self.detailed_perf:
                print(f"Pruning: tokens={len(parallel_tokens)}->{len(pruned_tokens)}, "
                      f"scoring={self.perf_stats['scoring_time']:.4f}s, "
                      f"threshold={self.perf_stats['threshold_time']:.4f}s, "
                      f"total={total_time:.4f}s")
                
            return pruned_tokens, all_pruned_sets
        else:
            # Simple pruning without dynamic threshold
            pruning_start = time.time()
            pruned_tokens = self.strategy.prune_tokens(input_ids, parallel_tokens)
            pruning_time = time.time() - pruning_start
            
            # Track pruning stats
            self.perf_stats["tokens_pruned"] += len(parallel_tokens) - len(pruned_tokens)
            self.perf_stats["prune_time"] += time.time() - start_time
            
            # Log detailed performance if enabled
            if hasattr(self, 'detailed_perf') and self.detailed_perf:
                print(f"Pruning: tokens={len(parallel_tokens)}->{len(pruned_tokens)}, "
                      f"time={pruning_time:.4f}s")
                
            return pruned_tokens, [pruned_tokens]
    
    def get_final_pruned_sets(self) -> List[List[Tuple[int, float]]]:
        """
        Get the final pruned sets after applying dynamic thresholding.
        
        Returns:
            List[List[Tuple[int, float]]]: Final pruned token sets
        """
        if self.use_dynamic_threshold:
            threshold_start = time.time()
            final_sets = self.threshold_manager.reapply_threshold_to_all_sets()
            self.perf_stats["threshold_time"] += time.time() - threshold_start
            return final_sets
        return []
    
    def reset(self):
        """Reset the pruner for a new generation."""
        # Reset strategy and threshold manager
        if hasattr(self.strategy, 'reset'):
            self.strategy.reset()
        
        if self.use_dynamic_threshold:
            self.threshold_manager.reset()
            
        # Optionally reset performance stats
        if hasattr(self, 'reset_perf') and self.reset_perf:
            self.perf_stats = {
                "prune_calls": 0,
                "prune_time": 0,
                "scoring_time": 0,
                "threshold_time": 0,
                "tokens_pruned": 0,
                "total_tokens": 0,
                "steps": 0
            }
    
    def print_performance_stats(self):
        """Print performance statistics for pruning operations."""
        print("\nPruner Performance Stats:")
        print(f"  Pruning calls: {self.perf_stats['prune_calls']}")
        print(f"  Total pruning time: {self.perf_stats['prune_time']:.4f}s")
        
        if self.perf_stats['prune_calls'] > 0:
            avg_time = self.perf_stats['prune_time'] / self.perf_stats['prune_calls']
            print(f"  Average pruning time: {avg_time:.4f}s per call")
        
        if self.use_dynamic_threshold:
            print(f"  Token scoring time: {self.perf_stats['scoring_time']:.4f}s")
            print(f"  Threshold application time: {self.perf_stats['threshold_time']:.4f}s")
            
        total_tokens = self.perf_stats['total_tokens']
        tokens_pruned = self.perf_stats['tokens_pruned']
        if total_tokens > 0:
            prune_pct = (tokens_pruned / total_tokens) * 100
            print(f"  Tokens pruned: {tokens_pruned}/{total_tokens} ({prune_pct:.1f}%)")
        
        print(f"  Steps processed: {self.perf_stats['steps']}")
        
        # Print strategy-specific stats if available
        if hasattr(self.strategy, 'print_performance_stats'):
            self.strategy.print_performance_stats()
    
    def enable_detailed_perf(self, enabled=True):
        """Enable or disable detailed per-call performance logging."""
        self.detailed_perf = enabled
        
        # Also enable for strategy if it supports it
        if hasattr(self.strategy, 'enable_detailed_perf'):
            self.strategy.enable_detailed_perf(enabled)
            
    def set_reset_perf(self, reset=True):
        """Set whether to reset performance stats on reset()."""
        self.reset_perf = reset 