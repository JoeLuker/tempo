"""Performance tracking implementation for the TEMPO system.

This module implements performance tracking for various operations
in the token generation pipeline.
"""

import time
from src.utils.logging_utils import LoggingMixin
from src.domain.interfaces.performance_tracker import PerformanceTrackerInterface


class PerformanceTracker(LoggingMixin, PerformanceTrackerInterface):
    """Tracks performance metrics for token generation operations."""
    
    def __init__(self):
        """Initialize the performance tracker."""
        super().__init__()
        
        # Performance statistics
        self.stats = {
            "tokenization_calls": 0,
            "tokenization_time": 0.0,
            "tokenization_cache_hits": 0,
            "model_calls": 0,
            "model_time": 0.0,
            "model_tokens_processed": 0,
            "decode_calls": 0,
            "decode_time": 0.0,
            "decode_tokens": 0,
            "decode_cache_hits": 0,
            "isolated_tokens_processed": 0,
            "start_time": time.time()
        }
        
        # Setup logging
        self.setup_logging("performance_tracker", "performance_tracker_debug.log")
    
    def track_tokenization(self, duration: float, cache_hit: bool) -> None:
        """Track tokenization performance.
        
        Args:
            duration: Time taken for tokenization
            cache_hit: Whether this was a cache hit
        """
        self.stats["tokenization_calls"] += 1
        self.stats["tokenization_time"] += duration
        if cache_hit:
            self.stats["tokenization_cache_hits"] += 1
    
    def track_model_call(self, duration: float, num_tokens: int = 1) -> None:
        """Track model forward pass performance.
        
        Args:
            duration: Time taken for model forward pass
            num_tokens: Number of tokens processed (for parallel generation)
        """
        self.stats["model_calls"] += 1
        self.stats["model_time"] += duration
        self.stats["model_tokens_processed"] += num_tokens
        
        # Track isolated tokens separately
        if num_tokens > 1:
            self.stats["isolated_tokens_processed"] += num_tokens
    
    def track_decode(self, duration: float, num_tokens: int, cache_hits: int) -> None:
        """Track token decoding performance.
        
        Args:
            duration: Time taken for decoding
            num_tokens: Number of tokens decoded
            cache_hits: Number of cache hits during decoding
        """
        self.stats["decode_calls"] += 1
        self.stats["decode_time"] += duration
        self.stats["decode_tokens"] += num_tokens
        self.stats["decode_cache_hits"] += cache_hits
    
    def get_stats(self) -> Dict:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        # Calculate derived statistics
        stats = self.stats.copy()
        
        # Calculate rates and averages
        if stats["tokenization_calls"] > 0:
            stats["avg_tokenization_time"] = stats["tokenization_time"] / stats["tokenization_calls"]
            stats["tokenization_cache_hit_rate"] = (
                stats["tokenization_cache_hits"] / stats["tokenization_calls"] * 100
            )
        else:
            stats["avg_tokenization_time"] = 0.0
            stats["tokenization_cache_hit_rate"] = 0.0
        
        if stats["model_calls"] > 0:
            stats["avg_model_time"] = stats["model_time"] / stats["model_calls"]
            stats["avg_tokens_per_call"] = stats["model_tokens_processed"] / stats["model_calls"]
        else:
            stats["avg_model_time"] = 0.0
            stats["avg_tokens_per_call"] = 0.0
        
        if stats["decode_calls"] > 0:
            stats["avg_decode_time"] = stats["decode_time"] / stats["decode_calls"]
            stats["avg_tokens_per_decode"] = stats["decode_tokens"] / stats["decode_calls"]
        else:
            stats["avg_decode_time"] = 0.0
            stats["avg_tokens_per_decode"] = 0.0
        
        if stats["decode_tokens"] > 0:
            stats["decode_cache_hit_rate"] = stats["decode_cache_hits"] / stats["decode_tokens"] * 100
        else:
            stats["decode_cache_hit_rate"] = 0.0
        
        # Calculate total elapsed time
        stats["total_elapsed_time"] = time.time() - stats["start_time"]
        
        return stats
    
    def print_stats(self) -> None:
        """Print performance statistics."""
        stats = self.get_stats()
        
        print("\nPerformance Statistics:")
        print(f"  Total elapsed time: {stats['total_elapsed_time']:.2f}s")
        
        # Tokenization stats
        print(f"\n  Tokenization:")
        print(f"    Calls: {stats['tokenization_calls']}")
        print(f"    Total time: {stats['tokenization_time']:.4f}s")
        print(f"    Average time: {stats['avg_tokenization_time']*1000:.2f}ms")
        print(f"    Cache hit rate: {stats['tokenization_cache_hit_rate']:.1f}%")
        
        # Model stats
        print(f"\n  Model Forward Pass:")
        print(f"    Calls: {stats['model_calls']}")
        print(f"    Total time: {stats['model_time']:.4f}s")
        print(f"    Average time: {stats['avg_model_time']*1000:.2f}ms")
        print(f"    Tokens processed: {stats['model_tokens_processed']}")
        print(f"    Average tokens/call: {stats['avg_tokens_per_call']:.2f}")
        
        # Isolated token optimization
        if stats['isolated_tokens_processed'] > 0:
            print(f"\n  Isolated Parallel Token Optimization:")
            print(f"    Tokens processed: {stats['isolated_tokens_processed']}")
            # Estimate time saved by processing multiple tokens in one pass
            potential_calls = stats['isolated_tokens_processed']
            actual_calls = stats['model_calls']
            saved_calls = max(0, potential_calls - actual_calls)
            if stats['avg_model_time'] > 0:
                saved_time = saved_calls * stats['avg_model_time']
                print(f"    Estimated time saved: {saved_time*1000:.1f}ms")
        
        # Decode stats
        print(f"\n  Token Decoding:")
        print(f"    Calls: {stats['decode_calls']}")
        print(f"    Total time: {stats['decode_time']:.4f}s")
        print(f"    Average time: {stats['avg_decode_time']*1000:.2f}ms")
        print(f"    Tokens decoded: {stats['decode_tokens']}")
        print(f"    Cache hit rate: {stats['decode_cache_hit_rate']:.1f}%")
    
    def reset(self) -> None:
        """Reset all performance statistics."""
        self.stats = {
            "tokenization_calls": 0,
            "tokenization_time": 0.0,
            "tokenization_cache_hits": 0,
            "model_calls": 0,
            "model_time": 0.0,
            "model_tokens_processed": 0,
            "decode_calls": 0,
            "decode_time": 0.0,
            "decode_tokens": 0,
            "decode_cache_hits": 0,
            "isolated_tokens_processed": 0,
            "start_time": time.time()
        }
        
        if self.debug_mode:
            self.log("Reset performance statistics")
