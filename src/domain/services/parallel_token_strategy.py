"""Smart strategy selection for parallel token processing.

This module selects the optimal processing strategy based on the isolation mode:
- Isolated mode: Use KV cache (sequential processing naturally isolates tokens)
- Visible mode: Disable KV cache and use batched processing (allows cross-attention)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class ParallelProcessingStrategy(Enum):
    """Strategy for processing parallel tokens."""

    SEQUENTIAL_CACHED = "sequential_cached"
    """Process tokens sequentially with KV cache. Fast, naturally isolated."""

    BATCHED_NO_CACHE = "batched_no_cache"
    """Process tokens in batch without KV cache. Slower, allows cross-attention."""


@dataclass
class StrategySelection:
    """Result of strategy selection."""

    strategy: ParallelProcessingStrategy
    reason: str
    use_kv_cache: bool
    use_batched_generator: bool

    def __str__(self) -> str:
        return f"{self.strategy.value}: {self.reason}"


class ParallelTokenStrategySelector:
    """Selects optimal processing strategy based on configuration.

    The key insight:
    - Isolated mode: Sequential KV-cached processing naturally prevents cross-attention (optimal!)
    - Visible mode: Need batched processing to allow symmetric cross-attention (necessary trade-off)
    """

    @staticmethod
    def select_strategy(
        isolate_parallel_tokens: bool,
        num_parallel_tokens: int,
        disable_kv_cache_override: bool = False
    ) -> StrategySelection:
        """Select the optimal processing strategy.

        Args:
            isolate_parallel_tokens: Whether to prevent cross-parallel attention
            num_parallel_tokens: Number of parallel tokens to process
            disable_kv_cache_override: Force disable KV cache (for testing)

        Returns:
            StrategySelection with chosen strategy and rationale
        """
        # Single token: always use sequential cached (no parallelism to manage)
        if num_parallel_tokens <= 1:
            return StrategySelection(
                strategy=ParallelProcessingStrategy.SEQUENTIAL_CACHED,
                reason="Single token - use standard KV-cached generation",
                use_kv_cache=True and not disable_kv_cache_override,
                use_batched_generator=False
            )

        # Override: KV cache disabled globally
        if disable_kv_cache_override:
            return StrategySelection(
                strategy=ParallelProcessingStrategy.BATCHED_NO_CACHE,
                reason="KV cache disabled globally - use batched processing",
                use_kv_cache=False,
                use_batched_generator=True
            )

        # ISOLATED MODE: Sequential KV-cached is optimal!
        # Later tokens naturally can't see earlier ones because they're processed separately
        if isolate_parallel_tokens:
            return StrategySelection(
                strategy=ParallelProcessingStrategy.SEQUENTIAL_CACHED,
                reason=(
                    f"Isolated mode with {num_parallel_tokens} parallel tokens - "
                    "sequential KV-cached processing naturally prevents cross-attention (optimal!)"
                ),
                use_kv_cache=True,
                use_batched_generator=False
            )

        # VISIBLE MODE: Need batched processing for symmetric cross-attention
        # Without batch processing, later tokens see earlier ones but not vice versa (asymmetric)
        else:
            return StrategySelection(
                strategy=ParallelProcessingStrategy.BATCHED_NO_CACHE,
                reason=(
                    f"Visible mode with {num_parallel_tokens} parallel tokens - "
                    "batched processing required for symmetric cross-attention "
                    "(slower but correct)"
                ),
                use_kv_cache=False,
                use_batched_generator=True
            )

    @staticmethod
    def explain_strategy(isolate_parallel_tokens: bool) -> str:
        """Get human-readable explanation of strategy for a given mode.

        Args:
            isolate_parallel_tokens: Isolation mode

        Returns:
            Explanation string
        """
        if isolate_parallel_tokens:
            return """
ISOLATED MODE (isolate_parallel_tokens=True):
Strategy: Sequential KV-cached processing (optimal!)

Why this is optimal:
- Each parallel token is processed in its own forward pass
- Later tokens naturally cannot see earlier tokens (already in cache)
- No cross-parallel attention possible (desired behavior)
- Full KV cache benefits (fast, memory efficient)
- No trade-offs required!

Performance: O(n) per token, KV cache reused
"""
        else:
            return """
VISIBLE MODE (isolate_parallel_tokens=False):
Strategy: Batched processing without KV cache (necessary trade-off)

Why batched processing is needed:
- Sequential processing creates asymmetric visibility:
  * Token1 processed first → in cache
  * Token2 processed → sees Token1 (but Token1 can't see Token2!)
  * Token3 processed → sees Token1 AND Token2 (completely asymmetric!)
- Batched processing processes all tokens simultaneously:
  * All tokens see the same base context
  * All tokens can attend to each other (symmetric, desired)
  * Custom attention masks control exactly what they see

Trade-off: Slower (no KV cache) but correct cross-attention

Performance: O(n²) for sequence reprocessing
"""


def get_optimal_strategy_for_config(config) -> StrategySelection:
    """Helper to get strategy from a GenerationConfig.

    Args:
        config: GenerationConfig object

    Returns:
        StrategySelection for this configuration
    """
    return ParallelTokenStrategySelector.select_strategy(
        isolate_parallel_tokens=config.isolate_parallel_tokens,
        num_parallel_tokens=1,  # Will be updated per step
        disable_kv_cache_override=config.disable_kv_cache
    )
