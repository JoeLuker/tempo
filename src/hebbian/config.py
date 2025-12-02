"""
Consolidated configuration for Hebbian consolidation experiments.

Single source of truth for all Hebbian parameters. Both engines and
benchmarks import from here to ensure consistency.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class HebbianConfig:
    """Configuration for Hebbian consolidation engine.

    The engine uses a memory bank approach: when tokens are evicted from the
    sliding window, their K/V vectors are stored in a memory bank that all
    future queries can attend to. This enables recall of information that
    would otherwise be lost.

    Attributes:
        memory_enabled: Whether to store evicted K/V in memory bank.
                       When False, acts as standard sliding window attention.
        window_size: Maximum tokens in context before eviction.
        decay: Importance decay rate per step (0.0-1.0).
        max_memory_per_layer: Maximum K/V pairs to store per layer.
        n_sink_tokens: Number of initial tokens to keep as attention sinks.
                      These tokens are never evicted and serve as anchors.
        min_importance: Minimum importance to store in memory bank.
                       Higher = only store important tokens. Lower = store more.
    """
    memory_enabled: bool = True
    window_size: int = 32
    decay: float = 0.99
    max_memory_per_layer: int = 500  # Higher = better recall, more memory
    n_sink_tokens: int = 4  # StreamingLLM default
    min_importance: float = 0.1  # Store most tokens by default
    memory_top_k: int = 64  # 0 = use all memory, >0 = retrieve top-k per query

    # Legacy alias for update_scale (for backward compatibility)
    @property
    def update_scale(self) -> float:
        return 1.0 if self.memory_enabled else 0.0


# Preset configurations
BASELINE = HebbianConfig(memory_enabled=False)
HEBBIAN = HebbianConfig(memory_enabled=True)

# For experiments with different window sizes
def config_for_window(
    window_size: int,
    memory_enabled: bool = True,
    n_sink_tokens: int = 4,
) -> HebbianConfig:
    """Create config with specified window size."""
    return HebbianConfig(
        memory_enabled=memory_enabled,
        window_size=window_size,
        n_sink_tokens=n_sink_tokens,
    )


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments.

    Attributes:
        n_seeds: Number of random seeds to test.
        model_name: HuggingFace model identifier.
        output_dir: Directory for benchmark results.
        variants: List of (name, scale) tuples to test.
    """
    n_seeds: int = 10
    model_name: str = "deepcogito/cogito-v1-preview-llama-3B"
    output_dir: str = "experiments/hebbian/results"
    min_memory_gb: float = 10.0

    # Default variants for formula comparison
    variants: list = field(default_factory=lambda: [
        ("baseline", 0.0),
        ("hebbian", 1e-6),
    ])


# Default benchmark config
DEFAULT_BENCHMARK = BenchmarkConfig()
