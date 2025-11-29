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

    Attributes:
        update_scale: Scale factor for Hebbian weight updates.
                     Use 0.0 for baseline (no modifications).
        window_size: Maximum tokens in context before eviction.
        decay: Importance decay rate per step (0.0-1.0).
        max_mods_per_layer: Maximum modifications to store per layer.
        min_memory_gb: Minimum available memory before skipping modifications.
        n_sink_tokens: Number of initial tokens to keep as attention sinks.
                      These tokens are never evicted and serve as anchors.
                      Critical for models not trained with sliding window attention.
        update_target: Which projections to modify: "k", "v", or "both".
                      - "k": Modify key projections (affects what queries match)
                      - "v": Modify value projections (stores content for retrieval)
                      - "both": Modify both K and V projections
    """
    update_scale: float = 1e-6
    window_size: int = 32
    decay: float = 0.99
    max_mods_per_layer: int = 100
    min_memory_gb: float = 2.0
    n_sink_tokens: int = 4  # StreamingLLM default
    update_target: str = "v"  # "k", "v", or "both" - default to V for content storage

    def with_scale(self, scale: float) -> "HebbianConfig":
        """Return new config with different scale."""
        return HebbianConfig(
            update_scale=scale,
            window_size=self.window_size,
            decay=self.decay,
            max_mods_per_layer=self.max_mods_per_layer,
            min_memory_gb=self.min_memory_gb,
            n_sink_tokens=self.n_sink_tokens,
            update_target=self.update_target,
        )

    def with_target(self, target: str) -> "HebbianConfig":
        """Return new config with different update target."""
        return HebbianConfig(
            update_scale=self.update_scale,
            window_size=self.window_size,
            decay=self.decay,
            max_mods_per_layer=self.max_mods_per_layer,
            min_memory_gb=self.min_memory_gb,
            n_sink_tokens=self.n_sink_tokens,
            update_target=target,
        )


# Preset configurations
BASELINE = HebbianConfig(update_scale=0.0)
HEBBIAN = HebbianConfig(update_scale=1e-6)

# For experiments with different window sizes
def config_for_window(
    window_size: int,
    hebbian: bool = True,
    n_sink_tokens: int = 4,
) -> HebbianConfig:
    """Create config with specified window size."""
    return HebbianConfig(
        update_scale=1e-6 if hebbian else 0.0,
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
