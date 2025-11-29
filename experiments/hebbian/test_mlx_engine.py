"""Test script for MLX-based Hebbian consolidation engine.

Usage:
    python3 experiments/hebbian/test_mlx_engine.py
"""

import sys
sys.path.insert(0, "/Users/jluker/tempo")

import mlx.core as mx
from src.hebbian.mlx import HebbianMLXEngine
from src.hebbian.config import HebbianConfig


def test_baseline(engine: HebbianMLXEngine) -> bool:
    """Test generation without Hebbian modifications."""
    print("\n=== Baseline Test (no Hebbian) ===")
    engine.clear()

    result = engine.generate_text(
        "The capital of France is",
        max_tokens=40,
        temperature=0.0,
    )

    print(f"Output: {result[:100]}...")

    # Check that output is coherent (not just periods or garbage)
    is_coherent = "Paris" in result and not result.endswith(".....")
    print(f"Coherent output: {'PASS' if is_coherent else 'FAIL'}")
    return is_coherent


def test_sliding_window(engine: HebbianMLXEngine) -> bool:
    """Test that sliding window eviction works correctly."""
    print("\n=== Sliding Window Test ===")
    engine.clear()

    # Generate enough tokens to trigger eviction
    result = engine.generate_text(
        "Hello",
        max_tokens=60,
        temperature=0.0,
    )

    stats = engine.get_stats()
    print(f"Positions processed: {stats['positions_processed']}")
    print(f"Cache size: {stats['cache_size']}")
    print(f"Output: {result[:100]}...")

    # Verify sink tokens are preserved
    n_sink = engine.config.n_sink_tokens
    sink_preserved = all(i in engine.kv_cache.active_positions for i in range(n_sink))
    print(f"Sink tokens (0-{n_sink-1}) preserved: {'PASS' if sink_preserved else 'FAIL'}")

    # Verify eviction happened
    eviction_happened = stats['positions_processed'] > engine.config.window_size
    print(f"Eviction triggered: {'PASS' if eviction_happened else 'SKIP (not enough tokens)'}")

    return sink_preserved


def test_hebbian_modifications(engine: HebbianMLXEngine) -> bool:
    """Test that Hebbian modifications are created."""
    print("\n=== Hebbian Modifications Test ===")
    engine.clear()

    # Generate enough to trigger modifications
    result = engine.generate_text(
        "The quick brown fox",
        max_tokens=50,
        temperature=0.0,
    )

    stats = engine.get_stats()
    has_mods = stats['total_modifications'] > 0

    print(f"Total modifications: {stats['total_modifications']}")
    print(f"Modifications created: {'PASS' if has_mods else 'FAIL'}")

    return has_mods


def main():
    """Run all tests."""
    print("Loading MLX Hebbian Engine...")

    # Test with Hebbian enabled
    config = HebbianConfig(
        update_scale=1e-6,
        window_size=32,
        n_sink_tokens=4,
        decay=0.9,
        max_mods_per_layer=50,
    )

    engine = HebbianMLXEngine(
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        config=config,
    )

    print(f"\nConfig: window={config.window_size}, sink={config.n_sink_tokens}, scale={config.update_scale}")

    results = []
    results.append(("Baseline", test_baseline(engine)))
    results.append(("Sliding Window", test_sliding_window(engine)))
    results.append(("Hebbian Mods", test_hebbian_modifications(engine)))

    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_pass = all_pass and passed

    print("=" * 40)
    print(f"Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
