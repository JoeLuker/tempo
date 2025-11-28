#!/usr/bin/env python3
"""Test suite for validating experiment runner layers.

This ensures that each optimization layer (model persistence, multi-process,
batched KV cache) produces IDENTICAL results to the baseline implementation.

Test methodology:
1. Run same prompt/seed with baseline
2. Run same prompt/seed with optimized layer
3. Compare ALL outputs bit-by-bit:
   - Generated token IDs (exact match)
   - Attention matrices (within floating point tolerance)
   - Parallel sets structure (exact match)
   - Timing (should be faster, not exact)
"""

import pytest
import numpy as np
import json
import torch
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentValidator:
    """Validates that two experiment results are identical."""

    @staticmethod
    def compare_token_ids(result1: Dict, result2: Dict) -> bool:
        """Compare generated token IDs (must be EXACT)."""
        tokens1 = result1.get("generated_token_ids", [])
        tokens2 = result2.get("generated_token_ids", [])

        if tokens1 != tokens2:
            logger.error(f"Token ID mismatch!")
            logger.error(f"  Result 1: {tokens1}")
            logger.error(f"  Result 2: {tokens2}")
            return False

        logger.info(f"✓ Token IDs match: {len(tokens1)} tokens")
        return True

    @staticmethod
    def compare_attention_matrices(
        attn1: np.ndarray,
        attn2: np.ndarray,
        rtol: float = 1e-5,
        atol: float = 1e-8
    ) -> bool:
        """Compare attention matrices (within tolerance for floating point)."""
        if attn1.shape != attn2.shape:
            logger.error(f"Attention shape mismatch: {attn1.shape} vs {attn2.shape}")
            return False

        if not np.allclose(attn1, attn2, rtol=rtol, atol=atol):
            diff = np.abs(attn1 - attn2)
            max_diff = diff.max()
            mean_diff = diff.mean()

            logger.error(f"Attention matrix mismatch!")
            logger.error(f"  Max difference: {max_diff}")
            logger.error(f"  Mean difference: {mean_diff}")
            logger.error(f"  Tolerance: rtol={rtol}, atol={atol}")

            # Show where differences occur
            large_diffs = np.where(diff > atol)
            if len(large_diffs[0]) > 0:
                logger.error(f"  Large differences at {len(large_diffs[0])} locations")
                # Show first 5
                for i in range(min(5, len(large_diffs[0]))):
                    idx = tuple(d[i] for d in large_diffs)
                    logger.error(f"    {idx}: {attn1[idx]} vs {attn2[idx]} (diff={diff[idx]})")

            return False

        logger.info(f"✓ Attention matrices match: shape {attn1.shape}")
        return True

    @staticmethod
    def compare_parallel_sets(sets1: Dict, sets2: Dict) -> bool:
        """Compare parallel sets structure (must be EXACT)."""
        if sets1.keys() != sets2.keys():
            logger.error(f"Parallel sets keys mismatch: {sets1.keys()} vs {sets2.keys()}")
            return False

        for key in sets1.keys():
            if sets1[key] != sets2[key]:
                logger.error(f"Parallel sets mismatch at key '{key}':")
                logger.error(f"  Result 1: {sets1[key]}")
                logger.error(f"  Result 2: {sets2[key]}")
                return False

        logger.info(f"✓ Parallel sets match: {len(sets1)} entries")
        return True

    @staticmethod
    def compare_full_results(result1: Dict, result2: Dict, check_timing: bool = False) -> bool:
        """Compare all aspects of two experiment results."""
        logger.info("Comparing experiment results...")

        # 1. Token IDs (exact)
        if not ExperimentValidator.compare_token_ids(result1, result2):
            return False

        # 2. Attention matrices (within tolerance)
        if "attention_data" in result1 and "attention_data" in result2:
            attn1_data = result1["attention_data"]
            attn2_data = result2["attention_data"]

            # Compare keys
            if set(attn1_data.keys()) != set(attn2_data.keys()):
                logger.error(f"Attention data keys mismatch")
                return False

            # Compare each attention matrix
            for key in attn1_data.keys():
                if 'attention' in key:
                    attn1 = attn1_data[key]
                    attn2 = attn2_data[key]

                    if isinstance(attn1, np.ndarray) and isinstance(attn2, np.ndarray):
                        if not ExperimentValidator.compare_attention_matrices(attn1, attn2):
                            logger.error(f"  Failed at key: {key}")
                            return False

        # 3. Parallel sets (exact)
        if "parallel_sets" in result1 and "parallel_sets" in result2:
            if not ExperimentValidator.compare_parallel_sets(
                result1["parallel_sets"],
                result2["parallel_sets"]
            ):
                return False

        # 4. Generated text (should match if tokens match)
        if result1.get("generated_text") != result2.get("generated_text"):
            logger.warning("Generated text differs (may be formatting only)")
            logger.warning(f"  Result 1: {result1.get('generated_text')[:100]}")
            logger.warning(f"  Result 2: {result2.get('generated_text')[:100]}")

        # 5. Timing (optional check - should be faster, not identical)
        if check_timing:
            time1 = result1.get("generation_time", 0)
            time2 = result2.get("generation_time", 0)
            speedup = time1 / time2 if time2 > 0 else 0

            logger.info(f"Timing comparison:")
            logger.info(f"  Baseline: {time1:.2f}s")
            logger.info(f"  Optimized: {time2:.2f}s")
            logger.info(f"  Speedup: {speedup:.2f}x")

        logger.info("✓ All comparisons passed!")
        return True


def run_baseline_experiment(prompt: str, seed: int, mode: str) -> Dict:
    """Run experiment with baseline (fresh model load each time)."""
    logger.info(f"Running BASELINE: prompt='{prompt[:30]}...', seed={seed}, mode={mode}")

    # Import here to avoid loading at module level
    from experiments.baseline_runner import run_baseline_experiment as run_baseline

    result = run_baseline(
        prompt=prompt,
        seed=seed,
        isolated=(mode == "isolated"),
        max_tokens=10
    )

    return result


def run_layer1_experiment(prompt: str, seed: int, mode: str, runner) -> Dict:
    """Run experiment with Layer 1 (model persistence)."""
    logger.info(f"Running LAYER 1: prompt='{prompt[:30]}...', seed={seed}, mode={mode}")

    result = runner.run_experiment(
        prompt=prompt,
        seed=seed,
        isolated=(mode == "isolated"),
        max_tokens=10
    )

    return result


def run_layer2_experiment(prompt: str, seed: int, mode: str, suite) -> Dict:
    """Run experiment with Layer 2 (multi-process batching)."""
    logger.info(f"Running LAYER 2: prompt='{prompt[:30]}...', seed={seed}, mode={mode}")

    # Run as single-item batch
    results = suite.run_batch(
        prompts=[prompt],
        seeds=[seed],
        modes=[mode]
    )

    return results[0]


def run_layer3_experiment(prompt: str, seed: int, mode: str, batched_runner) -> Dict:
    """Run experiment with Layer 3 (batched KV cache)."""
    logger.info(f"Running LAYER 3: prompt='{prompt[:30]}...', seed={seed}, mode={mode}")

    # Run as single-item batch
    results = batched_runner.run_batch(
        prompts=[prompt],
        seeds=[seed],
        mode=mode
    )

    return results[0]


# ============================================================================
# PYTEST TEST CASES
# ============================================================================

class TestExperimentLayers:
    """Test suite for experiment runner layers."""

    @pytest.fixture
    def test_prompts(self):
        """Small set of test prompts."""
        return [
            "The cat sat on the",
            "Once upon a time",
            "The scientist discovered"
        ]

    @pytest.fixture
    def test_seeds(self):
        """Test seeds."""
        return [42, 123]

    @pytest.fixture
    def test_modes(self):
        """Test modes."""
        return ["isolated", "visible"]

    def test_layer1_vs_baseline_single(self, test_prompts):
        """Test Layer 1 produces identical results to baseline for single experiment."""
        from experiments.simple_persistent import SimplePersistentRunner

        prompt = test_prompts[0]
        seed = 42
        mode = "isolated"

        # Run baseline
        baseline_result = run_baseline_experiment(prompt, seed, mode)

        # Run Layer 1
        runner = SimplePersistentRunner()
        layer1_result = run_layer1_experiment(prompt, seed, mode, runner)

        # Compare
        assert ExperimentValidator.compare_full_results(baseline_result, layer1_result)

    def test_layer1_multiple_sequential(self, test_prompts, test_seeds):
        """Test Layer 1 with multiple sequential experiments (model persistence)."""
        from experiments.simple_persistent import SimplePersistentRunner

        runner = SimplePersistentRunner()

        for prompt in test_prompts[:2]:  # Use 2 prompts
            for seed in test_seeds[:1]:  # Use 1 seed
                for mode in ["isolated"]:  # Use 1 mode
                    # Run baseline
                    baseline_result = run_baseline_experiment(prompt, seed, mode)

                    # Run Layer 1
                    layer1_result = run_layer1_experiment(prompt, seed, mode, runner)

                    # Compare
                    assert ExperimentValidator.compare_full_results(baseline_result, layer1_result), \
                        f"Mismatch for prompt='{prompt}', seed={seed}, mode={mode}"

    def test_layer1_memory_stability(self, test_prompts):
        """Test Layer 1 doesn't leak memory across experiments."""
        import gc
        import psutil
        import os

        from experiments.simple_persistent import SimplePersistentRunner

        runner = SimplePersistentRunner()
        process = psutil.Process(os.getpid())

        memory_samples = []

        # Run 5 experiments and track memory
        for i in range(5):
            prompt = test_prompts[i % len(test_prompts)]
            runner.run_experiment(prompt, seed=42, mode="isolated")

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Measure memory
            mem_mb = process.memory_info().rss / 1024 / 1024
            memory_samples.append(mem_mb)

            logger.info(f"After experiment {i+1}: {mem_mb:.1f} MB")

        # Check memory didn't grow significantly
        initial_mem = memory_samples[0]
        final_mem = memory_samples[-1]
        growth = final_mem - initial_mem
        growth_percent = (growth / initial_mem) * 100

        logger.info(f"Memory growth: {growth:.1f} MB ({growth_percent:.1f}%)")

        # Allow up to 20% growth (some variance is expected)
        assert growth_percent < 20, f"Memory grew by {growth_percent:.1f}% (too much!)"

    def test_layer2_vs_layer1(self, test_prompts, test_seeds):
        """Test Layer 2 produces identical results to Layer 1."""
        from experiments.simple_persistent import SimplePersistentRunner
        from experiments.parallel_suite import ParallelExperimentSuite

        runner = SimplePersistentRunner()
        suite = ParallelExperimentSuite(num_workers=2)

        prompt = test_prompts[0]
        seed = 42
        mode = "isolated"

        # Run Layer 1
        layer1_result = run_layer1_experiment(prompt, seed, mode, runner)

        # Run Layer 2
        layer2_result = run_layer2_experiment(prompt, seed, mode, suite)

        # Compare
        assert ExperimentValidator.compare_full_results(layer1_result, layer2_result)

    def test_layer3_vs_layer2(self, test_prompts, test_seeds):
        """Test Layer 3 produces identical results to Layer 2."""
        from experiments.parallel_suite import ParallelExperimentSuite
        from experiments.batched_runner import BatchedExperimentRunner

        suite = ParallelExperimentSuite(num_workers=2)
        batched_runner = BatchedExperimentRunner(batch_size=5)

        prompt = test_prompts[0]
        seed = 42
        mode = "isolated"

        # Run Layer 2
        layer2_result = run_layer2_experiment(prompt, seed, mode, suite)

        # Run Layer 3
        layer3_result = run_layer3_experiment(prompt, seed, mode, batched_runner)

        # Compare
        assert ExperimentValidator.compare_full_results(layer2_result, layer3_result)

    def test_determinism_with_seed(self, test_prompts):
        """Test that same seed produces identical results."""
        from experiments.simple_persistent import SimplePersistentRunner

        runner = SimplePersistentRunner()
        prompt = test_prompts[0]
        seed = 42
        mode = "isolated"

        # Run twice with same seed
        result1 = run_layer1_experiment(prompt, seed, mode, runner)
        result2 = run_layer1_experiment(prompt, seed, mode, runner)

        # Should be IDENTICAL
        assert ExperimentValidator.compare_full_results(result1, result2)

    def test_different_seeds_produce_different_results(self, test_prompts):
        """Test that different seeds produce different results."""
        from experiments.simple_persistent import SimplePersistentRunner

        runner = SimplePersistentRunner()
        prompt = test_prompts[0]
        mode = "isolated"

        # Run with different seeds
        result1 = run_layer1_experiment(prompt, seed=42, mode=mode, runner=runner)
        result2 = run_layer1_experiment(prompt, seed=123, mode=mode, runner=runner)

        # Token IDs should differ (with high probability)
        tokens1 = result1.get("generated_token_ids", [])
        tokens2 = result2.get("generated_token_ids", [])

        assert tokens1 != tokens2, "Different seeds should produce different outputs"


# ============================================================================
# STANDALONE VALIDATION SCRIPT
# ============================================================================

def main():
    """Run validation tests manually (without pytest)."""
    print("="*80)
    print("TEMPO EXPERIMENT LAYER VALIDATION")
    print("="*80)

    validator = ExperimentValidator()

    # Test 1: Single experiment baseline vs Layer 1
    print("\n" + "="*80)
    print("TEST 1: Baseline vs Layer 1 (single experiment)")
    print("="*80)

    prompt = "The cat sat on the"
    seed = 42
    mode = "isolated"

    print(f"\nPrompt: '{prompt}'")
    print(f"Seed: {seed}")
    print(f"Mode: {mode}")

    try:
        baseline_result = run_baseline_experiment(prompt, seed, mode)
        print("✓ Baseline completed")

        from experiments.simple_persistent import SimplePersistentRunner
        runner = SimplePersistentRunner()
        layer1_result = run_layer1_experiment(prompt, seed, mode, runner)
        print("✓ Layer 1 completed")

        if validator.compare_full_results(baseline_result, layer1_result, check_timing=True):
            print("\n✓✓✓ TEST 1 PASSED: Results are identical!")
        else:
            print("\n✗✗✗ TEST 1 FAILED: Results differ!")
            return False

    except Exception as e:
        print(f"\n✗✗✗ TEST 1 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Multiple sequential experiments
    print("\n" + "="*80)
    print("TEST 2: Multiple sequential experiments with Layer 1")
    print("="*80)

    prompts = ["The cat sat on the", "Once upon a time"]
    from experiments.simple_persistent import SimplePersistentRunner
    runner = SimplePersistentRunner()

    all_passed = True
    for i, prompt in enumerate(prompts):
        print(f"\nExperiment {i+1}/{len(prompts)}: '{prompt}'")

        try:
            baseline_result = run_baseline_experiment(prompt, 42, "isolated")
            layer1_result = run_layer1_experiment(prompt, 42, "isolated", runner)

            if validator.compare_full_results(baseline_result, layer1_result):
                print(f"  ✓ Experiment {i+1} passed")
            else:
                print(f"  ✗ Experiment {i+1} failed")
                all_passed = False

        except Exception as e:
            print(f"  ✗ Experiment {i+1} error: {e}")
            all_passed = False

    if all_passed:
        print("\n✓✓✓ TEST 2 PASSED: All experiments match!")
    else:
        print("\n✗✗✗ TEST 2 FAILED: Some experiments differed!")
        return False

    print("\n" + "="*80)
    print("ALL VALIDATION TESTS PASSED!")
    print("="*80)
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
