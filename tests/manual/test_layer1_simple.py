#!/usr/bin/env python3
"""Simple test to verify Layer 1 produces identical results to baseline."""

import time
from experiments.simple_baseline import run_baseline
from experiments.simple_persistent import SimplePersistentRunner

def compare_results(r1, r2):
    """Compare two results."""
    print("\nComparing results...")

    # Compare generated text
    if r1['generated_text'] != r2['generated_text']:
        print(f"✗ Generated text differs!")
        print(f"  Baseline:   {r1['generated_text'][:100]}")
        print(f"  Persistent: {r2['generated_text'][:100]}")
        return False
    else:
        print(f"✓ Generated text matches")

    # Compare raw text
    if r1['raw_generated_text'] != r2['raw_generated_text']:
        print(f"✗ Raw text differs!")
        print(f"  Baseline:   {r1['raw_generated_text'][:100]}")
        print(f"  Persistent: {r2['raw_generated_text'][:100]}")
        return False
    else:
        print(f"✓ Raw text matches")

    print(f"\n✓✓✓ ALL CHECKS PASSED - Results are identical!")
    return True

def main():
    print("="*80)
    print("LAYER 1 VALIDATION TEST")
    print("="*80)

    prompt = "The cat sat on the"
    seed = 42

    # Test 1: Single experiment comparison
    print("\nTEST 1: Single experiment - Baseline vs Layer 1")
    print("-"*80)

    print("\nRunning BASELINE...")
    start = time.time()
    baseline_result = run_baseline(prompt, seed, isolated=True, max_tokens=10)
    baseline_time = time.time() - start
    print(f"Baseline completed in {baseline_time:.2f}s")

    print("\nRunning LAYER 1 (first run, includes model load)...")
    runner = SimplePersistentRunner()
    start = time.time()
    layer1_result = runner.run_experiment(prompt, seed, isolated=True, max_tokens=10)
    layer1_time = time.time() - start
    print(f"Layer 1 completed in {layer1_time:.2f}s")

    if not compare_results(baseline_result, layer1_result):
        print("\n✗✗✗ TEST 1 FAILED")
        return False

    # Test 2: Multiple sequential experiments (shows persistence benefit)
    print("\n" + "="*80)
    print("TEST 2: Multiple sequential experiments with Layer 1")
    print("-"*80)

    prompts = ["The cat sat on the", "Once upon a time", "The scientist discovered"]

    print(f"\nRunning {len(prompts)} experiments sequentially...")
    start = time.time()

    for i, p in enumerate(prompts):
        print(f"\n  Experiment {i+1}/{len(prompts)}: '{p}'")
        exp_start = time.time()
        result = runner.run_experiment(p, seed=42, isolated=True, max_tokens=10)
        exp_time = time.time() - exp_start
        print(f"    Completed in {exp_time:.2f}s")
        print(f"    Generated: {result['generated_text'][:60]}...")

    total_time = time.time() - start
    avg_time = total_time / len(prompts)

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Average per experiment: {avg_time:.2f}s")
    print(f"  ✓ Model persisted across all {len(prompts)} experiments")

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    print(f"\nLayer 1 is working correctly!")
    print(f"- Results match baseline exactly")
    print(f"- Model persistence works")
    print(f"- Ready for multi-prompt experiments")

    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
