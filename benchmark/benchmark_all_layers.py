#!/usr/bin/env python3
"""Benchmark all three optimization layers with real timing data.

This script runs the same set of experiments through all three layers
and measures actual wall-clock time to compare performance.
"""

import time
import json
from pathlib import Path
from typing import List, Dict
import logging

from experiments.baseline_runner import run_baseline_experiment
from experiments.simple_persistent import SimplePersistentRunner
from experiments.parallel_suite import ParallelExperimentSuite
from experiments.batched_runner import BatchedExperimentRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_benchmark(num_experiments: int = 10, num_workers: int = 4):
    """Run comprehensive benchmark of all layers.

    Args:
        num_experiments: Number of experiments to run in each layer
        num_workers: Number of workers for Layer 2
    """
    # Test data
    prompts = [
        "The cat sat on the",
        "Once upon a time",
        "The scientist discovered",
        "In the beginning",
        "The ancient city",
        "Far across the sea",
        "Deep in the forest",
        "High above the clouds",
        "Beneath the surface",
        "Beyond the horizon"
    ]

    # Cycle through prompts to get num_experiments
    test_prompts = [prompts[i % len(prompts)] for i in range(num_experiments)]
    test_seeds = [42 + i for i in range(num_experiments)]
    test_mode = "isolated"

    results = {
        "config": {
            "num_experiments": num_experiments,
            "num_workers": num_workers,
            "prompts": test_prompts,
            "seeds": test_seeds,
            "mode": test_mode
        },
        "baseline": {},
        "layer1": {},
        "layer2": {},
        "layer3": {}
    }

    print("\n" + "="*80)
    print(f"TEMPO OPTIMIZATION LAYERS BENCHMARK")
    print("="*80)
    print(f"Experiments: {num_experiments}")
    print(f"Workers (Layer 2): {num_workers}")
    print("="*80)

    # ========================================================================
    # BASELINE: Fresh model load per experiment
    # ========================================================================
    print("\n" + "="*80)
    print("BASELINE: Fresh model load per experiment")
    print("="*80)

    baseline_start = time.time()
    baseline_results = []

    for i in range(num_experiments):
        print(f"Baseline {i+1}/{num_experiments}...", end=" ", flush=True)
        exp_start = time.time()

        result = run_baseline_experiment(
            prompt=test_prompts[i],
            seed=test_seeds[i],
            isolated=True,
            max_tokens=10
        )

        exp_time = time.time() - exp_start
        print(f"{exp_time:.2f}s")
        baseline_results.append(result)

    baseline_total = time.time() - baseline_start
    baseline_per_exp = baseline_total / num_experiments

    results["baseline"] = {
        "total_time": baseline_total,
        "per_experiment": baseline_per_exp,
        "throughput": num_experiments / baseline_total
    }

    print(f"\n✓ Baseline completed:")
    print(f"  Total: {baseline_total:.2f}s ({baseline_total/60:.2f} min)")
    print(f"  Per experiment: {baseline_per_exp:.2f}s")
    print(f"  Throughput: {results['baseline']['throughput']:.2f} exp/s")

    # ========================================================================
    # LAYER 1: Model Persistence
    # ========================================================================
    print("\n" + "="*80)
    print("LAYER 1: Model Persistence")
    print("="*80)

    layer1_start = time.time()

    print("Loading model (one-time)...", end=" ", flush=True)
    runner = SimplePersistentRunner()
    load_time = time.time() - layer1_start
    print(f"{load_time:.2f}s")

    layer1_results = []
    gen_start = time.time()

    for i in range(num_experiments):
        print(f"Layer 1 {i+1}/{num_experiments}...", end=" ", flush=True)
        exp_start = time.time()

        result = runner.run_experiment(
            prompt=test_prompts[i],
            seed=test_seeds[i],
            isolated=True,
            max_tokens=10
        )

        exp_time = time.time() - exp_start
        print(f"{exp_time:.2f}s")
        layer1_results.append(result)

    gen_time = time.time() - gen_start
    layer1_total = time.time() - layer1_start
    layer1_per_exp = gen_time / num_experiments

    results["layer1"] = {
        "total_time": layer1_total,
        "load_time": load_time,
        "generation_time": gen_time,
        "per_experiment": layer1_per_exp,
        "throughput": num_experiments / layer1_total,
        "speedup_vs_baseline": baseline_total / layer1_total
    }

    print(f"\n✓ Layer 1 completed:")
    print(f"  Total: {layer1_total:.2f}s ({layer1_total/60:.2f} min)")
    print(f"  Load time: {load_time:.2f}s (one-time)")
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Per experiment: {layer1_per_exp:.2f}s")
    print(f"  Throughput: {results['layer1']['throughput']:.2f} exp/s")
    print(f"  Speedup: {results['layer1']['speedup_vs_baseline']:.2f}x vs baseline")

    # ========================================================================
    # LAYER 2: Multi-Process Batching
    # ========================================================================
    print("\n" + "="*80)
    print(f"LAYER 2: Multi-Process Batching ({num_workers} workers)")
    print("="*80)

    layer2_start = time.time()

    suite = ParallelExperimentSuite(num_workers=num_workers)

    layer2_results = suite.run_batch(
        prompts=test_prompts,
        seeds=test_seeds,
        modes=[test_mode] * num_experiments
    )

    layer2_total = time.time() - layer2_start
    layer2_per_exp = layer2_total / num_experiments

    results["layer2"] = {
        "total_time": layer2_total,
        "per_experiment": layer2_per_exp,
        "throughput": num_experiments / layer2_total,
        "speedup_vs_baseline": baseline_total / layer2_total,
        "speedup_vs_layer1": layer1_total / layer2_total
    }

    print(f"\n✓ Layer 2 completed:")
    print(f"  Total: {layer2_total:.2f}s ({layer2_total/60:.2f} min)")
    print(f"  Per experiment: {layer2_per_exp:.2f}s")
    print(f"  Throughput: {results['layer2']['throughput']:.2f} exp/s")
    print(f"  Speedup: {results['layer2']['speedup_vs_baseline']:.2f}x vs baseline")
    print(f"  Speedup: {results['layer2']['speedup_vs_layer1']:.2f}x vs Layer 1")

    # ========================================================================
    # LAYER 3: Batched Processing
    # ========================================================================
    print("\n" + "="*80)
    print("LAYER 3: Batched Processing")
    print("="*80)

    layer3_start = time.time()

    batched_runner = BatchedExperimentRunner(batch_size=4)

    layer3_results = batched_runner.run_batch(
        prompts=test_prompts,
        seeds=test_seeds,
        mode=test_mode
    )

    layer3_total = time.time() - layer3_start
    layer3_per_exp = layer3_total / num_experiments

    results["layer3"] = {
        "total_time": layer3_total,
        "per_experiment": layer3_per_exp,
        "throughput": num_experiments / layer3_total,
        "speedup_vs_baseline": baseline_total / layer3_total,
        "speedup_vs_layer1": layer1_total / layer3_total
    }

    print(f"\n✓ Layer 3 completed:")
    print(f"  Total: {layer3_total:.2f}s ({layer3_total/60:.2f} min)")
    print(f"  Per experiment: {layer3_per_exp:.2f}s")
    print(f"  Throughput: {results['layer3']['throughput']:.2f} exp/s")
    print(f"  Speedup: {results['layer3']['speedup_vs_baseline']:.2f}x vs baseline")
    print(f"  Speedup: {results['layer3']['speedup_vs_layer1']:.2f}x vs Layer 1")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"\n{num_experiments} experiments:")
    print(f"  Baseline:  {baseline_total:.2f}s ({baseline_total/60:.2f} min) - 1.00x")
    print(f"  Layer 1:   {layer1_total:.2f}s ({layer1_total/60:.2f} min) - {results['layer1']['speedup_vs_baseline']:.2f}x faster")
    print(f"  Layer 2:   {layer2_total:.2f}s ({layer2_total/60:.2f} min) - {results['layer2']['speedup_vs_baseline']:.2f}x faster")
    print(f"  Layer 3:   {layer3_total:.2f}s ({layer3_total/60:.2f} min) - {results['layer3']['speedup_vs_baseline']:.2f}x faster")

    print(f"\nBest performer: ", end="")
    times = {
        "Layer 1": layer1_total,
        "Layer 2": layer2_total,
        "Layer 3": layer3_total
    }
    best = min(times, key=times.get)
    print(f"{best} ({times[best]:.2f}s)")

    # Save results
    output_file = Path("benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("="*80)

    return results


if __name__ == "__main__":
    import sys

    # Default: 10 experiments with 4 workers
    num_experiments = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    run_benchmark(num_experiments, num_workers)
