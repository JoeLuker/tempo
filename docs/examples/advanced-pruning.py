#!/usr/bin/env python3
"""
Advanced pruning examples for TEMPO.

This script demonstrates various pruning strategies including retroactive
pruning, dynamic thresholds, and attention-based refinement.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.experiments.argument_parser import create_argument_parser
from src.experiments.experiment_runner import ExperimentRunner


def run_example(name: str, args: list):
    """Run a single example with given arguments."""
    print("=" * 60)
    print(f"Example: {name}")
    print("=" * 60)
    
    parser = create_argument_parser()
    parsed_args = parser.parse_args(args)
    
    runner = ExperimentRunner(parsed_args)
    result = runner.run()
    
    print(f"\nPrompt: {parsed_args.prompt}")
    print(f"Generated text: {result['generated_text']}")
    print(f"Clean text: {result['clean_text']}")
    
    # Show pruning statistics if available
    if 'pruning_stats' in result:
        stats = result['pruning_stats']
        print(f"\nPruning Statistics:")
        print(f"  - Total candidates: {stats.get('total_candidates', 'N/A')}")
        print(f"  - Pruned tokens: {stats.get('pruned_tokens', 'N/A')}")
        print(f"  - Pruning rate: {stats.get('pruning_rate', 'N/A')}%")
    
    print(f"\nGeneration time: {result.get('generation_time', 'N/A')} seconds")
    return result


def main():
    """Run advanced pruning examples."""
    
    # Example 1: Basic retroactive pruning
    run_example(
        "Basic Retroactive Pruning",
        [
            "--prompt", "The scientific method involves",
            "--max-tokens", "75",
            "--selection-threshold", "0.1",
            "--use-retroactive-pruning",
            "--attention-threshold", "0.02"
        ]
    )
    
    # Example 2: Retroactive pruning with dynamic threshold
    print("\n")
    run_example(
        "Dynamic Threshold with Bezier Curve",
        [
            "--prompt", "In the depths of the ocean, researchers discovered",
            "--max-tokens", "100",
            "--selection-threshold", "0.12",
            "--use-retroactive-pruning",
            "--attention-threshold", "0.01",
            "--dynamic-threshold",
            "--bezier-p1", "0.1",
            "--bezier-p2", "0.9",
            "--final-threshold", "0.05"
        ]
    )
    
    # Example 3: ReLU-based dynamic threshold
    print("\n")
    run_example(
        "ReLU Dynamic Threshold",
        [
            "--prompt", "The algorithm's time complexity analysis shows",
            "--max-tokens", "80",
            "--selection-threshold", "0.08",
            "--use-retroactive-pruning",
            "--attention-threshold", "0.015",
            "--dynamic-threshold",
            "--use-relu",
            "--relu-activation-point", "0.4"
        ]
    )
    
    # Example 4: Multi-scale attention with relative thresholds
    print("\n")
    run_example(
        "Multi-Scale Attention Integration",
        [
            "--prompt", "The Renaissance period was characterized by",
            "--max-tokens", "120",
            "--selection-threshold", "0.1",
            "--use-retroactive-pruning",
            "--attention-threshold", "0.01",
            "--use-relative-attention",
            "--relative-threshold", "0.3",
            "--use-multi-scale-attention",
            "--num-layers-to-use", "8"
        ]
    )
    
    # Example 5: Sigmoid decision boundary
    print("\n")
    run_example(
        "Sigmoid-Based Pruning",
        [
            "--prompt", "Quantum computing differs from classical computing in",
            "--max-tokens", "100",
            "--selection-threshold", "0.09",
            "--use-retroactive-pruning",
            "--attention-threshold", "0.008",
            "--use-sigmoid-threshold",
            "--sigmoid-steepness", "15.0"
        ]
    )
    
    # Example 6: Complete pruning with position removal
    print("\n")
    run_example(
        "Aggressive Pruning with Position Removal",
        [
            "--prompt", "The three laws of robotics state that",
            "--max-tokens", "80",
            "--selection-threshold", "0.15",
            "--use-retroactive-pruning",
            "--attention-threshold", "0.02",
            "--complete-pruning-mode", "remove_position",
            "--no-preserve-isolated-tokens"
        ]
    )
    
    # Example 7: Combined strategies for creative writing
    print("\n")
    run_example(
        "Combined Strategies for Creative Writing",
        [
            "--prompt", "In a world where time flows backwards,",
            "--max-tokens", "150",
            "--selection-threshold", "0.13",
            "--use-retroactive-pruning",
            "--attention-threshold", "0.012",
            "--dynamic-threshold",
            "--bezier-p1", "0.15",
            "--bezier-p2", "0.85",
            "--use-multi-scale-attention",
            "--use-relative-attention",
            "--relative-threshold", "0.4",
            "--temperature", "0.9"
        ]
    )


if __name__ == "__main__":
    print("TEMPO Advanced Pruning Examples")
    print("===============================")
    print("Demonstrating various pruning strategies and configurations\n")
    
    try:
        main()
        
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print("""
These examples demonstrate:
1. Basic retroactive pruning for coherence
2. Dynamic thresholds using Bezier curves
3. ReLU-based threshold transitions
4. Multi-scale attention integration
5. Sigmoid decision boundaries
6. Aggressive pruning with position removal
7. Combined strategies for creative outputs

Experiment with different threshold values and strategies to find
the best configuration for your use case!
        """)
        
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()