#!/usr/bin/env python3
"""
Basic TEMPO generation example.

This script demonstrates the simplest way to use TEMPO for text generation
with parallel token exploration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.experiments.argument_parser import create_argument_parser
from src.experiments.experiment_runner import ExperimentRunner


def main():
    """Run basic generation examples."""
    
    # Example 1: Simple generation with default settings
    print("=" * 50)
    print("Example 1: Basic generation")
    print("=" * 50)
    
    args = [
        "--prompt", "The future of artificial intelligence is",
        "--max-tokens", "50",
        "--selection-threshold", "0.1"
    ]
    
    parser = create_argument_parser()
    parsed_args = parser.parse_args(args)
    
    runner = ExperimentRunner(parsed_args)
    result = runner.run()
    
    print(f"\nGenerated text: {result['generated_text']}")
    print(f"Clean text: {result['clean_text']}")
    print(f"Token count: {result['total_tokens']}")
    
    # Example 2: Generation with higher threshold for more exploration
    print("\n" + "=" * 50)
    print("Example 2: Higher threshold for more parallel tokens")
    print("=" * 50)
    
    args = [
        "--prompt", "Once upon a time in a",
        "--max-tokens", "30",
        "--selection-threshold", "0.15"
    ]
    
    parsed_args = parser.parse_args(args)
    runner = ExperimentRunner(parsed_args)
    result = runner.run()
    
    print(f"\nGenerated text: {result['generated_text']}")
    print(f"Parallel positions: {result.get('parallel_positions', 'N/A')}")
    
    # Example 3: Technical writing with lower threshold
    print("\n" + "=" * 50)
    print("Example 3: Technical writing with focused generation")
    print("=" * 50)
    
    args = [
        "--prompt", "To implement a binary search algorithm in Python, you should",
        "--max-tokens", "100",
        "--selection-threshold", "0.05",
        "--temperature", "0.7"
    ]
    
    parsed_args = parser.parse_args(args)
    runner = ExperimentRunner(parsed_args)
    result = runner.run()
    
    print(f"\nGenerated text: {result['clean_text']}")
    
    # Example 4: Creative writing with dynamic settings
    print("\n" + "=" * 50)
    print("Example 4: Creative story generation")
    print("=" * 50)
    
    args = [
        "--prompt", "The robot discovered an ancient artifact that",
        "--max-tokens", "150",
        "--selection-threshold", "0.12",
        "--temperature", "0.9",
        "--top-p", "0.95"
    ]
    
    parsed_args = parser.parse_args(args)
    runner = ExperimentRunner(parsed_args)
    result = runner.run()
    
    print(f"\nGenerated text: {result['generated_text']}")
    print(f"\nGeneration time: {result.get('generation_time', 'N/A')} seconds")


if __name__ == "__main__":
    print("TEMPO Basic Generation Examples")
    print("==============================\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()