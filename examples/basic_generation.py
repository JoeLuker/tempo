#!/usr/bin/env python3
"""
TEMPO Basic Generation Examples
Simple examples to get started with TEMPO
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_tempo import main as run_tempo
import argparse


def example_simple_generation():
    """Basic text generation with default settings"""
    print("\n" + "="*60)
    print("Example 1: Simple Generation")
    print("="*60)
    
    args = [
        "--prompt", "The future of artificial intelligence is",
        "--selection-threshold", "0.1",
        "--max-tokens", "50"
    ]
    
    print(f"Running: python run_tempo.py {' '.join(args)}")
    print("-"*60)
    
    # Run TEMPO
    run_tempo(args)
    

def example_creative_writing():
    """Creative writing with higher threshold for more branching"""
    print("\n" + "="*60)
    print("Example 2: Creative Writing with More Branching")
    print("="*60)
    
    args = [
        "--prompt", "In a world where time flows backwards,",
        "--selection-threshold", "0.2",
        "--max-tokens", "100",
        "--temperature", "0.9"
    ]
    
    print(f"Running: python run_tempo.py {' '.join(args)}")
    print("-"*60)
    
    run_tempo(args)
    

def example_with_pruning():
    """Generation with retroactive pruning enabled"""
    print("\n" + "="*60)
    print("Example 3: Generation with Retroactive Pruning")
    print("="*60)
    
    args = [
        "--prompt", "The key to understanding quantum mechanics is",
        "--selection-threshold", "0.15",
        "--use-retroactive-pruning",
        "--attention-threshold", "0.01",
        "--max-tokens", "75"
    ]
    
    print(f"Running: python run_tempo.py {' '.join(args)}")
    print("-"*60)
    
    run_tempo(args)
    

def example_dynamic_threshold():
    """Using dynamic thresholding with Bezier curves"""
    print("\n" + "="*60)
    print("Example 4: Dynamic Thresholding")
    print("="*60)
    
    args = [
        "--prompt", "Once upon a time in a distant galaxy,",
        "--selection-threshold", "0.1",
        "--use-retroactive-pruning",
        "--attention-threshold", "0.005",
        "--dynamic-threshold",
        "--bezier-p1", "0.1",
        "--bezier-p2", "0.9",
        "--max-tokens", "100"
    ]
    
    print(f"Running: python run_tempo.py {' '.join(args)}")
    print("-"*60)
    
    run_tempo(args)
    

def example_thinking_mode():
    """Cogito thinking mode for reasoning tasks"""
    print("\n" + "="*60)
    print("Example 5: Cogito Thinking Mode")
    print("="*60)
    
    args = [
        "--prompt", "Explain step by step how photosynthesis works:",
        "--enable-thinking",
        "--selection-threshold", "0.12",
        "--max-tokens", "150"
    ]
    
    print(f"Running: python run_tempo.py {' '.join(args)}")
    print("-"*60)
    
    run_tempo(args)


def example_low_threshold():
    """Low threshold for minimal branching"""
    print("\n" + "="*60)
    print("Example 6: Low Threshold (Minimal Branching)")
    print("="*60)
    
    args = [
        "--prompt", "The capital of France is",
        "--selection-threshold", "0.02",
        "--max-tokens", "20"
    ]
    
    print(f"Running: python run_tempo.py {' '.join(args)}")
    print("-"*60)
    
    run_tempo(args)


def main():
    parser = argparse.ArgumentParser(
        description="TEMPO Basic Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/basic_generation.py --all        # Run all examples
  python examples/basic_generation.py --example 1  # Run specific example
  python examples/basic_generation.py --list       # List all examples
        """
    )
    
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="Run specific example number"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all examples"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available examples"
    )
    
    args = parser.parse_args()
    
    examples = {
        1: ("Simple Generation", example_simple_generation),
        2: ("Creative Writing", example_creative_writing),
        3: ("With Pruning", example_with_pruning),
        4: ("Dynamic Threshold", example_dynamic_threshold),
        5: ("Thinking Mode", example_thinking_mode),
        6: ("Low Threshold", example_low_threshold)
    }
    
    if args.list:
        print("\nAvailable Examples:")
        print("-" * 40)
        for num, (name, _) in examples.items():
            print(f"{num}. {name}")
        return
        
    if args.all:
        print("\nRunning all TEMPO examples...")
        print("This will demonstrate different generation modes.\n")
        
        for num, (name, func) in examples.items():
            try:
                func()
                input("\nPress Enter to continue to next example...")
            except KeyboardInterrupt:
                print("\n\nExamples interrupted by user.")
                break
                
    elif args.example:
        name, func = examples[args.example]
        print(f"\nRunning Example {args.example}: {name}")
        func()
        
    else:
        # Default: show usage
        parser.print_help()
        print("\n💡 Tip: Start with --example 1 for the simplest case")


if __name__ == "__main__":
    main()