#!/usr/bin/env python3
"""
TEMPO Advanced Pruning Examples
Demonstrates various pruning strategies and configurations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_tempo import main as run_tempo
import argparse


def example_basic_retroactive():
    """Basic retroactive pruning with default settings"""
    print("\n" + "="*60)
    print("Example 1: Basic Retroactive Pruning")
    print("="*60)
    print("Removes tokens that receive low attention from future tokens")
    print("-"*60)
    
    args = [
        "--prompt", "The three most important inventions in human history are",
        "--selection-threshold", "0.15",
        "--use-retroactive-pruning",
        "--attention-threshold", "0.01",
        "--max-tokens", "100"
    ]
    
    print(f"\nCommand: python run_tempo.py {' '.join(args)}\n")
    run_tempo(args)


def example_relative_attention():
    """Using relative attention thresholds"""
    print("\n" + "="*60)
    print("Example 2: Relative Attention Thresholds")
    print("="*60)
    print("Prunes based on relative attention within each set")
    print("-"*60)
    
    args = [
        "--prompt", "The most effective way to learn a new language is",
        "--selection-threshold", "0.12",
        "--use-retroactive-pruning",
        "--attention-threshold", "0.005",
        "--use-relative-attention",
        "--relative-threshold", "0.3",
        "--max-tokens", "80"
    ]
    
    print(f"\nCommand: python run_tempo.py {' '.join(args)}\n")
    run_tempo(args)


def example_multi_scale_attention():
    """Multi-scale attention across transformer layers"""
    print("\n" + "="*60)
    print("Example 3: Multi-Scale Attention Integration")
    print("="*60)
    print("Combines attention from multiple transformer layers")
    print("-"*60)
    
    args = [
        "--prompt", "Artificial intelligence will transform society by",
        "--selection-threshold", "0.1",
        "--use-retroactive-pruning",
        "--attention-threshold", "0.008",
        "--use-multi-scale-attention",
        "--num-layers-to-use", "8",
        "--max-tokens", "100"
    ]
    
    print(f"\nCommand: python run_tempo.py {' '.join(args)}\n")
    run_tempo(args)


def example_sigmoid_boundaries():
    """Sigmoid-based decision boundaries"""
    print("\n" + "="*60)
    print("Example 4: Sigmoid Decision Boundaries")
    print("="*60)
    print("Sharp transitions for pruning decisions")
    print("-"*60)
    
    args = [
        "--prompt", "The recipe for happiness includes",
        "--selection-threshold", "0.18",
        "--use-retroactive-pruning",
        "--attention-threshold", "0.006",
        "--use-sigmoid-threshold",
        "--sigmoid-steepness", "15.0",
        "--max-tokens", "75"
    ]
    
    print(f"\nCommand: python run_tempo.py {' '.join(args)}\n")
    run_tempo(args)


def example_dynamic_bezier():
    """Dynamic thresholding with Bezier curves"""
    print("\n" + "="*60)
    print("Example 5: Dynamic Bezier Thresholding")
    print("="*60)
    print("Smooth threshold increase over generation steps")
    print("-"*60)
    
    args = [
        "--prompt", "In the year 2050, everyday life will",
        "--selection-threshold", "0.1",
        "--use-retroactive-pruning",
        "--attention-threshold", "0.005",
        "--dynamic-threshold",
        "--final-threshold", "0.05",
        "--bezier-p1", "0.2",
        "--bezier-p2", "0.8",
        "--max-tokens", "120"
    ]
    
    print(f"\nCommand: python run_tempo.py {' '.join(args)}\n")
    run_tempo(args)


def example_dynamic_relu():
    """Dynamic thresholding with ReLU activation"""
    print("\n" + "="*60)
    print("Example 6: Dynamic ReLU Thresholding")
    print("="*60)
    print("Sharp threshold increase at specific point")
    print("-"*60)
    
    args = [
        "--prompt", "The secret to mastering any skill is",
        "--selection-threshold", "0.12",
        "--use-retroactive-pruning",
        "--attention-threshold", "0.004",
        "--dynamic-threshold",
        "--use-relu",
        "--relu-activation-point", "0.5",
        "--final-threshold", "0.04",
        "--max-tokens", "100"
    ]
    
    print(f"\nCommand: python run_tempo.py {' '.join(args)}\n")
    run_tempo(args)


def example_complete_removal():
    """Different pruning modes for handling positions"""
    print("\n" + "="*60)
    print("Example 7: Complete Position Removal")
    print("="*60)
    print("Aggressive pruning that removes entire positions")
    print("-"*60)
    
    args = [
        "--prompt", "The fundamental principles of democracy include",
        "--selection-threshold", "0.2",
        "--use-retroactive-pruning",
        "--attention-threshold", "0.015",
        "--complete-pruning-mode", "remove_position",
        "--max-tokens", "80"
    ]
    
    print(f"\nCommand: python run_tempo.py {' '.join(args)}\n")
    run_tempo(args)


def example_combined_features():
    """Combining multiple pruning features"""
    print("\n" + "="*60)
    print("Example 8: Combined Advanced Features")
    print("="*60)
    print("Multi-scale + Relative + Dynamic + Sigmoid")
    print("-"*60)
    
    args = [
        "--prompt", "To solve climate change, humanity must",
        "--selection-threshold", "0.1",
        "--use-retroactive-pruning",
        "--attention-threshold", "0.003",
        "--use-relative-attention",
        "--relative-threshold", "0.25",
        "--use-multi-scale-attention",
        "--num-layers-to-use", "12",
        "--use-sigmoid-threshold",
        "--sigmoid-steepness", "20.0",
        "--dynamic-threshold",
        "--bezier-p1", "0.15",
        "--bezier-p2", "0.85",
        "--max-tokens", "150"
    ]
    
    print(f"\nCommand: python run_tempo.py {' '.join(args)}\n")
    run_tempo(args)


def main():
    parser = argparse.ArgumentParser(
        description="TEMPO Advanced Pruning Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples demonstrate various pruning strategies:
  - Retroactive pruning based on attention
  - Relative vs absolute thresholds
  - Multi-scale attention integration
  - Dynamic thresholding curves
  - Different pruning modes

Usage:
  python examples/advanced_pruning.py --all        # Run all examples
  python examples/advanced_pruning.py --example 3  # Run specific example
  python examples/advanced_pruning.py --list       # List examples
        """
    )
    
    parser.add_argument(
        "--example",
        type=int,
        choices=range(1, 9),
        help="Run specific example (1-8)"
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
        1: ("Basic Retroactive Pruning", example_basic_retroactive),
        2: ("Relative Attention", example_relative_attention),
        3: ("Multi-Scale Attention", example_multi_scale_attention),
        4: ("Sigmoid Boundaries", example_sigmoid_boundaries),
        5: ("Dynamic Bezier", example_dynamic_bezier),
        6: ("Dynamic ReLU", example_dynamic_relu),
        7: ("Complete Removal", example_complete_removal),
        8: ("Combined Features", example_combined_features)
    }
    
    if args.list:
        print("\nAdvanced Pruning Examples:")
        print("-" * 50)
        for num, (name, _) in examples.items():
            print(f"{num}. {name}")
        return
        
    if args.all:
        print("\nRunning all advanced pruning examples...")
        print("This demonstrates different pruning strategies.\n")
        
        for num, (name, func) in examples.items():
            try:
                func()
                if num < len(examples):
                    input("\nPress Enter for next example...")
            except KeyboardInterrupt:
                print("\n\nExamples interrupted.")
                break
                
    elif args.example:
        name, func = examples[args.example]
        print(f"\nRunning Example {args.example}: {name}")
        func()
        
    else:
        parser.print_help()
        print("\n💡 Try --example 1 to see basic retroactive pruning")


if __name__ == "__main__":
    main()