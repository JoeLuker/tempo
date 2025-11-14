#!/usr/bin/env python3
"""
Demo: Optimized Compressed Thought Generation

Shows how position gaps + TEMPO parallel tokens = compressed thought vectors
where each parallel token encodes a complete semantic trajectory.

With optimized 4D masking, attention is perfectly preserved across all gap sizes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model
from src.algorithms.generation.compressed_thought_generator import (
    CompressedThoughtGenerator
)


def demo_basic_usage():
    """Basic example: Generate thought paths for a simple prompt."""

    print("="*80)
    print("COMPRESSED THOUGHT GENERATION - Basic Demo")
    print("="*80)
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    # Create generator
    generator = CompressedThoughtGenerator(
        model=model,
        tokenizer=tokenizer,
        device="mps"
    )

    # Generate compressed thoughts
    prompt = "The answer is"
    gap_size = 10

    print(f"\nPrompt: '{prompt}'")
    print(f"Gap size: {gap_size} tokens")
    print(f"\nGenerating {gap_size}-token thought paths...\n")

    paths = generator.generate_thought_paths(
        prompt=prompt,
        gap_size=gap_size,
        selection_threshold=0.05,
        max_parallel_paths=10,
        expand_paths=True,
    )

    print(f"Generated {len(paths)} complete thought paths:\n")

    for i, path in enumerate(paths, 1):
        print(f"{i}. [{path.probability:.4f}] {path.initial_token!r}")
        print(f"   Complete path: {path.full_path!r}")
        print(f"   Tokens: {' → '.join(repr(t) for t in path.path_tokens)}\n")

    print(f"\n✓ {len(paths)} different {gap_size}-token thoughts explored")
    print(f"✓ Each initial token encodes a complete semantic trajectory")
    print(f"✓ Same compute as sequential, multiple paths discovered")
    print(f"✓ Attention perfectly preserved with optimized masking")


def demo_multi_scale():
    """Show thoughts at different semantic scales."""

    print("\n\n" + "="*80)
    print("MULTI-SCALE THOUGHT GENERATION")
    print("="*80)
    print()

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    generator = CompressedThoughtGenerator(
        model=model,
        tokenizer=tokenizer,
        device="mps"
    )

    prompt = "In summary,"

    print(f"Prompt: '{prompt}'")
    print("\nGenerating thoughts at different scales:\n")

    # Generate at multiple gap sizes
    results = generator.generate_with_adaptive_gaps(
        prompt=prompt,
        gap_sizes=[3, 7, 15],
        selection_threshold=0.05,
    )

    for gap_size, paths in results.items():
        print(f"\n{gap_size}-token thoughts (semantic distance = {gap_size}):")
        print("-" * 70)

        # Show top 5 paths
        top_paths = generator.select_best_paths(paths, top_k=5)

        for i, path in enumerate(top_paths, 1):
            print(f"  {i}. [{path.probability:.3f}] '{path.full_path}'")


def demo_brainstorming():
    """Use compressed thoughts for brainstorming/ideation."""

    print("\n\n" + "="*80)
    print("BRAINSTORMING MODE: Exploring Answer Formats")
    print("="*80)
    print()

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    generator = CompressedThoughtGenerator(
        model=model,
        tokenizer=tokenizer,
        device="mps"
    )

    prompt = "The main benefit is"

    print(f"Prompt: '{prompt}'")
    print(f"\nExploring different ways to complete this thought:\n")

    paths = generator.generate_thought_paths(
        prompt=prompt,
        gap_size=12,
        selection_threshold=0.03,  # Lower threshold = more diversity
        max_parallel_paths=15,
    )

    print(f"Found {len(paths)} different conceptual directions:\n")

    for i, path in enumerate(paths, 1):
        print(f"{i:2d}. '{path.full_path}'")
        print(f"     (confidence: {path.probability:.3f}, starts with: '{path.initial_token}')\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OPTIMIZED COMPRESSED THOUGHT GENERATION DEMO")
    print("="*80)
    print()
    print("Demonstration of generating multiple complete thoughts")
    print("using position gaps + TEMPO + optimized 4D masking.")
    print()
    print("Key Innovation: Each parallel token at position N encodes")
    print("a complete semantic trajectory from current position to N.")
    print()
    print("Optimization: Explicit 4D boolean causal masks based on")
    print("sequence indices (not positions) preserve attention perfectly.")
    print()

    demo_basic_usage()
    demo_multi_scale()
    demo_brainstorming()

    print("\n" + "="*80)
    print("Demo complete!")
    print("="*80)
    print()
    print("Key Takeaways:")
    print("✓ Position gaps + TEMPO = compressed thought vectors")
    print("✓ Optimized masking preserves attention across all gap sizes")
    print("✓ Same compute, multiple complete thought paths explored")
    print("✓ Works for gap sizes from 1 to 20+ tokens")
    print()
    print("See docs/compressed_thought_optimization_results.md for details.")
    print("="*80)
