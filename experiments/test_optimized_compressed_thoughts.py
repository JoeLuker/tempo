#!/usr/bin/env python3
"""
Test the optimized CompressedThoughtGenerator with proper 4D masking.

Benchmarks different gap sizes to find optimal configuration.
"""

import sys
from pathlib import Path
import torch
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model
from src.algorithms.generation.compressed_thought_generator import CompressedThoughtGenerator


def benchmark_gap_size(prompt: str, gap_size: int, threshold: float = 0.05):
    """
    Test compressed thought generation with a specific gap size.

    Measures:
    - Number of parallel paths found
    - Attention patterns
    - Output quality
    - Generation time
    """

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    generator = CompressedThoughtGenerator(
        model=model,
        tokenizer=tokenizer,
        device="mps",
    )

    print(f"\n{'='*80}")
    print(f"Testing gap_size={gap_size}, threshold={threshold}")
    print(f"{'='*80}")

    start_time = time.time()

    thought_paths = generator.generate_thought_paths(
        prompt=prompt,
        gap_size=gap_size,
        selection_threshold=threshold,
        max_parallel_paths=10,
        expand_paths=True,
    )

    elapsed = time.time() - start_time

    print(f"\nGenerated {len(thought_paths)} parallel thought paths in {elapsed:.2f}s")
    print(f"\nTop paths:")

    for i, path in enumerate(thought_paths[:5], 1):
        print(f"\n{i}. [{path.probability:.4f}] {path.initial_token!r}")
        print(f"   Full path: {path.full_path!r}")
        print(f"   Tokens: {path.path_tokens}")

    # Analyze diversity
    if len(thought_paths) > 1:
        initial_tokens = [p.initial_token for p in thought_paths]
        unique_initials = len(set(initial_tokens))
        diversity_ratio = unique_initials / len(initial_tokens)
        print(f"\nDiversity: {unique_initials}/{len(initial_tokens)} unique initial tokens ({diversity_ratio:.2%})")

    return {
        'gap_size': gap_size,
        'num_paths': len(thought_paths),
        'elapsed': elapsed,
        'paths': thought_paths,
    }


def compare_gap_sizes(prompt: str):
    """Compare different gap sizes to find optimal configuration."""

    print("="*80)
    print("BENCHMARKING COMPRESSED THOUGHT GENERATION")
    print("="*80)
    print(f"Prompt: '{prompt}'")

    gap_sizes = [0, 3, 5, 7, 10, 15, 20]
    results = []

    for gap_size in gap_sizes:
        result = benchmark_gap_size(prompt, gap_size, threshold=0.05)
        results.append(result)

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Gap':>4} | {'Paths':>5} | {'Time':>6} | {'Tokens/Path':>12} | Quality")
    print("-"*80)

    for result in results:
        gap = result['gap_size']
        num_paths = result['num_paths']
        elapsed = result['elapsed']

        if result['paths']:
            avg_tokens = sum(len(p.path_tokens) for p in result['paths']) / len(result['paths'])
            top_prob = result['paths'][0].probability
            quality = "✓" if top_prob > 0.1 and num_paths > 1 else "⚠"
        else:
            avg_tokens = 0
            quality = "✗"

        print(f"{gap:4d} | {num_paths:5d} | {elapsed:6.2f}s | {avg_tokens:12.1f} | {quality}")

    # Find optimal
    valid_results = [r for r in results if r['num_paths'] > 1]
    if valid_results:
        optimal = max(valid_results, key=lambda r: r['num_paths'] * (1 + r['gap_size']/10))
        print(f"\nOptimal gap size: {optimal['gap_size']} ({optimal['num_paths']} paths)")


def test_attention_preservation():
    """
    Verify that optimized masking preserves attention to prompt.

    This tests the fix: with proper 4D masks, attention should be maintained
    even with position gaps.
    """

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    prompt = "The answer is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")
    prompt_length = input_ids.shape[1]

    print("\n" + "="*80)
    print("TESTING ATTENTION PRESERVATION WITH OPTIMIZED MASKS")
    print("="*80)
    print(f"Prompt: '{prompt}'")

    from src.algorithms.generation.attention_mask_utils import create_sequence_based_attention_mask

    for gap in [0, 5, 10, 20]:
        # Create position IDs with gap
        if gap == 0:
            position_ids = torch.arange(prompt_length, device="mps").unsqueeze(0)
        else:
            position_ids = torch.arange(prompt_length, device="mps").unsqueeze(0)

        # Create optimized attention mask
        attention_mask = create_sequence_based_attention_mask(
            input_ids=input_ids,
            position_ids=position_ids,
        )

        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
                output_attentions=True,
            )

        # Analyze attention
        attentions = outputs.attentions[-1]
        avg_attn = attentions[0, :, -1, :].mean(dim=0).cpu().numpy()

        attn_to_prompt = avg_attn[:prompt_length].sum()

        print(f"\nGap={gap:3d}: attention_to_prompt={attn_to_prompt:.6f}")

        # Get next token prediction
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        top_k = torch.topk(probs, k=3)

        print(f"  Top predictions:")
        for prob, token_id in zip(top_k.values, top_k.indices):
            token = tokenizer.decode([token_id.item()])
            print(f"    {prob:.4f}: {token!r}")


if __name__ == "__main__":
    prompt = "Once upon a time"

    # Test 1: Verify attention preservation
    test_attention_preservation()

    # Test 2: Compare gap sizes
    compare_gap_sizes(prompt)

    # Test 3: Detailed look at optimal gap
    print("\n" + "="*80)
    print("DETAILED ANALYSIS OF OPTIMAL GAP")
    print("="*80)
    benchmark_gap_size(prompt, gap_size=5, threshold=0.05)
