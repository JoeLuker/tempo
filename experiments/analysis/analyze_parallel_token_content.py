#!/usr/bin/env python3
"""Analyze what tokens actually appear at parallel positions.

This investigates whether token semantics/types correlate with attention patterns.
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_experiment_metadata(prompt_dir: Path) -> Dict:
    """Load metadata about the experiment."""
    metadata_file = prompt_dir / "experiment_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            return json.load(f)
    return {}


def analyze_token_content(prompt_dir: Path, prompt_name: str, tokenizer) -> Dict:
    """Analyze the actual tokens at parallel vs non-parallel positions."""

    # Load parallel sets
    parallel_sets_file = prompt_dir / "parallel_sets.json"
    if not parallel_sets_file.exists():
        return None

    with open(parallel_sets_file) as f:
        parallel_data = json.load(f)

    parallel_token_info = []

    for pset in parallel_data.get("parallel_sets", []):
        step = pset["step"]
        tokens = pset["tokens"]
        positions = pset["positions"]

        # Decode tokens
        decoded = [tokenizer.decode([tid]) for tid in tokens]

        parallel_token_info.append({
            "step": step,
            "positions": positions,
            "token_ids": tokens,
            "tokens": decoded
        })

    # Load attention data to get reduction percentage
    results_file = prompt_dir.parent / "final_results.json"
    reduction = None
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
            for metric in results.get("all_metrics", []):
                if metric["prompt_name"] == prompt_name:
                    reduction = metric["reduction_percentage"]
                    break

    return {
        "prompt_name": prompt_name,
        "reduction_percentage": reduction,
        "parallel_sets": parallel_token_info,
        "total_parallel_tokens": sum(len(ps["tokens"]) for ps in parallel_token_info)
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze parallel token content")
    parser.add_argument("--input-dir", type=Path, default=Path("experiments/results/multi_prompt_attention"))
    args = parser.parse_args()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepcogito/cogito-v1-preview-llama-3B")

    prompts = [
        "narrative_1", "narrative_2", "narrative_3",
        "factual_1", "factual_2", "factual_3",
        "technical_1", "technical_2",
        "conversational_1", "conversational_2",
        "simple_1", "simple_2",
        "complex_1"
    ]

    print("\n" + "="*70)
    print("PARALLEL TOKEN CONTENT ANALYSIS")
    print("="*70)

    all_analyses = []

    for prompt_name in prompts:
        prompt_dir = args.input_dir / prompt_name
        if not prompt_dir.exists():
            continue

        analysis = analyze_token_content(prompt_dir, prompt_name, tokenizer)
        if analysis:
            all_analyses.append(analysis)

    # Group by extreme cases
    sorted_by_reduction = sorted([a for a in all_analyses if a["reduction_percentage"] is not None],
                                   key=lambda x: x["reduction_percentage"])

    most_negative = sorted_by_reduction[:3]  # Parallel gets MOST attention
    most_positive = sorted_by_reduction[-3:]  # Parallel gets LEAST attention

    print("\n🔴 PARALLEL GETS MOST ATTENTION (negative reduction):")
    for analysis in most_negative:
        print(f"\n{analysis['prompt_name']} ({analysis['reduction_percentage']:+.1f}%):")
        print(f"  Total parallel tokens: {analysis['total_parallel_tokens']}")
        print(f"  Parallel token sets:")
        for pset in analysis["parallel_sets"]:
            tokens_str = " / ".join([f"'{t}'" for t in pset["tokens"]])
            print(f"    Step {pset['step']}: {tokens_str} (positions {pset['positions']})")

    print("\n\n🔵 PARALLEL GETS LEAST ATTENTION (positive reduction):")
    for analysis in most_positive:
        print(f"\n{analysis['prompt_name']} ({analysis['reduction_percentage']:+.1f}%):")
        print(f"  Total parallel tokens: {analysis['total_parallel_tokens']}")
        print(f"  Parallel token sets:")
        for pset in analysis["parallel_sets"]:
            tokens_str = " / ".join([f"'{t}'" for t in pset["tokens"]])
            print(f"    Step {pset['step']}: {tokens_str} (positions {pset['positions']})")

    # Aggregate token statistics
    print("\n\n" + "="*70)
    print("TOKEN FREQUENCY ANALYSIS")
    print("="*70)

    # Collect all parallel tokens
    high_attn_tokens = []  # Tokens that got high attention
    low_attn_tokens = []   # Tokens that got low attention

    for analysis in most_negative:
        for pset in analysis["parallel_sets"]:
            high_attn_tokens.extend(pset["tokens"])

    for analysis in most_positive:
        for pset in analysis["parallel_sets"]:
            low_attn_tokens.extend(pset["tokens"])

    print(f"\nTokens in HIGH attention prompts (n={len(high_attn_tokens)}):")
    high_counter = Counter(high_attn_tokens)
    for token, count in high_counter.most_common(10):
        decoded = tokenizer.decode([token])
        print(f"  '{decoded}' (id={token}): {count} times")

    print(f"\nTokens in LOW attention prompts (n={len(low_attn_tokens)}):")
    low_counter = Counter(low_attn_tokens)
    for token, count in low_counter.most_common(10):
        decoded = tokenizer.decode([token])
        print(f"  '{decoded}' (id={token}): {count} times")

    # Look for common vs unique tokens
    high_set = set(high_attn_tokens)
    low_set = set(low_attn_tokens)
    common = high_set & low_set
    high_only = high_set - low_set
    low_only = low_set - high_set

    print(f"\n\nToken Overlap:")
    print(f"  Tokens in both HIGH and LOW attention prompts: {len(common)}")
    print(f"  Tokens unique to HIGH attention prompts: {len(high_only)}")
    print(f"  Tokens unique to LOW attention prompts: {len(low_only)}")

    if common:
        print(f"\n  Common tokens (appear in both):")
        for token_id in list(common)[:10]:
            decoded = tokenizer.decode([token_id])
            print(f"    '{decoded}' (id={token_id})")

    print("\n" + "="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
