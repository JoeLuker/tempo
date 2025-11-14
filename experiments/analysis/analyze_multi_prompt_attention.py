#!/usr/bin/env python3
"""Analyze attention data from multi-prompt validation.

This script processes the attention data captured during multi-prompt generation
and computes statistics to validate the "parallel tokens receive reduced attention" finding.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class AttentionMetrics:
    """Metrics for attention analysis."""
    prompt_name: str
    category: str
    step: int
    parallel_mean_attn: float
    non_parallel_mean_attn: float
    ratio: float  # parallel / non_parallel
    num_parallel_tokens: int
    num_non_parallel_tokens: int


def analyze_prompt_attention(prompt_dir: Path, category: str) -> List[AttentionMetrics]:
    """Analyze attention data for a single prompt.

    Args:
        prompt_dir: Directory containing prompt experiment data
        category: Category of the prompt (narrative, factual, etc.)

    Returns:
        List of attention metrics for each step with parallel tokens
    """
    metrics = []

    # Check for attention weights file
    attention_file = prompt_dir / "attention_weights.npz"
    if not attention_file.exists():
        return metrics

    # Load attention data
    try:
        data = np.load(attention_file, allow_pickle=True)
    except Exception as e:
        print(f"  ✗ Failed to load attention from {prompt_dir.name}: {e}")
        return metrics

    # Load parallel sets to identify parallel vs non-parallel tokens
    parallel_sets_file = prompt_dir / "parallel_sets.json"
    parallel_sets_by_step = {}
    if parallel_sets_file.exists():
        with open(parallel_sets_file) as f:
            parallel_data = json.load(f)
            for pset in parallel_data.get("parallel_sets", []):
                parallel_sets_by_step[pset["step"]] = {
                    "positions": pset["positions"],
                    "count": pset["count"]
                }

    # Extract steps
    num_steps = 0
    for key in data.keys():
        if key.startswith("step_") and key.endswith("_logical"):
            num_steps += 1

    # Analyze each step with parallel tokens
    for i in range(num_steps):
        logical_step = int(data[f"step_{i}_logical"])
        positions = data[f"step_{i}_positions"]
        attention = data[f"step_{i}_attention"]  # Shape: [layers, batch, heads, seq_len, seq_len]

        # Check if this step has parallel tokens
        if logical_step not in parallel_sets_by_step:
            continue

        parallel_info = parallel_sets_by_step[logical_step]
        parallel_positions = set(parallel_info["positions"])

        if len(parallel_positions) < 2:
            continue  # Skip if only one token

        # Average across layers, batch, and heads
        if attention.ndim == 5:
            avg_attention = attention.mean(axis=(0, 1, 2))  # -> [seq_len, seq_len]
        elif attention.ndim == 4:
            avg_attention = attention.mean(axis=(0, 1))  # -> [seq_len, seq_len]
        elif attention.ndim == 3:
            avg_attention = attention.mean(axis=0)  # -> [seq_len, seq_len]
        else:
            avg_attention = attention

        # For each parallel token, measure its attention to non-parallel (context) tokens
        # and parallel tokens (should be near-zero in isolated mode)
        parallel_attention_scores = []
        non_parallel_attention_scores = []

        for pos in parallel_positions:
            if pos >= avg_attention.shape[0]:
                continue

            # Get attention from this parallel token to all previous positions
            attn_from_pos = avg_attention[pos, :pos]

            if len(attn_from_pos) == 0:
                continue

            # Split into parallel vs non-parallel
            for target_pos in range(pos):
                attn_score = attn_from_pos[target_pos]

                if target_pos in parallel_positions:
                    parallel_attention_scores.append(attn_score)
                else:
                    non_parallel_attention_scores.append(attn_score)

        # Calculate metrics
        if len(non_parallel_attention_scores) > 0:
            non_parallel_mean = float(np.mean(non_parallel_attention_scores))
            parallel_mean = float(np.mean(parallel_attention_scores)) if parallel_attention_scores else 0.0

            # Calculate ratio of parallel attention to non-parallel attention
            # This ratio should be close to 0 in isolated mode (parallel tokens don't attend to each other)
            # We're storing raw means for now
            ratio = parallel_mean / non_parallel_mean if non_parallel_mean > 0 else 0.0

            metrics.append(AttentionMetrics(
                prompt_name=prompt_dir.name,
                category=category,
                step=logical_step,
                parallel_mean_attn=parallel_mean,
                non_parallel_mean_attn=non_parallel_mean,
                ratio=ratio,
                num_parallel_tokens=len(parallel_positions),
                num_non_parallel_tokens=len(non_parallel_attention_scores)
            ))

    return metrics


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze attention data from multi-prompt validation"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/results/multi_prompt_attention"),
        help="Directory containing prompt experiment data"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output JSON file for results"
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return 1

    # Prompt categories
    prompt_categories = {
        "narrative_1": "narrative",
        "narrative_2": "narrative",
        "narrative_3": "narrative",
        "factual_1": "factual",
        "factual_2": "factual",
        "factual_3": "factual",
        "technical_1": "technical",
        "technical_2": "technical",
        "conversational_1": "conversational",
        "conversational_2": "conversational",
        "simple_1": "simple",
        "simple_2": "simple",
        "complex_1": "complex",
    }

    print("="*70)
    print("Multi-Prompt Attention Analysis")
    print("="*70)

    all_metrics = []
    category_metrics = {}

    # Analyze each prompt
    for prompt_name, category in prompt_categories.items():
        prompt_dir = args.input_dir / prompt_name
        if not prompt_dir.exists():
            print(f"⚠ Skipping {prompt_name}: directory not found")
            continue

        metrics = analyze_prompt_attention(prompt_dir, category)

        if metrics:
            all_metrics.extend(metrics)
            if category not in category_metrics:
                category_metrics[category] = []
            category_metrics[category].extend(metrics)
            print(f"✓ {prompt_name:20s} ({category:15s}): {len(metrics)} parallel steps")
        else:
            print(f"⚠ {prompt_name:20s} ({category:15s}): No parallel steps found")

    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)

    if not all_metrics:
        print("No attention metrics found. Make sure attention capture was enabled.")
        return 1

    # Overall statistics
    all_ratios = [m.ratio for m in all_metrics]
    all_parallel_means = [m.parallel_mean_attn for m in all_metrics]
    all_non_parallel_means = [m.non_parallel_mean_attn for m in all_metrics]

    print(f"\nOverall Statistics ({len(all_metrics)} parallel steps across {len(prompt_categories)} prompts):")
    print(f"  Parallel → Parallel attention:     {np.mean(all_parallel_means):.6f} ± {np.std(all_parallel_means):.6f}")
    print(f"  Parallel → Non-parallel attention: {np.mean(all_non_parallel_means):.6f} ± {np.std(all_non_parallel_means):.6f}")
    print(f"  Ratio (parallel/non-parallel):     {np.mean(all_ratios):.6f} ± {np.std(all_ratios):.6f}")
    print(f"  \nNote: In isolated mode, ratio should be near 0 (parallel tokens don't attend to each other)")

    # Category-wise statistics
    print(f"\nBy Category:")
    for category in sorted(category_metrics.keys()):
        cat_metrics = category_metrics[category]
        cat_ratios = [m.ratio for m in cat_metrics]
        cat_parallel = [m.parallel_mean_attn for m in cat_metrics]
        cat_non_parallel = [m.non_parallel_mean_attn for m in cat_metrics]

        print(f"\n  {category.capitalize()}:")
        print(f"    Steps: {len(cat_metrics)}")
        print(f"    Parallel → Parallel:     {np.mean(cat_parallel):.6f} ± {np.std(cat_parallel):.6f}")
        print(f"    Parallel → Non-parallel: {np.mean(cat_non_parallel):.6f} ± {np.std(cat_non_parallel):.6f}")
        print(f"    Ratio:                   {np.mean(cat_ratios):.6f} ± {np.std(cat_ratios):.6f}")

    # Save to JSON if requested
    if args.output_file:
        results = {
            "overall": {
                "num_prompts": len(prompt_categories),
                "num_parallel_steps": len(all_metrics),
                "parallel_to_parallel_mean": float(np.mean(all_parallel_means)),
                "parallel_to_parallel_std": float(np.std(all_parallel_means)),
                "parallel_to_non_parallel_mean": float(np.mean(all_non_parallel_means)),
                "parallel_to_non_parallel_std": float(np.std(all_non_parallel_means)),
                "ratio_mean": float(np.mean(all_ratios)),
                "ratio_std": float(np.std(all_ratios)),
            },
            "by_category": {},
            "all_metrics": [asdict(m) for m in all_metrics]
        }

        for category in category_metrics:
            cat_metrics = category_metrics[category]
            cat_ratios = [m.ratio for m in cat_metrics]
            cat_parallel = [m.parallel_mean_attn for m in cat_metrics]
            cat_non_parallel = [m.non_parallel_mean_attn for m in cat_metrics]

            results["by_category"][category] = {
                "num_steps": len(cat_metrics),
                "parallel_to_parallel_mean": float(np.mean(cat_parallel)),
                "parallel_to_parallel_std": float(np.std(cat_parallel)),
                "parallel_to_non_parallel_mean": float(np.mean(cat_non_parallel)),
                "parallel_to_non_parallel_std": float(np.std(cat_non_parallel)),
                "ratio_mean": float(np.mean(cat_ratios)),
                "ratio_std": float(np.std(cat_ratios)),
            }

        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {args.output_file}")

    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
