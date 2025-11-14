#!/usr/bin/env python3
"""Analyze if parallel tokens receive reduced attention (CORRECTED VERSION).

This analyzes attention patterns across the entire generation sequence to determine
if tokens attend less to parallel positions vs non-parallel positions.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class GlobalAttentionMetrics:
    """Metrics for attention to parallel vs non-parallel tokens across entire sequence."""
    prompt_name: str
    category: str

    # Global statistics
    mean_attn_to_parallel: float
    mean_attn_to_non_parallel: float
    reduction_percentage: float

    # Sample sizes
    num_parallel_positions: int
    num_non_parallel_positions: int
    num_attention_measurements: int


def analyze_global_attention(prompt_dir: Path, category: str) -> GlobalAttentionMetrics:
    """Analyze attention to parallel vs non-parallel tokens across entire generation.

    Args:
        prompt_dir: Directory containing experiment data
        category: Prompt category

    Returns:
        Global attention metrics, or None if no data
    """
    # Load data
    attention_file = prompt_dir / "attention_weights.npz"
    if not attention_file.exists():
        return None

    try:
        data = np.load(attention_file, allow_pickle=True)
    except Exception as e:
        print(f"  ✗ Failed to load {prompt_dir.name}: {e}")
        return None

    # Load parallel sets
    parallel_sets_file = prompt_dir / "parallel_sets.json"
    all_parallel_positions = set()
    if parallel_sets_file.exists():
        with open(parallel_sets_file) as f:
            parallel_data = json.load(f)
            for pset in parallel_data.get("parallel_sets", []):
                all_parallel_positions.update(pset["positions"])

    if len(all_parallel_positions) == 0:
        return None

    # Find prompt length (first generated token position)
    prompt_length = int(data["step_0_positions"][0] if isinstance(data["step_0_positions"], (list, np.ndarray)) else data["step_0_positions"])

    # Collect all attention weights across all steps
    attn_to_parallel = []
    attn_to_non_parallel = []

    num_steps = sum(1 for k in data.keys() if k.startswith("step_") and k.endswith("_logical"))

    for i in range(num_steps):
        attention = data[f"step_{i}_attention"]

        # Average across layers, batch, heads -> [seq_len, seq_len]
        if attention.ndim == 5:
            avg_attn = attention.mean(axis=(0, 1, 2))
        elif attention.ndim == 4:
            avg_attn = attention.mean(axis=(0, 1))
        elif attention.ndim == 3:
            avg_attn = attention.mean(axis=0)
        else:
            avg_attn = attention

        seq_len = avg_attn.shape[0]

        # For each position in the sequence (from), measure attention to all previous positions (to)
        for from_pos in range(prompt_length, seq_len):
            # Get attention weights from this position to all previous positions
            attn_weights = avg_attn[from_pos, :from_pos]

            # Split by parallel vs non-parallel TARGET positions
            for to_pos in range(from_pos):
                if to_pos < prompt_length:
                    continue  # Skip prompt

                weight = float(attn_weights[to_pos])

                if to_pos in all_parallel_positions:
                    attn_to_parallel.append(weight)
                else:
                    attn_to_non_parallel.append(weight)

    # Calculate metrics
    if len(attn_to_parallel) == 0 or len(attn_to_non_parallel) == 0:
        return None

    mean_parallel = float(np.mean(attn_to_parallel))
    mean_non_parallel = float(np.mean(attn_to_non_parallel))

    reduction = (mean_non_parallel - mean_parallel) / mean_non_parallel * 100 if mean_non_parallel > 0 else 0.0

    # Count unique positions
    all_generated_positions = set(range(prompt_length, seq_len))
    non_parallel_positions = all_generated_positions - all_parallel_positions

    return GlobalAttentionMetrics(
        prompt_name=prompt_dir.name,
        category=category,
        mean_attn_to_parallel=mean_parallel,
        mean_attn_to_non_parallel=mean_non_parallel,
        reduction_percentage=reduction,
        num_parallel_positions=len(all_parallel_positions),
        num_non_parallel_positions=len(non_parallel_positions),
        num_attention_measurements=len(attn_to_parallel) + len(attn_to_non_parallel)
    )


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze attention reduction (corrected version)")
    parser.add_argument("--input-dir", type=Path, default=Path("experiments/results/multi_prompt_attention"))
    parser.add_argument("--output-file", type=Path, default=None)
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: {args.input_dir} does not exist")
        return 1

    # Auto-discover all prompt directories and infer category from name
    prompt_categories = {}
    for item in args.input_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Infer category from prompt name (e.g., "narrative_1" -> "narrative")
            parts = item.name.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                category = parts[0]
            else:
                category = "unknown"
            prompt_categories[item.name] = category

    print("="*70)
    print("Attention Reduction Analysis (Corrected)")
    print("="*70)
    print("\nResearch Question: Do tokens attend less to parallel positions")
    print("(shared RoPE) vs non-parallel positions (unique RoPE)?")
    print("="*70)

    all_metrics = []
    category_metrics = {}

    for prompt_name, category in prompt_categories.items():
        prompt_dir = args.input_dir / prompt_name
        if not prompt_dir.exists():
            print(f"⚠ Skipping {prompt_name}")
            continue

        metrics = analyze_global_attention(prompt_dir, category)

        if metrics:
            all_metrics.append(metrics)
            if category not in category_metrics:
                category_metrics[category] = []
            category_metrics[category].append(metrics)
            print(f"✓ {prompt_name:20s} ({category:15s}): {metrics.reduction_percentage:+6.1f}% reduction")
        else:
            print(f"⚠ {prompt_name:20s} ({category:15s}): No data")

    if not all_metrics:
        print("\n❌ No metrics found")
        return 1

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    # Overall statistics
    reductions = [m.reduction_percentage for m in all_metrics]
    to_parallel = [m.mean_attn_to_parallel for m in all_metrics]
    to_non_parallel = [m.mean_attn_to_non_parallel for m in all_metrics]

    print(f"\n📊 Overall ({len(all_metrics)} prompts):")
    print(f"\n  Attention TO parallel tokens:     {np.mean(to_parallel):.6f} ± {np.std(to_parallel):.6f}")
    print(f"  Attention TO non-parallel tokens: {np.mean(to_non_parallel):.6f} ± {np.std(to_non_parallel):.6f}")
    print(f"\n  🎯 REDUCTION: {np.mean(reductions):.1f}% ± {np.std(reductions):.1f}%")
    print(f"     Range: [{np.min(reductions):.1f}%, {np.max(reductions):.1f}%]")
    print(f"     Median: {np.median(reductions):.1f}%")

    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(to_non_parallel, to_parallel)
    print(f"\n  📈 Paired t-test:")
    print(f"     t = {t_stat:.4f}, p = {p_value:.6f}")
    if p_value < 0.05:
        print(f"     ✅ Statistically significant (p < 0.05)")
    else:
        print(f"     ⚠️  Not significant (p ≥ 0.05)")

    # By category
    print(f"\n📁 By Category:")
    for category in sorted(category_metrics.keys()):
        cat = category_metrics[category]
        cat_red = [m.reduction_percentage for m in cat]
        print(f"  {category.capitalize():15s}: {np.mean(cat_red):+6.1f}% ± {np.std(cat_red):.1f}% ({len(cat)} prompts)")

    # Save results
    if args.output_file:
        results = {
            "summary": {
                "num_prompts": len(all_metrics),
                "mean_reduction_percentage": float(np.mean(reductions)),
                "std_reduction_percentage": float(np.std(reductions)),
                "median_reduction_percentage": float(np.median(reductions)),
                "min_reduction_percentage": float(np.min(reductions)),
                "max_reduction_percentage": float(np.max(reductions)),
                "mean_attn_to_parallel": float(np.mean(to_parallel)),
                "mean_attn_to_non_parallel": float(np.mean(to_non_parallel)),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "statistically_significant": bool(p_value < 0.05)
            },
            "by_category": {},
            "all_metrics": [asdict(m) for m in all_metrics]
        }

        for category in category_metrics:
            cat = category_metrics[category]
            cat_red = [m.reduction_percentage for m in cat]
            results["by_category"][category] = {
                "num_prompts": len(cat),
                "mean_reduction_percentage": float(np.mean(cat_red)),
                "std_reduction_percentage": float(np.std(cat_red))
            }

        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved to: {args.output_file}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    mean_red = np.mean(reductions)
    if 40 <= mean_red <= 60:
        print(f"✅ CONFIRMED: {mean_red:.1f}% reduction (within 40-60% range)")
    elif mean_red > 30:
        print(f"⚠️  PARTIAL: {mean_red:.1f}% reduction (outside 40-60% range)")
    elif mean_red > 10:
        print(f"⚠️  WEAK: {mean_red:.1f}% reduction (below expected range)")
    else:
        print(f"❌ NOT CONFIRMED: {mean_red:.1f}% reduction (minimal effect)")

    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
