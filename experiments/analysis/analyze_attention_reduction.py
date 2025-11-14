#!/usr/bin/env python3
"""Analyze if parallel tokens receive reduced attention compared to non-parallel tokens.

This script addresses the original research question: "Do tokens at position sharing steps
receive 40-60% less attention than tokens at unique position steps?"

The hypothesis is that because parallel tokens share the same RoPE position, other tokens
in the sequence may attend less to them compared to tokens at unique positions.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class AttentionReductionMetrics:
    """Metrics comparing attention to parallel vs non-parallel tokens."""
    prompt_name: str
    category: str

    # Attention FROM later tokens TO this step's tokens
    mean_attn_to_parallel: float  # Mean attention received by parallel tokens
    mean_attn_to_non_parallel: float  # Mean attention received by non-parallel tokens

    # Reduction ratio
    reduction_ratio: float  # (non_parallel - parallel) / non_parallel
    reduction_percentage: float  # reduction_ratio * 100

    # Sample sizes
    num_parallel_positions: int
    num_non_parallel_positions: int
    num_attending_positions: int  # Positions that attend to these tokens


def analyze_attention_reduction(prompt_dir: Path, category: str) -> List[AttentionReductionMetrics]:
    """Analyze whether parallel tokens receive less attention than non-parallel tokens.

    Args:
        prompt_dir: Directory containing prompt experiment data
        category: Category of the prompt

    Returns:
        List of metrics for each generation step
    """
    metrics = []

    # Load attention data
    attention_file = prompt_dir / "attention_weights.npz"
    if not attention_file.exists():
        return metrics

    try:
        data = np.load(attention_file, allow_pickle=True)
    except Exception as e:
        print(f"  ✗ Failed to load attention from {prompt_dir.name}: {e}")
        return metrics

    # Load parallel sets
    parallel_sets_file = prompt_dir / "parallel_sets.json"
    parallel_sets_by_step = {}
    if parallel_sets_file.exists():
        with open(parallel_sets_file) as f:
            parallel_data = json.load(f)
            for pset in parallel_data.get("parallel_sets", []):
                parallel_sets_by_step[pset["step"]] = set(pset["positions"])

    # Extract number of steps
    num_steps = sum(1 for key in data.keys() if key.startswith("step_") and key.endswith("_logical"))

    # Analyze each step
    for i in range(num_steps):
        logical_step = int(data[f"step_{i}_logical"])
        positions = data[f"step_{i}_positions"]
        attention = data[f"step_{i}_attention"]

        # Average across layers, batch, and heads to get [seq_len, seq_len]
        if attention.ndim == 5:
            avg_attention = attention.mean(axis=(0, 1, 2))
        elif attention.ndim == 4:
            avg_attention = attention.mean(axis=(0, 1))
        elif attention.ndim == 3:
            avg_attention = attention.mean(axis=0)
        else:
            avg_attention = attention

        seq_len = avg_attention.shape[0]

        # Get all parallel positions up to this step
        all_parallel_positions = set()
        for step in range(logical_step + 1):
            if step in parallel_sets_by_step:
                all_parallel_positions.update(parallel_sets_by_step[step])

        # Get non-parallel positions (all positions except parallel ones and prompt)
        # We need to know prompt length - assume first step's positions start after prompt
        if i == 0:
            prompt_length = positions[0] if isinstance(positions, (list, np.ndarray)) else positions
        else:
            prompt_length = data["step_0_positions"][0] if isinstance(data["step_0_positions"], (list, np.ndarray)) else data["step_0_positions"]

        all_non_parallel_positions = set(range(prompt_length, seq_len)) - all_parallel_positions

        if len(all_parallel_positions) == 0 or len(all_non_parallel_positions) == 0:
            continue

        # For each position AFTER this step, measure attention TO parallel vs non-parallel tokens
        # We want to see if tokens generated after parallel tokens attend less to them

        # Get positions that come after the current step
        current_step_end = max(positions) if isinstance(positions, (list, np.ndarray)) else positions
        attending_positions = range(current_step_end + 1, seq_len)

        if len(attending_positions) == 0:
            continue  # No future tokens to analyze

        # Collect attention scores TO parallel and non-parallel tokens FROM future tokens
        attn_to_parallel = []
        attn_to_non_parallel = []

        for from_pos in attending_positions:
            if from_pos >= avg_attention.shape[0]:
                continue

            # Attention from this position to all previous positions
            attn_weights = avg_attention[from_pos, :from_pos]

            # Split by parallel vs non-parallel targets
            for to_pos in range(from_pos):
                if to_pos < prompt_length:
                    continue  # Skip prompt tokens

                weight = attn_weights[to_pos]

                if to_pos in all_parallel_positions:
                    attn_to_parallel.append(weight)
                elif to_pos in all_non_parallel_positions:
                    attn_to_non_parallel.append(weight)

        # Calculate metrics
        if len(attn_to_parallel) > 0 and len(attn_to_non_parallel) > 0:
            mean_parallel = float(np.mean(attn_to_parallel))
            mean_non_parallel = float(np.mean(attn_to_non_parallel))

            # Calculate reduction
            reduction_ratio = (mean_non_parallel - mean_parallel) / mean_non_parallel if mean_non_parallel > 0 else 0.0
            reduction_percentage = reduction_ratio * 100

            metrics.append(AttentionReductionMetrics(
                prompt_name=prompt_dir.name,
                category=category,
                mean_attn_to_parallel=mean_parallel,
                mean_attn_to_non_parallel=mean_non_parallel,
                reduction_ratio=reduction_ratio,
                reduction_percentage=reduction_percentage,
                num_parallel_positions=len(all_parallel_positions),
                num_non_parallel_positions=len(all_non_parallel_positions),
                num_attending_positions=len(attending_positions)
            ))

    return metrics


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze attention reduction to parallel vs non-parallel tokens"
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
    print("Attention Reduction Analysis: Parallel vs Non-Parallel Tokens")
    print("="*70)
    print("\nResearch Question: Do tokens receive less attention when directed")
    print("at parallel tokens (shared RoPE position) vs non-parallel tokens?")
    print("="*70)

    all_metrics = []
    category_metrics = {}

    # Analyze each prompt
    for prompt_name, category in prompt_categories.items():
        prompt_dir = args.input_dir / prompt_name
        if not prompt_dir.exists():
            print(f"⚠ Skipping {prompt_name}: directory not found")
            continue

        metrics = analyze_attention_reduction(prompt_dir, category)

        if metrics:
            all_metrics.extend(metrics)
            if category not in category_metrics:
                category_metrics[category] = []
            category_metrics[category].extend(metrics)
            print(f"✓ {prompt_name:20s} ({category:15s}): {len(metrics)} analyzable steps")
        else:
            print(f"⚠ {prompt_name:20s} ({category:15s}): No analyzable steps")

    print("\n" + "="*70)
    print("RESULTS: Attention Reduction to Parallel Tokens")
    print("="*70)

    if not all_metrics:
        print("No metrics found. Ensure attention capture was enabled.")
        return 1

    # Overall statistics
    all_reductions = [m.reduction_percentage for m in all_metrics]
    all_to_parallel = [m.mean_attn_to_parallel for m in all_metrics]
    all_to_non_parallel = [m.mean_attn_to_non_parallel for m in all_metrics]

    print(f"\n📊 Overall Statistics ({len(all_metrics)} steps across {len(set(m.prompt_name for m in all_metrics))} prompts):")
    print(f"\n  Mean attention TO parallel tokens:     {np.mean(all_to_parallel):.6f} ± {np.std(all_to_parallel):.6f}")
    print(f"  Mean attention TO non-parallel tokens: {np.mean(all_to_non_parallel):.6f} ± {np.std(all_to_non_parallel):.6f}")
    print(f"\n  🎯 ATTENTION REDUCTION: {np.mean(all_reductions):.1f}% ± {np.std(all_reductions):.1f}%")
    print(f"     Range: [{np.min(all_reductions):.1f}%, {np.max(all_reductions):.1f}%]")
    print(f"     Median: {np.median(all_reductions):.1f}%")

    # Check if reduction is statistically significant
    from scipy import stats
    if len(all_to_parallel) > 1 and len(all_to_non_parallel) > 1:
        t_stat, p_value = stats.ttest_rel(all_to_non_parallel, all_to_parallel)
        print(f"\n  📈 Statistical Test (paired t-test):")
        print(f"     t-statistic: {t_stat:.4f}")
        print(f"     p-value: {p_value:.6f}")
        if p_value < 0.05:
            print(f"     ✅ Statistically significant (p < 0.05)")
        else:
            print(f"     ⚠️  Not statistically significant (p ≥ 0.05)")

    # Category-wise statistics
    print(f"\n📁 By Category:")
    for category in sorted(category_metrics.keys()):
        cat_metrics = category_metrics[category]
        cat_reductions = [m.reduction_percentage for m in cat_metrics]
        cat_to_parallel = [m.mean_attn_to_parallel for m in cat_metrics]
        cat_to_non_parallel = [m.mean_attn_to_non_parallel for m in cat_metrics]

        print(f"\n  {category.capitalize()}:")
        print(f"    Steps analyzed: {len(cat_metrics)}")
        print(f"    Attention to parallel:     {np.mean(cat_to_parallel):.6f} ± {np.std(cat_to_parallel):.6f}")
        print(f"    Attention to non-parallel: {np.mean(cat_to_non_parallel):.6f} ± {np.std(cat_to_non_parallel):.6f}")
        print(f"    Reduction: {np.mean(cat_reductions):.1f}% ± {np.std(cat_reductions):.1f}%")

    # Save results to JSON if requested
    if args.output_file:
        results = {
            "summary": {
                "research_question": "Do tokens receive less attention to parallel tokens vs non-parallel tokens?",
                "num_prompts": len(set(m.prompt_name for m in all_metrics)),
                "num_steps_analyzed": len(all_metrics),
                "mean_reduction_percentage": float(np.mean(all_reductions)),
                "std_reduction_percentage": float(np.std(all_reductions)),
                "median_reduction_percentage": float(np.median(all_reductions)),
                "min_reduction_percentage": float(np.min(all_reductions)),
                "max_reduction_percentage": float(np.max(all_reductions)),
                "mean_attn_to_parallel": float(np.mean(all_to_parallel)),
                "std_attn_to_parallel": float(np.std(all_to_parallel)),
                "mean_attn_to_non_parallel": float(np.mean(all_to_non_parallel)),
                "std_attn_to_non_parallel": float(np.std(all_to_non_parallel)),
            },
            "by_category": {},
            "all_metrics": [asdict(m) for m in all_metrics]
        }

        if len(all_to_parallel) > 1 and len(all_to_non_parallel) > 1:
            t_stat, p_value = stats.ttest_rel(all_to_non_parallel, all_to_parallel)
            results["summary"]["t_statistic"] = float(t_stat)
            results["summary"]["p_value"] = float(p_value)
            results["summary"]["statistically_significant"] = p_value < 0.05

        for category in category_metrics:
            cat_metrics = category_metrics[category]
            cat_reductions = [m.reduction_percentage for m in cat_metrics]
            cat_to_parallel = [m.mean_attn_to_parallel for m in cat_metrics]
            cat_to_non_parallel = [m.mean_attn_to_non_parallel for m in cat_metrics]

            results["by_category"][category] = {
                "num_steps": len(cat_metrics),
                "mean_reduction_percentage": float(np.mean(cat_reductions)),
                "std_reduction_percentage": float(np.std(cat_reductions)),
                "mean_attn_to_parallel": float(np.mean(cat_to_parallel)),
                "std_attn_to_parallel": float(np.std(cat_to_parallel)),
                "mean_attn_to_non_parallel": float(np.mean(cat_to_non_parallel)),
                "std_attn_to_non_parallel": float(np.std(cat_to_non_parallel)),
            }

        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {args.output_file}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    mean_reduction = np.mean(all_reductions)
    if mean_reduction > 30:
        print(f"✅ CONFIRMED: Parallel tokens receive {mean_reduction:.1f}% less attention")
        print(f"   on average compared to non-parallel tokens.")
    elif mean_reduction > 10:
        print(f"⚠️  PARTIAL: Parallel tokens receive {mean_reduction:.1f}% less attention,")
        print(f"   but the effect is smaller than initially hypothesized (40-60%).")
    else:
        print(f"❌ NOT CONFIRMED: Minimal reduction ({mean_reduction:.1f}%) observed.")

    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
