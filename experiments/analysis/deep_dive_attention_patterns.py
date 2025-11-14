#!/usr/bin/env python3
"""Deep dive analysis of attention pattern variability.

This script investigates WHY attention patterns vary so dramatically across prompts,
with some showing +30% MORE attention to parallel tokens and others showing -60% LESS.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class DetailedPromptAnalysis:
    """Detailed analysis of a single prompt's attention patterns."""
    prompt_name: str
    category: str
    prompt_text: str

    # Overall metrics
    mean_attn_to_parallel: float
    mean_attn_to_non_parallel: float
    reduction_percentage: float

    # Granular analysis
    num_parallel_positions: int
    num_non_parallel_positions: int
    parallel_positions: List[int]

    # Per-step breakdown
    attention_by_step: List[Dict]

    # Position-wise analysis
    attention_distribution_to_parallel: List[float]
    attention_distribution_to_non_parallel: List[float]

    # Temporal analysis
    early_vs_late_effect: Dict


def analyze_prompt_in_detail(prompt_dir: Path, category: str, prompt_text: str) -> DetailedPromptAnalysis:
    """Perform deep analysis of attention patterns for a single prompt."""

    # Load data
    attention_file = prompt_dir / "attention_weights.npz"
    if not attention_file.exists():
        return None

    data = np.load(attention_file, allow_pickle=True)

    # Load parallel sets
    parallel_sets_file = prompt_dir / "parallel_sets.json"
    all_parallel_positions = set()
    parallel_steps = {}

    if parallel_sets_file.exists():
        with open(parallel_sets_file) as f:
            parallel_data = json.load(f)
            for pset in parallel_data.get("parallel_sets", []):
                positions = pset["positions"]
                all_parallel_positions.update(positions)
                parallel_steps[pset["step"]] = positions

    if len(all_parallel_positions) == 0:
        return None

    # Get prompt length
    prompt_length = int(data["step_0_positions"][0] if isinstance(data["step_0_positions"], (list, np.ndarray)) else data["step_0_positions"])

    # Collect attention data
    num_steps = sum(1 for k in data.keys() if k.startswith("step_") and k.endswith("_logical"))

    attn_to_parallel_all = []
    attn_to_non_parallel_all = []

    attention_by_step = []

    for i in range(num_steps):
        attention = data[f"step_{i}_attention"]
        logical_step = int(data[f"step_{i}_logical"])

        # Average attention
        if attention.ndim == 5:
            avg_attn = attention.mean(axis=(0, 1, 2))
        elif attention.ndim == 4:
            avg_attn = attention.mean(axis=(0, 1))
        elif attention.ndim == 3:
            avg_attn = attention.mean(axis=0)
        else:
            avg_attn = attention

        seq_len = avg_attn.shape[0]

        # Track per-step attention
        step_parallel = []
        step_non_parallel = []

        for from_pos in range(prompt_length, seq_len):
            attn_weights = avg_attn[from_pos, :from_pos]

            for to_pos in range(from_pos):
                if to_pos < prompt_length:
                    continue

                weight = float(attn_weights[to_pos])

                if to_pos in all_parallel_positions:
                    attn_to_parallel_all.append(weight)
                    step_parallel.append(weight)
                else:
                    attn_to_non_parallel_all.append(weight)
                    step_non_parallel.append(weight)

        if len(step_parallel) > 0 or len(step_non_parallel) > 0:
            attention_by_step.append({
                "step": logical_step,
                "seq_len": seq_len,
                "mean_to_parallel": float(np.mean(step_parallel)) if step_parallel else None,
                "mean_to_non_parallel": float(np.mean(step_non_parallel)) if step_non_parallel else None,
                "num_measurements_parallel": len(step_parallel),
                "num_measurements_non_parallel": len(step_non_parallel),
                "has_parallel_tokens": logical_step in parallel_steps
            })

    # Calculate overall metrics
    mean_parallel = float(np.mean(attn_to_parallel_all)) if attn_to_parallel_all else 0.0
    mean_non_parallel = float(np.mean(attn_to_non_parallel_all)) if attn_to_non_parallel_all else 0.0
    reduction = (mean_non_parallel - mean_parallel) / mean_non_parallel * 100 if mean_non_parallel > 0 else 0.0

    # Temporal analysis: early vs late
    mid_step = num_steps // 2
    early_steps = attention_by_step[:mid_step]
    late_steps = attention_by_step[mid_step:]

    early_parallel = [s["mean_to_parallel"] for s in early_steps if s["mean_to_parallel"] is not None]
    early_non_parallel = [s["mean_to_non_parallel"] for s in early_steps if s["mean_to_non_parallel"] is not None]
    late_parallel = [s["mean_to_parallel"] for s in late_steps if s["mean_to_parallel"] is not None]
    late_non_parallel = [s["mean_to_non_parallel"] for s in late_steps if s["mean_to_non_parallel"] is not None]

    early_reduction = (np.mean(early_non_parallel) - np.mean(early_parallel)) / np.mean(early_non_parallel) * 100 if len(early_parallel) > 0 and len(early_non_parallel) > 0 else None
    late_reduction = (np.mean(late_non_parallel) - np.mean(late_parallel)) / np.mean(late_non_parallel) * 100 if len(late_parallel) > 0 and len(late_non_parallel) > 0 else None

    all_generated_positions = set(range(prompt_length, seq_len))
    non_parallel_positions = all_generated_positions - all_parallel_positions

    return DetailedPromptAnalysis(
        prompt_name=prompt_dir.name,
        category=category,
        prompt_text=prompt_text,
        mean_attn_to_parallel=mean_parallel,
        mean_attn_to_non_parallel=mean_non_parallel,
        reduction_percentage=reduction,
        num_parallel_positions=len(all_parallel_positions),
        num_non_parallel_positions=len(non_parallel_positions),
        parallel_positions=sorted(list(all_parallel_positions)),
        attention_by_step=attention_by_step,
        attention_distribution_to_parallel=attn_to_parallel_all,
        attention_distribution_to_non_parallel=attn_to_non_parallel_all,
        early_vs_late_effect={
            "early_reduction": early_reduction,
            "late_reduction": late_reduction,
            "early_parallel_mean": float(np.mean(early_parallel)) if early_parallel else None,
            "early_non_parallel_mean": float(np.mean(early_non_parallel)) if early_non_parallel else None,
            "late_parallel_mean": float(np.mean(late_parallel)) if late_parallel else None,
            "late_non_parallel_mean": float(np.mean(late_non_parallel)) if late_non_parallel else None,
        }
    )


def compare_extreme_cases(analyses: List[DetailedPromptAnalysis]) -> Dict:
    """Compare prompts with extreme positive vs negative reductions."""

    # Sort by reduction percentage
    sorted_analyses = sorted(analyses, key=lambda a: a.reduction_percentage)

    # Get extremes
    most_negative = sorted_analyses[:3]  # Most negative = parallel gets MOST attention
    most_positive = sorted_analyses[-3:]  # Most positive = parallel gets LEAST attention

    print("\n" + "="*70)
    print("EXTREME CASES COMPARISON")
    print("="*70)

    print("\n🔴 Parallel Tokens Get MOST Attention (negative reduction):")
    for a in most_negative:
        print(f"\n  {a.prompt_name} ({a.reduction_percentage:+.1f}%):")
        print(f"    Prompt: \"{a.prompt_text[:60]}...\"")
        print(f"    Parallel positions: {a.parallel_positions}")
        print(f"    To parallel: {a.mean_attn_to_parallel:.6f}")
        print(f"    To non-parallel: {a.mean_attn_to_non_parallel:.6f}")
        if a.early_vs_late_effect["early_reduction"] is not None:
            print(f"    Early vs Late: {a.early_vs_late_effect['early_reduction']:+.1f}% → {a.early_vs_late_effect['late_reduction']:+.1f}%")

    print("\n🔵 Parallel Tokens Get LEAST Attention (positive reduction):")
    for a in most_positive:
        print(f"\n  {a.prompt_name} ({a.reduction_percentage:+.1f}%):")
        print(f"    Prompt: \"{a.prompt_text[:60]}...\"")
        print(f"    Parallel positions: {a.parallel_positions}")
        print(f"    To parallel: {a.mean_attn_to_parallel:.6f}")
        print(f"    To non-parallel: {a.mean_attn_to_non_parallel:.6f}")
        if a.early_vs_late_effect["early_reduction"] is not None:
            print(f"    Early vs Late: {a.early_vs_late_effect['early_reduction']:+.1f}% → {a.early_vs_late_effect['late_reduction']:+.1f}%")

    return {
        "most_negative": [asdict(a) for a in most_negative],
        "most_positive": [asdict(a) for a in most_positive]
    }


def analyze_distributional_patterns(analyses: List[DetailedPromptAnalysis]) -> Dict:
    """Analyze the distribution of attention weights."""

    print("\n" + "="*70)
    print("DISTRIBUTIONAL ANALYSIS")
    print("="*70)

    for analysis in analyses:
        parallel_dist = analysis.attention_distribution_to_parallel
        non_parallel_dist = analysis.attention_distribution_to_non_parallel

        print(f"\n{analysis.prompt_name} ({analysis.reduction_percentage:+.1f}%):")
        print(f"  Parallel attention:")
        print(f"    Min: {np.min(parallel_dist):.6f}, Max: {np.max(parallel_dist):.6f}")
        print(f"    Median: {np.median(parallel_dist):.6f}, Std: {np.std(parallel_dist):.6f}")
        print(f"  Non-parallel attention:")
        print(f"    Min: {np.min(non_parallel_dist):.6f}, Max: {np.max(non_parallel_dist):.6f}")
        print(f"    Median: {np.median(non_parallel_dist):.6f}, Std: {np.std(non_parallel_dist):.6f}")

    return {}


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Deep dive into attention pattern variability")
    parser.add_argument("--input-dir", type=Path, default=Path("experiments/results/multi_prompt_attention"))
    parser.add_argument("--output-file", type=Path, default=None)
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: {args.input_dir} does not exist")
        return 1

    # Prompt metadata
    prompts_info = {
        "narrative_1": ("narrative", "Once upon a time in a distant galaxy"),
        "narrative_2": ("narrative", "The old wizard slowly climbed the mountain"),
        "narrative_3": ("narrative", "Deep in the forest, a mysterious creature"),
        "factual_1": ("factual", "The capital of France is"),
        "factual_2": ("factual", "Photosynthesis is the process by which"),
        "factual_3": ("factual", "The largest planet in our solar system"),
        "technical_1": ("technical", "Machine learning algorithms can be classified into"),
        "technical_2": ("technical", "The algorithm complexity of quicksort is"),
        "conversational_1": ("conversational", "How are you doing today? I'm"),
        "conversational_2": ("conversational", "Hey, did you see that movie? It was"),
        "simple_1": ("simple", "The cat sat on the"),
        "simple_2": ("simple", "I went to the store and bought"),
        "complex_1": ("complex", "Despite the significant challenges facing modern society"),
    }

    print("="*70)
    print("DEEP DIVE: Attention Pattern Variability Analysis")
    print("="*70)
    print("\nGoal: Understand WHY attention patterns vary across prompts")
    print("="*70)

    # Analyze each prompt in detail
    analyses = []
    for prompt_name, (category, prompt_text) in prompts_info.items():
        prompt_dir = args.input_dir / prompt_name
        if not prompt_dir.exists():
            continue

        analysis = analyze_prompt_in_detail(prompt_dir, category, prompt_text)
        if analysis:
            analyses.append(analysis)
            print(f"✓ Analyzed {prompt_name}")

    if not analyses:
        print("\n❌ No data to analyze")
        return 1

    # Compare extreme cases
    extreme_comparison = compare_extreme_cases(analyses)

    # Distributional analysis
    distributional_analysis = analyze_distributional_patterns(analyses)

    # Save results
    if args.output_file:
        results = {
            "analyses": [asdict(a) for a in analyses],
            "extreme_cases": extreme_comparison,
            "distributional": distributional_analysis
        }

        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Saved detailed analysis to: {args.output_file}")

    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("\n[To be filled in based on patterns observed...]")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
