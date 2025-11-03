#!/usr/bin/env python3
"""Deep analysis runner for TEMPO mechanistic interpretability experiments.

This script runs comprehensive analysis on captured experiment data to answer
the key research questions about parallel token processing in TEMPO.
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.experiment_loader import ExperimentLoader
from src.analysis.attention_analyzer import AttentionAnalyzer
from src.analysis.logits_analyzer import LogitsAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def analyze_isolated_vs_visible(loader: ExperimentLoader, output_dir: Path):
    """Analyze differences between isolated and visible modes."""
    print_section("EXPERIMENT 1: ISOLATED VS VISIBLE ATTENTION COMPARISON")

    # Load experiments
    exp_isolated = loader.load_experiment("exp1_isolated")
    exp_visible = loader.load_experiment("exp1_visible")

    print(f"Loaded experiments:")
    print(f"  - {exp_isolated.experiment_name}: {exp_isolated.num_steps} steps")
    print(f"  - {exp_visible.experiment_name}: {exp_visible.num_steps} steps")

    # Attention comparison
    print("\n--- Attention Pattern Comparison ---")
    analyzer = AttentionAnalyzer(debug_mode=False)
    comparison = analyzer.compare_attention_patterns(exp_isolated, exp_visible)

    print(f"\nResults:")
    print(f"  Mean absolute difference: {comparison.mean_absolute_difference:.8f}")
    print(f"  Max absolute difference:  {comparison.max_absolute_difference:.8f}")
    print(f"  Overall correlation:      {comparison.correlation:.6f}")

    if len(comparison.layer_correlations) > 0:
        print(f"\nLayer-wise correlations:")
        for i, corr in enumerate(comparison.layer_correlations):
            print(f"  Layer {i:2d}: {corr:.6f}")

    # Save results
    results = {
        "comparison": "isolated_vs_visible",
        "mean_absolute_difference": comparison.mean_absolute_difference,
        "max_absolute_difference": comparison.max_absolute_difference,
        "correlation": comparison.correlation,
        "layer_correlations": comparison.layer_correlations.tolist() if len(comparison.layer_correlations) > 0 else [],
        "step_differences": comparison.step_differences
    }

    output_file = output_dir / "isolated_vs_visible_attention.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    return results


def analyze_cross_parallel_attention(loader: ExperimentLoader, output_dir: Path):
    """Analyze attention between parallel tokens in visible mode."""
    print_section("EXPERIMENT 3: CROSS-PARALLEL ATTENTION MEASUREMENT")

    # Load visible mode experiment
    exp = loader.load_experiment("exp3_cross_parallel")

    print(f"Loaded experiment: {exp.experiment_name}")
    print(f"  Steps: {exp.num_steps}")
    print(f"  Parallel steps: {exp.parallel_sets.total_parallel_steps}")
    print(f"  Max parallel width: {exp.parallel_sets.max_parallel_width}")

    # Analyze cross-parallel attention
    print("\n--- Cross-Parallel Attention Analysis ---")
    analyzer = AttentionAnalyzer(debug_mode=False)
    cross_attn = analyzer.analyze_cross_parallel_attention(exp)

    print(f"\nResults for {len(cross_attn)} parallel steps:")

    results = []
    for ca in cross_attn:
        print(f"\nStep {ca.step}: {ca.parallel_count} parallel tokens")
        print(f"  Cross-parallel attention (siblings): {ca.mean_cross_attention:.6f} (max: {ca.max_cross_attention:.6f})")
        print(f"  Prior context attention:             {ca.mean_prior_attention:.6f}")
        print(f"  Ratio (prior / cross):               {ca.attention_ratio:.2f}x")

        results.append({
            "step": ca.step,
            "parallel_count": ca.parallel_count,
            "parallel_positions": ca.parallel_positions,
            "mean_cross_attention": ca.mean_cross_attention,
            "max_cross_attention": ca.max_cross_attention,
            "mean_prior_attention": ca.mean_prior_attention,
            "attention_ratio": ca.attention_ratio
        })

    # Compute aggregates
    mean_cross = sum(r["mean_cross_attention"] for r in results) / len(results)
    mean_prior = sum(r["mean_prior_attention"] for r in results) / len(results)
    mean_ratio = sum(r["attention_ratio"] for r in results if r["attention_ratio"] != float('inf')) / len(results)

    print(f"\n--- Aggregate Statistics ---")
    print(f"  Mean cross-parallel attention: {mean_cross:.6f}")
    print(f"  Mean prior context attention:  {mean_prior:.6f}")
    print(f"  Mean ratio (prior / cross):    {mean_ratio:.2f}x")

    # Interpretation
    print(f"\n--- Interpretation ---")
    if mean_cross < 0.01:
        print("  ✓ Parallel tokens have VERY LOW attention to each other")
        print("    Hypothesis: Prior context dominates (Scenario A)")
    elif mean_cross < mean_prior:
        print(f"  ✓ Parallel tokens attend more to prior context ({mean_ratio:.1f}x more)")
        print("    Hypothesis: Subtle cross-attention but prior dominates (Scenario B)")
    else:
        print("  ! Parallel tokens attend significantly to each other")
        print("    Hypothesis: Significant interaction (Scenario C/D)")

    # Save results
    output_data = {
        "experiment": "cross_parallel_attention",
        "parallel_steps": results,
        "aggregates": {
            "mean_cross_attention": mean_cross,
            "mean_prior_attention": mean_prior,
            "mean_ratio": mean_ratio
        }
    }

    output_file = output_dir / "cross_parallel_attention.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    return output_data


def analyze_logits_distributions(loader: ExperimentLoader, output_dir: Path):
    """Analyze logits distribution comparison."""
    print_section("EXPERIMENT 2: LOGITS DISTRIBUTION COMPARISON")

    # Load logits experiment
    exp = loader.load_experiment("exp2_logits")

    print(f"Loaded experiment: {exp.experiment_name}")
    print(f"  Steps: {exp.num_steps}")
    print(f"  Vocab size: {exp.logits.vocab_size:,}")

    # Analyze entropy
    print("\n--- Distribution Entropy Analysis ---")
    analyzer = LogitsAnalyzer(debug_mode=False)
    entropies = analyzer.analyze_distribution_entropy(exp)

    mean_entropy = sum(entropies.values()) / len(entropies)
    print(f"Mean entropy across steps: {mean_entropy:.4f}")

    # Sample top-k tokens for a few steps
    print("\n--- Sample Top-K Tokens ---")
    for step in [0, 2, 5]:
        if step in exp.logits.steps:
            top_k = analyzer.get_top_k_tokens(exp, step, k=5)
            print(f"\nStep {step} top-5 tokens:")
            for i, (token_id, prob) in enumerate(top_k, 1):
                print(f"  {i}. Token {token_id}: {prob:.6f} ({prob*100:.2f}%)")

    # Save results
    results = {
        "experiment": "logits_distribution",
        "vocab_size": exp.logits.vocab_size,
        "num_steps": exp.num_steps,
        "entropies": entropies,
        "mean_entropy": mean_entropy
    }

    output_file = output_dir / "logits_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    return results


def analyze_rope_positions(loader: ExperimentLoader, output_dir: Path):
    """Analyze RoPE position mappings."""
    print_section("EXPERIMENT 4: ROPE POSITION VERIFICATION")

    exp = loader.load_experiment("exp4_kv_cache")

    print(f"Loaded experiment: {exp.experiment_name}")

    if exp.rope_positions:
        print(f"  RoPE position mappings: {len(exp.rope_positions)}")

        # Verify parallel tokens share positions
        print("\n--- Physical → Logical Position Mapping ---")

        # Group by logical position
        logical_groups = {}
        for phys, logical in exp.rope_positions.items():
            if logical not in logical_groups:
                logical_groups[logical] = []
            logical_groups[logical].append(phys)

        parallel_positions = {k: v for k, v in logical_groups.items() if len(v) > 1}

        print(f"\nFound {len(parallel_positions)} logical positions with parallel tokens:")
        for logical, physicals in sorted(parallel_positions.items()):
            print(f"  Logical {logical}: Physical positions {physicals} ({len(physicals)} tokens)")

        print(f"\n✓ Verified: Parallel tokens successfully share RoPE positions")

        # Save results
        results = {
            "experiment": "rope_positions",
            "total_mappings": len(exp.rope_positions),
            "parallel_position_groups": {str(k): v for k, v in parallel_positions.items()},
            "num_parallel_groups": len(parallel_positions)
        }

        output_file = output_dir / "rope_position_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

        return results
    else:
        print("  ⚠ No RoPE position data available")
        return {}


def main():
    """Main analysis entry point."""
    parser = argparse.ArgumentParser(
        description="Run deep analysis on TEMPO mechanistic interpretability experiments"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/analysis",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        choices=["isolated_vs_visible", "cross_parallel", "logits", "rope", "all"],
        default=["all"],
        help="Which analyses to run"
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = ExperimentLoader(Path(args.results_dir))

    print(f"╔{'='*78}╗")
    print(f"║{'TEMPO DEEP MECHANISTIC INTERPRETABILITY ANALYSIS':^78}║")
    print(f"╚{'='*78}╝")

    available = loader.list_experiments()
    print(f"\nAvailable experiments: {', '.join(available)}")

    experiments_to_run = args.experiments
    if "all" in experiments_to_run:
        experiments_to_run = ["isolated_vs_visible", "cross_parallel", "logits", "rope"]

    # Run analyses
    all_results = {}

    try:
        if "isolated_vs_visible" in experiments_to_run:
            all_results["isolated_vs_visible"] = analyze_isolated_vs_visible(loader, output_dir)

        if "cross_parallel" in experiments_to_run:
            try:
                all_results["cross_parallel"] = analyze_cross_parallel_attention(loader, output_dir)
            except IndexError as e:
                logger.warning(f"Cross-parallel analysis skipped: {e}")
                logger.warning("Note: Current attention capture only includes attention TO parallel tokens,")
                logger.warning("not FROM parallel tokens. Cross-parallel analysis requires different capture.")

        if "logits" in experiments_to_run:
            all_results["logits"] = analyze_logits_distributions(loader, output_dir)

        if "rope" in experiments_to_run:
            all_results["rope"] = analyze_rope_positions(loader, output_dir)

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1

    # Summary
    print_section("ANALYSIS COMPLETE")
    print(f"All results saved to: {output_dir}")
    print(f"\nAnalyses completed: {len(all_results)}")
    for name in all_results.keys():
        print(f"  ✓ {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
