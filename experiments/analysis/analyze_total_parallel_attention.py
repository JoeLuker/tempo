#!/usr/bin/env python3
"""Analyze total attention to parallel token sets vs individual non-parallel tokens.

Key question: Does the sum of attention to all parallel tokens at a logical position
equal the attention to a single non-parallel token?

This tests the hypothesis that the model distributes attention across parallel
alternatives such that the total information extracted from the parallel set
is comparable to a single deterministic token.
"""

import numpy as np
from pathlib import Path
import json
from scipy.stats import ttest_ind, ttest_rel
from typing import Dict, List

def load_experiment(exp_dir: Path) -> Dict:
    """Load experiment data."""
    with open(exp_dir / "experiment_metadata.json") as f:
        metadata = json.load(f)
    with open(exp_dir / "parallel_sets.json") as f:
        parallel_sets = json.load(f)
    attention_data = np.load(exp_dir / "attention_weights.npz")

    return {
        "metadata": metadata,
        "parallel_sets": parallel_sets,
        "attention_data": attention_data
    }

def analyze_total_parallel_attention(exp_data: Dict, step: int) -> Dict:
    """Analyze total attention to parallel token SETS vs individual tokens."""
    attention_data = exp_data["attention_data"]
    parallel_steps = exp_data["parallel_sets"].get("parallel_sets", [])

    attn_key = f"step_{step}_attention"
    if attn_key not in attention_data:
        return None

    attn_weights = attention_data[attn_key]  # [layers, batch, heads, 1, keys]
    attn = attn_weights[:, 0, :, 0, :]  # [layers, heads, keys]

    num_layers = attn.shape[0]
    num_heads = attn.shape[1]
    num_keys = attn.shape[2]

    # Find all parallel steps in the context (before this step)
    parallel_sets_in_context = []
    for ps in parallel_steps:
        if ps["step"] < step:
            parallel_sets_in_context.append({
                "step": ps["step"],
                "positions": ps["positions"],
                "tokens": ps["tokens"],
                "count": ps["count"]
            })

    if not parallel_sets_in_context:
        return None

    # Collect total attention to each parallel SET
    parallel_set_totals = []

    # Collect attention to individual non-parallel tokens
    all_parallel_positions = []
    for ps in parallel_sets_in_context:
        all_parallel_positions.extend(ps["positions"])

    non_parallel_positions = [i for i in range(num_keys) if i not in all_parallel_positions]
    individual_non_parallel_attns = []

    # For each layer and head
    for layer in range(num_layers):
        for head in range(num_heads):
            # Sum attention to each parallel SET
            for ps in parallel_sets_in_context:
                positions = ps["positions"]
                total_attn_to_set = 0.0
                for pos in positions:
                    if pos < num_keys:
                        total_attn_to_set += attn[layer, head, pos]
                parallel_set_totals.append(total_attn_to_set)

            # Individual non-parallel token attention
            for pos in non_parallel_positions:
                if pos < num_keys:
                    individual_non_parallel_attns.append(attn[layer, head, pos])

    return {
        "step": step,
        "parallel_set_totals": np.array(parallel_set_totals),
        "individual_non_parallel": np.array(individual_non_parallel_attns),
        "num_parallel_sets": len(parallel_sets_in_context),
        "num_non_parallel_positions": len(non_parallel_positions),
        "num_layers": num_layers,
        "num_heads": num_heads
    }

def main():
    print("╔" + "="*78 + "╗")
    print("║" + "TOTAL PARALLEL SET ATTENTION ANALYSIS".center(78) + "║")
    print("╚" + "="*78 + "╝")

    print("\nResearch Question:")
    print("  Does SUM(attention to all parallel tokens at position P)")
    print("  equal attention to a single non-parallel token?")
    print()
    print("Hypothesis: If parallel tokens contain redundant information,")
    print("the model might distribute attention across them such that")
    print("TOTAL attention ≈ attention to one non-parallel token.")

    results_dir = Path("experiments/results")

    # Analyze both modes
    for mode_name, exp_dir in [("ISOLATED", "exp1_isolated"), ("VISIBLE", "exp1_visible")]:
        print(f"\n{'='*80}")
        print(f"{mode_name} MODE")
        print(f"{'='*80}")

        exp_data = load_experiment(results_dir / exp_dir)

        all_set_totals = []
        all_individual_attns = []

        for step in [3, 4, 5, 6]:
            result = analyze_total_parallel_attention(exp_data, step)

            if result is None:
                continue

            set_totals = result["parallel_set_totals"]
            individual_attns = result["individual_non_parallel"]

            all_set_totals.extend(set_totals)
            all_individual_attns.extend(individual_attns)

            print(f"\nStep {step}:")
            print(f"  Parallel sets in context: {result['num_parallel_sets']}")
            print(f"  Sample sizes: {len(set_totals)} set totals, {len(individual_attns)} individual tokens")

            print(f"\n  TOTAL attention to parallel SETS:")
            print(f"    Mean:   {set_totals.mean():.6f}")
            print(f"    Median: {np.median(set_totals):.6f}")
            print(f"    Std:    {set_totals.std():.6f}")
            print(f"    Min:    {set_totals.min():.6f}")
            print(f"    Max:    {set_totals.max():.6f}")

            print(f"\n  INDIVIDUAL non-parallel token attention:")
            print(f"    Mean:   {individual_attns.mean():.6f}")
            print(f"    Median: {np.median(individual_attns):.6f}")
            print(f"    Std:    {individual_attns.std():.6f}")
            print(f"    Min:    {individual_attns.min():.6f}")
            print(f"    Max:    {individual_attns.max():.6f}")

            # Comparison
            ratio = set_totals.mean() / individual_attns.mean()
            print(f"\n  Ratio (set total / individual): {ratio:.4f}")

            if ratio < 0.9:
                print(f"    → Parallel sets receive LESS total attention than individual tokens")
            elif ratio > 1.1:
                print(f"    → Parallel sets receive MORE total attention than individual tokens")
            else:
                print(f"    → Parallel sets receive SIMILAR total attention to individual tokens")

            # Statistical test
            t_stat, p_value = ttest_ind(set_totals, individual_attns)
            print(f"\n  Statistical test (t-test):")
            print(f"    t-statistic: {t_stat:.4f}")
            print(f"    p-value: {p_value:.2e}")
            if p_value < 0.001:
                print(f"    → *** HIGHLY SIGNIFICANT difference")
            elif p_value < 0.01:
                print(f"    → ** VERY SIGNIFICANT difference")
            elif p_value < 0.05:
                print(f"    → * SIGNIFICANT difference")
            else:
                print(f"    → NOT SIGNIFICANT (sets ≈ individual tokens)")

        # Aggregate across all steps
        all_set_totals_arr = np.array(all_set_totals)
        all_individual_attns_arr = np.array(all_individual_attns)

        print(f"\n{'='*80}")
        print(f"AGGREGATE STATISTICS - {mode_name}")
        print(f"{'='*80}")

        print(f"\nTotal parallel set attention:")
        print(f"  Mean:   {all_set_totals_arr.mean():.6f}")
        print(f"  Median: {np.median(all_set_totals_arr):.6f}")
        print(f"  Std:    {all_set_totals_arr.std():.6f}")
        print(f"  95% CI: [{all_set_totals_arr.mean() - 1.96*all_set_totals_arr.std()/np.sqrt(len(all_set_totals_arr)):.6f}, "
              f"{all_set_totals_arr.mean() + 1.96*all_set_totals_arr.std()/np.sqrt(len(all_set_totals_arr)):.6f}]")

        print(f"\nIndividual non-parallel token attention:")
        print(f"  Mean:   {all_individual_attns_arr.mean():.6f}")
        print(f"  Median: {np.median(all_individual_attns_arr):.6f}")
        print(f"  Std:    {all_individual_attns_arr.std():.6f}")
        print(f"  95% CI: [{all_individual_attns_arr.mean() - 1.96*all_individual_attns_arr.std()/np.sqrt(len(all_individual_attns_arr)):.6f}, "
              f"{all_individual_attns_arr.mean() + 1.96*all_individual_attns_arr.std()/np.sqrt(len(all_individual_attns_arr)):.6f}]")

        overall_ratio = all_set_totals_arr.mean() / all_individual_attns_arr.mean()
        print(f"\nOverall ratio: {overall_ratio:.4f}")
        print(f"  Interpretation: Parallel sets receive {overall_ratio*100:.1f}% of individual token attention")

        # Statistical significance
        t_stat, p_value = ttest_ind(all_set_totals_arr, all_individual_attns_arr)
        print(f"\nOverall statistical test:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.2e}")

    print(f"\n{'='*80}")
    print(f"INTERPRETATION")
    print(f"{'='*80}")

    print("""
Three possible outcomes:

1. Ratio ≈ 1.0: COMPENSATION hypothesis confirmed
   - Model distributes attention across parallel tokens
   - Total attention to set ≈ attention to one token
   - Parallel tokens collectively get "one token's worth" of attention

2. Ratio < 1.0: DOWN-WEIGHTING at SET level
   - Even summing all parallel tokens doesn't reach single token attention
   - Parallel sets are systematically under-attended
   - Suggests position-based penalty, not just distribution

3. Ratio > 1.0: OVER-WEIGHTING at SET level
   - Parallel sets receive MORE total attention
   - Model may be uncertain and attending to all alternatives
   - Could indicate exploration behavior
    """)

if __name__ == "__main__":
    main()
