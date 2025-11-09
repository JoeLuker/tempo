#!/usr/bin/env python3
"""Analyze how total attention to parallel sets changes with set size.

Key questions:
1. Does total attention scale with set size? (linear, sublinear, constant?)
2. Is the penalty fixed per position or per token?
3. What's the marginal attention benefit of each additional parallel token?

Three hypotheses:
A. Fixed penalty per position: Total attention same regardless of set size
B. Linear scaling: Total attention ∝ set size (each token adds equal attention)
C. Sublinear scaling: Total attention grows but with diminishing returns
"""

import numpy as np
from pathlib import Path
import json
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

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

def analyze_attention_by_set_size(exp_data: Dict, step: int) -> List[Dict]:
    """Analyze attention to parallel sets grouped by set size."""
    attention_data = exp_data["attention_data"]
    parallel_steps = exp_data["parallel_sets"].get("parallel_sets", [])

    attn_key = f"step_{step}_attention"
    if attn_key not in attention_data:
        return []

    attn_weights = attention_data[attn_key]  # [layers, batch, heads, 1, keys]
    attn = attn_weights[:, 0, :, 0, :]  # [layers, heads, keys]

    num_layers = attn.shape[0]
    num_heads = attn.shape[1]
    num_keys = attn.shape[2]

    # Find all parallel sets in context (before this step)
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
        return []

    # Group by set size and collect total attention
    results_by_size = {}

    for layer in range(num_layers):
        for head in range(num_heads):
            for ps in parallel_sets_in_context:
                size = ps["count"]
                positions = ps["positions"]

                # Sum attention to this set
                total_attn = 0.0
                for pos in positions:
                    if pos < num_keys:
                        total_attn += attn[layer, head, pos]

                if size not in results_by_size:
                    results_by_size[size] = []
                results_by_size[size].append(total_attn)

    return results_by_size

def main():
    print("╔" + "="*78 + "╗")
    print("║" + "ATTENTION SCALING BY PARALLEL SET SIZE".center(78) + "║")
    print("╚" + "="*78 + "╝")

    print("\nResearch Questions:")
    print("  1. Does total attention scale with set size?")
    print("  2. Is the penalty fixed per position or per token?")
    print("  3. What's the marginal attention benefit of adding a parallel token?")
    print()
    print("Hypotheses:")
    print("  A. FIXED: Total attention same regardless of set size")
    print("  B. LINEAR: Total attention ∝ set size")
    print("  C. SUBLINEAR: Total attention grows but with diminishing returns")

    results_dir = Path("experiments/results")

    # Analyze both modes
    for mode_name, exp_dir in [("ISOLATED", "exp1_isolated"), ("VISIBLE", "exp1_visible")]:
        print(f"\n{'='*80}")
        print(f"{mode_name} MODE")
        print(f"{'='*80}")

        exp_data = load_experiment(results_dir / exp_dir)

        # Aggregate across all steps
        all_results_by_size = {}

        for step in [3, 4, 5, 6]:
            results_by_size = analyze_attention_by_set_size(exp_data, step)

            for size, attns in results_by_size.items():
                if size not in all_results_by_size:
                    all_results_by_size[size] = []
                all_results_by_size[size].extend(attns)

        # Analyze results
        if not all_results_by_size:
            print("  No parallel sets found")
            continue

        sizes = sorted(all_results_by_size.keys())

        print(f"\nParallel set sizes found: {sizes}")
        print(f"\nTotal attention by set size:")

        size_means = []
        size_stds = []
        size_counts = []

        for size in sizes:
            attns = np.array(all_results_by_size[size])
            mean_attn = attns.mean()
            std_attn = attns.std()
            n = len(attns)

            size_means.append(mean_attn)
            size_stds.append(std_attn)
            size_counts.append(n)

            print(f"\n  Set size {size}:")
            print(f"    Samples: n={n}")
            print(f"    Mean total attention: {mean_attn:.6f}")
            print(f"    Std: {std_attn:.6f}")
            print(f"    Per-token attention: {mean_attn/size:.6f} (total / size)")
            print(f"    95% CI: [{mean_attn - 1.96*std_attn/np.sqrt(n):.6f}, "
                  f"{mean_attn + 1.96*std_attn/np.sqrt(n):.6f}]")

        # Analyze scaling
        print(f"\n{'='*80}")
        print(f"SCALING ANALYSIS - {mode_name}")
        print(f"{'='*80}")

        sizes_arr = np.array(sizes)
        means_arr = np.array(size_means)

        # Linear correlation
        if len(sizes) >= 2:
            pearson_r, pearson_p = pearsonr(sizes_arr, means_arr)
            spearman_r, spearman_p = spearmanr(sizes_arr, means_arr)

            print(f"\nCorrelation between set size and total attention:")
            print(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.4f})")
            print(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.4f})")

            if pearson_p < 0.05:
                if pearson_r > 0:
                    print(f"  → SIGNIFICANT positive correlation (larger sets get more attention)")
                else:
                    print(f"  → SIGNIFICANT negative correlation (larger sets get less attention)")
            else:
                print(f"  → NO SIGNIFICANT correlation (set size doesn't affect total attention)")

        # Compute per-token attention (total / size)
        per_token_means = means_arr / sizes_arr

        print(f"\nPer-token attention (total attention / set size):")
        for i, size in enumerate(sizes):
            print(f"  Size {size}: {per_token_means[i]:.6f}")

        if len(sizes) >= 2:
            # Test if per-token attention decreases with size
            pearson_r_per_token, pearson_p_per_token = pearsonr(sizes_arr, per_token_means)

            print(f"\nCorrelation between set size and PER-TOKEN attention:")
            print(f"  Pearson r: {pearson_r_per_token:.4f} (p={pearson_p_per_token:.4f})")

            if pearson_p_per_token < 0.05:
                if pearson_r_per_token < 0:
                    print(f"  → DIMINISHING RETURNS: Each token gets less attention in larger sets")
                else:
                    print(f"  → INCREASING RETURNS: Each token gets more attention in larger sets")
            else:
                print(f"  → CONSTANT: Each token gets same attention regardless of set size")

        # Marginal attention benefit
        if len(sizes) >= 2:
            print(f"\nMarginal attention benefit (attention gained per additional token):")
            for i in range(1, len(sizes)):
                delta_size = sizes[i] - sizes[i-1]
                delta_attn = means_arr[i] - means_arr[i-1]
                marginal_benefit = delta_attn / delta_size if delta_size > 0 else 0

                print(f"  Size {sizes[i-1]} → {sizes[i]}: +{marginal_benefit:.6f} per token")

        # Hypothesis testing
        print(f"\n{'='*80}")
        print(f"HYPOTHESIS TESTING - {mode_name}")
        print(f"{'='*80}")

        if len(sizes) >= 2:
            # Check if total attention is roughly constant (Hypothesis A)
            cv = means_arr.std() / means_arr.mean()
            print(f"\nHypothesis A: FIXED penalty per position")
            print(f"  Coefficient of variation in total attention: {cv:.4f}")
            if cv < 0.2:
                print(f"  → SUPPORTED: Total attention is relatively constant (~{means_arr.mean():.4f})")
            else:
                print(f"  → NOT SUPPORTED: Total attention varies significantly")

            # Check if scaling is linear (Hypothesis B)
            print(f"\nHypothesis B: LINEAR scaling")
            if pearson_p < 0.05 and 0.8 < pearson_r < 1.0:
                print(f"  → SUPPORTED: Strong positive correlation (r={pearson_r:.3f})")
            else:
                print(f"  → NOT SUPPORTED: Correlation is {pearson_r:.3f}")

            # Check if per-token attention decreases (Hypothesis C)
            print(f"\nHypothesis C: SUBLINEAR scaling (diminishing returns)")
            if pearson_p_per_token < 0.05 and pearson_r_per_token < -0.5:
                print(f"  → SUPPORTED: Per-token attention decreases with size (r={pearson_r_per_token:.3f})")
            elif pearson_p < 0.05 and pearson_r > 0 and pearson_r_per_token < 0:
                print(f"  → PARTIALLY SUPPORTED: Total attention increases but per-token decreases")
            else:
                print(f"  → NOT SUPPORTED")

        # Create visualization
        if len(sizes) >= 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Plot 1: Total attention by set size
            ax1.errorbar(sizes, size_means,
                        yerr=[1.96*s/np.sqrt(n) for s, n in zip(size_stds, size_counts)],
                        marker='o', capsize=5, capthick=2, markersize=8)
            ax1.set_xlabel('Parallel Set Size', fontsize=12)
            ax1.set_ylabel('Mean Total Attention to Set', fontsize=12)
            ax1.set_title(f'Total Attention vs Set Size\n({mode_name} mode)', fontsize=14)
            ax1.grid(True, alpha=0.3)

            # Add reference line for linear scaling
            if len(sizes) >= 2:
                # Fit line from first point
                baseline_per_token = size_means[0] / sizes[0]
                linear_prediction = [baseline_per_token * s for s in sizes]
                ax1.plot(sizes, linear_prediction, '--', color='red', alpha=0.5,
                        label=f'Linear (baseline: {baseline_per_token:.4f}/token)')
                ax1.legend()

            # Plot 2: Per-token attention by set size
            per_token_errors = [1.96*s/(sizes[i]*np.sqrt(n))
                               for i, (s, n) in enumerate(zip(size_stds, size_counts))]
            ax2.errorbar(sizes, per_token_means, yerr=per_token_errors,
                        marker='o', capsize=5, capthick=2, markersize=8, color='orange')
            ax2.set_xlabel('Parallel Set Size', fontsize=12)
            ax2.set_ylabel('Mean Per-Token Attention', fontsize=12)
            ax2.set_title(f'Per-Token Attention vs Set Size\n({mode_name} mode)', fontsize=14)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            output_file = f'attention_by_set_size_{mode_name.lower()}.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\n  Visualization saved to: {output_file}")
            plt.close()

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print("""
Key findings will reveal:

1. If total attention is CONSTANT across set sizes:
   → Fixed penalty per position (doesn't matter how many alternatives)
   → RoPE position sharing creates position-level black hole

2. If total attention INCREASES linearly with set size:
   → Each token adds equal attention
   → Penalty is per-token, not per-position

3. If per-token attention DECREASES with set size:
   → Diminishing returns (more tokens compete for same total attention)
   → Position has fixed "attention budget" split across alternatives
    """)

if __name__ == "__main__":
    main()
