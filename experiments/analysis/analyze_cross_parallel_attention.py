#!/usr/bin/env python3
"""Mechanistic interpretability: Cross-parallel attention analysis.

The key insight: TEMPO captures attention with shape [layers, batch, heads, 1, all_keys].
The query dimension is always 1 because we process one logical step at a time.
But the KEYS include all previous tokens, including parallel tokens from previous steps.

This script analyzes how CURRENT tokens attend to PREVIOUS parallel tokens.
"""

import numpy as np
import json
from pathlib import Path
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

def analyze_attention_to_parallel_siblings(exp_data: Dict, analyzing_step: int) -> None:
    """Analyze how tokens at analyzing_step attend to parallel tokens from PREVIOUS steps."""
    print(f"\n" + "="*80)
    print(f"ATTENTION TO PREVIOUS PARALLEL TOKENS - Step {analyzing_step}")
    print("="*80)

    attention_data = exp_data["attention_data"]
    parallel_steps = exp_data["parallel_sets"].get("parallel_sets", [])
    isolated = exp_data["metadata"]["config"]["allow_intraset_token_visibility"] == False

    attn_key = f"step_{analyzing_step}_attention"
    if attn_key not in attention_data:
        print(f"  ⚠ No attention data for step {analyzing_step}")
        return

    attn_weights = attention_data[attn_key]  # [layers, batch, heads, 1, keys]
    print(f"Mode: {'ISOLATED' if isolated else 'VISIBLE'}")
    print(f"Attention shape: {attn_weights.shape}")

    # Find all parallel tokens in previous steps
    previous_parallel_positions = {}
    for ps in parallel_steps:
        if ps["step"] < analyzing_step:
            step = ps["step"]
            positions = ps["positions"]
            tokens = ps["tokens"]
            previous_parallel_positions[step] = {
                "positions": positions,
                "tokens": tokens,
                "count": ps["count"]
            }

    if not previous_parallel_positions:
        print(f"  ℹ No previous parallel tokens to analyze")
        return

    print(f"\nPrevious parallel steps found: {sorted(previous_parallel_positions.keys())}")

    # Average attention over layers and batch
    avg_attn = attn_weights.mean(axis=(0, 1))[0]  # [heads, keys]
    num_heads = avg_attn.shape[0]
    num_keys = avg_attn.shape[1]

    print(f"  Attention heads: {num_heads}")
    print(f"  Total key positions: {num_keys}")

    # Analyze attention to each previous parallel step
    for prev_step in sorted(previous_parallel_positions.keys()):
        info = previous_parallel_positions[prev_step]
        positions = info["positions"]
        tokens = info["tokens"]
        count = info["count"]

        print(f"\n  Step {prev_step} parallel tokens (positions {positions}):")

        # Get attention to these specific positions
        # Note: positions are physical token indices
        parallel_attns = []
        for pos in positions:
            if pos < num_keys:
                attn_to_pos = avg_attn[:, pos]  # [heads]
                mean_attn = attn_to_pos.mean()
                max_attn = attn_to_pos.max()
                parallel_attns.append(mean_attn)
                print(f"    Position {pos} (token {tokens[positions.index(pos)]}): "
                      f"mean={mean_attn:.6f}, max={max_attn:.6f}")

        if parallel_attns:
            overall_mean = np.mean(parallel_attns)
            print(f"    Overall mean attention to step {prev_step} parallel tokens: {overall_mean:.6f}")

    # Compare attention to parallel vs non-parallel tokens
    all_parallel_positions = []
    for info in previous_parallel_positions.values():
        all_parallel_positions.extend(info["positions"])

    parallel_attns = [avg_attn[:, pos].mean() for pos in all_parallel_positions if pos < num_keys]
    non_parallel_positions = [i for i in range(num_keys) if i not in all_parallel_positions]
    non_parallel_attns = [avg_attn[:, pos].mean() for pos in non_parallel_positions]

    if parallel_attns and non_parallel_attns:
        mean_parallel = np.mean(parallel_attns)
        mean_non_parallel = np.mean(non_parallel_attns)
        print(f"\n  COMPARISON:")
        print(f"    Mean attention to previous parallel tokens:     {mean_parallel:.6f}")
        print(f"    Mean attention to previous non-parallel tokens: {mean_non_parallel:.6f}")
        print(f"    Ratio (parallel / non-parallel): {mean_parallel / mean_non_parallel:.3f}x")

        if abs(mean_parallel - mean_non_parallel) < 0.001:
            print(f"    → Parallel and non-parallel tokens receive similar attention")
        elif mean_parallel > mean_non_parallel:
            print(f"    → Parallel tokens receive MORE attention")
        else:
            print(f"    → Parallel tokens receive LESS attention")

def analyze_within_step_attention(exp_data: Dict, step: int) -> None:
    """Analyze attention WITHIN a parallel step (sibling attention).

    This requires looking at a step where we process multiple parallel tokens.
    Unfortunately, current capture shows queries=1, meaning we process one token at a time.
    """
    print(f"\n" + "="*80)
    print(f"WITHIN-STEP SIBLING ATTENTION - Step {step}")
    print("="*80)

    parallel_steps = exp_data["parallel_sets"].get("parallel_sets", [])
    step_data = None
    for ps in parallel_steps:
        if ps["step"] == step:
            step_data = ps
            break

    if not step_data:
        print(f"  ℹ No parallel tokens at step {step}")
        return

    print(f"  Parallel tokens at step {step}: {step_data['count']}")
    print(f"  Positions: {step_data['positions']}")
    print(f"  Token IDs: {step_data['tokens']}")

    # Check if we captured attention FROM these tokens
    attention_data = exp_data["attention_data"]
    positions = step_data["positions"]

    # Look for attention captured at these positions
    found_attention = False
    for pos in positions:
        # Try to find attention data where this position is the query
        for key in attention_data.keys():
            if f"step_{step}_" in key and "positions" in key:
                captured_positions = attention_data[key]
                if pos in captured_positions:
                    found_attention = True
                    break

    if not found_attention:
        print(f"  ℹ Attention not captured FROM parallel tokens (queries=1 limitation)")
        print(f"  NOTE: Current implementation captures attention TO tokens, not FROM them")
    else:
        print(f"  ✓ Found attention FROM parallel tokens")

def main():
    print("╔" + "="*78 + "╗")
    print("║" + "CROSS-PARALLEL ATTENTION ANALYSIS".center(78) + "║")
    print("╚" + "="*78 + "╝")

    results_dir = Path("experiments/results")
    exp_isolated = load_experiment(results_dir / "exp1_isolated")
    exp_visible = load_experiment(results_dir / "exp1_visible")

    print("\nAnalyzing ISOLATED mode:")
    print("="*80)

    # Analyze how later tokens attend to previous parallel tokens
    for step in [3, 4, 5, 6]:
        analyze_attention_to_parallel_siblings(exp_isolated, step)

    print("\n\nAnalyzing VISIBLE mode:")
    print("="*80)

    for step in [3, 4, 5, 6]:
        analyze_attention_to_parallel_siblings(exp_visible, step)

    # Try to analyze within-step attention (will likely show limitation)
    print("\n\nWITHIN-STEP ANALYSIS:")
    print("="*80)
    analyze_within_step_attention(exp_isolated, 2)
    analyze_within_step_attention(exp_isolated, 3)

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print("""
The current attention capture shows:
1. How CURRENT tokens attend TO PREVIOUS parallel tokens (measurable)
2. Whether parallel tokens from previous steps are treated differently
3. Comparison between isolated and visible modes

Limitation: queries=1 means we can't directly see attention BETWEEN
parallel tokens at the SAME step. That would require queries > 1.

However, we CAN infer cross-parallel behavior by comparing:
- Attention to previous parallel tokens in isolated vs visible mode
- Whether the isolation mechanism affects attention to prior parallel tokens
    """)

if __name__ == "__main__":
    main()
