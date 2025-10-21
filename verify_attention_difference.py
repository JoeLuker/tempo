#!/usr/bin/env python3
"""
Verify that attention patterns are actually different between isolated and visible modes.

This script checks:
1. Are the attention matrices different between modes?
2. In visible mode, is there non-zero cross-parallel attention?
3. In isolated mode, is cross-parallel attention zero?
"""

import numpy as np
import json
from pathlib import Path


def load_experiment_data(exp_dir: Path):
    """Load attention and parallel set data."""
    attn_file = exp_dir / "attention_weights.npz"
    parallel_file = exp_dir / "parallel_sets.json"

    if not attn_file.exists():
        raise FileNotFoundError(f"No attention data: {attn_file}")
    if not parallel_file.exists():
        raise FileNotFoundError(f"No parallel sets data: {parallel_file}")

    attn_data = np.load(attn_file, allow_pickle=True)
    with open(parallel_file) as f:
        parallel_sets = json.load(f)

    return attn_data, parallel_sets


def analyze_cross_parallel_attention(attn_data, parallel_sets, step: int):
    """Analyze cross-parallel attention for a specific step.

    Returns:
        dict with analysis results
    """
    attn_key = f'step_{step}_attention'
    pos_key = f'step_{step}_positions'

    if attn_key not in attn_data or pos_key not in attn_data:
        return None

    # Get attention (tuple of layer tensors)
    attention_layers = attn_data[attn_key]
    positions = attn_data[pos_key]

    # Get parallel token info
    step_data = parallel_sets.get(str(step))
    if not step_data:
        return None

    num_parallel = len(positions)

    if num_parallel <= 1:
        return {
            'step': step,
            'num_parallel': num_parallel,
            'cross_parallel_attention': None,
            'reason': 'Only one parallel token'
        }

    # Analyze last layer (most interpretable)
    last_layer_attn = attention_layers[-1]  # Shape: [batch, heads, seq_len, seq_len]

    # Extract attention between parallel tokens
    # positions contains the physical indices of parallel tokens
    min_pos = min(positions)
    max_pos = max(positions)

    # Get attention from parallel tokens to each other
    # attn[batch=0, :, parallel_token_i, parallel_token_j]
    cross_parallel_values = []

    for i, pos_i in enumerate(positions):
        for j, pos_j in enumerate(positions):
            if i != j:  # Don't include self-attention
                # Average across all heads
                attn_value = last_layer_attn[0, :, pos_i, pos_j].mean()
                cross_parallel_values.append(float(attn_value))

    return {
        'step': step,
        'num_parallel': num_parallel,
        'positions': positions.tolist() if hasattr(positions, 'tolist') else list(positions),
        'cross_parallel_mean': np.mean(cross_parallel_values) if cross_parallel_values else 0.0,
        'cross_parallel_max': np.max(cross_parallel_values) if cross_parallel_values else 0.0,
        'cross_parallel_min': np.min(cross_parallel_values) if cross_parallel_values else 0.0,
        'num_cross_parallel_pairs': len(cross_parallel_values),
        'all_values': cross_parallel_values[:20]  # First 20 for inspection
    }


def compare_modes():
    """Compare isolated vs visible modes."""

    print("="*80)
    print("VERIFYING ATTENTION DIFFERENCES BETWEEN ISOLATION MODES")
    print("="*80)

    iso_dir = Path("experiments/results/exp1_isolated")
    vis_dir = Path("experiments/results/exp1_visible")

    print(f"\nLoading isolated mode data from: {iso_dir}")
    iso_attn, iso_parallel = load_experiment_data(iso_dir)

    print(f"Loading visible mode data from: {vis_dir}")
    vis_attn, vis_parallel = load_experiment_data(vis_dir)

    # Find steps with multiple parallel tokens
    print("\n" + "="*80)
    print("ANALYZING CROSS-PARALLEL ATTENTION")
    print("="*80)

    for step in range(10):  # Check all 10 steps
        print(f"\n--- Step {step} ---")

        iso_analysis = analyze_cross_parallel_attention(iso_attn, iso_parallel, step)
        vis_analysis = analyze_cross_parallel_attention(vis_attn, vis_parallel, step)

        if not iso_analysis or not vis_analysis:
            print("  No data for this step")
            continue

        if iso_analysis.get('cross_parallel_attention') is None:
            print(f"  Only {iso_analysis['num_parallel']} parallel token(s) - skipping")
            continue

        print(f"  Parallel tokens: {iso_analysis['num_parallel']}")
        print(f"  Physical positions: {iso_analysis['positions']}")
        print(f"  Cross-parallel pairs: {iso_analysis['num_cross_parallel_pairs']}")

        print(f"\n  ISOLATED MODE:")
        print(f"    Mean cross-parallel attention: {iso_analysis['cross_parallel_mean']:.6f}")
        print(f"    Max:  {iso_analysis['cross_parallel_max']:.6f}")
        print(f"    Min:  {iso_analysis['cross_parallel_min']:.6f}")
        print(f"    Sample values: {[f'{v:.6f}' for v in iso_analysis['all_values'][:5]]}")

        print(f"\n  VISIBLE MODE:")
        print(f"    Mean cross-parallel attention: {vis_analysis['cross_parallel_mean']:.6f}")
        print(f"    Max:  {vis_analysis['cross_parallel_max']:.6f}")
        print(f"    Min:  {vis_analysis['cross_parallel_min']:.6f}")
        print(f"    Sample values: {[f'{v:.6f}' for v in vis_analysis['all_values'][:5]]}")

        # Compare
        diff = abs(vis_analysis['cross_parallel_mean'] - iso_analysis['cross_parallel_mean'])
        print(f"\n  DIFFERENCE:")
        print(f"    Absolute difference in mean: {diff:.6f}")

        if diff < 1e-6:
            print(f"    🚩 WARNING: Virtually identical (< 1e-6)")
        elif diff < 1e-4:
            print(f"    ⚠️  Very similar (< 1e-4)")
        else:
            print(f"    ✓ Measurable difference")

    # Now check if the full attention matrices are identical
    print("\n" + "="*80)
    print("COMPARING FULL ATTENTION MATRICES")
    print("="*80)

    for step in range(min(3, 10)):  # Check first 3 steps
        print(f"\nStep {step}:")

        iso_attn_layers = iso_attn[f'step_{step}_attention']
        vis_attn_layers = vis_attn[f'step_{step}_attention']

        # Compare last layer
        iso_layer = iso_attn_layers[-1]
        vis_layer = vis_attn_layers[-1]

        # Check if identical
        are_identical = np.allclose(iso_layer, vis_layer, atol=1e-7)
        max_diff = np.max(np.abs(iso_layer - vis_layer))

        print(f"  Last layer shape: {iso_layer.shape}")
        print(f"  Are identical (atol=1e-7): {are_identical}")
        print(f"  Max absolute difference: {max_diff:.10f}")

        if are_identical:
            print(f"  🚩 RED FLAG: Attention matrices are identical!")
        else:
            print(f"  ✓ Attention matrices differ")


def main():
    """Main entry point."""
    try:
        compare_modes()

        print("\n" + "="*80)
        print("DIAGNOSIS")
        print("="*80)
        print("""
If cross-parallel attention is near-zero in BOTH modes:
  → Attention masking may not be working as expected
  → OR the model naturally ignores same-position tokens

If cross-parallel attention is identical between modes:
  → RED FLAG: Isolation mode is not actually isolating
  → Attention mask is not being applied

If cross-parallel attention differs significantly:
  → ✓ Modes are working as intended
  → Model behavior is genuinely identical despite different attention
        """)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
