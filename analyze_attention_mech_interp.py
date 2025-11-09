#!/usr/bin/env python3
"""Mechanistic interpretability analysis of TEMPO attention patterns.

This script analyzes the actual attention weight matrices to understand:
1. How parallel tokens attend to prior context
2. Whether parallel tokens truly don't see each other (isolation verification)
3. Which attention heads are responsible for different behaviors
4. Layer-wise attention patterns
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

def load_experiment(exp_dir: Path) -> Dict:
    """Load experiment data including attention weights and metadata."""
    metadata_file = exp_dir / "experiment_metadata.json"
    results_file = exp_dir / "results.json"
    attention_file = exp_dir / "attention_weights.npz"
    parallel_sets_file = exp_dir / "parallel_sets.json"

    with open(metadata_file) as f:
        metadata = json.load(f)

    with open(results_file) as f:
        results = json.load(f)

    with open(parallel_sets_file) as f:
        parallel_sets = json.load(f)

    attention_data = np.load(attention_file)

    return {
        "metadata": metadata,
        "results": results,
        "parallel_sets": parallel_sets,
        "attention_data": attention_data
    }

def analyze_attention_shape(attention_data: np.lib.npyio.NpzFile) -> None:
    """Analyze the structure of attention weight tensors."""
    print("\n" + "="*80)
    print("ATTENTION TENSOR STRUCTURE ANALYSIS")
    print("="*80)

    # Get a sample attention matrix
    first_key = [k for k in attention_data.keys() if 'attention' in k][0]
    sample = attention_data[first_key]

    print(f"\nSample key: {first_key}")
    print(f"Shape: {sample.shape}")
    print(f"  Dimension breakdown (likely):")
    print(f"    [0] = {sample.shape[0]} layers")
    print(f"    [1] = {sample.shape[1]} batch size")
    print(f"    [2] = {sample.shape[2]} attention heads")
    print(f"    [3] = {sample.shape[3]} query tokens (current step)")
    print(f"    [4] = {sample.shape[4]} key tokens (context)")

    return sample.shape

def analyze_parallel_isolation(exp_data: Dict, step: int) -> None:
    """Analyze whether parallel tokens at a given step can see each other."""
    print(f"\n" + "="*80)
    print(f"PARALLEL TOKEN ISOLATION ANALYSIS - Step {step}")
    print("="*80)

    attention_data = exp_data["attention_data"]
    parallel_sets = exp_data["parallel_sets"]
    isolated = exp_data["metadata"]["config"]["allow_intraset_token_visibility"] == False

    # Get attention weights for this step
    attn_key = f"step_{step}_attention"
    pos_key = f"step_{step}_positions"
    logical_key = f"step_{step}_logical"

    if attn_key not in attention_data:
        print(f"  ⚠ No attention data for step {step}")
        return

    attn_weights = attention_data[attn_key]  # Shape: [layers, batch, heads, queries, keys]
    positions = attention_data[pos_key]
    logical_positions = attention_data[logical_key]

    print(f"\nMode: {'ISOLATED' if isolated else 'VISIBLE'}")
    print(f"Attention shape: {attn_weights.shape}")
    print(f"Physical positions: {positions}")
    print(f"Logical positions: {logical_positions}")

    # Find parallel tokens at this step
    # parallel_sets has structure: {"parallel_sets": [{step, count, tokens, positions}, ...]}
    parallel_steps = parallel_sets.get("parallel_sets", [])
    step_data = None
    for ps in parallel_steps:
        if ps["step"] == step:
            step_data = ps
            break

    if step_data is None:
        print(f"  ℹ No parallel tokens at step {step}")
        return

    num_parallel = step_data["count"]
    parallel_token_ids = step_data["tokens"]
    parallel_physical_positions = step_data["positions"]

    if num_parallel <= 1:
        print(f"  ℹ Only {num_parallel} token at step {step}, no parallelism")
        return

    print(f"\nParallel tokens at step {step}: {num_parallel} tokens")
    print(f"  Token IDs: {parallel_token_ids}")
    print(f"  Physical positions: {parallel_physical_positions}")

    # Average over layers and batch (focus on heads)
    # attn_weights shape: [layers, batch, heads, queries, keys]
    avg_over_layers = attn_weights.mean(axis=0)  # [batch, heads, queries, keys]
    avg_over_batch = avg_over_layers.mean(axis=0)  # [heads, queries, keys]

    num_heads = avg_over_batch.shape[0]
    num_queries = avg_over_batch.shape[1]
    num_keys = avg_over_batch.shape[2]

    print(f"  Heads: {num_heads}, Queries: {num_queries}, Keys: {num_keys}")

    # Attention from current step tokens to each other vs to prior context
    # Assuming queries are the current step tokens and keys include all context

    # Get attention to last few positions (likely the parallel tokens)
    if num_queries == num_parallel:
        # Current step queries attending to all keys
        # Last `num_parallel` keys should be the parallel tokens themselves

        # Attention to parallel siblings (last num_parallel keys)
        sibling_attention = avg_over_batch[:, :, -num_parallel:]  # [heads, queries, parallel_keys]

        # Attention to prior context (all except last num_parallel keys)
        prior_context_attention = avg_over_batch[:, :, :-num_parallel] if num_keys > num_parallel else np.array([])

        print(f"\n  Cross-parallel (sibling) attention:")
        for q in range(num_queries):
            # Attention from query q to its siblings (excluding self)
            sibling_weights = sibling_attention[:, q, :]  # [heads, parallel_keys]

            # Exclude self-attention (diagonal)
            sibling_weights_no_self = np.concatenate([
                sibling_weights[:, :q],
                sibling_weights[:, q+1:]
            ], axis=1) if sibling_weights.shape[1] > 1 else np.array([])

            if sibling_weights_no_self.size > 0:
                mean_sibling = sibling_weights_no_self.mean()
                max_sibling = sibling_weights_no_self.max()
                print(f"    Token {q} → siblings: mean={mean_sibling:.6f}, max={max_sibling:.6f}")
            else:
                print(f"    Token {q} → siblings: (only one token)")

        if prior_context_attention.size > 0:
            print(f"\n  Prior context attention:")
            for q in range(num_queries):
                prior_weights = prior_context_attention[:, q, :]  # [heads, context_keys]
                mean_prior = prior_weights.mean()
                max_prior = prior_weights.max()
                print(f"    Token {q} → prior context: mean={mean_prior:.6f}, max={max_prior:.6f}")

            # Overall comparison
            mean_cross = sibling_attention[:, :, :].mean()
            mean_prior = prior_context_attention.mean()

            print(f"\n  Summary:")
            print(f"    Mean cross-parallel attention: {mean_cross:.6f}")
            print(f"    Mean prior context attention:  {mean_prior:.6f}")
            print(f"    Ratio (prior / cross): {mean_prior / mean_cross:.2f}x" if mean_cross > 0 else "    Ratio: undefined (zero cross-attention)")

            if isolated and mean_cross < 0.01:
                print(f"    ✓ ISOLATION VERIFIED: Cross-parallel attention near zero")
            elif isolated and mean_cross >= 0.01:
                print(f"    ⚠ ISOLATION INCOMPLETE: Cross-parallel attention = {mean_cross:.6f}")
            elif not isolated and mean_cross > mean_prior * 0.1:
                print(f"    ✓ VISIBLE MODE: Parallel tokens attend to each other")
        else:
            print(f"    ℹ No prior context (first step)")

def analyze_layer_wise_attention(exp_data: Dict, step: int) -> None:
    """Analyze how attention patterns differ across layers."""
    print(f"\n" + "="*80)
    print(f"LAYER-WISE ATTENTION PATTERNS - Step {step}")
    print("="*80)

    attention_data = exp_data["attention_data"]
    attn_key = f"step_{step}_attention"

    if attn_key not in attention_data:
        print(f"  ⚠ No attention data for step {step}")
        return

    attn_weights = attention_data[attn_key]  # [layers, batch, heads, queries, keys]
    num_layers = attn_weights.shape[0]

    print(f"\nTotal layers: {num_layers}")

    # Analyze attention entropy per layer (measure of focus vs diffusion)
    for layer in range(num_layers):
        layer_attn = attn_weights[layer]  # [batch, heads, queries, keys]

        # Compute entropy for this layer
        # Entropy = -sum(p * log(p)) where p are attention weights
        epsilon = 1e-10
        entropy = -(layer_attn * np.log(layer_attn + epsilon)).sum(axis=-1)  # [batch, heads, queries]
        mean_entropy = entropy.mean()

        # Compute attention concentration (max attention weight)
        max_attention = layer_attn.max(axis=-1).mean()  # [batch, heads, queries] -> scalar

        print(f"  Layer {layer:2d}: entropy={mean_entropy:.4f}, max_attn={max_attention:.4f}")

    # Find which layers have most focused attention
    entropies = []
    for layer in range(num_layers):
        layer_attn = attn_weights[layer]
        epsilon = 1e-10
        entropy = -(layer_attn * np.log(layer_attn + epsilon)).sum(axis=-1).mean()
        entropies.append(entropy)

    most_focused = np.argmin(entropies)
    most_diffuse = np.argmax(entropies)

    print(f"\n  Most focused layer:  Layer {most_focused} (entropy={entropies[most_focused]:.4f})")
    print(f"  Most diffuse layer:  Layer {most_diffuse} (entropy={entropies[most_diffuse]:.4f})")

def analyze_head_specialization(exp_data: Dict, step: int) -> None:
    """Analyze whether different attention heads specialize in different patterns."""
    print(f"\n" + "="*80)
    print(f"ATTENTION HEAD SPECIALIZATION - Step {step}")
    print("="*80)

    attention_data = exp_data["attention_data"]
    attn_key = f"step_{step}_attention"

    if attn_key not in attention_data:
        print(f"  ⚠ No attention data for step {step}")
        return

    attn_weights = attention_data[attn_key]  # [layers, batch, heads, queries, keys]

    # Average over layers and batch
    avg_over_layers_batch = attn_weights.mean(axis=(0, 1))  # [heads, queries, keys]
    num_heads = avg_over_layers_batch.shape[0]
    num_keys = avg_over_layers_batch.shape[2]

    print(f"\nAnalyzing {num_heads} attention heads:")

    # For each head, compute:
    # 1. How much it attends to recent vs distant context
    # 2. How focused vs diffuse it is

    for head in range(num_heads):
        head_attn = avg_over_layers_batch[head]  # [queries, keys]

        # Recent vs distant: attention to last 3 keys vs rest
        if num_keys >= 3:
            recent_attn = head_attn[:, -3:].mean()
            distant_attn = head_attn[:, :-3].mean() if num_keys > 3 else 0.0
        else:
            recent_attn = head_attn.mean()
            distant_attn = 0.0

        # Focus: max attention weight
        max_attn = head_attn.max()

        # Entropy
        epsilon = 1e-10
        entropy = -(head_attn * np.log(head_attn + epsilon)).sum(axis=-1).mean()

        print(f"  Head {head:2d}: recent={recent_attn:.4f}, distant={distant_attn:.4f}, "
              f"max={max_attn:.4f}, entropy={entropy:.4f}")

    # Identify specialized heads
    print(f"\n  Pattern observations:")
    print(f"    - Heads with high 'recent' focus on local context")
    print(f"    - Heads with high 'distant' focus on long-range dependencies")
    print(f"    - Heads with low entropy are highly focused")
    print(f"    - Heads with high entropy are more diffuse")

def main():
    """Run mechanistic interpretability analysis."""
    print("╔" + "="*78 + "╗")
    print("║" + "TEMPO MECHANISTIC INTERPRETABILITY ANALYSIS".center(78) + "║")
    print("╚" + "="*78 + "╝")

    # Load experiments
    results_dir = Path("experiments/results")

    exp_isolated = load_experiment(results_dir / "exp1_isolated")
    exp_visible = load_experiment(results_dir / "exp1_visible")

    print(f"\nLoaded experiments:")
    print(f"  • exp1_isolated: {exp_isolated['metadata']['config']['description']}")
    print(f"  • exp1_visible: {exp_visible['metadata']['config']['description']}")

    # Analyze attention structure
    analyze_attention_shape(exp_isolated["attention_data"])

    # Analyze isolation at a specific step with parallel tokens
    # Step 1 should have parallel tokens based on threshold 0.1
    for step in [1, 2, 3]:
        analyze_parallel_isolation(exp_isolated, step)

    # Compare with visible mode
    print("\n" + "="*80)
    print("VISIBLE MODE COMPARISON")
    print("="*80)
    for step in [1, 2, 3]:
        analyze_parallel_isolation(exp_visible, step)

    # Layer-wise analysis
    analyze_layer_wise_attention(exp_isolated, step=1)

    # Head specialization
    analyze_head_specialization(exp_isolated, step=1)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Questions Answered:")
    print("  1. ✓ Attention tensor structure documented")
    print("  2. ✓ Parallel token isolation verified (or not)")
    print("  3. ✓ Layer-wise attention patterns analyzed")
    print("  4. ✓ Attention head specialization examined")

if __name__ == "__main__":
    main()
