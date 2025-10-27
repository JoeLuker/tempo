#!/usr/bin/env python3
"""
Comprehensive analysis of isolation mechanism effects on TEMPO generation.

This script runs multiple experiments to understand:
1. How isolation affects text quality and coherence
2. Impact on diversity and creativity
3. Effect on different text types (narrative, factual, dialogue)
4. Relationship between selection threshold and isolation impact
"""

import subprocess
import yaml
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def run_experiment(
    prompt: str,
    selection_threshold: float,
    allow_visibility: bool,
    experiment_id: str,
    max_tokens: int = 30
) -> Dict:
    """Run a single TEMPO experiment."""
    config = {
        'experiment_name': experiment_id,
        'prompt': prompt,
        'max_tokens': max_tokens,
        'selection_threshold': selection_threshold,
        'seed': 42,
        'use_retroactive_removal': False,
        'allow_intraset_token_visibility': allow_visibility,
        'use_custom_rope': True,
        'debug_mode': False,
        'output_dir': f'./analysis_results/{experiment_id}',
        'capture_attention': False,
        'capture_logits': True,
    }

    config_path = f'./analysis_configs/{experiment_id}.yaml'
    Path('./analysis_configs').mkdir(exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    result = subprocess.run(
        ['python3', 'run_tempo.py', '--config', config_path],
        capture_output=True,
        text=True,
        check=True
    )

    # Load results
    with open(f'./analysis_results/{experiment_id}/results.json') as f:
        return json.load(f)


def compute_diversity_metrics(text: str) -> Dict:
    """Compute text diversity metrics."""
    tokens = text.split()
    unique_tokens = set(tokens)

    # Type-token ratio
    ttr = len(unique_tokens) / len(tokens) if tokens else 0

    # Bigram diversity
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
    unique_bigrams = set(bigrams)
    bigram_diversity = len(unique_bigrams) / len(bigrams) if bigrams else 0

    return {
        'type_token_ratio': ttr,
        'bigram_diversity': bigram_diversity,
        'unique_tokens': len(unique_tokens),
        'total_tokens': len(tokens)
    }


def analyze_coherence(text: str) -> Dict:
    """Simple coherence metrics."""
    # Count repeated words (might indicate loops)
    tokens = text.split()
    repeats = sum(1 for i in range(len(tokens)-1) if tokens[i] == tokens[i+1])
    repeat_ratio = repeats / len(tokens) if tokens else 0

    return {
        'immediate_repeats': repeats,
        'repeat_ratio': repeat_ratio
    }


def compare_logits(iso_dir: str, vis_dir: str) -> Dict:
    """Compare logits between isolated and visible modes."""
    iso_logits = np.load(f'{iso_dir}/logits_distributions.npz')
    vis_logits = np.load(f'{vis_dir}/logits_distributions.npz')

    iso_steps = [k for k in iso_logits.files if k.startswith('step_') and k.endswith('_logits')]

    kl_divergences = []
    entropy_diffs = []

    for step_idx in range(len(iso_steps)):
        iso_key = f'step_{step_idx}_logits'
        vis_key = f'step_{step_idx}_logits'

        if iso_key not in iso_logits.files or vis_key not in vis_logits.files:
            continue

        iso_log = iso_logits[iso_key]
        vis_log = vis_logits[vis_key]

        # Convert to probabilities
        iso_probs = np.exp(iso_log - np.max(iso_log, axis=-1, keepdims=True))
        iso_probs = iso_probs / np.sum(iso_probs, axis=-1, keepdims=True)

        vis_probs = np.exp(vis_log - np.max(vis_log, axis=-1, keepdims=True))
        vis_probs = vis_probs / np.sum(vis_probs, axis=-1, keepdims=True)

        # KL divergence
        kl_div = np.sum(iso_probs * np.log((iso_probs + 1e-10) / (vis_probs + 1e-10)))
        kl_divergences.append(kl_div)

        # Entropy difference
        iso_entropy = -np.sum(iso_probs * np.log(iso_probs + 1e-10))
        vis_entropy = -np.sum(vis_probs * np.log(vis_probs + 1e-10))
        entropy_diffs.append(iso_entropy - vis_entropy)

    return {
        'kl_divergences': kl_divergences,
        'mean_kl_divergence': np.mean(kl_divergences) if kl_divergences else 0,
        'max_kl_divergence': np.max(kl_divergences) if kl_divergences else 0,
        'entropy_differences': entropy_diffs,
        'mean_entropy_diff': np.mean(entropy_diffs) if entropy_diffs else 0
    }


# Test prompts covering different domains
TEST_PROMPTS = [
    # Narrative/Creative
    ("Once upon a time", "narrative"),
    ("The detective entered the room and", "narrative"),

    # Factual/Informative
    ("The capital of France is", "factual"),
    ("Photosynthesis is the process by which", "factual"),

    # Dialogue/Conversational
    ("'How are you today?' she asked.", "dialogue"),
    ("The teacher said to the students,", "dialogue"),
]

# Test different selection thresholds
THRESHOLDS = [0.05, 0.1, 0.2]

print("="*80)
print("COMPREHENSIVE ISOLATION ANALYSIS")
print("="*80)
print()

all_results = []

for prompt, category in TEST_PROMPTS:
    print(f"\n{'='*80}")
    print(f"Category: {category.upper()}")
    print(f"Prompt: '{prompt}'")
    print(f"{'='*80}\n")

    for threshold in THRESHOLDS:
        print(f"  Threshold: {threshold}")

        # Run both modes
        exp_id_iso = f"{category}_t{threshold}_isolated"
        exp_id_vis = f"{category}_t{threshold}_visible"

        try:
            iso_result = run_experiment(prompt, threshold, False, exp_id_iso)
            vis_result = run_experiment(prompt, threshold, True, exp_id_vis)

            # Extract texts
            iso_text = iso_result['raw_generated_text']
            vis_text = vis_result['raw_generated_text']

            print(f"    Isolated: {iso_text[:60]}...")
            print(f"    Visible:  {vis_text[:60]}...")

            # Compute metrics
            iso_diversity = compute_diversity_metrics(iso_text)
            vis_diversity = compute_diversity_metrics(vis_text)

            iso_coherence = analyze_coherence(iso_text)
            vis_coherence = analyze_coherence(vis_text)

            logit_comparison = compare_logits(
                f'./analysis_results/{exp_id_iso}',
                f'./analysis_results/{exp_id_vis}'
            )

            # Store results
            result_entry = {
                'prompt': prompt,
                'category': category,
                'threshold': threshold,
                'isolated_text': iso_text,
                'visible_text': vis_text,
                'isolated_diversity': iso_diversity,
                'visible_diversity': vis_diversity,
                'isolated_coherence': iso_coherence,
                'visible_coherence': vis_coherence,
                'logit_comparison': logit_comparison,
                'generation_time_iso': iso_result['generation_time'],
                'generation_time_vis': vis_result['generation_time']
            }

            all_results.append(result_entry)

            print(f"    KL Divergence: {logit_comparison['mean_kl_divergence']:.4f}")
            print(f"    Diversity Δ: {iso_diversity['type_token_ratio'] - vis_diversity['type_token_ratio']:+.4f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

print("\n" + "="*80)
print("SUMMARY ANALYSIS")
print("="*80)

# Group by category
by_category = {}
for r in all_results:
    cat = r['category']
    if cat not in by_category:
        by_category[cat] = []
    by_category[cat].append(r)

for category, results in by_category.items():
    print(f"\n{category.upper()}:")

    kl_divs = [r['logit_comparison']['mean_kl_divergence'] for r in results]
    diversity_diffs = [
        r['isolated_diversity']['type_token_ratio'] - r['visible_diversity']['type_token_ratio']
        for r in results
    ]

    print(f"  Average KL Divergence: {np.mean(kl_divs):.4f}")
    print(f"  Average Diversity Difference: {np.mean(diversity_diffs):+.4f}")
    print(f"  Isolation increases diversity: {sum(1 for d in diversity_diffs if d > 0)}/{len(diversity_diffs)} cases")

# Convert numpy types to Python types for JSON serialization
def convert_to_json_serializable(obj):
    """Convert numpy types to Python types."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

# Save detailed results
with open('./analysis_results/comprehensive_analysis.json', 'w') as f:
    json.dump(convert_to_json_serializable(all_results), f, indent=2)

print("\n" + "="*80)
print("Detailed results saved to: ./analysis_results/comprehensive_analysis.json")
print("="*80)
