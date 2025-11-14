#!/usr/bin/env python3
"""Demonstration of how to use TEMPO's JSON output programmatically.

This script shows real-world usage patterns for the JSON output format,
including accessing structured parallel token data with probabilities.
"""

import json
import subprocess
import sys


def run_tempo_json(prompt, threshold=0.05, max_tokens=10):
    """Run TEMPO and get JSON output."""
    cmd = [
        sys.executable, "run_tempo.py",
        "--prompt", prompt,
        "--selection-threshold", str(threshold),
        "--max-tokens", str(max_tokens),
        "--output-json"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Extract JSON from output (skip log messages)
    output_lines = result.stdout.strip().split('\n')
    json_started = False
    json_lines = []

    for line in output_lines:
        if line.startswith('{'):
            json_started = True
        if json_started:
            json_lines.append(line)
            if line.startswith('}'):
                break

    json_text = '\n'.join(json_lines)
    return json.loads(json_text)


def main():
    print("=" * 80)
    print("TEMPO JSON OUTPUT - PRACTICAL USAGE EXAMPLES")
    print("=" * 80)
    print()

    # Example 1: Analyze parallel token alternatives
    print("Example 1: Analyzing Token Alternatives")
    print("-" * 80)

    data = run_tempo_json("The weather today is", threshold=0.1, max_tokens=4)

    print(f"Prompt: '{data['prompt']}'")
    print(f"\nTokens at each step:")

    for step_data in data['tokens']:
        step = step_data['step']
        tokens = step_data['tokens']
        was_pruned = step_data['was_pruned']

        print(f"\n  Step {step}: {len(tokens)} alternative(s)")

        # Show top 3 tokens with probabilities
        for i, token in enumerate(tokens[:3]):
            marker = "→" if i == 0 else " "
            print(f"    {marker} '{token['text']}' (prob={token['probability']:.3f})")

        if was_pruned:
            orig_count = step_data['original_count']
            print(f"    (pruned from {orig_count} original tokens)")

    print()

    # Example 2: Build text from highest probability tokens
    print("Example 2: Reconstructing Text from Tokens")
    print("-" * 80)

    data = run_tempo_json("What is the meaning of life?", threshold=0.08, max_tokens=5)

    # Get highest probability token from each step
    highest_prob_tokens = []
    for step_data in data['tokens']:
        if step_data['tokens']:
            best_token = max(step_data['tokens'], key=lambda t: t['probability'])
            highest_prob_tokens.append(best_token['text'])

    reconstructed_text = ''.join(highest_prob_tokens)
    print(f"Prompt: '{data['prompt']}'")
    print(f"Highest probability path: {reconstructed_text}")
    print()

    # Example 3: Metrics and statistics
    print("Example 3: Generation Metrics")
    print("-" * 80)

    data = run_tempo_json("Once upon a time", threshold=0.05, max_tokens=8)

    metrics = data['metrics']
    config = data['config']

    print(f"Configuration:")
    print(f"  Threshold: {config['selection_threshold']}")
    print(f"  Max tokens: {config['max_tokens']}")
    print(f"  Retroactive pruning: {config['use_retroactive_removal']}")
    print()
    print(f"Generation Metrics:")
    print(f"  Total steps: {metrics['total_steps']}")
    print(f"  Parallel steps: {metrics['parallel_steps']}")
    print(f"  Pruned steps: {metrics['pruned_steps']}")
    print(f"  Generation time: {metrics['generation_time']:.3f}s")
    print(f"  Speed: {metrics['tokens_per_second']:.2f} tokens/sec")
    print()

    # Example 4: Token analysis
    print("Example 4: Token Analysis")
    print("-" * 80)

    data = run_tempo_json("The capital of France is", threshold=0.1, max_tokens=3)

    print(f"Prompt: '{data['prompt']}'")
    print()

    total_alternatives = sum(len(step['tokens']) for step in data['tokens'])
    avg_alternatives = total_alternatives / len(data['tokens']) if data['tokens'] else 0

    print(f"Statistics:")
    print(f"  Total steps: {len(data['tokens'])}")
    print(f"  Total token alternatives: {total_alternatives}")
    print(f"  Average alternatives per step: {avg_alternatives:.2f}")
    print()

    # Show token diversity
    print("Token diversity by step:")
    for step_data in data['tokens']:
        num_alts = len(step_data['tokens'])
        top_prob = step_data['tokens'][0]['probability'] if step_data['tokens'] else 0
        print(f"  Step {step_data['step']}: {num_alts} alternatives (top prob: {top_prob:.3f})")
    print()

    # Example 5: Save to file
    print("Example 5: Saving to File")
    print("-" * 80)

    print("You can also save directly to a JSON file:")
    print()
    print("  python3 run_tempo.py \\")
    print("    --prompt 'Your prompt here' \\")
    print("    --selection-threshold 0.05 \\")
    print("    --max-tokens 10 \\")
    print("    --output-json \\")
    print("    --json-output-file results.json")
    print()
    print("Then load it with:")
    print()
    print("  import json")
    print("  with open('results.json', 'r') as f:")
    print("      data = json.load(f)")
    print()
    print("  # Access token data")
    print("  for step_data in data['tokens']:")
    print("      for token in step_data['tokens']:")
    print("          print(f\"{token['text']} (p={token['probability']:.3f})\")")
    print()

    print("=" * 80)
    print("END OF DEMONSTRATION")
    print("=" * 80)


if __name__ == "__main__":
    main()
