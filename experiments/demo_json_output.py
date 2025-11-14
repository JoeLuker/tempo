"""
TEMPO JSON Output Demo

Demonstrates how to use TEMPO's JSON output format for programmatic access
to generation results, including parallel tokens, clean text, and metrics.
"""

import subprocess
import json
import sys
from pathlib import Path

def run_tempo_json(prompt, threshold=0.05, max_tokens=5):
    """Run TEMPO and get JSON output."""

    cmd = [
        sys.executable,
        "run_tempo.py",
        "--prompt", prompt,
        "--selection-threshold", str(threshold),
        "--max-tokens", str(max_tokens),
        "--output-json"
    ]

    print(f"Running TEMPO with prompt: '{prompt}'")
    print(f"Threshold: {threshold}, Max tokens: {max_tokens}")
    print()

    # Run and capture output
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )

    # Find the JSON in the output (it starts with {)
    lines = result.stdout.split('\n')
    json_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('{'):
            json_start = i
            break

    if json_start is None:
        print("Error: No JSON output found")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None

    # Extract JSON (from { to the end of the object)
    json_lines = []
    brace_count = 0
    for line in lines[json_start:]:
        json_lines.append(line)
        brace_count += line.count('{') - line.count('}')
        if brace_count == 0:
            break

    json_text = '\n'.join(json_lines)

    try:
        data = json.loads(json_text)
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print("JSON text:", json_text)
        return None


def print_json_results(data):
    """Print formatted JSON results."""

    print("="*80)
    print("TEMPO GENERATION RESULTS (JSON Format)")
    print("="*80)
    print()

    # Prompt
    print(f"Prompt: '{data['prompt']}'")
    print()

    # Tokens array
    print(f"Generated tokens ({len(data['tokens'])} steps):")
    print()

    # Show first few token steps
    for i, step_data in enumerate(data['tokens'][:5]):
        tokens = step_data['tokens']
        print(f"  Step {step_data['step']}: {len(tokens)} alternative(s)")
        for j, tok in enumerate(tokens):
            marker = "→" if j == 0 else " "
            print(f"    {marker} '{tok['text']}' (p={tok['probability']:.3f})")
        print()

    if len(data['tokens']) > 5:
        print(f"  ... and {len(data['tokens']) - 5} more steps")
        print()

    # Configuration
    print("Configuration:")
    for key, value in data['config'].items():
        print(f"  {key}: {value}")
    print()

    # Metrics
    print("Performance Metrics:")
    print(f"  Generation time: {data['metrics']['generation_time']:.4f}s")
    print(f"  Tokens per second: {data['metrics']['tokens_per_second']:.2f}")
    print(f"  Total steps: {data['metrics']['total_steps']}")
    print(f"  Parallel steps: {data['metrics']['parallel_steps']}")
    print(f"  Removal time: {data['metrics']['removal_time']:.4f}s")
    print(f"  Removal steps: {data['metrics']['removal_steps']}")
    print()


def main():
    print("\n" + "="*80)
    print("TEMPO JSON OUTPUT DEMONSTRATION")
    print("="*80)
    print()
    print("This demo shows how to use TEMPO's JSON output format for")
    print("programmatic access to generation results.")
    print()

    # Example 1: Hotdog sandwich question
    print("\nExample 1: Hotdog Sandwich Question")
    print("-" * 80)

    data1 = run_tempo_json(
        "The answer to whether a hotdog is a sandwich or not is",
        threshold=0.05,
        max_tokens=5
    )

    if data1:
        print_json_results(data1)

    # Example 2: Story beginning
    print("\nExample 2: Story Beginning")
    print("-" * 80)

    data2 = run_tempo_json(
        "Once upon a time",
        threshold=0.08,
        max_tokens=4
    )

    if data2:
        print_json_results(data2)

    # Show how to use the JSON programmatically
    print("\n" + "="*80)
    print("PROGRAMMATIC ACCESS EXAMPLE")
    print("="*80)
    print()
    print("You can access the JSON data programmatically:")
    print()
    print("  import subprocess, json")
    print("  result = subprocess.run([")
    print("      'python3', 'run_tempo.py',")
    print("      '--prompt', 'Your prompt here',")
    print("      '--selection-threshold', '0.05',")
    print("      '--max-tokens', '10',")
    print("      '--output-json'")
    print("  ], capture_output=True, text=True)")
    print()
    print("  # Parse JSON from output")
    print("  data = json.loads(result.stdout.split('{', 1)[1].rsplit('}', 1)[0] + '}')")
    print()
    print("  # Access token data")
    print("  prompt = data['prompt']")
    print("  tokens = data['tokens']  # Array of token steps with alternatives")
    print("  config = data['config']")
    print("  metrics = data['metrics']")
    print()
    print("  # Process tokens")
    print("  for step_data in tokens:")
    print("      step = step_data['step']")
    print("      alternatives = step_data['tokens']")
    print("      for tok in alternatives:")
    print("          print(f\"Token: {tok['text']} (p={tok['probability']})\")")
    print()

    print("="*80)
    print("Use --json-output-file to save to a file:")
    print("  python3 run_tempo.py --prompt 'Your prompt' --output-json --json-output-file results.json")
    print("="*80)


if __name__ == "__main__":
    main()
