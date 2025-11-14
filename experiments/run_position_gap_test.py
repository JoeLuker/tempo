#!/usr/bin/env python3
"""Run position gap experiment using existing TEMPO infrastructure.

Tests whether position gaps affect temporal perception by modifying
the generation process to skip position indices.
"""

import sys
from pathlib import Path
import json
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model
import torch


def generate_with_position_gap(
    prompt: str,
    position_offset: int,
    max_tokens: int = 30,
    model_name: str = "deepcogito/cogito-v1-preview-llama-3B",
    device: str = "mps"
) -> Dict:
    """Generate text with a position gap after the prompt.

    Args:
        prompt: The prompt text
        position_offset: Position index to jump to after prompt (0 = no gap)
        max_tokens: Number of tokens to generate
        model_name: Model to use
        device: Device for computation

    Returns:
        Dict with results including generated text and positions used
    """
    print(f"\n{'='*70}")
    if position_offset == 0:
        print(f"CONTROL: Normal sequential positions")
    else:
        print(f"TREATMENT: Position gap to {position_offset}")
    print(f"{'='*70}")

    # Load model and tokenizer
    print(f"Loading model...")
    model, tokenizer = load_model(model_name, device=device, load_tokenizer=True)

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]

    print(f"Prompt: '{prompt}'")
    print(f"Prompt length: {prompt_length} tokens")

    # Generate tokens with position control
    all_input_ids = input_ids

    for step in range(max_tokens):
        current_length = all_input_ids.shape[1]

        # Create position IDs with gap
        if position_offset == 0:
            # Normal sequential positions
            position_ids = torch.arange(current_length, device=device).unsqueeze(0)
        else:
            # Position gap: prompt at 0...N-1, then jump to offset
            prompt_positions = torch.arange(prompt_length, device=device)
            generated_positions = torch.arange(
                position_offset,
                position_offset + (current_length - prompt_length),
                device=device
            )
            position_ids = torch.cat([prompt_positions, generated_positions]).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=all_input_ids,
                position_ids=position_ids,
                return_dict=True,
                use_cache=False
            )

        # Get next token (greedy for consistency)
        next_token_logits = outputs.logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)

        # Append to sequence
        all_input_ids = torch.cat([all_input_ids, next_token_id], dim=1)

        # Check for EOS
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # Decode generated text
    generated_ids = all_input_ids[0, prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(all_input_ids[0], skip_special_tokens=True)

    # Calculate position info
    if position_offset == 0:
        start_pos = prompt_length
        end_pos = prompt_length + len(generated_ids) - 1
        gap = 0
    else:
        start_pos = position_offset
        end_pos = position_offset + len(generated_ids) - 1
        gap = position_offset - prompt_length

    print(f"Generated {len(generated_ids)} tokens")
    print(f"Position range: {start_pos} to {end_pos}")
    if gap > 0:
        print(f"Position gap: {gap}")
    print(f"\nGenerated text: '{generated_text}'")
    print(f"Full text: '{full_text}'")

    return {
        "prompt": prompt,
        "prompt_length": prompt_length,
        "position_offset": position_offset,
        "position_gap": gap,
        "start_position": start_pos,
        "end_position": end_pos,
        "tokens_generated": len(generated_ids),
        "generated_text": generated_text,
        "full_text": full_text,
    }


def run_experiment():
    """Run the full position-as-time perception experiment."""

    print("="*70)
    print("POSITION-AS-TIME PERCEPTION EXPERIMENT")
    print("="*70)
    print()
    print("Hypothesis: LLMs perceive elapsed time through position indices")
    print()

    # Test prompts
    test_prompts = [
        "We've been talking for",
        "How long has this conversation been going?",
        "This conversation started",
        "After all this time,",
    ]

    # Position gaps to test
    position_gaps = [
        0,      # Control
        1000,   # Moderate gap
        5000,   # Large gap
        10000,  # Massive gap
    ]

    results = []

    # Run each prompt with each gap
    for prompt in test_prompts:
        print(f"\n{'#'*70}")
        print(f"# TESTING PROMPT: '{prompt}'")
        print(f"{'#'*70}")

        for gap in position_gaps:
            try:
                result = generate_with_position_gap(
                    prompt=prompt,
                    position_offset=gap,
                    max_tokens=25,
                    device="mps"
                )
                results.append(result)

            except Exception as e:
                print(f"\nERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "position_gap_results.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_file}")

    # Print comparative analysis
    print(f"\n{'='*70}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*70}")

    for prompt in test_prompts:
        print(f"\n\nPrompt: '{prompt}'")
        print("-"*70)

        prompt_results = [r for r in results if r['prompt'] == prompt]

        for r in prompt_results:
            gap_label = "CONTROL" if r['position_gap'] == 0 else f"GAP={r['position_gap']:,}"
            print(f"\n{gap_label:15s} (pos {r['start_position']:,}-{r['end_position']:,}):")
            print(f"  Generated: {r['generated_text'][:80]}")

    return results


def quick_test():
    """Quick test with one prompt."""
    print("="*70)
    print("QUICK TEST - Position Gap Effect on Temporal Perception")
    print("="*70)

    prompt = "We've been talking for"

    print(f"\nPrompt: '{prompt}'")
    print("\nTesting 3 conditions:")
    print("  1. Control (normal positions)")
    print("  2. Gap to position 1000")
    print("  3. Gap to position 10000")

    results = []

    for gap in [0, 1000, 10000]:
        result = generate_with_position_gap(
            prompt=prompt,
            position_offset=gap,
            max_tokens=20,
            device="mps"
        )
        results.append(result)

    print(f"\n{'='*70}")
    print("RESULTS COMPARISON")
    print(f"{'='*70}")

    for r in results:
        gap_label = "CONTROL" if r['position_gap'] == 0 else f"GAP={r['position_gap']:,}"
        print(f"\n{gap_label}:")
        print(f"  Full: {r['full_text']}")
        print(f"  Positions: {r['start_position']:,} to {r['end_position']:,}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Position gap temporal perception test")
    parser.add_argument("--quick", action="store_true", help="Quick test with one prompt")
    parser.add_argument("--device", default="mps", help="Device (mps, cuda, cpu)")

    args = parser.parse_args()

    if args.quick:
        quick_test()
    else:
        run_experiment()
