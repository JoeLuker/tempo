#!/usr/bin/env python3
"""Find the position gap threshold that affects generation while maintaining coherence.

Large gaps (1000+) break coherence completely.
We need to find smaller gaps that show temporal perception without breaking the model.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model
import torch


def test_position_gap(prompt: str, position_gap: int, max_tokens: int = 25) -> dict:
    """Test a single position gap."""
    print(f"\n{'='*70}")
    print(f"Position Gap: {position_gap}")
    print(f"{'='*70}")

    # Load model
    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")
    prompt_length = input_ids.shape[1]

    print(f"Prompt: '{prompt}'")
    print(f"Prompt length: {prompt_length} tokens")

    # Generate
    all_input_ids = input_ids

    for step in range(max_tokens):
        current_length = all_input_ids.shape[1]

        # Create position IDs with gap
        if position_gap == 0:
            # Normal sequential
            position_ids = torch.arange(current_length, device="mps").unsqueeze(0)
        else:
            # Gap: prompt at normal positions, then offset generation
            prompt_positions = torch.arange(prompt_length, device="mps")
            gen_start = prompt_length + position_gap
            generated_positions = torch.arange(
                gen_start,
                gen_start + (current_length - prompt_length),
                device="mps"
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

        # Get next token
        next_token_logits = outputs.logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)

        all_input_ids = torch.cat([all_input_ids, next_token_id], dim=1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # Decode
    generated_ids = all_input_ids[0, prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(all_input_ids[0], skip_special_tokens=True)

    result = {
        "prompt": prompt,
        "position_gap": position_gap,
        "generated_text": generated_text,
        "full_text": full_text,
        "coherent": True  # We'll judge manually
    }

    print(f"\nGenerated: '{generated_text}'")
    print(f"Full: '{full_text}'")

    return result


def main():
    """Test gradually increasing position gaps to find coherence threshold."""

    print("="*70)
    print("FINDING COHERENCE-PRESERVING POSITION THRESHOLD")
    print("="*70)
    print()
    print("Testing small position gaps to find where temporal perception")
    print("begins to change WITHOUT breaking coherence.")
    print()

    # Test prompt
    prompt = "We've been talking for"

    # Test increasing gaps
    gaps = [
        0,      # Control
        5,      # Tiny gap (5 positions ahead)
        10,     # Small gap
        20,     # Moderate gap
        50,     # Larger gap
        100,    # Big gap
        200,    # Very big gap
    ]

    results = []

    for gap in gaps:
        try:
            result = test_position_gap(prompt, gap, max_tokens=25)
            results.append(result)
        except Exception as e:
            print(f"\nERROR with gap={gap}: {e}")
            import traceback
            traceback.print_exc()
            break

    # Save results
    output_file = Path("experiments/results/coherence_threshold_test.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)

    for r in results:
        gap = r['position_gap']
        text = r['generated_text'][:80]
        print(f"\nGap={gap:3d}: {text}")

    print("\n" + "="*70)
    print(f"Results saved to: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
