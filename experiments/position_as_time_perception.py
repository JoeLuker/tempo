#!/usr/bin/env python3
"""Experiment: Do LLMs perceive time through position indices?

Tests whether large gaps in position indices affect the model's perception
of elapsed conversation time.

Hypothesis: Position indices serve as temporal markers independent of content.
Prediction: Large position gaps → model perceives more "elapsed time"

Experimental Design:
- Control: Normal sequential positions (0, 1, 2, 3...)
- Treatment 1: Skip to position 1000 after prompt
- Treatment 2: Skip to position 10000 after prompt
- Treatment 3: Skip to position 50000 after prompt

Test each with prompts that probe temporal perception.
"""

import sys
from pathlib import Path
import json
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.model.model_adapter import ModelAdapter
from src.infrastructure.tokenization.tokenizer_adapter import TokenizerAdapter
from src.application.services.rope_service import RoPEService


class PositionGapExperiment:
    """Test temporal perception through position gaps."""

    def __init__(self, model_name: str = "deepcogito/cogito-v1-preview-llama-3B", device: str = "mps"):
        """Initialize experiment."""
        self.model_name = model_name
        self.device = device

        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.model_adapter = ModelAdapter(model_name, device=device)
        self.tokenizer = TokenizerAdapter(model_name)

        # Initialize services
        self.rope_service = RoPEService(device=device, debug_mode=False)

    def run_control(self, prompt: str, max_tokens: int = 30) -> dict:
        """Run control condition with normal sequential positions."""
        print("\n" + "="*70)
        print("CONTROL: Normal Sequential Positions")
        print("="*70)

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        prompt_length = len(input_ids)

        print(f"Prompt: '{prompt}'")
        print(f"Prompt length: {prompt_length} tokens")
        print(f"Starting position: 0")
        print(f"Generating {max_tokens} tokens...")

        # Initialize RoPE service
        self.rope_service.initialize(prompt_length)

        # Generate with normal positions
        generated_ids = self._generate_tokens(input_ids, max_tokens, position_offset=0)
        generated_text = self.tokenizer.decode(generated_ids[prompt_length:])

        final_position = prompt_length + max_tokens - 1

        result = {
            "condition": "control",
            "prompt": prompt,
            "prompt_length": prompt_length,
            "start_position": 0,
            "end_position": final_position,
            "position_gap": 0,
            "generated_text": generated_text,
            "full_text": prompt + generated_text
        }

        print(f"Final position: {final_position}")
        print(f"Generated: '{generated_text}'")

        return result

    def run_with_gap(self, prompt: str, position_gap: int, max_tokens: int = 30) -> dict:
        """Run treatment with position gap after prompt."""
        print("\n" + "="*70)
        print(f"TREATMENT: Position Gap = {position_gap}")
        print("="*70)

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        prompt_length = len(input_ids)

        print(f"Prompt: '{prompt}'")
        print(f"Prompt length: {prompt_length} tokens")
        print(f"SKIPPING from position {prompt_length-1} to position {position_gap}")
        print(f"Generating {max_tokens} tokens...")

        # Initialize RoPE service
        self.rope_service.initialize(prompt_length)

        # Generate with position offset
        generated_ids = self._generate_tokens(input_ids, max_tokens, position_offset=position_gap)
        generated_text = self.tokenizer.decode(generated_ids[prompt_length:])

        final_position = position_gap + max_tokens - 1

        result = {
            "condition": "treatment",
            "prompt": prompt,
            "prompt_length": prompt_length,
            "start_position": position_gap,
            "end_position": final_position,
            "position_gap": position_gap - prompt_length,
            "generated_text": generated_text,
            "full_text": prompt + generated_text
        }

        print(f"Final position: {final_position}")
        print(f"Generated: '{generated_text}'")

        return result

    def _generate_tokens(self, input_ids: list, max_tokens: int, position_offset: int = 0) -> list:
        """Generate tokens with specified position offset.

        Args:
            input_ids: Prompt token IDs
            max_tokens: Number of tokens to generate
            position_offset: Position to start generation from (0 = normal, >prompt_length = gap)

        Returns:
            List of all token IDs (prompt + generated)
        """
        prompt_length = len(input_ids)
        all_ids = input_ids.copy()

        # Prepare prompt tensor
        input_tensor = torch.tensor([input_ids], device=self.device)

        for step in range(max_tokens):
            # Calculate position IDs
            if position_offset == 0:
                # Normal sequential positions
                position_ids = list(range(len(all_ids)))
            else:
                # Position gap: prompt uses 0...N-1, then jump to offset
                position_ids = (
                    list(range(prompt_length)) +  # Prompt positions
                    list(range(position_offset, position_offset + len(all_ids) - prompt_length))  # Generated positions
                )

            position_tensor = torch.tensor([position_ids], device=self.device)

            # Modify positions through RoPE service
            modified_positions = self.rope_service.get_modified_position_ids(position_tensor)

            # Get model outputs
            with torch.no_grad():
                outputs = self.model_adapter.model(
                    input_ids=input_tensor,
                    position_ids=modified_positions,
                    return_dict=True,
                    use_cache=False
                )

            # Get next token (greedy decoding for consistency)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()

            all_ids.append(next_token_id)

            # Update input tensor for next iteration
            input_tensor = torch.tensor([all_ids], device=self.device)

        return all_ids


def run_full_experiment():
    """Run complete position-as-time perception experiment."""

    # Test prompts designed to probe temporal perception
    test_prompts = [
        "We've been talking for",
        "How long has this conversation been going?",
        "I just started this conversation and",
        "After all this time discussing",
        "When we first started talking, I",
    ]

    # Position gaps to test
    position_gaps = [
        0,      # Control
        1000,   # Moderate gap
        10000,  # Large gap
        50000,  # Massive gap
    ]

    # Initialize experiment
    exp = PositionGapExperiment(device="mps")

    results = []

    # Run each prompt with each position gap
    for prompt in test_prompts:
        print("\n" + "#"*70)
        print(f"# TESTING PROMPT: '{prompt}'")
        print("#"*70)

        for gap in position_gaps:
            try:
                if gap == 0:
                    result = exp.run_control(prompt, max_tokens=30)
                else:
                    result = exp.run_with_gap(prompt, position_gap=gap, max_tokens=30)

                results.append(result)

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    output_file = Path("experiments/results/position_as_time_perception.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        prompt_results = [r for r in results if r['prompt'] == prompt]

        for r in prompt_results:
            gap_desc = "CONTROL (normal)" if r['position_gap'] == 0 else f"GAP={r['position_gap']}"
            print(f"  {gap_desc:20s}: {r['generated_text'][:60]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Position-as-time perception experiment")
    parser.add_argument("--device", default="mps", help="Device to use (mps, cuda, cpu)")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer prompts")

    args = parser.parse_args()

    if args.quick:
        # Quick test with one prompt
        print("Running QUICK test...")
        exp = PositionGapExperiment(device=args.device)

        prompt = "We've been talking for"

        results = []
        for gap in [0, 1000, 10000]:
            if gap == 0:
                result = exp.run_control(prompt)
            else:
                result = exp.run_with_gap(prompt, position_gap=gap)
            results.append(result)

        print("\n" + "="*70)
        print("QUICK TEST RESULTS")
        print("="*70)
        for r in results:
            gap_desc = "CONTROL" if r['position_gap'] == 0 else f"GAP={r['position_gap']}"
            print(f"{gap_desc:15s}: {r['generated_text']}")
    else:
        run_full_experiment()
