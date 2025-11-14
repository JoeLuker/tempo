#!/usr/bin/env python3
"""
Test using cache_position to properly handle position gaps.

The transformers library has a cache_position parameter that tells it
which positions we're actually querying, separate from the sequence indices.
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model


def test_cache_position_approach(prompt: str, gap: int):
    """
    Test if using cache_position helps with position gaps.

    cache_position tells the model: "these sequence indices correspond to these positions"
    """

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")
    prompt_length = input_ids.shape[1]

    print(f"\n{'='*80}")
    print(f"Testing cache_position approach with gap={gap}")
    print('='*80)
    print(f"Prompt: '{prompt}'")
    print(f"Prompt length: {prompt_length}")

    # Method 1: Using position_ids (what we've been doing)
    print(f"\nMethod 1: position_ids only")
    position_ids_with_gap = torch.cat([
        torch.arange(prompt_length, device="mps"),
        torch.tensor([prompt_length + gap], device="mps")
    ]).unsqueeze(0)

    extended_input = torch.cat([
        input_ids,
        torch.tensor([[tokenizer.encode("test")[0]]], device="mps")
    ], dim=1)

    try:
        with torch.no_grad():
            outputs1 = model(
                input_ids=extended_input,
                position_ids=position_ids_with_gap,
                return_dict=True,
                use_cache=False,
            )
        print(f"  ✓ Works without explicit attention_mask")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Method 2: Using cache_position
    print(f"\nMethod 2: cache_position parameter")

    # cache_position should be the actual positions we want
    cache_position = torch.cat([
        torch.arange(prompt_length, device="mps"),
        torch.tensor([prompt_length + gap], device="mps")
    ])

    try:
        with torch.no_grad():
            outputs2 = model(
                input_ids=extended_input,
                cache_position=cache_position,
                return_dict=True,
                use_cache=False,
            )
        print(f"  ✓ Works with cache_position")

        # Compare results
        if 'outputs1' in locals():
            diff = (outputs1.logits - outputs2.logits).abs().max().item()
            print(f"  Difference from method 1: {diff:.6f}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Method 3: Both cache_position and position_ids
    print(f"\nMethod 3: Both cache_position AND position_ids")
    try:
        with torch.no_grad():
            outputs3 = model(
                input_ids=extended_input,
                position_ids=position_ids_with_gap,
                cache_position=cache_position,
                return_dict=True,
                use_cache=False,
            )
        print(f"  ✓ Works with both")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def test_proper_mask_creation(prompt: str, gap: int):
    """
    Create a proper 4D attention mask that allows position gaps.
    """

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")
    prompt_length = input_ids.shape[1]

    print(f"\n{'='*80}")
    print(f"Creating proper 4D causal mask for gap={gap}")
    print('='*80)

    # Add one generated token
    extended_input = torch.cat([
        input_ids,
        torch.tensor([[tokenizer.encode("test")[0]]], device="mps")
    ], dim=1)

    seq_length = extended_input.shape[1]

    # Create 4D causal mask based on SEQUENCE indices, not positions
    # Shape: (batch_size, 1, seq_length, seq_length)
    # True = can attend, False = cannot attend
    causal_mask_4d = torch.tril(torch.ones(seq_length, seq_length, dtype=torch.bool, device="mps"))
    causal_mask_4d = causal_mask_4d.unsqueeze(0).unsqueeze(0)

    print(f"Causal mask shape: {causal_mask_4d.shape}")
    print(f"Mask (sequence-based, not position-based):")
    print(causal_mask_4d[0, 0].int())

    # Position IDs with gap
    position_ids = torch.cat([
        torch.arange(prompt_length, device="mps"),
        torch.tensor([prompt_length + gap], device="mps")
    ]).unsqueeze(0)

    print(f"\nPosition IDs: {position_ids[0].tolist()}")

    with torch.no_grad():
        outputs = model(
            input_ids=extended_input,
            position_ids=position_ids,
            attention_mask=causal_mask_4d,
            return_dict=True,
            use_cache=False,
            output_attentions=True,
        )

    # Check attention
    attentions = outputs.attentions[-1]
    avg_attn = attentions[0, :, -1, :].mean(dim=0).cpu().numpy()

    print(f"\nAttention from last token:")
    for i in range(len(avg_attn)):
        pos = position_ids[0, i].item()
        print(f"  Seq[{i}] Pos[{pos:3d}]: {avg_attn[i]:.6f}")

    attn_to_prompt = avg_attn[:prompt_length].sum()
    print(f"\nTotal attention to prompt: {attn_to_prompt:.6f}")


if __name__ == "__main__":
    prompt = "The answer is"

    print("="*80)
    print("OPTIMIZING POSITION GAP HANDLING")
    print("="*80)

    # Test different approaches
    for gap in [0, 5, 10]:
        test_cache_position_approach(prompt, gap)

    # Test proper mask creation
    for gap in [0, 5, 10, 100]:
        test_proper_mask_creation(prompt, gap)
