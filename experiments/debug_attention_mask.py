#!/usr/bin/env python3
"""Debug what attention mask is being used."""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model


def debug_attention_mask(prompt: str, gap: int):
    """Check what attention mask the model sees."""

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")
    prompt_length = input_ids.shape[1]

    # Generate first token
    position_ids = torch.arange(prompt_length, device="mps").unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            return_dict=True,
            use_cache=False,
        )

    next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
    all_input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device="mps")], dim=1)

    # Build position IDs for second token
    current_length = all_input_ids.shape[1]

    if gap == 0:
        prompt_positions = torch.arange(prompt_length, device="mps")
        gen_start = prompt_length
        num_generated = current_length - prompt_length
        generated_positions = torch.arange(gen_start, gen_start + num_generated, device="mps")
        position_ids = torch.cat([prompt_positions, generated_positions]).unsqueeze(0)
    else:
        prompt_positions = torch.arange(prompt_length, device="mps")
        gen_start = prompt_length + gap
        num_generated = current_length - prompt_length
        generated_positions = torch.arange(gen_start, gen_start + num_generated, device="mps")
        position_ids = torch.cat([prompt_positions, generated_positions]).unsqueeze(0)

    print(f"\nGap={gap}:")
    print(f"  Input IDs shape: {all_input_ids.shape}")
    print(f"  Position IDs: {position_ids[0].tolist()}")

    # Try calling with explicit attention_mask
    attention_mask = torch.ones_like(all_input_ids)
    print(f"  Attention mask: {attention_mask[0].tolist()} (all ones)")

    # Let's also try WITHOUT position_ids to see what happens
    print("\n  Test 1: WITH position_ids, WITH attention_mask")
    with torch.no_grad():
        outputs1 = model(
            input_ids=all_input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
            output_attentions=True
        )

    attentions1 = outputs1.attentions[-1]
    avg_attn1 = attentions1[0, :, -1, :].mean(dim=0).cpu().numpy()
    print(f"    Attention to prompt: {avg_attn1[:prompt_length].sum():.6f}")

    print("\n  Test 2: WITHOUT position_ids, WITH attention_mask")
    with torch.no_grad():
        outputs2 = model(
            input_ids=all_input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
            output_attentions=True
        )

    attentions2 = outputs2.attentions[-1]
    avg_attn2 = attentions2[0, :, -1, :].mean(dim=0).cpu().numpy()
    print(f"    Attention to prompt: {avg_attn2[:prompt_length].sum():.6f}")

    print("\n  Test 3: WITH position_ids, WITHOUT attention_mask (None)")
    with torch.no_grad():
        outputs3 = model(
            input_ids=all_input_ids,
            position_ids=position_ids,
            attention_mask=None,
            return_dict=True,
            use_cache=False,
            output_attentions=True
        )

    attentions3 = outputs3.attentions[-1]
    avg_attn3 = attentions3[0, :, -1, :].mean(dim=0).cpu().numpy()
    print(f"    Attention to prompt: {avg_attn3[:prompt_length].sum():.6f}")

    # Check if model is creating its own causal mask based on position_ids
    print("\n  Checking internal mask creation...")
    print(f"    Does model config have 'is_causal'? {hasattr(model.config, 'is_causal')}")
    if hasattr(model.config, 'is_causal'):
        print(f"    is_causal: {model.config.is_causal}")


if __name__ == "__main__":
    prompt = "We've been talking for"

    print("="*80)
    print("DEBUGGING ATTENTION MASK")
    print("="*80)

    debug_attention_mask(prompt, gap=0)
    debug_attention_mask(prompt, gap=1)

    print("\n" + "="*80)
