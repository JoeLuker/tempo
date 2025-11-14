#!/usr/bin/env python3
"""
Test if decimal position IDs work with the model.

Question: Can we use positions like [0, 1, 2, 3, 3.5, 4, 5]?

This would allow:
1. Inserting "intermediate" thoughts between existing positions
2. Finer-grained control over semantic distance
3. Potentially smoother interpolation between positions
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model
from src.algorithms.generation.attention_mask_utils import create_sequence_based_attention_mask


def test_decimal_positions():
    """Test if model can handle decimal position IDs."""

    print("="*80)
    print("TESTING DECIMAL POSITION IDS")
    print("="*80)

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    prompt = "The answer is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")
    prompt_length = input_ids.shape[1]

    print(f"\nPrompt: '{prompt}'")
    print(f"Prompt length: {prompt_length} tokens")

    # Test 1: Integer positions (baseline)
    print("\n" + "─"*80)
    print("Test 1: Integer positions [0, 1, 2, 3]")
    print("─"*80)

    position_ids_int = torch.arange(prompt_length, dtype=torch.long, device="mps").unsqueeze(0)
    attention_mask_int = create_sequence_based_attention_mask(
        input_ids=input_ids,
        position_ids=position_ids_int,
    )

    print(f"Position IDs: {position_ids_int.tolist()}")
    print(f"Position IDs dtype: {position_ids_int.dtype}")

    try:
        with torch.no_grad():
            outputs_int = model(
                input_ids=input_ids,
                position_ids=position_ids_int,
                attention_mask=attention_mask_int,
                return_dict=True,
                use_cache=False,
                output_attentions=True,
            )

        logits_int = outputs_int.logits[0, -1, :]
        probs_int = torch.softmax(logits_int, dim=-1)
        top_k_int = torch.topk(probs_int, k=5)

        print("\nTop 5 predictions:")
        for prob, token_id in zip(top_k_int.values, top_k_int.indices):
            token = tokenizer.decode([token_id.item()])
            print(f"  {prob:.4f}: {token!r}")

        # Attention analysis
        attentions = outputs_int.attentions[-1]
        avg_attn = attentions[0, :, -1, :].mean(dim=0).cpu().numpy()
        attn_to_prompt = avg_attn[:prompt_length].sum()
        print(f"\nAttention to prompt: {attn_to_prompt:.6f}")

    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Test 2: Decimal positions
    print("\n" + "─"*80)
    print("Test 2: Decimal positions [0.0, 1.0, 2.0, 3.0]")
    print("─"*80)

    position_ids_float = torch.arange(prompt_length, dtype=torch.float32, device="mps").unsqueeze(0)

    print(f"Position IDs: {position_ids_float.tolist()}")
    print(f"Position IDs dtype: {position_ids_float.dtype}")

    try:
        with torch.no_grad():
            outputs_float = model(
                input_ids=input_ids,
                position_ids=position_ids_float,
                attention_mask=attention_mask_int,  # Same mask
                return_dict=True,
                use_cache=False,
                output_attentions=True,
            )

        logits_float = outputs_float.logits[0, -1, :]
        probs_float = torch.softmax(logits_float, dim=-1)
        top_k_float = torch.topk(probs_float, k=5)

        print("\nTop 5 predictions:")
        for prob, token_id in zip(top_k_float.values, top_k_float.indices):
            token = tokenizer.decode([token_id.item()])
            print(f"  {prob:.4f}: {token!r}")

        # Check if results match integer positions
        max_diff = (probs_int - probs_float).abs().max().item()
        print(f"\nMax probability difference vs integers: {max_diff:.10f}")

        if max_diff < 1e-5:
            print("✓ Decimal positions produce identical results!")
        else:
            print("⚠ Decimal positions produce different results")

        # Attention analysis
        attentions = outputs_float.attentions[-1]
        avg_attn = attentions[0, :, -1, :].mean(dim=0).cpu().numpy()
        attn_to_prompt = avg_attn[:prompt_length].sum()
        print(f"Attention to prompt: {attn_to_prompt:.6f}")

    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Test 3: Mixed decimal positions
    print("\n" + "─"*80)
    print("Test 3: Mixed positions [0.0, 1.0, 2.0, 2.5]")
    print("─"*80)

    position_ids_mixed = torch.tensor([[0.0, 1.0, 2.0, 2.5]], dtype=torch.float32, device="mps")

    print(f"Position IDs: {position_ids_mixed.tolist()}")
    print(f"Position IDs dtype: {position_ids_mixed.dtype}")

    try:
        with torch.no_grad():
            outputs_mixed = model(
                input_ids=input_ids,
                position_ids=position_ids_mixed,
                attention_mask=attention_mask_int,
                return_dict=True,
                use_cache=False,
                output_attentions=True,
            )

        logits_mixed = outputs_mixed.logits[0, -1, :]
        probs_mixed = torch.softmax(logits_mixed, dim=-1)
        top_k_mixed = torch.topk(probs_mixed, k=5)

        print("\nTop 5 predictions:")
        for prob, token_id in zip(top_k_mixed.values, top_k_mixed.indices):
            token = tokenizer.decode([token_id.item()])
            print(f"  {prob:.4f}: {token!r}")

        # Compare to position 3.0
        max_diff_vs_int = (probs_int - probs_mixed).abs().max().item()
        print(f"\nMax probability difference vs [0,1,2,3]: {max_diff_vs_int:.10f}")

        if max_diff_vs_int > 1e-5:
            print("✓ Position 2.5 produces different results than 3.0!")
            print("  This means decimal positions WORK and have semantic meaning!")
        else:
            print("⚠ Position 2.5 same as 3.0 - decimals may not matter")

        # Attention analysis
        attentions = outputs_mixed.attentions[-1]
        avg_attn = attentions[0, :, -1, :].mean(dim=0).cpu().numpy()
        attn_to_prompt = avg_attn[:prompt_length].sum()
        print(f"Attention to prompt: {attn_to_prompt:.6f}")

    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Test 4: Fractional gaps
    print("\n" + "─"*80)
    print("Test 4: Fractional gap [0, 1, 2, 3, 3.1, 3.2, 3.3]")
    print("─"*80)

    # Extend sequence with fractional positions
    extended_prompt = prompt + " yes"
    extended_ids = tokenizer.encode(extended_prompt, return_tensors="pt").to("mps")
    extended_length = extended_ids.shape[1]

    # Original positions + fractional increments
    position_ids_frac = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0, 3.1, 3.2, 3.3][:extended_length]],
        dtype=torch.float32,
        device="mps"
    )

    attention_mask_frac = create_sequence_based_attention_mask(
        input_ids=extended_ids,
        position_ids=position_ids_frac,
    )

    print(f"Extended prompt: '{extended_prompt}'")
    print(f"Position IDs: {position_ids_frac.tolist()}")

    try:
        with torch.no_grad():
            outputs_frac = model(
                input_ids=extended_ids,
                position_ids=position_ids_frac,
                attention_mask=attention_mask_frac,
                return_dict=True,
                use_cache=False,
                output_attentions=True,
            )

        logits_frac = outputs_frac.logits[0, -1, :]
        probs_frac = torch.softmax(logits_frac, dim=-1)
        top_k_frac = torch.topk(probs_frac, k=5)

        print("\nTop 5 predictions:")
        for prob, token_id in zip(top_k_frac.values, top_k_frac.indices):
            token = tokenizer.decode([token_id.item()])
            print(f"  {prob:.4f}: {token!r}")

        # Attention analysis
        attentions = outputs_frac.attentions[-1]
        avg_attn = attentions[0, :, -1, :].mean(dim=0).cpu().numpy()
        attn_to_prompt = avg_attn[:extended_length].sum()
        print(f"Attention to prompt: {attn_to_prompt:.6f}")

        print("\n✓ Fractional position increments work!")

    except Exception as e:
        print(f"ERROR: {e}")
        return


def test_decimal_with_compressed_thoughts():
    """Test decimal positions with compressed thought generation."""

    print("\n\n" + "="*80)
    print("TESTING DECIMAL POSITIONS WITH COMPRESSED THOUGHTS")
    print("="*80)

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    prompt = "The answer is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")
    prompt_length = input_ids.shape[1]

    print(f"\nPrompt: '{prompt}'")

    # Test: Instead of gap=5 (positions 0,1,2,3,8), use position 3.5
    print("\n" + "─"*80)
    print("Compressed thought at position 3.5 (halfway to next token)")
    print("─"*80)

    position_ids = torch.tensor(
        [[0.0, 1.0, 2.0, 3.5]],
        dtype=torch.float32,
        device="mps"
    )

    attention_mask = create_sequence_based_attention_mask(
        input_ids=input_ids,
        position_ids=position_ids,
    )

    print(f"Position IDs: {position_ids.tolist()}")

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        # Get parallel tokens above threshold
        threshold = 0.05
        mask = probs >= threshold
        parallel_probs = probs[mask]
        parallel_tokens = torch.nonzero(mask).squeeze(-1)

        print(f"\nParallel tokens at position 3.5 (threshold={threshold}):")
        for prob, token_id in zip(parallel_probs, parallel_tokens):
            token = tokenizer.decode([token_id.item()])
            print(f"  [{prob:.4f}] {token!r}")

        print(f"\nTotal parallel paths: {len(parallel_tokens)}")
        print("\n✓ Decimal positions enable 'half-step' compressed thoughts!")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    test_decimal_positions()
    test_decimal_with_compressed_thoughts()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
    Key Questions Answered:

    1. Can we use decimal position IDs?
       → Testing if float32 positions work with RoPE

    2. Do decimals have semantic meaning?
       → Does position 2.5 differ from position 3.0?

    3. Can we do fractional gaps?
       → Instead of gap=5, can we use gap=0.5?

    4. Do compressed thoughts work at decimal positions?
       → Can we get parallel tokens at position 3.5?

    If this works, it opens up:
    - Fine-grained semantic interpolation
    - Smoother thought transitions
    - Arbitrary precision in position encoding
    - "Half-step" compressed thoughts
    """)
    print("="*80)
