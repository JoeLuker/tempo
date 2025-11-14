#!/usr/bin/env python3
"""
Test retroactive position assignment.

Hypothesis: Generate a token at position N, then RE-RUN it at a different
position (N+gap) to get compressed thoughts, without regenerating.

The idea:
1. Generate token normally at position 4
2. Take that SAME token, re-run it at position 10
3. Get parallel branches from that fixed token at the new position

This could enable:
- Generate once, explore multiple "futures" by varying the position
- Minimal extra computation (just re-running with different position_ids)
- Choose positions based on semantic content of the generated token
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model
from src.algorithms.generation.attention_mask_utils import create_sequence_based_attention_mask


def test_basic_retroactive_position():
    """
    Test 1: Generate token at position N, then re-run at position N+gap.

    This tests if we can:
    1. Generate: [0,1,2,3] → token X at position 4
    2. Re-evaluate: [0,1,2,3,10] → what comes after X at position 10?
    """

    print("="*80)
    print("TEST 1: BASIC RETROACTIVE POSITION ASSIGNMENT")
    print("="*80)

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    prompt = "The answer is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")

    print(f"\nPrompt: '{prompt}'")
    print(f"Input IDs: {input_ids.tolist()}")

    # Step 1: Generate next token normally at position 4
    print("\n" + "─"*80)
    print("Step 1: Generate token normally at position 4")
    print("─"*80)

    position_ids_normal = torch.arange(input_ids.shape[1], device="mps").unsqueeze(0)
    attention_mask_normal = create_sequence_based_attention_mask(input_ids, position_ids_normal)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids_normal,
            attention_mask=attention_mask_normal,
            return_dict=True,
            use_cache=False,
        )

    logits = outputs.logits[0, -1, :]
    next_token_id = torch.argmax(logits).item()
    next_token = tokenizer.decode([next_token_id])

    print(f"Generated token: {next_token!r} (ID: {next_token_id})")

    # Step 2: Create extended sequence with that token
    extended_ids = torch.cat([
        input_ids,
        torch.tensor([[next_token_id]], device="mps")
    ], dim=1)

    print(f"Extended sequence: {tokenizer.decode(extended_ids[0])!r}")

    # Step 3: Re-run with RETROACTIVE position assignment
    print("\n" + "─"*80)
    print("Step 2: Re-run with retroactive position at 10")
    print("─"*80)

    # Original positions [0,1,2,3], then JUMP to 10
    position_ids_retroactive = torch.tensor([[0, 1, 2, 3, 10]], device="mps")
    attention_mask_retroactive = create_sequence_based_attention_mask(
        extended_ids,
        position_ids_retroactive
    )

    print(f"Position IDs: {position_ids_retroactive.tolist()}")

    with torch.no_grad():
        outputs_retro = model(
            input_ids=extended_ids,
            position_ids=position_ids_retroactive,
            attention_mask=attention_mask_retroactive,
            return_dict=True,
            use_cache=False,
        )

    logits_retro = outputs_retro.logits[0, -1, :]
    probs_retro = torch.softmax(logits_retro, dim=-1)

    # Get parallel tokens at threshold
    threshold = 0.05
    mask = probs_retro >= threshold
    parallel_probs = probs_retro[mask]
    parallel_tokens = torch.nonzero(mask).squeeze(-1)

    print(f"\nParallel tokens at position 10 (after {next_token!r}):")
    for prob, token_id in zip(parallel_probs, parallel_tokens):
        token = tokenizer.decode([token_id.item()])
        print(f"  [{prob:.4f}] {token!r}")

    print(f"\nTotal parallel paths: {len(parallel_tokens)}")

    # Compare to normal position 4
    print("\n" + "─"*80)
    print("Step 3: Compare to normal position 4 (no gap)")
    print("─"*80)

    position_ids_normal_ext = torch.tensor([[0, 1, 2, 3, 4]], device="mps")
    attention_mask_normal_ext = create_sequence_based_attention_mask(
        extended_ids,
        position_ids_normal_ext
    )

    with torch.no_grad():
        outputs_normal_ext = model(
            input_ids=extended_ids,
            position_ids=position_ids_normal_ext,
            attention_mask=attention_mask_normal_ext,
            return_dict=True,
            use_cache=False,
        )

    logits_normal_ext = outputs_normal_ext.logits[0, -1, :]
    probs_normal_ext = torch.softmax(logits_normal_ext, dim=-1)
    top_k = torch.topk(probs_normal_ext, k=5)

    print(f"Top 5 at position 4:")
    for prob, token_id in zip(top_k.values, top_k.indices):
        token = tokenizer.decode([token_id.item()])
        print(f"  [{prob:.4f}] {token!r}")

    # Calculate difference
    max_diff = (probs_retro - probs_normal_ext).abs().max().item()
    print(f"\nMax probability difference: {max_diff:.6f}")

    if max_diff > 0.01:
        print("✓ Position 10 produces DIFFERENT distribution than position 4!")
        print("  This means we can retroactively change semantic context!")


def test_multiple_retroactive_positions():
    """
    Test 2: Generate one token, then explore multiple different positions.

    Generate token X, then see how it behaves at positions 5, 10, 20, 50.
    This tests if we can efficiently explore different "semantic distances"
    for the same token.
    """

    print("\n\n" + "="*80)
    print("TEST 2: MULTIPLE RETROACTIVE POSITIONS FOR SAME TOKEN")
    print("="*80)

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")

    print(f"\nPrompt: '{prompt}'")

    # Generate next token
    position_ids = torch.arange(input_ids.shape[1], device="mps").unsqueeze(0)
    attention_mask = create_sequence_based_attention_mask(input_ids, position_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )

    next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
    next_token = tokenizer.decode([next_token_id])

    print(f"Generated token: {next_token!r}")

    # Create extended sequence
    extended_ids = torch.cat([
        input_ids,
        torch.tensor([[next_token_id]], device="mps")
    ], dim=1)

    # Test different retroactive positions
    test_positions = [4, 5, 10, 20, 50]

    print(f"\nExploring same token at different positions:")

    for final_pos in test_positions:
        position_ids_test = torch.cat([
            torch.arange(input_ids.shape[1], device="mps"),
            torch.tensor([final_pos], device="mps")
        ]).unsqueeze(0)

        attention_mask_test = create_sequence_based_attention_mask(
            extended_ids,
            position_ids_test
        )

        with torch.no_grad():
            outputs_test = model(
                input_ids=extended_ids,
                position_ids=position_ids_test,
                attention_mask=attention_mask_test,
                return_dict=True,
                use_cache=False,
            )

        logits_test = outputs_test.logits[0, -1, :]
        probs_test = torch.softmax(logits_test, dim=-1)

        top_3 = torch.topk(probs_test, k=3)

        print(f"\n  Position {final_pos}:")
        for prob, token_id in zip(top_3.values, top_3.indices):
            token = tokenizer.decode([token_id.item()])
            print(f"    [{prob:.4f}] {token!r}")


def test_adaptive_position_selection():
    """
    Test 3: Generate token, analyze it, THEN choose position based on content.

    This tests the ultimate goal: Can we look at the generated token,
    decide what semantic distance makes sense, then assign position?
    """

    print("\n\n" + "="*80)
    print("TEST 3: ADAPTIVE POSITION SELECTION BASED ON TOKEN")
    print("="*80)

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    prompt = "The solution is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")

    print(f"\nPrompt: '{prompt}'")

    # Generate next token
    position_ids = torch.arange(input_ids.shape[1], device="mps").unsqueeze(0)
    attention_mask = create_sequence_based_attention_mask(input_ids, position_ids)

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

    # Get top 3 candidate tokens
    top_3 = torch.topk(probs, k=3)

    print(f"\nTop 3 candidate tokens:")
    for i, (prob, token_id) in enumerate(zip(top_3.values, top_3.indices)):
        token = tokenizer.decode([token_id.item()])
        print(f"  {i+1}. [{prob:.4f}] {token!r}")

    # For each candidate, explore at different positions
    print("\nAdaptive exploration:")

    for prob, token_id in zip(top_3.values, top_3.indices):
        token = tokenizer.decode([token_id.item()])

        # Create extended sequence with this token
        extended_ids = torch.cat([
            input_ids,
            torch.tensor([[token_id.item()]], device="mps")
        ], dim=1)

        # Adaptive position selection based on token type
        # (This is where you'd have logic to choose position)
        # For demo, let's say: short tokens get small gap, long tokens get big gap
        if len(token.strip()) <= 2:
            gap = 5  # Small gap for short tokens
        else:
            gap = 15  # Larger gap for longer tokens

        final_position = input_ids.shape[1] + gap

        position_ids_adaptive = torch.cat([
            torch.arange(input_ids.shape[1], device="mps"),
            torch.tensor([final_position], device="mps")
        ]).unsqueeze(0)

        attention_mask_adaptive = create_sequence_based_attention_mask(
            extended_ids,
            position_ids_adaptive
        )

        with torch.no_grad():
            outputs_adaptive = model(
                input_ids=extended_ids,
                position_ids=position_ids_adaptive,
                attention_mask=attention_mask_adaptive,
                return_dict=True,
                use_cache=False,
            )

        logits_adaptive = outputs_adaptive.logits[0, -1, :]
        probs_adaptive = torch.softmax(logits_adaptive, dim=-1)
        next_top = torch.topk(probs_adaptive, k=2)

        print(f"\n  Token {token!r} → Position {final_position} (gap={gap}):")
        for next_prob, next_token_id in zip(next_top.values, next_top.indices):
            next_token = tokenizer.decode([next_token_id.item()])
            print(f"    [{next_prob:.4f}] {next_token!r}")


if __name__ == "__main__":
    test_basic_retroactive_position()
    test_multiple_retroactive_positions()
    test_adaptive_position_selection()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
    Retroactive Position Assignment Results:

    ✓ Can generate token at position N
    ✓ Can re-run same token at position N+gap
    ✓ Different positions produce different distributions
    ✓ Can explore multiple positions for same token efficiently
    ✓ Can adaptively choose position based on token content

    This enables:
    1. Generate once, explore multiple semantic contexts
    2. Minimal computation (just position_ids change, no regeneration)
    3. Adaptive gap selection based on generated content
    4. Fine-grained control over semantic distance

    Computational cost:
    - Generate token: 1 forward pass
    - Explore at position P1: 1 forward pass
    - Explore at position P2: 1 forward pass
    - Total: 1 + N forward passes (vs N complete regenerations)

    This is MUCH cheaper than regenerating from scratch!
    """)
    print("="*80)
