#!/usr/bin/env python3
"""
Deep dive into understanding the overhead.

Questions:
1. Why is changing position_ids causing 62ms overhead?
2. Is it mask computation? Tensor operations? GPU sync?
3. Are there conditions where retroactive IS beneficial?
4. What's actually happening inside the model?
"""

import sys
from pathlib import Path
import torch
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model
from src.algorithms.generation.attention_mask_utils import create_sequence_based_attention_mask


def profile_components():
    """Break down where time is spent."""

    print("="*80)
    print("PROFILING: Where does the time go?")
    print("="*80)

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    prompt = "The answer is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")

    # Warmup
    for _ in range(5):
        position_ids = torch.arange(input_ids.shape[1], device="mps").unsqueeze(0)
        attention_mask = create_sequence_based_attention_mask(input_ids, position_ids)
        with torch.no_grad():
            _ = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)

    # Test 1: Mask creation time
    print("\n" + "─"*80)
    print("Test 1: Mask creation overhead")
    print("─"*80)

    times_mask = []
    for _ in range(100):
        position_ids = torch.arange(input_ids.shape[1], device="mps").unsqueeze(0)
        start = time.time()
        attention_mask = create_sequence_based_attention_mask(input_ids, position_ids)
        elapsed = time.time() - start
        times_mask.append(elapsed)

    print(f"\nMask creation: {sum(times_mask)/len(times_mask)*1000:.3f}ms (mean of 100)")

    # Test 2: Position IDs tensor creation
    print("\n" + "─"*80)
    print("Test 2: Position IDs creation overhead")
    print("─"*80)

    times_pos = []
    for gap in [0, 5, 10, 20]:
        for _ in range(100):
            start = time.time()
            if gap == 0:
                position_ids = torch.arange(input_ids.shape[1], device="mps").unsqueeze(0)
            else:
                position_ids = torch.cat([
                    torch.arange(input_ids.shape[1], device="mps"),
                    torch.tensor([input_ids.shape[1] + gap], device="mps")
                ]).unsqueeze(0)
            elapsed = time.time() - start
            times_pos.append(elapsed)

    print(f"\nPosition creation: {sum(times_pos)/len(times_pos)*1000:.3f}ms (mean of {len(times_pos)})")

    # Test 3: Model forward pass with different position patterns
    print("\n" + "─"*80)
    print("Test 3: Model forward pass with different positions")
    print("─"*80)

    # Generate a token first
    position_ids = torch.arange(input_ids.shape[1], device="mps").unsqueeze(0)
    attention_mask = create_sequence_based_attention_mask(input_ids, position_ids)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
    next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
    extended_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device="mps")], dim=1)

    # Test sequential positions
    print("\nSequential positions [0,1,2,3,4]:")
    times_seq = []
    for _ in range(20):
        position_ids = torch.arange(extended_ids.shape[1], device="mps").unsqueeze(0)
        attention_mask = create_sequence_based_attention_mask(extended_ids, position_ids)

        start = time.time()
        with torch.no_grad():
            _ = model(input_ids=extended_ids, position_ids=position_ids, attention_mask=attention_mask)
        elapsed = time.time() - start
        times_seq.append(elapsed)

    print(f"  Mean: {sum(times_seq)/len(times_seq)*1000:.2f}ms ± {torch.tensor(times_seq).std().item()*1000:.2f}ms")

    # Test gap positions
    for gap in [5, 10, 20]:
        print(f"\nGap positions [0,1,2,3,{3+gap}]:")
        times_gap = []
        for _ in range(20):
            position_ids = torch.cat([
                torch.arange(input_ids.shape[1], device="mps"),
                torch.tensor([input_ids.shape[1] + gap], device="mps")
            ]).unsqueeze(0)
            attention_mask = create_sequence_based_attention_mask(extended_ids, position_ids)

            start = time.time()
            with torch.no_grad():
                _ = model(input_ids=extended_ids, position_ids=position_ids, attention_mask=attention_mask)
            elapsed = time.time() - start
            times_gap.append(elapsed)

        mean_gap = sum(times_gap)/len(times_gap)
        mean_seq = sum(times_seq)/len(times_seq)
        overhead = (mean_gap - mean_seq) * 1000

        print(f"  Mean: {mean_gap*1000:.2f}ms ± {torch.tensor(times_gap).std().item()*1000:.2f}ms")
        print(f"  Overhead vs sequential: {overhead:.2f}ms ({overhead/mean_seq/10:.1f}%)")


def test_batch_benefits():
    """
    Maybe retroactive is beneficial if we batch the explorations?
    Test if we can do multiple position explorations in parallel.
    """

    print("\n\n" + "="*80)
    print("TEST: Can we batch retroactive explorations?")
    print("="*80)

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    prompt = "The answer is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")

    # Generate token
    position_ids = torch.arange(input_ids.shape[1], device="mps").unsqueeze(0)
    attention_mask = create_sequence_based_attention_mask(input_ids, position_ids)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
    next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
    extended_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device="mps")], dim=1)

    gaps = [5, 10, 20]

    # Sequential exploration (current approach)
    print("\n" + "─"*80)
    print("Sequential: Explore each gap separately")
    print("─"*80)

    start = time.time()
    for gap in gaps:
        position_ids = torch.cat([
            torch.arange(input_ids.shape[1], device="mps"),
            torch.tensor([input_ids.shape[1] + gap], device="mps")
        ]).unsqueeze(0)
        attention_mask = create_sequence_based_attention_mask(extended_ids, position_ids)
        with torch.no_grad():
            _ = model(input_ids=extended_ids, position_ids=position_ids, attention_mask=attention_mask)
    sequential_time = time.time() - start

    print(f"\nSequential time: {sequential_time*1000:.2f}ms")

    # Batched exploration
    print("\n" + "─"*80)
    print("Batched: Explore all gaps in one batch")
    print("─"*80)

    # Create batched input
    batch_input_ids = extended_ids.repeat(len(gaps), 1)
    batch_position_ids = torch.stack([
        torch.cat([
            torch.arange(input_ids.shape[1], device="mps"),
            torch.tensor([input_ids.shape[1] + gap], device="mps")
        ])
        for gap in gaps
    ])

    # Create batched attention mask
    # This is tricky - each batch item needs its own mask
    batch_masks = []
    for gap in gaps:
        pos_ids = torch.cat([
            torch.arange(input_ids.shape[1], device="mps"),
            torch.tensor([input_ids.shape[1] + gap], device="mps")
        ]).unsqueeze(0)
        mask = create_sequence_based_attention_mask(extended_ids, pos_ids)
        batch_masks.append(mask)

    batch_attention_mask = torch.cat(batch_masks, dim=0)

    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch_input_ids.shape}")
    print(f"  position_ids: {batch_position_ids.shape}")
    print(f"  attention_mask: {batch_attention_mask.shape}")

    start = time.time()
    with torch.no_grad():
        _ = model(
            input_ids=batch_input_ids,
            position_ids=batch_position_ids,
            attention_mask=batch_attention_mask
        )
    batched_time = time.time() - start

    print(f"\nBatched time: {batched_time*1000:.2f}ms")
    print(f"Sequential time: {sequential_time*1000:.2f}ms")
    print(f"Speedup: {sequential_time/batched_time:.2f}x")

    if batched_time < sequential_time:
        print(f"\n✓ BATCHING HELPS! {(1 - batched_time/sequential_time)*100:.1f}% faster")
    else:
        print(f"\n✗ BATCHING SLOWER: {(batched_time/sequential_time - 1)*100:.1f}% slower")


def test_real_world_scenario():
    """
    What's the actual use case where retroactive makes sense?
    Maybe: Generate many tokens, then explore each at multiple positions?
    """

    print("\n\n" + "="*80)
    print("TEST: Real-world scenario - explore top-k tokens at multiple gaps")
    print("="*80)

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    prompt = "The answer is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")

    # Get top-k candidate tokens
    position_ids = torch.arange(input_ids.shape[1], device="mps").unsqueeze(0)
    attention_mask = create_sequence_based_attention_mask(input_ids, position_ids)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)

    logits = outputs.logits[0, -1, :]
    top_k = torch.topk(logits, k=3)
    candidate_tokens = top_k.indices.tolist()

    print(f"\nPrompt: '{prompt}'")
    print(f"Top {len(candidate_tokens)} candidate tokens:")
    for token_id in candidate_tokens:
        token = tokenizer.decode([token_id])
        print(f"  {token!r}")

    gaps = [5, 10, 20]

    # Approach 1: For each token, explore all gaps sequentially
    print("\n" + "─"*80)
    print("Approach 1: Sequential (current retroactive)")
    print("─"*80)

    start = time.time()
    for token_id in candidate_tokens:
        extended_ids = torch.cat([input_ids, torch.tensor([[token_id]], device="mps")], dim=1)
        for gap in gaps:
            position_ids = torch.cat([
                torch.arange(input_ids.shape[1], device="mps"),
                torch.tensor([input_ids.shape[1] + gap], device="mps")
            ]).unsqueeze(0)
            attention_mask = create_sequence_based_attention_mask(extended_ids, position_ids)
            with torch.no_grad():
                _ = model(input_ids=extended_ids, position_ids=position_ids, attention_mask=attention_mask)
    sequential_time = time.time() - start

    print(f"\nSequential: {sequential_time*1000:.2f}ms")
    print(f"Total forward passes: {len(candidate_tokens) * len(gaps)}")

    # Approach 2: Batch all token-gap combinations
    print("\n" + "─"*80)
    print("Approach 2: Fully batched")
    print("─"*80)

    # Create all combinations
    batch_inputs = []
    batch_positions = []
    batch_masks = []

    for token_id in candidate_tokens:
        extended_ids = torch.cat([input_ids, torch.tensor([[token_id]], device="mps")], dim=1)
        for gap in gaps:
            position_ids = torch.cat([
                torch.arange(input_ids.shape[1], device="mps"),
                torch.tensor([input_ids.shape[1] + gap], device="mps")
            ]).unsqueeze(0)
            mask = create_sequence_based_attention_mask(extended_ids, position_ids)

            batch_inputs.append(extended_ids)
            batch_positions.append(position_ids)
            batch_masks.append(mask)

    batch_input_ids = torch.cat(batch_inputs, dim=0)
    batch_position_ids = torch.cat(batch_positions, dim=0)
    batch_attention_mask = torch.cat(batch_masks, dim=0)

    print(f"\nBatch size: {batch_input_ids.shape[0]}")

    start = time.time()
    with torch.no_grad():
        _ = model(
            input_ids=batch_input_ids,
            position_ids=batch_position_ids,
            attention_mask=batch_attention_mask
        )
    batched_time = time.time() - start

    print(f"\nBatched: {batched_time*1000:.2f}ms")
    print(f"Speedup vs sequential: {sequential_time/batched_time:.2f}x")

    if batched_time < sequential_time:
        print(f"\n✓ BATCHING IS THE KEY! {(1 - batched_time/sequential_time)*100:.1f}% faster")
        print("  → Retroactive CAN be faster if you batch properly!")
    else:
        print(f"\n✗ Even batching doesn't help: {(batched_time/sequential_time - 1)*100:.1f}% slower")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DEEP DIVE: Understanding the Overhead")
    print("="*80)
    print("\nLet's understand what's actually happening...\n")

    profile_components()
    test_batch_benefits()
    test_real_world_scenario()

    print("\n" + "="*80)
    print("INSIGHTS")
    print("="*80)
    print("""
    Key findings:

    1. Where is the overhead?
       → Measure mask creation, position tensor creation, model forward pass

    2. Can batching help?
       → Test if we can do multiple gap explorations in one batch

    3. What's the real use case?
       → When would retroactive actually be beneficial?

    Maybe the speedup comes from batching, not from retroactive assignment itself!
    """)
    print("="*80)
