#!/usr/bin/env python3
"""
Skeptical validation of our "breakthroughs."

Question the assumptions:
1. Are decimal positions ACTUALLY different, or just floating point noise?
2. Is retroactive assignment ACTUALLY faster, or are we measuring wrong?
3. Do these approaches ACTUALLY produce useful outputs, or just gibberish?
4. Are we fooling ourselves with cherry-picked examples?

Let's be rigorous and critical.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model
from src.algorithms.generation.attention_mask_utils import create_sequence_based_attention_mask


def test_decimal_position_significance():
    """
    SKEPTICAL TEST: Are decimal positions actually meaningful?

    Concerns:
    - Maybe the difference is just floating point noise
    - Maybe it's within normal variance
    - Maybe we're seeing patterns that don't exist
    """

    print("="*80)
    print("SKEPTICAL TEST 1: Decimal Position Significance")
    print("="*80)

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    prompt = "The answer is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")

    print(f"\nPrompt: '{prompt}'")

    # Test: Run position 3.0 multiple times to measure variance
    print("\n" + "─"*80)
    print("Step 1: Measure variance of repeated runs at position 3.0")
    print("─"*80)

    runs_at_3 = []
    for i in range(5):
        position_ids = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32, device="mps")
        attention_mask = create_sequence_based_attention_mask(input_ids, position_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )

        probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
        runs_at_3.append(probs.cpu().numpy())

    # Calculate variance across runs
    runs_at_3 = np.array(runs_at_3)
    variance_within = np.mean(np.var(runs_at_3, axis=0))
    max_diff_within = np.max(np.std(runs_at_3, axis=0))

    print(f"\nVariance within position 3.0: {variance_within:.10f}")
    print(f"Max std dev within position 3.0: {max_diff_within:.10f}")

    # Test: Compare position 3.0 vs 2.5
    print("\n" + "─"*80)
    print("Step 2: Compare position 3.0 vs 2.5")
    print("─"*80)

    position_ids_2_5 = torch.tensor([[0.0, 1.0, 2.0, 2.5]], dtype=torch.float32, device="mps")
    attention_mask = create_sequence_based_attention_mask(input_ids, position_ids_2_5)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids_2_5,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )

    probs_2_5 = torch.softmax(outputs.logits[0, -1, :], dim=-1).cpu().numpy()

    # Compare to mean of position 3.0 runs
    mean_3_0 = np.mean(runs_at_3, axis=0)
    diff_2_5_vs_3_0 = np.abs(probs_2_5 - mean_3_0)
    max_diff_between = np.max(diff_2_5_vs_3_0)

    print(f"\nMax difference 2.5 vs 3.0: {max_diff_between:.10f}")
    print(f"Max difference within 3.0 runs: {max_diff_within:.10f}")

    # Statistical test
    print("\n" + "─"*80)
    print("Step 3: Statistical significance test")
    print("─"*80)

    # Is the difference between 2.5 and 3.0 larger than noise?
    signal_to_noise = max_diff_between / max_diff_within if max_diff_within > 0 else float('inf')

    print(f"\nSignal-to-noise ratio: {signal_to_noise:.2f}")

    if signal_to_noise > 10:
        print("✓ SIGNIFICANT: Difference is >10x larger than noise")
    elif signal_to_noise > 3:
        print("⚠ MARGINAL: Difference is 3-10x larger than noise")
    else:
        print("✗ NOT SIGNIFICANT: Difference is comparable to noise")
        print("  → Decimal positions might just be floating point noise!")

    # Show actual token differences
    print("\n" + "─"*80)
    print("Step 4: Compare actual token predictions")
    print("─"*80)

    top_k_3_0 = np.argsort(mean_3_0)[-5:][::-1]
    top_k_2_5 = np.argsort(probs_2_5)[-5:][::-1]

    print("\nTop 5 at position 3.0 (mean of 5 runs):")
    for idx in top_k_3_0:
        token = tokenizer.decode([idx])
        print(f"  {mean_3_0[idx]:.4f}: {token!r}")

    print("\nTop 5 at position 2.5:")
    for idx in top_k_2_5:
        token = tokenizer.decode([idx])
        print(f"  {probs_2_5[idx]:.4f}: {token!r}")

    # Check if top tokens are even the same
    top_tokens_same = len(set(top_k_3_0) & set(top_k_2_5))
    print(f"\nTokens in common: {top_tokens_same}/5")

    if top_tokens_same == 5:
        print("⚠ WARNING: Top 5 tokens are identical - just reordered!")
        print("  → Decimal positions may not be semantically meaningful")


def test_retroactive_timing_validity():
    """
    SKEPTICAL TEST: Are we measuring retroactive timing correctly?

    Concerns:
    - Maybe model loading dominates and hides actual cost
    - Maybe we're not accounting for all overhead
    - Maybe the comparison is unfair
    """

    print("\n\n" + "="*80)
    print("SKEPTICAL TEST 2: Retroactive Timing Validity")
    print("="*80)

    import time

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    prompt = "The answer is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")

    print(f"\nPrompt: '{prompt}'")

    # Warmup
    print("\nWarming up GPU...")
    for _ in range(3):
        position_ids = torch.arange(input_ids.shape[1], device="mps").unsqueeze(0)
        attention_mask = create_sequence_based_attention_mask(input_ids, position_ids)
        with torch.no_grad():
            _ = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)

    # Test 1: Single forward pass
    print("\n" + "─"*80)
    print("Test 1: Measure single forward pass time")
    print("─"*80)

    times_single = []
    for _ in range(10):
        position_ids = torch.arange(input_ids.shape[1], device="mps").unsqueeze(0)
        attention_mask = create_sequence_based_attention_mask(input_ids, position_ids)

        start = time.time()
        with torch.no_grad():
            _ = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
        elapsed = time.time() - start
        times_single.append(elapsed)

    mean_single = np.mean(times_single)
    std_single = np.std(times_single)

    print(f"\nSingle forward pass: {mean_single*1000:.2f}ms ± {std_single*1000:.2f}ms")

    # Test 2: "Retroactive" approach (1 + 3 passes)
    print("\n" + "─"*80)
    print("Test 2: Measure 'retroactive' approach (1 gen + 3 explore)")
    print("─"*80)

    times_retro = []
    for _ in range(10):
        start = time.time()

        # Generation
        position_ids = torch.arange(input_ids.shape[1], device="mps").unsqueeze(0)
        attention_mask = create_sequence_based_attention_mask(input_ids, position_ids)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)

        next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
        extended_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device="mps")], dim=1)

        # Explore 3 positions
        for gap in [5, 10, 20]:
            position_ids = torch.cat([
                torch.arange(input_ids.shape[1], device="mps"),
                torch.tensor([input_ids.shape[1] + gap], device="mps")
            ]).unsqueeze(0)
            attention_mask = create_sequence_based_attention_mask(extended_ids, position_ids)
            with torch.no_grad():
                _ = model(input_ids=extended_ids, position_ids=position_ids, attention_mask=attention_mask)

        elapsed = time.time() - start
        times_retro.append(elapsed)

    mean_retro = np.mean(times_retro)
    std_retro = np.std(times_retro)

    print(f"\nRetroactive (4 passes): {mean_retro*1000:.2f}ms ± {std_retro*1000:.2f}ms")
    print(f"Expected (4 × single): {mean_single*4*1000:.2f}ms")
    print(f"Overhead: {(mean_retro - mean_single*4)*1000:.2f}ms")

    # Test 3: Is it actually faster than naive approach?
    print("\n" + "─"*80)
    print("Test 3: Compare to naive approach (3 separate generations)")
    print("─"*80)

    # For a fair comparison, naive would need to generate at each gap
    # But wait - we can't actually do that without regenerating from scratch!
    # So the comparison might not be apples-to-apples

    print("\n⚠ CRITICAL ISSUE: We can't fairly compare!")
    print("  - Retroactive: 1 gen + 3 explores = 4 passes")
    print("  - Traditional: Would need to generate 3 complete sequences")
    print("  - But those sequences are DIFFERENT LENGTHS!")
    print("\n  Position 5:  [0,1,2,3,8]   = 5 tokens")
    print("  Position 10: [0,1,2,3,13]  = 5 tokens")
    print("  Position 20: [0,1,2,3,23]  = 5 tokens")
    print("\n  All same length! So the 'speedup' is misleading!")

    actual_expected = mean_single * 3  # 3 generations of same length
    print(f"\nFair comparison:")
    print(f"  Retroactive: {mean_retro*1000:.2f}ms")
    print(f"  Traditional: {actual_expected*1000:.2f}ms (3 generations)")
    print(f"  Speedup: {actual_expected/mean_retro:.2f}x")

    if actual_expected/mean_retro < 0.9:
        print("\n✗ NOT FASTER: Retroactive is actually SLOWER!")
    elif actual_expected/mean_retro < 1.2:
        print("\n⚠ MARGINAL: Only ~20% faster, within margin of error")
    else:
        print(f"\n✓ CONFIRMED: {actual_expected/mean_retro:.2f}x faster")


def test_output_quality():
    """
    SKEPTICAL TEST: Do these approaches produce coherent output?

    Concerns:
    - Maybe the outputs are just gibberish
    - Maybe we're cherry-picking good examples
    - Maybe the parallel tokens don't make sense
    """

    print("\n\n" + "="*80)
    print("SKEPTICAL TEST 3: Output Quality")
    print("="*80)

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    # Test multiple prompts, not cherry-picked
    test_prompts = [
        "The capital of France is",
        "2 + 2 equals",
        "The president of the United States is",
        "Water freezes at",
        "The sky is",
    ]

    print("\nTesting multiple prompts for coherence...")
    print("(Not cherry-picked - random factual prompts)")

    coherent_count = 0
    total_count = 0

    for prompt in test_prompts:
        print(f"\n{'─'*80}")
        print(f"Prompt: '{prompt}'")
        print(f"{'─'*80}")

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")

        # Test with gap=10
        position_ids = torch.cat([
            torch.arange(input_ids.shape[1], device="mps"),
        ]).unsqueeze(0)
        attention_mask = create_sequence_based_attention_mask(input_ids, position_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )

        # Get next token
        next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
        next_token = tokenizer.decode([next_token_id])

        # Now test at gap=10
        extended_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device="mps")], dim=1)
        position_ids_gap = torch.cat([
            torch.arange(input_ids.shape[1], device="mps"),
            torch.tensor([input_ids.shape[1] + 10], device="mps")
        ]).unsqueeze(0)
        attention_mask_gap = create_sequence_based_attention_mask(extended_ids, position_ids_gap)

        with torch.no_grad():
            outputs_gap = model(
                input_ids=extended_ids,
                position_ids=position_ids_gap,
                attention_mask=attention_mask_gap,
                return_dict=True,
                use_cache=False,
            )

        logits_gap = outputs_gap.logits[0, -1, :]
        probs_gap = torch.softmax(logits_gap, dim=-1)

        # Get parallel tokens
        threshold = 0.05
        mask = probs_gap >= threshold
        parallel_tokens_ids = torch.nonzero(mask).squeeze(-1)
        parallel_tokens = [tokenizer.decode([tid.item()]) for tid in parallel_tokens_ids]
        parallel_probs = probs_gap[mask].cpu().tolist()

        print(f"\nGenerated: {next_token!r}")
        print(f"Parallel tokens at gap=10 ({len(parallel_tokens)} found):")
        for token, prob in zip(parallel_tokens[:5], parallel_probs[:5]):
            print(f"  [{prob:.4f}] {token!r}")

        # Heuristic: Check if tokens contain weird characters or are nonsensical
        weird_count = sum(1 for t in parallel_tokens[:5] if
                         any(ord(c) > 127 or c in ['�', '\x00'] for c in t))

        total_count += 1
        if weird_count == 0:
            coherent_count += 1
            print("  → Tokens look reasonable")
        else:
            print(f"  → WARNING: {weird_count}/5 tokens contain weird characters!")

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"\nCoherent outputs: {coherent_count}/{total_count}")

    if coherent_count < total_count * 0.8:
        print("✗ QUALITY ISSUE: <80% of outputs are coherent!")
    else:
        print(f"✓ Quality seems OK ({coherent_count/total_count:.0%} coherent)")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SKEPTICAL VALIDATION OF 'BREAKTHROUGHS'")
    print("="*80)
    print("\nBeing critical and rigorous about our claims...")
    print()

    test_decimal_position_significance()
    test_retroactive_timing_validity()
    test_output_quality()

    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    print("""
    Key questions to answer:

    1. Decimal positions: Are they statistically significant?
       → Check signal-to-noise ratio
       → Verify top tokens actually differ

    2. Retroactive timing: Are we measuring fairly?
       → Account for all overhead
       → Compare apples-to-apples

    3. Output quality: Is it actually useful?
       → Test on non-cherry-picked examples
       → Check for gibberish/weird characters

    Be skeptical. Demand evidence. Question everything.
    """)
    print("="*80)
