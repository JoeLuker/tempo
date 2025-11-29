#!/usr/bin/env python3
"""
Test the proper Hebbian implementation with real attention hooks.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.hebbian import HebbianEngine


def main():
    device = 'mps' if torch.backends.mps.is_available() else \
             'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model on {device}...")
    model_name = "deepcogito/cogito-v1-preview-llama-3B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
        attn_implementation="eager",
    ).to(device)

    print("Model loaded. Creating HebbianEngine...")

    # Test 1: Basic generation with evictions
    print("\n" + "="*60)
    print("TEST 1: Generation with eviction")
    print("="*60)

    engine = HebbianEngine(
        model=model,
        tokenizer=tokenizer,
        window_size=64,  # Small window to force evictions
        decay=0.99,
        update_scale=1e-6,
        device=device,
    )

    result = engine.generate(
        prompt="The quick brown fox",
        max_new_tokens=100,
        temperature=0.8,
    )

    print(f"Generated: {result['text'][:200]}...")
    print(f"Evictions: {len(result['evictions'])}")
    print(f"Updater stats: {result['updater_stats']}")
    print(f"Importance stats: {result['importance_stats']}")

    if result['evictions']:
        print("\nFirst 5 evictions:")
        for e in result['evictions'][:5]:
            token_text = tokenizer.decode([e['token_id']]) if e['token_id'] else "?"
            print(f"  pos={e['position']}, importance={e['importance']:.4f}, token='{token_text}'")

    # Test 2: Compare with vs without Hebbian
    print("\n" + "="*60)
    print("TEST 2: Compare Hebbian vs No Hebbian")
    print("="*60)

    prompt = "Once upon a time in a magical kingdom, there lived"

    # With Hebbian (model already has updates from Test 1)
    result_with = engine.generate(
        prompt=prompt,
        max_new_tokens=80,
        temperature=0.7,
    )
    perp_with = sum(result_with['perplexity_curve']) / len(result_with['perplexity_curve'])

    # Without Hebbian - reload fresh model
    print("Loading fresh model for comparison...")
    model_fresh = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
        attn_implementation="eager",
    ).to(device)

    engine_fresh = HebbianEngine(
        model=model_fresh,
        tokenizer=tokenizer,
        window_size=64,
        decay=0.99,
        update_scale=0.0,  # No updates
        device=device,
    )

    result_without = engine_fresh.generate(
        prompt=prompt,
        max_new_tokens=80,
        temperature=0.7,
    )
    perp_without = sum(result_without['perplexity_curve']) / len(result_without['perplexity_curve'])

    diff = (perp_with - perp_without) / perp_without * 100

    print(f"\nWith Hebbian updates: perplexity = {perp_with:.4f}")
    print(f"Without Hebbian: perplexity = {perp_without:.4f}")
    print(f"Difference: {diff:+.2f}% ({'better' if diff < 0 else 'worse'})")

    print(f"\nTotal weight updates applied: {result_with['updater_stats']['total_updates']}")
    print(f"Max update ratio: {result_with['updater_stats']['max_update_ratio']:.6f}")

    # Test 3: Pattern learning
    print("\n" + "="*60)
    print("TEST 3: Pattern learning")
    print("="*60)

    engine_pattern = HebbianEngine(
        model=model_fresh,  # Start fresh
        tokenizer=tokenizer,
        window_size=128,
        decay=0.99,
        update_scale=1e-6,
        device=device,
    )

    pattern = "XYZXYZXYZ" * 5
    result_pattern = engine_pattern.generate(
        prompt=pattern,
        max_new_tokens=50,
        temperature=0.5,
    )

    perp_start = sum(result_pattern['perplexity_curve'][:10]) / 10 if len(result_pattern['perplexity_curve']) >= 10 else 0
    perp_end = sum(result_pattern['perplexity_curve'][-10:]) / 10 if len(result_pattern['perplexity_curve']) >= 10 else 0

    print(f"Pattern continuation: {result_pattern['text'][:100]}")
    print(f"Perplexity start: {perp_start:.4f}")
    print(f"Perplexity end: {perp_end:.4f}")
    print(f"Change: {((perp_end - perp_start) / perp_start * 100):+.1f}%")

    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)


if __name__ == "__main__":
    main()
