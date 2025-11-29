#!/usr/bin/env python3
"""Test the minimal Hebbian engine - memory-efficient version."""

import torch
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.hebbian.minimal_engine import MinimalHebbianEngine


def load_model(model_name: str, device: str):
    """Load model with proper memory management."""
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
    ).to(device)


def main():
    device = 'mps' if torch.backends.mps.is_available() else \
             'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = "deepcogito/cogito-v1-preview-llama-3B"
    print(f"Loading model on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = load_model(model_name, device)

    # Test 1: Basic generation with eviction
    print("\n" + "="*60)
    print("TEST 1: Basic generation with eviction")
    print("="*60)

    engine = MinimalHebbianEngine(
        model=model, tokenizer=tokenizer,
        window_size=32,  # Small window to force evictions
        update_scale=1e-4, device=device,
    )

    result = engine.generate(
        prompt="The quick brown fox",
        max_new_tokens=60,  # Enough to exceed window
        temperature=0.0,
    )

    print(f"Generated: {result['text'][:100]}...")
    print(f"Evictions: {len(result['evictions'])}, Updates: {result['total_updates']}")

    if result['evictions']:
        print("\nEviction samples (last 5):")
        for e in result['evictions'][-5:]:
            tok = tokenizer.decode([e['token_id']])
            print(f"  pos={e['position']}, importance={e['importance']:.2f}, token={repr(tok)}")

    # Test 2: Compare update scales (reuse model, reset engine each time)
    print("\n" + "="*60)
    print("TEST 2: Effect of update scale")
    print("="*60)

    prompt = "Once upon a time"
    results = {}

    for scale in [0.0, 0.01]:
        # Reload model for clean comparison
        del model
        model = load_model(model_name, device)

        engine = MinimalHebbianEngine(
            model=model, tokenizer=tokenizer,
            window_size=24,  # Small window to force evictions
            update_scale=scale, device=device,
        )

        result = engine.generate(prompt=prompt, max_new_tokens=50, temperature=0.0)
        perp = sum(result['perplexity_curve']) / len(result['perplexity_curve']) if result['perplexity_curve'] else 0
        results[scale] = (perp, result['text'], result['total_updates'])

        print(f"\nScale {scale}:")
        print(f"  Perplexity: {perp:.3f}, Updates: {result['total_updates']}")
        print(f"  Output: {result['text'][:60]}...")

    # Compare outputs
    baseline_text = results[0.0][1]
    hebbian_text = results[0.01][1]
    baseline_perp = results[0.0][0]
    hebbian_perp = results[0.01][0]

    diff = ((hebbian_perp - baseline_perp) / baseline_perp) * 100 if baseline_perp > 0 else 0
    print(f"\nPerplexity change: {diff:+.2f}% ({'better' if diff < 0 else 'worse'})")
    print(f"Outputs match: {baseline_text == hebbian_text}")

    print("\n" + "="*60)
    print("Tests complete!")
    print("="*60)


if __name__ == "__main__":
    main()
