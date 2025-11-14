#!/usr/bin/env python3
"""Analyze attention patterns across different position gaps."""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model


def analyze_attention_with_gap(prompt: str, gap: int, num_tokens: int = 10):
    """Analyze attention patterns for each token generated with a gap."""

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")
    prompt_length = input_ids.shape[1]

    attention_history = []

    print(f"\n{'='*80}")
    print(f"Gap={gap}")
    print('='*80)

    for step in range(num_tokens):
        current_length = input_ids.shape[1]

        # Build position IDs
        if gap == 0:
            position_ids = torch.arange(current_length, device="mps").unsqueeze(0)
        else:
            prompt_positions = torch.arange(prompt_length, device="mps")
            gen_start = prompt_length + gap
            num_generated = current_length - prompt_length
            generated_positions = torch.arange(gen_start, gen_start + num_generated, device="mps")
            position_ids = torch.cat([prompt_positions, generated_positions]).unsqueeze(0)

        # Explicit attention mask
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
                output_attentions=True
            )

        # Analyze attention from last token
        attentions = outputs.attentions[-1]  # Last layer
        last_token_attn = attentions[0, :, -1, :].cpu().numpy()  # [num_heads, seq_len]
        avg_attn = last_token_attn.mean(axis=0)  # Average across heads

        # Calculate key metrics
        attn_to_prompt = avg_attn[:prompt_length].sum()
        attn_to_prev_generated = avg_attn[prompt_length:-1].sum() if current_length > prompt_length else 0.0
        attn_to_self = avg_attn[-1]

        # Get token info
        next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
        next_token_text = tokenizer.decode([next_token_id])

        # Get top-5 token probabilities to see distribution
        probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
        top_probs, top_indices = torch.topk(probs, k=5)
        top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]

        attention_history.append({
            'step': step,
            'token': next_token_text,
            'attn_to_prompt': attn_to_prompt,
            'attn_to_prev_gen': attn_to_prev_generated,
            'attn_to_self': attn_to_self,
            'top_prob': top_probs[0].item(),
            'entropy': -(probs * torch.log(probs + 1e-10)).sum().item(),
            'position_id': position_ids[0, -1].item()
        })

        # Print step info
        print(f"\nStep {step}: Token '{next_token_text}' (pos={position_ids[0, -1].item()})")
        print(f"  Attention: prompt={attn_to_prompt:.4f}, prev_gen={attn_to_prev_generated:.4f}, self={attn_to_self:.4f}")
        print(f"  Top prob: {top_probs[0].item():.4f}, Entropy: {attention_history[-1]['entropy']:.4f}")
        print(f"  Top-5: {', '.join([f'{tok}({prob:.3f})' for tok, prob in zip(top_tokens, top_probs)])}")

        if next_token_id == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device="mps")], dim=1)

    return attention_history


def compare_gaps(prompt: str, gaps: list, num_tokens: int = 10):
    """Compare attention patterns across multiple gaps."""

    all_results = {}

    for gap in gaps:
        history = analyze_attention_with_gap(prompt, gap, num_tokens)
        all_results[gap] = history

    # Print summary comparison
    print(f"\n\n{'='*80}")
    print("SUMMARY COMPARISON")
    print('='*80)

    for step in range(num_tokens):
        print(f"\nStep {step}:")
        for gap in gaps:
            if step < len(all_results[gap]):
                data = all_results[gap][step]
                print(f"  Gap={gap:4d}: '{data['token']:>10s}' | "
                      f"attn_prompt={data['attn_to_prompt']:.3f} | "
                      f"attn_self={data['attn_to_self']:.3f} | "
                      f"top_prob={data['top_prob']:.3f} | "
                      f"entropy={data['entropy']:.2f}")

    # Identify where breakdown occurs
    print(f"\n\n{'='*80}")
    print("BREAKDOWN ANALYSIS")
    print('='*80)

    for gap in gaps:
        if gap == 0:
            continue

        print(f"\nGap={gap}:")

        # Check when attention to prompt drops below threshold
        attn_drops = []
        for i, data in enumerate(all_results[gap]):
            if data['attn_to_prompt'] < 0.1:
                attn_drops.append(i)

        if attn_drops:
            print(f"  Attention to prompt <0.1 at steps: {attn_drops}")
            print(f"  First drop at step {attn_drops[0]}")
        else:
            print(f"  Attention to prompt stays >0.1")

        # Check for repetition
        tokens = [data['token'] for data in all_results[gap]]
        if len(tokens) > 2 and len(set(tokens[-3:])) == 1:
            print(f"  Repetition detected: '{tokens[-1]}' repeats")

        # Check entropy collapse
        entropies = [data['entropy'] for data in all_results[gap]]
        if len(entropies) > 0:
            avg_entropy = np.mean(entropies)
            print(f"  Average entropy: {avg_entropy:.2f}")
            if avg_entropy < 2.0:
                print(f"  ⚠️  Low entropy - model is very certain (possibly collapsed)")


if __name__ == "__main__":
    prompt = "We've been talking for"

    print("="*80)
    print("ATTENTION PATTERN ANALYSIS ACROSS POSITION GAPS")
    print("="*80)
    print(f"Prompt: '{prompt}'")

    # Test gaps: 0, 5, 10, 20, 50, 100
    gaps = [0, 5, 10, 20, 50, 100]

    compare_gaps(prompt, gaps, num_tokens=10)

    print("\n" + "="*80)
