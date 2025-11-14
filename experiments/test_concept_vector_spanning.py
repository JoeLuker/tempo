#!/usr/bin/env python3
"""
Test if asking for position N returns tokens that encode paths from current_pos to N.

Hypothesis: When we ask "what token is at position 10?" after a prompt at position 5,
the model's parallel token options might encode different conceptual paths/thoughts
that span positions 6-10.
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_utils import load_model


def test_position_jump_tokens(
    prompt: str,
    jump_size: int,
    selection_threshold: float = 0.05,
):
    """
    Ask: What token appears at position (prompt_end + jump_size)?

    See if the parallel tokens at that position encode different conceptual paths.
    """

    model, tokenizer = load_model(
        "deepcogito/cogito-v1-preview-llama-3B",
        device="mps",
        load_tokenizer=True
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("mps")
    prompt_length = input_ids.shape[1]

    target_position = prompt_length - 1 + jump_size  # -1 because last prompt token is at prompt_length-1

    print(f"\n{'='*80}")
    print(f"Position Jump Test: jump_size={jump_size}")
    print('='*80)
    print(f"Prompt: '{prompt}'")
    print(f"Prompt tokens: {prompt_length}")
    print(f"Last prompt position: {prompt_length - 1}")
    print(f"Target position: {target_position}")
    print(f"Asking: 'What token is at position {target_position}?'")
    print(f"This spans {jump_size} positions\n")

    # Position IDs: prompt at normal positions, then we're asking about target_position
    # But we need to actually have a token there to query...
    # Let me just use normal positions for the prompt
    position_ids = torch.arange(prompt_length, device="mps").unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )

    # Get probabilities for next token
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)

    # Get parallel tokens above threshold
    above_threshold = probs >= selection_threshold
    parallel_token_ids = torch.where(above_threshold)[0]
    parallel_probs = probs[parallel_token_ids]

    # Sort
    sorted_indices = torch.argsort(parallel_probs, descending=True)
    parallel_token_ids = parallel_token_ids[sorted_indices]
    parallel_probs = parallel_probs[sorted_indices]

    num_parallel = len(parallel_token_ids)
    parallel_tokens = [tokenizer.decode([tid.item()]) for tid in parallel_token_ids]

    print(f"Found {num_parallel} parallel tokens at next position:")
    print(f"\nTop 20 tokens (these are what the model thinks could come next):")
    for i in range(min(20, num_parallel)):
        print(f"  {i+1:2d}. '{parallel_tokens[i]:25s}' (p={parallel_probs[i]:.4f})")

    # Now continue each of these for jump_size steps
    # This shows what thought/path each starting token leads to
    print(f"\n\nContinuing each path for {jump_size} tokens:")
    print("(This shows if different starting tokens lead to different 'thoughts')\n")

    paths = []

    for i in range(min(10, num_parallel)):  # Top 10
        current_ids = torch.cat([
            input_ids,
            parallel_token_ids[i].unsqueeze(0).unsqueeze(0)
        ], dim=1)

        path_tokens = [parallel_tokens[i]]

        # Generate rest of path
        for step in range(jump_size - 1):
            current_length = current_ids.shape[1]
            position_ids = torch.arange(current_length, device="mps").unsqueeze(0)
            attention_mask = torch.ones_like(current_ids)

            with torch.no_grad():
                outputs = model(
                    input_ids=current_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False,
                )

            logits = outputs.logits[0, -1, :]
            next_token_id = torch.argmax(logits).item()
            next_token = tokenizer.decode([next_token_id])
            path_tokens.append(next_token)

            current_ids = torch.cat([
                current_ids,
                torch.tensor([[next_token_id]], device="mps")
            ], dim=1)

        path_text = ''.join(path_tokens)
        paths.append(path_text)
        print(f"  Path {i+1:2d} (p={parallel_probs[i]:.3f}): '{path_text}'")

    return parallel_tokens, paths


def compare_jump_sizes():
    """Compare how jump size affects the diversity of conceptual paths."""

    prompt = "The answer is"

    jump_sizes = [1, 3, 5, 10, 20]

    print("="*80)
    print("CONCEPT VECTOR SPANNING TEST")
    print("="*80)
    print("\nHypothesis: Larger jumps → more diverse conceptual paths")
    print("because each token at position N must encode the path to get there\n")

    all_results = {}

    for jump_size in jump_sizes:
        tokens, paths = test_position_jump_tokens(
            prompt=prompt,
            jump_size=jump_size,
            selection_threshold=0.05
        )
        all_results[jump_size] = {'tokens': tokens, 'paths': paths}

    # Analysis
    print(f"\n\n{'='*80}")
    print("ANALYSIS: How Jump Size Affects Conceptual Diversity")
    print('='*80)

    for jump_size, data in all_results.items():
        paths = data['paths']
        unique_paths = len(set(paths))

        print(f"\nJump size {jump_size}:")
        print(f"  Total paths: {len(paths)}")
        print(f"  Unique paths: {unique_paths}")
        print(f"  Diversity: {unique_paths/len(paths)*100:.0f}%")

        # Show the paths
        for i, path in enumerate(paths[:5]):
            print(f"    {i+1}. '{path}'")


if __name__ == "__main__":
    compare_jump_sizes()
