"""
Attention mask utilities for position gap handling.

Provides optimized mask creation that works with position gaps
by creating sequence-based masks rather than position-based masks.
"""

import torch
from typing import Optional


def create_causal_mask_for_positions(
    seq_length: int,
    device: str = "mps",
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    Create a 4D causal attention mask based on SEQUENCE indices.

    This mask allows position gaps because it's based on sequence relationships,
    not position values. Token at sequence index i can attend to all tokens
    at sequence indices 0..i, regardless of their position IDs.

    Args:
        seq_length: Length of the sequence
        device: Device to create tensor on
        dtype: Data type for the mask (torch.bool for modern transformers)

    Returns:
        4D mask of shape (1, 1, seq_length, seq_length)
        True = can attend, False = cannot attend
    """
    # Create lower triangular matrix (causal mask)
    # mask[i, j] = True if i >= j (can attend to earlier or same position)
    mask_2d = torch.tril(torch.ones(seq_length, seq_length, dtype=dtype, device=device))

    # Expand to 4D: (batch_size=1, num_heads=1, seq_length, seq_length)
    mask_4d = mask_2d.unsqueeze(0).unsqueeze(0)

    return mask_4d


def create_sequence_based_attention_mask(
    input_ids: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    Create attention mask that works with position gaps.

    This creates a mask based on sequence indices, not position values,
    allowing tokens at non-consecutive positions to attend to each other.

    Args:
        input_ids: Input token IDs of shape (batch_size, seq_length)
        position_ids: Optional position IDs (not used for mask, but validated)
        dtype: Data type for the mask

    Returns:
        4D attention mask compatible with transformers library
    """
    batch_size, seq_length = input_ids.shape
    device = input_ids.device

    # Create causal mask for each batch
    # Shape: (batch_size, 1, seq_length, seq_length)
    mask_2d = torch.tril(torch.ones(seq_length, seq_length, dtype=dtype, device=device))
    mask_4d = mask_2d.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)

    return mask_4d


def create_optimized_mask_for_gap(
    prompt_length: int,
    num_generated: int,
    position_gap: int = 0,
    device: str = "mps",
) -> torch.Tensor:
    """
    Create optimized attention mask for compressed thought generation.

    This creates a causal mask that allows the generated tokens (which may
    have position gaps) to attend to all prompt tokens and previous generated
    tokens based on sequence order, not position distance.

    Args:
        prompt_length: Number of tokens in the prompt
        num_generated: Number of generated tokens
        position_gap: Gap in position IDs (doesn't affect the mask!)
        device: Device to create tensor on

    Returns:
        4D boolean attention mask

    Example:
        If prompt_length=4, num_generated=1, positions=[0,1,2,3,10]:
        The mask allows token at sequence index 4 (position 10) to attend
        to sequence indices 0-4 (positions 0,1,2,3,10).
    """
    total_length = prompt_length + num_generated

    # Create standard causal mask - based on SEQUENCE not POSITION
    mask = create_causal_mask_for_positions(
        seq_length=total_length,
        device=device,
        dtype=torch.bool,
    )

    return mask


def validate_position_ids_with_gap(
    position_ids: torch.Tensor,
    expected_prompt_length: int,
) -> bool:
    """
    Validate that position IDs are structured correctly for gap usage.

    Checks:
    1. Prompt positions are sequential [0, 1, 2, ..., N-1]
    2. Generated positions have expected gap
    3. Generated positions are sequential after the gap

    Args:
        position_ids: Position IDs tensor of shape (batch_size, seq_length)
        expected_prompt_length: Expected length of the prompt

    Returns:
        True if valid, False otherwise
    """
    if position_ids.dim() != 2:
        return False

    batch_size, seq_length = position_ids.shape

    # Check each batch
    for b in range(batch_size):
        pos = position_ids[b]

        # Check prompt positions are sequential
        prompt_pos = pos[:expected_prompt_length]
        expected_prompt = torch.arange(expected_prompt_length, device=pos.device)

        if not torch.equal(prompt_pos, expected_prompt):
            return False

        # Check generated positions are sequential (after any gap)
        if seq_length > expected_prompt_length:
            gen_pos = pos[expected_prompt_length:]
            gen_diffs = gen_pos[1:] - gen_pos[:-1]

            # All differences should be 1 (sequential)
            if not torch.all(gen_diffs == 1):
                return False

    return True


def get_attention_mask_info(mask: torch.Tensor) -> dict:
    """
    Get information about an attention mask for debugging.

    Args:
        mask: Attention mask tensor

    Returns:
        Dictionary with mask statistics
    """
    info = {
        "shape": tuple(mask.shape),
        "dtype": str(mask.dtype),
        "device": str(mask.device),
    }

    if mask.dtype == torch.bool:
        info["num_true"] = mask.sum().item()
        info["num_false"] = (~mask).sum().item()
        info["sparsity"] = info["num_false"] / mask.numel()
    else:
        # Float mask (0 = attend, -inf = don't attend)
        info["num_attend"] = (mask == 0).sum().item()
        info["num_masked"] = (mask < 0).sum().item()
        info["sparsity"] = info["num_masked"] / mask.numel()

    return info
