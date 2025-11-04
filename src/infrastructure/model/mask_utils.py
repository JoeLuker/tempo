"""Pure functions for attention mask manipulation."""

import torch
from typing import Optional


def extract_position_from_cache(cache_position: Optional[torch.LongTensor]) -> Optional[int]:
    """Extract scalar position from cache_position tensor."""
    if cache_position is None:
        return None

    if cache_position.dim() == 0:
        return cache_position.item()
    elif cache_position.dim() == 1:
        return cache_position[0].item()
    else:
        return cache_position[0, 0].item()


def clamp_position(position: int, max_position: int) -> int:
    """Clamp position to valid range [0, max_position]."""
    return min(max(0, position), max_position)


def extract_mask_row(mask: torch.Tensor, row_index: int) -> torch.Tensor:
    """Extract a single row from mask [seq_len, seq_len] -> [seq_len]."""
    return mask[row_index, :]


def slice_to_length(tensor: torch.Tensor, length: int) -> torch.Tensor:
    """Slice tensor's last dimension to specified length."""
    return tensor[..., :length]


def reshape_to_attention_mask(mask_row: torch.Tensor) -> torch.Tensor:
    """Reshape [seq_len] to [1, 1, 1, seq_len] for attention."""
    return mask_row.unsqueeze(0).unsqueeze(0).unsqueeze(0)


def pad_mask_to_length(mask: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pad mask's last dimension to target_length with zeros (allow attention)."""
    current_length = mask.shape[-1]
    if current_length >= target_length:
        return mask

    pad_size = target_length - current_length
    return torch.nn.functional.pad(mask, (0, pad_size), value=0.0)


def align_mask_shapes(mask_a: torch.Tensor, mask_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align two masks to same shape by padding shorter one.

    Returns:
        (aligned_mask_a, aligned_mask_b) with matching shapes
    """
    len_a = mask_a.shape[-1]
    len_b = mask_b.shape[-1]

    if len_a == len_b:
        return mask_a, mask_b

    target_len = max(len_a, len_b)
    return (
        pad_mask_to_length(mask_a, target_len),
        pad_mask_to_length(mask_b, target_len)
    )


def merge_masks(mask_a: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    """Merge two masks by taking minimum (most restrictive)."""
    return torch.minimum(mask_a, mask_b)


def prepare_custom_mask_for_generation(
    full_mask: torch.Tensor,
    cache_position: Optional[torch.LongTensor],
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Prepare custom mask for a single generation step.

    Args:
        full_mask: Full attention mask [seq_len, seq_len]
        cache_position: Current position being generated
        device: Target device
        dtype: Target dtype

    Returns:
        Mask slice [1, 1, 1, seq_len] for this generation step
    """
    mask_size = full_mask.shape[0]

    # Extract position
    position = extract_position_from_cache(cache_position)
    if position is None:
        position = mask_size - 1

    # Clamp to valid range
    position = clamp_position(position, mask_size - 1)

    # Extract row, slice, and reshape
    mask_row = extract_mask_row(full_mask, position)
    mask_row = slice_to_length(mask_row, mask_size)
    mask_reshaped = reshape_to_attention_mask(mask_row)

    # Move to target device/dtype
    return mask_reshaped.to(device, dtype)


def apply_custom_mask(
    causal_mask: Optional[torch.Tensor],
    custom_mask: torch.Tensor
) -> torch.Tensor:
    """
    Apply custom mask on top of causal mask.

    Args:
        causal_mask: Existing causal mask from model (or None)
        custom_mask: Custom mask to apply

    Returns:
        Combined mask with custom restrictions applied
    """
    if causal_mask is None:
        return custom_mask

    # Align shapes (pad shorter one)
    aligned_causal, aligned_custom = align_mask_shapes(causal_mask, custom_mask)

    # Merge (most restrictive wins)
    return merge_masks(aligned_causal, aligned_custom)
