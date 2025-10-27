"""Pure functions for attention mask manipulation."""

import torch
from typing import Optional


def get_mask_row_for_position(
    full_mask: torch.Tensor,
    position: int,
    kv_length: int
) -> torch.Tensor:
    """
    Extract a single row from a full attention mask for a specific position.

    Args:
        full_mask: Full mask [seq_len, seq_len]
        position: Which position's mask to extract
        kv_length: Length of keys/values

    Returns:
        Mask for this position: [1, 1, 1, kv_length]
    """
    mask_size = full_mask.shape[0]

    # Clamp position to valid range
    pos = min(position, mask_size - 1)

    # Get the row for this position
    mask_row = full_mask[pos, :kv_length]  # [kv_length]

    # Reshape to [batch=1, heads=1, query=1, kv_length]
    return mask_row.unsqueeze(0).unsqueeze(0).unsqueeze(0)


def combine_masks(
    causal_mask: Optional[torch.Tensor],
    custom_mask: torch.Tensor
) -> torch.Tensor:
    """
    Combine causal and custom masks by taking the minimum (most restrictive).

    Args:
        causal_mask: Existing causal mask or None
        custom_mask: Custom mask to apply

    Returns:
        Combined mask
    """
    if causal_mask is None:
        return custom_mask

    # If shapes don't match, pad custom mask to match causal mask size
    if causal_mask.shape != custom_mask.shape:
        causal_shape = causal_mask.shape
        custom_shape = custom_mask.shape

        # Pad custom mask to match causal mask
        # Padding with 0 (allow attention) for positions beyond our mask
        if causal_shape[-1] > custom_shape[-1]:
            pad_size = causal_shape[-1] - custom_shape[-1]
            # Pad on the right (last dimension)
            custom_padded = torch.nn.functional.pad(
                custom_mask,
                (0, pad_size),  # Pad right side of last dim
                value=0.0  # 0 = allow attention
            )
            return torch.minimum(causal_mask, custom_padded)
        else:
            # Custom mask is larger, trim it
            custom_trimmed = custom_mask[..., :causal_shape[-1]]
            return torch.minimum(causal_mask, custom_trimmed)

    return torch.minimum(causal_mask, custom_mask)


def prepare_mask_for_kv_cache(
    full_mask: torch.Tensor,
    cache_position: Optional[torch.LongTensor],
    attention_mask: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Prepare a full mask for KV-cached generation.

    During KV caching, we only process 1 new token but it needs to attend
    to all previous tokens. Extract the appropriate row from the full mask.

    Args:
        full_mask: Full attention mask [seq_len, seq_len]
        cache_position: Position(s) being processed
        attention_mask: Existing attention mask (tells us KV length)
        device: Target device
        dtype: Target dtype

    Returns:
        Mask slice ready for this forward pass
    """
    mask_size = full_mask.shape[0]

    # Determine which position we're processing
    if cache_position is not None:
        if cache_position.dim() == 0:  # Scalar
            pos = cache_position.item()
        elif cache_position.dim() == 1:
            pos = cache_position[0].item()  # Take first
        else:
            pos = cache_position[0, 0].item()  # [batch, seq]
    else:
        # Fallback: assume we're processing the last position
        pos = mask_size - 1

    # Clamp position to mask bounds
    pos = min(pos, mask_size - 1)

    # KV length should match the mask's second dimension
    # Since we're extracting from a [seq_len, seq_len] mask, use mask_size
    kv_len = mask_size

    # Extract row and reshape
    mask_slice = get_mask_row_for_position(full_mask, pos, kv_len)

    # Move to correct device and dtype
    return mask_slice.to(device, dtype)
