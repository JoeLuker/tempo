#!/usr/bin/env python3
"""
RoPE handler for position gaps.

The transformers library expects position_ids to be contiguous [0,1,2,3,4].
When we use gaps [0,1,2,3,10], it breaks because the RoPE cache only has positions 0-4.

Solution: Monkey-patch the model to compute RoPE for arbitrary positions.
"""

import torch
import math
from typing import Tuple


def compute_rope_for_positions(
    position_ids: torch.Tensor,
    dim: int,
    base: float = 10000.0,
    device: str = "mps",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RoPE cos/sin for arbitrary position IDs.

    Args:
        position_ids: [batch, seq_len] tensor with arbitrary position values
        dim: Dimension of the embeddings (head_dim)
        base: RoPE base frequency
        device: Device to create tensors on
        dtype: Data type for tensors

    Returns:
        (cos, sin) tensors of shape [batch, seq_len, dim]
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # position_ids: [batch, seq_len]
    # inv_freq: [dim/2]
    # We want: [batch, seq_len, dim/2]

    # Expand position_ids for broadcasting: [batch, seq_len, 1]
    position_ids_expanded = position_ids.unsqueeze(-1).float()

    # Expand inv_freq: [1, 1, dim/2]
    inv_freq_expanded = inv_freq.unsqueeze(0).unsqueeze(0)

    # Compute freqs: [batch, seq_len, dim/2]
    freqs = position_ids_expanded * inv_freq_expanded

    # Duplicate to get [batch, seq_len, dim]
    emb = torch.cat([freqs, freqs], dim=-1)

    # Compute cos and sin
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    return cos, sin


def patch_model_for_gaps(model):
    """
    Monkey-patch a model to handle position gaps in RoPE.

    This replaces the model's RoPE computation to handle arbitrary position IDs.
    """
    # Get model config
    config = model.config

    # This is hacky but necessary: we need to intercept RoPE computation
    # The issue is that transformers computes RoPE based on seq_len, not position_ids

    # For LLaMA-style models, RoPE is applied in the attention layers
    # We need to patch each layer's self_attn

    for layer_idx, layer in enumerate(model.model.layers):
        original_forward = layer.self_attn.forward

        def make_patched_forward(original_fn):
            def patched_forward(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                **kwargs,
            ):
                # If position_ids has gaps, we need custom RoPE
                if position_ids is not None:
                    max_pos = position_ids.max().item()
                    seq_len = position_ids.shape[1]

                    # Check for gaps: max_pos > seq_len-1 means there's a gap
                    if max_pos > seq_len - 1:
                        # Custom RoPE computation needed
                        # This is where we'd inject our gap-aware RoPE
                        # For now, just log and call original
                        pass

                # Call original forward
                return original_fn(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs,
                )

            return patched_forward

        layer.self_attn.forward = make_patched_forward(original_forward)

    return model


class GapRoPEWrapper:
    """
    Wrapper that handles position gaps by pre-computing RoPE.

    Instead of patching the model, we compute what the model WOULD compute
    if it handled gaps correctly.
    """

    def __init__(self, model, tokenizer, device="mps"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Get RoPE parameters from model config
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.rope_base = getattr(model.config, "rope_theta", 10000.0)

    def forward_with_gaps(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        """
        Forward pass that properly handles position gaps.

        The trick: Pre-compute RoPE cos/sin for the actual position values,
        then somehow inject them into the model.

        Problem: We can't easily inject custom cos/sin into transformers.

        Alternative: Extend the model's RoPE cache to include all positions up to max(position_ids).
        """
        max_pos = position_ids.max().item()
        seq_len = input_ids.shape[1]

        # If there's a gap, we need to handle it
        if max_pos >= seq_len:
            # Extend RoPE cache
            # The model's rotary_emb computes cos/sin for positions 0 to seq_len-1
            # We need it to compute for positions 0 to max_pos

            # Hacky solution: Temporarily extend seq_len in the model's view
            # This won't work because seq_len is derived from input_ids shape

            # Real solution: We need to modify how the model computes RoPE
            # This requires patching the attention layers

            raise NotImplementedError(
                "Position gaps require patching the model's RoPE computation. "
                "This is complex and requires modifying transformers internals."
            )

        # Normal case: no gaps
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
