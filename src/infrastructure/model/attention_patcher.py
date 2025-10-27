"""Monkey-patch Llama attention to support custom attention masks."""

import torch
from typing import Optional
import functools


class AttentionPatcher:
    """Patches model attention layers to use custom attention masks."""

    def __init__(self):
        self.custom_mask = None
        self.original_forwards = {}

    def set_custom_mask(self, mask: Optional[torch.Tensor]):
        """Set the custom attention mask to be used."""
        self.custom_mask = mask

    def patch_model(self, model):
        """Patch all attention layers in the model."""
        # For Llama models
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer_idx, layer in enumerate(model.model.layers):
                if hasattr(layer, 'self_attn'):
                    self._patch_llama_attention(layer.self_attn, layer_idx)

    def _patch_llama_attention(self, attn_module, layer_idx: int):
        """Patch a Llama attention module."""
        # Store original forward
        original_forward = attn_module.forward
        self.original_forwards[layer_idx] = original_forward

        # Create patched forward
        @functools.wraps(original_forward)
        def patched_forward(
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[any] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs
        ):
            # If we have a custom mask, use it
            if self.custom_mask is not None:
                # Custom mask shape: [seq_len, seq_len] - full mask for all positions
                # During KV-cached generation, hidden_states is [batch, 1, hidden_dim]
                # We need to extract the relevant row(s) from the custom mask

                custom = self.custom_mask
                query_len = hidden_states.shape[1]  # Number of tokens being processed (usually 1 with KV cache)

                # Determine which row(s) of the mask to use
                # The mask is for the FULL sequence, we need the row(s) for current query position(s)
                if cache_position is not None:
                    # Use cache_position to index into mask
                    # cache_position shape: [batch, query_len] or [query_len]
                    if cache_position.dim() == 1:
                        pos_indices = cache_position
                    else:
                        pos_indices = cache_position[0]  # Assume batch=1

                    # Extract the rows for these positions
                    # custom[pos_indices] gives us [query_len, key_len]
                    if custom.dim() == 2:
                        custom_slice = custom[pos_indices, :]  # [query_len, key_len]
                        custom_slice = custom_slice.unsqueeze(0).unsqueeze(1)  # [1, 1, query_len, key_len]
                    else:
                        # Already has batch/head dims
                        custom_slice = custom[:, :, pos_indices, :]
                else:
                    # Fallback: assume we want the last row(s)
                    # Get KV sequence length from attention_mask if available
                    if attention_mask is not None:
                        kv_len = attention_mask.shape[-1]
                    elif past_key_values is not None:
                        # Try to get sequence length from past_key_values
                        if hasattr(past_key_values, 'get_seq_length'):
                            kv_len = past_key_values.get_seq_length() or custom.shape[0]
                        else:
                            kv_len = custom.shape[0]
                    else:
                        kv_len = custom.shape[0]

                    # Extract last query_len rows
                    if custom.dim() == 2:
                        start_pos = kv_len - 1  # Last position
                        custom_slice = custom[start_pos:start_pos+query_len, :kv_len]  # [query_len, kv_len]
                        custom_slice = custom_slice.unsqueeze(0).unsqueeze(1)  # [1, 1, query_len, kv_len]
                    else:
                        start_pos = kv_len - 1
                        custom_slice = custom[:, :, start_pos:start_pos+query_len, :kv_len]

                # Make sure it's on the right device and dtype
                custom_slice = custom_slice.to(hidden_states.device, hidden_states.dtype)

                # Combine with existing attention mask if present
                if attention_mask is not None:
                    # Both masks should be same shape now: [batch, 1, query_len, kv_len]
                    # Combine by taking minimum (most restrictive)
                    attention_mask = torch.minimum(attention_mask, custom_slice)
                else:
                    attention_mask = custom_slice

            # Call original forward with potentially modified mask
            return original_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs
            )

        # Replace forward method
        attn_module.forward = patched_forward

    def unpatch_model(self, model):
        """Restore original forward methods."""
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer_idx, layer in enumerate(model.model.layers):
                if layer_idx in self.original_forwards:
                    layer.self_attn.forward = self.original_forwards[layer_idx]

        self.original_forwards.clear()
