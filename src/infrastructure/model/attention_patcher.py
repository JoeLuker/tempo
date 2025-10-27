"""Monkey-patch Llama attention to support custom attention masks."""

import torch
from typing import Optional
from src.infrastructure.model.mask_utils import prepare_mask_for_kv_cache, combine_masks


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
        # Unwrap TEMPOModelWrapper or other wrappers to get to the actual model
        actual_model = model
        if hasattr(model, 'model'):
            actual_model = model.model

        # For Llama models (LlamaForCausalLM has .model.layers)
        if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
            layers = actual_model.model.layers
            for layer_idx, layer in enumerate(layers):
                if hasattr(layer, 'self_attn'):
                    self._patch_llama_attention(layer.self_attn, layer_idx)

    def _patch_llama_attention(self, attn_module, layer_idx: int):
        """Patch a Llama attention module."""
        original_forward = attn_module.forward
        self.original_forwards[layer_idx] = original_forward

        def patched_forward(
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[any] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs
        ):
            # Apply custom mask if we have one
            if self.custom_mask is not None:
                try:
                    # Prepare mask for KV-cached generation
                    custom_slice = prepare_mask_for_kv_cache(
                        full_mask=self.custom_mask,
                        cache_position=cache_position,
                        attention_mask=attention_mask,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype
                    )

                    # Combine with existing mask
                    attention_mask = combine_masks(attention_mask, custom_slice)

                except Exception as e:
                    # If masking fails, log and continue without it
                    if layer_idx == 0:
                        print(f"[PATCHER] Mask application failed: {e}")
                    pass

            # Call original forward
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
        # Unwrap to actual model
        actual_model = model
        if hasattr(model, 'model'):
            actual_model = model.model

        if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
            for layer_idx, layer in enumerate(actual_model.model.layers):
                if layer_idx in self.original_forwards:
                    layer.self_attn.forward = self.original_forwards[layer_idx]

        self.original_forwards.clear()
