"""Patch model attention layers to inject custom masks."""

import torch
from typing import Optional
from src.infrastructure.model.mask_utils import (
    prepare_custom_mask_for_generation,
    apply_custom_mask
)


def unwrap_model(model):
    """Extract base model from wrapper if present."""
    return model.model if hasattr(model, 'model') else model


def get_attention_layers(model):
    """Extract attention layers from Llama model."""
    base_model = unwrap_model(model)
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        return base_model.model.layers
    return []


class AttentionPatcher:
    """Injects custom attention masks into model forward passes."""

    def __init__(self):
        self.custom_mask = None
        self.original_forwards = {}

    def set_custom_mask(self, mask: Optional[torch.Tensor]):
        """Set custom attention mask for next forward pass."""
        self.custom_mask = mask

    def patch_model(self, model):
        """Replace attention forward methods with patched versions."""
        for layer_idx, layer in enumerate(get_attention_layers(model)):
            if hasattr(layer, 'self_attn'):
                self._patch_attention_layer(layer.self_attn, layer_idx)

    def unpatch_model(self, model):
        """Restore original attention forward methods."""
        for layer_idx, layer in enumerate(get_attention_layers(model)):
            if layer_idx in self.original_forwards:
                layer.self_attn.forward = self.original_forwards[layer_idx]
        self.original_forwards.clear()

    def _patch_attention_layer(self, attn_module, layer_idx: int):
        """Patch single attention layer to inject custom mask."""
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
            if self.custom_mask is not None:
                attention_mask = self._apply_mask(
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                    layer_idx=layer_idx
                )

            return original_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs
            )

        attn_module.forward = patched_forward

    def _apply_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        cache_position: Optional[torch.LongTensor],
        device: torch.device,
        dtype: torch.dtype,
        layer_idx: int
    ) -> torch.Tensor:
        """Apply custom mask to attention_mask."""
        try:
            custom_prepared = prepare_custom_mask_for_generation(
                full_mask=self.custom_mask,
                cache_position=cache_position,
                device=device,
                dtype=dtype
            )
            return apply_custom_mask(attention_mask, custom_prepared)

        except Exception as e:
            if layer_idx == 0:
                print(f"[PATCHER] Failed to apply mask: {e}")
            return attention_mask
