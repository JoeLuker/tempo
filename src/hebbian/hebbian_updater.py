"""Hebbian weight updates - rank-one modifications on token eviction."""

import torch
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class UpdateStats:
    """Statistics about weight updates."""
    total_updates: int = 0
    total_importance_mass: float = 0.0
    updates_per_layer: Dict[int, int] = field(default_factory=dict)
    max_update_norm: float = 0.0
    weight_norm_before: Dict[str, float] = field(default_factory=dict)
    weight_norm_after: Dict[str, float] = field(default_factory=dict)


class HebbianUpdater:
    """
    Apply Hebbian weight updates when tokens are evicted.

    When token i is evicted:
        ΔW_K += α * importance_i * outer(k_i, x_i)
        ΔW_V += α * importance_i * outer(v_i, x_i)

    Where:
        - k_i, v_i are the token's key/value vectors
        - x_i is the input that produced this token
        - importance_i is cumulative attention score
        - α is learning rate

    The intuition: tokens that received more attention leave deeper marks.
    This is "what fired together, wired together" - no loss function needed.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        alpha: float = 1e-5,
        attention_scaled: bool = True,
        normalize_update: bool = True,
        clip_update_norm: Optional[float] = 0.01,
        update_keys: bool = True,
        update_values: bool = True,
    ):
        """
        Args:
            model: The transformer model to update
            alpha: Base scaling factor (when attention_scaled=False) or
                   scaling multiplier for attention-derived rate (when attention_scaled=True)
            attention_scaled: If True, learning rate = attention_importance * alpha
                             The attention score directly determines update magnitude.
                             If False, learning rate = alpha (fixed).
            normalize_update: If True, normalize outer product before applying
            clip_update_norm: Max norm for update tensor (None = no clipping)
            update_keys: Whether to update key projection weights
            update_values: Whether to update value projection weights
        """
        self.model = model
        self.alpha = alpha
        self.attention_scaled = attention_scaled
        self.normalize_update = normalize_update
        self.clip_update_norm = clip_update_norm
        self.update_keys = update_keys
        self.update_values = update_values

        # Find the attention layers
        self.layer_info = self._find_attention_layers()
        logger.info(f"Found {len(self.layer_info)} attention layers for Hebbian updates")

        # Track statistics
        self.stats = UpdateStats()

        # Store original weight norms for monitoring drift
        self._record_initial_norms()

    def _find_attention_layers(self) -> List[dict]:
        """Find all attention layers and their weight matrices."""
        layers = []

        # Try common architectures
        # Llama-style
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for idx, layer in enumerate(self.model.model.layers):
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    layers.append({
                        'idx': idx,
                        'k_proj': attn.k_proj if hasattr(attn, 'k_proj') else None,
                        'v_proj': attn.v_proj if hasattr(attn, 'v_proj') else None,
                        'name': f'layer_{idx}'
                    })

        # GPT2-style
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            for idx, layer in enumerate(self.model.transformer.h):
                if hasattr(layer, 'attn'):
                    attn = layer.attn
                    # GPT2 uses combined c_attn, harder to update separately
                    layers.append({
                        'idx': idx,
                        'c_attn': attn.c_attn if hasattr(attn, 'c_attn') else None,
                        'name': f'layer_{idx}'
                    })

        return layers

    def _record_initial_norms(self) -> None:
        """Record initial weight norms for monitoring."""
        for layer in self.layer_info:
            if layer.get('k_proj'):
                name = f"{layer['name']}_k"
                self.stats.weight_norm_before[name] = layer['k_proj'].weight.norm().item()
            if layer.get('v_proj'):
                name = f"{layer['name']}_v"
                self.stats.weight_norm_before[name] = layer['v_proj'].weight.norm().item()

    def consolidate(
        self,
        layer_idx: int,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        input_embedding: torch.Tensor,
        importance: float
    ) -> Tuple[float, float]:
        """
        Apply Hebbian update for an evicted token.

        Args:
            layer_idx: Which transformer layer
            key_vector: The evicted token's key [hidden_dim] or [heads, head_dim]
            value_vector: The evicted token's value [hidden_dim] or [heads, head_dim]
            input_embedding: Input that produced this token [hidden_dim]
            importance: Cumulative attention score (gates update magnitude)

        Returns:
            Tuple of (key_update_norm, value_update_norm)
        """
        if layer_idx >= len(self.layer_info):
            logger.warning(f"Layer {layer_idx} not found, skipping update")
            return 0.0, 0.0

        layer = self.layer_info[layer_idx]

        # Flatten if needed (multi-head → single vector)
        k = key_vector.flatten() if key_vector.dim() > 1 else key_vector
        v = value_vector.flatten() if value_vector.dim() > 1 else value_vector
        x = input_embedding.flatten() if input_embedding.dim() > 1 else input_embedding

        # Compute learning rate from attention
        # When attention_scaled=True: lr = importance * alpha (attention determines magnitude)
        # When attention_scaled=False: lr = alpha (fixed rate)
        if self.attention_scaled:
            effective_alpha = importance * self.alpha
        else:
            effective_alpha = self.alpha

        k_norm, v_norm = 0.0, 0.0

        with torch.no_grad():
            # Update key projection
            if self.update_keys and layer.get('k_proj'):
                k_norm = self._apply_update(layer['k_proj'].weight, k, x, effective_alpha)

            # Update value projection
            if self.update_values and layer.get('v_proj'):
                v_norm = self._apply_update(layer['v_proj'].weight, v, x, effective_alpha)

        # Track stats
        self.stats.total_updates += 1
        self.stats.total_importance_mass += importance
        self.stats.updates_per_layer[layer_idx] = self.stats.updates_per_layer.get(layer_idx, 0) + 1
        self.stats.max_update_norm = max(self.stats.max_update_norm, k_norm, v_norm)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Layer {layer_idx} update: importance={importance:.4f}, "
                f"k_norm={k_norm:.6f}, v_norm={v_norm:.6f}"
            )

        return k_norm, v_norm

    def _apply_update(
        self,
        weight: torch.Tensor,
        output_vec: torch.Tensor,
        input_vec: torch.Tensor,
        alpha: float
    ) -> float:
        """
        Apply rank-one update: W += α * outer(output, input)

        Args:
            weight: Weight matrix [out_dim, in_dim]
            output_vec: Output vector [out_dim]
            input_vec: Input vector [in_dim]
            alpha: Gated learning rate

        Returns:
            Norm of the update
        """
        # Ensure same device and dtype
        output_vec = output_vec.to(weight.device, weight.dtype)
        input_vec = input_vec.to(weight.device, weight.dtype)

        # Handle dimension mismatch (common with multi-head attention)
        if output_vec.size(0) != weight.size(0):
            # Reshape or truncate
            if output_vec.size(0) > weight.size(0):
                output_vec = output_vec[:weight.size(0)]
            else:
                # Pad with zeros
                padded = torch.zeros(weight.size(0), device=weight.device, dtype=weight.dtype)
                padded[:output_vec.size(0)] = output_vec
                output_vec = padded

        if input_vec.size(0) != weight.size(1):
            if input_vec.size(0) > weight.size(1):
                input_vec = input_vec[:weight.size(1)]
            else:
                padded = torch.zeros(weight.size(1), device=weight.device, dtype=weight.dtype)
                padded[:input_vec.size(0)] = input_vec
                input_vec = padded

        # Compute outer product
        update = torch.outer(output_vec, input_vec)

        # Normalize if requested
        if self.normalize_update:
            update_norm = update.norm()
            if update_norm > 0:
                update = update / update_norm

        # Clip if requested
        if self.clip_update_norm is not None:
            update_norm = update.norm()
            if update_norm > self.clip_update_norm:
                update = update * (self.clip_update_norm / update_norm)

        # Apply update
        weight.add_(update, alpha=alpha)

        return update.norm().item()

    def get_weight_drift(self) -> Dict[str, float]:
        """
        Calculate how much weights have drifted from initial values.

        Returns:
            Dict mapping layer names to drift percentages
        """
        drift = {}
        for layer in self.layer_info:
            if layer.get('k_proj'):
                name = f"{layer['name']}_k"
                current = layer['k_proj'].weight.norm().item()
                initial = self.stats.weight_norm_before.get(name, current)
                drift[name] = abs(current - initial) / initial * 100 if initial > 0 else 0

            if layer.get('v_proj'):
                name = f"{layer['name']}_v"
                current = layer['v_proj'].weight.norm().item()
                initial = self.stats.weight_norm_before.get(name, current)
                drift[name] = abs(current - initial) / initial * 100 if initial > 0 else 0

        return drift

    def get_stats(self) -> dict:
        """Get update statistics."""
        drift = self.get_weight_drift()
        return {
            "total_updates": self.stats.total_updates,
            "total_importance_mass": self.stats.total_importance_mass,
            "max_update_norm": self.stats.max_update_norm,
            "mean_drift_pct": sum(drift.values()) / len(drift) if drift else 0,
            "max_drift_pct": max(drift.values()) if drift else 0,
            "updates_per_layer": dict(self.stats.updates_per_layer),
        }

    def reset_stats(self) -> None:
        """Reset statistics (but not weight changes)."""
        self.stats = UpdateStats()
        self._record_initial_norms()
