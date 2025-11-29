"""
Hooks into actual attention layers to capture K, V, and attention weights.

The math:
    k = W_K @ x     # key projection
    v = W_V @ x     # value projection

    Hebbian update when token evicted:
    ΔW_K += α * importance * outer(k, x)
    ΔW_V += α * importance * outer(v, x)

    This strengthens the association: "input x should produce key k"

For GQA (Grouped Query Attention):
    - num_kv_heads < num_heads
    - k_proj: (hidden_dim,) → (num_kv_heads * head_dim,)
    - The update shapes still work: outer(k, x) has shape (kv_dim, hidden_dim)
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class CapturedActivations:
    """Activations captured from a single forward pass."""
    # Per-layer, per-position: the input that went into k_proj/v_proj
    # Shape per entry: (hidden_dim,)
    layer_inputs: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)

    # Per-layer, per-position: the key vector output
    # Shape per entry: (num_kv_heads * head_dim,)
    layer_keys: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)

    # Per-layer, per-position: the value vector output
    # Shape per entry: (num_kv_heads * head_dim,)
    layer_values: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)

    # Per-layer: full attention weights from last forward
    # Shape per entry: (batch, num_heads, seq_len, seq_len)
    attention_weights: Dict[int, torch.Tensor] = field(default_factory=dict)


class AttentionHooks:
    """
    Hook into transformer attention layers to capture real K, V, and attention.

    Works with Llama-style architectures (including Mistral, Cogito, etc.)
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.captures = CapturedActivations()
        self.enabled = False

        # Find attention layers
        self.attention_layers = self._find_attention_layers()
        logger.info(f"Found {len(self.attention_layers)} attention layers to hook")

    def _find_attention_layers(self) -> List[Tuple[int, torch.nn.Module]]:
        """Find all attention modules in the model."""
        layers = []

        # Llama/Mistral style
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for idx, layer in enumerate(self.model.model.layers):
                if hasattr(layer, 'self_attn'):
                    layers.append((idx, layer.self_attn))

        # GPT2 style
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            for idx, layer in enumerate(self.model.transformer.h):
                if hasattr(layer, 'attn'):
                    layers.append((idx, layer.attn))

        return layers

    def register_hooks(self) -> None:
        """Register forward hooks on all attention layers."""
        if self.hooks:
            self.remove_hooks()

        for layer_idx, attn_module in self.attention_layers:
            # Hook the k_proj to capture input and key output
            if hasattr(attn_module, 'k_proj'):
                hook = attn_module.k_proj.register_forward_hook(
                    self._make_kv_hook(layer_idx, 'key')
                )
                self.hooks.append(hook)

            # Hook the v_proj to capture value output
            if hasattr(attn_module, 'v_proj'):
                hook = attn_module.v_proj.register_forward_hook(
                    self._make_kv_hook(layer_idx, 'value')
                )
                self.hooks.append(hook)

        self.enabled = True
        logger.debug(f"Registered {len(self.hooks)} hooks")

    def _make_kv_hook(self, layer_idx: int, which: str) -> Callable:
        """Create a hook function for k_proj or v_proj."""
        def hook(module, input_tuple, output):
            if not self.enabled:
                return

            # input_tuple[0] is the hidden states: (batch, seq_len, hidden_dim)
            hidden_states = input_tuple[0]

            # output is the projected vectors: (batch, seq_len, kv_dim)
            projected = output

            batch_size, seq_len, _ = hidden_states.shape

            # Initialize storage for this layer
            if layer_idx not in self.captures.layer_inputs:
                self.captures.layer_inputs[layer_idx] = {}
            if layer_idx not in self.captures.layer_keys:
                self.captures.layer_keys[layer_idx] = {}
            if layer_idx not in self.captures.layer_values:
                self.captures.layer_values[layer_idx] = {}

            # Store per-position (assuming batch=1 for inference)
            for pos in range(seq_len):
                # Input that produced this position's K/V
                self.captures.layer_inputs[layer_idx][pos] = hidden_states[0, pos].detach().clone()

                if which == 'key':
                    self.captures.layer_keys[layer_idx][pos] = projected[0, pos].detach().clone()
                else:
                    self.captures.layer_values[layer_idx][pos] = projected[0, pos].detach().clone()

        return hook

    def capture_attention_weights(self, attention_outputs: Tuple) -> None:
        """
        Capture attention weights from model output.

        Args:
            attention_outputs: The attentions tuple from model output
                              Each element: (batch, num_heads, seq_len, seq_len)
        """
        if attention_outputs is None:
            return

        for layer_idx, attn in enumerate(attention_outputs):
            self.captures.attention_weights[layer_idx] = attn.detach().clone()

    def get_token_data(
        self,
        position: int
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Get captured K, V, and input for a specific position across all layers.

        Args:
            position: Token position to retrieve

        Returns:
            Dict mapping layer_idx -> (key, value, input) tensors
        """
        result = {}

        for layer_idx in self.captures.layer_inputs:
            if position in self.captures.layer_inputs[layer_idx]:
                result[layer_idx] = (
                    self.captures.layer_keys[layer_idx].get(position),
                    self.captures.layer_values[layer_idx].get(position),
                    self.captures.layer_inputs[layer_idx].get(position),
                )

        return result

    def get_cumulative_attention(self, position: int) -> float:
        """
        Compute cumulative attention received by a position.

        Sums attention weights pointing TO this position, across all layers and heads.

        Args:
            position: Token position

        Returns:
            Total attention received
        """
        total = 0.0

        for layer_idx, attn in self.captures.attention_weights.items():
            # attn: (batch, num_heads, seq_len, seq_len)
            # We want attention TO position, so we look at [:, :, :, position]
            if position < attn.size(-1):
                incoming = attn[0, :, :, position].sum().item()
                total += incoming

        return total

    def clear_position(self, position: int) -> None:
        """Remove captured data for a position (after eviction)."""
        for layer_idx in list(self.captures.layer_inputs.keys()):
            self.captures.layer_inputs[layer_idx].pop(position, None)
            self.captures.layer_keys[layer_idx].pop(position, None)
            self.captures.layer_values[layer_idx].pop(position, None)

    def clear_all(self) -> None:
        """Clear all captured activations."""
        self.captures = CapturedActivations()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.enabled = False

    @contextmanager
    def capture_mode(self):
        """Context manager for capturing activations."""
        self.register_hooks()
        try:
            yield self
        finally:
            self.remove_hooks()


class HebbianWeightUpdater:
    """
    Apply Hebbian updates to actual weight matrices.

    The math:
        When token at position p is evicted with importance I:

        For each layer:
            k_p = k_proj(x_p)   # captured during forward
            v_p = v_proj(x_p)   # captured during forward

            ΔW_K = I * outer(k_p, x_p)   # shape: (kv_dim, hidden_dim)
            ΔW_V = I * outer(v_p, x_p)   # shape: (kv_dim, hidden_dim)

            W_K += scale * ΔW_K
            W_V += scale * ΔW_V

    The scale factor normalizes by:
        - Weight matrix norm (to prevent explosion)
        - Update norm (for stability)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scale: float = 1e-6,
        normalize_by_weight_norm: bool = True,
        clip_update_ratio: float = 0.001,  # Max update as fraction of weight norm
    ):
        """
        Args:
            model: The transformer model
            scale: Base scaling factor for updates
            normalize_by_weight_norm: If True, scale update relative to weight magnitude
            clip_update_ratio: Maximum update norm as fraction of weight norm
        """
        self.model = model
        self.scale = scale
        self.normalize_by_weight_norm = normalize_by_weight_norm
        self.clip_update_ratio = clip_update_ratio

        # Find projection weights
        self.weight_refs = self._find_projection_weights()

        # Track update statistics
        self.total_updates = 0
        self.total_update_norm = 0.0
        self.max_update_ratio = 0.0

    def _find_projection_weights(self) -> Dict[int, Dict[str, torch.nn.Parameter]]:
        """Find k_proj and v_proj weights for each layer."""
        weights = {}

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for idx, layer in enumerate(self.model.model.layers):
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    weights[idx] = {}
                    if hasattr(attn, 'k_proj'):
                        weights[idx]['k'] = attn.k_proj.weight
                    if hasattr(attn, 'v_proj'):
                        weights[idx]['v'] = attn.v_proj.weight

        return weights

    def apply_update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        input_hidden: torch.Tensor,
        importance: float,
    ) -> Tuple[float, float]:
        """
        Apply Hebbian update for an evicted token.

        Args:
            layer_idx: Which layer
            key: Key vector that was produced, shape (kv_dim,)
            value: Value vector that was produced, shape (kv_dim,)
            input_hidden: Input that produced them, shape (hidden_dim,)
            importance: Cumulative attention (gates update magnitude)

        Returns:
            (k_update_ratio, v_update_ratio) - update norms relative to weight norms
        """
        if layer_idx not in self.weight_refs:
            return 0.0, 0.0

        refs = self.weight_refs[layer_idx]
        k_ratio, v_ratio = 0.0, 0.0

        with torch.no_grad():
            # Update key projection
            if 'k' in refs and key is not None:
                k_ratio = self._apply_single_update(
                    refs['k'], key, input_hidden, importance
                )

            # Update value projection
            if 'v' in refs and value is not None:
                v_ratio = self._apply_single_update(
                    refs['v'], value, input_hidden, importance
                )

        self.total_updates += 1
        self.max_update_ratio = max(self.max_update_ratio, k_ratio, v_ratio)

        return k_ratio, v_ratio

    def _apply_single_update(
        self,
        weight: torch.nn.Parameter,
        output_vec: torch.Tensor,
        input_vec: torch.Tensor,
        importance: float,
    ) -> float:
        """
        Apply single Hebbian update to a weight matrix.

        ΔW = importance * outer(output, input)
        W += scale * ΔW (with clipping)

        Returns update ratio (update_norm / weight_norm)
        """
        # Ensure same device and dtype
        output_vec = output_vec.to(weight.device, weight.dtype)
        input_vec = input_vec.to(weight.device, weight.dtype)

        # Compute outer product: (out_dim,) x (in_dim,) -> (out_dim, in_dim)
        update = torch.outer(output_vec, input_vec)

        # Scale by importance (attention determines magnitude)
        update = importance * update

        # Compute norms
        weight_norm = weight.norm().item()
        update_norm = update.norm().item()

        if weight_norm == 0 or update_norm == 0:
            return 0.0

        # Normalize update if requested
        if self.normalize_by_weight_norm:
            update = update / update_norm * weight_norm
            update_norm = weight_norm

        # Clip update to prevent explosion
        max_update_norm = weight_norm * self.clip_update_ratio
        if update_norm > max_update_norm:
            update = update * (max_update_norm / update_norm)
            update_norm = max_update_norm

        # Apply scaled update
        weight.add_(update, alpha=self.scale)

        ratio = update_norm / weight_norm
        self.total_update_norm += update_norm

        return ratio

    def get_stats(self) -> dict:
        """Get update statistics."""
        return {
            "total_updates": self.total_updates,
            "total_update_norm": self.total_update_norm,
            "max_update_ratio": self.max_update_ratio,
            "scale": self.scale,
        }
