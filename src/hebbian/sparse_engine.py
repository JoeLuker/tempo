"""
Hebbian inference with sparse positions.

Tokens keep their original positions forever.
Evicted positions leave holes - gaps in the position sequence.
New tokens append at the end.
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer

from .sparse_context import SparseContext, SparseKVCache
from .attention_hooks import HebbianWeightUpdater

logger = logging.getLogger(__name__)


class SparseHebbianEngine:
    """
    Hebbian inference with sparse position management.

    Key differences from standard inference:
    1. Positions have gaps (evicted tokens leave holes)
    2. KV cache is sparse (keyed by position, not contiguous)
    3. Position IDs explicitly passed to model
    4. Hebbian updates on eviction
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        window_size: int = 512,
        decay: float = 0.99,
        update_scale: float = 1e-6,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.device = device

        # Model config
        config = model.config
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_dim = config.hidden_size

        # Ensure attention outputs are available
        self.model.config.output_attentions = True

        # Sparse context management
        self.context = SparseContext(
            max_positions=window_size * 10,  # Allow many positions before wrap
            decay=decay,
            device=device,
        )

        # Sparse KV cache
        self.kv_cache = SparseKVCache(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            device=device,
            dtype=torch.float16 if device != 'cpu' else torch.float32,
        )

        # Weight updater
        self.updater = HebbianWeightUpdater(
            model=model,
            scale=update_scale,
            normalize_by_weight_norm=True,
            clip_update_ratio=0.001,
        )

        # Register hooks to capture K, V projections
        self.hooks = []
        self._register_kv_hooks()

        # Temporary storage for current forward pass
        self._current_keys: Dict[int, torch.Tensor] = {}
        self._current_values: Dict[int, torch.Tensor] = {}
        self._current_inputs: Dict[int, torch.Tensor] = {}

        # Stats
        self.perplexity_curve = []

        logger.info(
            f"SparseHebbianEngine: window={window_size}, decay={decay}, "
            f"scale={update_scale}, layers={self.num_layers}, kv_heads={self.num_kv_heads}"
        )

    def _register_kv_hooks(self) -> None:
        """Register hooks on K, V projections to capture outputs."""

        def make_hook(layer_idx: int, which: str):
            def hook(module, input_tuple, output):
                # input_tuple[0]: (batch, seq_len, hidden_dim)
                # output: (batch, seq_len, kv_dim)
                hidden = input_tuple[0]
                proj = output

                # Store for current forward pass
                if which == 'key':
                    self._current_keys[layer_idx] = proj.detach()
                    # Store input hidden states (same for K and V)
                    self._current_inputs[layer_idx] = hidden.detach()
                else:
                    self._current_values[layer_idx] = proj.detach()

            return hook

        # Find and hook attention layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for idx, layer in enumerate(self.model.model.layers):
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    if hasattr(attn, 'k_proj'):
                        h = attn.k_proj.register_forward_hook(make_hook(idx, 'key'))
                        self.hooks.append(h)
                    if hasattr(attn, 'v_proj'):
                        h = attn.v_proj.register_forward_hook(make_hook(idx, 'value'))
                        self.hooks.append(h)

        logger.debug(f"Registered {len(self.hooks)} KV hooks")

    def _store_kv_for_position(self, position: int, seq_idx: int) -> None:
        """Store captured K, V for a specific position."""
        for layer_idx in range(self.num_layers):
            if layer_idx in self._current_keys and layer_idx in self._current_values:
                k = self._current_keys[layer_idx][0, seq_idx]  # (kv_dim,)
                v = self._current_values[layer_idx][0, seq_idx]  # (kv_dim,)
                inp = self._current_inputs[layer_idx][0, seq_idx]  # (hidden_dim,)

                self.kv_cache.store(
                    position=position,
                    layer_idx=layer_idx,
                    key=k,
                    value=v,
                    input_hidden=inp if layer_idx == 0 else None,
                )

    def _evict_one(self) -> Optional[int]:
        """Evict lowest importance token, apply Hebbian update."""
        evict_pos = self.context.get_eviction_candidate()
        if evict_pos is None:
            return None

        # Get K, V, input for Hebbian update
        kv_data = self.kv_cache.get_for_hebbian(evict_pos)

        # Apply Hebbian updates
        slot = self.context.slots.get(evict_pos)
        importance = slot.importance if slot else 0.0

        for layer_idx, (key, value, input_hidden) in kv_data.items():
            self.updater.apply_update(
                layer_idx=layer_idx,
                key=key,
                value=value,
                input_hidden=input_hidden,
                importance=importance,
            )

        # Evict from context and cache
        self.context.evict(evict_pos)
        self.kv_cache.remove(evict_pos)

        return evict_pos

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        forced_tokens: Optional[Dict[int, int]] = None,
        temperature: float = 1.0,
        protect_prompt: bool = True,
    ) -> dict:
        """
        Generate with sparse positions and Hebbian updates.

        Args:
            prompt: Input text
            max_new_tokens: Max tokens to generate
            forced_tokens: step -> token_id for forced injection
            temperature: Sampling temp (0 = greedy)
            protect_prompt: Protect prompt from eviction

        Returns:
            Dict with text, tokens, stats
        """
        forced_tokens = forced_tokens or {}

        # Reset state
        self.context = SparseContext(
            max_positions=self.window_size * 10,
            decay=self.context.decay,
            device=self.device,
        )
        self.kv_cache.clear()
        self.perplexity_curve = []

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.size(1)

        # Add prompt tokens to context
        prompt_positions = []
        for i in range(prompt_len):
            pos = self.context.add_token(input_ids[0, i].item())
            prompt_positions.append(pos)

        if protect_prompt:
            self.context.protect(range(prompt_len))

        # Process prompt to build initial KV cache
        position_ids = torch.tensor([prompt_positions], device=self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                position_ids=position_ids,
                output_attentions=True,
                use_cache=False,
            )

        # Store KV for each prompt position
        for seq_idx, pos in enumerate(prompt_positions):
            self._store_kv_for_position(pos, seq_idx)

        # Update importance from prompt attention
        if outputs.attentions:
            avg_attn = torch.stack(outputs.attentions).mean(dim=0)
            self.context.update_importance(avg_attn)

        # Generation loop
        all_token_ids = input_ids[0].tolist()

        for step in range(max_new_tokens):
            # Evict if needed
            while self.context.num_filled() >= self.window_size:
                self._evict_one()

            # Get current filled positions
            filled_positions = self.context.get_filled_positions()

            # Build input tensors for filled positions only
            current_tokens = torch.tensor(
                [[self.context.slots[p].token_id for p in filled_positions]],
                device=self.device
            )
            current_position_ids = torch.tensor(
                [filled_positions],
                device=self.device
            )

            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    current_tokens,
                    position_ids=current_position_ids,
                    output_attentions=True,
                    use_cache=False,
                )

            logits = outputs.logits[0, -1, :]  # Last position logits

            # Perplexity
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum().item()
            self.perplexity_curve.append(entropy)

            # Update importance
            if outputs.attentions:
                avg_attn = torch.stack(outputs.attentions).mean(dim=0)
                self.context.update_importance(avg_attn)

            # Get next token
            if step in forced_tokens:
                next_token = forced_tokens[step]
            elif temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = logits.argmax().item()

            # Add new token
            new_pos = self.context.add_token(next_token)
            all_token_ids.append(next_token)

            # Store KV for new token (it's the last in current sequence)
            self._store_kv_for_position(new_pos, len(filled_positions) - 1)

            # Check EOS
            if next_token == self.tokenizer.eos_token_id:
                break

        # Decode
        full_text = self.tokenizer.decode(all_token_ids, skip_special_tokens=True)
        generated_text = full_text[len(prompt):]

        return {
            "text": generated_text,
            "tokens": all_token_ids,
            "perplexity_curve": self.perplexity_curve,
            "evictions": self.context.evictions,
            "context_stats": self.context.get_stats(),
            "updater_stats": self.updater.get_stats(),
        }

    def cleanup(self) -> None:
        """Remove hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
