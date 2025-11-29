"""MLX-based Hebbian consolidation engine.

This module provides the main engine for running Hebbian consolidation
on Apple Silicon using MLX. It loads models via mlx-lm and implements
a custom forward pass that:
1. Uses custom attention to expose attention weights
2. Applies Hebbian modifications to K projections
3. Tracks importance via attention patterns
4. Evicts old positions and creates new modifications
"""

import logging
from dataclasses import dataclass
from typing import Generator

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.tokenizer_utils import TokenizerWrapper

from ..config import HebbianConfig, HEBBIAN
from .attention import attention_with_weights, create_causal_mask
from .cache import HebbianKVCache
from .modifications import HebbianModifications

logger = logging.getLogger(__name__)


@dataclass
class TokenState:
    """State for a single token position."""
    position: int
    token_id: int
    hidden: mx.array  # Input hidden state for Hebbian update
    # Attention pattern: which positions this token attended to (averaged across layers/heads)
    # Shape: (n_attended_positions,) with position indices
    attended_positions: list[int] = None
    # Attention weights for those positions (for weighting the associations)
    attention_weights: mx.array = None


class HebbianMLXEngine:
    """MLX-based Hebbian consolidation engine.

    This engine loads a model via mlx-lm and runs inference with
    Hebbian consolidation. Key features:
    - Custom attention that exposes weights for importance tracking
    - Sparse KV cache with eviction for sliding window
    - Batched Hebbian modifications applied during forward pass

    Attributes:
        model_name: HuggingFace model identifier
        config: Hebbian consolidation configuration
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3-4B-4bit",
        config: HebbianConfig = HEBBIAN,
    ):
        """Initialize the engine.

        Args:
            model_name: HuggingFace model identifier (preferably MLX-quantized)
            config: Hebbian consolidation configuration
        """
        self.model_name = model_name
        self.config = config

        logger.info(f"Loading model: {model_name}")
        self.model, self.tokenizer = load(model_name)

        # Extract model architecture info
        self.args = self.model.args
        self.n_layers = self.args.num_hidden_layers
        self.n_heads = self.args.num_attention_heads
        self.n_kv_heads = self.args.num_key_value_heads
        self.hidden_dim = self.args.hidden_size
        self.head_dim = self.args.head_dim

        logger.info(
            f"Model loaded: {self.n_layers} layers, "
            f"{self.n_heads} heads, {self.n_kv_heads} kv_heads, "
            f"hidden_dim={self.hidden_dim}, head_dim={self.head_dim}"
        )

        # Initialize Hebbian components
        self._init_hebbian_state()

        # Compute K projection dimension
        self.k_dim = self.n_kv_heads * self.head_dim

        logger.info(
            f"Hebbian config: scale={config.update_scale}, "
            f"window={config.window_size}, max_mods={config.max_mods_per_layer}"
        )

    def _init_hebbian_state(self) -> None:
        """Initialize or reset Hebbian tracking state."""
        self.kv_cache = HebbianKVCache(
            n_layers=self.n_layers,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            window_size=self.config.window_size,
            n_sink_tokens=self.config.n_sink_tokens,
        )

        # K-projection modifications
        self.k_modifications: dict[int, HebbianModifications] = {
            layer: HebbianModifications(
                k_dim=self.n_kv_heads * self.head_dim,
                hidden_dim=self.hidden_dim,
                max_mods=self.config.max_mods_per_layer,
            )
            for layer in range(self.n_layers)
        }

        # V-projection modifications (same dimensions as K for GQA models)
        self.v_modifications: dict[int, HebbianModifications] = {
            layer: HebbianModifications(
                k_dim=self.n_kv_heads * self.head_dim,  # V has same dim as K
                hidden_dim=self.hidden_dim,
                max_mods=self.config.max_mods_per_layer,
            )
            for layer in range(self.n_layers)
        }

        # Alias for backward compatibility
        self.modifications = self.k_modifications

        # Token state tracking
        self.slots: dict[int, TokenState] = {}
        self.next_position: int = 0

        # Store ALL hidden states for attention-based associations
        # When token A evicts and attended to token B, we need B's hidden state
        self.all_hidden_states: dict[int, mx.array] = {}

    def clear(self) -> None:
        """Clear all state for a new generation."""
        self._init_hebbian_state()

    def _get_layer(self, layer_idx: int):
        """Get a transformer layer by index."""
        return self.model.model.layers[layer_idx]

    def _apply_rope(
        self,
        x: mx.array,
        positions: list[int],
        rope: nn.RoPE,
    ) -> mx.array:
        """Apply rotary position embeddings.

        Args:
            x: Tensor of shape (batch, n_heads, seq_len, head_dim)
            positions: List of absolute positions
            rope: RoPE module from the layer

        Returns:
            Tensor with RoPE applied
        """
        # RoPE expects (batch, seq, n_heads, head_dim)
        x = mx.transpose(x, (0, 2, 1, 3))

        # Get the offset (minimum position)
        offset = min(positions) if positions else 0

        # Apply RoPE with offset
        x = rope(x, offset=offset)

        # Transpose back to (batch, n_heads, seq, head_dim)
        return mx.transpose(x, (0, 2, 1, 3))

    def _forward_layer(
        self,
        layer_idx: int,
        hidden: mx.array,
        positions: list[int],
        mask: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        """Forward pass through a single layer with Hebbian modifications.

        Args:
            layer_idx: Layer index
            hidden: Input hidden states (batch, seq_len, hidden_dim)
            positions: Absolute positions for each token
            mask: Attention mask

        Returns:
            output: Output hidden states
            attn_weights: Attention weights for importance tracking
        """
        layer = self._get_layer(layer_idx)
        B, L, D = hidden.shape

        # Input layer norm
        normed = layer["input_layernorm"](hidden)

        # Q, K, V projections
        q = layer["self_attn"]["q_proj"](normed)
        k_base = layer["self_attn"]["k_proj"](normed)
        v_base = layer["self_attn"]["v_proj"](normed)

        # Apply Hebbian modifications based on config
        if self.config.update_scale > 0:
            target = self.config.update_target

            # K modifications
            if target in ("k", "both"):
                k_delta = self.k_modifications[layer_idx].apply(normed)
                k = k_base + k_delta
            else:
                k = k_base

            # V modifications - this stores content for retrieval
            if target in ("v", "both"):
                v_delta = self.v_modifications[layer_idx].apply(normed)
                v = v_base + v_delta
            else:
                v = v_base
        else:
            k = k_base
            v = v_base

        # Reshape and transpose to (batch, heads, seq, head_dim)
        # This must be done BEFORE RoPE because RoPE uses axis=-2 as sequence
        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply QK normalization if present (Qwen3-style)
        if "q_norm" in layer["self_attn"]:
            q = layer["self_attn"]["q_norm"](q)
        if "k_norm" in layer["self_attn"]:
            k = layer["self_attn"]["k_norm"](k)

        # Apply RoPE (expects batch, heads, seq, dim with axis=-2 as sequence)
        rope = layer["self_attn"]["rope"]
        offset = min(positions) if positions else 0
        q = rope(q, offset=offset)
        k = rope(k, offset=offset)

        # Store K, V in cache for each position
        for i, pos in enumerate(positions):
            self.kv_cache.add(
                layer_idx,
                pos,
                k[0, :, i, :],  # (n_kv_heads, head_dim)
                v[0, :, i, :],
            )

        # Get all cached K, V for attention
        cached_k, cached_v, cached_positions = self.kv_cache.get_kv_for_attention(
            layer_idx
        )

        if cached_k is not None:
            # GQA: expand K, V heads to match Q heads
            if self.n_kv_heads < self.n_heads:
                repeat_factor = self.n_heads // self.n_kv_heads
                cached_k = mx.repeat(cached_k, repeat_factor, axis=1)
                cached_v = mx.repeat(cached_v, repeat_factor, axis=1)

            # Custom attention with weights
            attn_output, attn_weights = attention_with_weights(
                q, cached_k, cached_v, mask=mask
            )
        else:
            # No cache yet - shouldn't happen in practice
            attn_output = mx.zeros_like(q)
            attn_weights = mx.zeros((B, self.n_heads, L, 1))

        # Reshape attention output: (batch, heads, seq, head_dim) -> (batch, seq, dim)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Output projection
        attn_output = layer["self_attn"]["o_proj"](attn_output)

        # Residual connection
        hidden = hidden + attn_output

        # MLP block (handle different model architectures)
        if "pre_feedforward_layernorm" in layer:
            # Gemma-style: pre and post feedforward norms
            mlp_input = layer["pre_feedforward_layernorm"](hidden)
            mlp_output = layer["mlp"](mlp_input)
            if "post_feedforward_layernorm" in layer:
                mlp_output = layer["post_feedforward_layernorm"](mlp_output)
        elif "post_attention_layernorm" in layer:
            # Llama-style: single post-attention norm
            mlp_input = layer["post_attention_layernorm"](hidden)
            mlp_output = layer["mlp"](mlp_input)
        else:
            # Fallback
            mlp_output = layer["mlp"](hidden)

        hidden = hidden + mlp_output

        return hidden, attn_weights

    def _forward(
        self,
        token_ids: list[int],
        positions: list[int],
    ) -> tuple[mx.array, dict[int, mx.array]]:
        """Full forward pass through all layers.

        Args:
            token_ids: Token IDs to process
            positions: Absolute positions for each token

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            attn_weights_by_layer: Attention weights per layer
        """
        # Embed tokens
        x = mx.array([token_ids])  # (1, seq_len)
        hidden = self.model.model.embed_tokens(x)  # (1, seq_len, hidden_dim)

        # Create causal mask
        # Query positions are the new tokens (positions)
        # Key positions are the current cache + new tokens (will be added in forward_layer)
        seq_len = len(token_ids)
        all_key_positions = sorted(self.kv_cache.active_positions | set(positions))
        n_keys = len(all_key_positions)

        if n_keys > 0:
            # Build mask: (seq_len, n_keys)
            # Row i can attend to key j if positions[i] >= all_key_positions[j]
            query_pos = mx.array(positions)[:, None]  # (seq_len, 1)
            key_pos = mx.array(all_key_positions)[None, :]  # (1, n_keys)
            # Can't attend to future positions
            causal_mask = query_pos < key_pos  # True where we should NOT attend
            mask = mx.where(causal_mask, mx.array(-1e9), mx.array(0.0))
            # Expand for attention: (1, 1, seq_len, n_keys)
            mask = mask[None, None, :, :]
        else:
            mask = create_causal_mask(seq_len)
            mask = mask[None, None, :, :]

        # Store attention weights for importance tracking
        attn_weights_by_layer: dict[int, mx.array] = {}

        # Forward through all layers
        for layer_idx in range(self.n_layers):
            hidden, attn_weights = self._forward_layer(
                layer_idx, hidden, positions, mask
            )
            attn_weights_by_layer[layer_idx] = attn_weights

        # Final layer norm
        hidden = self.model.model.norm(hidden)

        # LM head
        if hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(hidden)
        else:
            # Tied embeddings
            logits = self.model.model.embed_tokens.as_linear(hidden)

        return logits, attn_weights_by_layer

    def _extract_attention_pattern(
        self,
        attn_weights_by_layer: dict[int, mx.array],
        query_position: int,
    ) -> tuple[list[int], mx.array]:
        """Extract which positions a token attended to.

        This is the KEY for correct Hebbian consolidation:
        When token A evicts, we want to store associations with positions
        A attended TO, not just A's own hidden state.

        Args:
            attn_weights_by_layer: Attention weights from forward pass
            query_position: The position we're extracting patterns for

        Returns:
            attended_positions: List of position indices this token attended to
            attention_weights: Corresponding attention weights (normalized)
        """
        if not attn_weights_by_layer:
            return [], None

        # Average attention across layers and heads
        avg_attn = None
        for layer_idx, attn in attn_weights_by_layer.items():
            # attn shape: (batch, n_heads, seq_q, seq_k)
            layer_avg = mx.mean(attn, axis=1)[0]  # (seq_q, seq_k)
            if avg_attn is None:
                avg_attn = layer_avg
            else:
                avg_attn = avg_attn + layer_avg

        avg_attn = avg_attn / len(attn_weights_by_layer)

        # Get the attention pattern for the last query (current token)
        # avg_attn shape: (seq_q, seq_k), we want the last row
        if avg_attn.shape[0] == 0:
            return [], None

        attn_pattern = avg_attn[-1, :]  # (seq_k,)

        # Get the positions that were attended to
        cached_positions = sorted(self.kv_cache.active_positions)

        # Only keep positions with significant attention (top-k or above threshold)
        # Use top-8 most attended positions to avoid storing too many associations
        n_to_keep = min(8, len(cached_positions))

        if n_to_keep == 0:
            return [], None

        # Get attention values for cached positions
        attn_values = []
        for i, pos in enumerate(cached_positions):
            if i < attn_pattern.shape[0]:
                attn_values.append((pos, float(attn_pattern[i])))

        # Sort by attention weight, keep top-k
        attn_values.sort(key=lambda x: x[1], reverse=True)
        top_k = attn_values[:n_to_keep]

        attended_positions = [pos for pos, _ in top_k]
        weights = mx.array([w for _, w in top_k])

        # Normalize weights to sum to 1
        weights = weights / (mx.sum(weights) + 1e-8)

        return attended_positions, weights

    def _update_importance(self, attn_weights_by_layer: dict[int, mx.array]) -> None:
        """Update importance scores from attention weights.

        Importance is measured by how much attention each position receives
        from subsequent tokens (averaged across layers and heads).
        """
        if not attn_weights_by_layer:
            return

        # Average attention weights across layers
        avg_attn = None
        for layer_idx, attn in attn_weights_by_layer.items():
            # attn shape: (batch, n_heads, seq_q, seq_k)
            # Average across heads
            layer_avg = mx.mean(attn, axis=1)[0]  # (seq_q, seq_k)
            if avg_attn is None:
                avg_attn = layer_avg
            else:
                avg_attn = avg_attn + layer_avg

        avg_attn = avg_attn / len(attn_weights_by_layer)

        # Sum attention received by each position (column sum)
        importance_per_pos = mx.sum(avg_attn, axis=0)  # (seq_k,)

        # Update importance for cached positions
        cached_positions = sorted(self.kv_cache.active_positions)
        for i, pos in enumerate(cached_positions):
            if i < importance_per_pos.shape[0]:
                old_imp = self.kv_cache.get_importance(pos)
                new_imp = float(importance_per_pos[i])
                # Exponential moving average
                updated = old_imp * self.config.decay + new_imp
                self.kv_cache.update_importance(pos, updated)

    def _process_token(self, token_id: int, logits_in: mx.array = None) -> tuple[mx.array, float]:
        """Process a single token - shared logic for all generation methods.

        This is the single source of truth for token processing:
        1. Forward pass
        2. Store hidden state and attention pattern
        3. Update importance scores
        4. Handle eviction

        Args:
            token_id: Token to process
            logits_in: Previous logits (for log prob calculation), None for prompt tokens

        Returns:
            logits: Output logits from this token
            log_prob: Log probability of this token (0.0 for prompt tokens)
        """
        position = self.next_position
        self.next_position += 1

        # Forward pass
        logits, attn_weights = self._forward([token_id], [position])

        # Compute log prob if we have previous logits
        log_prob = 0.0
        if logits_in is not None:
            last_logits = logits_in[0, -1, :]
            log_softmax = last_logits - mx.logsumexp(last_logits)
            log_prob = float(log_softmax[token_id])

        # Store hidden state for attention-based associations
        hidden = self.model.model.embed_tokens(mx.array([[token_id]]))[0, 0]
        self.all_hidden_states[position] = hidden

        # Extract attention pattern
        attended_positions, attention_wts = self._extract_attention_pattern(
            attn_weights, position
        )

        # Store token state
        self.slots[position] = TokenState(
            position=position,
            token_id=token_id,
            hidden=hidden,
            attended_positions=attended_positions,
            attention_weights=attention_wts,
        )

        # Update importance
        self._update_importance(attn_weights)

        # Evict if needed
        while self.kv_cache.should_evict():
            oldest = self.kv_cache.get_oldest_position()
            if oldest is not None:
                self._evict_and_consolidate(oldest)

        mx.eval(logits)
        return logits, log_prob

    def _evict_and_consolidate(self, position: int) -> None:
        """Evict a position and create Hebbian modifications.

        KEY INSIGHT: Store associations based on ATTENTION PATTERNS.
        When token A evicts and was attending to tokens B, C, D, we store:
            outer(V_A, hidden_B), outer(V_A, hidden_C), outer(V_A, hidden_D)

        This means: when future tokens have hidden states similar to B, C, D,
        they will retrieve A's value content. This is TRUE associative memory.

        Args:
            position: Position to evict
        """
        if position not in self.slots:
            return

        state = self.slots.pop(position)
        importance = self.kv_cache.get_importance(position)

        # Evict from KV cache and get the K, V values
        evicted_kv = self.kv_cache.evict(position)

        if self.config.update_scale <= 0:
            return

        target = self.config.update_target
        base_scale = self.config.update_scale * importance

        # Get which positions this token attended to
        attended_positions = state.attended_positions or []
        attention_weights = state.attention_weights

        # If no attention pattern recorded, fall back to self-association
        if not attended_positions:
            attended_positions = [position]
            attention_weights = mx.array([1.0])

        # Create Hebbian modification for each layer
        for layer_idx, (k, v) in evicted_kv.items():
            k_flat = k.flatten()
            v_flat = v.flatten()
            k_norm = mx.linalg.norm(k_flat)
            v_norm = mx.linalg.norm(v_flat)

            if float(k_norm) < 1e-8 or float(v_norm) < 1e-8:
                continue

            k_normalized = k_flat / k_norm
            v_normalized = v_flat / v_norm

            # For each position this token attended to, create an association
            for i, attended_pos in enumerate(attended_positions):
                if attended_pos not in self.all_hidden_states:
                    continue

                # Get the hidden state of the attended position
                attended_hidden = self.all_hidden_states[attended_pos]
                inp_flat = attended_hidden.flatten()
                inp_norm = mx.linalg.norm(inp_flat)

                if float(inp_norm) < 1e-8:
                    continue

                inp_normalized = inp_flat / inp_norm

                # Weight by attention: higher attention = stronger association
                attn_weight = float(attention_weights[i]) if attention_weights is not None else 1.0
                scale = base_scale * attn_weight

                # K modification: when input matches attended hidden, produce this K
                if target in ("k", "both"):
                    self.k_modifications[layer_idx].add(k_normalized, inp_normalized, scale)

                # V modification: when input matches attended hidden, produce this V
                # This is the key for memory - retrieve A's content when context matches
                if target in ("v", "both"):
                    self.v_modifications[layer_idx].add(v_normalized, inp_normalized, scale)

        # Prune old modifications if we have too many
        for layer_idx in range(self.n_layers):
            if target in ("k", "both"):
                if self.k_modifications[layer_idx].n_active > self.config.max_mods_per_layer:
                    self.k_modifications[layer_idx].prune_oldest(self.config.max_mods_per_layer)
            if target in ("v", "both"):
                if self.v_modifications[layer_idx].n_active > self.config.max_mods_per_layer:
                    self.v_modifications[layer_idx].prune_oldest(self.config.max_mods_per_layer)

    def _sample_next_token(self, logits: mx.array, temperature: float) -> int:
        """Sample next token from logits."""
        last_logits = logits[0, -1, :]
        if temperature == 0:
            return int(mx.argmax(last_logits))
        else:
            probs = mx.softmax(last_logits / temperature)
            return int(mx.random.categorical(mx.log(probs)))

    def _get_eos_id(self) -> int:
        """Get EOS token ID."""
        if hasattr(self.tokenizer, "eos_token_id"):
            return self.tokenizer.eos_token_id
        return self.tokenizer._tokenizer.eos_token_id

    def _encode_prompt(self, prompt: str) -> list[int]:
        """Encode prompt to token IDs."""
        if isinstance(self.tokenizer, TokenizerWrapper):
            return self.tokenizer.encode(prompt)
        return self.tokenizer.encode(prompt)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> Generator[tuple[int, str], None, None]:
        """Generate tokens with Hebbian consolidation.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)

        Yields:
            (token_id, token_text) tuples for each generated token
        """
        self.clear()

        # Process prompt
        prompt_tokens = self._encode_prompt(prompt)
        for token_id in prompt_tokens:
            logits, _ = self._process_token(token_id)

        # Generate
        eos_id = self._get_eos_id()
        for _ in range(max_tokens):
            next_token = self._sample_next_token(logits, temperature)
            yield next_token, self.tokenizer.decode([next_token])

            if next_token == eos_id:
                break

            logits, _ = self._process_token(next_token)

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        """Generate text and return as a single string."""
        tokens = []
        for token_id, token_text in self.generate(prompt, max_tokens, temperature):
            tokens.append(token_text)
        return "".join(tokens)

    def _reset_for_generation(self, preserve_modifications: bool = False) -> None:
        """Reset state for a new generation.

        Args:
            preserve_modifications: If True, keep Hebbian modifications but reset
                                   KV cache and position tracking (cross-generation learning)
        """
        if preserve_modifications:
            # Keep modifications but reset runtime state
            self.kv_cache = HebbianKVCache(
                n_layers=self.n_layers,
                n_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
                window_size=self.config.window_size,
                n_sink_tokens=self.config.n_sink_tokens,
            )
            self.slots = {}
            self.all_hidden_states = {}
            self.next_position = 0
        else:
            self.clear()

    def generate_with_metrics(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
        preserve_modifications: bool = False,
    ) -> dict:
        """Generate tokens and track metrics including log probabilities.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            preserve_modifications: If True, keep Hebbian modifications from
                                   previous generations (for cross-generation learning)

        Returns:
            Dictionary with:
                - text: Generated text
                - tokens: List of token IDs
                - log_probs: List of log probabilities for each token
                - perplexity: Perplexity (exp of mean negative log prob)
                - n_evictions: Number of evictions
                - n_modifications: Total modifications created
                - tokens_per_second: Generation speed
        """
        import time
        import math

        self._reset_for_generation(preserve_modifications)
        start_time = time.time()

        # Process prompt tokens
        prompt_tokens = self._encode_prompt(prompt)
        for token_id in prompt_tokens:
            logits, _ = self._process_token(token_id)

        # Generate new tokens with log probability tracking
        generated_tokens = []
        log_probs = []
        eos_id = self._get_eos_id()

        for _ in range(max_tokens):
            next_token = self._sample_next_token(logits, temperature)
            generated_tokens.append(next_token)

            if next_token == eos_id:
                break

            # Process token and get log probability
            logits, log_prob = self._process_token(next_token, logits_in=logits)
            log_probs.append(log_prob)

        elapsed = time.time() - start_time

        # Compute perplexity: exp(mean(-log_prob))
        if log_probs:
            mean_neg_log_prob = -sum(log_probs) / len(log_probs)
            perplexity = math.exp(mean_neg_log_prob)
        else:
            perplexity = float('inf')

        text = self.tokenizer.decode(generated_tokens)
        stats = self.get_stats()

        return {
            "text": text,
            "tokens": generated_tokens,
            "log_probs": log_probs,
            "perplexity": perplexity,
            "n_tokens": len(generated_tokens),
            "n_evictions": stats["positions_processed"] - stats["cache_size"],
            "n_modifications": stats["total_modifications"],
            "tokens_per_second": len(generated_tokens) / elapsed if elapsed > 0 else 0,
        }

    def get_stats(self) -> dict:
        """Get current engine statistics."""
        k_mods = sum(
            self.k_modifications[layer].n_active for layer in range(self.n_layers)
        )
        v_mods = sum(
            self.v_modifications[layer].n_active for layer in range(self.n_layers)
        )
        return {
            "cache_size": self.kv_cache.size,
            "total_modifications": k_mods + v_mods,
            "k_modifications": k_mods,
            "v_modifications": v_mods,
            "positions_processed": self.next_position,
        }
