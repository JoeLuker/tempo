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

logger = logging.getLogger(__name__)


@dataclass
class TokenState:
    """State for a single token position."""
    position: int
    token_id: int


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
            f"Hebbian config: memory={config.memory_enabled}, "
            f"window={config.window_size}, max_memory={config.max_memory_per_layer}"
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

        # Token state tracking
        self.slots: dict[int, TokenState] = {}
        self.next_position: int = 0

        # Think block tracking - tokens inside <think>...</think> are not stored
        self._in_think_block: bool = False
        self._think_positions: set[int] = set()  # Positions that are inside think blocks

        # Memory bank: store evicted K/V for direct attention access
        # Use pre-stacked tensors for GPU efficiency instead of Python lists
        # Shape per layer: (n_entries, n_kv_heads, head_dim)
        self._mem_k: dict[int, mx.array | None] = {l: None for l in range(self.n_layers)}
        self._mem_v: dict[int, mx.array | None] = {l: None for l in range(self.n_layers)}
        self._mem_importance: dict[int, mx.array | None] = {l: None for l in range(self.n_layers)}

    @property
    def memory_bank(self) -> dict[int, int]:
        """Legacy interface - returns count per layer."""
        return {l: self._mem_count(l) for l in range(self.n_layers)}

    def _mem_count(self, layer: int) -> int:
        """Get number of entries in memory bank for a layer."""
        if self._mem_k[layer] is None:
            return 0
        return self._mem_k[layer].shape[0]

    def _mem_add(self, layer: int, k: mx.array, v: mx.array, importance: float) -> None:
        """Add entry to memory bank with GPU-efficient concatenation."""
        k = k[None, :, :]  # (1, n_kv_heads, head_dim)
        v = v[None, :, :]
        imp = mx.array([importance])

        if self._mem_k[layer] is None:
            self._mem_k[layer] = k
            self._mem_v[layer] = v
            self._mem_importance[layer] = imp
        else:
            self._mem_k[layer] = mx.concatenate([self._mem_k[layer], k], axis=0)
            self._mem_v[layer] = mx.concatenate([self._mem_v[layer], v], axis=0)
            self._mem_importance[layer] = mx.concatenate([self._mem_importance[layer], imp], axis=0)

    def _mem_evict_lowest(self, layer: int, keep: int) -> None:
        """Keep only top-k by importance using vectorized ops."""
        if self._mem_k[layer] is None or self._mem_k[layer].shape[0] <= keep:
            return

        # Vectorized top-k selection
        top_indices = mx.argsort(self._mem_importance[layer])[-keep:]
        self._mem_k[layer] = self._mem_k[layer][top_indices]
        self._mem_v[layer] = self._mem_v[layer][top_indices]
        self._mem_importance[layer] = self._mem_importance[layer][top_indices]

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
        k = layer["self_attn"]["k_proj"](normed)
        v = layer["self_attn"]["v_proj"](normed)

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

            # Concatenate memory bank K/V if available
            # Memory bank entries are always attendable (no mask needed)
            if self._mem_k[layer_idx] is not None:
                # Already stacked tensors: (n_mem, n_kv_heads, head_dim)
                mem_k = self._mem_k[layer_idx]
                mem_v = self._mem_v[layer_idx]

                # Top-k retrieval: select most relevant memories based on query
                top_k = self.config.memory_top_k
                if top_k > 0 and mem_k.shape[0] > top_k:
                    # Use mean query across heads for relevance scoring
                    # q shape: (1, n_heads, seq, head_dim)
                    q_mean = mx.mean(q[0], axis=0)  # (seq, head_dim)
                    q_last = q_mean[-1]  # Use last query token (current position)

                    # mem_k shape: (n_mem, n_kv_heads, head_dim)
                    # Average across kv_heads for scoring
                    mem_k_mean = mx.mean(mem_k, axis=1)  # (n_mem, head_dim)

                    # Compute relevance scores via dot product (fully vectorized)
                    scores = mx.sum(q_last * mem_k_mean, axis=-1)  # (n_mem,)

                    # Get top-k indices
                    top_indices = mx.argsort(scores)[-top_k:]

                    # Select top-k memories
                    mem_k = mem_k[top_indices]
                    mem_v = mem_v[top_indices]

                # Reshape to (1, n_heads, n_mem, head_dim)
                mem_k = mem_k[None, :, :, :].transpose(0, 2, 1, 3)
                mem_v = mem_v[None, :, :, :].transpose(0, 2, 1, 3)

                # GQA expansion for memory
                if self.n_kv_heads < self.n_heads:
                    mem_k = mx.repeat(mem_k, repeat_factor, axis=1)
                    mem_v = mx.repeat(mem_v, repeat_factor, axis=1)

                # Concatenate: memory first (always attendable), then cache
                # Shape: (1, n_heads, n_mem + n_cache, head_dim)
                cached_k = mx.concatenate([mem_k, cached_k], axis=2)
                cached_v = mx.concatenate([mem_v, cached_v], axis=2)

                # Extend mask to allow attending to memory (all zeros = attend)
                n_mem = mem_k.shape[2]  # Use actual count after top-k
                if mask is not None:
                    mem_mask = mx.zeros((1, 1, L, n_mem))  # No masking for memory
                    mask = mx.concatenate([mem_mask, mask], axis=3)

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

    def _update_importance(self, attn_weights_by_layer: dict[int, mx.array]) -> None:
        """Update importance scores from attention weights.

        Importance is measured by how much attention each position receives
        from subsequent tokens (averaged across layers and heads).

        Optimized to use vectorized operations instead of Python loops.
        """
        if not attn_weights_by_layer:
            return

        # Stack all layers and compute mean in one operation
        # Each attn: (batch, n_heads, seq_q, seq_k)
        all_attn = mx.stack(list(attn_weights_by_layer.values()), axis=0)
        # Mean across layers, batch, and heads -> (seq_q, seq_k)
        avg_attn = mx.mean(all_attn, axis=(0, 1, 2))

        # Sum attention received by each key position
        importance_per_pos = mx.sum(avg_attn, axis=0)  # (seq_k,)

        # Force evaluation to get values
        mx.eval(importance_per_pos)

        # Batch update: convert to numpy once, then update dict
        importance_vals = importance_per_pos.tolist()
        cached_positions = sorted(self.kv_cache.active_positions)
        decay = self.config.decay
        floor = self.config.min_importance

        for i, pos in enumerate(cached_positions):
            if i < len(importance_vals):
                old_imp = self.kv_cache._importance.get(pos, floor)
                new_imp = importance_vals[i]
                self.kv_cache._importance[pos] = max(floor, old_imp * decay + new_imp)

    # Qwen3 think block tokens
    _THINK_OPEN_TOKEN: int = 151667   # <think>
    _THINK_CLOSE_TOKEN: int = 151668  # </think>

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

        # Track think blocks - these tokens won't be stored in memory bank
        if token_id == self._THINK_OPEN_TOKEN:
            self._in_think_block = True
            self._think_positions.add(position)
        elif token_id == self._THINK_CLOSE_TOKEN:
            self._in_think_block = False
            self._think_positions.add(position)
        elif self._in_think_block:
            self._think_positions.add(position)

        # Forward pass
        logits, attn_weights = self._forward([token_id], [position])

        # Compute log prob if we have previous logits
        log_prob = 0.0
        if logits_in is not None:
            last_logits = logits_in[0, -1, :]
            log_softmax = last_logits - mx.logsumexp(last_logits)
            log_prob = float(log_softmax[token_id])

        # Store token state (minimal - just for eviction tracking)
        self.slots[position] = TokenState(position=position, token_id=token_id)

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
        """Evict a position and store in memory bank for persistent attention.

        The memory bank approach: evicted K/V become permanent attention
        positions that all future queries can attend to.

        Args:
            position: Position to evict
        """
        if position not in self.slots:
            return

        state = self.slots.pop(position)
        importance = self.kv_cache.get_importance(position)

        # Evict from KV cache and get the K, V values
        evicted_kv = self.kv_cache.evict(position)

        if not self.config.memory_enabled:
            return

        # Skip think block tokens - they're ephemeral reasoning, not worth storing
        if position in self._think_positions:
            self._think_positions.discard(position)  # Clean up
            return

        # Filter by importance - only store high-importance tokens
        if importance < self.config.min_importance:
            return

        logger.debug(
            f"Storing to memory bank: token {state.token_id} at pos {position}, "
            f"importance={importance:.4f}"
        )

        # Store K/V in memory bank using GPU-efficient tensor ops
        for layer_idx, (k, v) in evicted_kv.items():
            self._mem_add(layer_idx, k, v, importance)

            # Limit memory bank size - evict LOWEST importance, not oldest
            max_memory = self.config.max_memory_per_layer
            self._mem_evict_lowest(layer_idx, max_memory)

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
        memory_entries = sum(
            self._mem_count(layer) for layer in range(self.n_layers)
        )
        return {
            "cache_size": self.kv_cache.size,
            "memory_entries": memory_entries,
            "total_modifications": memory_entries,  # For backward compatibility
            "positions_processed": self.next_position,
        }
