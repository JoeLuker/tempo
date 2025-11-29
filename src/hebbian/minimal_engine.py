"""
Minimal inference engine - no HuggingFace generate(), no hooks.
Direct control over the forward pass.

We extract the weights and run inference ourselves:
1. Embedding lookup
2. For each layer: attention + MLP
3. Final LM head

This gives us direct access to K, V, attention at each step.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class TokenState:
    """State for a token in context."""
    position: int
    token_id: int
    importance: float = 0.0


class MinimalHebbianEngine:
    """
    Direct inference with Hebbian learning.

    No hooks, no HuggingFace generate(). We run the forward pass ourselves.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        window_size: int = 512,
        decay: float = 0.99,
        update_scale: float = 1e-6,
        device: str = "cuda",
    ):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.decay = decay
        self.update_scale = update_scale
        self.device = device

        # Extract model components
        self._extract_model(model)

        # Sparse position tracking
        self.slots: Dict[int, TokenState] = {}
        self.protected: set = set()
        self.next_pos = 0

        # Sparse KV cache: layer -> position -> (K, V)
        # K, V shape: (num_kv_heads, head_dim)
        self.kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {
            i: {} for i in range(self.num_layers)
        }

        # Store inputs for Hebbian: position -> hidden_state
        self.input_cache: Dict[int, torch.Tensor] = {}

        # Stats
        self.evictions: List[dict] = []
        self.perplexity_curve: List[float] = []
        self.total_updates = 0

        logger.info(
            f"MinimalHebbianEngine: {self.num_layers} layers, "
            f"{self.num_heads} heads, {self.num_kv_heads} kv_heads, "
            f"hidden={self.hidden_dim}, head_dim={self.head_dim}"
        )

    def _extract_model(self, model: torch.nn.Module) -> None:
        """Extract weights and config from HuggingFace model."""
        config = model.config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.hidden_dim = config.hidden_size
        self.head_dim = self.hidden_dim // self.num_heads
        self.vocab_size = config.vocab_size
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)

        # Heads per KV group (for GQA)
        self.num_heads_per_kv = self.num_heads // self.num_kv_heads

        # Extract embedding
        if hasattr(model, 'model'):
            base = model.model
        else:
            base = model

        self.embed_tokens = base.embed_tokens

        # Extract layers
        self.layers = []
        for layer in base.layers:
            self.layers.append({
                'input_layernorm': layer.input_layernorm,
                'q_proj': layer.self_attn.q_proj,
                'k_proj': layer.self_attn.k_proj,
                'v_proj': layer.self_attn.v_proj,
                'o_proj': layer.self_attn.o_proj,
                'post_attention_layernorm': layer.post_attention_layernorm,
                'gate_proj': layer.mlp.gate_proj,
                'up_proj': layer.mlp.up_proj,
                'down_proj': layer.mlp.down_proj,
            })

        # Final norm and LM head
        self.norm = base.norm
        self.lm_head = model.lm_head

        # Precompute RoPE frequencies
        self._init_rope()

    def _init_rope(self) -> None:
        """Initialize rotary position embedding frequencies."""
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim)
        )
        self.inv_freq = inv_freq

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embeddings (Llama-style).

        Args:
            x: (batch, num_heads, seq_len, head_dim)
            positions: (seq_len,) position IDs

        Returns:
            x with RoPE applied
        """
        seq_len = positions.size(0)

        # Compute position encodings
        freqs = torch.outer(positions.float(), self.inv_freq)  # (seq_len, head_dim/2)

        # Llama uses cos/sin with interleaved application
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)

        # Rotate half (Llama style)
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        x_rot = (x * cos) + (rotate_half(x) * sin)
        return x_rot

    def _attention(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        positions: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Run attention for one layer.

        Args:
            layer_idx: Which layer
            hidden: (1, seq_len, hidden_dim) input hidden states
            positions: List of position IDs for each token

        Returns:
            output: (1, seq_len, hidden_dim)
            attn_weights: (1, num_heads, seq_len, seq_len)
            new_kv: position -> (K, V) for newly computed positions
        """
        layer = self.layers[layer_idx]
        seq_len = hidden.size(1)

        # Layer norm
        normed = layer['input_layernorm'](hidden)

        # Project Q, K, V
        q = layer['q_proj'](normed)  # (1, seq_len, num_heads * head_dim)
        k = layer['k_proj'](normed)  # (1, seq_len, num_kv_heads * head_dim)
        v = layer['v_proj'](normed)  # (1, seq_len, num_kv_heads * head_dim)

        # Reshape for attention
        q = q.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(1, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Store K, V for cache (before RoPE, as that's position-dependent)
        new_kv = {}
        for seq_idx, pos in enumerate(positions):
            new_kv[pos] = (
                k[0, :, seq_idx, :].clone(),  # (num_kv_heads, head_dim)
                v[0, :, seq_idx, :].clone(),
            )

        # Apply RoPE to Q and K
        position_ids = torch.tensor(positions, device=self.device)
        q = self._apply_rope(q, position_ids)
        k = self._apply_rope(k, position_ids)

        # Expand KV for GQA
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads_per_kv, dim=1)
            v = v.repeat_interleave(self.num_heads_per_kv, dim=1)

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=self.device) * float('-inf'),
            diagonal=1
        )
        attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(1, seq_len, self.hidden_dim)
        attn_output = layer['o_proj'](attn_output)

        return attn_output, attn_weights, new_kv

    def _mlp(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        """Run MLP for one layer."""
        layer = self.layers[layer_idx]
        normed = layer['post_attention_layernorm'](hidden)
        gate = F.silu(layer['gate_proj'](normed))
        up = layer['up_proj'](normed)
        return layer['down_proj'](gate * up)

    def _forward(
        self,
        token_ids: List[int],
        positions: List[int],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Full forward pass.

        Args:
            token_ids: Token IDs to process
            positions: Position IDs (can have gaps!)

        Returns:
            logits: (vocab_size,) for last position
            all_attn: List of attention weights per layer
        """
        seq_len = len(token_ids)

        # Embed
        input_ids = torch.tensor([token_ids], device=self.device)
        hidden = self.embed_tokens(input_ids)  # (1, seq_len, hidden_dim)

        # Store inputs for Hebbian (using first layer's input)
        for seq_idx, pos in enumerate(positions):
            self.input_cache[pos] = hidden[0, seq_idx].clone()

        all_attn = []

        # Run layers
        for layer_idx in range(self.num_layers):
            attn_out, attn_weights, new_kv = self._attention(layer_idx, hidden, positions)
            hidden = hidden + attn_out

            mlp_out = self._mlp(layer_idx, hidden)
            hidden = hidden + mlp_out

            all_attn.append(attn_weights)

            # Store KV in cache
            for pos, (k, v) in new_kv.items():
                self.kv_cache[layer_idx][pos] = (k, v)

        # Final norm and LM head
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden[0, -1, :])  # Last position only

        return logits, all_attn

    def _update_importance(self, attn_list: List[torch.Tensor], positions: List[int]) -> None:
        """Update importance scores from attention."""
        # Average attention across layers
        avg_attn = torch.stack(attn_list).mean(dim=0)  # (1, heads, seq, seq)

        # Incoming attention to each position (sum over queries, mean over heads)
        incoming = avg_attn.sum(dim=2).mean(dim=1).squeeze(0)  # (seq_len,)

        # Decay and update
        for pos, slot in self.slots.items():
            slot.importance *= self.decay

        for seq_idx, pos in enumerate(positions):
            if pos in self.slots:
                self.slots[pos].importance += incoming[seq_idx].item()

    def _evict_one(self) -> Optional[int]:
        """Evict lowest importance token, apply Hebbian update."""
        # Find candidate
        candidates = [
            (pos, slot.importance)
            for pos, slot in self.slots.items()
            if pos not in self.protected
        ]
        if not candidates:
            return None

        evict_pos, importance = min(candidates, key=lambda x: x[1])

        # Apply Hebbian update for each layer
        for layer_idx in range(self.num_layers):
            if evict_pos in self.kv_cache[layer_idx] and evict_pos in self.input_cache:
                k, v = self.kv_cache[layer_idx][evict_pos]
                inp = self.input_cache[evict_pos]

                # ΔW = importance * outer(output, input)
                self._apply_hebbian(layer_idx, k.flatten(), v.flatten(), inp, importance)

        # Log and remove
        slot = self.slots.pop(evict_pos)
        self.evictions.append({
            "position": evict_pos,
            "token_id": slot.token_id,
            "importance": importance,
        })

        # Clean caches
        for layer_idx in range(self.num_layers):
            self.kv_cache[layer_idx].pop(evict_pos, None)
        self.input_cache.pop(evict_pos, None)

        self.total_updates += self.num_layers

        return evict_pos

    def _apply_hebbian(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        input_hidden: torch.Tensor,
        importance: float,
    ) -> None:
        """Apply Hebbian weight update."""
        layer = self.layers[layer_idx]

        with torch.no_grad():
            for proj, output in [('k_proj', key), ('v_proj', value)]:
                W = layer[proj].weight

                # Ensure shapes match
                out = output.to(W.device, W.dtype)
                inp = input_hidden.to(W.device, W.dtype)

                if out.size(0) != W.size(0):
                    out = out[:W.size(0)]
                if inp.size(0) != W.size(1):
                    inp = inp[:W.size(1)]

                # Compute update: importance * outer(output, input)
                update = torch.outer(out, inp)

                # Normalize to unit norm, then scale by update_scale * importance
                u_norm = update.norm()
                if u_norm > 0:
                    update = update / u_norm  # Unit normalize

                W.add_(update, alpha=self.update_scale * importance)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        forced_tokens: Optional[Dict[int, int]] = None,
        temperature: float = 1.0,
        protect_prompt: bool = True,
    ) -> dict:
        """Generate with Hebbian consolidation."""
        forced_tokens = forced_tokens or {}

        # Reset
        self.slots = {}
        self.kv_cache = {i: {} for i in range(self.num_layers)}
        self.input_cache = {}
        self.evictions = []
        self.perplexity_curve = []
        self.next_pos = 0
        self.protected = set()

        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        all_tokens = prompt_ids.copy()

        # Add prompt tokens
        prompt_positions = []
        for tid in prompt_ids:
            pos = self.next_pos
            self.next_pos += 1
            self.slots[pos] = TokenState(position=pos, token_id=tid)
            prompt_positions.append(pos)

        if protect_prompt:
            self.protected = set(prompt_positions)

        # Process prompt
        logits, attn_list = self._forward(prompt_ids, prompt_positions)
        self._update_importance(attn_list, prompt_positions)

        # Generate
        for step in range(max_new_tokens):
            # Get current state
            positions = sorted(self.slots.keys())
            token_ids = [self.slots[p].token_id for p in positions]

            # Forward pass first - this updates importance for ALL current tokens
            logits, attn_list = self._forward(token_ids, positions)
            self._update_importance(attn_list, positions)

            # Perplexity (use float32 for numerical stability)
            logits_f32 = logits.float()
            probs = F.softmax(logits_f32, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum().item()
            if not (math.isnan(entropy) or math.isinf(entropy)):
                self.perplexity_curve.append(entropy)

            # Now evict if needed - all tokens have importance from this step
            while len(self.slots) >= self.window_size:
                self._evict_one()

            # Sample
            if step in forced_tokens:
                next_token = forced_tokens[step]
            elif temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = logits.argmax().item()

            # Add token
            pos = self.next_pos
            self.next_pos += 1
            self.slots[pos] = TokenState(position=pos, token_id=next_token)
            all_tokens.append(next_token)

            if next_token == self.tokenizer.eos_token_id:
                break

        # Decode
        text = self.tokenizer.decode(all_tokens, skip_special_tokens=True)
        generated = text[len(prompt):]

        return {
            "text": generated,
            "tokens": all_tokens,
            "perplexity_curve": self.perplexity_curve,
            "evictions": self.evictions,
            "total_updates": self.total_updates,
            "context_size": len(self.slots),
        }
