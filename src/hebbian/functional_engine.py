#!/usr/bin/env python3
"""
Functional Hebbian engine using modification vectors.

Instead of directly modifying weights, we store rank-1 updates as vectors
and apply them functionally during the forward pass.

Benefits:
- Reversible: clear modifications = restore original model
- Minimal memory: just store (output_vec, input_vec, scale) tuples
- Composable: accumulate updates without compounding numerical error
- Efficient: rank-1 updates are O(d) to apply, not O(d²)

Math:
  Original: y = W @ x
  Modified: y = W @ x + Σᵢ (scaleᵢ * outᵢ * (inᵢ · x))

Each modification adds: scale * outer(out, in) to the effective weight matrix,
but we never materialize the full matrix update.
"""

import torch
import torch.nn.functional as F
import math
import logging
import psutil
from dataclasses import dataclass, field
from typing import Optional, Union

from .config import HebbianConfig, HEBBIAN

logger = logging.getLogger(__name__)


def get_memory_info() -> dict:
    """Get current memory usage info."""
    info = {
        "system_available_gb": psutil.virtual_memory().available / (1024**3),
        "system_percent_used": psutil.virtual_memory().percent,
    }

    if torch.backends.mps.is_available():
        # MPS doesn't have direct memory query, use system memory as proxy
        info["device"] = "mps"
        info["allocated_gb"] = torch.mps.current_allocated_memory() / (1024**3) if hasattr(torch.mps, 'current_allocated_memory') else 0
    elif torch.cuda.is_available():
        info["device"] = "cuda"
        info["allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        info["reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
    else:
        info["device"] = "cpu"

    return info


def check_memory_available(min_gb: float = 4.0) -> bool:
    """Check if at least min_gb of memory is available."""
    info = get_memory_info()
    available = info["system_available_gb"]
    if available < min_gb:
        logger.warning(f"Low memory: {available:.1f}GB available, need {min_gb:.1f}GB")
        return False
    return True


def log_memory(prefix: str = ""):
    """Log current memory state."""
    info = get_memory_info()
    logger.debug(f"{prefix} Memory: {info['system_available_gb']:.1f}GB available, "
                 f"{info['system_percent_used']:.0f}% used")


@dataclass
class RankOneModification:
    """A single rank-1 weight modification."""
    output_vec: torch.Tensor  # Shape: (out_dim,)
    input_vec: torch.Tensor   # Shape: (in_dim,)
    scale: float


@dataclass
class TokenState:
    """State for a single token in context."""
    position: int
    token_id: int
    importance: float = 0.0


class FunctionalHebbianEngine:
    """
    Hebbian consolidation using functional modification vectors.

    Instead of W += outer(k, x), we store (k, x, scale) and apply during forward.
    This makes the modifications reversible with zero cost.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Union[HebbianConfig, None] = None,
        device: str = "cpu",
        # Legacy params for backward compatibility - prefer using config
        window_size: Optional[int] = None,
        decay: Optional[float] = None,
        update_scale: Optional[float] = None,
    ):
        # Use config or build from legacy params
        if config is None:
            config = HebbianConfig(
                update_scale=update_scale if update_scale is not None else HEBBIAN.update_scale,
                window_size=window_size if window_size is not None else HEBBIAN.window_size,
                decay=decay if decay is not None else HEBBIAN.decay,
            )

        self.hebbian_config = config
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = config.window_size
        self.decay = config.decay
        self.update_scale = config.update_scale
        self.device = device

        # Extract model structure
        self.model_config = model.config
        self.num_layers = self.model_config.num_hidden_layers
        self.num_heads = self.model_config.num_attention_heads
        self.num_kv_heads = getattr(self.model_config, 'num_key_value_heads', self.num_heads)
        self.hidden_size = self.model_config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        # Get layer references
        self.layers = self._get_layers()
        self.embed = self._get_embedding()
        self.norm = self._get_final_norm()
        self.lm_head = self._get_lm_head()

        # Modification vectors per layer: layer_idx -> list of RankOneModification
        self.k_modifications: dict[int, list[RankOneModification]] = {
            i: [] for i in range(self.num_layers)
        }

        # RoPE parameters
        self.rope_theta = getattr(self.model_config, 'rope_theta', 10000.0)

        # Context state
        self.slots: dict[int, TokenState] = {}
        self.input_cache: dict[int, torch.Tensor] = {}  # position -> hidden state before projection
        self.next_pos = 0
        self.protected: set[int] = set()

        # Stats
        self.total_modifications = 0
        self.evictions: list[dict] = []

        # Memory check at init
        mem_info = get_memory_info()
        logger.info(f"FunctionalHebbianEngine: {self.num_layers} layers, "
                   f"{self.num_heads} heads, {self.num_kv_heads} kv_heads")
        logger.info(f"Memory available: {mem_info['system_available_gb']:.1f}GB "
                   f"({100 - mem_info['system_percent_used']:.0f}% free)")

    def _get_layers(self):
        """Get transformer layer references."""
        if hasattr(self.model, 'model'):
            return self.model.model.layers
        return self.model.transformer.h

    def _get_embedding(self):
        """Get embedding layer."""
        if hasattr(self.model, 'model'):
            return self.model.model.embed_tokens
        return self.model.transformer.wte

    def _get_final_norm(self):
        """Get final layer norm."""
        if hasattr(self.model, 'model'):
            return self.model.model.norm
        return self.model.transformer.ln_f

    def _get_lm_head(self):
        """Get language model head."""
        return self.model.lm_head

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings."""
        seq_len = x.size(2)
        dim = x.size(-1)

        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
        freqs = torch.einsum('i,j->ij', positions.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)

        def rotate_half(t):
            t1 = t[..., : t.shape[-1] // 2]
            t2 = t[..., t.shape[-1] // 2:]
            return torch.cat([-t2, t1], dim=-1)

        return (x * cos) + (rotate_half(x) * sin)

    def _apply_k_modifications(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Apply accumulated rank-1 modifications to K projection output.

        Instead of K = W_k @ x, we compute:
        K = W_k @ x + Σᵢ (scaleᵢ * outᵢ * (inᵢ · x))

        This is equivalent to using W_k + Σᵢ (scaleᵢ * outer(outᵢ, inᵢ))
        but without materializing the full matrix.
        """
        mods = self.k_modifications[layer_idx]
        if not mods:
            return torch.zeros_like(x[:, :, :self.num_kv_heads * self.head_dim])

        # x shape: (batch, seq, hidden)
        delta = torch.zeros(
            x.size(0), x.size(1), self.num_kv_heads * self.head_dim,
            device=x.device, dtype=x.dtype
        )

        for mod in mods:
            # Compute: scale * out * (in · x)
            # Move CPU-stored vectors to device and dtype for computation
            input_vec = mod.input_vec.to(device=x.device, dtype=x.dtype)
            output_vec = mod.output_vec.to(device=x.device, dtype=x.dtype)
            # in · x for each position: (hidden,) · (batch, seq, hidden) -> (batch, seq)
            dot = torch.einsum('h,bsh->bs', input_vec, x)
            # scale * out * dot: (out_dim,) * (batch, seq) -> (batch, seq, out_dim)
            contribution = mod.scale * torch.einsum('o,bs->bso', output_vec, dot)
            delta += contribution

        return delta

    def _forward_layer(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward through a single layer with functional modifications."""
        layer = self.layers[layer_idx]

        # Input layernorm
        if hasattr(layer, 'input_layernorm'):
            normed = layer.input_layernorm(hidden)
        else:
            normed = layer.ln_1(hidden)

        # Store input for Hebbian (before projection)
        # We'll cache this per-position later in the main forward

        # Get projections
        attn = layer.self_attn if hasattr(layer, 'self_attn') else layer.attn

        # Q, K, V projections
        q = attn.q_proj(normed)
        k_base = attn.k_proj(normed)
        v = attn.v_proj(normed)

        # Apply functional modifications to K
        k_delta = self._apply_k_modifications(layer_idx, normed)
        k = k_base + k_delta

        batch, seq, _ = q.shape

        # Reshape for attention
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        pos_tensor = positions.to(hidden.device)
        q = self._apply_rope(q, pos_tensor)
        k = self._apply_rope(k, pos_tensor)

        # GQA: expand KV heads
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, -1)

        # Output projection
        attn_output = attn.o_proj(attn_output)

        # Residual
        hidden = hidden + attn_output

        # MLP
        if hasattr(layer, 'post_attention_layernorm'):
            normed = layer.post_attention_layernorm(hidden)
        else:
            normed = layer.ln_2(hidden)

        if hasattr(layer, 'mlp'):
            mlp_out = layer.mlp(normed)
        else:
            mlp_out = layer.feed_forward(normed)

        hidden = hidden + mlp_out

        return hidden, attn_weights

    def _forward(
        self,
        token_ids: list[int],
        positions: list[int],
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[int, torch.Tensor]]:
        """Full forward pass with functional modifications."""
        input_ids = torch.tensor([token_ids], device=self.device)
        pos_tensor = torch.tensor(positions, device=self.device)

        # Embedding
        hidden = self.embed(input_ids)

        # Causal mask
        seq_len = len(token_ids)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=self.device),
            diagonal=1
        )

        attn_weights_list = []
        input_states = {}  # position -> hidden state before attention

        for layer_idx in range(self.num_layers):
            # Cache input states for Hebbian
            if hasattr(self.layers[layer_idx], 'input_layernorm'):
                normed = self.layers[layer_idx].input_layernorm(hidden)
            else:
                normed = self.layers[layer_idx].ln_1(hidden)

            for seq_idx, pos in enumerate(positions):
                input_states[pos] = normed[0, seq_idx].detach().clone()

            hidden, attn_weights = self._forward_layer(
                layer_idx, hidden, pos_tensor, causal_mask
            )
            attn_weights_list.append(attn_weights.detach())

        # Final norm and LM head
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        return logits[0, -1], attn_weights_list, input_states

    def _update_importance(self, attn_list: list[torch.Tensor], positions: list[int]):
        """Update importance scores from attention."""
        avg_attn = torch.stack(attn_list).mean(dim=0)
        incoming = avg_attn.sum(dim=2).mean(dim=1).squeeze(0)

        for pos, slot in self.slots.items():
            slot.importance *= self.decay

        for seq_idx, pos in enumerate(positions):
            if pos in self.slots:
                self.slots[pos].importance += incoming[seq_idx].item()

    def _add_modification(
        self,
        layer_idx: int,
        key_output: torch.Tensor,
        input_hidden: torch.Tensor,
        importance: float,
    ):
        """Add a rank-1 modification for a layer."""
        # Check memory before adding
        if not check_memory_available(min_gb=self.hebbian_config.min_memory_gb):
            logger.warning("Skipping modification due to low memory")
            return

        # Limit max modifications per layer to prevent memory blowup
        if len(self.k_modifications[layer_idx]) >= self.hebbian_config.max_mods_per_layer:
            # Remove oldest modification
            self.k_modifications[layer_idx].pop(0)

        # Normalize vectors and move to CPU to save GPU memory
        # Match minimal_engine: normalize by ||outer(k, x)|| ≈ ||k|| * ||x||
        # So we store normalized vectors and DON'T multiply scale by norms
        out = key_output.detach().cpu()
        inp = input_hidden.detach().cpu()

        out_norm = out.norm()
        inp_norm = inp.norm()

        if out_norm > 0 and inp_norm > 0:
            out = out / out_norm
            inp = inp / inp_norm

            # Scale WITHOUT multiplying by norms - this matches minimal_engine's
            # normalized outer product: outer(k, x) / ||outer(k, x)||
            mod = RankOneModification(
                output_vec=out,  # Stored on CPU, normalized
                input_vec=inp,   # Stored on CPU, normalized
                scale=self.update_scale * importance,  # No norm multiplication!
            )
            self.k_modifications[layer_idx].append(mod)
            self.total_modifications += 1

            if self.total_modifications % 100 == 0:
                log_memory(f"After {self.total_modifications} modifications:")

    def _evict_one(self) -> Optional[int]:
        """Evict lowest importance token, add modification vectors."""
        candidates = [
            (pos, slot.importance)
            for pos, slot in self.slots.items()
            if pos not in self.protected
        ]
        if not candidates:
            return None

        evict_pos, evict_importance = min(candidates, key=lambda x: x[1])
        evicted = self.slots.pop(evict_pos)

        self.evictions.append({
            'position': evict_pos,
            'token_id': evicted.token_id,
            'importance': evict_importance,
        })

        # Get cached input state
        if evict_pos in self.input_cache:
            input_hidden = self.input_cache.pop(evict_pos)

            # Add modifications for each layer
            for layer_idx in range(self.num_layers):
                # Compute what K would have been for this position
                layer = self.layers[layer_idx]
                attn = layer.self_attn if hasattr(layer, 'self_attn') else layer.attn

                with torch.no_grad():
                    k = attn.k_proj(input_hidden.unsqueeze(0)).squeeze(0)
                    k = k[:self.num_kv_heads * self.head_dim]  # Only KV head dims

                self._add_modification(layer_idx, k, input_hidden, evict_importance)

        return evict_pos

    def clear_modifications(self):
        """Clear all modifications - instantly restore original model behavior."""
        import gc

        for layer_idx in range(self.num_layers):
            self.k_modifications[layer_idx].clear()
        self.total_modifications = 0

        # Also clear cached states
        self.input_cache.clear()
        self.evictions.clear()

        # Force garbage collection
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        log_memory("After clear_modifications:")

    def get_modification_count(self) -> dict[int, int]:
        """Get number of modifications per layer."""
        return {i: len(mods) for i, mods in self.k_modifications.items()}

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> dict:
        """Generate with functional Hebbian consolidation."""
        logger.debug(f"CHECKPOINT: generate() start, max_tokens={max_new_tokens}")
        log_memory("generate() entry")

        # Reset context (but keep modifications if any)
        self.slots = {}
        self.kv_cache = {i: {} for i in range(self.num_layers)}
        self.input_cache = {}
        self.evictions = []
        self.next_pos = 0
        self.protected = set()

        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        logger.debug(f"CHECKPOINT: Encoded {len(prompt_ids)} tokens")
        all_tokens = prompt_ids.copy()

        # Add prompt tokens
        prompt_positions = []
        for tid in prompt_ids:
            pos = self.next_pos
            self.next_pos += 1
            self.slots[pos] = TokenState(position=pos, token_id=tid)
            prompt_positions.append(pos)

        self.protected = set(prompt_positions)

        # Process prompt
        logits, attn_list, input_states = self._forward(prompt_ids, prompt_positions)
        self._update_importance(attn_list, prompt_positions)
        self.input_cache.update(input_states)

        perplexity_curve = []

        # Generate
        for step in range(max_new_tokens):
            if step % 10 == 0:
                logger.debug(f"CHECKPOINT: Generation step {step}/{max_new_tokens}")
                log_memory(f"Step {step}")

            positions = sorted(self.slots.keys())
            token_ids = [self.slots[p].token_id for p in positions]

            logits, attn_list, input_states = self._forward(token_ids, positions)
            self._update_importance(attn_list, positions)
            self.input_cache.update(input_states)

            # Perplexity
            logits_f32 = logits.float()
            probs = F.softmax(logits_f32, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum().item()
            if not (math.isnan(entropy) or math.isinf(entropy)):
                perplexity_curve.append(entropy)

            # Evict if needed
            while len(self.slots) >= self.window_size:
                self._evict_one()

            # Sample
            if temperature > 0:
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

        text = self.tokenizer.decode(all_tokens, skip_special_tokens=True)
        generated = text[len(prompt):]

        return {
            "text": generated,
            "tokens": all_tokens,
            "perplexity_curve": perplexity_curve,
            "evictions": self.evictions,
            "total_modifications": self.total_modifications,
            "modifications_per_layer": self.get_modification_count(),
        }

