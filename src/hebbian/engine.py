"""
Hebbian inference engine with proper attention hooks.

This is the clean implementation that:
1. Hooks into actual k_proj/v_proj layers to capture real K, V, and inputs
2. Tracks cumulative attention importance with decay
3. Evicts by importance, not recency
4. Applies proper Hebbian updates: ΔW = importance * outer(output, input)
"""

import torch
import logging
from typing import Dict, List, Optional, Iterator, Tuple
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer

from .attention_hooks import AttentionHooks, HebbianWeightUpdater

logger = logging.getLogger(__name__)


@dataclass
class TokenState:
    """State for a single token in the context."""
    position: int
    token_id: int
    importance: float  # Cumulative attention received
    age: int  # Steps since added


class ImportanceManager:
    """
    Track cumulative attention importance for each token.

    importance[pos] = Σ_t (decay^(T-t) * attention_received_at_t)

    Where attention_received = sum of attention weights pointing TO this position.
    """

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.importance: Dict[int, float] = {}
        self.ages: Dict[int, int] = {}
        self.current_step = 0
        self.protected: set = set()

    def add_token(self, position: int) -> None:
        """Register a new token."""
        self.importance[position] = 0.0
        self.ages[position] = self.current_step

    def protect(self, positions: range) -> None:
        """Mark positions as protected from eviction."""
        self.protected.update(positions)

    def update(self, attention_weights: torch.Tensor) -> None:
        """
        Update importance from attention weights.

        Args:
            attention_weights: (batch, heads, seq_len, seq_len)
                              We sum incoming attention to each position.
        """
        self.current_step += 1

        # Decay existing importance
        for pos in self.importance:
            self.importance[pos] *= self.decay

        # Sum attention received by each position (sum over queries, mean over heads)
        # attention[:, :, q, k] = attention from query q to key k
        # We want incoming to k, so sum over q dimension
        incoming = attention_weights.sum(dim=2).mean(dim=(0, 1))  # (seq_len,)

        for pos in self.importance:
            if pos < incoming.size(0):
                self.importance[pos] += incoming[pos].item()

    def get_eviction_candidate(self) -> Optional[Tuple[int, float]]:
        """
        Get position with lowest importance that isn't protected.

        Returns:
            (position, importance) or None if all protected
        """
        candidates = [
            (pos, imp) for pos, imp in self.importance.items()
            if pos not in self.protected
        ]

        if not candidates:
            return None

        # Return position with minimum importance
        return min(candidates, key=lambda x: x[1])

    def remove(self, position: int) -> float:
        """Remove a position, return its final importance."""
        imp = self.importance.pop(position, 0.0)
        self.ages.pop(position, None)
        return imp

    def get_stats(self) -> dict:
        if not self.importance:
            return {"count": 0, "mean": 0, "max": 0, "min": 0}

        values = list(self.importance.values())
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "max": max(values),
            "min": min(values),
            "protected": len(self.protected),
        }


class HebbianEngine:
    """
    Transformer inference with Hebbian weight updates.

    The core loop:
    1. Forward pass with attention hooks to capture K, V, inputs
    2. Update importance from attention weights
    3. When context full, evict lowest-importance token
    4. Before eviction, apply Hebbian update: W += importance * outer(output, input)
    5. Continue generation
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

        # Ensure model outputs attention
        self.model.config.output_attentions = True

        # Set up hooks
        self.hooks = AttentionHooks(model)

        # Set up weight updater
        self.updater = HebbianWeightUpdater(
            model,
            scale=update_scale,
            normalize_by_weight_norm=True,
            clip_update_ratio=0.001,
        )

        # Importance tracking
        self.importance = ImportanceManager(decay=decay)

        # Token tracking
        self.tokens: Dict[int, TokenState] = {}
        self.current_seq_len = 0

        # Statistics
        self.evictions = []
        self.perplexity_curve = []

        logger.info(
            f"HebbianEngine: window={window_size}, decay={decay}, "
            f"scale={update_scale}, device={device}"
        )

    def _evict_one(self) -> Optional[int]:
        """Evict the least important token."""
        candidate = self.importance.get_eviction_candidate()
        if candidate is None:
            return None

        position, imp = candidate

        # Get the actual K, V, input that were captured for this position
        token_data = self.hooks.get_token_data(position)

        if not token_data:
            logger.debug(f"No token data found for position {position}, skipping update")

        # Apply Hebbian update for each layer
        for layer_idx, (key, value, input_hidden) in token_data.items():
            if key is not None and value is not None and input_hidden is not None:
                self.updater.apply_update(
                    layer_idx=layer_idx,
                    key=key,
                    value=value,
                    input_hidden=input_hidden,
                    importance=imp,
                )

        # Log eviction
        token_state = self.tokens.get(position)
        self.evictions.append({
            "position": position,
            "importance": imp,
            "token_id": token_state.token_id if token_state else None,
            "age": token_state.age if token_state else 0,
        })

        # Clean up
        self.importance.remove(position)
        self.hooks.clear_position(position)
        self.tokens.pop(position, None)

        logger.debug(f"Evicted position {position} with importance {imp:.4f}")

        return position

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        forced_tokens: Optional[Dict[int, int]] = None,
        temperature: float = 1.0,
        protect_prompt: bool = True,
    ) -> dict:
        """
        Generate text with Hebbian consolidation.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            forced_tokens: Dict mapping generation step -> token_id to force
            temperature: Sampling temperature (0 = greedy)
            protect_prompt: If True, prompt tokens can't be evicted

        Returns:
            Dict with text, tokens, perplexity_curve, evictions, stats
        """
        forced_tokens = forced_tokens or {}

        # Reset state
        self.tokens = {}
        self.importance = ImportanceManager(decay=self.importance.decay)
        self.evictions = []
        self.perplexity_curve = []
        self.hooks.clear_all()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.size(1)

        # Register hooks
        self.hooks.register_hooks()

        try:
            # Process prompt
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    output_attentions=True,
                    use_cache=False,
                )

            # Capture attention and register prompt tokens
            if outputs.attentions:
                self.hooks.capture_attention_weights(outputs.attentions)
                # Average attention across layers for importance
                avg_attn = torch.stack(outputs.attentions).mean(dim=0)
                self.importance.update(avg_attn)

            # Register prompt tokens
            for pos in range(prompt_len):
                self.importance.add_token(pos)
                self.tokens[pos] = TokenState(
                    position=pos,
                    token_id=input_ids[0, pos].item(),
                    importance=0.0,
                    age=0,
                )

            # Protect prompt
            if protect_prompt:
                self.importance.protect(range(prompt_len))

            self.current_seq_len = prompt_len

            # Generation loop
            generated_ids = input_ids.clone()

            for step in range(max_new_tokens):
                # Check if we need to evict
                while len(self.tokens) >= self.window_size:
                    evicted_pos = self._evict_one()
                    if evicted_pos is None:
                        break

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(
                        generated_ids,
                        output_attentions=True,
                        use_cache=False,
                    )

                logits = outputs.logits[:, -1, :]

                # Compute perplexity (entropy of distribution)
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log(probs.clamp(min=1e-10))
                entropy = -(probs * log_probs).sum().item()
                if not torch.isnan(torch.tensor(entropy)):
                    self.perplexity_curve.append(entropy)

                # Capture attention
                if outputs.attentions:
                    self.hooks.capture_attention_weights(outputs.attentions)
                    avg_attn = torch.stack(outputs.attentions).mean(dim=0)
                    self.importance.update(avg_attn)

                # Get next token
                if step in forced_tokens:
                    next_token = forced_tokens[step]
                else:
                    if temperature > 0:
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                    else:
                        next_token = logits.argmax(dim=-1).item()

                # Register new token
                new_pos = generated_ids.size(1)
                self.importance.add_token(new_pos)
                self.tokens[new_pos] = TokenState(
                    position=new_pos,
                    token_id=next_token,
                    importance=0.0,
                    age=0,
                )

                # Append to sequence
                next_tensor = torch.tensor([[next_token]], device=self.device)
                generated_ids = torch.cat([generated_ids, next_tensor], dim=1)

                # Check for EOS
                if next_token == self.tokenizer.eos_token_id:
                    break

        finally:
            self.hooks.remove_hooks()

        # Decode
        full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt):]

        return {
            "text": generated_text,
            "tokens": generated_ids[0].tolist(),
            "perplexity_curve": self.perplexity_curve,
            "evictions": self.evictions,
            "importance_stats": self.importance.get_stats(),
            "updater_stats": self.updater.get_stats(),
        }

    def reset_model(self) -> None:
        """
        Note: This doesn't actually reset weights.
        Once Hebbian updates are applied, they persist.
        To reset, reload the model from checkpoint.
        """
        logger.warning("reset_model called - Hebbian changes are permanent")
        self.updater.total_updates = 0
        self.updater.total_update_norm = 0.0
        self.updater.max_update_ratio = 0.0
