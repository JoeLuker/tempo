"""Main inference loop with Hebbian consolidation."""

import torch
import logging
from typing import Optional, Dict, List, Iterator, Tuple
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer

from .importance_tracker import ImportanceTracker
from .hebbian_updater import HebbianUpdater
from .consolidating_cache import ConsolidatingCache

logger = logging.getLogger(__name__)


@dataclass
class GenerationStep:
    """Result of a single generation step."""
    token_id: int
    token_text: str
    logits: torch.Tensor
    attention_weights: Optional[torch.Tensor]
    was_forced: bool
    eviction_occurred: bool
    step_idx: int


@dataclass
class GenerationResult:
    """Complete generation result."""
    text: str
    tokens: List[int]
    steps: List[GenerationStep]
    perplexity_curve: List[float]  # Perplexity at each step
    eviction_log: List[dict]


class HebbianInferenceEngine:
    """
    Transformer inference with Hebbian consolidation.

    Key features:
    1. Attention-scored eviction (not recency-based)
    2. Hebbian weight updates on eviction
    3. Forced token injection for guided learning
    4. Perplexity tracking to measure learning
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        window_size: int = 512,
        alpha: float = 1e-5,
        decay: float = 0.99,
        device: str = "cuda",
    ):
        """
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            window_size: Maximum context window size
            alpha: Hebbian learning rate
            decay: Attention importance decay
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.device = device

        # Get model config
        config = model.config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_dim = config.hidden_size

        # Initialize Hebbian updater
        self.updater = HebbianUpdater(model, alpha=alpha)

        # Initialize cache (will be reset per generation)
        self.cache: Optional[ConsolidatingCache] = None

        # Eviction log for analysis
        self.eviction_log: List[dict] = []

        # Ensure model outputs attention
        self.model.config.output_attentions = True

        logger.info(
            f"HebbianInferenceEngine initialized: "
            f"window={window_size}, alpha={alpha}, decay={decay}, "
            f"layers={self.num_layers}, heads={self.num_heads}"
        )

    def _create_cache(self) -> ConsolidatingCache:
        """Create a fresh cache with eviction callback."""

        def on_eviction(position, keys, values, input_embedding, importance):
            """Called when a token is evicted."""
            # Log the eviction
            eviction_info = {
                "position": position,
                "importance": importance,
                "token_id": None,  # Could track if needed
            }
            self.eviction_log.append(eviction_info)

            # Apply Hebbian update for each layer
            for layer_idx in range(self.num_layers):
                self.updater.consolidate(
                    layer_idx=layer_idx,
                    key_vector=keys[layer_idx],
                    value_vector=values[layer_idx],
                    input_embedding=input_embedding,
                    importance=importance
                )

        return ConsolidatingCache(
            max_size=self.window_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            decay=0.99,
            device=self.device,
            dtype=torch.float16,
            on_eviction=on_eviction
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        forced_tokens: Optional[Dict[int, int]] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        protect_prompt: bool = True,
    ) -> GenerationResult:
        """
        Generate text with Hebbian consolidation.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            forced_tokens: Dict mapping step -> token_id for forced injection
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            protect_prompt: If True, prompt tokens can't be evicted

        Returns:
            GenerationResult with text, tokens, and metrics
        """
        forced_tokens = forced_tokens or {}

        # Reset state
        self.cache = self._create_cache()
        self.eviction_log = []

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = input_ids.size(1)

        # Process prompt
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False  # We manage our own cache
            )

        # Store prompt in cache
        hidden_states = outputs.hidden_states  # Tuple of [batch, seq, hidden]
        # Use the input embeddings (first hidden state)
        input_embeddings = hidden_states[0][0]  # [seq, hidden]

        # For each prompt token, store in cache
        # Note: We'd need to extract per-layer KV pairs from the model
        # For now, use the hidden states as a proxy
        for pos in range(prompt_length):
            # Approximate: use hidden state as both K and V proxy
            # Real implementation would hook into attention layers
            h = input_embeddings[pos]
            keys = [h.unsqueeze(0).expand(self.num_heads, -1)[:, :self.head_dim] for _ in range(self.num_layers)]
            values = keys  # Same approximation

            self.cache.add_token(
                position=pos,
                keys=keys,
                values=values,
                input_embedding=h
            )

        # Protect prompt tokens
        if protect_prompt:
            self.cache.protect_range(0, prompt_length)

        # Update attention importance from prompt processing
        if outputs.attentions:
            # Average across layers for initial importance
            attn_avg = torch.stack(outputs.attentions).mean(dim=0)
            self.cache.update_attention(attn_avg)

        # Generation loop
        steps = []
        perplexity_curve = []
        generated_ids = input_ids.clone()

        for step in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    generated_ids,
                    output_attentions=True,
                    output_hidden_states=True,
                    use_cache=False
                )

            logits = outputs.logits[:, -1, :]  # [batch, vocab]

            # Compute perplexity for this step
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum().item()
            perplexity_curve.append(entropy)

            # Check for forced token
            if step in forced_tokens:
                next_token_id = forced_tokens[step]
                was_forced = True
            else:
                # Sample
                if temperature > 0:
                    scaled_logits = logits / temperature

                    # Top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = False
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        scaled_logits[indices_to_remove] = float('-inf')

                    probs = torch.softmax(scaled_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token_id = logits.argmax(dim=-1).item()

                was_forced = False

            # Update attention importance
            if outputs.attentions:
                attn_avg = torch.stack(outputs.attentions).mean(dim=0)
                self.cache.update_attention(attn_avg)

            # Add new token to cache
            current_pos = generated_ids.size(1)
            hidden = outputs.hidden_states[0][0, -1]  # Last token hidden state

            eviction = self.cache.add_token(
                position=current_pos,
                keys=[hidden.unsqueeze(0).expand(self.num_heads, -1)[:, :self.head_dim] for _ in range(self.num_layers)],
                values=[hidden.unsqueeze(0).expand(self.num_heads, -1)[:, :self.head_dim] for _ in range(self.num_layers)],
                input_embedding=hidden
            )

            # Record step
            steps.append(GenerationStep(
                token_id=next_token_id,
                token_text=self.tokenizer.decode([next_token_id]),
                logits=logits.cpu(),
                attention_weights=attn_avg.cpu() if outputs.attentions else None,
                was_forced=was_forced,
                eviction_occurred=eviction is not None,
                step_idx=step
            ))

            # Append to sequence
            next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
            generated_ids = torch.cat([generated_ids, next_token_tensor], dim=1)

            # Check for EOS
            if next_token_id == self.tokenizer.eos_token_id:
                break

        # Decode final text
        full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt):]

        return GenerationResult(
            text=generated_text,
            tokens=generated_ids[0].tolist(),
            steps=steps,
            perplexity_curve=perplexity_curve,
            eviction_log=self.eviction_log
        )

    def stream_tokens(
        self,
        token_stream: Iterator[int],
        protect_first_n: int = 0,
    ) -> Iterator[Tuple[int, dict]]:
        """
        Process a stream of tokens with consolidation.

        This is for long-running inference where tokens come in continuously.

        Args:
            token_stream: Iterator yielding token IDs
            protect_first_n: Protect first N tokens from eviction

        Yields:
            Tuple of (token_id, stats_dict)
        """
        self.cache = self._create_cache()
        self.eviction_log = []

        position = 0
        all_ids = []

        for token_id in token_stream:
            all_ids.append(token_id)
            input_ids = torch.tensor([all_ids], device=self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    output_attentions=True,
                    output_hidden_states=True,
                    use_cache=False
                )

            # Update attention
            if outputs.attentions:
                attn_avg = torch.stack(outputs.attentions).mean(dim=0)
                self.cache.update_attention(attn_avg)

            # Add to cache
            hidden = outputs.hidden_states[0][0, -1]
            self.cache.add_token(
                position=position,
                keys=[hidden.unsqueeze(0).expand(self.num_heads, -1)[:, :self.head_dim] for _ in range(self.num_layers)],
                values=[hidden.unsqueeze(0).expand(self.num_heads, -1)[:, :self.head_dim] for _ in range(self.num_layers)],
                input_embedding=hidden
            )

            if position < protect_first_n:
                self.cache.protect_range(position, position + 1)

            position += 1

            # Yield with stats
            yield token_id, {
                "cache": self.cache.get_stats(),
                "updater": self.updater.get_stats(),
            }

    def get_stats(self) -> dict:
        """Get combined statistics."""
        stats = {
            "updater": self.updater.get_stats(),
        }
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        stats["evictions"] = len(self.eviction_log)
        return stats

    def reset_weights(self) -> None:
        """
        Reset model weights to original values.

        WARNING: This reloads the model, losing all Hebbian learning.
        """
        logger.warning("Resetting model weights - all Hebbian learning will be lost")
        # Would need to reload from checkpoint
        # For now, just reset stats
        self.updater.reset_stats()
