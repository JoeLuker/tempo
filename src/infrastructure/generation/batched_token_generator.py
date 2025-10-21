"""Batched token generator for parallel token processing with custom attention masking.

This module implements a hybrid approach that:
1. Processes parallel tokens together in a single batched forward pass
2. Applies custom attention masks to control cross-parallel visibility
3. Splits/merges KV caches appropriately for efficient incremental generation
"""

import torch
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass

from ...domain.entities.generation_state import GenerationState
from ...domain.entities.logits import TokenLogits
from ...utils.logging_utils import LoggingMixin


@dataclass
class BatchedTokenLogits:
    """Logits for multiple parallel tokens processed in a batch."""
    tensors: List[torch.Tensor]  # List of logit tensors, one per parallel token
    sequence_position: int
    batch_size: int


class BatchedTokenGenerator(LoggingMixin):
    """Generates logits for parallel tokens using batched processing.

    This generator handles the complexity of:
    - Replicating base KV cache for each parallel token
    - Processing tokens in a single batched forward pass
    - Applying custom attention masks
    - Splitting the resulting KV cache back into individual caches
    """

    def __init__(self,
                 model_adapter,
                 cache_manager=None,
                 performance_tracker=None,
                 debug_mode: bool = False):
        """Initialize the batched token generator.

        Args:
            model_adapter: Adapter for the language model
            cache_manager: Optional cache manager for attention/KV cache
            performance_tracker: Optional performance tracking
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        self.setup_logging("batched_token_generator", "batched_token_generator.log", debug_mode)

        self.model = model_adapter
        self.cache_manager = cache_manager
        self.performance_tracker = performance_tracker

    def generate_parallel_logits(
        self,
        base_state: GenerationState,
        num_parallel: int,
        custom_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[BatchedTokenLogits, List[GenerationState]]:
        """Generate logits for multiple parallel tokens in a single batch.

        This method:
        1. Replicates the base state KV cache for each parallel token
        2. Processes all tokens in a single batched forward pass
        3. Applies custom attention mask to control cross-parallel visibility
        4. Returns individual logits and states for each parallel token

        Args:
            base_state: Base generation state (before parallel tokens)
            num_parallel: Number of parallel tokens to process
            custom_attention_mask: Optional mask [seq_len, seq_len] to control attention

        Returns:
            Tuple of (BatchedTokenLogits, List[GenerationState]) - one state per parallel token
        """
        start_time = time.time()

        if self.debug_mode:
            self.log(f"Processing {num_parallel} parallel tokens in batch")

        # 1. Replicate base state for batch processing
        batch_input_ids, batch_attention_mask, batch_kv_cache = self._replicate_state_for_batch(
            base_state, num_parallel
        )

        if self.debug_mode:
            self.log(f"Replicated state: input_ids shape {batch_input_ids.shape}")

        # 2. Build batched attention mask
        if custom_attention_mask is not None:
            # Expand custom mask for batch: [batch, 1, seq_len, seq_len]
            batch_custom_mask = custom_attention_mask.unsqueeze(0).unsqueeze(0)
            batch_custom_mask = batch_custom_mask.expand(num_parallel, 1, -1, -1)

            if self.debug_mode:
                self.log(f"Custom attention mask shape: {batch_custom_mask.shape}")
        else:
            batch_custom_mask = None

        # 3. Run batched forward pass
        with torch.inference_mode():
            outputs = self.model.forward(
                input_ids=batch_input_ids,
                attention_mask=batch_custom_mask if batch_custom_mask is not None else batch_attention_mask,
                past_key_values=batch_kv_cache,
                use_cache=True,
                output_attentions=True,
                custom_attention_mask=batch_custom_mask
            )

        # 4. Extract logits for each parallel token
        # outputs.logits shape: [batch_size, seq_len, vocab_size]
        logits_list = []
        for i in range(num_parallel):
            # Get logits for the last position of this batch element
            token_logits = outputs.logits[i, -1, :]  # [vocab_size]
            logits_list.append(token_logits)

        # 5. Split KV cache into individual states
        individual_states = self._split_kv_cache(
            base_state,
            outputs.past_key_values,
            num_parallel
        )

        # 6. Cache attention if available
        if hasattr(outputs, "attentions") and outputs.attentions and self.cache_manager:
            # For batched processing, cache the attention from first token (representative)
            self.cache_manager.cache_attention(outputs.attentions, base_state.get_current_sequence_length())

        # Track performance
        if self.performance_tracker:
            duration = time.time() - start_time
            self.performance_tracker.track_model_call(duration, num_parallel)

        batched_logits = BatchedTokenLogits(
            tensors=logits_list,
            sequence_position=base_state.get_current_sequence_length(),
            batch_size=num_parallel
        )

        if self.debug_mode:
            self.log(f"Generated {len(logits_list)} parallel logit tensors in {time.time() - start_time:.4f}s")

        return batched_logits, individual_states

    def _replicate_state_for_batch(
        self,
        base_state: GenerationState,
        num_parallel: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[tuple]]:
        """Replicate generation state for batch processing.

        Args:
            base_state: Base state to replicate
            num_parallel: Number of replicas needed

        Returns:
            Tuple of (batch_input_ids, batch_attention_mask, batch_kv_cache)
        """
        # Replicate input_ids: [1, seq_len] -> [num_parallel, seq_len]
        batch_input_ids = base_state.input_ids.expand(num_parallel, -1)

        # Replicate attention_mask
        batch_attention_mask = base_state.attention_mask.expand(num_parallel, -1)

        # Replicate KV cache if it exists
        batch_kv_cache = None
        if base_state.past_key_values is not None:
            batch_kv_cache = self._replicate_kv_cache(base_state.past_key_values, num_parallel)

        return batch_input_ids, batch_attention_mask, batch_kv_cache

    def _replicate_kv_cache(self, kv_cache: tuple, num_parallel: int) -> tuple:
        """Replicate KV cache for batch processing.

        KV cache structure: tuple of (num_layers) where each layer is:
        (key_states, value_states) with shapes [batch=1, num_heads, seq_len, head_dim]

        We need to expand batch dimension from 1 to num_parallel.

        Args:
            kv_cache: Original KV cache tuple
            num_parallel: Number of parallel tokens

        Returns:
            Replicated KV cache tuple
        """
        replicated_cache = []

        for layer_cache in kv_cache:
            if layer_cache is None:
                replicated_cache.append(None)
                continue

            key_states, value_states = layer_cache

            # Expand batch dimension: [1, num_heads, seq_len, head_dim] -> [num_parallel, num_heads, seq_len, head_dim]
            expanded_key = key_states.expand(num_parallel, -1, -1, -1)
            expanded_value = value_states.expand(num_parallel, -1, -1, -1)

            replicated_cache.append((expanded_key, expanded_value))

        return tuple(replicated_cache)

    def _split_kv_cache(
        self,
        base_state: GenerationState,
        batch_kv_cache: tuple,
        num_parallel: int
    ) -> List[GenerationState]:
        """Split batched KV cache into individual generation states.

        After processing parallel tokens in a batch, we need to split the
        resulting KV cache so each parallel token has its own state for
        future generation.

        Args:
            base_state: Original base state
            batch_kv_cache: KV cache from batched forward pass [num_parallel, ...]
            num_parallel: Number of parallel tokens

        Returns:
            List of GenerationState, one per parallel token
        """
        individual_states = []

        for i in range(num_parallel):
            # Extract this token's KV cache from the batch
            individual_kv = self._extract_kv_for_index(batch_kv_cache, i)

            # Create new state for this token
            # Note: input_ids will be updated by the orchestrator when token is selected
            state = GenerationState(
                input_ids=base_state.input_ids.clone(),  # Will be updated
                attention_mask=base_state.attention_mask.clone(),  # Will be updated
                past_key_values=individual_kv,
                sequence_length=base_state.sequence_length,
                generated_tokens=base_state.generated_tokens.copy()
            )

            individual_states.append(state)

        return individual_states

    def _extract_kv_for_index(self, batch_kv_cache: tuple, index: int) -> tuple:
        """Extract KV cache for a specific batch index.

        Args:
            batch_kv_cache: Batched KV cache [num_parallel, ...]
            index: Index of the token to extract

        Returns:
            KV cache tuple for the specified token [1, ...]
        """
        extracted_cache = []

        for layer_cache in batch_kv_cache:
            if layer_cache is None:
                extracted_cache.append(None)
                continue

            key_states, value_states = layer_cache

            # Extract and add batch dimension back: [num_parallel, ...] -> [1, ...]
            extracted_key = key_states[index:index+1]  # Keep batch dim
            extracted_value = value_states[index:index+1]

            extracted_cache.append((extracted_key, extracted_value))

        return tuple(extracted_cache)
