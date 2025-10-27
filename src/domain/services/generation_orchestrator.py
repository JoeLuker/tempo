"""Core orchestration logic for parallel text generation.

This module contains the domain service that orchestrates the generation process,
coordinating between various components while maintaining domain logic.
"""

import time
from typing import Optional
import torch

from ..entities.parallel_generation import (
    GenerationConfig, GenerationResult, ParallelTokenSet,
    LogicalPosition, MCTSState
)
from ..entities.generation_state import GenerationState
from ..entities.token import TokenSet
from ..entities.logits import TokenLogits
from ..interfaces.token_generation import TokenGeneratorInterface
from ..interfaces.generation_strategy import GenerationStrategy
from ..interfaces.attention_manager import AttentionManagerInterface
from ..interfaces.data_capture import DataCaptureInterface
from ..interfaces.retroactive_remover import RetroactiveRemoverInterface
from .sequence_tracker import SequenceTracker
from .retroactive_removal_coordinator import RetroactiveRemovalCoordinator
from ...utils.logging_utils import LoggingMixin
from ...extensions import GenState, Extension, run_extensions
import math


class GenerationOrchestrator(LoggingMixin):
    """Domain service that orchestrates the parallel generation process."""
    
    def __init__(self, debug_mode: bool = False, extensions: Optional[list[Extension]] = None):
        """Initialize the generation orchestrator.

        Args:
            debug_mode: Whether to enable debug logging
            extensions: Optional list of extension functions to run during generation
        """
        super().__init__()
        self.setup_logging("generation_orchestrator", "orchestrator.log", debug_mode)

        # Track logical layout of parallel tokens
        self.logical_layout: list[LogicalPosition] = []

        # Initialize coordinators
        self.sequence_tracker = SequenceTracker(debug_mode)
        self.removal_coordinator = RetroactiveRemovalCoordinator(debug_mode)

        # Store extensions
        self.extensions = extensions or []

        # Extension state tracking
        self.extension_metadata: dict = {}
    
    def orchestrate_generation(
        self,
        initial_state: GenerationState,
        config: GenerationConfig,
        strategy: GenerationStrategy,
        token_generator: TokenGeneratorInterface,
        retroactive_remover: Optional[RetroactiveRemoverInterface] = None,
        data_capture: Optional[DataCaptureInterface] = None,
        attention_manager: Optional[AttentionManagerInterface] = None,
        json_collector: Optional['GenerationDataCollector'] = None
    ) -> tuple[GenerationResult, GenerationState]:
        """Orchestrate the parallel text generation process.

        This method coordinates the generation process, managing state,
        applying strategies, and tracking progress.

        Args:
            initial_state: Initial generation state with prompt
            config: Generation configuration
            strategy: Strategy for token selection and pruning
            token_generator: Interface for generating token logits
            retroactive_remover: Optional retroactive pruning component
            data_capture: Optional experiment data capture instance
            attention_manager: Optional attention service for controlling parallel token visibility
            json_collector: Optional JSON data collector for rich output

        Returns:
            GenerationResult with generated text and metadata
        """
        # Initialize timing
        generation_start = time.time()
        removal_time = 0.0
        removal_steps = 0
        
        # Initialize tracking
        self.sequence_tracker.initialize(initial_state.sequence_length)
        self.logical_layout = [LogicalPosition(0, 0, initial_state.sequence_length - 1)]
        
        # Track token sets for processing
        all_original_token_sets: dict[int, list[tuple[int, float]]] = {}
        all_surviving_token_sets: dict[int, list[tuple[int, float]]] = {}
        
        # Current state
        current_state = initial_state
        
        # Notify initial sequence callback
        if config.sequence_callback:
            config.sequence_callback(0, 0, initial_state.sequence_length)
        
        # Main generation loop
        for logical_step in range(config.max_tokens):
            self.log(f"\n--- Logical Step {logical_step} ---")
            step_start_time = time.time()

            # 1. Generate logits for next tokens
            # Note: We don't pass custom attention mask here because:
            # - Parallel tokens are selected from ONE logit distribution (can't see each other)
            # - Future token isolation is handled by registering parallel sets after append
            logits, new_state = token_generator.generate_logits_with_cache(
                current_state,
                custom_attention_mask=None
            )
            current_state = new_state

            # 3. Apply generation strategy to select tokens
            token_set = strategy.select_tokens(
                logits=logits,
                step=logical_step,
                config=config,
                state=current_state
            )

            if not token_set.tokens:
                self.log("No tokens selected, ending generation", "warning")
                break

            # 3.5. Run extensions (may modify config threshold or inject prompts)
            config, prompt_to_inject = self._run_extensions(
                logical_step=logical_step,
                token_set=token_set,
                current_config=config,
                prompt_length=initial_state.sequence_length
            )

            # Check if extension wants to inject a prompt
            if prompt_to_inject:
                # Return special signal to use case to handle prompt injection
                # Store injection info in metadata for caller to handle
                self.extension_metadata['pending_injection'] = {
                    'prompt': prompt_to_inject,
                    'step': logical_step
                }
                self.log(f"Extension requested prompt injection at step {logical_step}: '{prompt_to_inject[:50]}...'")
                # Note: Actual injection must be handled by use case which has tokenizer access

            # Store original token set
            all_original_token_sets[logical_step] = [
                (t.id, t.probability) for t in token_set.tokens
            ]

            # Capture JSON data if collector provided
            if json_collector:
                step_time_ms = (time.time() - step_start_time) * 1000
                self._capture_json_step(
                    json_collector,
                    logical_step,
                    current_state.sequence_length,
                    token_set,
                    logits,
                    step_time_ms
                )

            # 4. Apply retroactive removal if enabled
            if config.use_retroactive_removal and logical_step > 0 and retroactive_remover:
                removal_start = time.time()
                
                # Apply retroactive removal
                surviving_history = self.removal_coordinator.apply_retroactive_removal(
                    retroactive_remover,
                    initial_state.sequence_length,
                    all_original_token_sets,
                    logical_step
                )
                all_surviving_token_sets.update(surviving_history)
                
                removal_time += time.time() - removal_start
                removal_steps += 1
            
            # Get final token IDs
            token_ids = [t.id for t in token_set.tokens]

            # 5. Capture experimental data if enabled
            if data_capture:
                # Get physical positions for this step
                physical_start_idx = current_state.input_ids.size(1)
                physical_positions = list(range(physical_start_idx, physical_start_idx + len(token_ids)))

                # Get cached attention if available
                attention_weights = None
                if hasattr(token_generator, 'get_cached_attention'):
                    cached = token_generator.get_cached_attention()
                    if cached:
                        attention_pattern = cached[0]  # AttentionPattern entity
                        # Stack all attention layers into a single tensor for capture
                        attention_weights = tuple(attention_pattern.layers)

                # Capture this step
                data_capture.capture_step_data(
                    logical_step=logical_step,
                    physical_positions=physical_positions,
                    token_ids=token_ids,
                    logits=logits.tensor,  # Pass the raw logits tensor
                    attention=attention_weights,
                    kv_cache=current_state.past_key_values if data_capture.capture_kv_cache else None
                )

            # 6. Update state with new tokens
            physical_start_idx = current_state.input_ids.size(1)
            new_tokens_tensor = torch.tensor([token_ids], device=current_state.input_ids.device)

            new_input_ids = torch.cat([current_state.input_ids, new_tokens_tensor], dim=1)
            new_attention_mask = torch.cat([
                current_state.attention_mask,
                torch.ones((1, len(token_ids)), device=current_state.attention_mask.device)
            ], dim=1)

            physical_end_idx = new_input_ids.size(1) - 1
            
            # Create new state
            current_state = GenerationState(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                past_key_values=current_state.past_key_values,
                sequence_length=new_input_ids.size(1),
                generated_tokens=current_state.generated_tokens + token_ids
            )

            # Update logical layout
            self.logical_layout.append(
                LogicalPosition(logical_step, physical_start_idx, physical_end_idx)
            )

            # Register parallel token set with attention manager
            if attention_manager and len(token_ids) > 1:
                attention_manager.register_parallel_set(physical_start_idx, physical_end_idx)
                if self.debug_mode:
                    self.log(f"Registered parallel set: {physical_start_idx}-{physical_end_idx} ({len(token_ids)} tokens)")

            # Update sequence tracking
            self.sequence_tracker.update_sequence_length(
                current_state.sequence_length - initial_state.sequence_length,
                config.sequence_callback
            )
            
            # 7. Check termination conditions
            if logical_step >= config.min_steps:
                if strategy.should_terminate(token_set, current_state):
                    self.log(f"Termination condition met at step {logical_step}")
                    break
        
        # Calculate total generation time
        generation_time = time.time() - generation_start
        
        # Build result
        result = GenerationResult(
            generated_text="",  # Will be filled by formatter
            raw_generated_text="",  # Will be filled by formatter
            generation_time=generation_time,
            removal_time=removal_time,
            removal_steps=removal_steps,
            prompt="",  # Will be filled by caller
            selection_threshold=config.selection_threshold,
            use_retroactive_removal=config.use_retroactive_removal,
            min_steps=config.min_steps,
            disable_kv_cache=config.disable_kv_cache,
            isolate_parallel_tokens=config.isolate_parallel_tokens,
            logical_layout=self.logical_layout,
            all_original_token_sets=all_original_token_sets,
            all_surviving_token_sets=all_surviving_token_sets
        )
        
        return result, current_state
    
    def get_sequence_metrics(self) -> dict:
        """Get metrics about the generation sequence."""
        metrics = self.sequence_tracker.get_metrics()
        metrics["logical_layout_size"] = len(self.logical_layout)
        return metrics

    def _calculate_entropy(self, token_set: TokenSet) -> float:
        """Calculate Shannon entropy of token probabilities.

        Args:
            token_set: Set of tokens with probabilities

        Returns:
            Shannon entropy value
        """
        if not token_set.tokens:
            return 0.0

        entropy = 0.0
        for token in token_set.tokens:
            if token.probability > 0:
                entropy -= token.probability * math.log2(token.probability)

        return entropy

    def _run_extensions(
        self,
        logical_step: int,
        token_set: TokenSet,
        current_config: GenerationConfig,
        prompt_length: int
    ) -> tuple[GenerationConfig, Optional[str]]:
        """Run extensions and return updated config and optional prompt injection.

        Args:
            logical_step: Current generation step
            token_set: Selected tokens for this step
            current_config: Current generation config
            prompt_length: Length of the prompt

        Returns:
            Tuple of (modified config, prompt to inject or None)
        """
        if not self.extensions:
            return current_config, None

        # Calculate entropy for this step
        entropy = self._calculate_entropy(token_set)

        # Convert to extension state
        ext_state = GenState(
            step=logical_step,
            entropy=entropy,
            threshold=current_config.selection_threshold,
            selected_tokens=tuple((t.id, t.probability) for t in token_set.tokens),
            branching_factor=len(token_set.tokens),
            prompt_length=prompt_length,
            metadata=self.extension_metadata
        )

        # Run extensions
        ext_state = run_extensions(ext_state, self.extensions)

        # Update extension metadata (extensions can write to it)
        self.extension_metadata = ext_state.metadata

        # Check for prompt injection request
        prompt_to_inject = ext_state.metadata.pop('inject_prompt', None)

        # Check if threshold was modified
        if ext_state.threshold != current_config.selection_threshold:
            if self.debug_mode:
                self.log(f"Extension modified threshold: {current_config.selection_threshold:.4f} → {ext_state.threshold:.4f}")

            # Create new config with updated threshold
            current_config = GenerationConfig(
                max_tokens=current_config.max_tokens,
                selection_threshold=ext_state.threshold,
                min_steps=current_config.min_steps,
                use_retroactive_removal=current_config.use_retroactive_removal,
                disable_kv_cache=current_config.disable_kv_cache,
                isolate_parallel_tokens=current_config.isolate_parallel_tokens,
                show_token_ids=current_config.show_token_ids,
                system_content=current_config.system_content,
                return_parallel_sets=current_config.return_parallel_sets,
                sequence_callback=current_config.sequence_callback
            )

        return current_config, prompt_to_inject

    def _capture_json_step(
        self,
        collector: 'GenerationDataCollector',
        step: int,
        prompt_tokens_so_far: int,
        token_set: TokenSet,
        logits: TokenLogits,
        generation_time_ms: float
    ) -> None:
        """Capture data for JSON output.

        Args:
            collector: The data collector
            step: Current logical step
            prompt_tokens_so_far: Number of prompt tokens processed
            token_set: Selected token set
            logits: Full logits distribution
            generation_time_ms: Time taken for this step
        """
        # Extract probabilities from logits
        # Handle both 2D (batch_size, vocab_size) and 3D (batch_size, seq_len, vocab_size) logits
        if logits.tensor.dim() == 3:
            probs = torch.softmax(logits.tensor[0, -1, :], dim=-1)
        else:
            probs = torch.softmax(logits.tensor[-1, :], dim=-1)

        # Get top tokens for rejected comparison (top 20 to find 10 that weren't selected)
        top_probs, top_indices = torch.topk(probs, k=min(20, probs.size(0)))

        # Build selected tokens list
        selected_ids = {t.id for t in token_set.tokens}
        selected_tokens = []
        for token in token_set.tokens:
            # Find logit for this token
            if logits.tensor.dim() == 3:
                logit_value = logits.tensor[0, -1, token.id].item()
            else:
                logit_value = logits.tensor[-1, token.id].item()
            selected_tokens.append((
                token.id,
                "",  # Token text will be decoded later if needed
                token.probability,
                logit_value
            ))

        # Build rejected tokens list (top tokens not selected)
        rejected_tokens = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            if idx not in selected_ids and len(rejected_tokens) < 10:
                if logits.tensor.dim() == 3:
                    logit_value = logits.tensor[0, -1, idx].item()
                else:
                    logit_value = logits.tensor[-1, idx].item()
                rejected_tokens.append((
                    idx,
                    "",  # Token text will be decoded later if needed
                    prob,
                    logit_value
                ))

        # Call collector
        collector.add_step(
            step_num=step,
            position=step,
            prompt_tokens_so_far=prompt_tokens_so_far,
            selected_tokens=selected_tokens,
            rejected_tokens=rejected_tokens,
            generation_time_ms=generation_time_ms,
            attention_summary=None  # Could add attention data if needed
        )
