"""Core orchestration logic for parallel text generation.

This module contains the domain service that orchestrates the generation process,
coordinating between various components while maintaining domain logic.
"""

import time
from typing import Optional, Any
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
from .sequence_tracker import SequenceTracker
from .retroactive_removal_coordinator import RetroactiveRemovalCoordinator
from ...utils.logging_utils import LoggingMixin


class GenerationOrchestrator(LoggingMixin):
    """Domain service that orchestrates the parallel generation process."""
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the generation orchestrator.
        
        Args:
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        self.setup_logging("generation_orchestrator", "orchestrator.log", debug_mode)
        
        # Track logical layout of parallel tokens
        self.logical_layout: list[LogicalPosition] = []
        
        # Initialize coordinators
        self.sequence_tracker = SequenceTracker(debug_mode)
        self.removal_coordinator = RetroactiveRemovalCoordinator(debug_mode)
    
    def orchestrate_generation(
        self,
        initial_state: GenerationState,
        config: GenerationConfig,
        strategy: GenerationStrategy,
        token_generator: TokenGeneratorInterface,
        retroactive_remover: Optional[Any] = None,
        data_capture: Optional[Any] = None,
        attention_manager: Optional[Any] = None
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
            data_capture: Optional experiment data capture interface
            attention_manager: Optional attention manager for mask construction

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

        # Initialize attention manager if provided
        if attention_manager:
            attention_manager.initialize(initial_state.sequence_length)
            self.log(f"Attention manager initialized (isolation={config.isolate_parallel_tokens})")
        
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

            # 1. Build custom attention mask if needed
            custom_mask = None
            if attention_manager and config.isolate_parallel_tokens:
                custom_mask = attention_manager.build_attention_mask(
                    seq_length=current_state.sequence_length,
                    dtype=torch.float32
                )

            # 2. Generate logits for next tokens
            logits, new_state = token_generator.generate_logits_with_cache(
                current_state,
                custom_attention_mask=custom_mask
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
            
            # Store original token set
            all_original_token_sets[logical_step] = [
                (t.id, t.probability) for t in token_set.tokens
            ]

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

            # 5. Capture experiment data if requested
            if data_capture:
                physical_start_idx = current_state.input_ids.size(1)
                physical_positions = list(range(physical_start_idx, physical_start_idx + len(token_ids)))

                # Get cached attention from token generator
                attention_data = token_generator.get_cached_attention()
                attention_tensor = None
                if attention_data:
                    attention_pattern, _ = attention_data
                    # attention_pattern.layers is a list of tensors, one per layer
                    attention_tensor = attention_pattern.layers

                # Capture step data
                data_capture.capture_step_data(
                    logical_step=logical_step,
                    physical_positions=physical_positions,
                    token_ids=token_ids,
                    logits=logits.tensor if hasattr(logits, 'tensor') else logits,
                    attention=attention_tensor,
                    kv_cache=current_state.past_key_values
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

            # Register parallel set with attention manager
            if attention_manager and len(token_ids) > 1 and config.isolate_parallel_tokens:
                attention_manager.register_parallel_set(physical_start_idx, physical_end_idx)
                self.log(f"Registered parallel set: positions {physical_start_idx}-{physical_end_idx}")
            
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

        # Save captured data if requested
        if data_capture:
            self.log("Saving experiment data...")
            data_capture.save_all()

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
    
    def get_sequence_metrics(self) -> dict[str, Any]:
        """Get metrics about the generation sequence."""
        metrics = self.sequence_tracker.get_metrics()
        metrics["logical_layout_size"] = len(self.logical_layout)
        return metrics
