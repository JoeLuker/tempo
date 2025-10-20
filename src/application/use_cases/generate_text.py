"""Main text generation use case for TEMPO.

This module implements the primary use case for generating text
using the parallel token generation approach.
"""

import time
import torch
from typing import Any, Optional

from ...domain.entities.parallel_generation import GenerationConfig, GenerationResult
from ...domain.entities.generation_state import GenerationState, TokenizationResult
from ...domain.services.generation_orchestrator import GenerationOrchestrator
from ...domain.interfaces.token_generation import TokenGeneratorInterface
from ...domain.interfaces.tokenizer import TokenizerInterface
from ...domain.interfaces.generation_strategy import GenerationStrategy
from ..services.sequence_manager import SequenceManager
from ...utils.logging_utils import LoggingMixin


class GenerateTextUseCase(LoggingMixin):
    """Use case for generating text with parallel tokens."""
    
    def __init__(
        self,
        token_generator: TokenGeneratorInterface,
        tokenizer: TokenizerInterface,
        generation_strategy: GenerationStrategy,
        sequence_manager: SequenceManager,
        rope_modifier: Optional[Any] = None,
        attention_manager: Optional[Any] = None,
        formatter: Optional[Any] = None,
        debug_mode: bool = False
    ):
        """Initialize the generate text use case.
        
        Args:
            token_generator: Interface for generating token logits
            tokenizer: Interface for tokenization operations
            generation_strategy: Strategy for token selection
            sequence_manager: Manager for sequence operations
            rope_modifier: Optional RoPE modifier for position embeddings
            attention_manager: Optional attention manager
            formatter: Optional text formatter
            debug_mode: Whether to enable debug logging
        """
        super().__init__()
        self.setup_logging("generate_text_use_case", "use_case.log", debug_mode)
        
        self.token_generator = token_generator
        self.tokenizer = tokenizer
        self.generation_strategy = generation_strategy
        self.sequence_manager = sequence_manager
        self.rope_modifier = rope_modifier
        self.attention_manager = attention_manager
        self.formatter = formatter
        
        # Create orchestrator
        self.orchestrator = GenerationOrchestrator(debug_mode=debug_mode)
    
    def execute(
        self,
        prompt: str,
        config: GenerationConfig,
        retroactive_remover: Optional[Any] = None
    ) -> GenerationResult:
        """Execute the text generation use case.
        
        Args:
            prompt: Text prompt to generate from
            config: Generation configuration
            retroactive_remover: Optional retroactive pruning component
            
        Returns:
            GenerationResult with generated text and metadata
        """
        try:
            # 1. Prepare the prompt
            formatted_prompt = self._prepare_prompt(prompt, config.system_content)
            
            # 2. Tokenize the prompt
            tokenization_result = self.tokenizer.tokenize_prompt(formatted_prompt)
            
            # 3. Initialize generation state
            initial_state = GenerationState(
                input_ids=tokenization_result.input_ids,
                attention_mask=tokenization_result.attention_mask,
                sequence_length=tokenization_result.token_count
            )
            
            # 4. Prime KV cache if not disabled
            if not config.disable_kv_cache:
                initial_state = self._prime_kv_cache(initial_state)
            
            # 5. Configure components
            self._configure_components(config, tokenization_result.token_count)
            
            # 6. Orchestrate generation
            result, final_state = self.orchestrator.orchestrate_generation(
                initial_state=initial_state,
                config=config,
                strategy=self.generation_strategy,
                token_generator=self.token_generator,
                retroactive_remover=retroactive_remover
            )
            
            # 7. Format the output
            result = self._format_output(
                result,
                final_state,
                formatted_prompt,
                tokenization_result.token_count,
                config
            )
            
            # 8. Check for repetition
            result.had_repetition_loop = self._check_repetition(result.raw_generated_text)
            
            # 9. Clean up
            self._cleanup()
            
            return result
            
        except Exception as e:
            self.log(f"Error in text generation: {e}", "error")
            raise
    
    def _prepare_prompt(self, prompt: str, system_content: Optional[str]) -> str:
        """Prepare the prompt with system content if provided."""
        if not system_content:
            return prompt
        
        # This would ideally use a proper template manager
        # For now, using simple concatenation
        return f"System: {system_content}\n\nUser: {prompt}\n\nAssistant:"
    
    def _prime_kv_cache(self, state: GenerationState) -> GenerationState:
        """Prime the KV cache with the initial prompt."""
        self.log("Priming KV cache with initial prompt...")
        
        # Generate with cache to initialize it
        _, state_with_cache = self.token_generator.generate_logits_with_cache(state)
        return state_with_cache
    
    def _configure_components(self, config: GenerationConfig, prompt_length: int) -> None:
        """Configure components based on generation config."""
        # Configure RoPE modifier if available
        if self.rope_modifier and config.isolate_parallel_tokens:
            self.rope_modifier.reset()
            if hasattr(self.rope_modifier, 'set_isolation_mode'):
                self.rope_modifier.set_isolation_mode(True)
        
        # Configure attention manager if available
        if self.attention_manager:
            self.attention_manager.reset_cache()
        
        # Configure sequence manager
        self.sequence_manager.initialize(prompt_length)
    
    def _format_output(
        self,
        result: GenerationResult,
        final_state: GenerationState,
        prompt: str,
        prompt_length: int,
        config: GenerationConfig
    ) -> GenerationResult:
        """Format the generation output."""
        # Extract generated tokens
        generated_token_ids = final_state.input_ids[0][prompt_length:].tolist()

        if self.debug_mode:
            self.log(f"Prompt length: {prompt_length}, Final sequence length: {final_state.input_ids.size(1)}")
            self.log(f"Generated {len(generated_token_ids)} token IDs: {generated_token_ids[:20]}")  # Show first 20

        # Decode raw text - join all decoded tokens
        decoded_tokens = self.tokenizer.decode_tokens(generated_token_ids)
        result.raw_generated_text = "".join(decoded_tokens) if isinstance(decoded_tokens, list) else decoded_tokens
        result.prompt = prompt

        if self.debug_mode:
            self.log(f"Raw generated text: '{result.raw_generated_text[:200]}'")  # Show first 200 chars
        
        # Format with layout if formatter available
        if self.formatter:
            # Use colored bracket formatting
            all_token_ids = final_state.input_ids[0].tolist()
            result.generated_text = self.formatter.format_with_parallel_indicators(
                token_ids=all_token_ids,
                logical_layout=result.logical_layout,
                prompt_length=prompt_length,
                all_original_token_sets=result.all_original_token_sets,
                all_surviving_token_sets=result.all_surviving_token_sets
            )

            # Extract clean text
            result.clean_text = self.formatter.extract_clean_text(
                token_ids=all_token_ids,
                logical_layout=result.logical_layout,
                prompt_length=prompt_length
            )
        else:
            result.generated_text = prompt + result.raw_generated_text
            result.clean_text = result.raw_generated_text
        
        # Build visualization data if requested
        if config.return_parallel_sets:
            result.token_sets = self._build_visualization_data(
                result.all_original_token_sets,
                result.all_surviving_token_sets
            )
        
        return result
    
    def _build_visualization_data(
        self,
        original_sets: dict[int, list],
        surviving_sets: dict[int, list]
    ) -> list:
        """Build visualization data from token sets."""
        viz_data = []
        
        for step in range(len(original_sets)):
            original = original_sets.get(step, [])
            surviving = surviving_sets.get(step, original)
            
            # Extract data
            original_ids = [tid for tid, _ in original]
            original_probs = [prob for _, prob in original]
            
            # Find removed tokens
            removed_ids = []
            removed_probs = []
            surviving_ids = {tid for tid, _ in surviving}
            
            for tid, prob in original:
                if tid not in surviving_ids:
                    removed_ids.append(tid)
                    removed_probs.append(prob)
            
            viz_data.append((
                step,
                (original_ids, original_probs),
                (removed_ids, removed_probs)
            ))
        
        return viz_data
    
    def _check_repetition(self, text: str, min_length: int = 5, max_length: int = 20, min_repeats: int = 3) -> bool:
        """Check if the generated text contains repetition patterns."""
        if len(text) < min_length * min_repeats:
            return False
        
        for seq_len in range(min_length, min(max_length, len(text) // min_repeats)):
            for i in range(len(text) - seq_len * min_repeats):
                seq = text[i:i + seq_len]
                
                # Skip trivial sequences
                if seq.isspace() or seq == seq[0] * seq_len:
                    continue
                
                # Count non-overlapping occurrences
                count = 0
                pos = i
                while pos < len(text):
                    found_pos = text.find(seq, pos)
                    if found_pos == -1:
                        break
                    count += 1
                    pos = found_pos + seq_len
                
                if count >= min_repeats:
                    self.log(f"Repetition detected: '{seq}' repeats {count} times")
                    return True
        
        return False
    
    def _cleanup(self) -> None:
        """Clean up after generation."""
        if self.rope_modifier:
            self.rope_modifier.reset()
        
        if self.attention_manager:
            self.attention_manager.reset_cache()
        
        if hasattr(self.token_generator, 'clear_kv_cache'):
            self.token_generator.clear_kv_cache()
