"""Service for formatting generation responses.

This module handles the formatting of generation results into API responses.
"""

import logging
from typing import Any
from src.presentation.api.models.responses import (
    TokenInfo, StepInfo, RetroactiveRemovalInfo, 
    TimingInfo, ModelInfo, GenerationResponse
)
from src.presentation.api.models.requests import GenerationRequest
from .text_processing import TextProcessingService

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """Formats generation results into API responses."""
    
    def __init__(self):
        """Initialize the response formatter."""
        self.text_processor = TextProcessingService()
    
    def format_response(
        self,
        request: GenerationRequest,
        generation_result: dict[str, Any],
        elapsed_time: float,
        system_content: str,
        tokenizer: Any,
        device: str
    ) -> GenerationResponse:
        """Format the generation result into a response.
        
        Args:
            request: Original generation request
            generation_result: Raw generation result
            elapsed_time: Total elapsed time
            system_content: System content used
            tokenizer: Tokenizer for decoding
            device: Device used for generation
            
        Returns:
            Formatted GenerationResponse
        """
        try:
            # Process text outputs
            text_outputs = self.text_processor.process_generation_output(
                generation_result["generated_text"],
                generation_result.get("raw_generated_text", "")
            )
            
            # Debug logging
            if request.debug_mode:
                self._log_debug_info(text_outputs)
            
            # Create base response
            response = self._create_base_response(
                request, generation_result, text_outputs, 
                elapsed_time, system_content, device
            )
            
            # Process token sets
            self._process_token_sets(response, generation_result, tokenizer)
            
            # Add removal info if applicable
            if request.use_retroactive_removal:
                response.retroactive_removal = self._create_removal_info(request)
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            raise Exception(f"Failed to format generation response: {str(e)}")
    
    def _log_debug_info(self, text_outputs: dict[str, str]) -> None:
        """Log debug information about text outputs."""
        logger.debug(f"Generated text length: {len(text_outputs.get('generated_text', ''))}")
        logger.debug(f"Raw text length: {len(text_outputs.get('raw_generated_text', ''))}")
        logger.debug(f"Clean text length: {len(text_outputs.get('clean_text', ''))}")
        logger.debug(f"Clean text preview: {repr(text_outputs.get('clean_text', '')[:100])}")
    
    def _create_base_response(
        self,
        request: GenerationRequest,
        generation_result: dict[str, Any],
        text_outputs: dict[str, str],
        elapsed_time: float,
        system_content: str,
        device: str
    ) -> GenerationResponse:
        """Create the base response object."""
        return GenerationResponse(
            generated_text=text_outputs["generated_text"],
            raw_generated_text=text_outputs["raw_generated_text"],
            clean_text=text_outputs["clean_text"],
            steps=[],
            timing=TimingInfo(
                generation_time=generation_result.get("generation_time", elapsed_time),
                removal_time=generation_result.get("removal_time", 0.0),
                elapsed_time=elapsed_time,
            ),
            model_info=ModelInfo(
                model_name="deepcogito/cogito-v1-preview-llama-3B",
                is_qwen_model=generation_result.get("is_qwen_model", False),
                use_custom_rope=request.use_custom_rope,
                device=device,
                model_type="llama"
            ),
            selection_threshold=request.selection_threshold,
            max_tokens=request.max_tokens,
            min_steps=request.min_steps,
            prompt=request.prompt,
            had_repetition_loop=generation_result.get("had_repetition_loop", False),
            system_content=system_content,
            token_sets=[],
            token_sets_with_text=[],
            position_to_tokens=generation_result.get("position_to_tokens", {}),
            original_parallel_positions=list(
                generation_result.get("original_parallel_positions", set())
            ),
            final_removed_sets=generation_result.get("final_removed_sets", {}),
        )
    
    def _create_removal_info(self, request: GenerationRequest) -> RetroactiveRemovalInfo:
        """Create retroactive removal info."""
        return RetroactiveRemovalInfo(
            attention_threshold=request.attention_threshold,
            use_relative_attention=not request.no_relative_attention,
            relative_threshold=request.relative_threshold,
            use_multi_scale_attention=not request.no_multi_scale_attention,
            num_layers_to_use=request.num_layers_to_use,
            use_sigmoid_threshold=not request.no_sigmoid_threshold,
            sigmoid_steepness=request.sigmoid_steepness,
            complete_removal_mode=request.complete_removal_mode,
        )
    
    def _process_token_sets(
        self,
        response: GenerationResponse,
        generation_result: dict[str, Any],
        tokenizer: Any
    ) -> None:
        """Process token sets data for the response."""
        token_sets_data = generation_result.get("token_sets", [])
        if not token_sets_data:
            return
            
        steps_list = []
        formatted_token_sets = []
        token_sets_with_text = []
        
        for step_data in token_sets_data:
            try:
                processed = self._process_step_data(step_data, tokenizer)
                if processed:
                    step_info, formatted_set, set_with_text = processed
                    steps_list.append(step_info)
                    formatted_token_sets.append(formatted_set)
                    token_sets_with_text.append(set_with_text)
            except Exception as e:
                logger.warning(f"Error processing step data: {e}")
                continue
        
        response.token_sets = formatted_token_sets
        response.steps = steps_list
        response.token_sets_with_text = token_sets_with_text
    
    def _process_step_data(
        self,
        step_data: Any,
        tokenizer: Any
    ) -> tuple[StepInfo, Tuple, Tuple]:
        """Process a single step's token data."""
        if not isinstance(step_data, tuple) or len(step_data) != 3:
            logger.warning(f"Invalid step data format: {step_data}")
            return None
            
        position, original_data, removed_data = step_data
        
        if not (isinstance(original_data, tuple) and len(original_data) == 2 and
                isinstance(removed_data, tuple) and len(removed_data) == 2):
            logger.warning(f"Invalid token data format at position {position}")
            return None
        
        original_ids, original_probs = original_data
        removed_ids, removed_probs = removed_data
        
        # Convert to basic types
        original_pairs = [
            (int(tid), float(prob))
            for tid, prob in zip(original_ids, original_probs)
        ]
        removed_pairs = [
            (int(tid), float(prob))
            for tid, prob in zip(removed_ids, removed_probs)
        ]
        
        # Build step info
        parallel_tokens = [
            TokenInfo(
                token_text=tokenizer.decode([tid]),
                token_id=tid,
                probability=prob,
            )
            for tid, prob in original_pairs
        ]
        removed_tokens = [
            TokenInfo(
                token_text=tokenizer.decode([tid]),
                token_id=tid,
                probability=prob,
            )
            for tid, prob in removed_pairs
        ]
        
        step_info = StepInfo(
            position=position,
            parallel_tokens=parallel_tokens,
            removed_tokens=removed_tokens,
        )
        
        # Create formatted sets
        formatted_set = (position, original_pairs, removed_pairs)
        
        # Create sets with text
        original_with_text = [
            (int(tid), float(prob), tokenizer.decode([tid]))
            for tid, prob in zip(original_ids, original_probs)
        ]
        removed_with_text = [
            (int(tid), float(prob), tokenizer.decode([tid]))
            for tid, prob in zip(removed_ids, removed_probs)
        ]
        set_with_text = (position, original_with_text, removed_with_text)
        
        return step_info, formatted_set, set_with_text